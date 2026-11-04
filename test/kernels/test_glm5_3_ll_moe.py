# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.moe`` with ``ll.hops`` on 4 SM100-family GPUs (GB300, W=4): one GLM-5.2 MoE
layer at TP4 shapes (256 experts, top-8, H 6144, I 512 per rank) with synthetic
NVFP4 weights processed as the server processes them
(``align_fp4_moe_weights_for_flashinfer_trtllm``).

1. The copied FlashInfer TRT-LLM NVFP4 MoE call (``ll.moe.routed_deferred``) against
   SGLang's runner (``fused_experts_none_to_flashinfer_trtllm_fp4`` under the
   deferred-finalize context, bypassed top-k) at 5, 6, 12, 24 and 48 rows: the same
   routing weights and the same finalized rows bit for bit (the permuted row order
   may differ between runs at 24+ rows; the finalize reads it).
2. The layer (post-attention hop with the NVFP4 output, router, routed experts,
   shared expert on the alternate stream, the finalize hop into the next layer's
   norm) against the stock path (FlashInfer's AR + residual + RMSNorm, the router,
   SGLang's runner, ``moe_finalize_fuse_shared``, FlashInfer's AR + residual +
   RMSNorm):
   - bit for bit where every hop of both runs FlashInfer's one-shot geometry (<= 21
     rows) when the stock runner is handed the hop's NVFP4 input (its FP4-dispatch
     input), and also with its own quantize at the P10 skip rows (3, 21);
   - with its own quantize (SGLang's cute-dsl ``fp4_quantize``) the FP4 codes differ
     from the hop's epilogue (FlashInfer's MNNVL quant, bit-identical to FlashInfer's
     cuda ``fp4_quantize``) on about 1e-3 of the codes: reported, and the layer
     outputs gated by ``ll.moe_layer.max_abs`` / ``ll.moe_layer.rel_l2``.
3. The layer captured in a CUDA graph (shared expert on the alternate stream) equals
   its eager run bit for bit.
4. The shared expert on a stream of the greatest priority (option
   ``ll.moe.alt_stream_priority``): the same bits as on the alternate stream, eager
   and replayed, and the captured graph's kernel nodes keep the stream's priority
   (``CU_KERNEL_NODE_ATTRIBUTE_PRIORITY``).

Run with ``python test/kernels/test_glm5_3_ll_moe.py`` (torchrun on 4 GPUs).
"""

from __future__ import annotations

import os
import sys
import time
import types

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="run via torchrun: python test/kernels/test_glm5_3_ll_moe.py",
)

H, E, TOP_K, I_RANK, I_SHARED = 6144, 256, 8, 512, 512
EPS = 1e-5
RSF = 2.5  # routed scaling factor
ROWS = [5, 6, 12, 24, 48]
GS_X = 448.0 * 6.0 / 10.0  # FC1 input global scale (|x| up to ~10)
GS_W, GS_I = 150.0, 448.0 * 6.0 / 8.0


def log(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg, flush=True)


class _Gate(torch.nn.Module):
    """The router (MoEGate's shape and contract: bf16 [E, H] weight, FP32 logits;
    its own GEMM below ``tiny_router_gemm_max_tokens`` + 1 rows, where ll.gemm keeps
    the stock router)."""

    tiny_router_gemm_max_tokens = 16

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight
        self.bias = None

    def forward(self, x):
        return torch.mm(x, self.weight.t(), out_dtype=torch.float32)


class _Shared(torch.nn.Module):
    """The shared expert's math (bf16 GEMMs, SiLU-and-mul), this rank's shard."""

    def __init__(self, gu: torch.Tensor, dn: torch.Tensor) -> None:
        super().__init__()
        self.gu, self.dn = gu, dn

    def forward(self, x):
        g, u = (x @ self.gu.t()).chunk(2, dim=-1)
        return (torch.nn.functional.silu(g) * u) @ self.dn.t()


def synthetic_layer(dev):
    """This rank's experts (its TP shard), processed like the server, and the
    SGLang quant info / runner config / top-k config of the same layer."""
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp4MoeQuantInfo,
        align_fp4_moe_weights_for_flashinfer_trtllm,
        get_activation_type,
    )
    from sglang.srt.layers.moe.topk import TopKConfig
    from sglang.srt.layers.moe.utils import RoutingMethodType

    from sgl_nvidia_mega.glm5_3.ll.moe import RoutedArgs

    rank = dist.get_rank()
    g = torch.Generator(device=dev).manual_seed(100 + rank)
    gs = torch.Generator(device=dev).manual_seed(7)  # replicated: router, bias
    f8 = torch.float8_e4m3fn
    p = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
    u8 = lambda *s: torch.randint(0, 256, s, generator=g, device=dev, dtype=torch.uint8)  # noqa: E731
    sf = lambda *s: (1 + torch.rand(s, generator=g, device=dev)).to(f8)  # noqa: E731
    layer = torch.nn.Module()
    layer.w13_weight = p(u8(E, 2 * I_RANK, H // 2))
    layer.w2_weight = p(u8(E, H, I_RANK // 2))
    layer.w13_weight_scale = p(sf(E, 2 * I_RANK, H // 16))
    layer.w2_weight_scale = p(sf(E, H, I_RANK // 16))
    ones = torch.ones(E, dtype=torch.float32, device=dev)
    layer.w2_input_scale_quant = p(ones * GS_I)
    layer.g1_alphas = p(ones / (GS_X * GS_W))
    layer.g2_alphas = p(ones / (GS_I * GS_W))
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")
    align_fp4_moe_weights_for_flashinfer_trtllm(layer)
    w13_input_scale_quant = torch.tensor([GS_X], dtype=torch.float32, device=dev)
    bias = (torch.randn(E, generator=gs, device=dev) * 0.05).float()
    router = (torch.randn(E, H, generator=gs, device=dev) * 0.02).bfloat16()
    shared = _Shared(
        (torch.randn(2 * I_SHARED, H, generator=g, device=dev) * 0.02).bfloat16(),
        (torch.randn(H, I_SHARED, generator=g, device=dev) * 0.03).bfloat16(),
    )
    rmt = RoutingMethodType.DeepSeekV3
    args = RoutedArgs(
        w13_weight=layer.w13_weight.data,
        w13_weight_scale=layer.w13_weight_scale.data.view(f8),
        w2_weight=layer.w2_weight.data,
        w2_weight_scale=layer.w2_weight_scale.data.view(f8),
        g1_scale_c=layer.g1_scale_c.data,
        g1_alphas=layer.g1_alphas.data,
        g2_alphas=layer.g2_alphas.data,
        w13_input_scale_quant=w13_input_scale_quant,
        num_experts=E,
        local_expert_offset=0,
        local_num_experts=E,
        intermediate_size=int(layer.intermediate_size_per_partition),
        routing_bias=bias,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=RSF,
        routing_method_type=int(rmt),
        activation_type=get_activation_type("silu", is_gated=True),
    )
    quant_info = FlashInferTrtllmFp4MoeQuantInfo(
        w13_weight=layer.w13_weight.data,
        w2_weight=layer.w2_weight.data,
        w13_weight_scale=layer.w13_weight_scale.data,
        w2_weight_scale=layer.w2_weight_scale.data,
        g1_scale_c=layer.g1_scale_c.data,
        g1_alphas=layer.g1_alphas.data,
        g2_alphas=layer.g2_alphas.data,
        w13_input_scale_quant=w13_input_scale_quant,
        global_num_experts=E,
        local_expert_offset=0,
        local_num_experts=E,
        intermediate_size_per_partition=int(layer.intermediate_size_per_partition),
        routing_method_type=rmt,
    )
    runner_config = MoeRunnerConfig(
        num_experts=E,
        top_k=TOP_K,
        activation="silu",
        is_gated=True,
        routed_scaling_factor=RSF,
    )
    topk_config = TopKConfig(
        top_k=TOP_K,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        renormalize=True,
        correction_bias=bias,
        routed_scaling_factor=RSF,
        scoring_func="sigmoid",
    )
    gate = _Gate(router)
    mlp = types.SimpleNamespace(
        gate=gate,
        shared_experts=shared,
        alt_stream=torch.cuda.Stream(device=dev),
        experts=types.SimpleNamespace(w13_input_scale_quant=w13_input_scale_quant),
    )
    return mlp, args, quant_info, runner_config, topk_config


@pytest.fixture(scope="module")
def env():
    # SGLang's MoE runner modules import each other; load them in the server's order
    # (the model module first), as a server process does.
    import sglang.srt.models.deepseek_v2  # noqa: F401
    from flashinfer.comm import Mapping, MNNVLAllReduceFusionWorkspace
    from flashinfer.comm.mnnvl import TorchDistBackend

    from sgl_nvidia_mega.glm5_3.ll import hops, transport

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=dev)
    world, rank = dist.get_world_size(), dist.get_rank()
    cpu = dist.new_group(backend="gloo")
    t0 = time.time()
    fam = hops.LlHopFamily(
        device_group=dist.group.WORLD, cpu_group=cpu, eps=EPS, deadline_s=10.0
    )
    fam.compile([48])
    ws = MNNVLAllReduceFusionWorkspace(
        Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world),
        buffer_size_in_bytes=transport.buffer_bytes(world, 48, H),
        comm_backend=TorchDistBackend(group=cpu),
    )
    layer = synthetic_layer(dev)
    # the router role on ll.gemm (when present) at every row count the tests run
    from sgl_nvidia_mega.glm5_3 import ll

    gemm = ll.build_linear("router", layer[0].gate, sorted(set(ROWS + [3, 21])))
    log(f"[setup] W={world}: hops + synthetic layer in {time.time() - t0:.1f} s; "
        f"ll modules {sorted(ll.modules())}; router: {gemm or 'stock (no ll.gemm)'}")  # fmt: skip
    yield fam, ws, layer, dev
    dist.barrier(group=cpu)


def inputs(rows, seed, dev):
    rank = dist.get_rank()
    g = torch.Generator().manual_seed(1000 * seed + 3)
    residual = torch.randn(rows, H, generator=g) * 0.8
    gr = torch.Generator().manual_seed(1000 * seed + 11 + rank)
    partial = torch.randn(rows, H, generator=gr) * 0.3
    gam = lambda k: 1 + 0.2 * torch.randn(H, generator=torch.Generator().manual_seed(k))  # noqa: E731
    bf = lambda t: t.to(dev, torch.bfloat16)  # noqa: E731
    return bf(partial), bf(residual), bf(gam(5)), bf(gam(6))


def same(a, b) -> bool:
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def stock_deferred(layer, x, logits, fp4=None):
    """SGLang's runner on a bypassed top-k under the deferred-finalize context;
    ``fp4``: (codes, scales) handed in as an FP4-dispatched input."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_deferred_finalize_context,
        fused_experts_none_to_flashinfer_trtllm_fp4,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import BypassedTopKOutput

    mlp, args, quant_info, runner_config, topk_config = layer
    dispatch = StandardDispatchOutput(
        hidden_states=x if fp4 is None else fp4[0],
        hidden_states_scale=None if fp4 is None else fp4[1],
        topk_output=BypassedTopKOutput(x, logits, topk_config),
    )
    with flashinfer_trtllm_deferred_finalize_context():
        out = fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch, quant_info, runner_config
        )
    return out.hidden_states


def finalize(gemm2, idx, weights, shared):
    from sglang.kernels.ops.moe.moe_finalize_fuse_shared import moe_finalize_fuse_shared

    return moe_finalize_fuse_shared(gemm2, idx.reshape(-1), weights, shared, TOP_K)


def test_routed_call_matches_sglang_runner(env):
    from sgl_nvidia_mega.glm5_3.ll import moe

    fam, ws, layer, dev = env
    mlp, args = layer[0], layer[1]
    bad = []
    for i, rows in enumerate(ROWS):
        _, x, _, _ = inputs(rows, i, dev)
        logits = mlp.gate(x)
        ours = moe.routed_deferred(args, logits, *moe.quantize_input(x, args))
        ref = stock_deferred(layer, x, logits)
        same_w = same(ours[1], ref.expert_weights)
        same_i = same(ours[2].reshape(-1), ref.expanded_idx_to_permuted_idx.reshape(-1))
        fin_ours = finalize(ours[0], ours[2], ours[1], None)
        fin_ref = finalize(
            ref.gemm2_out, ref.expanded_idx_to_permuted_idx, ref.expert_weights, None
        )
        ok = same_w and same(fin_ours, fin_ref)
        log(f"  routed M={rows:2d}: weights {same_w} idx {same_i} finalized bitwise {same(fin_ours, fin_ref)} "
            f"(|out| max {fin_ref.float().abs().max().item():.3f}, weights {ours[1].dtype})")  # fmt: skip
        if not ok:
            bad.append(f"M={rows}: the copied MoE call differs from SGLang's runner")
    torch.cuda.synchronize()
    assert not bad, "\n".join(bad)


def _ours(fam, layer, partial, residual, g_post, g_next, stream=None):
    """ll.hops post-attention hop (NVFP4 out) -> ll.moe -> the finalize hop."""
    from sgl_nvidia_mega.glm5_3.ll import hops, moe

    mlp, args = layer[0], layer[1]
    out = fam.allreduce_norm_nvfp4(
        partial, residual, g_post, args.w13_input_scale_quant, site=1
    )
    if out is None:  # a P10 skip row
        x, res1 = fam.allreduce_norm(partial, residual, g_post, site=1)
        nv = None
    else:
        x, res1, fp4, sf = out
        nv = hops.Nvfp4Input(3, x, fp4, sf, args.w13_input_scale_quant)
    fin = moe.moe_forward(mlp, args, x, nv, stream)
    norm, res2 = fam.finalize(fin, res1, g_next, site=5)
    return x, res1, norm, res2


def _stock(ws, layer, partial, residual, g_post, g_next, fp4_of=None):
    """FlashInfer AR fusion (FlashInfer's strategy) -> router -> SGLang's runner (its
    own quantize, or ``fp4_of(x)``) -> shared expert -> moe_finalize_fuse_shared ->
    AR fusion."""
    import flashinfer.comm as comm

    from sgl_nvidia_mega.glm5_3.ll import moe

    mlp, args = layer[0], layer[1]

    def ar(x, res, gamma):
        n, r = torch.empty_like(x), torch.empty_like(res)
        comm.allreduce_fusion(
            input=x, workspace=ws, pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True, residual_out=r, norm_out=n, residual_in=res,
            rms_gamma=gamma, rms_eps=EPS, use_oneshot=None,
        )  # fmt: skip
        return n, r

    x, res1 = ar(partial, residual, g_post)
    d = stock_deferred(layer, x, mlp.gate(x), None if fp4_of is None else fp4_of(x))
    part = finalize(
        d.gemm2_out,
        d.expanded_idx_to_permuted_idx,
        d.expert_weights,
        mlp.shared_experts(x),
    )
    norm, res2 = ar(part, res1, g_next)
    del moe
    return x, res1, norm, res2


def test_layer_matches_stock(env):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        quantize_hidden_states_fp4,
    )

    from sgl_nvidia_mega.glm5_3.ll import hops, moe

    fam, ws, layer, dev = env
    mlp, args = layer[0], layer[1]
    tol, tol_rel = (
        moe.TOLERANCES["ll.moe_layer.max_abs"],
        moe.TOLERANCES["ll.moe_layer.rel_l2"],
    )
    bad = []
    for i, rows in enumerate(ROWS + [3, 21]):
        partial, residual, g_post, g_next = inputs(rows, 10 + i, dev)
        ours = _ours(fam, layer, partial, residual, g_post, g_next)
        skip = hops.nvfp4_skipped(fam.world, rows)
        fi_band = rows * H * fam.world * 2 <= hops.FI_ONESHOT_MAX_BYTES
        line = f"  layer M={rows:2d}{' [P10 skip]' if skip else ''}:"
        if not skip:
            # the hop's FP4 vs SGLang's quantize of the same norm output
            out = fam.allreduce_norm_nvfp4(
                partial, residual, g_post, args.w13_input_scale_quant, site=1
            )
            q_ref, _ = quantize_hidden_states_fp4(out[0], args.w13_input_scale_quant)
            flips = (out[2] != q_ref).float().mean().item()
            line += f" FP4 codes vs SGLang's quantize {flips:.2e};"
            if flips > 1e-2:
                bad.append(f"M={rows}: FP4 codes vs SGLang's quantize {flips:.2e}")
            # the stock layer handed the same FP4 input
            handed = _stock(ws, layer, partial, residual, g_post, g_next,
                            fp4_of=lambda x: fam.allreduce_norm_nvfp4(partial, residual, g_post,
                                                                      args.w13_input_scale_quant, site=1)[2:])  # fmt: skip
            same_handed = all(same(a, b) for a, b in zip(ours, handed))
            line += f" == stock(hop FP4) {same_handed};"
            if fi_band and not same_handed:
                bad.append(f"M={rows}: not bitwise to the stock layer on the hop's FP4")
        stock = _stock(ws, layer, partial, residual, g_post, g_next)
        bitwise = all(same(a, b) for a, b in zip(ours, stock))
        d = max(
            (a.float() - b.float()).abs().max().item()
            for a, b in zip(ours[2:], stock[2:])
        )
        rel = (
            (ours[3].float() - stock[3].float()).norm() / stock[3].float().norm()
        ).item()
        line += f" == stock {bitwise}; max abs {d:.3e}, residual rel L2 {rel:.2e}"
        log(line)
        if skip and fi_band and not bitwise:
            bad.append(f"M={rows}: P10 skip row not bitwise to the stock layer")
        if d > tol or rel > tol_rel:
            bad.append(
                f"M={rows}: max abs {d:.3e} / rel L2 {rel:.2e} > {tol} / {tol_rel}"
            )
    torch.cuda.synchronize()
    assert not bad, "\n".join(bad)


def test_layer_graph_equals_eager(env):
    fam, ws, layer, dev = env
    for rows in (6, 24, 48):
        args = inputs(rows, 50 + rows, dev)
        eager = [t.clone() for t in _ours(fam, layer, *args)]
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = _ours(fam, layer, *args)
        for replay in range(3):
            graph.replay()
            torch.cuda.synchronize()
            assert all(same(o, e) for o, e in zip(outs, eager)), (
                f"M={rows} replay {replay}"
            )
        log(f"  graph M={rows:2d}: 3 replays bitwise equal to eager")


def _node_priorities(graph):
    """{priority: [kernel names]} of the captured graph's kernel nodes (driver API on
    torch's cudaGraph_t), or None when the graph or the attribute is not reachable."""
    from cuda.bindings import driver as cu

    def ok(res):
        if int(res[0]) != 0:
            raise RuntimeError(str(res[0]))
        return res[1:] if len(res) > 2 else res[1]

    # CUkernelNodeAttrID is CUlaunchAttributeID in cuda-python 13 (the PRIORITY member's
    # name follows the launch-attribute enum)
    attr = next(
        getattr(enum, name)
        for enum, name in (
            (getattr(cu, "CUlaunchAttributeID", None), "CU_LAUNCH_ATTRIBUTE_PRIORITY"),
            (getattr(cu, "CUkernelNodeAttrID", None), "CU_LAUNCH_ATTRIBUTE_PRIORITY"),
            (
                getattr(cu, "CUkernelNodeAttrID", None),
                "CU_KERNEL_NODE_ATTRIBUTE_PRIORITY",
            ),
        )
        if enum is not None and hasattr(enum, name)
    )
    g = cu.CUgraph(int(graph.raw_cuda_graph()))
    _, n = ok(cu.cuGraphGetNodes(g, 0))
    nodes, _ = ok(cu.cuGraphGetNodes(g, n))
    out = {}
    for node in nodes:
        if (
            ok(cu.cuGraphNodeGetType(node))
            != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
        ):
            continue
        val = ok(cu.cuGraphKernelNodeGetAttribute(node, attr))
        name = "?"
        try:
            params = ok(cu.cuGraphKernelNodeGetParams(node))
            fn = params.func if int(params.func) else cu.CUfunction(int(params.kern))
            name = ok(cu.cuFuncGetName(fn))
            name = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:  # noqa: BLE001 - a name is a nicety
            pass
        out.setdefault(int(val.priority), []).append(name[:60])
    return out


def test_shared_expert_priority_stream(env):
    from sgl_nvidia_mega.glm5_3.ll import moe

    fam, ws, layer, dev = env
    stream = moe.priority_stream()
    log(
        f"  priority stream: priority {stream.priority} (alt stream {layer[0].alt_stream.priority})"
    )
    for rows in (6, 24):
        args = inputs(rows, 70 + rows, dev)
        base = [t.clone() for t in _ours(fam, layer, *args)]
        prio = [t.clone() for t in _ours(fam, layer, *args, stream=stream)]
        assert all(same(a, b) for a, b in zip(base, prio)), (
            f"M={rows}: eager bits differ"
        )
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph(keep_graph=True)
        except TypeError:
            graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = _ours(fam, layer, *args, stream=stream)
        graph.replay()
        torch.cuda.synchronize()
        assert all(same(o, b) for o, b in zip(outs, base)), (
            f"M={rows}: replay bits differ"
        )
        try:
            prios = _node_priorities(graph)
        except Exception as e:  # noqa: BLE001
            prios = None
            log(
                f"  graph M={rows}: node priorities not readable ({type(e).__name__}: {e})"
            )
        if prios is not None:
            summary = {p: len(v) for p, v in sorted(prios.items())}
            log(f"  graph M={rows}: kernel nodes per priority {summary}")
            for p, names in sorted(prios.items()):
                if p != 0:
                    log(f"    priority {p}: {names}")
            assert any(p == stream.priority for p in prios), (
                f"M={rows}: no captured kernel node kept priority {stream.priority}: {summary}"
            )


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(
            pytest.main([__file__, "-q", "-s", "-p", "no:cacheprovider", *sys.argv[1:]])
        )
    from sglang.test.kernels.utils import multigpu_pytest_main

    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)
