# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_rowselect, the fused draft's draft-extend row select, on 4 SM100-family GPUs (GB300).

- Exactness of the selection: ``draft_attention.select_rows`` on a prepared stream,
  then a row-wise MLP stand-in, SGLang's residual stream and the NextN final fused
  add + RMSNorm (``batch.final_norm``): the selected rows equal the full-row path's
  rows ``[select_index]`` bit for bit, at 5 and 6 tokens per request and 1-8 requests,
  eager and replayed from a CUDA graph whose ``select_index`` buffer is rewritten
  between replays (as the draft-extend runner does).
- The draft MoE on the selected rows: per rank, the kernels SGLang runs for GLM-5's BF16
  NextN MoE (the router GEMM, FlashInfer ``trtllm_bf16_moe`` with weights shuffled as
  ``UnquantizedFusedMoEMethod`` does and autotuned, the BF16 shared expert), the TP
  all-reduce and the final fused add + RMSNorm, on every draft-extend row vs on the
  ``select_index`` rows. The MoE picks its tactic and the router its GEMM by the row
  count, so this is not bit for bit: ``draft.rowselect.mlp`` in ``draft.NUMERICS``
  bounds the rows whose routing is unchanged (BF16 accumulation-order rounding is
  ~1e-3, a wrong row ~1) and the share of rows whose top-8 near-tie flips.
- Rank 0 times the MoE and the whole MLP block from CUDA-graph replay, cold (L2 flushed,
  the flush subtracted) and hot, and writes ``$TEST_OUT/draft_rowselect.{md,json}``.

Shapes per rank: TP8 (I = 256) at 6 tokens per request (EAGLE 5/1/6, the 8-rank group
emulated on 4 ranks: the kernels' shapes are per rank) and TP4 (I = 512) at 5 and 6.
With ``MODEL`` set, the router and its correction bias are the checkpoint's NextN ones.

Run with ``python test/kernels/test_glm5_3_draft_rowselect.py --num-gpu 4 -s``.
"""

import json
import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sglang.test.kernels.utils import multigpu_pytest_main

from sgl_nvidia_mega.glm5_3 import draft_attention
from sgl_nvidia_mega.glm5_3.draft import NUMERICS

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="run via torchrun: python test/kernels/test_glm5_3_draft_rowselect.py --num-gpu 4 -s",
)

H, E, TOPK, MOE_INTER, RSF, EPS = 6144, 256, 8, 2048, 2.5, 1e-5
TINY_MAX_M = 16
# name -> (TP size of the per-rank shapes, tokens per request, requests per batch)
CONFIGS = {
    "tp8-516": (8, 6, (1, 2)),
    "tp4-516": (4, 6, (1, 2, 4, 8)),
    "tp4-415": (4, 5, (1, 2, 4, 8)),
}
MAX_ROW_REL = NUMERICS.tolerances["draft.rowselect.mlp.row_rel"]
MIN_ROW_COS = NUMERICS.tolerances["draft.rowselect.mlp.row_cos"]
MAX_FLIP_FRACTION = NUMERICS.tolerances["draft.rowselect.mlp.flip_fraction"]


def log(*a):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, flush=True)


@pytest.fixture(scope="module")
def rank():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    yield rank
    dist.barrier()
    dist.destroy_process_group()


def _select_index(bs, w, gen):
    accept = torch.randint(0, w, (bs,), generator=gen, device="cuda")
    return torch.arange(bs, device="cuda") * w + accept


def _gamma(gen):
    return (1.0 + 0.1 * torch.randn(H, generator=gen, device="cuda")).to(torch.bfloat16)


class _FinalNorm:
    """The NextN final norm: a fused add + RMSNorm in place on (x, residual)."""

    def __init__(self, weight):
        from sgl_kernel import fused_add_rmsnorm

        self._kernel, self.weight = fused_add_rmsnorm, weight

    def __call__(self, x, residual):
        self._kernel(x, residual, self.weight, EPS)
        return x, residual


# --------------------------------------------------------------- exactness of the selection


class _Batch:
    def __init__(self, num_tokens, residual):
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

        self.input_ids = torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
        self.residual_stream = ResidualStream()
        # the state the MLP prepare leaves: a written residual, nothing pending
        self.residual_stream.write(residual)


def _mlp(x):
    # a row-wise stand-in for the MLP: any row-independent op keeps the rows exact
    return F.silu(x.float()).mul(0.5).to(x.dtype)


def _tail(x, fb, norm):
    """The MLP output recorded on the stream, then the stack's completion and final norm."""
    from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
    from sglang.srt.layers.layer_boundary.residual.batch import (
        complete_output,
        final_norm,
    )

    handle = fb.residual_stream.record(_mlp(x), PLAIN_ADD)
    return final_norm(complete_output(handle, fb), fb, norm)


@pytest.mark.parametrize("w", [5, 6])
def test_selected_rows_equal_the_full_rows(rank, w):
    gen = torch.Generator(device="cuda").manual_seed(1234 + w)
    norm = _FinalNorm(_gamma(gen))
    for bs in (1, 2, 3, 4, 8):
        n = bs * w
        x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=gen)
        res = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=gen)
        sel = _select_index(bs, w, gen)
        ref = _tail(x, _Batch(n, res.clone()), norm)[sel]
        fb = _Batch(n, res.clone())
        xs = draft_attention.select_rows(x, fb, sel)
        assert torch.equal(fb.residual_stream.residual, res[sel])
        out = _tail(xs, fb, norm)
        assert out.shape[0] == bs and torch.equal(out, ref), f"w {w} bs {bs}"
    log(
        f"select_rows at {w} tokens per request: bitwise equal to full rows [select_index]"
    )


@pytest.mark.parametrize("w", [5, 6])
def test_graph_replay_follows_the_rewritten_select_index(rank, w):
    gen = torch.Generator(device="cuda").manual_seed(99 + w)
    norm = _FinalNorm(_gamma(gen))
    bs = 4
    n = bs * w
    x_buf = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=gen)
    res_src = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=gen)
    res_buf = res_src.clone()
    sel_buf = torch.arange(bs, device="cuda", dtype=torch.int64) * w + w - 1

    def run():
        fb = _Batch(n, res_buf)
        return _tail(draft_attention.select_rows(x_buf, fb, sel_buf), fb, norm)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = run()
    for trial in range(5):
        sel = _select_index(bs, w, gen)
        sel_buf.copy_(sel)
        x_buf.copy_(
            torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=gen)
        )
        res_buf.copy_(res_src)
        graph.replay()
        ref = _tail(x_buf.clone(), _Batch(n, res_src.clone()), norm)[sel]
        assert torch.equal(out_static, ref), f"replay {trial}"
        # the selection copies the residual rows: the full-row buffer stays intact
        assert torch.equal(res_buf, res_src), f"replay {trial}"
    log(
        f"select_rows at {w} tokens per request: graph replay with a rewritten select_index matches eager"
    )


# --------------------------------------------------------------- the draft MoE on the selected rows


def _next_pow2(x):
    return 1 << (max(int(x), 1) - 1).bit_length()


def _load_gate(model_dir):
    """The checkpoint's NextN router weight and correction bias, or None."""
    from safetensors import safe_open

    with open(os.path.join(model_dir, "config.json")) as f:
        layer = json.load(f)["num_hidden_layers"]
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    out = []
    for name in (
        f"model.layers.{layer}.mlp.gate.weight",
        f"model.layers.{layer}.mlp.gate.e_score_correction_bias",
    ):
        with safe_open(os.path.join(model_dir, weight_map[name]), framework="pt") as fh:
            out.append(fh.get_tensor(name))
    w, b = out
    assert tuple(w.shape) == (E, H), w.shape
    return w.cuda().to(torch.bfloat16), b.cuda().float()


def _shuffle_for_trtllm_bf16(w13, w2):
    """In place, as UnquantizedFusedMoEMethod.process_weights_after_loading prepares
    FlashInfer's trtllm BF16 MoE weights."""
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    cache = {}
    old13, old2 = w13[0].shape, w2[0].shape
    new13 = new2 = None
    for i in range(w13.shape[0]):
        p = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13[i].view(torch.uint8), 128, is_gated_act_gemm=True
        )
        t1 = w13[i].clone().view(torch.uint8)[p.to(w13.device)].contiguous()
        p = get_w2_permute_indices_with_cache(cache, w2[i].view(torch.uint8), 128)
        t2 = w2[i].clone().view(torch.uint8)[p.to(w2.device)].contiguous()
        t1 = convert_to_block_layout(t1.view(torch.uint8), 128)
        t2 = convert_to_block_layout(t2.view(torch.uint8), 128)
        new13, new2 = t1.view(torch.bfloat16).shape, t2.view(torch.bfloat16).shape
        w13[i] = t1.view(torch.bfloat16).contiguous().reshape(old13)
        w2[i] = t2.view(torch.bfloat16).contiguous().reshape(old2)
    return w13.reshape(w13.shape[0], *new13), w2.reshape(w2.shape[0], *new2)


class _NextnMlp:
    """One rank's NextN MoE block (routed + shared, before the all-reduce)."""

    def __init__(self, inter, rank, gate):
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm

        g = torch.Generator(device="cuda").manual_seed(7 + rank)
        self.inter = inter
        w13 = torch.randn(
            E, 2 * inter, H, generator=g, device="cuda", dtype=torch.bfloat16
        )
        w2 = torch.randn(E, H, inter, generator=g, device="cuda", dtype=torch.bfloat16)
        w13.mul_(1.0 / math.sqrt(H))
        w2.mul_(1.0 / math.sqrt(inter))
        self.w13, self.w2 = _shuffle_for_trtllm_bf16(w13, w2)
        self.w_gate_up = (
            torch.randn(2 * inter, H, generator=g, device="cuda") / math.sqrt(H)
        ).to(torch.bfloat16)
        self.w_down = (
            torch.randn(H, inter, generator=g, device="cuda") / math.sqrt(inter)
        ).to(torch.bfloat16)
        self.w_router, self.bias = gate
        self.tiny_ok = can_use_tiny_gemm(E, H, max_m=TINY_MAX_M)

    def router(self, x):
        # MoEGate.forward: the tiny GEMM up to its row bound, else the BF16 -> FP32 GEMM
        if self.tiny_ok and x.shape[0] <= TINY_MAX_M:
            from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

            return tiny_gemm_bf16(
                x, self.w_router, out_dtype=torch.float32, max_m=TINY_MAX_M
            )
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

        return linear_bf16_fp32(x, self.w_router)

    def routed(self, x, logits, replay=None):
        from flashinfer.fused_moe import (
            ActivationType,
            RoutingMethodType,
            trtllm_bf16_moe,
        )

        kw = {} if replay is None else {"routing_replay_out": replay}
        out = trtllm_bf16_moe(
            routing_logits=logits,
            routing_bias=self.bias,
            hidden_states=x,
            gemm1_weights=self.w13,
            gemm2_weights=self.w2,
            num_experts=E,
            top_k=TOPK,
            n_group=1,
            topk_group=1,
            intermediate_size=self.inter,
            local_expert_offset=0,
            local_num_experts=E,
            routing_method_type=int(RoutingMethodType.DeepSeekV3),
            routed_scaling_factor=RSF,
            tune_max_num_tokens=_next_pow2(x.shape[0]),
            activation_type=int(ActivationType.Swiglu),
            **kw,
        )
        return out[0] if isinstance(out, (list, tuple)) else out

    def shared(self, x):
        gu = F.linear(x, self.w_gate_up)
        return F.linear(F.silu(gu[:, : self.inter]) * gu[:, self.inter :], self.w_down)

    def __call__(self, x):
        return self.routed(x, self.router(x)) + self.shared(x)

    def autotune(self, x):
        from flashinfer import autotune

        with torch.inference_mode(), autotune(True):
            self(x)
        torch.cuda.synchronize()


@pytest.fixture(scope="module")
def gate(rank):
    model = os.environ.get("MODEL", "")
    real = None
    if rank == 0 and model and os.path.exists(os.path.join(model, "config.json")):
        real = _load_gate(model)
    flag = torch.tensor([real is not None], device="cuda", dtype=torch.int32)
    dist.broadcast(flag, 0)
    if real is None and flag.item():
        real = (
            torch.empty(E, H, dtype=torch.bfloat16, device="cuda"),
            torch.empty(E, dtype=torch.float32, device="cuda"),
        )
    if real is None:
        g = torch.Generator(device="cuda").manual_seed(3)
        real = (
            (torch.randn(E, H, generator=g, device="cuda") * (2.0 / math.sqrt(H))).to(
                torch.bfloat16
            ),
            0.01 * torch.randn(E, generator=g, device="cuda"),
        )
    for t in real:
        dist.broadcast(t, 0)
    log(f"router: {'the checkpoint NextN gate' if flag.item() else 'random'}")
    return real


def _inputs(cfg_id, bs, w):
    g = torch.Generator(device="cuda").manual_seed(100 * cfg_id + bs)
    n = bs * w
    x = torch.randn(n, H, generator=g, device="cuda", dtype=torch.bfloat16)
    res = (4 * torch.randn(n, H, generator=g, device="cuda")).to(torch.bfloat16)
    sel = _select_index(bs, w, g)
    return x, res, sel


def _graph_us(fn, flush=None, iters=100, warmup=10):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.inference_mode():
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        if flush is not None:
            flush()
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    def timed():
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / iters

    return graph, timed


def _bench_pair(old_fn, new_fn, flush, reset_old, rounds=3):
    """Medians over ABBA rounds; with the flush, minus a flush-only graph."""
    g_old, t_old = _graph_us(old_fn, flush)
    g_new, t_new = _graph_us(new_fn, flush)
    base = 0.0
    if flush is not None:
        _, t_flush = _graph_us(lambda: None, flush)
        base = sorted(t_flush() for _ in range(3))[1]

    def timed_old():
        reset_old()
        return t_old()

    olds, news = [], []
    for _ in range(rounds):
        olds.append(timed_old())
        news += [t_new(), t_new()]
        olds.append(timed_old())
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    del g_old, g_new
    return med(olds) - base, med(news) - base


def test_draft_moe_on_the_selected_rows(rank, gate):
    world = dist.get_world_size()
    gen = torch.Generator(device="cuda").manual_seed(11)
    norm = _FinalNorm(_gamma(gen))
    numerics, bench, failures = [], [], []
    for cfg_id, (name, (tp, w, batch_sizes)) in enumerate(CONFIGS.items()):
        mlp = _NextnMlp(MOE_INTER // tp, rank, gate)
        flips = rows_total = 0
        for bs in batch_sizes:
            x, res, sel = _inputs(cfg_id, bs, w)
            xs = x.index_select(0, sel)
            mlp.autotune(x)
            mlp.autotune(xs)
            with torch.inference_mode():
                # every draft-extend row, then the logits processor's gather
                y = mlp(x)
                dist.all_reduce(y)
                ref = norm(y, res.clone())[0][sel]
                # the select_index rows of the MLP input and of the residual
                y = mlp(xs)
                dist.all_reduce(y)
                out = norm(y, res.index_select(0, sel))[0]
                # the kernel's own routing of the same rows, to tell flips from rounding
                rep_old = torch.full(
                    (x.shape[0], TOPK), -1, dtype=torch.int16, device="cuda"
                )
                rep_new = torch.full((bs, TOPK), -1, dtype=torch.int16, device="cuda")
                mlp.routed(x, mlp.router(x), rep_old)
                mlp.routed(xs, mlp.router(xs), rep_new)
                flipped = (
                    rep_old.long()[sel].sort(dim=-1).values
                    != rep_new.long().sort(dim=-1).values
                ).any(dim=-1)
            a, b = ref.float(), out.float()
            row_rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)
            row_cos = F.cosine_similarity(a, b, dim=-1)
            keep = ~flipped
            worst_rel = float(row_rel[keep].max()) if keep.any() else 0.0
            worst_cos = float(row_cos[keep].min()) if keep.any() else 1.0
            flips += int(flipped.sum())
            rows_total += bs
            row = dict(
                cfg=name,
                bs=bs,
                m_old=x.shape[0],
                m_new=bs,
                bitwise=bool(torch.equal(ref, out)),
                equal_frac=float((ref == out).float().mean()),
                rel_l2=float((a - b).norm() / a.norm().clamp_min(1e-12)),
                worst_row_rel=worst_rel,
                worst_row_cos=worst_cos,
                routing_flips=int(flipped.sum()),
                experts_old=int(rep_old.long().unique().numel()),
                experts_new=int(rep_new.long().unique().numel()),
            )
            numerics.append(row)
            log(
                f"[{name}] bs {bs} M {x.shape[0]} -> {bs}: bitwise {row['bitwise']} "
                f"equal {row['equal_frac']:.4f} rel_l2 {row['rel_l2']:.2e} "
                f"worst row rel {worst_rel:.2e} cos {worst_cos:.6f} flips {row['routing_flips']}"
            )
            if worst_rel > MAX_ROW_REL or worst_cos < MIN_ROW_COS:
                failures.append(
                    f"{name} bs {bs}: row rel {worst_rel:.2e} cos {worst_cos:.6f}"
                )
        if flips > max(1, MAX_FLIP_FRACTION * rows_total):
            failures.append(f"{name}: {flips} of {rows_total} rows flip their routing")
        dist.barrier()
        if rank == 0:
            bench += _microbench(mlp, norm, name, cfg_id, w, batch_sizes)
        dist.barrier()
        del mlp
        torch.cuda.empty_cache()
    if rank == 0:
        _report(world, numerics, bench, failures)
    flag = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    assert flag.item() == 0, "; ".join(failures) or "a peer rank failed"


def _microbench(mlp, norm, name, cfg_id, w, batch_sizes):
    flush_buf = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    flush = flush_buf.zero_
    rows = []
    for bs in batch_sizes:
        x, res, sel = _inputs(cfg_id, bs, w)
        xs = x.index_select(0, sel)
        logits_old, logits_new = mlp.router(x), mlp.router(xs)
        # the old block's fused add + RMSNorm writes its residual in place: a private
        # copy, restored (untimed) before each timing; the new block reads `res` only
        res_old = res.clone()

        def block_old():
            return norm(mlp(x), res_old)[0].index_select(0, sel)

        def block_new():
            return norm(mlp(x.index_select(0, sel)), res.index_select(0, sel))[0]

        ops = {
            "moe_trtllm_bf16": (
                lambda: mlp.routed(x, logits_old),
                lambda: mlp.routed(xs, logits_new),
                lambda: None,
            ),
            "nextn_mlp_block": (block_old, block_new, lambda: res_old.copy_(res)),
        }
        for op, (old_fn, new_fn, reset) in ops.items():
            cold = _bench_pair(old_fn, new_fn, flush, reset)
            hot = _bench_pair(old_fn, new_fn, None, reset)
            rows.append(
                dict(
                    op=op,
                    cfg=name,
                    bs=bs,
                    m_old=x.shape[0],
                    m_new=bs,
                    old_us=cold[0],
                    new_us=cold[1],
                    hot_old_us=hot[0],
                    hot_new_us=hot[1],
                )
            )
            log(
                f"[bench {name}] {op} bs {bs} M {x.shape[0]} -> {bs}: cold {cold[0]:.1f} -> "
                f"{cold[1]:.1f} us, hot {hot[0]:.1f} -> {hot[1]:.1f} us"
            )
    del flush_buf
    return rows


def _report(world, numerics, bench, failures):
    lines = [
        f"## Draft-extend row select (FU-1): the draft MoE on all rows vs the select_index rows ({world} ranks)",
        "",
        "| cfg | bs | M old->new | bitwise | equal frac | rel L2 | worst row rel | worst row cos | routing flips | experts old->new |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in numerics:
        lines.append(
            f"| {r['cfg']} | {r['bs']} | {r['m_old']}->{r['m_new']} | {r['bitwise']} | {r['equal_frac']:.4f} | "
            f"{r['rel_l2']:.2e} | {r['worst_row_rel']:.2e} | {r['worst_row_cos']:.6f} | {r['routing_flips']} | "
            f"{r['experts_old']}->{r['experts_new']} |"
        )
    lines += [
        "",
        "## CUDA-graph replay on rank 0 (cold: L2 flushed, flush subtracted)",
        "",
        "| op | cfg | M/bs | old us | new us | saved us | hot old us | hot new us |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in bench:
        lines.append(
            f"| {r['op']} | {r['cfg']} | {r['m_old']}/{r['bs']} | {r['old_us']:.1f} | {r['new_us']:.1f} | "
            f"{r['old_us'] - r['new_us']:.1f} | {r['hot_old_us']:.1f} | {r['hot_new_us']:.1f} |"
        )
    lines += ["", "failures: " + ("; ".join(failures) if failures else "none")]
    text = "\n".join(lines)
    print(text, flush=True)
    out = os.environ.get("TEST_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "draft_rowselect.md"), "w") as f:
            f.write(text + "\n")
        with open(os.path.join(out, "draft_rowselect.json"), "w") as f:
            json.dump(
                dict(world=world, numerics=numerics, bench=bench, failures=failures),
                f,
                indent=1,
            )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)
