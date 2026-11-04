# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM low-latency hops (``ll.hops``) on the plugin-owned Lamport buffer, on 4
SM100-family GPUs of one tray (W=4), or 8 GPUs of two trays (W=8, the R7 probe's
numerics: ``test_glm5_3_ll_hops_two_tray.py`` drives it).

Ported from the in-tree stack's ``08-mnnvl-hop-fast/test_hop_dist.py`` and
``45-unified-moe-hop/test_unified_hop.py``; the reference is FlashInfer's MNNVL
``allreduce_fusion`` on FlashInfer's own workspace.

1. Plain hop (P4), every row count 1-48 the TP4 verify / draft graphs replay plus
   edges: residual exact (up to the sign of zero) and norm within
   ``ll.hops.max_abs`` of the fp32 rank-order reference; bit-identical to
   FlashInfer's one-shot where the plan keeps FlashInfer's geometry and to
   FlashInfer's two-shot + rmsNormLamport where the plan is two-shot; unicast ==
   multicast for the same geometry.
2. Finalize hop (P9): equal bit for bit to SGLang's ``moe_finalize_fuse_shared``
   followed by the plain hop with the same launch (fp32 and bf16 routing weights,
   empty slots, with and without the shared output).
3. NVFP4 hop (P10): norm and residual bitwise to the plain hop with the same
   launch; norm, residual, FP4 codes and scales bitwise to FlashInfer's NVFP4 hop
   where the geometry is FlashInfer's; FP4 against ``fp4_quantize(norm)`` reported.
4. Lamport protocol: a seeded mix of 120 hops of every kind and row count, eager,
   then 24 of them in one CUDA graph replayed 3 times with fresh inputs; every
   result bitwise equal to the same hop run alone.
5. A per-hop microbenchmark (CUDA-graph replay): ours vs FlashInfer one-/two-shot.

Run with ``python test/kernels/test_glm5_3_ll_hops.py`` (relaunches itself under
torchrun on 4 GPUs).
"""

from __future__ import annotations

import os
import random
import statistics
import sys
import time

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="run via torchrun: python test/kernels/test_glm5_3_ll_hops.py",
)

H = 6144
EPS = 1e-5
TOP_K = 8
ALLOCATOR = os.environ.get("LL_HOP_ALLOCATOR", "symm_mem")
BENCH = os.environ.get("LL_HOP_BENCH", "1") == "1"
ROWS = {
    4: sorted(set(range(1, 17)) | {18, 20, 21, 24, 25, 30, 35, 36, 37, 40, 42, 45, 48}),
    8: [1, 2, 4, 5, 6, 8, 10, 11, 12, 16, 17, 18, 19, 24, 30, 36, 42, 48],
}


def log(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg, flush=True)


@pytest.fixture(scope="module")
def env():
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
        device_group=dist.group.WORLD,
        cpu_group=cpu,
        eps=EPS,
        deadline_s=10.0,
        allocator=ALLOCATOR,
    )
    fam.compile([48])
    log(f"[setup] W={world} sm={fam.sm_count} built + buffer in {time.time() - t0:.1f} s: "
        f"{fam.buffer.describe()}")  # fmt: skip
    ws = MNNVLAllReduceFusionWorkspace(
        Mapping(world_size=world, rank=rank, gpus_per_node=int(os.environ.get("LOCAL_WORLD_SIZE", world)), tp_size=world),
        buffer_size_in_bytes=transport.buffer_bytes(world, 48, H),
        comm_backend=TorchDistBackend(group=cpu),
    )  # fmt: skip
    yield fam, ws, world, rank, dev
    dist.barrier(group=cpu)


# ------------------------------------------------------------------ data / refs
def inputs(rows: int, seed: int, dev):
    """(this rank's partial, the replicated residual): outlier channels, exact zeros
    and -0.0 in both."""
    rank = dist.get_rank()
    g = torch.Generator().manual_seed(1000 * seed + 17)
    residual = torch.randn(rows, H, generator=g) * 0.6
    out = torch.randint(0, H, (24,), generator=g)
    residual[:, out] *= 60.0
    gr = torch.Generator().manual_seed(1000 * seed + 31 + rank)
    partial = torch.randn(rows, H, generator=gr) * 0.25
    partial[:, out[:4]] *= 40.0
    partial[0, :64] = 0.0
    partial[0, 64:128] = -0.0
    residual[0, 96:160] = -0.0
    return partial.to(dev, torch.bfloat16), residual.to(dev, torch.bfloat16)


def gamma_of(seed: int, dev):
    g = torch.Generator().manual_seed(7 + seed)
    return (1 + 0.2 * torch.randn(H, generator=g)).to(dev, torch.bfloat16)


def reference(partial, residual, gamma):
    world = dist.get_world_size()
    parts = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(parts, partial)
    acc = torch.zeros(partial.shape, dtype=torch.float32, device=partial.device)
    for p in parts:
        acc = acc + p.float()
    res = acc.to(torch.bfloat16) + residual
    sq = (res * res).float()
    rcp = torch.rsqrt(sq.sum(-1, keepdim=True) / H + EPS)
    return (res.float() * rcp * gamma.float()).to(torch.bfloat16), res


def same(a, b) -> bool:
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def same_value(a, b) -> bool:
    return torch.equal(a.float() + 0.0, b.float() + 0.0)


def max_abs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def max_ulp(a, ref) -> float:
    """The largest difference in bf16 ulps of ``ref`` (an ulp of a value in
    [2^e, 2^(e+1)) is 2^(e-7))."""
    r = ref.float()
    ulp = torch.ldexp(torch.ones_like(r), torch.frexp(r).exponent - 8)
    return ((a.float() - r).abs() / ulp).max().item()


def fi_hop(ws, x, res, gamma, oneshot: bool, fp4_scale=None):
    import flashinfer.comm as comm

    norm, r = torch.empty_like(x), torch.empty_like(res)
    kw = dict(
        input=x,
        workspace=ws,
        launch_with_pdl=True,
        residual_out=r,
        norm_out=norm,
        residual_in=res,
        rms_gamma=gamma,
        rms_eps=EPS,
        use_oneshot=oneshot,
    )
    if fp4_scale is None:
        comm.allreduce_fusion(
            pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm, **kw
        )
        return norm, r
    rows = x.shape[0]
    q = torch.empty((rows, H // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((rows, H // 16), dtype=torch.float8_e4m3fn, device=x.device)
    comm.allreduce_fusion(
        pattern=comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        quant_out=q,
        scale_out=sf,
        scale_factor=fp4_scale,
        layout_code=comm.QuantizationSFLayout.LINEAR,
        **kw,
    )
    return norm, r, q, sf


def moe_handoff(rows: int, seed: int, dev, wdtype=torch.float32, shared=True):
    """A synthetic trtllm-gen do_finalize=False handoff of this rank: FC2 rows in a
    permuted (padded) layout, top-8 weights, some empty slots."""
    from sgl_nvidia_mega.glm5_3.ll.hops import DeferredFinalize

    rank = dist.get_rank()
    g = torch.Generator().manual_seed(5000 * seed + 101 + rank)
    p = rows * TOP_K + 64
    gemm2 = (torch.randn(p, H, generator=g) * 0.5).to(dev, torch.bfloat16)
    perm = torch.randperm(p, generator=g)[: rows * TOP_K].view(rows, TOP_K).int()
    perm[0, -1] = -1
    if rows > 2:
        perm[2, 3] = -1
    w = torch.rand(rows, TOP_K, generator=g) * 0.3
    sh = (
        (torch.randn(rows, H, generator=g) * 0.4).to(dev, torch.bfloat16)
        if shared
        else None
    )
    return DeferredFinalize(gemm2, w.to(dev, wdtype), perm.to(dev), sh, TOP_K)


def finalize_ref(fin):
    from sglang.kernels.ops.moe.moe_finalize_fuse_shared import moe_finalize_fuse_shared

    return moe_finalize_fuse_shared(
        fin.gemm2_out,
        fin.permuted_idx.reshape(-1),
        fin.expert_weights,
        fin.shared_output,
        fin.top_k,
    )


def plans(fam, rows):
    from sgl_nvidia_mega.glm5_3.ll import hops

    a = (fam.world, rows, H, fam.sm_count)
    return hops.plain_plan(*a), hops.finalize_plan(*a), hops.nvfp4_plan(*a)


def fi_geometry(fam, rows):
    from sgl_nvidia_mega.glm5_3.ll import hops

    return hops.oneshot_geometry(rows, H, fam.sm_count)


# ------------------------------------------------------------------ 1. plain hop
def test_plain_hop(env):
    from sgl_nvidia_mega.glm5_3.ll import hops, transport

    fam, ws, world, rank, dev = env
    tol = hops.TOLERANCES["ll.hops.max_ulp"]
    bad = []
    for i, rows in enumerate(ROWS[world]):
        gamma = gamma_of(i % 3, dev)
        x, res = inputs(rows, i, dev)
        ref_norm, ref_res = reference(x, res, gamma)
        plan, _, _ = plans(fam, rows)
        norm, r = fam.allreduce_norm(x, res, gamma, site=7)
        line = f"M={rows:2d} {'ts' if plan.two_shot else ('uc' if plan.unicast else 'mc')}{plan.cluster}x{plan.block}"
        if not same_value(r, ref_res):
            bad.append(f"M={rows}: residual != fp32 rank-order reference")
        err = max_ulp(norm, ref_norm)
        if err > tol:
            bad.append(f"M={rows}: norm vs reference {err:.2f} ulp > {tol}")
        if plan.two_shot:
            fn, fr = fi_hop(ws, x, res, gamma, oneshot=False)
            ok = same(norm, fn) and same(r, fr)
            line += f" ==fi_twoshot {ok}"
            if not ok:
                bad.append(f"M={rows}: two-shot not bitwise to FlashInfer two-shot")
        elif (plan.block, plan.cluster) == fi_geometry(fam, rows):
            fn, fr = fi_hop(ws, x, res, gamma, oneshot=True)
            ok = same(norm, fn) and same(r, fr)
            line += f" ==fi_oneshot {ok}"
            if not ok:
                bad.append(f"M={rows}: one-shot not bitwise to FlashInfer one-shot "
                           f"(norm max diff {max_abs(norm, fn):.3e})")  # fmt: skip
            # the other publish, same geometry: same bits
            other = transport.Launch(
                plan.block, plan.cluster, not plan.unicast, plan.max_carveout
            )
            n2, r2 = transport.ar_rmsnorm(fam.buffer, x, res, gamma, EPS, other)
            if not (same(n2, norm) and same(r2, r)):
                bad.append(f"M={rows}: unicast != multicast")
        line += f" norm vs ref {err:.2f} ulp"
        log("  " + line)
    torch.cuda.synchronize()
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------ 2. finalize hop
@pytest.mark.parametrize("wdtype", [torch.float32, torch.bfloat16])
def test_finalize_hop(env, wdtype):
    from sgl_nvidia_mega.glm5_3.ll import hops, transport

    fam, ws, world, rank, dev = env
    tol = hops.TOLERANCES["ll.hops.max_ulp"]
    bad = []
    for i, rows in enumerate(ROWS[world]):
        for shared in (True, False) if rows in (5, 6, 24, 48) else (True,):
            gamma = gamma_of(i % 3, dev)
            fin = moe_handoff(rows, i, dev, wdtype, shared)
            _, res = inputs(rows, i, dev)
            _, plan, _ = plans(fam, rows)
            norm, r = fam.finalize(fin, res, gamma, site=11)
            part = finalize_ref(fin)
            n2, r2 = transport.ar_rmsnorm(fam.buffer, part, res, gamma, EPS, plan)
            if not (same(norm, n2) and same(r, r2)):
                bad.append(f"M={rows} shared={shared}: finalize hop != finalize + hop "
                           f"(norm {max_abs(norm, n2):.3e})")  # fmt: skip
            ref_norm, ref_res = reference(part, res, gamma)
            if not same_value(r, ref_res) or max_ulp(norm, ref_norm) > tol:
                bad.append(f"M={rows}: finalize hop vs reference")
        log(
            f"  finalize {str(wdtype)[6:]} M={rows:2d} {'ts' if plan.two_shot else ('uc' if plan.unicast else 'mc')}{plan.cluster}x{plan.block}"
        )
    torch.cuda.synchronize()
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------ 3. NVFP4 hop
def test_nvfp4_hop(env):
    from flashinfer import fp4_quantize

    from sgl_nvidia_mega.glm5_3.ll import hops, transport

    fam, ws, world, rank, dev = env
    gs = torch.tensor([448.0 * 6.0 / 37.0], dtype=torch.float32, device=dev)
    bad = []
    for i, rows in enumerate(ROWS[world]):
        plan = hops.plain_plan(world, rows, H, fam.sm_count)
        gamma = gamma_of(i % 3, dev)
        x, res = inputs(rows, i, dev)
        n, r, q, sf = transport.ar_rmsnorm_nvfp4(
            fam.buffer, x, res, gamma, EPS, plan, global_scale=gs
        )
        n2, r2 = transport.ar_rmsnorm(fam.buffer, x, res, gamma, EPS, plan)
        if not (same(n, n2) and same(r, r2)):
            bad.append(f"M={rows}: NVFP4 hop norm / residual != plain hop")
        line = f"  nvfp4 M={rows:2d}"
        geo_fi = plan.two_shot or (plan.block, plan.cluster) == fi_geometry(fam, rows)
        if geo_fi:
            fn, fr, fq, fsf = fi_hop(
                ws, x, res, gamma, oneshot=not plan.two_shot, fp4_scale=gs
            )
            ok = same(n, fn) and same(r, fr) and same(q, fq) and same(sf, fsf)
            line += f" ==fi_nvfp4 {ok}"
            if not ok:
                bad.append(f"M={rows}: NVFP4 hop != FlashInfer's (codes "
                           f"{(q != fq).float().mean().item():.2e}, scales {same(sf, fsf)})")  # fmt: skip
        rq, rsf = fp4_quantize(n, gs, 16, False, False)
        diff = (q != rq.view(q.shape)).float().mean().item()
        line += f" codes vs fp4_quantize {diff:.2e}"
        if diff > 1e-2:
            bad.append(f"M={rows}: FP4 codes vs fp4_quantize(norm) {diff:.2e}")
        if fam.nvfp4 and hops.nvfp4_skipped(world, rows):
            line += " (P10 skip row)"
        log(line)
    torch.cuda.synchronize()
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------ 4. protocol
def _op(fam, kind, rows, seed, dev):
    """One hop of ``kind`` on inputs seeded by (seed, rows): its outputs."""
    from sgl_nvidia_mega.glm5_3.ll import hops, transport

    gamma = gamma_of(seed % 3, dev)
    if kind == "finalize":
        fin = moe_handoff(rows, seed, dev)
        _, res = inputs(rows, seed, dev)
        return lambda: fam.finalize(fin, res, gamma, site=1)
    x, res = inputs(rows, seed, dev)
    if kind == "nvfp4":
        gs = torch.tensor([448.0 * 6.0 / 29.0], dtype=torch.float32, device=dev)
        plan = hops.plain_plan(fam.world, rows, H, fam.sm_count)
        return lambda: transport.ar_rmsnorm_nvfp4(
            fam.buffer, x, res, gamma, EPS, plan, global_scale=gs
        )
    return lambda: fam.allreduce_norm(x, res, gamma, site=0)


def test_protocol_mix_eager_and_graph(env):
    fam, ws, world, rank, dev = env
    rng = random.Random(1234)  # same sequence on every rank
    kinds = ("plain", "finalize", "nvfp4")
    seq = [
        (rng.choice(kinds), rng.choice(ROWS[world]), rng.randrange(50))
        for _ in range(120)
    ]
    expected, fns = {}, {}
    for key in sorted(set(seq)):  # the same order on every rank
        fns[key] = _op(fam, *key, dev)
        expected[key] = [t.clone() for t in fns[key]()]
    torch.cuda.synchronize()
    for i, key in enumerate(seq):
        got = fns[key]()
        assert all(same(g, e) for g, e in zip(got, expected[key])), (
            f"eager op {i} {key}"
        )
    torch.cuda.synchronize()
    log(f"  eager mix of {len(seq)} hops: bitwise")
    # a CUDA graph of 24 hops, replayed with fresh inputs
    sub = seq[:24]
    graph_ops = {k: _op(fam, *k, dev) for k in sorted(set(sub))}
    for k in sorted(graph_ops):
        graph_ops[k]()  # warm
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    outs = []
    with torch.cuda.graph(graph):
        for k in sub:
            outs.append(graph_ops[k]())
    for replay in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for k, o in zip(sub, outs):
            assert all(same(g, e) for g, e in zip(o, expected[k])), (
                f"replay {replay} {k}"
            )
        fns[seq[-1]]()  # an eager hop between replays moves the buffer index
    torch.cuda.synchronize()
    log("  CUDA graph of 24 hops x 3 replays (eager hops between): bitwise")


# ------------------------------------------------------------------ 5. timing
def _time(fn, n=20, reps=15) -> float:
    graph = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(n):
            fn()
    ts = []
    for _ in range(reps):
        dist.barrier()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / n)
    t = torch.tensor([statistics.median(ts)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


@pytest.mark.skipif(not BENCH, reason="LL_HOP_BENCH=0")
def test_bench(env):
    from sgl_nvidia_mega.glm5_3.ll import hops

    fam, ws, world, rank, dev = env
    log("  op      M  plan            ours_us  fi_one  fi_two")
    for rows in [
        r
        for r in ROWS[world]
        if r in (1, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24, 30, 36, 42, 48)
    ]:
        gamma = gamma_of(0, dev)
        x, res = inputs(rows, 3, dev)
        plan = hops.plain_plan(world, rows, H, fam.sm_count)
        ours = _time(lambda: fam.allreduce_norm(x, res, gamma, site=0))
        fio = _time(lambda: fi_hop(ws, x, res, gamma, True))
        fit = _time(lambda: fi_hop(ws, x, res, gamma, False))
        label = f"{'ts' if plan.two_shot else ('uc' if plan.unicast else 'mc')}{plan.cluster}x{plan.block}{'+c' if plan.max_carveout else ''}"
        log(f"  plain  {rows:2d}  {label:14s} {ours:7.2f} {fio:7.2f} {fit:7.2f}")
        fin = moe_handoff(rows, 3, dev)
        fplan = hops.finalize_plan(world, rows, H, fam.sm_count)
        tf = _time(lambda: fam.finalize(fin, res, gamma, site=0))
        tsep = _time(lambda: fam.allreduce_norm(finalize_ref(fin), res, gamma, site=0))
        label = f"{'ts' if fplan.two_shot else ('uc' if fplan.unicast else 'mc')}{fplan.cluster}x{fplan.block}"
        log(
            f"  final  {rows:2d}  {label:14s} {tf:7.2f}  (finalize kernel + hop {tsep:.2f})"
        )


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(
            pytest.main([__file__, "-q", "-s", "-p", "no:cacheprovider", *sys.argv[1:]])
        )
    from sglang.test.kernels.utils import multigpu_pytest_main

    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)
