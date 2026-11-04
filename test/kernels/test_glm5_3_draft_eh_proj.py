# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""P6a, the draft's row-parallel eh_proj at TP4 (the ``ll.draft_eh`` unit,
``ll.draft.RowParallelEhFamily``), on 4 SM100-family GPUs (GB300).

- Numerics at every TP4 draft row count (1-48): the served path (each rank's K window
  of the front's output times its weight columns, SGLang's TP all-reduce of the BF16
  partials, replayed from a CUDA graph through the unit's ``eh_proj`` step) is the FP32 sum
  of the four BF16 partials within ``ll.draft_eh.sum_max_ulp`` BF16 ulp (the all-reduce's
  own order and rounding), and its error against FP64 is within
  ``ll.draft_eh.max_rel_rms`` of each row's RMS (each partial rounds to BF16 before the
  sum, as in every row-parallel projection; the stock replicated GEMM's error is
  reported beside it). Every rank holds the same result.
- With the target's TP4 LL hop (``ll.hops``, when present): the unit sums the partials
  and applies the draft layer's input norm in the hop (``PreparedInput``): the residual
  is the FP32 rank-order sum of the BF16 partials and the norm within ``ll.hops.max_ulp``
  of RMSNorm of it, replayed from a CUDA graph.

Run with ``python test/kernels/test_glm5_3_draft_eh_proj.py --num-gpu 4 -s``.
"""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

from sgl_nvidia_mega.glm5_3 import draft_units as draft
from sgl_nvidia_mega.glm5_3.ll.draft import (
    TOLERANCES,
    RowParallelEhFamily,
    RowParallelEhProj,
)
from sgl_nvidia_mega.glm5_3.runtime import DerivedWeights

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="run via torchrun: python test/kernels/test_glm5_3_draft_eh_proj.py --num-gpu 4 -s",
)

H = draft.HIDDEN
ROWS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 18, 24, 30, 36, 48)
SUM_MAX_ULP = TOLERANCES["ll.draft_eh.sum_max_ulp"]
MAX_REL_RMS = TOLERANCES["ll.draft_eh.max_rel_rms"]


def log(*a):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, flush=True)


@pytest.fixture(scope="module")
def tp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # SGLang's default all-reduce dispatch (custom all-reduce on one tray), as the draft's other sums
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield ps.get_tp_group()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


@pytest.fixture(scope="module")
def eh(tp):
    # the same weight and inputs on every rank, as the replicated checkpoint tensor is
    gen = torch.Generator(device="cuda").manual_seed(17)
    w = (torch.randn(H, 2 * H, generator=gen, device="cuda") / (2 * H) ** 0.5).to(
        torch.bfloat16
    )
    x = torch.randn(max(ROWS), 2 * H, generator=gen, device="cuda").to(torch.bfloat16)
    return w, x


def _p6a(w):
    """The ll.draft_eh family built on a NextN stand-in holding ``w`` (this rank's TP group)."""
    nextn = SimpleNamespace(model=SimpleNamespace(eh_proj=SimpleNamespace(weight=w)))
    family = RowParallelEhFamily(nextn, DerivedWeights(), None)
    family.compile(())
    return family


def _capture(fn):
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with ps.graph_capture() as context:
        with torch.cuda.graph(graph, stream=context.stream):
            out = fn()
    torch.cuda.synchronize()
    return graph, out


def _ulps(got, ref):
    """max |got - ref| in BF16 ulp at |ref| (normal range)."""
    ref = ref.double()
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(2.0**-126))) - 7)
    return ((got.double() - ref).abs() / ulp).max().item()


def _rel_rms(got, ref64):
    """max |got - ref| over each row's RMS."""
    rms = ref64.pow(2).mean(dim=-1, keepdim=True).sqrt()
    return ((got.double() - ref64).abs() / rms).max().item()


def test_row_parallel_eh_proj_within_the_named_ulps(tp, eh):
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    w, x_all = eh
    p6a = _p6a(w)
    # cuBLAS and the all-reduce initialize outside the capture, as in a server
    tensor_model_parallel_all_reduce(p6a.rp.partial(x_all[:1].contiguous()))
    lines = []
    for rows in ROWS:
        x = x_all[:rows].contiguous()
        p6a.rp.partial(x)
        graph, out = _capture(lambda: p6a.provider(None, rows, "decode").eh_proj(x))
        assert out is not None
        graph.replay()
        torch.cuda.synchronize()
        ref = x.double() @ w.double().T
        stock = torch.nn.functional.linear(x, w)
        # every rank's BF16 partial (the same cuBLAS GEMM), summed in FP32 in rank order
        partials = [
            RowParallelEhProj(w, r, tp.world_size).partial(x)
            for r in range(tp.world_size)
        ]
        fp32_sum = (
            torch.stack([p.float() for p in partials]).sum(dim=0).to(torch.bfloat16)
        )
        sum_ulps = _ulps(out, fp32_sum)
        rel, stock_rel = _rel_rms(out, ref), _rel_rms(stock, ref)
        lines.append(f"| {rows} | {stock_rel:.2e} | {rel:.2e} | {sum_ulps:.0f} |")
        assert sum_ulps <= SUM_MAX_ULP, (
            f"{rows} rows: {sum_ulps} ulp from the FP32 sum of the partials"
        )
        assert rel <= MAX_REL_RMS, (
            f"{rows} rows: {rel:.2e} of the row RMS from FP64 (stock {stock_rel:.2e})"
        )
        gathered = [torch.empty_like(out) for _ in range(tp.world_size)]
        torch.distributed.all_gather(gathered, out, group=tp.device_group)
        assert all(torch.equal(g, out) for g in gathered), f"{rows} rows: ranks differ"
    log(
        "P6a numerics at TP4 (max |error| vs FP64 over the row RMS; ulp from the FP32 sum of the BF16 partials)\n"
        "| rows | stock GEMM | P6a | P6a vs partial sum (ulp) |\n|---|---|---|---|\n"
        + "\n".join(lines)
    )


def _graph_us(fn, flush, iters=100):
    """Per call: cold = each replay right after an L2 flush (the flush keeps the GPU busy
    while the replay is queued), hot = back-to-back replays."""
    graph, _ = _capture(fn)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    if flush is None:
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e3 / iters
    graph_flush, _ = _capture(flush)
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        graph_flush.replay()
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in events) * 1e3 / iters


@pytest.mark.skipif(
    importlib.util.find_spec("sgl_nvidia_mega.glm5_3.ll.hops") is None,
    reason="the TP4 LL hops (ll.hops) are not in this tree",
)
def test_row_parallel_sum_in_the_ll_hop(tp, eh, monkeypatch):
    from sgl_nvidia_mega.glm5_3.draft_attention import PreparedInput
    from sgl_nvidia_mega.glm5_3.ll import draft as ll_draft
    from sgl_nvidia_mega.glm5_3.ll import hops

    eps = 1e-5
    w, x_all = eh
    hop = hops.LlHopFamily(
        device_group=tp.device_group, cpu_group=tp.cpu_group, eps=eps, deadline_s=10.0
    )
    hop.compile([max(ROWS)])
    monkeypatch.setattr(ll_draft, "target_hop_family", lambda nextn: hop)
    gen = torch.Generator(device="cuda").manual_seed(23)
    gamma = (1 + 0.2 * torch.randn(H, generator=gen, device="cuda")).to(torch.bfloat16)
    norm_module = SimpleNamespace(weight=gamma, variance_epsilon=eps)
    nextn = SimpleNamespace(
        model=SimpleNamespace(
            eh_proj=SimpleNamespace(weight=w),
            decoder=SimpleNamespace(input_layernorm=norm_module),
        )
    )
    family = RowParallelEhFamily(nextn, DerivedWeights(), None)
    family.compile(ROWS)
    assert family.hop is hop and set(ROWS) <= family.hop_rows
    tol = hops.TOLERANCES["ll.hops.max_ulp"]
    worst = 0.0
    for rows in ROWS:
        x = x_all[:rows].contiguous()
        family.rp.partial(x)
        graph, out = _capture(lambda: family.eh_proj(x, prepared=True))
        graph.replay()
        torch.cuda.synchronize()
        assert isinstance(out, PreparedInput)
        # under the stock draft attention (prepared False) only the summed projection
        plain = family.eh_proj(x)
        assert isinstance(plain, torch.Tensor) and not isinstance(plain, PreparedInput)
        partials = [
            RowParallelEhProj(w, r, tp.world_size).partial(x)
            for r in range(tp.world_size)
        ]
        acc = torch.zeros(rows, H, dtype=torch.float32, device="cuda")
        for p in partials:
            acc = acc + p.float()
        summed = acc.to(torch.bfloat16)
        assert torch.equal(out.residual.float() + 0.0, summed.float() + 0.0), (
            f"{rows} rows"
        )
        rms = summed.float().pow(2).mean(-1).sqrt()
        rel = ((plain.float() - summed.float()).abs().amax(-1) / rms).max().item()
        assert rel <= ll_draft.TOLERANCES["ll.draft_eh.max_rel_rms"], (
            f"{rows} rows (plain sum): {rel}"
        )
        rcp = torch.rsqrt(summed.float().pow(2).sum(-1, keepdim=True) / H + eps)
        ref = (summed.float() * rcp * gamma.float()).to(torch.bfloat16)
        r = ref.float()
        ulp = torch.ldexp(torch.ones_like(r), torch.frexp(r).exponent - 8)
        err = ((out.norm.float() - r).abs() / ulp).max().item()
        worst = max(worst, err)
        assert err <= tol, (
            f"{rows} rows: the hop's norm is {err:.2f} ulp from RMSNorm of the sum"
        )
    log(
        f"P6a in the LL hop: residual = the FP32 sum of the partials; norm within {worst:.2f} ulp at {ROWS}"
    )
    # per draft forward, hot (CUDA-graph replay): the hop (sum + input norm in one launch) vs SGLang's all-reduce
    # of the partials + the stock RMSNorm
    from sgl_kernel import rmsnorm

    lines = []
    for rows in (1, 2, 5, 6, 12, 24, 48):
        x = x_all[:rows].contiguous()
        t_hop = _graph_us(lambda: family.eh_proj(x, prepared=True), None)
        t_ar = _graph_us(lambda: rmsnorm(family.rp(x), gamma, eps), None)
        lines.append(f"| {rows} | {t_ar:.1f} | {t_hop:.1f} | {t_ar - t_hop:.1f} |")
    log(
        "P6a sum + draft input norm (TP4, us, hot)\n| rows | all-reduce + RMSNorm | LL hop | saved |\n"
        "|---|---|---|---|\n" + "\n".join(lines)
    )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)
