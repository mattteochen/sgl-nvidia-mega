# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.sparse_mla_fd`` under CUDA graphs and PDL on one SM100-family GPU (GB300).

1. Replay: the kernel captured once and replayed 30 times with new q / indices / lengths
   in its static inputs equals the eager call bit for bit: one-tile at the plan's rows
   (8 and 16 heads), the multi-tile mode at 12 / 18 / 24 rows on the cluster sizes the
   plan picks and at 2 CTAs, and the v_b fold at 1 / 2 / 7 rows.
2. The PDL chain of the decode step: RoPE + FP8 quant (FlashInfer, PDL) -> KV append
   (``set_mla_kv_buffer``, PDL) -> the flash decode (PDL, triggers after its gathers)
   -> v_b (``torch.bmm``, or the fold) -> o_proj (``F.linear``) in one graph. Before each
   of 50 replays the fresh KV rows, the FP8 query, the attention output, the o_proj
   input and output are filled with NaN bytes, so a read that overtakes its producer
   shows up. Each replay equals the same chain run eagerly bit for bit; the last one is
   also held against the TRTLLM-Gen chain and the FP64 oracle.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from test_glm5_3_ll_sparse_mla import pytestmark  # noqa: F401  (the SM100 skip)
from test_glm5_3_ll_sparse_mla import (
    D_QK,
    D_V,
    LOG2E,
    SM_SCALE,
    TOP_K,
    V_HEAD,
    TrtllmGen,
    bmm_vb,
    caps,
    gen,
    make_pool,
    make_q,
    make_rows,
    nrms,
    oracle,
)

from sgl_nvidia_mega.glm5_3.ll import sparse_mla as fd

NUM_SLOTS = 1 << 17
HIDDEN = 6144


def _cases(heads: int):
    """(rows, splits, fold) under test: the plan's one-tile rows, the multi-tile sizes,
    the fold rows (8 heads)."""
    cp = caps(heads)
    cases = [(r, 16, False) for r in ((1, 5, 6, 7) if heads == 16 else (1, 2, 6, 7))]
    if heads == 8:
        cases += [(r, 16, True) for r in (1, 2, 7)]
    picked = {(r, fd.mt_splits_for_rows(r, cp.mt_rows), False) for r in (12, 18, 24)}
    cases += sorted({c for c in picked if c[1]} | {(24, 2, False)})
    return cases


@pytest.mark.parametrize("heads", fd.NUM_HEADS)
def test_graph_replay(heads):
    g = gen(4321 + heads)
    pool = make_pool(NUM_SLOTS, 1.4, g)
    cp = caps(heads)
    w_uv = (torch.randn(heads, V_HEAD, D_V, generator=g, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    bad = []
    for rows, splits, fold in _cases(heads):
        idx, lens = make_rows("1k", rows, NUM_SLOTS, g)
        q = make_q(rows, heads, 1.4, g)

        def run():
            if fold:
                return fd.flash_decode(
                    q, pool, idx, lens, SM_SCALE * LOG2E, vb_proj=w_uv
                )
            smem = cp.spread_smem_bytes if splits == 16 else None
            return fd.flash_decode(
                q, pool, idx, lens, SM_SCALE * LOG2E, splits=splits, smem_bytes=smem
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        differ = 0
        for it in range(30):
            i2, l2 = make_rows(("1k", "8k", "2048")[it % 3], rows, NUM_SLOTS, g)
            idx.copy_(i2)
            lens.copy_(l2)
            q.copy_(make_q(rows, heads, 1.4, g))
            out.fill_(float("nan"))
            graph.replay()
            got = out.clone()
            ref = run()
            torch.cuda.synchronize()
            differ += int(not torch.equal(got, ref))
        tag = f"{heads}h rows={rows} splits={splits} fold={fold}"
        print(f"graph replay {tag}: {30 - differ}/30 bitwise equal")
        if differ:
            bad.append(f"{tag}: {differ}/30 replays differ")
    assert not bad, "\n".join(bad)


class Chain:
    """q / k RoPE + FP8 quant -> KV append -> attention -> v_b -> o_proj, static buffers."""

    def __init__(self, heads, rows, attention, g, *, fold=False):
        self.heads, self.rows, self.attention, self.fold = heads, rows, attention, fold
        self.pool = make_pool(NUM_SLOTS, 1.4, g)
        self.loc = (torch.randperm(NUM_SLOTS - 1, device="cuda")[:rows] + 1).to(
            torch.int64
        )
        self.positions = torch.randint(0, 16384, (rows,), device="cuda")
        self.cos_sin = torch.randn(16384, 64, device="cuda").float()
        self.q_nope = torch.empty(rows, heads, D_V, device="cuda", dtype=torch.bfloat16)
        self.q_pe = torch.empty(
            rows, heads, D_QK - D_V, device="cuda", dtype=torch.bfloat16
        )
        self.k_nope = torch.empty(rows, D_V, device="cuda", dtype=torch.bfloat16)
        self.k_pe = torch.empty(rows, D_QK - D_V, device="cuda", dtype=torch.bfloat16)
        self.idx = torch.empty(rows, TOP_K, device="cuda", dtype=torch.int32)
        self.lens = torch.empty(rows, device="cuda", dtype=torch.int32)
        self.w_uv = (torch.randn(heads, V_HEAD, D_V, device="cuda") * 0.05).to(
            torch.bfloat16
        )
        self.w_o = (torch.randn(HIDDEN, heads * V_HEAD, device="cuda") * 0.02).to(
            torch.bfloat16
        )
        self.refill(g)

    def refill(self, g):
        r, h = self.rows, self.heads
        self.q_nope.copy_(torch.randn(r, h, D_V, device="cuda", generator=g))
        self.q_pe.copy_(torch.randn(r, h, D_QK - D_V, device="cuda", generator=g))
        self.k_nope.copy_(torch.randn(r, D_V, device="cuda", generator=g))
        self.k_pe.copy_(torch.randn(r, D_QK - D_V, device="cuda", generator=g))
        idx, lens = make_rows("1k", r, NUM_SLOTS, g)
        # every row reads all of this step's rows (written by the KV append)
        perm = torch.randperm(int(lens.min().item()), device="cuda")[:r]
        idx[:, perm] = self.loc.to(torch.int32)[None, :]
        self.idx.copy_(idx)
        self.lens.copy_(lens)

    def run(self):
        from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8
        from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer

        q, k, k_rope = mla_quantize_and_rope_for_fp8(
            self.q_nope,
            self.q_pe,
            self.k_nope,
            self.k_pe,
            self.positions,
            self.cos_sin,
            False,
            D_V,
            D_QK - D_V,
        )
        set_mla_kv_buffer(self.pool, self.loc, k, k_rope)
        if self.fold:
            attn = None
            vb = self.attention(q, self.pool, self.idx, self.lens, vb_proj=self.w_uv)
        else:
            attn = self.attention(q, self.pool, self.idx, self.lens)
            vb = bmm_vb(attn, self.w_uv)
        return q, attn, vb, F.linear(vb, self.w_o)

    def poison(self, q, attn, vb, o):
        self.pool.view(torch.uint8)[self.loc] = 0xFF
        q.view(torch.uint8).fill_(0xFF)
        for t in (attn, vb, o):
            if t is not None:
                t.fill_(float("nan"))


@pytest.mark.parametrize("heads", fd.NUM_HEADS)
def test_pdl_chain(heads):
    g = gen(777 + heads)
    cp = caps(heads)
    failures = []
    for rows, splits, fold in _cases(heads):
        if splits == 2:
            continue

        def ours(q, pool, idx, lens, vb_proj=None, splits=splits):
            smem = cp.spread_smem_bytes if splits == 16 and vb_proj is None else None
            return fd.flash_decode(
                q,
                pool,
                idx,
                lens,
                SM_SCALE * LOG2E,
                splits=splits,
                smem_bytes=smem,
                vb_proj=vb_proj,
            )

        chain = Chain(heads, rows, ours, g, fold=fold)
        chain.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = chain.run()
        mismatches = 0
        for _ in range(50):
            chain.refill(g)
            chain.poison(*captured)
            graph.replay()
            got = [t.clone() if t is not None else None for t in captured[1:]]
            ref = chain.run()[1:]  # eager, same inputs; rewrites the same KV rows
            torch.cuda.synchronize()
            ok = all(
                (a is None and b is None)
                or (bool(torch.isfinite(a.float()).all()) and torch.equal(a, b))
                for a, b in zip(got, ref)
            )
            mismatches += int(not ok)
        tag = f"{heads}h rows={rows} splits={splits} fold={fold}"
        if mismatches:
            failures.append(f"PDL chain {tag}: {mismatches}/50 replays wrong")
        # the same final inputs through the TRTLLM-Gen chain, against the oracle
        trt = TrtllmGen(heads)
        old = Chain(heads, rows, lambda q, p, i, l: trt(q, p, i, l, SM_SCALE), g)
        for name in (
            "pool",
            "q_nope",
            "q_pe",
            "k_nope",
            "k_pe",
            "idx",
            "lens",
            "positions",
            "cos_sin",
            "loc",
            "w_uv",
        ):
            getattr(old, name).copy_(getattr(chain, name))
        graph.replay()
        q_old, attn_old, _, _ = old.run()
        q_new = captured[0]
        ref = oracle(q_new, chain.pool, chain.idx, chain.lens, SM_SCALE)
        o_new = (
            captured[1]
            if not fold
            else fd.flash_decode(
                q_new, chain.pool, chain.idx, chain.lens, SM_SCALE * LOG2E
            )
        )
        torch.cuda.synchronize()
        e_new, e_old = nrms(o_new, ref), nrms(attn_old, ref)
        print(
            f"PDL chain {tag}: {50 - mismatches}/50 replays equal eager; nrms ours {e_new:.2e} TRTLLM-Gen {e_old:.2e}"
        )
        if e_new > 1.25 * e_old + 2e-4:
            failures.append(
                f"PDL chain {tag}: nrms {e_new:.3e} vs TRTLLM-Gen {e_old:.3e}"
            )
    assert not failures, "\n".join(failures)
