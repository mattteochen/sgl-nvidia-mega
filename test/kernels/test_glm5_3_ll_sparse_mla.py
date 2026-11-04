# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.sparse_mla_fd`` numerics on one SM100-family GPU (GB300), against TRTLLM-Gen's
sparse MLA decode (as SGLang's DSA backend calls it) and an FP64 oracle.

Inputs follow the ``trtllm`` DSA decode path: an FP8 E4M3 query [rows, H, 576], a flat
FP8 KV pool [slots, 576] (slot 0 is the graph padding slot), [rows, 2048] int32 global
slots padded with -1, per-row int32 KV lengths and bmm1_scale = 1 / sqrt(256).

- Per head count (8: TP8, 16: TP4), rows 1-48 (the one-tile kernel at every row count,
  multi-wave beyond its wave; the multi-tile mode at 9-48 rows at the cluster size the
  plan picks and at 2 CTAs), contexts 1k (a -1 tail and idle CTAs), 2048, 8k unsorted
  and sorted, three query distributions: every case finite, nrms(ours, oracle) <= 1.25
  nrms(TRTLLM-Gen, oracle) + 2e-4, |ours - TRTLLM-Gen| <= ll.sparse_mla_fd.max_abs, and
  the spread launch bitwise equal to the packed one.
- The multi-tile mode with a peak in the first / last tile of a CTA (the kRefWindow
  clamp and the 2^20 rescale), two equal peaks in different CTAs (the merge weighs each
  partial by its own reference) and flat queries.
- 16 heads against two 8-head launches; edge rows (empty, all -1, -1 inside the prefix,
  out-of-range slots, duplicates, a padding row on slot 0, one key, one partly active
  CTA) at every cluster size against the oracle.
- The v_b fold (8 heads, 1-7 rows, 1k / 8k / 128k, random W_UV and, with
  ``SGL_NVIDIA_MEGA_GLM_CKPT``, two checkpoint layers): bit for bit the one-tile output
  followed by the stock ``torch.bmm(o.transpose(0, 1), w_vc)``; deterministic; edge
  rows; a strided o_proj-input buffer.
- The provider steps (``SparseMlaFdSteps``) on a stub DSA backend against the stock
  ``AttentionStep`` steps: verify at 5 / 6 rows (16 heads), draft extend at 6 rows
  (8 heads), draft decode at 1 / 2 rows (8 heads, folded): the stock TRTLLM-Gen output
  within the bound, the o_proj input equal to ``flash_decode`` + the stock BMM bit for
  bit, eager and replayed from a CUDA graph.

Run with ``pytest -s test/kernels/test_glm5_3_ll_sparse_mla.py`` (``-s`` prints the
tables).
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.glm5_3.ll import sparse_mla as fd

TOP_K, D_QK, D_V, V_HEAD = 2048, 576, 512, 256
PAGE_SIZE = 64
QK_NOPE = 192
SM_SCALE = 1.0 / math.sqrt(256)
LOG2E = math.log2(math.e)
# max_seq_len of SGLang's captured TRTLLM-Gen calls: the page-table width, baked in at
# capture; FlashInfer sizes its KV split from min(max_seq_len, top-k), so any value >= 2048
# selects the server's kernel.
GRAPH_MAX_SEQ_LEN = 16384
NUM_SLOTS = 1 << 18
KINDS = ("1k", "2048", "8k", "8k_sorted")
CKPT_ENV = "SGL_NVIDIA_MEGA_GLM_CKPT"


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer.decode  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")


# ------------------------------------------------------------------- inputs


def gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def fp8_randn(shape, std: float, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * std
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def make_pool(num_slots: int, std: float, g: torch.Generator) -> torch.Tensor:
    pool = fp8_randn((num_slots, D_QK), std, g)
    pool[0].zero_()  # slot 0 is the CUDA-graph padding slot
    return pool


def make_rows(kind: str, rows: int, num_slots: int, g: torch.Generator):
    """[rows, 2048] int32 slots and [rows] int32 lengths of one context kind.

    1k: L_r in [700, 2040], every token selected, -1 tail; 2048: L_r = 2048 + r; 8k /
    8k_sorted / 128k: L_r = 8192 (131072) + r, 2048 distinct positions (unsorted / sorted
    slots). Tokens map to random slots >= 1.
    """
    cpu = torch.Generator()
    cpu.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g, device="cuda")))
    idx = torch.full((rows, TOP_K), -1, dtype=torch.int32)
    lens = torch.empty(rows, dtype=torch.int32)
    for r in range(rows):
        if kind == "1k":
            length = int(torch.randint(700, 2041, (1,), generator=cpu))
            positions = torch.randperm(length, generator=cpu)
        elif kind == "2048":
            length = TOP_K + r
            positions = torch.randperm(length, generator=cpu)[:TOP_K]
        else:
            length = (131072 if kind == "128k" else 8192) + r
            positions = torch.randperm(length, generator=cpu)[:TOP_K]
        if length <= num_slots - 1:
            tokens = torch.randperm(num_slots - 1, generator=cpu)[:length] + 1
        else:
            tokens = torch.randint(1, num_slots, (length,), generator=cpu)
        slots = tokens[positions]
        if kind == "8k_sorted":
            slots = slots.sort().values
        idx[r, : slots.numel()] = slots.to(torch.int32)
        lens[r] = length
    return idx.cuda(), lens.cuda()


def make_q(rows: int, heads: int, std: float, g: torch.Generator) -> torch.Tensor:
    return fp8_randn((rows, heads, D_QK), std, g)


def oracle(q, pool, indices, seq_lens, sm_scale, *, chunk: int = 8) -> torch.Tensor:
    """Exact FP64 attention on the FP8 inputs under the kernel's masking rules."""
    rows, num_slots = q.shape[0], pool.shape[0]
    out = torch.zeros(rows, q.shape[1], D_V, dtype=torch.float64, device="cuda")
    pos = torch.arange(TOP_K, device="cuda")
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        idx = indices[r0:r1, :TOP_K].long()
        valid = (
            (pos[None, :] < seq_lens[r0:r1, None].long())
            & (idx >= 0)
            & (idx < num_slots)
        )
        k = pool[torch.where(valid, idx, 0)].double()
        s = torch.einsum("rhd,rkd->rhk", q[r0:r1].double(), k) * sm_scale
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        m = s.amax(-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(s - m)
        den = p.sum(-1, keepdim=True)
        o = torch.einsum("rhk,rkd->rhd", p, k[..., :D_V])
        out[r0:r1] = torch.where(
            den > 0, o / den.clamp_min(1e-300), torch.zeros_like(o)
        )
    return out


def nrms(x: torch.Tensor, ref: torch.Tensor) -> float:
    x, ref = x.double(), ref.double()
    return (x - ref).norm().item() / max(ref.norm().item(), 1e-30)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


class TrtllmGen:
    """TRTLLM-Gen's sparse MLA call as SGLang's DSA backend makes it (_forward_trtllm)."""

    def __init__(self, heads: int, max_rows: int = 512):
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )

        self.workspace = torch.zeros(384 << 20, dtype=torch.uint8, device="cuda")
        self.counter = make_persistent_multi_ctas_kv_counter_buffer(
            device=torch.device("cuda"), num_q_heads=heads, max_batch_size=max_rows
        )

    def __call__(self, q, pool, indices, seq_lens, bmm1_scale, *, max_seq_len=None):
        import flashinfer.decode

        rows, heads, _ = q.shape
        if max_seq_len is None:
            capturing = torch.cuda.is_current_stream_capturing()
            max_seq_len = GRAPH_MAX_SEQ_LEN if capturing else int(seq_lens[:rows].max())
        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q.view(rows, 1, heads, D_QK),
            kv_cache=pool.view(-1, 1, PAGE_SIZE, D_QK),
            workspace_buffer=self.workspace,
            qk_nope_head_dim=QK_NOPE,
            kv_lora_rank=D_V,
            qk_rope_head_dim=D_QK - D_V,
            block_tables=indices.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            sparse_mla_top_k=TOP_K,
            bmm1_scale=bmm1_scale,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=None,
            sparse_mla_top_k_lens=None,
            multi_ctas_kv_counter_buffer=self.counter,
        )
        return out.view(rows, heads, D_V)


def ours(q, pool, idx, lens, **kw) -> torch.Tensor:
    return fd.flash_decode(q, pool, idx, lens, SM_SCALE * LOG2E, **kw)


def bmm_vb(o: torch.Tensor, w_uv: torch.Tensor, out=None) -> torch.Tensor:
    """forward_absorb_core's v_b BMM: o [R, H, 512] x w_vc = W_UV.transpose(1, 2)."""
    rows, heads = o.shape[0], o.shape[1]
    if out is None:
        out = torch.empty(rows, heads * V_HEAD, dtype=torch.bfloat16, device=o.device)
    torch.bmm(
        o.transpose(0, 1),
        w_uv.transpose(1, 2),
        out=out.view(-1, heads, V_HEAD)[:rows].transpose(0, 1),
    )
    return out


_caps: dict = {}


def caps(heads: int) -> fd.Caps:
    if heads not in _caps:
        _caps[heads] = fd.device_caps(
            heads, with_mt=True, with_fold=heads == fd.FOLD_HEADS
        )
    return _caps[heads]


def table(title: str, header: list, rows: list) -> None:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}" if (abs(v) < 1e-2 and v) else f"{v:.4f}"
        return str(v)

    widths = [
        max(len(str(h)), *(len(fmt(r[i])) for r in rows)) for i, h in enumerate(header)
    ]
    print(f"\n## {title}")
    print("  ".join(str(h).rjust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(fmt(v).rjust(w) for v, w in zip(r, widths)))


@pytest.fixture(scope="module")
def pool():
    torch.cuda.set_device(0)
    return make_pool(NUM_SLOTS, 1.4, gen(1234))


def query(case: str, rows: int, heads: int, pool, idx, lens, g) -> torch.Tensor:
    if case == "normal":
        return make_q(rows, heads, 1.4, g)
    if case == "flat":
        return make_q(rows, heads, 0.2, g)
    # peaked: every head aligns with one valid key of its row
    q = make_q(rows, heads, 0.5, g).float()
    for r in range(rows):
        j = int(
            torch.randint(
                0, int(min(lens[r].item(), TOP_K)), (1,), generator=g, device="cuda"
            )
        )
        q[r] += 2.0 * pool[int(idx[r, j].item())].float()
    return q.clamp(-448, 448).to(torch.float8_e4m3fn)


# ------------------------------------------------------------------- attention numerics


MAX_ABS_SEEN: dict = {}


@pytest.mark.parametrize("heads", fd.NUM_HEADS)
def test_numerics_vs_trtllm_gen(pool, heads):
    g = gen(100 + heads)
    trt = TrtllmGen(heads)
    cp = caps(heads)
    print(f"\n{heads} heads: caps {cp}")
    width = 6 if heads == 8 else 5
    failures, rows_out = [], []
    worst = 0.0
    for rows in sorted({1, 2, 4, width, 6, 7, 9, 12, 18, 24, 36, 48}):
        for kind in KINDS:
            idx, lens = make_rows(kind, rows, NUM_SLOTS, g)
            for qc in ("normal", "peaked", "flat") if rows == width else ("normal",):
                q = query(qc, rows, heads, pool, idx, lens, g)
                ref = oracle(q, pool, idx, lens, SM_SCALE)
                old = trt(q, pool, idx, lens, SM_SCALE)
                packed = ours(q, pool, idx, lens, smem_bytes=cp.smem_bytes)
                spread = ours(q, pool, idx, lens, smem_bytes=cp.spread_smem_bytes)
                e_old = nrms(old, ref)
                bound = 1.25 * e_old + 2e-4
                tag = f"{heads}h rows={rows} {kind} q={qc}"
                outs = {"one-tile": packed}
                if rows >= 9:
                    picked = fd.mt_splits_for_rows(rows, cp.mt_rows)
                    for s in sorted({s for s in (picked, 2) if s}):
                        outs[f"mt{s}"] = ours(q, pool, idx, lens, splits=s)
                torch.cuda.synchronize()
                if not torch.equal(packed, spread):
                    failures.append(f"{tag}: spread launch differs from packed")
                for name, out in outs.items():
                    e, d = nrms(out, ref), max_abs(out, old)
                    worst = max(worst, d)
                    if not bool(torch.isfinite(out.float()).all()):
                        failures.append(f"{tag} {name}: non-finite output")
                    if e > bound:
                        failures.append(f"{tag} {name}: nrms {e:.3e} > {bound:.3e}")
                    if d > fd.TOLERANCES["ll.sparse_mla_fd.max_abs"]:
                        failures.append(f"{tag} {name}: |ours - TRTLLM-Gen| {d:.4g}")
                    rows_out.append(
                        [
                            rows,
                            kind,
                            qc,
                            name,
                            e,
                            e_old,
                            d,
                            "ok" if e <= bound else "FAIL",
                        ]
                    )
    MAX_ABS_SEEN[heads] = worst
    table(
        f"{heads} heads: ours vs TRTLLM-Gen vs FP64 oracle",
        [
            "rows",
            "ctx",
            "q",
            "kernel",
            "nrms_ours",
            "nrms_trtllm",
            "max_abs_vs_trtllm",
            "verdict",
        ],
        rows_out,
    )
    print(f"{heads} heads: max |ours - TRTLLM-Gen| = {worst:.6g}")
    assert not failures, "\n".join(failures[:40])


def cta_tiles(split: int, splits: int, length: int) -> list:
    """Active 128-position tiles of CTA ``split`` in the multi-tile mode, in order."""
    active = (min(length, TOP_K) + 127) // 128
    return list(range(split, active, splits))


def tile_peaked_query(rows, heads, splits, where, pool, idx, lens, g) -> torch.Tensor:
    """Every head of a row peaks on one key in the first or the last tile of a CTA,
    about 400 log2 above the row's other keys (beyond kRefWindow = 20)."""
    q = make_q(rows, heads, 0.5, g).float()
    for r in range(rows):
        length = int(lens[r].item())
        multi = [s for s in range(splits) if len(cta_tiles(s, splits, length)) >= 2]
        ctas = multi or [s for s in range(splits) if cta_tiles(s, splits, length)]
        for h in range(heads):
            tiles = cta_tiles(ctas[(r + h) % len(ctas)], splits, length)
            tile = tiles[0] if where == "first" else tiles[-1]
            hi = min(tile * 128 + 128, length, TOP_K)
            j = int(torch.randint(tile * 128, hi, (1,), generator=g, device="cuda"))
            q[r, h] += 4.0 * pool[int(idx[r, j].item())].float()
    return q.clamp(-448, 448).to(torch.float8_e4m3fn)


def tie_case(rows, heads, splits, tpool, idx, lens, g):
    """Two equal peaks per row, in the first tile of CTA a and the last tile of CTA b,
    with different V rows: the output is (V_A + V_B) / 2 only if the merge weighs each
    partial by the reference it was accumulated against. The peak keys go to fresh
    slots past NUM_SLOTS."""
    idx = idx.clone()
    q = torch.empty(rows, heads, D_QK, device="cuda")
    for h in range(heads):
        q[:, h] = 2.0 + 0.5 * (h % 4)  # exact in e4m3
    for r in range(rows):
        length = int(lens[r].item())
        ctas = [s for s in range(splits) if cta_tiles(s, splits, length)]
        multi = [s for s in ctas if len(cta_tiles(s, splits, length)) >= 2]
        a = (multi or ctas)[r % len(multi or ctas)]
        b = next((s for s in ctas if s != a), a)
        positions = []
        for cta, tile in (
            (a, cta_tiles(a, splits, length)[0]),
            (b, cta_tiles(b, splits, length)[-1]),
        ):
            hi = min(tile * 128 + 128, length, TOP_K)
            choices = [j for j in range(tile * 128, hi) if j not in positions]
            positions.append(
                choices[
                    int(
                        torch.randint(0, len(choices), (1,), generator=g, device="cuda")
                    )
                ]
            )
        latent = (torch.randn(D_V, generator=g, device="cuda").abs() * 1.4).to(
            torch.float8_e4m3fn
        )
        rope = fp8_randn((D_QK - D_V,), 1.4, g)
        perm = torch.randperm(D_V, generator=g, device="cuda")
        for k, (j, lat) in enumerate(zip(positions, (latent, latent[perm]))):
            slot = NUM_SLOTS + 2 * r + k
            tpool[slot, :D_V] = lat
            tpool[slot, D_V:] = rope
            idx[r, j] = slot
    return q.to(torch.float8_e4m3fn), idx


@pytest.mark.parametrize("heads", fd.NUM_HEADS)
def test_multi_tile_peaked(pool, heads):
    g = gen(200 + heads)
    trt = TrtllmGen(heads)
    cp = caps(heads)
    tpool = torch.cat(
        [pool, torch.zeros(PAGE_SIZE, D_QK, device="cuda").to(pool.dtype)]
    )
    failures, rows_out = [], []
    for rows in (12, 18, 24):
        splits = fd.mt_splits_for_rows(rows, cp.mt_rows)
        assert splits, f"no multi-tile size for {rows} rows ({cp})"
        for kind in ("8k", "1k"):
            idx, lens = make_rows(kind, rows, NUM_SLOTS, g)
            for qc in ("peak-first", "peak-last", "tie", "flat"):
                rows_idx = idx
                if qc == "flat":
                    q = make_q(rows, heads, 0.2, g)
                elif qc == "tie":
                    q, rows_idx = tie_case(rows, heads, splits, tpool, idx, lens, g)
                else:
                    q = tile_peaked_query(
                        rows, heads, splits, qc.split("-")[1], pool, idx, lens, g
                    )
                ref = oracle(q, tpool, rows_idx, lens, SM_SCALE)
                old = trt(q, tpool, rows_idx, lens, SM_SCALE)
                mt = ours(q, tpool, rows_idx, lens, splits=splits)
                torch.cuda.synchronize()
                e_mt, e_old = nrms(mt, ref), nrms(old, ref)
                bound = 1.25 * e_old + 2e-4
                tag = f"{heads}h mt{splits} rows={rows} {kind} q={qc}"
                if not bool(torch.isfinite(mt.float()).all()):
                    failures.append(f"{tag}: non-finite")
                if e_mt > bound:
                    failures.append(f"{tag}: nrms {e_mt:.3e} > {bound:.3e}")
                rows_out.append(
                    [
                        rows,
                        splits,
                        kind,
                        qc,
                        e_mt,
                        e_old,
                        "ok" if e_mt <= bound else "FAIL",
                    ]
                )
    table(
        f"{heads} heads: multi-tile peaked / tie / flat",
        ["rows", "splits", "ctx", "q", "nrms_mt", "nrms_trtllm", "verdict"],
        rows_out,
    )
    assert not failures, "\n".join(failures)


def test_head_split(pool):
    """16 heads equal two 8-head launches (the per-head arithmetic is the same)."""
    g = gen(300)
    for kind in ("1k", "8k"):
        idx, lens = make_rows(kind, 6, NUM_SLOTS, g)
        q = make_q(6, 16, 1.4, g)
        both = torch.cat(
            [
                ours(q[:, :8].contiguous(), pool, idx, lens),
                ours(q[:, 8:].contiguous(), pool, idx, lens),
            ],
            dim=1,
        )
        out16 = ours(q, pool, idx, lens)
        e = nrms(out16, both)
        print(f"head split {kind}: bitwise {torch.equal(out16, both)}, nrms {e:.2e}")
        assert e <= 1e-3


@pytest.mark.parametrize("heads", fd.NUM_HEADS)
def test_edge_rows(pool, heads):
    """Rows TRTLLM-Gen does not define the same way, against the oracle, at every
    cluster size."""
    g = gen(400 + heads)
    rows = 8
    idx, lens = make_rows("8k", rows, NUM_SLOTS, g)
    idx, lens = idx.clone(), lens.clone()
    lens[0] = 0  # empty row
    lens[1], idx[1] = 3000, -1  # all masked
    idx[2, ::3] = -1  # -1 inside the prefix
    idx[3, 5], idx[3, 700] = NUM_SLOTS + 5, 2**31 - 1  # out of range inside the prefix
    idx[4, :1500], idx[4, 1500:] = 17, 4242  # duplicates count per occurrence
    lens[5], idx[5] = 1, 0  # graph padding row: slot 0
    lens[6] = 1  # one key: the output is that key's V row exactly
    lens[7] = 100  # one partly active CTA
    q = make_q(rows, heads, 1.4, g)
    ref = oracle(q, pool, idx, lens, SM_SCALE)
    v_row = pool[int(idx[6, 0].item()), :D_V].to(torch.bfloat16)
    failures = []
    for splits in (16, *fd.MT_SPLITS):
        out = ours(q, pool, idx, lens, splits=splits)
        torch.cuda.synchronize()
        tag = f"edge {heads}h splits={splits}"
        if not bool(torch.isfinite(out.float()).all()):
            failures.append(f"{tag}: non-finite")
        if not (bool((out[0] == 0).all()) and bool((out[1] == 0).all())):
            failures.append(f"{tag}: empty rows not zero")
        if not bool((out[6] == v_row[None, :]).all()):
            failures.append(f"{tag}: one-key row is not that V row")
        for r in (2, 3, 4, 5, 7):
            e = (
                nrms(out[r], ref[r])
                if ref[r].abs().max() > 0
                else max_abs(out[r], torch.zeros_like(out[r]))
            )
            if e > 2e-2:
                failures.append(f"{tag} row {r}: nrms {e:.3e}")
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------------- v_b fold


def _w_uvs(g) -> list:
    """Random W_UV, plus two checkpoint layers' TP8 rank-0 heads when the env names one."""
    w = [
        (torch.randn(8, V_HEAD, D_V, generator=g, device="cuda") * 0.05).to(
            torch.bfloat16
        )
    ]
    ckpt = os.environ.get(CKPT_ENV)
    if ckpt and os.path.isdir(ckpt):
        from safetensors import safe_open

        with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
            wmap = json.load(f)["weight_map"]
        for layer in (3, 40):
            name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
            if name in wmap:
                with safe_open(os.path.join(ckpt, wmap[name]), "pt", device="cpu") as f:
                    t = f.get_tensor(name)
                if t.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    t = t.view(-1, QK_NOPE + V_HEAD, D_V)[:8, QK_NOPE:, :]
                    w.append(t.to(device="cuda", dtype=torch.bfloat16).contiguous())
    return w


def test_fold_bitwise_vs_bmm():
    g = gen(500)
    w_uvs = _w_uvs(g)
    cp = caps(8)
    pools = {"1k": 8192, "8k": 65536, "128k": 262144}
    failures, rows_out = [], []
    for ctx, num_slots in pools.items():
        p = make_pool(num_slots, 1.4, g)
        for rows in (1, 2, 3, 6, 7):
            for wi, w_uv in enumerate(w_uvs):
                q = make_q(rows, 8, 1.4, g)
                idx, lens = make_rows(ctx, rows, num_slots, g)
                o = ours(q, p, idx, lens, smem_bytes=cp.spread_smem_bytes)
                ref = bmm_vb(o, w_uv)
                got = ours(q, p, idx, lens, vb_proj=w_uv)
                again = ours(q, p, idx, lens, vb_proj=w_uv)
                torch.cuda.synchronize()
                differ = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
                tag = f"fold {ctx} rows={rows} W{wi}"
                if differ:
                    failures.append(f"{tag}: {differ} elements differ from the BMM")
                if not torch.equal(got, again):
                    failures.append(f"{tag}: not deterministic")
                exact = torch.einsum("rhd,hvd->rhv", o.double(), w_uv.double()).reshape(
                    rows, -1
                )
                rows_out.append(
                    [ctx, rows, wi, differ, nrms(ref, exact), nrms(got, exact)]
                )
    table(
        "v_b fold vs flash decode + torch.bmm",
        ["ctx", "rows", "W", "differ", "nrms_bmm", "nrms_fold"],
        rows_out,
    )
    assert not failures, "\n".join(failures)


def test_fold_edge_rows_and_strided_out():
    g = gen(510)
    num_slots, rows = 4096, 7
    p = make_pool(num_slots, 1.4, g)
    w_uv = (torch.randn(8, V_HEAD, D_V, generator=g, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    q = make_q(rows, 8, 1.4, g)
    idx = torch.randint(1, num_slots, (rows, TOP_K), generator=g, device="cuda").to(
        torch.int32
    )
    lens = torch.full((rows,), 1500, dtype=torch.int32, device="cuda")
    lens[0] = 0  # empty row
    idx[1] = -1  # all -1
    lens[2], idx[2] = 1, 0  # graph padding row: length 1, slot 0
    idx[3, 100:200] = num_slots + 5  # out-of-range slots inside the prefix
    idx[4, :1500] = idx[4, 7]  # one slot repeated
    lens[5] = 129  # one active position in the second CTA
    ref = bmm_vb(ours(q, p, idx, lens, smem_bytes=caps(8).spread_smem_bytes), w_uv)
    got = ours(q, p, idx, lens, vb_proj=w_uv)
    assert torch.equal(got, ref)
    assert bool((got[:2].float() == 0).all()), "empty rows not zero"
    # the fold writes only its 8 x 256 columns of each row of a wider buffer
    wide = torch.full((rows, 3000), 7.0, dtype=torch.bfloat16, device="cuda")
    view = wide[:, 100 : 100 + 8 * V_HEAD]
    ours(q, p, idx, lens, vb_proj=w_uv, out=view)
    assert torch.equal(view.contiguous(), got)
    rest = torch.cat([wide[:, :100], wide[:, 100 + 8 * V_HEAD :]], dim=1)
    assert bool((rest.float() == 7.0).all()), "wrote outside its columns"


# ------------------------------------------------------------------- provider steps


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "decode"

    def is_target_verify(self):
        return self.name == "verify"

    def is_draft_extend_v2(self):
        return self.name == "extend"


class _TopkBackend:
    def is_sgl_kernel(self):
        return True

    def is_flashinfer(self):
        return False


class _Backend:
    """The DSA backend attributes the stock sparse_mla step and ours read, with SGLang's
    own helpers bound from DeepseekSparseAttnBackend."""

    def __init__(self, heads, pool, lens):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend as B,
        )

        rows = lens.shape[0]
        self.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda layer_id: pool)
        self.physical_page_size = PAGE_SIZE
        self.kv_cache_dim = D_QK
        self.qk_nope_head_dim = QK_NOPE
        self.kv_lora_rank = D_V
        self.qk_rope_head_dim = D_QK - D_V
        self.dsa_topk_backend = _TopkBackend()
        self.use_fused_topk = True
        self.dcp_size = 1
        self.workspace_buffer = torch.zeros(384 << 20, dtype=torch.uint8, device="cuda")
        self._trt = TrtllmGen(heads)
        self.forward_metadata = SimpleNamespace(
            cache_seqlens_int32=lens,
            dsa_cache_seqlens_int32=lens.clamp(max=TOP_K),
            max_seq_len_k=GRAPH_MAX_SEQ_LEN,
        )
        for name in (
            "_get_fused_topk_page_table",
            "_pad_topk_indices",
            "_pad_trtllm_sparse_page_table",
        ):
            setattr(self, name, getattr(B, name).__get__(self))
        self.rows = rows

    def _multi_ctas_kv_counter_for(self, batch_size):
        return self._trt.counter

    def view(self):
        """SGLang's ForwardView of this backend (Step.view), the members the steps read."""
        return SimpleNamespace(
            metadata=self.forward_metadata,
            kv_pool=self.token_to_kv_pool,
            dsa=SimpleNamespace(
                physical_page_size=self.physical_page_size,
                kv_cache_dim=self.kv_cache_dim,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                workspace_buffer=self.workspace_buffer,
                multi_ctas_kv_counter_buffer=self._trt.counter,
                use_fused_topk=self.use_fused_topk,
                topk_backend=self.dsa_topk_backend,
                dcp_size=self.dcp_size,
            ),
        )


def _attn(heads, w_uv):
    w_vc = w_uv.transpose(1, 2)  # [H, 512, 256], the loader's view
    return SimpleNamespace(
        layer_id=0,
        attn_mqa=SimpleNamespace(
            layer_id=0,
            k_scale_float=None,
            scaling=SM_SCALE,
            tp_q_head_num=heads,
            head_dim=D_QK,
        ),
        w_vc=w_vc,
        num_local_heads=heads,
        v_head_dim=V_HEAD,
        kv_lora_rank=D_V,
    )


@pytest.mark.parametrize(
    "heads,rows,mode",
    [
        (16, 5, "verify"),
        (16, 6, "verify"),
        (16, 2, "decode"),
        (8, 6, "extend"),
        (8, 1, "decode"),
        (8, 2, "decode"),
    ],
)
def test_provider_steps(monkeypatch, heads, rows, mode):
    import sglang.srt.model_executor.forward_context as fc

    from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep
    from sgl_nvidia_mega.glm5_3.draft_attention import DraftAttention

    g = gen(600 + rows)
    num_slots = 1 << 16
    p = make_pool(num_slots, 1.4, g).view(num_slots, 1, D_QK)
    idx, lens = make_rows("8k" if rows % 2 else "1k", rows, num_slots, g)
    q = make_q(rows, heads, 1.4, g)
    w_uv = (torch.randn(heads, V_HEAD, D_V, generator=g, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    backend = _Backend(heads, p, lens)
    monkeypatch.setattr(fc, "get_attn_backend", lambda: backend)
    attn = _attn(heads, w_uv)
    step = SimpleNamespace(
        forward_batch=SimpleNamespace(forward_mode=_Mode(mode)), view=backend.view()
    )

    profile = SimpleNamespace(tp=64 // heads, heads=heads, sm=(10, 3), sms=152)
    draft = mode != "verify"
    model = SimpleNamespace(
        model=SimpleNamespace(decoder=SimpleNamespace(self_attn=attn))
        if draft
        else SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    )
    assert fd.covers(profile, rows, mode) is None
    family = fd.SparseMlaFdFamily(model, None, profile)
    family.compile([rows])
    reg_mode = {"verify": fd.VERIFY, "extend": fd.DRAFT_EXTEND}.get(
        mode, fd.DRAFT_DECODE
    )
    steps = family.provider(fd.CANDIDATES[1 if draft else 0], rows, reg_mode)

    def run_ours():
        return steps.uv_proj(attn, steps.sparse_mla(attn, q, idx, step))

    decode = mode == "decode"
    # the stock step: the verify's, or the draft decode's (its full cache lengths)
    stock = DraftAttention() if decode else AttentionStep()
    o_stock = stock.sparse_mla(attn, q, idx, step)
    seq = lens if decode else lens.clamp(max=TOP_K)
    o_ours = fd.flash_decode(q, p.view(-1, D_QK), idx, seq, SM_SCALE * LOG2E)
    vb_ref = bmm_vb(o_ours, w_uv)
    vb = run_ours()
    torch.cuda.synchronize()
    folded = family.plan.folds(rows, fd.DRAFT_DECODE if decode else fd.VERIFY)
    assert folded == (heads == 8 and decode)
    assert torch.equal(vb, vb_ref), (
        "o_proj input differs from flash_decode + the stock BMM"
    )
    ref = oracle(q, p.view(-1, D_QK), idx, seq, SM_SCALE)
    assert nrms(o_ours, ref) <= 1.25 * nrms(o_stock, ref) + 2e-4
    assert max_abs(o_ours, o_stock) <= fd.TOLERANCES["ll.sparse_mla_fd.max_abs"]
    # captured once, replayed on new inputs: equal to eager
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run_ours()
    for it in range(5):
        i2, l2 = make_rows(("1k", "8k")[it % 2], rows, num_slots, g)
        idx.copy_(i2)
        lens.copy_(l2)
        backend.forward_metadata.dsa_cache_seqlens_int32.copy_(l2.clamp(max=TOP_K))
        q.copy_(make_q(rows, heads, 1.4, g))
        out.fill_(float("nan"))
        graph.replay()
        got = out.clone()
        want = run_ours()
        torch.cuda.synchronize()
        assert torch.equal(got, want), f"replay {it} differs from eager"
