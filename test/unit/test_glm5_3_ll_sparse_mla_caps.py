# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.sparse_mla_fd`` launch plan on CPU: the candidate's ``covers``, the plan derived
from the co-resident clusters, and the TP ranks' agreement on them (two gloo
processes).

The agreement takes the MIN of every count over the TP group's CPU group, so GPUs with
different floorsweeping launch one plan; a rank whose build or occupancy query failed
makes every rank raise (fail-stop), never a silent stock fallback on some ranks only.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import tempfile
from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.glm5_3.ll import sparse_mla as fd

# GB300 (152 SMs), job 4488724: one-tile spread / packed waves and the multi-tile
# clusters per size (2, 3, 4, 6, 8 CTAs).
GB300_8 = fd.Caps(105416, 116736, 7, 14, (76, 46, 36, 23, 15), 7, 221976)
GB300_16 = fd.Caps(134984, 134984, 7, 7)
MT = (76, 46, 36, 23, 15)


def profile(heads=16, sm=(10, 3), sms=152):
    return SimpleNamespace(tp=64 // heads, heads=heads, sm=sm, sms=sms)


def test_candidates():
    from sgl_nvidia_mega.glm5_3 import registry

    assert {fd.VERIFY, fd.DRAFT_DECODE, fd.DRAFT_EXTEND} == set(registry.MODES)
    cands, families = registry.collect([fd])
    assert {(c.group, c.name, c.steps, c.family) for c in cands} == {
        (
            "attention",
            "ll.sparse_mla_fd",
            ("sparse_mla", "uv_proj"),
            "ll.sparse_mla_fd",
        ),
        (
            "draft_attention",
            "ll.sparse_mla_fd",
            ("sparse_mla", "uv_proj"),
            "ll.sparse_mla_fd",
        ),
    }
    assert families == {"ll.sparse_mla_fd": fd.SparseMlaFdFamily}


def test_covers(monkeypatch):
    monkeypatch.setattr(fd, "static_reasons", lambda: [])
    for rows in range(1, 8):
        assert fd.covers(profile(16), rows, fd.VERIFY) is None
        assert fd.covers(profile(8), rows, fd.DRAFT_DECODE) is None
    for rows in range(8, 36):
        assert fd.covers(profile(8), rows, fd.VERIFY) is None
        assert "16 heads" in fd.covers(profile(16), rows, fd.VERIFY)
    assert fd.covers(profile(8), 36, fd.VERIFY) is not None
    assert fd.covers(profile(16), 48, fd.VERIFY) is not None
    assert "sm_103" in fd.covers(profile(16, sm=(10, 0)), 5, fd.VERIFY)
    assert "SMs" in fd.covers(profile(16, sms=148), 5, fd.VERIFY)
    assert "heads" in fd.covers(profile(heads=4), 5, fd.VERIFY)


def test_covers_declines_skip_softmax(monkeypatch):
    monkeypatch.setattr(
        fd, "static_reasons", lambda: ["TRTLLM-Gen skip-softmax decode is on"]
    )
    assert "skip-softmax" in fd.covers(profile(16), 5, fd.VERIFY)


def test_mt_splits_for_rows():
    # the most CTAs per row whose clusters fit one wave with one cluster of headroom
    assert [
        fd.mt_splits_for_rows(r, MT) for r in (8, 14, 15, 22, 23, 35, 36, 45, 75, 76)
    ] == [8, 8, 6, 6, 4, 4, 3, 3, 2, 0]
    assert fd.mt_splits_for_rows(15, MT, headroom=0) == 8
    assert fd.mt_splits_for_rows(5, ()) == 0


def test_plan_gb300():
    p8 = fd.make_plan(8, GB300_8)
    assert p8.one_tile_rows == 7 and p8.spread_rows == 7 and p8.fold_rows == 7
    assert [p8.splits(r) for r in (1, 7, 8, 12, 14, 15, 22, 23, 35, 36, 48, 0)] == [
        16,
        16,
        8,
        8,
        8,
        6,
        6,
        4,
        4,
        0,
        0,
        0,
    ]
    assert p8.smem_bytes(6) == GB300_8.spread_smem_bytes
    assert p8.folds(2, fd.DRAFT_DECODE) and not p8.folds(2, fd.VERIFY)
    assert not p8.folds(8, fd.DRAFT_DECODE) and not p8.folds(6, fd.DRAFT_EXTEND)
    p16 = fd.make_plan(16, GB300_16)
    assert p16.one_tile_rows == 7 and p16.mt_splits == () and p16.fold_rows == 0
    assert [p16.splits(r) for r in (1, 5, 6, 7, 8, 12)] == [16, 16, 16, 16, 0, 0]
    # a smaller device: the plan shrinks to its waves (compile then refuses the widths
    # covers promised beyond them)
    small = fd.make_plan(
        8, fd.Caps(105416, 116736, 5, 10, (60, 40, 30, 20, 12), 5, 221976)
    )
    assert small.one_tile_rows == 5 and small.splits(6) == 8 and small.fold_rows == 5
    # every covered width of every profile has a kernel in the GB300 plan
    fd.static_reasons, saved = (lambda: []), fd.static_reasons
    for heads, caps in ((8, GB300_8), (16, GB300_16)):
        plan = fd.make_plan(heads, caps)
        for rows in range(1, 49):
            if fd.covers(profile(heads), rows, fd.VERIFY) is None:
                assert plan.splits(rows), (heads, rows)
    fd.static_reasons = saved


# (case, per-rank (caps, failed), with_mt, with_fold, expected caps or None = raises)
CASES = (
    ("equal", ((GB300_8, False), (GB300_8, False)), True, True, GB300_8),
    (
        "rank 1 has fewer clusters",
        (
            (GB300_8, False),
            (fd.Caps(105416, 116736, 6, 13, (72, 44, 34, 22, 14), 6, 221976), False),
        ),
        True,
        True,
        fd.Caps(105416, 116736, 6, 13, (72, 44, 34, 22, 14), 6, 221976),
    ),
    (
        "per cluster size minimum",
        (
            (fd.Caps(105416, 116736, 7, 14, (76, 44, 36, 22, 15), 7, 221976), False),
            (fd.Caps(105416, 116736, 7, 14, (72, 46, 34, 23, 14), 7, 221976), False),
        ),
        True,
        True,
        fd.Caps(105416, 116736, 7, 14, (72, 44, 34, 22, 14), 7, 221976),
    ),
    (
        "16 heads, one-tile only",
        ((GB300_16, False), (GB300_16, False)),
        False,
        False,
        GB300_16,
    ),
    ("rank 1 build failed", ((GB300_8, False), (None, True)), True, True, None),
    ("rank 0 build failed", ((None, True), (GB300_8, False)), True, True, None),
    (
        "a rank reports 0 clusters",
        ((GB300_16, False), (fd.Caps(134984, 134984, 0, 0), False)),
        False,
        False,
        None,
    ),
    (
        "a rank without the fold it was asked for",
        ((GB300_8, False), (fd.Caps(105416, 116736, 7, 14, MT, 0, 0), False)),
        True,
        True,
        None,
    ),
)


def _worker(rank: int, init_file: str, queue) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    group = SimpleNamespace(world_size=2, cpu_group=dist.group.WORLD)
    got = []
    for _, per_rank, with_mt, with_fold, _ in CASES:
        caps, failed = per_rank[rank]
        try:
            agreed = fd.agree_caps(
                caps,
                RuntimeError("simulated build failure") if failed else None,
                group,
                with_mt=with_mt,
                with_fold=with_fold,
            )
            got.append(("ok", agreed))
        except RuntimeError as e:
            got.append(("raised", str(e)))
    dist.destroy_process_group()
    queue.put((rank, got))


def test_ranks_agree():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    init_file = os.path.join(tempfile.mkdtemp(prefix="fd_caps_"), "init")
    procs = [ctx.Process(target=_worker, args=(r, init_file, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    try:
        while len(results) < 2:
            rank, got = queue.get(timeout=120)
            results[rank] = got
    except queue_mod.Empty:
        pytest.fail(f"only ranks {sorted(results)} reported")
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
    for i, (case, _, _, _, expected) in enumerate(CASES):
        outcomes = [results[r][i] for r in range(2)]
        if expected is None:
            assert all(o[0] == "raised" for o in outcomes), (case, outcomes)
        else:
            assert all(o == ("ok", expected) for o in outcomes), (case, outcomes)
