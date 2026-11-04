# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM low-latency hops' launch plan on CPU: the FlashInfer geometry replica and
the measured per-world plan tables (08-mnnvl-hop-fast section 10.4, 45-unified-moe-hop
section 11, 02-moe-input-fp4-from-ar section 12) on 152-SM GB300 trays at H6144."""

from __future__ import annotations

import pytest

from sgl_nvidia_mega.glm5_3.ll import hops, transport

H, SM = 6144, 152


def _ref_grid(rows: int, hidden: int, sm: int):
    """FlashInfer v0.7.1 adjustGridConfig(useCluster=true), transcribed from C++."""
    threads_per_token = (hidden + 7) // 8
    cluster, loads = 8, 1
    while threads_per_token % cluster != 0 and cluster > 1:
        cluster //= 2
    max_cluster = cluster
    block = (threads_per_token + cluster - 1) // cluster
    while block < 128 and cluster >= 2:
        block, cluster = block * 2, cluster // 2
    while rows * cluster > sm and cluster > 1 and block <= 512:
        block, cluster = block * 2, cluster // 2
    while cluster < max_cluster:
        cc = cluster * 2
        cb = (threads_per_token + cc - 1) // cc
        if cb < 64 or rows * cc > sm:
            break
        cluster, block = cc, cb
    while block > 1024:
        if cluster < 8:
            cluster <<= 1
        elif loads < 8:
            loads += 1
        else:
            break
        block = (threads_per_token + cluster * loads - 1) // (cluster * loads)
    return block, cluster, loads


@pytest.mark.parametrize("hidden", [2048, 4096, 6144, 7168, 8192, 5120])
@pytest.mark.parametrize("sm", [132, 148, 152, 160])
def test_geometry_replica(hidden, sm):
    for rows in range(1, 130):
        assert hops.flashinfer_oneshot_grid(rows, hidden, sm) == _ref_grid(
            rows, hidden, sm
        )


def _label(plan):
    if plan is None:
        return None
    kind = "ts" if plan.two_shot else ("uc" if plan.unicast else "mc")
    return f"{kind}{plan.cluster}x{plan.block}{'+c' if plan.max_carveout else ''}"


def _expect_w4_plain(rows):
    if rows <= 7:
        return "mc8x96+c"
    if rows == 8:
        return "uc8x96+c"
    if rows <= 16:
        return "uc8x96"
    if rows <= 19:
        return "uc8x96+c"
    if rows <= 36:
        return "uc4x192+c"
    if rows <= 38:
        return "ts4x192+c"
    if rows <= 48:
        return "ts2x384+c"
    return None


def test_w4_plain_plan():
    for rows in range(1, 60):
        assert _label(hops.plain_plan(4, rows, H, SM)) == _expect_w4_plain(rows), rows


def test_w4_finalize_plan():
    for rows in range(1, 60):
        got = _label(hops.finalize_plan(4, rows, H, SM))
        if rows <= 36:
            want = _expect_w4_plain(rows)
        elif rows <= 38:
            want = "uc4x192+c"  # one-shot on FlashInfer's geometry above the cap
        elif rows <= 48:
            want = "uc2x384+c"
        else:
            want = None
        assert got == want, rows


def test_w4_nvfp4_plan():
    for rows in range(1, 49):
        got = _label(hops.nvfp4_plan(4, rows, H, SM))
        want = None if rows <= 3 or rows == 21 else _expect_w4_plain(rows)
        assert got == want, rows


def test_w8_plans():
    for rows in range(1, 60):
        if rows <= 11 or 13 <= rows <= 15:
            want = "mc8x96+c"
        elif rows in (12, 16):
            want = "mc8x96"
        elif rows <= 18:
            want = "mc4x192+c"
        elif rows <= 19:
            want = "ts8x96+c"
        elif rows <= 38:
            want = "ts4x192+c"
        elif rows <= 48:
            want = "ts2x384+c"
        else:
            want = None
        assert _label(hops.plain_plan(8, rows, H, SM)) == want, rows
        assert _label(hops.finalize_plan(8, rows, H, SM)) == want, rows
        skip = 11 <= rows <= 16 or rows == 36
        assert _label(hops.nvfp4_plan(8, rows, H, SM)) == (None if skip else want), rows


def test_buffer_holds_every_plan():
    for world in (4, 8):
        nbytes = transport.buffer_bytes(world, 48, H)
        assert nbytes % transport.BUFFER_ALIGN == 0
        for rows in range(1, 49):
            for plan in (
                hops.plain_plan(world, rows, H, SM),
                hops.finalize_plan(world, rows, H, SM),
            ):
                need = (
                    transport.twoshot_bytes
                    if plan.two_shot
                    else transport.oneshot_bytes
                )(world, rows, H)
                assert need <= nbytes, (world, rows)


def test_plans_cover_whole_rows():
    for world in (4, 8):
        for rows in range(1, 49):
            for plan in (
                hops.plain_plan(world, rows, H, SM),
                hops.finalize_plan(world, rows, H, SM),
            ):
                assert plan.block * plan.cluster * 8 == H
                assert plan.block % 32 == 0 and plan.block <= 1024
                assert 1 <= plan.cluster <= 8 and (plan.two_shot or plan.cluster >= 2)


# ------------------------------------------------------------------ the units
def _compose(profile, rows, allowed=lambda c: True):
    from sgl_nvidia_mega.glm5_3 import registry, units

    cands, _ = registry.collect(registry.target_modules())
    return registry.select(
        profile,
        rows,
        "target_verify",
        candidates=cands,
        preference=units.PREFERENCE,
        groups=registry.TARGET_GROUPS,
        allowed=allowed,
    )


def _profile(tp, tokens):
    from sgl_nvidia_mega.glm5_3.profile import MAX_ROWS, Profile, clip_widths

    return Profile(
        tp=tp,
        heads=64 // tp,
        tokens_per_request=tokens,
        widths=clip_widths(tuple(range(tokens, MAX_ROWS + 1, tokens)), tp, tokens),
        sm=(10, 3),
        sms=SM,
    )


@pytest.mark.parametrize("tokens", [4, 5, 6])
def test_tp4_hops_and_moe_are_taken_at_every_width(tokens):
    from sgl_nvidia_mega.glm5_3 import ll

    assert {"hops", "moe"} <= set(ll.modules())
    p = _profile(4, tokens)
    assert p.widths and max(p.widths) <= 48
    for rows in p.widths:
        comp = _compose(p, rows)
        owners = {f"{g}.{s}": c.name for (g, s), c in comp.owners.items()}
        for s in hops.HOP_STEPS:
            assert owners[f"hops.{s}"] == "ll.hops", (rows, owners)
        assert owners["mlp.forward"] == "ll.moe.nvfp4", (rows, owners)
        comp = _compose(p, rows, allowed=lambda c: c.name != "ll.moe.nvfp4")
        assert {c.name for c in comp.chosen} >= {"ll.hops", "ll.moe"}, rows


def test_ll_moe_needs_the_finalizing_hop():
    p = _profile(4, 6)
    comp = _compose(p, 6, allowed=lambda c: c.name != "ll.hops")
    names = {c.name for c in comp.chosen}
    assert not names & {"ll.hops", "ll.moe", "ll.moe.nvfp4"}, names


def test_tp8_and_other_gpus_decline():
    from dataclasses import replace

    for rows in (4, 6, 8, 12, 16):
        names = {c.name for c in _compose(_profile(8, 4), rows).chosen}
        assert not names & {"ll.hops", "ll.moe"}, (rows, names)
    p = replace(_profile(4, 6), sm=(10, 0))
    assert hops.covers(p, 6, "target_verify").startswith("sm_100")
    assert hops.covers(_profile(4, 6), 6, "decode") is not None
    assert hops.covers(_profile(4, 6), 60, "target_verify") is not None


def test_shared_expert_priority_stream_is_gated_by_rows():
    from types import SimpleNamespace

    from sgl_nvidia_mega.glm5_3.ll import moe

    stream = object()
    fam = SimpleNamespace(shared_stream=stream)
    for rows in range(1, 49):
        want = stream if rows <= moe.PRIORITY_MAX_ROWS else None
        assert moe.LlMlp(fam, rows).shared_stream is want, rows
    off = SimpleNamespace(shared_stream=None)
    assert all(moe.LlMlp(off, r).shared_stream is None for r in (1, 5, 6, 12))
