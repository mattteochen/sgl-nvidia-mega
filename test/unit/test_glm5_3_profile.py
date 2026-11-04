# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 / GLM-5.2 build profile: the accepted (TP, tokens per request) table, the MTP
pairing and the widths a build serves (CPU)."""

from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.glm5_3 import profile as P


def _args(tp=8, tokens=4, steps=None, **kw):
    base = dict(
        tp_size=tp,
        nnodes=1,
        speculative_algorithm="EAGLE",
        speculative_num_draft_tokens=tokens,
        speculative_num_steps=tokens - 1 if steps is None else steps,
        enable_hierarchical_cache=False,
        disaggregation_decode_enable_offload_kvcache=False,
        enable_lmcache=False,
        enable_flexkv=False,
        disaggregation_mode="null",
    )
    base.update(kw)
    return SimpleNamespace(**base)


HF = SimpleNamespace(num_attention_heads=64)


def test_accepted_table():
    # the TP8 envelope (plan.build_providers) and the TP4 kernel-unit profiles
    assert P.TP8 == {(8, 4): 16, (8, 6): 16}
    assert P.UNITS == {(4, t): 48 for t in (3, 4, 5, 6)}
    assert P.ACCEPTED == {**P.TP8, **P.UNITS}
    assert P.MAX_ROWS == max(P.ACCEPTED.values())


@pytest.mark.parametrize(
    "tp,tokens,units", [(8, 4, False), (8, 6, False), (4, 5, True), (4, 6, True)]
)
def test_the_units_run_the_tp4_profiles_only(tp, tokens, units):
    p = P.Profile.from_args(_args(tp, tokens), HF, device_capability=(10, 3))
    assert p.units is units


@pytest.mark.parametrize(
    "tp,tokens,widths",
    [
        (8, 4, (4, 8, 12, 16)),
        (8, 6, (6, 12)),
        (8, 5, ()),
        (4, 6, (6, 12, 18, 24, 30, 36, 42, 48)),
        (4, 5, (5, 10, 15, 20, 25, 30, 35, 40, 45)),
        (2, 4, ()),
    ],
)
def test_widths_from_args(tp, tokens, widths):
    p = P.Profile.from_args(_args(tp, tokens), HF, device_capability=(10, 3))
    assert p.widths == widths
    assert p.heads == 64 // tp and p.tp == tp and p.tokens_per_request == tokens
    assert p.sm == (10, 3)


def test_clip_widths_to_the_envelope():
    # SGLang asks for range(tokens, max_rows + 1, tokens)
    assert P.clip_widths(range(6, 49, 6), 8, 6) == (6, 12)
    assert P.clip_widths((4, 8, 12, 16), 8, 4) == (4, 8, 12, 16)
    assert P.clip_widths((6, 12), 2, 6) == ()


def test_features():
    p = P.Profile.from_args(_args(enable_hierarchical_cache=True, nnodes=2), HF)
    assert p.kv_offload and p.multi_node and not p.pd_decode
    p = P.Profile.from_args(_args(4, 6, disaggregation_mode="decode"), HF)
    assert p.pd_decode and not p.kv_offload
    p = P.Profile.from_args(_args(4, 6, disaggregation_mode="prefill"), HF)
    assert not p.pd_decode


@pytest.mark.parametrize("tp,tokens", sorted(P.ACCEPTED))
def test_accepted_pairs_have_no_reason(tp, tokens):
    assert P.profile_reasons(_args(tp, tokens)) == []


def test_reasons_name_the_pairing_and_the_envelope():
    (why,) = P.profile_reasons(_args(8, 4, steps=1))
    assert "--speculative-num-steps" in why
    (why,) = P.profile_reasons(_args(2, 4))
    assert "TP2 with 4 tokens" in why
    assert len(P.profile_reasons(_args(8, 7, steps=2))) == 2


def test_from_build_takes_the_context(monkeypatch):
    import sglang.srt.runtime_context as rc

    monkeypatch.setattr(rc, "get_server_args", lambda: _args(8, 6))
    model = SimpleNamespace(config=HF)
    ctx = SimpleNamespace(
        widths=tuple(range(6, 49, 6)), tokens_per_request=6, deadline_s=30.0
    )
    p = P.Profile.from_build(model, ctx)
    assert p.widths == (6, 12) and p.deadline_s == 30.0 and p.max_rows == 16


def test_from_build_reads_the_resolved_arguments(monkeypatch):
    """A model's resolved defaults are declarations over the record (whose fields keep
    the raw 'auto'); the build reads them as resolution decided."""
    import sglang.srt.runtime_context as rc

    raw = _args(4, 6)
    raw.tp_size = 4
    raw._resolved_overrides = [("model", {"enable_hierarchical_cache": True})]
    monkeypatch.setattr(rc, "get_server_args", lambda: raw)
    p = P.Profile.from_build(SimpleNamespace(config=HF), None)
    assert p.kv_offload and p.tp == 4 and p.widths[-1] == 48
