# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The ``ll.attention`` unit in the registry, on CPU: its coverage, the composition
at TP4 (the target's verify steps and the NextN draft layer's) and at the TP8
envelope (none), the units switch and the fold option unit, the provider object a composed
group keeps, and its named tolerance."""

from types import SimpleNamespace

import pytest
from glm5_3_slots import set_slots

from sgl_nvidia_mega.glm5_3 import ll, registry, units
from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep as AttentionDecode
from sgl_nvidia_mega.glm5_3.ll import attention as lla
from sgl_nvidia_mega.glm5_3.ll import norm_backend
from sgl_nvidia_mega.glm5_3.profile import Profile

TP4 = Profile(
    tp=4,
    heads=16,
    tokens_per_request=6,
    widths=tuple(range(6, 49, 6)),
    sm=(10, 3),
    sms=152,
)
TP8 = Profile(
    tp=8, heads=8, tokens_per_request=4, widths=(4, 8, 12, 16), sm=(10, 3), sms=152
)
CLAIMED = {f"attention.{s}" for s in lla.STEPS}


@pytest.fixture(autouse=True)
def _all_units(monkeypatch):
    set_slots(monkeypatch)


def _select(profile, rows, mode="target_verify", groups=registry.TARGET_GROUPS):
    cands, _ = registry.collect(registry.target_modules())
    return registry.select(
        profile,
        rows,
        mode,
        candidates=cands,
        preference=units.PREFERENCE,
        groups=groups,
        allowed=units.allowed,
    ), {c.key: c for c in cands}


def _owners(comp, group="attention"):
    return {f"attention.{s}": c.name for (g, s), c in comp.owners.items() if g == group}


def test_registered():
    assert "attention" in ll.modules() and ll.module("attention") is lla
    assert lla.FAMILIES == {"ll.attention": lla._family}
    assert {c.group for c in lla.CANDIDATES} == {"attention", "draft_attention"}
    for c in lla.CANDIDATES:
        assert c.name == "ll.attention" and c.steps == lla.STEPS
        for s in c.steps:
            assert "ll.attention" in [p.name for p in units.PREFERENCE[(c.group, s)]]
    assert lla.FOLD_UNIT in units.KNOWN
    assert ll.tolerances()["ll.attention.qkv_a_wide.max_rel"] == 8e-3


@pytest.mark.parametrize(
    "profile,rows,mode,why",
    [
        (TP4, 6, "target_verify", None),
        (TP4, 48, "target_verify", None),
        (TP4, 6, "draft_extend", None),
        (TP4, 3, "decode", None),
        (TP4, 49, "target_verify", "serves 1-48 rows"),
        (TP8, 4, "target_verify", "needs 16 attention heads per rank"),
        (Profile(4, 16, 6, (6,), (10, 0), 148), 6, "target_verify", "SM103 only"),
    ],
)
def test_covers(profile, rows, mode, why):
    got = lla._covers(profile, rows, mode)
    assert (got is None) if why is None else (why in got), got


@pytest.mark.parametrize("rows", (5, 6, 10, 12, 18, 24, 30, 36, 42, 48))
def test_tp4_verify_composition(rows):
    comp, _ = _select(TP4, rows)
    owners = _owners(comp)
    assert {k for k, v in owners.items() if v == "ll.attention"} == CLAIMED
    assert all(c.name.startswith("ll.") for c in comp.chosen)


@pytest.mark.parametrize("rows,mode", [(6, "draft_extend"), (2, "decode")])
def test_tp4_draft_composition(rows, mode):
    comp, _ = _select(TP4, rows, mode, groups=("draft_attention",))
    owners = _owners(comp, "draft_attention")
    assert {k for k, v in owners.items() if v == "ll.attention"} == CLAIMED


@pytest.mark.parametrize("rows", (4, 8, 12, 16))
def test_tp8_takes_no_ll_attention(rows):
    comp, cands = _select(TP8, rows)
    assert "ll.attention" not in {c.name for c in comp.chosen}


def test_switched_off(monkeypatch):
    set_slots(monkeypatch, without=["ll.attention", lla.FOLD_UNIT])
    comp, cands = _select(TP4, 6)
    assert "ll.attention" not in {c.name for c in comp.chosen}
    assert "switched off" in comp.stock_reasons(cands)["ll.attention"]


def _family_stub(fold=True):
    attn = SimpleNamespace(kv_a_layernorm=None)
    fam = object.__new__(lla.AttentionFamily)
    fam.attns, fam.eps, fam.device = [attn], 1e-5, "cpu"
    fam.forms = {6: lla.plan_forms(6), 12: lla.plan_forms(12)}
    fam.fold = norm_backend.NormFold(fold, "test")
    fam.trigger_at_entry = False
    fam._served = {}
    return fam


def test_fold_option_unit(monkeypatch):
    assert units.enabled(lla.FOLD_UNIT)
    set_slots(monkeypatch, without=[lla.FOLD_UNIT])
    assert not units.enabled(lla.FOLD_UNIT) and units.enabled("ll.attention")
    fold = _family_stub()._decide_fold((6, 12))  # no probe runs (no norm module here)
    assert not fold.fold and lla.FOLD_UNIT in fold.reason


def test_provider_object_per_width():
    fam = _family_stub()
    c = lla.CANDIDATES[0]
    p6 = fam.provider(c, 6, "target_verify")
    assert isinstance(p6, AttentionDecode) and p6.rows == 6 and p6.fold
    assert fam.provider(lla.CANDIDATES[1], 6, "draft_extend") is p6
    assert fam.provider(c, 12, "target_verify").forms.q_b == ("ll", True)
    with pytest.raises(RuntimeError, match="no forms built for 18 rows"):
        fam.provider(c, 18, "target_verify")


def test_target_group_is_the_provider():
    """Owning every claimed step of the target group alone, the composed group is the
    provider itself: the stock index steps then project through its _index_kw."""
    p = _family_stub().provider(lla.CANDIDATES[0], 6, "target_verify")
    group = registry.compose_group(
        AttentionDecode(), registry.GROUPS["attention"], {s: p for s in lla.STEPS}
    )
    assert group is p and group._stock_index_steps()
    assert group._index_kw.__func__ is lla.LLAttention._index_kw


def test_mixed_group_keeps_the_projection():
    """Beside another unit's core steps (ll.sparse_mla_fd at TP4 <= 7 rows) the
    composed group is the stock one with ll.attention's steps and its _index_kw
    bound: the two-stream index path stays and projects through ll.linear."""

    class Core(AttentionDecode):
        def sparse_mla(self, *a):
            return "core"

        def uv_proj(self, *a):
            return "core"

    p = _family_stub().provider(lla.CANDIDATES[0], 6, "target_verify")
    core = Core()
    owners = {s: p for s in lla.STEPS}
    owners.update(sparse_mla=core, uv_proj=core)
    group = registry.compose_group(
        AttentionDecode(), registry.GROUPS["attention"], owners
    )
    assert group is not p and isinstance(group, AttentionDecode)
    for s in lla.STEPS:
        # ll.attention's own steps run on it; the step bodies it inherits are the stock
        # ones (on the composed group), with its helpers
        assert getattr(group, s).__func__ is getattr(type(p), s), s
        if getattr(type(p), s) is not getattr(AttentionDecode, s):
            assert getattr(group, s).__self__ is p, s
    assert group._index_kw.__self__ is p
    assert group._index_kw.__func__ is lla.LLAttention._index_kw
    assert group._stock_index_steps()
    assert group.sparse_mla.__self__ is core


def test_plan_forms():
    assert lla.plan_forms(5) == lla.Forms(("ll", False, True), None)
    assert lla.plan_forms(9) == lla.Forms(("ll", True, True), ("ll", True))
    assert lla.plan_forms(16).q_b == ("ll", True)
    assert lla.plan_forms(17) == lla.Forms(("wide", lla.QKV_A_WIDE), None)
    assert lla.plan_forms(24).qkv_a[0] == "wide"
    assert lla.plan_forms(25) == lla.Forms(None, None)
    # the occupancy gate keeps SGLang's GEMM above 16 rows
    assert lla.plan_forms(24, wide_max_rows=16).qkv_a is None
