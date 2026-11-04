# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 / GLM-5.2 unit registry on CPU: selection, multi-step atomicity, the
contract tokens' fixed point, the legacy group token, decline vs failure, the units
switch, parity mode and step-group composition."""

import pytest

from sgl_nvidia_mega.glm5_3 import registry, units
from sgl_nvidia_mega.glm5_3.profile import Profile
from sgl_nvidia_mega.glm5_3.registry import Candidate, Decline, Pref

TP8 = Profile(
    tp=8, heads=8, tokens_per_request=4, widths=(4, 8, 12, 16), sm=(10, 3), sms=152
)
TP4 = Profile(
    tp=4, heads=16, tokens_per_request=6, widths=(6, 12, 18), sm=(10, 3), sms=152
)
V = "target_verify"


def covers_upto(n):
    return lambda p, rows, mode: None if rows <= n else f"serves <= {n} rows"


def ok(p, rows, mode):
    return None


def _select(cands, pref, rows=4, profile=TP8, allowed=lambda c: True, excluded=None):
    return registry.select(
        profile,
        rows,
        V,
        candidates=cands,
        preference=pref,
        groups=("attention", "hops", "mlp"),
        allowed=allowed,
        excluded=excluded,
    )


def _names(comp):
    return {f"{g}.{s}": c.name for (g, s), c in comp.owners.items()}


def test_first_covering_candidate_per_step():
    a = Candidate("a", "mlp", ("forward",), "f", covers_upto(8))
    b = Candidate("b", "mlp", ("forward",), "f", ok)
    pref = {("mlp", "forward"): (Pref("a"), Pref("b"))}
    assert _names(_select([a, b], pref, rows=8)) == {"mlp.forward": "a"}
    comp = _select([a, b], pref, rows=12)
    assert _names(comp) == {"mlp.forward": "b"}
    # a covered nothing at 12 rows: passed over with its reason, but no step is stock
    assert comp.passed[("mlp", "a")] == "serves <= 8 rows"
    assert comp.stock_reasons({c.key: c for c in (a, b)}) == {}


def test_absent_and_unpreferred_names_are_skipped():
    b = Candidate("b", "mlp", ("forward",), "f", ok)
    pref = {
        ("mlp", "forward"): (Pref("absent"), Pref("b", when=lambda p, r, m: p.tp == 4))
    }
    assert _names(_select([b], pref, profile=TP8)) == {}
    assert _names(_select([b], pref, profile=TP4, rows=6)) == {"mlp.forward": "b"}


def test_multi_step_candidates_are_atomic():
    steps = ("qkv_a", "prologue")
    front = Candidate("front", "attention", steps, "f", ok)
    qa = Candidate("qa", "attention", ("qkv_a",), "g", ok)
    pref = {
        ("attention", "qkv_a"): (Pref("qa"), Pref("front")),
        ("attention", "prologue"): (Pref("front"),),
    }
    comp = _select([front, qa], pref)
    # front is not first for qkv_a: not taken at all, prologue stays stock
    assert _names(comp) == {"attention.qkv_a": "qa"}
    assert (
        "not preferred for attention.qkv_a (qa is)"
        == comp.passed[("attention", "front")]
    )
    assert comp.stock_reasons({c.key: c for c in (front, qa)}) == {
        "front": "not preferred for attention.qkv_a (qa is)"
    }


def test_contract_tokens_reach_a_fixed_point():
    moe = Candidate(
        "moe", "mlp", ("forward",), "m", covers_upto(8), provides=("mlp:records",)
    )
    other = Candidate("other_mlp", "mlp", ("forward",), "o", ok)
    fold = Candidate(
        "fold",
        "hops",
        registry.GROUPS["hops"],
        "h",
        ok,
        provides=("hops:dnorm",),
        requires=("mlp:records",),
    )
    far = Candidate("far", "hops", registry.GROUPS["hops"], "h", ok)
    dn = Candidate(
        "front.dn", "attention", ("qkv_a",), "a", ok, requires=("hops:dnorm",)
    )
    front = Candidate("front", "attention", ("qkv_a",), "a", ok)
    pref = {
        ("mlp", "forward"): (Pref("moe"), Pref("other_mlp")),
        **{("hops", s): (Pref("fold"), Pref("far")) for s in registry.GROUPS["hops"]},
        ("attention", "qkv_a"): (Pref("front.dn"), Pref("front")),
    }
    cands = [moe, other, fold, far, dn, front]
    at8 = _names(_select(cands, pref, rows=8))
    assert at8["mlp.forward"] == "moe" and at8["hops.final"] == "fold"
    assert at8["attention.qkv_a"] == "front.dn"
    # at 12 rows the other MLP does not provide the records: the fold is passed over,
    # then the deferred-norm consumer loses its producer
    comp = _select(cands, pref, rows=12)
    at12 = _names(comp)
    assert at12["mlp.forward"] == "other_mlp" and at12["hops.final"] == "far"
    assert at12["attention.qkv_a"] == "front"
    assert comp.passed[("hops", "fold")].startswith("requires mlp:records")


def test_a_bare_group_token_is_any_taken_unit_of_that_group():
    m = Candidate("m", "mlp", ("forward",), "m", covers_upto(8))
    h = Candidate("h", "hops", ("final",), "h", ok, requires=("mlp",))
    pref = {("mlp", "forward"): (Pref("m"),), ("hops", "final"): (Pref("h"),)}
    assert _names(_select([m, h], pref, rows=8)) == {
        "mlp.forward": "m",
        "hops.final": "h",
    }
    assert _names(_select([m, h], pref, rows=12)) == {}


def test_switched_off_units_are_skipped_with_a_reason():
    a = Candidate("a", "mlp", ("forward",), "f", ok)
    pref = {("mlp", "forward"): (Pref("a"),)}
    comp = _select([a], pref, allowed=lambda c: c.name != "a")
    assert (
        comp.all_stock()
        and "SGLANG_NVIDIA_MEGAKERNEL_SLOTS" in comp.passed[("mlp", "a")]
    )


def _slots(monkeypatch, names):
    """SGLANG_NVIDIA_MEGAKERNEL_SLOTS as read at import: None = unset (every shipped slot)."""
    from sgl_nvidia_mega.glm5_3 import runtime

    if names is None:
        monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
        monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))
    else:
        monkeypatch.setenv(runtime.SLOTS_ENV, ",".join(names))
        monkeypatch.setattr(runtime, "SLOTS", frozenset(names))


def test_units_switch(monkeypatch):
    _slots(monkeypatch, None)
    assert units.enabled("ll.moe") and not units.parity()
    assert not units.listed("ll.gemm.q_b")
    _slots(monkeypatch, ["ll.moe", "ll.hops", "ll.gemm.q_b"])
    assert units.enabled("ll.moe") and not units.enabled("ll.attention")
    assert units.listed("ll.gemm.q_b")
    _slots(monkeypatch, [])
    assert units.parity() and not any(units.enabled(n) for n in units.KNOWN)
    # the draft front is the draft slot
    _slots(monkeypatch, ["draft"])
    assert units.enabled("draft.front") and not units.enabled("draft_rowselect")


def test_without_slot_draft_the_draft_units_are_off(monkeypatch):
    from sgl_nvidia_mega.glm5_3 import runtime

    _slots(monkeypatch, [s for s in runtime.SHIPPED_SLOTS if s != "draft"])
    assert not units.enabled("draft_rowselect") and not units.enabled("ll.draft_eh")
    # a unit serving both: off in the draft's groups only
    assert units.enabled("ll.sparse_mla_fd", "attention")
    assert not units.enabled("ll.sparse_mla_fd", "draft_attention")


def test_preference_table_is_consistent():
    for (g, s), prefs in units.PREFERENCE.items():
        assert s in registry.GROUPS[g]
        assert all(isinstance(p, Pref) for p in prefs)
    # every target unit the ll modules register is listed for every step it claims
    cands, families = registry.collect(registry.target_modules())
    for c in cands:
        if c.group in registry.DRAFT_GROUPS:
            continue
        for s in c.steps:
            assert c.name in [p.name for p in units.PREFERENCE.get((c.group, s), ())], (
                c,
                s,
            )
        assert c.name in units.KNOWN
    assert families


# ---- build: families, decline vs failure, composition, wiring


class StockMlp:
    def forward(self):
        return "stock"


class StockHops:
    def pre_attention(self):
        return "stock"

    def post_attention(self):
        return "stock"

    def final(self):
        return "stock"


class Mlp(StockMlp):
    def __init__(self, tag):
        self.tag = tag

    def forward(self):
        return self.tag


class Hops(StockHops):
    def __init__(self):
        self.wired = []

    def final(self):
        return "hop"

    def post_attention(self):
        return "hop"

    def pre_attention(self):
        return "hop"

    def wire(self, token):
        self.wired.append(token)


class Fam:
    def __init__(self, log, name, make, decline=(), fail_compile=False, fail_rows=()):
        self.log, self.name, self.make = log, name, make
        self.decline, self.fail_compile, self.fail_rows = (
            decline,
            fail_compile,
            fail_rows,
        )

    def compile(self, widths):
        self.log.append((self.name, tuple(widths)))
        if self.fail_compile:
            raise RuntimeError("nvcc says no")

    def provider(self, c, rows, mode):
        if rows in self.decline:
            raise Decline(f"{self.name} declines {rows}")
        if rows in self.fail_rows:
            raise ValueError("broken")
        return self.make(c, rows)


def _build(families, cands, pref, widths=(4, 8, 12), gather=None, allowed=None):
    return registry.build(
        None,
        None,
        TP8,
        steps=[(V, w) for w in widths],
        groups={"hops": StockHops, "mlp": StockMlp},
        candidates=cands,
        families=families,
        preference=pref,
        allowed=allowed or (lambda c: True),
        gather=gather or (lambda x: [x]),
    )


HOPS = registry.GROUPS["hops"]


def _units():
    moe = Candidate(
        "moe", "mlp", ("forward",), "m", covers_upto(8), provides=("mlp:records",)
    )
    alt = Candidate("alt", "mlp", ("forward",), "a", ok)
    fold = Candidate("fold", "hops", HOPS, "h", ok, requires=("mlp:records",))
    pref = {
        ("mlp", "forward"): (Pref("moe"), Pref("alt")),
        **{("hops", s): (Pref("fold"),) for s in HOPS},
    }
    return [moe, alt, fold], pref


def test_build_compiles_taken_families_at_their_covered_widths():
    log = []
    cands, pref = _units()
    fams = {
        "m": lambda *a: Fam(log, "m", lambda c, r: Mlp(f"moe{r}")),
        "a": lambda *a: Fam(log, "a", lambda c, r: Mlp(f"alt{r}")),
        "h": lambda *a: Fam(log, "h", lambda c, r: Hops()),
    }
    b = _build(fams, cands, pref)
    assert log == [("m", (4, 8)), ("a", (4, 8, 12)), ("h", (4, 8, 12))]
    assert b.groups[(V, 4)]["mlp"].forward() == "moe4"
    assert b.groups[(V, 12)]["mlp"].forward() == "alt12"
    assert isinstance(b.groups[(V, 12)]["hops"], StockHops)
    assert type(b.groups[(V, 12)]["hops"]) is StockHops  # fold needs the records
    assert b.groups[(V, 8)]["hops"].final() == "hop"


def test_a_decline_is_agreed_and_recomposed():
    log = []
    cands, pref = _units()
    fams = {
        "m": lambda *a: Fam(log, "m", lambda c, r: Mlp("moe"), decline=(8,)),
        "a": lambda *a: Fam(log, "a", lambda c, r: Mlp("alt")),
        "h": lambda *a: Fam(log, "h", lambda c, r: Hops()),
    }
    # rank 1 declines 4 rows as well: every rank recomposes both widths
    calls = []

    def gather(x):
        calls.append(x)
        if isinstance(x, tuple) and x[1]:
            other = (None, [((V, 4), [(("mlp", "moe"), "rank-1 cap")])])
            return [x, other]
        return [x, x]

    b = _build(fams, cands, pref, gather=gather)
    assert b.groups[(V, 8)]["mlp"].forward() == "alt"
    assert b.groups[(V, 4)]["mlp"].forward() == "alt"
    assert b.compositions[(V, 4)].passed[("mlp", "moe")] == "rank 1: rank-1 cap"
    assert b.compositions[(V, 8)].passed[("mlp", "moe")] == "m declines 8"


def test_a_family_failure_raises_on_every_rank():
    cands, pref = _units()
    fams = {
        "m": lambda *a: Fam([], "m", lambda c, r: Mlp("moe")),
        "a": lambda *a: Fam([], "a", lambda c, r: Mlp("alt")),
        "h": lambda *a: Fam([], "h", lambda c, r: Hops()),
    }

    # this rank builds fine; rank 3 failed to compile
    def gather(x):
        return (
            [x, x, x, "RuntimeError: nvcc says no"]
            if not isinstance(x, tuple)
            else [x] * 4
        )

    with pytest.raises(RuntimeError, match="rank 3: RuntimeError: nvcc says no") as e:
        _build(fams, cands, pref, gather=gather)
    assert "SGLANG_NVIDIA_MEGAKERNEL_SLOTS: moe" in str(e.value)
    # the failing rank itself
    fams["m"] = lambda *a: Fam([], "m", lambda c, r: Mlp("moe"), fail_compile=True)
    with pytest.raises(RuntimeError, match="'m' kernel family failed"):
        _build(fams, cands, pref)
    fams["m"] = lambda *a: None
    with pytest.raises(RuntimeError, match="returned None"):
        _build(fams, cands, pref)
    fams["m"] = lambda *a: Fam([], "m", lambda c, r: Mlp("moe"), fail_rows=(8,))
    with pytest.raises(RuntimeError, match="moe at 8 rows"):
        _build(fams, cands, pref)


def test_parity_mode_builds_nothing():
    log = []
    cands, pref = _units()
    fams = {
        k: (lambda k: lambda *a: Fam(log, k, lambda c, r: Mlp(k)))(k) for k in "mah"
    }
    b = _build(fams, cands, pref, allowed=lambda c: False)
    assert log == [] and b.families == {}
    assert all(c.all_stock() for c in b.compositions.values())
    assert type(b.groups[(V, 4)]["mlp"]) is StockMlp


def test_wiring_and_mixed_groups():
    dn = Candidate("dn", "mlp", ("forward",), "m", ok, requires=("hops:dnorm",))
    pre = Candidate(
        "pre", "hops", ("pre_attention",), "h", ok, provides=("hops:dnorm",)
    )
    post = Candidate("post", "hops", ("post_attention", "final"), "g", ok)
    pref = {
        ("mlp", "forward"): (Pref("dn"),),
        ("hops", "pre_attention"): (Pref("pre"),),
        ("hops", "post_attention"): (Pref("post"),),
        ("hops", "final"): (Pref("post"),),
    }

    class Pre(StockHops):
        def __init__(self):
            self.wired = []

        def pre_attention(self):
            return "pre"

        def wire(self, token):
            self.wired.append(token)

    class Post(StockHops):
        def post_attention(self):
            return "post"

        def final(self):
            return "post-final"

    made = {}
    fams = {
        "m": lambda *a: Fam([], "m", lambda c, r: Mlp("dn")),
        "h": lambda *a: Fam([], "h", lambda c, r: made.setdefault(r, Pre())),
        "g": lambda *a: Fam([], "g", lambda c, r: Post()),
    }
    b = _build(fams, [dn, pre, post], pref, widths=(4,))
    hops = b.groups[(V, 4)]["hops"]
    assert (hops.pre_attention(), hops.post_attention(), hops.final()) == (
        "pre",
        "post",
        "post-final",
    )
    assert made[4].wired == ["hops:dnorm"]


def test_compose_group_keeps_a_single_provider_object():
    class Stock:
        def forward(self):
            return self.step()

        def step(self):
            return "stock"

        def other(self):
            return "stock"

    class Full(Stock):
        def step(self):
            return "full"

    class OverridesOther(Stock):
        def step(self):
            return "x"

        def other(self):
            return "x-other"

    full = Full()
    assert (
        registry.compose_group(Stock(), ("forward", "step", "other"), {"step": full})
        is full
    )
    # the provider overrides a step it does not claim: the stock body serves that step
    g = registry.compose_group(
        Stock(), ("forward", "step", "other"), {"step": OverridesOther()}
    )
    assert (g.forward(), g.other()) == ("x", "stock")


def test_describe_and_index_steps_follow_the_composition():
    from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep as AttentionDecode
    from sgl_nvidia_mega.glm5_3.boundary.boundaries import HopSteps
    from sgl_nvidia_mega.glm5_3.moe.step import MlpSteps
    from sgl_nvidia_mega.glm5_3.plan import Providers

    class Core(AttentionDecode):
        def sparse_mla(self, *a):
            return "core"

    class Index(AttentionDecode):
        def index_k(self, *a):
            return "k"

    att = registry.compose_group(
        AttentionDecode(),
        registry.GROUPS["attention"],
        {"sparse_mla": Core(), "index_k": Index()},
    )
    d = Providers(attention=att, hops=HopSteps(), mlp=MlpSteps()).describe()
    assert d["attention.sparse_mla"].endswith("Core")
    assert d["attention.index_k"].endswith("Index")
    assert d["attention.qkv_a"] == "AttentionStep"
    assert not att._stock_index_steps() and AttentionDecode()._stock_index_steps()


# ---- the target's units: the TP4 profiles only (the TP8 envelope builds its own families)


def test_the_target_units_are_the_ll_modules():
    cands, _ = registry.collect(registry.target_modules())
    assert cands and all(c.name.startswith("ll.") for c in cands)


def test_forced_build_error(monkeypatch):
    cands, pref = _units()
    fams = {k: (lambda k: lambda *a: Fam([], k, lambda c, r: Mlp(k)))(k) for k in "mah"}
    monkeypatch.setenv("SGLANG_NVIDIA_MEGAKERNEL_GLM_FORCE_BUILD_ERROR", "m")
    with pytest.raises(RuntimeError, match="forced build error"):
        _build(fams, cands, pref)


def test_a_claimed_step_keeps_its_providers_helpers():
    """WP2's case: ll.attention claims index_k with its own _index_kw (registered wk|wp
    GEMM) and inherits the stock index_k; another unit claims the core."""

    from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep as AttentionDecode

    calls = []

    class Front(AttentionDecode):
        def qkv_a(self, *a):
            return "front"

        @staticmethod
        def _index_kw(indexer, x, step):
            calls.append("front._index_kw")
            return "k", "w"

    class Core(AttentionDecode):
        def sparse_mla(self, *a):
            return "core"

    front, core = Front(), Core()
    group = registry.compose_group(
        AttentionDecode(),
        registry.GROUPS["attention"],
        {"qkv_a": front, "index_k": front, "sparse_mla": core},
    )
    assert group._index_kw(None, None, None) == ("k", "w")
    # the stock two-stream index path still runs (index_k is the inherited body)
    assert group._stock_index_steps()
    assert calls == ["front._index_kw"]
    # one provider alone stays that provider (its helper is its own)
    alone = {"qkv_a": front, "index_k": front}
    assert (
        registry.compose_group(AttentionDecode(), registry.GROUPS["attention"], alone)
        is front
    )

    class OtherKw(AttentionDecode):
        @staticmethod
        def _index_kw(indexer, x, step):
            return None

        def index_q(self, *a):
            return "q"

    with pytest.raises(ValueError, match="_index_kw"):
        registry.compose_group(
            AttentionDecode(),
            registry.GROUPS["attention"],
            {"index_k": front, "index_q": OtherKw()},
        )

    # a lone provider overriding the helper of a step it does not claim is not
    # served as itself: the unclaimed steps keep the stock helper
    class SneakyKw(AttentionDecode):
        @staticmethod
        def _index_kw(indexer, x, step):
            return "sneaky"

        def sparse_mla(self, *a):
            return "sneaky-core"

    sneaky = SneakyKw()
    g = registry.compose_group(
        AttentionDecode(), registry.GROUPS["attention"], {"sparse_mla": sneaky}
    )
    assert g is not sneaky and isinstance(g, AttentionDecode)
    assert g._index_kw is AttentionDecode._index_kw
    assert g.sparse_mla() == "sneaky-core"
