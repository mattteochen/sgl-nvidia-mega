# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.gemm``'s plan (``gemm_plan``) and its selection logic, on CPU.

- Every split-K range's tactic is valid at every row it serves (the vendored copy of
  FlashInfer's ``validate_tactic``: tiles, stages, shared memory, K partition) and fits
  one wave of GB300's 152 SMs; ranges are ordered, disjoint and within ``MAX_M``; TGV
  serves BF16 roles only.
- The preferred entries are plan.md C.4's TP4 roles; the TP4 LM head and the TP8
  shapes are measured but deferred. wk_weights_proj is on split-K at 1-64 rows and on
  SGLang's GEMM above (branch 50 item 14).
- The plan engages only on SM103 with 152 SMs.
- The switches (``..._GLM_GEMM_ROLES``, ``..._GLM_GEMM_SPLITK``), the detection of
  FlashInfer's explicit-tactic API (fi-1) and of its synchronization-fix marker: by
  default FlashInfer's kernel runs only with the marker, else the vendored copy.
- The stock GEMM per module kind, and that a row the plan serves but nobody built
  raises instead of compiling under capture.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from glm5_3_slots import set_slots

from sgl_nvidia_mega.glm5_3 import ll, units
from sgl_nvidia_mega.glm5_3.ll import gemm
from sgl_nvidia_mega.glm5_3.ll import gemm_plan as plan

SM103, GB300_SMS = plan.MEASURED_DEVICE


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    set_slots(monkeypatch)
    monkeypatch.delenv(gemm.SPLITK_ENV, raising=False)
    gemm.reset()
    yield
    gemm.reset()


class _Weight:
    """What the plan reads of a weight, without allocating it."""

    def __init__(self, shape, dtype=torch.bfloat16, *, contiguous=True, ptr=0):
        self.shape, self.dtype = torch.Size(shape), dtype
        self._contiguous, self._ptr = contiguous, ptr

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._ptr


def _module(shape, *, bias=None, dtype=torch.bfloat16, contiguous=True, ptr=0):
    return SimpleNamespace(
        weight=_Weight(shape, dtype, contiguous=contiguous, ptr=ptr), bias=bias
    )


# ------------------------------------------------------------------- the plan


def test_entries_are_unique_and_ranges_disjoint():
    seen = set()
    for e in plan.PLAN:
        assert e.role in plan.ROLE_DTYPES
        assert (e.role, e.shape) not in seen
        seen.add((e.role, e.shape))
        last = 0
        for r in e.ranges:
            assert last < r.m_lo <= r.m_hi <= plan.MAX_M, e
            last = r.m_hi
            if r.kernel == plan.TGV:
                assert plan.ROLE_DTYPES[e.role] == "bfloat16" and not r.tactic
            else:
                assert r.kernel == plan.SPLITK and len(r.tactic) == 4


def test_preferred_entries_are_the_tp4_roles():
    preferred = {(e.role, e.shape) for e in plan.PLAN if e.preferred}
    assert preferred == {
        ("router", (256, 6144)),
        ("wk_weights_proj", (160, 6144)),
        ("o_proj", (6144, 4096)),
        ("dense_down", (6144, 3072)),
        ("eh_proj", (6144, 12288)),
        ("eh_proj", (6144, 3072)),
        ("lm_head", (38720, 6144)),
    }
    deferred = {(e.role, e.shape) for e in plan.PLAN if not e.preferred}
    assert deferred == {
        ("o_proj", (6144, 2048)),
        ("q_b", (2048, 2048)),
    }


def test_splitk_tactics_valid_and_one_wave_at_every_served_row():
    splitk = pytest.importorskip("sgl_nvidia_mega.glm5_3.ll.kernels.cute.splitk")
    checked = 0
    for e in plan.PLAN:
        n, k = e.shape
        for r in e.ranges:
            if r.kernel != plan.SPLITK:
                continue
            for m in range(r.m_lo, r.m_hi + 1):
                splitk.SplitKTactic(*r.tactic).validate(m, n, k)
                assert plan.splitk_fits_one_wave(r.tactic, m, n, GB300_SMS), (e.role, m)
                checked += 1
    assert checked == (96 - 16) + 64  # router 17-96, wk 1-64


def test_splitk_grid_sizes():
    # notes/branches/04-pdl-splitk-gemms.md 11.5: the router's 64x8/k4 is 96 CTAs at
    # 48 rows; wk keeps about 24 CTAs (12 at <= 8 rows) beside the indexer's wq_b
    assert plan.splitk_ctas((64, 8, 4, 12), 48, 256) == 96
    assert plan.splitk_ctas((64, 16, 4, 10), 96, 256) == 96
    assert plan.splitk_ctas((64, 8, 4, 12), 8, 160) == 12
    assert plan.splitk_ctas((64, 32, 4, 8), 64, 160) == 24
    # one wave: SMs - (split_k - 1) * 8 GPCs
    assert plan.splitk_fits_one_wave((64, 8, 4, 12), 60, 256, GB300_SMS)  # 128 CTAs
    assert not plan.splitk_fits_one_wave((64, 8, 4, 12), 72, 256, GB300_SMS)


def test_wk_weights_proj_split_k_at_1_to_64_rows():
    e = plan.entry("wk_weights_proj", (160, 6144))
    assert {m for m in range(1, 97) if plan.range_at(e, m) is not None} == set(
        range(1, 65)
    )
    assert all(plan.range_at(e, m).kernel == plan.SPLITK for m in range(1, 65))


def test_router_split_k_from_17_rows():
    e = plan.entry("router", (256, 6144))
    assert plan.range_at(e, 16) is None
    assert plan.kernel_name(plan.range_at(e, 17)) == "64x8/k4/s12"
    assert plan.kernel_name(plan.range_at(e, 48)) == "64x8/k4/s12"
    assert plan.kernel_name(plan.range_at(e, 49)) == "64x16/k4/s10"
    assert plan.range_at(e, 97) is None


def test_m_ranges():
    assert plan.m_ranges((5, 1, 2, 3, 9, 10)) == "1-3,5,9-10"
    assert plan.m_ranges(()) == ""


def test_tolerances_are_named_per_kernel_family():
    assert set(plan.TOLERANCES) == {"ll.gemm.bf16.rel", "ll.gemm.router.rel"}
    assert all(0 < v < 1e-3 for v in plan.TOLERANCES.values())


# ------------------------------------------------------------- engagement rules


@pytest.mark.parametrize(
    "sm, sms",
    [((10, 3), 152), ((10, 0), 148), ((10, 3), 160), ((10, 7), 152), ((12, 0), 152)],
)
def test_plan_engages_only_on_sm103_with_152_sms(sm, sms):
    e, why = gemm._planned("o_proj", _module((6144, 4096)), sm=sm, sms=sms)
    if (sm, sms) == plan.MEASURED_DEVICE:
        assert e is not None and why == ""
    else:
        assert e is None and "SM103 with 152 SMs" in why


@pytest.mark.parametrize(
    "module, reason",
    [
        (_module((6144, 4096), dtype=torch.float8_e4m3fn), "not 2-D BF16"),
        (_module((6144, 4096), bias=torch.zeros(6144)), "bias"),
        (_module((6144, 1024)), "no measured plan"),
        (_module((6144, 4096), contiguous=False), "strided"),
        (_module((6144, 4096), ptr=16), "aligned"),
    ],
)
def test_declines(module, reason):
    e, why = gemm._planned("o_proj", module, sm=SM103, sms=GB300_SMS)
    assert e is None and reason in why


def test_build_declines_off_device_without_touching_the_gpu():
    line = gemm.build(
        "o_proj", _module((6144, 4096)), range(1, 49), sm=(10, 0), sms=148
    )
    assert "stock" in line and "SM103" in line
    assert gemm.served_rows("o_proj", 6144, 4096) == {}
    assert "o_proj (N=6144, K=4096)" in gemm.describe()


def test_units_switch(monkeypatch):
    o_proj8, o_proj = _module((6144, 2048)), _module((6144, 4096))
    router = _module((256, 6144))

    def planned(role, module):
        return gemm._planned(role, module, sm=SM103, sms=GB300_SMS)[0] is not None

    # unset: the preferred entries; the deferred TP8 o_proj only when its unit is listed
    assert planned("o_proj", o_proj) and not planned("o_proj", o_proj8)
    set_slots(monkeypatch, ["ll.gemm.o_proj"])
    assert planned("o_proj", o_proj8) and not planned("router", router)
    # leave-one-out
    set_slots(monkeypatch, without=["ll.gemm.o_proj"])
    assert not planned("o_proj", o_proj) and planned("router", router)
    assert not planned("o_proj", o_proj8)
    # parity: every role on SGLang's GEMM
    set_slots(monkeypatch, [])
    assert not planned("o_proj", o_proj) and not planned("router", router)


def test_gemm_units_are_the_roles():
    assert ll.GEMM_ROLES == plan.ROLES
    assert units.GEMM_UNITS == tuple(gemm.unit(r) for r in plan.ROLES)
    assert set(units.GEMM_UNITS) <= set(units.KNOWN)


def test_shared_stock_body_and_tolerances():
    assert gemm.stock is ll.stock_linear
    assert gemm.TOLERANCES == plan.TOLERANCES
    assert ll.module("gemm") is gemm
    assert plan.TOLERANCES.items() <= ll.tolerances().items()


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="unknown role"):
        gemm._planned("v_b", _module((6144, 4096)), sm=SM103, sms=GB300_SMS)


def test_router_keeps_the_tiny_gemm_rows():
    assert (
        gemm._min_rows("router", SimpleNamespace(tiny_router_gemm_max_tokens=16)) == 17
    )
    assert (
        gemm._min_rows("router", SimpleNamespace(tiny_router_gemm_max_tokens=-1)) == 1
    )
    assert gemm._min_rows("o_proj", SimpleNamespace()) == 1


def test_out_dtype_casts_like_ll_linear():
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    linear = SimpleNamespace(weight=torch.randn(4, 8, dtype=torch.bfloat16), bias=None)
    y = gemm.linear("eh_proj", linear, x, out_dtype=torch.float32)
    assert y.dtype == torch.float32
    assert torch.equal(y, ll.linear("eh_proj", linear, x, out_dtype=torch.float32))
    with pytest.raises(ValueError, match="unknown role"):
        gemm.linear("v_b", linear, x)


# ---------------------------------------------------------- split-K source switch


def _fake_flashinfer(monkeypatch, *, with_tactic: bool, sync_fixed: bool = False):
    fi, fi_gemm = ModuleType("flashinfer"), ModuleType("flashinfer.gemm")

    if with_tactic:

        def mm_bf16(
            a,
            b,
            bias=None,
            pdl=False,
            out=None,
            out_dtype=None,
            backend="cudnn",
            *,
            tactic=None,
        ):
            return out

        attrs = {gemm.FI1_SYNC_MARKER: True} if sync_fixed else {}
        fi_gemm.SplitKTactic = type("SplitKTactic", (), attrs)
    else:

        def mm_bf16(
            a, b, bias=None, pdl=False, out=None, out_dtype=None, backend="cudnn"
        ):
            return out

    fi_gemm.mm_bf16 = mm_bf16
    fi.gemm = fi_gemm
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", fi_gemm)
    import sgl_nvidia_mega.glm5_3.sm103 as cute_gate

    monkeypatch.setattr(cute_gate, "cutlass_supported", lambda: (True, ""))


def test_fi1_with_the_sync_fixes_is_the_default(monkeypatch):
    _fake_flashinfer(monkeypatch, with_tactic=True, sync_fixed=True)
    assert gemm.flashinfer_tactic_api() is sys.modules["flashinfer.gemm"]
    assert gemm.splitk_source().name == "flashinfer"


def test_fi1_without_the_sync_fixes_defaults_to_the_vendored_copy(monkeypatch):
    pytest.importorskip("cutlass")
    _fake_flashinfer(monkeypatch, with_tactic=True)
    assert gemm.flashinfer_tactic_api() is not None
    assert gemm.splitk_source().name == "vendored"
    monkeypatch.setenv(gemm.SPLITK_ENV, "flashinfer")  # tests compare the two
    assert gemm.splitk_source().name == "flashinfer"


def test_stock_flashinfer_falls_back_to_the_vendored_copy(monkeypatch):
    pytest.importorskip("cutlass")
    _fake_flashinfer(monkeypatch, with_tactic=False)
    assert gemm.flashinfer_tactic_api() is None
    assert gemm.splitk_source().name == "vendored"
    monkeypatch.setenv(gemm.SPLITK_ENV, "flashinfer")
    with pytest.raises(RuntimeError, match="no mm_bf16"):
        gemm.splitk_source()


def test_switch_forces_the_vendored_copy(monkeypatch):
    pytest.importorskip("cutlass")
    _fake_flashinfer(monkeypatch, with_tactic=True, sync_fixed=True)
    monkeypatch.setenv(gemm.SPLITK_ENV, "vendored")
    assert gemm.splitk_source().name == "vendored"
    monkeypatch.setenv(gemm.SPLITK_ENV, "bogus")
    with pytest.raises(ValueError, match="bogus"):
        gemm.splitk_source()


def test_unvalidated_cutlass_raises(monkeypatch):
    import sgl_nvidia_mega.glm5_3.sm103 as cute_gate

    monkeypatch.setattr(
        cute_gate, "cutlass_supported", lambda: (False, "cutlass-dsl 9.9 not validated")
    )
    with pytest.raises(RuntimeError, match="not validated"):
        gemm.splitk_source()


# ------------------------------------------------------------------ stock path


class _QuantMethod:
    def __init__(self):
        self.calls = []

    def apply(self, layer, x, bias=None):
        self.calls.append(bias)
        return x @ layer.weight.T + 1


def test_stock_runs_the_modules_own_gemm():
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    w = torch.randn(4, 8, dtype=torch.bfloat16)
    qm = _QuantMethod()
    linear = SimpleNamespace(weight=w, bias=None, quant_method=qm)
    assert torch.equal(gemm.stock("o_proj", linear, x), x @ w.T + 1) and qm.calls == [
        None
    ]
    plain = SimpleNamespace(weight=w, bias=None)
    assert torch.equal(
        gemm.stock("eh_proj", plain, x), torch.nn.functional.linear(x, w)
    )

    class Gate(torch.nn.Module):  # MoEGate.forward: FP32 logits
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(w, requires_grad=False)

        def forward(self, h):
            return h.float() @ self.weight.float().T

    assert gemm.stock("router", Gate(), x).dtype == torch.float32
    assert torch.equal(gemm.stock("lm_head", plain, x), torch.matmul(x, w.T))


def test_unplanned_rows_run_stock(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gemm, "_device_facts", lambda index: plan.MEASURED_DEVICE)
    w = torch.randn(6144, 3072, dtype=torch.bfloat16)
    module = SimpleNamespace(weight=w, bias=None, quant_method=_QuantMethod())
    x = torch.randn(49, 3072, dtype=torch.bfloat16)  # dense_down's TGV range ends at 48
    assert torch.equal(gemm.linear("dense_down", module, x), x @ w.T + 1)


def test_planned_but_unbuilt_row_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gemm, "_device_facts", lambda index: plan.MEASURED_DEVICE)
    module = SimpleNamespace(
        weight=torch.zeros(6144, 3072, dtype=torch.bfloat16),
        bias=None,
        quant_method=_QuantMethod(),
    )
    with pytest.raises(RuntimeError, match="not built"):
        gemm.linear("dense_down", module, torch.zeros(6, 3072, dtype=torch.bfloat16))
    # with the role off (or the plan declining), the same call is stock
    set_slots(monkeypatch, [])
    assert gemm.linear(
        "dense_down", module, torch.zeros(6, 3072, dtype=torch.bfloat16)
    ).shape == (6, 6144)


def test_declined_build_keeps_the_role_stock(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(gemm, "_device_facts", lambda index: plan.MEASURED_DEVICE)
    module = SimpleNamespace(
        weight=torch.zeros(6144, 3072, dtype=torch.bfloat16),
        bias=torch.zeros(6144, dtype=torch.bfloat16),
        quant_method=_QuantMethod(),
    )
    assert "bias" in gemm.build("dense_down", module, [6], sm=SM103, sms=GB300_SMS)
    assert gemm.linear(
        "dense_down", module, torch.zeros(6, 3072, dtype=torch.bfloat16)
    ).shape == (6, 6144)


class _Processor:
    """SGLang's LogitsProcessor stand-in: its _compute_lm_head marks the stock path."""

    use_fp32_lm_head = False
    rl_on_policy_target = None

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None, **kwargs):
        return ("stock", kwargs)


class _MixedProcessor(gemm.LmHeadMixin, _Processor):
    pass


def test_lm_head_mixin_serves_only_built_rows(monkeypatch):
    w = torch.randn(38720, 64, dtype=torch.bfloat16)
    lm_head = SimpleNamespace(weight=w, quant_method=None)
    key = ("lm_head", 38720, 64)
    served = gemm._Served(plan.TGV, None, plan.TGV)
    monkeypatch.setitem(gemm._READY, key, {6: served})
    monkeypatch.setattr(
        gemm, "_run", lambda s, x, weight, dtype: ("ll", s.kernel, tuple(x.shape))
    )
    proc = _MixedProcessor()
    x6 = torch.randn(6, 64, dtype=torch.bfloat16)
    assert proc._compute_lm_head(x6, lm_head) == ("ll", plan.TGV, (6, 64))
    # a row count build did not prepare, a bias, FP32 logits, a strided input: stock
    assert (
        proc._compute_lm_head(torch.randn(5, 64, dtype=torch.bfloat16), lm_head)[0]
        == "stock"
    )
    assert proc._compute_lm_head(x6, lm_head, torch.zeros(38720))[0] == "stock"
    assert proc._compute_lm_head(x6, lm_head, out=None) == ("ll", plan.TGV, (6, 64))
    strided = torch.randn(64, 6, dtype=torch.bfloat16).T
    assert proc._compute_lm_head(strided, lm_head)[0] == "stock"
    proc.use_fp32_lm_head = True
    assert proc._compute_lm_head(x6, lm_head)[0] == "stock"
    proc.use_fp32_lm_head = False
    lora = SimpleNamespace(weight=w, quant_method=None, set_lora=1, apply_lora=1)
    assert proc._compute_lm_head(x6, lora)[0] == "stock"
