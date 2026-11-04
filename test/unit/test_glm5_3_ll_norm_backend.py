# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.norm_backend`` on CPU: FlashInfer's norm backend rebuilt from its public
surface, the probe's inputs and comparison, and the fold decision (fold only on CuTe
DSL with a bitwise probe; a failing probe or a CUDA backend keeps the stock norm, and
says why)."""

import sys
import types

import pytest
import torch

from sgl_nvidia_mega.glm5_3.ll import norm_backend as nb

DEV = torch.device("cpu")


def _rmsnorm(x, w, eps=1e-5):
    xf = x.float()
    return (
        xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    ).bfloat16()


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """A stand-in for ``flashinfer.norm`` / ``flashinfer.cute_dsl`` whose answers the
    test sets."""
    state = {"available": True, "arch": True}
    fi = types.ModuleType("flashinfer")
    norm = types.ModuleType("flashinfer.norm")
    cute = types.ModuleType("flashinfer.cute_dsl")
    cute.is_cute_dsl_available = lambda: state["available"]

    def arch(major, minor):
        if isinstance(state["arch"], Exception):
            raise state["arch"]
        return state["arch"]

    cute.is_cute_dsl_arch_supported = arch
    fi.norm, fi.cute_dsl = norm, cute
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setitem(sys.modules, "flashinfer.norm", norm)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl", cute)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (10, 3)
    )
    monkeypatch.delenv(nb.FLASHINFER_CUDA_NORM_ENV, raising=False)
    return state


def test_backend_cute_dsl(fake_flashinfer):
    assert nb.fi_norm_backend(DEV) == nb.CUTE_DSL


@pytest.mark.parametrize(
    "value,want", [("1", nb.CUDA), ("0", nb.CUTE_DSL), ("", nb.CUTE_DSL)]
)
def test_backend_env(fake_flashinfer, monkeypatch, value, want):
    monkeypatch.setenv(nb.FLASHINFER_CUDA_NORM_ENV, value)
    assert nb.fi_norm_backend(DEV) == want


def test_backend_without_cute_dsl(fake_flashinfer):
    fake_flashinfer["available"] = False
    assert nb.fi_norm_backend(DEV) == nb.CUDA


def test_backend_arch(fake_flashinfer):
    fake_flashinfer["arch"] = False
    assert nb.fi_norm_backend(DEV) == nb.CUDA
    # FlashInfer counts a failing capability probe as supported
    fake_flashinfer["arch"] = RuntimeError("probe failed")
    assert nb.fi_norm_backend(DEV) == nb.CUTE_DSL


def test_probe_inputs_layout():
    xs = list(
        nb.probe_inputs((1, 5, 48), 512, row_stride=2624, col_offset=2048, device=DEV)
    )
    assert [x.shape for x in xs] == [(1, 512), (5, 512), (48, 512)]
    for x in xs:
        assert x.dtype == torch.bfloat16 and x.stride() == (2624, 1)
        assert x.storage_offset() == 2048
    x = xs[2]
    assert bool((x[1] == 0).all())
    assert float(x[2].float().abs().max()) < 1e-15
    # order-sensitive rows: magnitudes spread over many binades
    big = x[:24].float().abs().amax(dim=0)
    assert float(big.max() / big[big > 0].min()) > 2**15


def test_probe_bits():
    w = torch.rand(512).bfloat16() + 0.5
    same = lambda x: _rmsnorm(x, w)  # noqa: E731
    ok, detail = nb.probe_bits(
        same, same, (1, 6, 24), 512, row_stride=2624, col_offset=2048, device=DEV
    )
    assert ok and "bitwise" in detail

    def one_ulp_off(x):
        y = _rmsnorm(x, w).view(torch.int16).clone()
        y[-1, 7] += 1
        return y.view(torch.bfloat16)

    ok, detail = nb.probe_bits(
        same, one_ulp_off, (1, 6), 512, row_stride=2624, col_offset=2048, device=DEV
    )
    assert not ok and "1 of" in detail and "1 rows" in detail


def _decide(stock, ours, backend):
    return nb.decide(
        stock,
        ours,
        (5, 6, 24),
        512,
        row_stride=2624,
        col_offset=2048,
        device=DEV,
        backend=lambda device: backend,
    )


def test_decide():
    w = torch.rand(512).bfloat16() + 0.5
    norm = lambda x: _rmsnorm(x, w)  # noqa: E731
    d = _decide(norm, norm, nb.CUTE_DSL)
    assert d.fold and "cute-dsl" in d.reason and "[5, 6, 24]" in d.reason
    # a CUDA backend never folds, even if the probe would match
    calls = []
    d = _decide(lambda x: calls.append(1) or norm(x), norm, nb.CUDA)
    assert not d.fold and "cuda" in d.reason and not calls
    # a probe mismatch keeps the stock norm and says where
    d = _decide(norm, lambda x: _rmsnorm(x, w, eps=1e-3), nb.CUTE_DSL)
    assert not d.fold and "probe against the stock norm" in d.reason


def test_decide_raises_on_probe_error():
    def broken(x):
        raise RuntimeError("kernel build failed")

    with pytest.raises(RuntimeError, match="kernel build failed"):
        _decide(broken, broken, nb.CUTE_DSL)
