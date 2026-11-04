# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.jit`` on CPU: the SGLang headers the unit kernels include exist in the installed
SGLang with the names they use, and ``load`` refuses what it must."""

import re
from pathlib import Path

import pytest
import sglang

from sgl_nvidia_mega.glm5_3 import ll
from sgl_nvidia_mega.glm5_3.ll import jit

INCLUDE = Path(sglang.__file__).resolve().parent / "kernels" / "jit" / "include"


@pytest.mark.parametrize("header", sorted(jit.SGLANG_HEADERS))
def test_sglang_provides_the_headers(header):
    path = INCLUDE / header
    assert path.is_file(), (
        f"SGLang no longer ships {header}; repair ll.jit.SGLANG_HEADERS"
    )
    text = path.read_text()
    missing = [
        n for n in jit.SGLANG_HEADERS[header] if not re.search(rf"\b{n}\b", text)
    ]
    assert not missing, f"{header} no longer defines {missing}"


def test_load_refuses_a_missing_source():
    with pytest.raises(FileNotFoundError, match="no source"):
        jit.load("nothing", "attention/does_not_exist.cuh", [("f", "f")])


def test_forced_error_hook(monkeypatch):
    monkeypatch.delenv(jit.FORCE_ERROR_ENV, raising=False)
    assert not jit.forced_error("x")
    monkeypatch.setenv(jit.FORCE_ERROR_ENV, "y, x")
    assert jit.forced_error("x") and not jit.forced_error("z")
    # a rank-scoped entry needs torch.distributed (absent here: never this rank)
    monkeypatch.setenv(jit.FORCE_ERROR_ENV, "x@3")
    assert not jit.forced_error("x")


def test_no_unit_module_and_stock_linear():
    import torch

    assert set(ll.modules()) <= set(ll.CANDIDATE_MODULES)
    lin = torch.nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 8)
    if ll.module("gemm") is None:
        out = ll.linear("o_proj", lin, x, out_dtype=torch.float64)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out, (x @ lin.weight.T).double())
        assert ll.build_linear("o_proj", lin, (4, 8)) == ""
