# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The covered verify step's logits processor (``runtime.composed.verify_logits_processor``), on CPU.

It runs the LM head on ``ll.gemm``'s lm_head role and must share the model processor's logits all-gather: SGLang
builds the gatherer's multimem state lazily on an eager call and never under CUDA-graph capture, and the covered
verify step only runs under capture, so a gatherer of its own would bake NCCL's all-gather into the verify graphs."""

from types import SimpleNamespace

import pytest
from sglang.srt.layers import logits_processor as lp

from sgl_nvidia_mega.glm5_3.runtime import composed as glm


class _Processor:
    """SGLang's LogitsProcessor stand-in: a fresh gatherer per instance, as SGLang builds one in ``__init__``."""

    def __init__(self, config):
        self.config = config
        self._logits_gatherer = object()


class _Mixin:
    pass


def _model():
    config = SimpleNamespace(name="glm")
    return SimpleNamespace(
        logits_processor=_Processor(config), lm_head=SimpleNamespace(), config=config
    )


@pytest.fixture
def stubs(monkeypatch):
    built = {"rows": True}
    gemm = SimpleNamespace(
        LmHeadMixin=_Mixin, built=lambda role, module, rows: built["rows"]
    )
    monkeypatch.setattr(lp, "LogitsProcessor", _Processor)
    monkeypatch.setattr(glm.ll, "module", lambda name: gemm if name == "gemm" else None)
    monkeypatch.setattr(glm.ll, "build_linear", lambda *a, **k: "")
    return built


PROFILE = SimpleNamespace(sm=(10, 3), sms=152)


def test_the_verify_processor_shares_the_models_gatherer(stubs):
    model = _model()
    proc = glm.verify_logits_processor(model, (6, 12, 24, 48), PROFILE)
    assert proc is not model.logits_processor
    assert isinstance(proc, _Mixin) and isinstance(proc, _Processor)
    assert proc._logits_gatherer is model.logits_processor._logits_gatherer
    assert proc.config is model.config


def test_no_lm_head_rows_keep_the_models_processor(stubs):
    stubs["rows"] = False
    model = _model()
    assert (
        glm.verify_logits_processor(model, (6, 12), PROFILE) is model.logits_processor
    )


def test_no_gemm_module_or_widths_keep_the_models_processor(stubs, monkeypatch):
    model = _model()
    assert glm.verify_logits_processor(model, (), PROFILE) is model.logits_processor
    monkeypatch.setattr(glm.ll, "module", lambda name: None)
    assert glm.verify_logits_processor(model, (6,), PROFILE) is model.logits_processor
