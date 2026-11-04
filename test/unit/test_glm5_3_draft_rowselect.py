# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_rowselect (the NextN draft extend runs its MLP on the select_index rows),
host logic on CPU against the installed SGLang.

- The gate (``draft_step.FusedDraft.extend_rows``) selects only captured draft-extend batches whose
  layer output is read on the ``select_index`` rows alone.
- ``draft_attention.select_rows`` keeps those rows of the MLP input and of the residual
  SGLang's prepare wrote, and refuses any other stream layout.
- The hand-off: the rows SGLang's logits processor feeds the draft's LM head, and the
  hidden state it stores, are the same whether the layer output was row-selected
  (``FusedDraft.run`` hands metadata without ``select_index``) or not (needs CUDA).

Run with ``python -m pytest test/unit/test_glm5_3_draft_rowselect.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.glm5_3 import draft_attention, draft_step
from sgl_nvidia_mega.glm5_3.runtime import DRAFT_SLOTS, SHIPPED_SLOTS, SLOT_NEEDS

H = 64  # any width: the host logic is row-wise


def test_the_row_select_is_a_shipped_slot_on_the_draft():
    assert "draft_rowselect" in SHIPPED_SLOTS and "draft_rowselect" in DRAFT_SLOTS
    assert ("draft_rowselect", "draft") in SLOT_NEEDS


# ------------------------------------------------------------------------ the gate


class _Batch(SimpleNamespace):
    """ForwardBatch stand-in: any field not set reads as None."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None


def _select_index(bs, w, seed=0):
    gen = torch.Generator().manual_seed(seed)
    accept = torch.randint(0, w, (bs,), generator=gen)
    return torch.arange(bs) * w + accept


def _extend_batch(bs=4, w=6, **kw):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    fields = dict(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        return_logprob=False,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_token_non_padded=None,
        spec_info=SimpleNamespace(select_index=_select_index(bs, w)),
        input_ids=torch.zeros(bs * w, dtype=torch.int64),
    )
    fields.update(kw)
    return _Batch(**fields)


def _row_select_draft(on=True):
    """A FusedDraft without its build (no GPU): the gate's state, no attention kernels,
    not rank 0 (no capture log)."""
    fd = draft_step.FusedDraft.__new__(draft_step.FusedDraft)
    fd.row_select = on
    fd._attn_family = None
    fd.rank = 1
    fd._reported = set()
    return fd


def test_extend_rows_selects_the_captured_draft_extend_rows():
    fb = _extend_batch()
    assert _row_select_draft().extend_rows(fb) is fb.spec_info.select_index


def test_extend_rows_declines_everything_else():
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    on = _row_select_draft()
    cases = {
        "slot off / unsupported MLP": (_row_select_draft(False), _extend_batch()),
        "draft decode": (on, _extend_batch(forward_mode=ForwardMode.DECODE)),
        "target verify": (on, _extend_batch(forward_mode=ForwardMode.TARGET_VERIFY)),
        "logprobs": (on, _extend_batch(return_logprob=True)),
        "FULL hidden capture": (
            on,
            _extend_batch(capture_hidden_mode=CaptureHiddenMode.FULL),
        ),
        "padded-row mask": (on, _extend_batch(num_token_non_padded=torch.tensor(20))),
        "eager (no select_index)": (
            on,
            _extend_batch(spec_info=SimpleNamespace(select_index=None)),
        ),
        "nothing to drop": (
            on,
            _extend_batch(input_ids=torch.zeros(4, dtype=torch.int64)),
        ),
    }
    for name, (fd, fb) in cases.items():
        assert fd.extend_rows(fb) is None, name


def test_extend_rows_also_selects_without_hidden_capture():
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    fb = _extend_batch(capture_hidden_mode=CaptureHiddenMode.NULL)
    assert _row_select_draft().extend_rows(fb) is fb.spec_info.select_index


# ------------------------------------------------------------------------ select_rows


def _prepared_batch(num_tokens, residual):
    """The stream state SGLang's FFN prepare leaves: a written residual, nothing pending."""
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    fb = _Batch(
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        residual_stream=ResidualStream(),
    )
    fb.residual_stream.write(residual)
    return fb


@pytest.mark.parametrize("w", [4, 6])
@pytest.mark.parametrize("bs", [1, 2, 4])
def test_select_rows_keeps_the_index_rows_of_the_input_and_the_residual(bs, w):
    n = bs * w
    x = torch.randn(n, H, dtype=torch.bfloat16)
    res = torch.randn(n, H, dtype=torch.bfloat16)
    sel = _select_index(bs, w, seed=bs)
    fb = _prepared_batch(n, res)
    out = draft_attention.select_rows(x, fb, sel)
    assert torch.equal(out, x[sel])
    assert torch.equal(fb.residual_stream.residual, res[sel])
    assert fb.residual_stream.pending is None


def test_select_rows_refuses_layouts_the_gate_does_not_admit():
    from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD

    n = 12
    x = torch.randn(n, H, dtype=torch.bfloat16)
    res = torch.randn(n, H, dtype=torch.bfloat16)
    sel = torch.tensor([5, 11])
    pending = _prepared_batch(n, res)
    pending.residual_stream.record(torch.zeros_like(x), PLAIN_ADD)
    cases = {
        "a pending contribution": (x, pending),
        "rows that are not this rank's tokens": (x, _prepared_batch(n + 3, res)),
        "a quantized MLP input": ((x, torch.ones(n)), _prepared_batch(n, res)),
    }
    for name, (hidden, fb) in cases.items():
        with pytest.raises(RuntimeError):
            draft_attention.select_rows(hidden, fb, sel)
        assert fb.residual_stream.residual is res, name


# ------------------------------------------------------------------------ hand-off


def _rows_read(hidden, meta):
    """What LogitsProcessor.forward feeds the LM head and stores, for this metadata."""
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor

    if not isinstance(meta, LogitsMetadata):
        meta = LogitsMetadata.from_forward_batch(meta)
    pruned, before, aux, sample_idx, _, _ = LogitsProcessor._get_pruned_states(
        None, hidden, None, None, meta
    )
    stored = LogitsProcessor._get_hidden_states_to_store(
        None, hidden, None, None, pruned, before, aux, sample_idx, meta
    )
    return pruned, stored


# SGLang's logits processor initializes the DSA handlers, which need a CUDA driver
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SGLang's logits processor needs CUDA"
)
@pytest.mark.parametrize("w", [4, 6])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("mode", ["LAST", "NULL"])
def test_the_logits_processor_reads_the_same_rows_on_both_paths(
    monkeypatch, bs, w, mode
):
    from sglang.srt.layers.logits_processor import LogitsMetadata
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    n = bs * w
    full = torch.randn(n, H, dtype=torch.bfloat16)
    fb = _extend_batch(bs, w, capture_hidden_mode=getattr(CaptureHiddenMode, mode))
    sel = fb.spec_info.select_index
    ref_pruned, ref_stored = _rows_read(full, fb)
    assert torch.equal(ref_pruned, full[sel])

    for on in (True, False):
        fd = _row_select_draft(on)
        got = {}

        def forward_steps(
            model, ids, pos, batch, attention=None, view=None, ffn_rows=None
        ):
            got["ffn_rows"] = ffn_rows
            return full if ffn_rows is None else full.index_select(0, ffn_rows)

        monkeypatch.setattr(fd, "forward_steps", forward_steps)
        hidden, meta = fd.run(None, fb.input_ids, torch.zeros(n), fb)
        if not on:
            # None: draft_logits passes the forward batch itself
            assert meta is None
            meta = fb
        if on:
            assert got["ffn_rows"] is sel
            assert isinstance(meta, LogitsMetadata)
            # the processor must not select the selected rows again
            assert meta.draft_extend_select_index is None
            assert meta.capture_hidden_mode == fb.capture_hidden_mode
        else:
            assert got["ffn_rows"] is None
        pruned, stored = _rows_read(hidden, meta)
        assert torch.equal(pruned, ref_pruned)
        assert (stored is None and ref_stored is None) or torch.equal(
            stored, ref_stored
        )
