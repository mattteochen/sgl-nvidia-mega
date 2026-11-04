# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The composed NextN draft of the TP4 profiles (``draft_units``), host logic on CPU
against the installed SGLang.

- The draft graphs' row counts pair with the target's verify widths
  (``draft_units.draft_rows``): draft extend at the verify rows, draft decode at one row
  per request.
- The draft's units compose per (mode, rows) from ``units.PREFERENCE``: the draft
  step's fused front, P6a (``ll.draft_eh``) and the row select in the draft extend; the
  rest stock (the ll attention units are left out here). Every draft unit is listed for
  every step it claims.
- The draft_rowselect step group runs the TP8 draft step's gate
  (``draft_step.select_index_rows``).
- The hand-off: the rows SGLang's logits processor feeds the draft's LM head, and the
  hidden state it stores, are the same whether the layer output was row-selected
  (``ComposedDraft.run`` hands metadata without ``select_index``) or not (CPU tensors,
  but SGLang's logits processor needs a CUDA driver: skipped without one).

Run with ``python -m pytest test/unit/test_glm5_3_draft_units.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from glm5_3_slots import set_slots

from sgl_nvidia_mega.glm5_3 import draft as draft_spec
from sgl_nvidia_mega.glm5_3 import draft_units as draft
from sgl_nvidia_mega.glm5_3 import spec as target

H = 64  # any width: the host logic is row-wise


@pytest.mark.parametrize(
    "tokens, widths, rows",
    [
        (6, (6, 12), (1, 2, 6, 12)),
        (5, (5, 10, 15), (1, 2, 3, 5, 10, 15)),
        (
            6,
            tuple(range(6, 49, 6)),
            (1, 2, 3, 4, 5, 6, 7, 8, 12, 18, 24, 30, 36, 42, 48),
        ),
    ],
)
def test_draft_rows_pair_with_the_verify_widths(tokens, widths, rows):
    assert draft.draft_rows(tokens, widths) == rows


def test_draft_rows_refuse_widths_off_the_request_grid():
    with pytest.raises(ValueError):
        draft.draft_rows(6, (6, 8))


def test_draft_steps_pair_every_verify_width_with_both_draft_graphs():
    assert draft.draft_steps(6, (6, 12)) == (
        ("decode", 1),
        ("decode", 2),
        ("draft_extend", 6),
        ("draft_extend", 12),
    )


def test_the_draft_spec_covers_the_targets_verify_widths():
    from sgl_nvidia_mega.glm5_3 import runtime

    if "draft" in runtime.SLOTS:
        assert draft_spec.SPEC.max_rows == target.SPEC.max_rows == 48


# ------------------------------------------------------------------------ units per draft graph


def _profile(tp, tokens, widths):
    from sgl_nvidia_mega.glm5_3.profile import Profile

    return Profile(
        tp=tp,
        heads=64 // tp,
        tokens_per_request=tokens,
        widths=widths,
        sm=(10, 3),
        sms=152,
    )


def _composition(profile, mode, rows, monkeypatch, slots=None):
    from sgl_nvidia_mega.glm5_3 import registry, units

    set_slots(monkeypatch, slots)
    candidates, _ = registry.collect(draft.draft_modules())
    # the ll attention units are left out: these compositions are the draft's own
    candidates = [
        c for c in candidates if c.name not in ("ll.sparse_mla_fd", "ll.attention")
    ]
    comp = registry.select(
        profile,
        rows,
        mode,
        candidates=candidates,
        preference=units.PREFERENCE,
        groups=registry.DRAFT_GROUPS,
        allowed=units.allowed,
    )
    owners = {}
    for (g, _), c in comp.owners.items():
        owners.setdefault(g, set()).add(c.name)
    return {g: "+".join(sorted(n)) for g, n in owners.items()}, comp


def test_every_draft_unit_is_listed_for_its_steps():
    from sgl_nvidia_mega.glm5_3 import registry, units

    candidates, families = registry.collect(draft.draft_modules())
    names = {c.name for c in candidates}
    assert {"draft.front", "draft_rowselect", "ll.draft_eh"} <= names
    for c in candidates:
        if c.group in registry.DRAFT_GROUPS:
            for step in c.steps:
                assert c.name in [p.name for p in units.PREFERENCE[(c.group, step)]], (
                    c,
                    step,
                )
    assert {"draft.front", "draft.rowselect", "ll.draft_eh"} <= set(families)


@pytest.mark.parametrize("tokens", [5, 6])
def test_tp4_draft_graphs_take_p6a_and_the_row_select(tokens, monkeypatch):
    widths = tuple(range(tokens, 49, tokens))
    p = _profile(4, tokens, widths)
    for mode, rows in draft.draft_steps(tokens, widths):
        owners, _ = _composition(p, mode, rows, monkeypatch)
        want = {"draft_front": "draft.front", "draft_eh": "ll.draft_eh"}
        if mode == "draft_extend":
            want["draft_mlp"] = "draft_rowselect"
        # the draft top-1 stays SGLang's at TP4 (E-D2)
        assert owners == want, (mode, rows)


def test_the_slots_turn_draft_units_off(monkeypatch):
    p = _profile(4, 6, (6, 12))
    owners, _ = _composition(
        p, "draft_extend", 6, monkeypatch, slots=["draft", "ll.draft_eh"]
    )
    assert owners == {"draft_front": "draft.front", "draft_eh": "ll.draft_eh"}
    owners, _ = _composition(p, "draft_extend", 6, monkeypatch, slots=[])
    assert owners == {}


def test_the_row_select_group_runs_the_shared_gate():
    from sgl_nvidia_mega.glm5_3 import draft_step

    assert draft.select_index_rows is draft_step.select_index_rows


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


def _draft_for(mode, rows, mlp):
    """A ComposedDraft whose built composition at (mode, rows) holds only ``mlp`` (the rest stock)."""
    fd = draft.ComposedDraft()
    groups = {g: factory() for g, factory in draft.DRAFT_STOCK.items()}
    groups["draft_mlp"] = mlp
    fd.build = SimpleNamespace(
        groups={(mode, rows): groups},
        compositions={(mode, rows): SimpleNamespace(owners={})},
    )
    return fd


# SGLang's logits processor initializes the DSA handlers, which need a CUDA driver
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SGLang's logits processor needs CUDA"
)
@pytest.mark.parametrize("w", [5, 6])
@pytest.mark.parametrize("bs", [1, 2, 8])
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
        fd = _draft_for(
            "draft_extend", n, draft.RowSelect() if on else draft.DraftMlp()
        )
        got = {}

        def forward_steps(
            model,
            ids,
            pos,
            batch,
            attention=None,
            ffn_rows=None,
            groups=None,
            view=None,
        ):
            got["ffn_rows"], got["attention"] = ffn_rows, attention
            return full if ffn_rows is None else full.index_select(0, ffn_rows)

        monkeypatch.setattr(fd, "forward_steps", forward_steps)
        hidden, meta = fd.run(None, fb.input_ids, torch.zeros(n), fb)
        # no unit claims a draft_attention step here: the stock attention
        assert got["attention"] is None
        if on:
            assert got["ffn_rows"] is sel
            assert isinstance(meta, LogitsMetadata)
            # the processor must not select the selected rows again
            assert meta.draft_extend_select_index is None
            assert meta.capture_hidden_mode == fb.capture_hidden_mode
        else:
            assert got["ffn_rows"] is None and meta is fb
        pruned, stored = _rows_read(hidden, meta)
        assert torch.equal(pruned, ref_pruned)
        assert (stored is None and ref_stored is None) or torch.equal(
            stored, ref_stored
        )
