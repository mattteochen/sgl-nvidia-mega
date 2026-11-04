# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The draft_eh step's prepared input (``ll.draft_eh``, P6a), on CPU.

The P6a sum in the target's LL hop also applies the draft layer's input norm and hands the
layer a ``draft_attention.PreparedInput``, so the layer skips its attention prepare. Only a
draft attention step group takes that input (it sets the attention inputs itself); the stock
``self_attn`` reads the attention inputs that the prepare sets. So under the stock draft
attention (``-ll.attention``, parity, a width no attention unit serves, a step-reasons
decline) the step returns the plain sum, and the layer refuses a prepared input without a
step group."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.glm5_3 import draft_attention
from sgl_nvidia_mega.glm5_3 import draft_units as draft
from sgl_nvidia_mega.glm5_3.ll import draft as ll_draft

H = ll_draft.HIDDEN


class _Hop:
    """The target's LL hop family: records the residual-free sum + norm calls."""

    def __init__(self):
        self.calls = []

    def allreduce_norm(self, partial, residual, gamma, *, site):
        self.calls.append((partial.shape[0], residual, site))
        return partial + 1, partial


class _RowParallel:
    """``RowParallelEhProj``: the plain sum of the partials (here the partial itself)."""

    def __init__(self):
        self.calls = 0

    def partial(self, eh_input):
        return eh_input[:, :H].clone()

    def __call__(self, eh_input):
        self.calls += 1
        return self.partial(eh_input)


def _family(hop_rows=(5, 10, 20)):
    family = ll_draft.RowParallelEhFamily(model=None, derived=None, profile=None)
    family.rp = _RowParallel()
    family.hop = _Hop()
    family.hop_rows = frozenset(hop_rows)
    family.gamma = torch.ones(H, dtype=torch.bfloat16)
    return family


def _eh_input(rows):
    return torch.randn(rows, 2 * H).to(torch.bfloat16)


@pytest.mark.parametrize("rows", [5, 10, 20])
def test_the_stock_draft_attention_gets_the_plain_sum(rows):
    family = _family()
    x = _eh_input(rows)
    out = draft.eh_sum(family, x, attention=None)
    assert isinstance(out, torch.Tensor) and not isinstance(
        out, draft_attention.PreparedInput
    )
    assert torch.equal(out, x[:, :H])
    assert family.rp.calls == 1 and family.hop.calls == []


@pytest.mark.parametrize("rows", [5, 10, 20])
def test_a_draft_attention_step_group_gets_the_prepared_input(rows):
    family = _family()
    x = _eh_input(rows)
    out = draft.eh_sum(family, x, attention=object())
    assert isinstance(out, draft_attention.PreparedInput)
    assert torch.equal(out.residual, x[:, :H]) and torch.equal(out.norm, x[:, :H] + 1)
    assert (
        family.hop.calls == [(rows, None, ll_draft.SITE_EH_SUM)]
        and family.rp.calls == 0
    )


def test_rows_the_hop_does_not_serve_get_the_plain_sum():
    family = _family(hop_rows=(5,))
    out = draft.eh_sum(family, _eh_input(15), attention=object())
    assert isinstance(out, torch.Tensor) and not isinstance(
        out, draft_attention.PreparedInput
    )
    assert family.rp.calls == 1 and family.hop.calls == []


def test_without_the_target_hops_the_sum_is_plain():
    family = _family()
    family.hop = None
    out = draft.eh_sum(family, _eh_input(5), attention=object())
    assert not isinstance(out, draft_attention.PreparedInput) and family.rp.calls == 1


def test_a_strided_input_stays_stock():
    family = _family()
    x = _eh_input(5)[:, ::2]
    assert draft.eh_sum(family, x, attention=None) is None
    assert draft.eh_sum(family, x, attention=object()) is None


def test_the_stock_eh_step_takes_the_switch():
    x = _eh_input(5)
    assert draft.eh_sum(draft.DraftEh(), x, attention=None) is None
    assert draft.eh_sum(draft.DraftEh(), x, attention=object()) is None


def test_the_layer_refuses_a_prepared_input_without_a_step_group():
    layer = SimpleNamespace()  # never read: the refusal comes first
    with pytest.raises(RuntimeError, match="needs a draft attention step group"):
        draft_attention.decoder_layer_forward(
            layer,
            None,
            positions=torch.zeros(5, dtype=torch.int64),
            hidden_states=torch.zeros(5, H, dtype=torch.bfloat16),
            forward_batch=None,
            zero_allocator=None,
            prepared=True,
        )
