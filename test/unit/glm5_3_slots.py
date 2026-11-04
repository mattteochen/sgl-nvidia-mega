# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test helper: set SGLANG_NVIDIA_MEGAKERNEL_SLOTS as glm5_3 reads it (``runtime.SLOTS``
is resolved at import, so tests patch it together with the variable)."""


def set_slots(monkeypatch, names=None, *, without=()):
    """``names``: the listed slots (None: the variable unset, every shipped slot);
    ``without``: every shipped slot but these (a leave-one-out list)."""
    from sgl_nvidia_mega.glm5_3 import runtime

    if names is None and not without:
        monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
        monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))
        return
    if names is None:
        names = [s for s in runtime.SHIPPED_SLOTS if s not in set(without)]
    monkeypatch.setenv(runtime.SLOTS_ENV, ",".join(names))
    monkeypatch.setattr(runtime, "SLOTS", frozenset(names))
