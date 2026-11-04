# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 / GLM-5.2 spec through SGLang's own argument-resolution checks
(``kernel_plugins/nvidia/binding.megakernel_disable_reasons``, which
``handle_nvidia_megakernel`` runs), on CPU with the toolchain, binding and draft
capture probes patched: the TP8 envelope (one node or two trays) and the TP4
kernel-unit profiles are accepted; every refused configuration names its reason."""

from types import SimpleNamespace

import pytest

guards = pytest.importorskip("sglang.srt.kernel_plugins.nvidia.binding")
from sglang.kernels.plugin_kernels.nvidia import spec as contract  # noqa: E402

from sgl_nvidia_mega.glm5_3 import HF_CONFIG  # noqa: E402
from sgl_nvidia_mega.glm5_3 import runtime  # noqa: E402
from sgl_nvidia_mega.glm5_3 import spec as glm  # noqa: E402

ARCH = "GlmMoeDsaForCausalLM"

# the published GLM-5.2 config fields the specs pin (glm5_3.HF_CONFIG)
HF = SimpleNamespace(architectures=[ARCH], model_type="glm_moe_dsa", **HF_CONFIG)

_FALSE = """enable_dp_attention enable_eplb enable_prefill_cp enable_two_batch_overlap
enable_single_batch_overlap enable_lora enable_return_routed_experts
enable_return_indexer_topk enable_hierarchical_cache
disaggregation_decode_enable_offload_kvcache enable_lmcache enable_flexkv
enable_hisparse enable_memory_saver enable_weights_cpu_backup enable_torch_compile
enable_deterministic_inference rl_on_policy_target""".split()


def cfg(tp=8, tokens=4, steps=None, topk=1, **kw):
    c = dict.fromkeys(_FALSE, False)
    c.update(
        tp_size=tp,
        pp_size=1,
        nnodes=1,
        attn_dp_size=1,
        dp_size=1,
        ep_size=1,
        attn_cp_size=1,
        dcp_size=1,
        lora_paths=None,
        cpu_offload_gb=0,
        offload_group_size=0,
        disaggregation_mode="null",
        watchdog_timeout=300,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
        kv_cache_dtype="fp8_e4m3",
        page_size=64,
        attention_backend="dsa",
        dsa_decode_backend="trtllm",
        moe_runner_backend="flashinfer_trtllm",
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=topk,
        speculative_num_draft_tokens=tokens,
        speculative_num_steps=tokens - 1 if steps is None else steps,
    )
    c.update(kw)
    # the other server arguments the spec pins: an accepted value each
    for key, ok in glm.SERVER_ARGS.items():
        c.setdefault(key, None if None in ok else sorted(ok, key=str)[0])
    return SimpleNamespace(**c)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setitem(contract._REGISTRY, ARCH, "sgl_nvidia_mega.glm5_3.spec:SPEC")
    monkeypatch.setattr(glm, "toolchain_reasons", lambda: [])
    # an SGLang with the DSA binding view and the draft capture check (sglang#46210)
    monkeypatch.setattr(glm, "binding_reasons", lambda: [])
    monkeypatch.setattr(glm, "draft_capture_reasons", lambda: [])
    monkeypatch.setenv("SGLANG_ENABLE_EMBED_REPLICATION", "1")
    monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))


def reasons(c, isa="sm_103"):
    spec, why = guards.megakernel_disable_reasons(cfg=c, hf_config=HF, isa=isa)
    assert spec is None or spec is glm.SPEC
    return why


@pytest.mark.parametrize(
    "c",
    [
        cfg(8, 4),  # TP8, EAGLE 3/1/4 (one B300 node)
        cfg(8, 6, nnodes=2, enable_hierarchical_cache=True),  # tp8-c1 (2 trays)
        cfg(4, 5, enable_hierarchical_cache=True),  # tp4-c1
        cfg(4, 6),  # the decode role's TP4 5/1/6, standalone
        cfg(4, 4),
    ],
    ids=["tp8-3-1-4", "tp8-c1", "tp4-c1", "tp4-5-1-6", "tp4-3-1-4"],
)
def test_accepted(c):
    assert reasons(c) == []


@pytest.mark.parametrize(
    "c,needle",
    [
        (
            cfg(8, 6, enable_dp_attention=True, attn_dp_size=8, ep_size=8),
            "DP attention",
        ),
        (cfg(1, 4, pp_size=4), "pipeline parallelism"),
        (cfg(1, 4), "tp_size 1 is not one of"),
        (cfg(2, 4), "tp_size 2 is not one of"),
        (cfg(8, 4, topk=2), "speculative_eagle_topk 2"),
        (cfg(8, 6, steps=3), "--speculative-num-steps"),
        (cfg(8, 3), "TP8 with 3 tokens per request is not served"),
        (cfg(8, 5), "TP8 with 5 tokens per request is not served"),
        (cfg(4, 6, disaggregation_mode="decode"), "PD disaggregation"),
    ],
    ids=[
        "dep8",
        "pp",
        "tp1",
        "eagle-3-1-4-tp2",
        "topk2",
        "steps",
        "tp8-3-tokens",
        "tp8-5-tokens",
        "pd-decode",
    ],
)
def test_refused(c, needle):
    why = reasons(c)
    assert any(needle in w for w in why), why


def test_unknown_slot_is_refused(monkeypatch):
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS) | {"ll.nope"}
    )
    assert any("unknown slot 'll.nope'" in w for w in reasons(cfg(4, 5)))


def test_a_subset_of_the_slots_is_served(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"ll.hops", "ll.moe"}))
    assert reasons(cfg(4, 5)) == []
    monkeypatch.setattr(runtime, "SLOTS", frozenset())
    assert reasons(cfg(8, 4)) == []


def test_wrong_isa():
    assert reasons(cfg(8, 4), isa="sm_90")


def test_spec_constants():
    assert glm.SPEC.max_rows == 48
    assert glm.REQUIREMENTS.tp_sizes == frozenset({4, 8})
    # the TP8 kernels' tolerances and the TP4 units' (ll.tolerances)
    assert {"attn.sparse_mla.rel_l2", "ll.sparse_mla_fd.max_abs"} <= set(
        glm.NUMERICS.tolerances
    )
    names = {f.name for f in glm.SPEC.features}
    assert {"KV_OFFLOAD", "MULTI_NODE_TP"} <= names
    assert "DP_ATTENTION" not in names
    # every unit is a slot (units.KNOWN; the draft front is the draft slot)
    from sgl_nvidia_mega.glm5_3 import units

    assert {units.slot(n) for n in units.KNOWN} <= {
        *runtime.SHIPPED_SLOTS,
        *runtime.OPT_IN_SLOTS,
    }
    assert not set(runtime.SHIPPED_SLOTS) & set(runtime.OPT_IN_SLOTS)
