# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.attention`` against the stock ``attention.step.AttentionStep`` on one TP4 layer
(one SM100-family GPU).

The layer is a GLM-5.2 DSA attention at a TP4 rank's shapes whose stock code is
SGLang's: ``DeepseekV2AttentionMLA.prepare_qkv_latent`` / ``q_b_proj_forward``
(SGLang's JIT fused-A kernel up to 16 rows), the mixin's ``_split_q_nope_pe``, real
``RMSNorm`` modules (FlashInfer's CuTe DSL norm) and a real ``MLATokenToKVPool`` write
door. Both providers run ``qkv_a`` -> ``prologue`` -> ``q_absorb`` -> ``rope_cache``
and the indexer's key / head-gate projection from the same inputs and cache bytes:

- with the fold (the family's own decision on this GPU) and without it, with the
  capture-mode side stream on and off: the q-LoRA (the indexer input), the FP8 query,
  every cache byte and the key / head-gate projection bitwise at 5, 6, 10, 12, 36 and
  48 rows; at 24 rows ``qkv_a`` within ``ll.attention.qkv_a_wide.max_rel`` of fp32 and
  every later step bitwise from the same latent;
- both providers captured in CUDA graphs and replayed on fresh inputs: equal bytes;
- the family's checks (16 heads only), its log line naming which way the fold went,
  and ``step_reasons`` on the pool.
"""

import logging
import types

import pytest
import torch
import torch.nn.functional as F

from sgl_nvidia_mega.glm5_3 import ll
from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep as AttentionDecode
from sgl_nvidia_mega.glm5_3.attention.step import StepContext
from sgl_nvidia_mega.glm5_3.ll import attention as lla
from sgl_nvidia_mega.glm5_3.ll import norm_backend

HEADS, HIDDEN, Q_LORA, KV_LORA, ROPE, NOPE = (
    lla.HEADS,
    lla.HIDDEN,
    lla.Q_LORA,
    lla.KV_LORA,
    lla.ROPE,
    lla.NOPE,
)
ROW = lla.MQA_DIM
SLOTS, PAGE = 4096, 64
WIDTHS = (5, 6, 10, 12, 24, 36, 48)
TOL = lla.TOLERANCES["ll.attention.qkv_a_wide.max_rel"]
EPS = 1e-5


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import sglang.srt.models.deepseek_v2  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _ready(), reason="needs an SM100-family GPU and SGLang"
)


class _Linear(torch.nn.Module):
    """A bf16 ``ReplicatedLinear`` stand-in: ``(x @ W^T, None)``."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight), None


def _rmsnorm(width, gen):
    from sglang.srt.layers.layernorm import RMSNorm

    norm = RMSNorm(width, eps=EPS)
    w = (1.0 + 0.2 * torch.randn(width, generator=gen, device="cuda")).bfloat16()
    norm.weight = torch.nn.Parameter(w, requires_grad=False)
    return norm


def _attn_class():
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
        DeepseekMLAForwardMixin,
    )
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

    class Attn:
        """The ``DeepseekV2AttentionMLA`` state the steps read; SGLang's own methods."""

        prepare_qkv_latent = DeepseekV2AttentionMLA.prepare_qkv_latent
        q_b_proj_forward = DeepseekV2AttentionMLA.q_b_proj_forward
        _split_q_nope_pe = DeepseekMLAForwardMixin._split_q_nope_pe

        def __init__(self, gen, heads=HEADS):
            dev = "cuda"
            self.layer_id = 0
            self.num_local_heads = heads
            self.q_lora_rank, self.kv_lora_rank = Q_LORA, KV_LORA
            self.qk_nope_head_dim, self.qk_rope_head_dim = NOPE, ROPE
            self.qk_head_dim = NOPE + ROPE
            self.v_head_dim = 256
            self.has_fused_proj, self.is_packed_weight = True, False
            self.fused_a_gemm_backend = "auto"
            # resolved as SGLang resolves them for this layer (bf16, verified q_b
            # shape, deterministic inference off)
            self._use_min_latency_fused_a_gemm = True
            self._use_min_latency_q_b_gemm = True
            self._q_b_proj_verified_shape = True
            self.alt_stream = torch.cuda.Stream()

            def w(*shape, scale):
                return (
                    torch.randn(*shape, generator=gen, device=dev) * scale
                ).bfloat16()

            self.fused_qkv_a_proj_with_mqa = _Linear(w(*lla.QKV_A_SHAPE, scale=0.02))
            self.q_b_proj = _Linear(w(heads * (NOPE + ROPE), Q_LORA, scale=0.02))
            self.q_a_layernorm = _rmsnorm(Q_LORA, gen)
            self.kv_a_layernorm = _rmsnorm(KV_LORA, gen)
            # deepseek_weight_loader keeps W_UK as a [H, 192, 512] view of [H, 512, 192]
            self.w_kc = w(heads, KV_LORA, NOPE, scale=0.05).transpose(1, 2)
            theta = 8000000.0
            inv = 1.0 / (
                theta
                ** (torch.arange(0, ROPE, 2, dtype=torch.float, device=dev) / ROPE)
            )
            freqs = torch.einsum(
                "i,j -> ij", torch.arange(1 << 16, dtype=torch.float, device=dev), inv
            )
            self.rotary_emb = types.SimpleNamespace(
                is_neox_style=False,
                cos_sin_cache=torch.cat((freqs.cos(), freqs.sin()), -1).contiguous(),
            )
            self.attn_mqa = types.SimpleNamespace(
                layer_id=0, tp_k_head_num=1, v_head_dim=KV_LORA
            )
            self.indexer = types.SimpleNamespace(
                wk_weights_proj=_Linear(w(128 + 32, HIDDEN, scale=0.02)),
                head_dim=128,
                n_heads=32,
            )

    return Attn


def _pool():
    """A real ``MLATokenToKVPool`` write door over one layer of raw FP8 rows."""
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    pool = object.__new__(MLATokenToKVPool)
    pool.size, pool.page_size, pool.start_layer = SLOTS - PAGE, PAGE, 0
    pool.dtype, pool.store_dtype = torch.float8_e4m3fn, torch.uint8
    pool.use_dsa, pool.dsa_kv_cache_store_fp8 = True, False
    pool.kv_lora_rank, pool.qk_rope_head_dim, pool.kv_cache_dim = KV_LORA, ROPE, ROW
    pool.dcp_replicated = True  # DCP 1: the same scatter, without the parallel state
    pool.layer_transfer_counter = None
    pool.kv_buffer = [torch.zeros(SLOTS, 1, ROW, dtype=torch.uint8, device="cuda")]
    return pool


class _Ctx:
    """``get_attn_tp_context()`` for one DSA layer: ``fetch_qkv_latent`` is
    ``prepare_qkv_latent`` of the current input (no attention-TP gather for DSA)."""

    attn = x = fb = None

    def fetch_qkv_latent(self):
        return self.attn.prepare_qkv_latent(self.x, self.fb)


@pytest.fixture(scope="module")
def env():
    """(Attn, pool, ctx, family) with SGLang's getters patched for the module."""
    import sglang.srt.layers.layer_boundary as lb
    import sglang.srt.model_executor.forward_context as fc
    import sglang.srt.model_executor.runner as runner
    import sglang.srt.model_executor.runner_utils.capture_mode as cm

    gen = torch.Generator(device="cuda").manual_seed(2026)
    attn = _attn_class()(gen)
    pool = _pool()
    ctx = _Ctx()
    ctx.attn = attn
    backend = types.SimpleNamespace(
        token_to_kv_pool=pool, kv_lora_rank=KV_LORA, qk_rope_head_dim=ROPE
    )
    # the running forward's ForwardView (Step.view) as the steps read it
    VIEW["view"] = _view(pool)
    capture = {"on": False}
    mp = pytest.MonkeyPatch()
    mp.setattr(fc, "get_attn_backend", lambda: backend)
    mp.setattr(lb, "get_attn_tp_context", lambda: ctx)
    mp.setattr(runner, "get_is_capture_mode", lambda: capture["on"])
    mp.setattr(cm, "get_is_capture_mode", lambda: capture["on"])
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)])
    )
    family = lla.AttentionFamily(model)
    family.compile(WIDTHS)
    # the other fold form, for the comparisons without it
    lla._kv_half_module(not family.fold.fold, False, lla._pdl(), False)
    yield attn, pool, ctx, family, capture
    mp.undo()


VIEW = {}


def _view(pool):
    """SGLang's ForwardView of a DSA forward on ``pool``, the members the steps read."""
    return types.SimpleNamespace(
        kv_pool=types.SimpleNamespace(
            pool=pool,
            get_key_buffer=pool.get_key_buffer,
            set_mla_kv_buffer=pool.set_mla_kv_buffer,
        ),
        dsa=types.SimpleNamespace(kv_lora_rank=KV_LORA, qk_rope_head_dim=ROPE),
    )


def _batch(rows, gen):
    loc = torch.randperm(SLOTS - 1, generator=gen, device="cuda")[:rows] + 1
    if rows >= 3:
        loc[-1] = 0  # a graph padding row on the reserved slot
    fb = types.SimpleNamespace(
        positions=torch.randint(0, 1 << 16, (rows,), generator=gen, device="cuda"),
        out_cache_loc=loc,
    )
    x = torch.randn(rows, HIDDEN, generator=gen, device="cuda")
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16()
    return x, fb


def _run(provider, attn, pool, ctx, x, fb, kv0, qkv=None):
    """The claimed steps of one provider from cache ``kv0``: (qkv, q_lora, q_fp8,
    cache, key / head-gate projection)."""
    ctx.x, ctx.fb = x, fb
    step = StepContext(
        forward_batch=fb,
        positions=fb.positions,
        zero_allocator=None,
        view=VIEW["view"],
    )
    pool.kv_buffer[0].copy_(kv0)
    if qkv is None:
        qkv = provider.qkv_a(attn, x, step)
    pro = provider.prologue(attn, qkv, step)
    kw = torch.cat(provider._index_kw(attn.indexer, x, step), dim=-1)
    q_nope = provider.q_absorb(attn, pro.q_nope, step)
    q = provider.rope_cache(attn, q_nope, pro.q_pe, pro.k_nope, pro.k_pe, step)
    return qkv, pro.q_lora, q, pool.kv_buffer[0].clone(), kw


def _provider(family, rows, fold):
    """The width's provider with the fold forced on or off (the family's own is
    ``family.provider``)."""
    saved = family.fold
    family.fold = norm_backend.NormFold(fold, "test")
    try:
        return lla.LLAttention(family, rows)
    finally:
        family.fold = saved


def _same_kw():
    """Whether the stock key / head-gate projection is ``ll.linear``'s bit for bit:
    true while no ``ll.gemm`` is present (its own test holds the registered GEMM)."""
    return ll.module("gemm") is None


def _assert_same(tag, stock, ours, kw=True):
    names = ("qkv", "q_lora", "q_fp8", "cache", "index_kw")[: 5 if kw else 4]
    for name, a, b in zip(names, stock, ours):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), (tag, name)
    assert (
        ours[2].shape == (stock[0].shape[0], HEADS, ROW) and ours[2].is_contiguous()
    ), tag


def test_family_on_this_gpu(env, caplog):
    attn, pool, ctx, family, _ = env
    print(
        f"LL_ATTENTION fold {family.fold}, forms {family.forms}, occupancy {family.wide_occupancy}"
    )
    assert family.fold.fold, family.fold.reason
    assert family.forms[24].qkv_a[0] == "wide" and family.forms[36].qkv_a is None
    assert family.forms[10].q_b is not None and family.forms[6].q_b is None
    with caplog.at_level(logging.INFO, logger=lla.__name__):
        family._log(WIDTHS)
    assert "kv_a RMSNorm folded into the KV half" in caplog.text


@pytest.mark.parametrize("rows", WIDTHS)
@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("capture_mode", [False, True])
def test_provider_matches_stock(env, rows, fold, capture_mode):
    attn, pool, ctx, family, capture = env
    gen = torch.Generator(device="cuda").manual_seed(rows * 4 + fold * 2 + capture_mode)
    x, fb = _batch(rows, gen)
    kv0 = torch.randint(
        0, 256, (SLOTS, 1, ROW), generator=gen, device="cuda", dtype=torch.int32
    ).to(torch.uint8)
    ours_p = _provider(family, rows, fold)
    capture["on"] = capture_mode
    try:
        stock = _run(AttentionDecode(), attn, pool, ctx, x, fb, kv0)
        ours = _run(ours_p, attn, pool, ctx, x, fb, kv0)
        wide = (
            family.forms[rows].qkv_a is not None
            and family.forms[rows].qkv_a[0] == "wide"
        )
        if wide:
            exact = x.float() @ attn.fused_qkv_a_proj_with_mqa.weight.float().T
            err = ((ours[0].float() - exact).abs().max() / exact.abs().max()).item()
            print(f"LL_ATTENTION_REL rows={rows} qkv_a={err:.3e}")
            assert err <= TOL, (rows, err)
            # every later step bitwise from the same latent
            ours = _run(ours_p, attn, pool, ctx, x, fb, kv0, qkv=stock[0])
        torch.cuda.synchronize()
    finally:
        capture["on"] = False
    _assert_same((rows, fold, capture_mode), stock, ours, kw=_same_kw())
    # the projection goes through ll.linear: the registered GEMM where ll.gemm serves
    # wk|wp (within its named tolerance of the module's GEMM), else the module's GEMM
    kw = ll.linear("wk_weights_proj", attn.indexer.wk_weights_proj, x)
    assert torch.equal(ours[4], kw), (rows, "ll.linear")
    a, b = stock[4].float(), ours[4].float()
    rel = ((a - b).norm() / a.norm()).item()
    print(f"LL_ATTENTION_KW_REL rows={rows} {rel:.2e}")
    assert rel <= ll.tolerances().get("ll.gemm.bf16.rel", 0.0), (rows, rel)
    # the cache changed exactly in the written rows
    written = fb.out_cache_loc[fb.out_cache_loc != 0]
    changed = (stock[3] != kv0).flatten(1).any(-1).nonzero().flatten()
    assert set(changed.tolist()) <= set(written.tolist())


@pytest.mark.parametrize("rows", (6, 12, 24))
def test_graph_replay(env, rows):
    """Both providers captured (capture mode on, as the graph runner records them),
    replayed on 6 fresh inputs: the same bytes as each other and as eager stock."""
    attn, pool, ctx, family, capture = env
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    x, fb = _batch(rows, gen)
    kv0 = torch.zeros(SLOTS, 1, ROW, dtype=torch.uint8, device="cuda")
    ours_p = family.provider(None, rows, "target_verify")
    outs = {}
    graphs = {}
    capture["on"] = True
    try:
        for name, p in (("stock", AttentionDecode()), ("ours", ours_p)):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _run(p, attn, pool, ctx, x, fb, kv0)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs[name] = _run(p, attn, pool, ctx, x, fb, kv0)
            graphs[name] = g
        for it in range(6):
            x2, fb2 = _batch(rows, gen)
            x.copy_(x2)
            fb.positions.copy_(fb2.positions)
            fb.out_cache_loc.copy_(fb2.out_cache_loc)
            kv0.copy_(
                torch.randint(
                    0, 256, kv0.shape, generator=gen, device="cuda", dtype=torch.int32
                ).to(torch.uint8)
            )
            got = {}
            for name in ("stock", "ours"):
                graphs[name].replay()
                torch.cuda.synchronize()
                got[name] = tuple(t.clone() for t in outs[name])
            eager_stock = _run(AttentionDecode(), attn, pool, ctx, x, fb, kv0)
            eager_ours = _run(ours_p, attn, pool, ctx, x, fb, kv0)
            torch.cuda.synchronize()
            _assert_same((rows, it, "graph stock"), eager_stock, got["stock"])
            _assert_same((rows, it, "graph ours"), eager_ours, got["ours"])
            if family.forms[rows].qkv_a[0] != "wide":
                _assert_same(
                    (rows, it, "ours vs stock"), eager_stock, got["ours"], kw=_same_kw()
                )
    finally:
        capture["on"] = False


def test_family_refuses_tp8_heads():
    gen = torch.Generator(device="cuda").manual_seed(3)
    attn = _attn_class()(gen, heads=8)
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)])
    )
    with pytest.raises(ValueError, match="8 heads per rank"):
        lla.AttentionFamily(model)


def test_step_reasons(env):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    attn, pool, ctx, family, _ = env
    assert family.step_reasons(None, None, VIEW["view"]) == []

    class OwnDoor(MLATokenToKVPool):
        def set_mla_kv_buffer(self, *a, **k):
            raise AssertionError

    saved = pool.__class__
    pool.__class__ = OwnDoor
    try:
        reasons = family.step_reasons(None, None, VIEW["view"])
    finally:
        pool.__class__ = saved
    assert reasons and "OwnDoor" in reasons[0]
