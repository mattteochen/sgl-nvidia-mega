# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.attention``'s KV-half kernel (``csrc/attention/mla_kv_half.cuh``) on one
SM100-family GPU (GB300), at a TP4 rank's 16 heads, against what SGLang's DSA TRT-LLM
FP8 path runs: ``sgl_kernel.rmsnorm`` (FlashInfer's CuTe DSL norm) on the strided kv_a
latent, ``mla_quantize_and_rope_for_fp8`` (FlashInfer ``mla_rope_quantize_fp8``) and
``set_mla_kv_buffer_triton`` (SGLang's JIT KV append). All bitwise:

- the FP8 query and every byte of the MLA cache (written rows, the skipped padding
  slot 0, untouched rows), with the norm folded and without it, at the TP4 widths
  1-48, int32 / int64 positions and slots, outlier rows, and both entry-trigger forms;
- the folded norm alone (the probe's instance) against the stock norm on 8192 real
  (or synthetic), order-sensitive and tiny rows;
- ``norm_backend.decide`` on this device: CuTe DSL and the probe bitwise (the fold
  engages), and ``FLASHINFER_USE_CUDA_NORM=1`` in a child process (it does not);
- a CUDA graph of the kernel, replayed on fresh inputs.

Weights: synthetic, or the kv_a norm / projection of GLM-5.2 layers 3, 40, 77 when
``SGL_NVIDIA_MEGA_GLM_CKPT`` names the checkpoint.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from sgl_nvidia_mega.glm5_3.ll import attention as lla
from sgl_nvidia_mega.glm5_3.ll import norm_backend

CKPT_ENV = "SGL_NVIDIA_MEGA_GLM_CKPT"
CKPT_LAYERS = (3, 40, 77)
HEADS = lla.HEADS
KV_LORA, ROPE, Q_LORA, NOPE, HIDDEN = (
    lla.KV_LORA,
    lla.ROPE,
    lla.Q_LORA,
    lla.NOPE,
    lla.HIDDEN,
)
ROW = lla.MQA_DIM
FUSED_A_N = lla.QKV_A_SHAPE[0]
ROWS = (1, 2, 3, 5, 6, 8, 10, 12, 16, 18, 24, 30, 36, 48)
SLOTS = 4096


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer.rope  # noqa: F401
        import sgl_kernel  # noqa: F401
        from sglang.kernels.ops.attention.utils import (  # noqa: F401
            mla_quantize_and_rope_for_fp8,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _ready(), reason="needs an SM100-family GPU and SGLang"
)


@pytest.fixture(scope="module")
def model():
    """(eps, rope_theta, {layer: {"kv_norm", "kv_a", "in_norm"}})."""
    root = os.environ.get(CKPT_ENV)
    if not root:
        return 1e-5, 8000000.0, {None: {}}
    from safetensors import safe_open

    cfg = json.loads((Path(root) / "config.json").read_text())
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    rp = cfg.get("rope_parameters") or {}
    theta = float(rp.get("rope_theta", cfg.get("rope_theta", 8000000.0)))
    wmap = json.loads((Path(root) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    layers = {}
    for layer in CKPT_LAYERS:
        names = {
            "kv_norm": f"model.layers.{layer}.self_attn.kv_a_layernorm.weight",
            "kv_a": f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight",
            "in_norm": f"model.layers.{layer}.input_layernorm.weight",
        }
        got = {}
        for key, name in names.items():
            with safe_open(str(Path(root) / wmap[name]), framework="pt") as f:
                got[key] = f.get_tensor(name).bfloat16().cuda()
        layers[layer] = got
    print(f"KV_HALF weights: GLM-5.2 layers {sorted(layers)}, eps {eps}")
    return eps, theta, layers


@pytest.fixture(scope="module")
def cos_sin(model):
    # RotaryEmbedding._compute_cos_sin_cache, GPT-J interleaved, fp32
    theta = model[1]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, ROPE, 2, dtype=torch.float, device="cuda") / ROPE)
    )
    t = torch.arange(0, 1 << 16, dtype=torch.float, device="cuda")
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


class Case:
    """One forward's per-rank inputs, laid out as the verify step produces them."""

    def __init__(self, rows, gen, weights=None, outliers=False):
        self.rows = rows
        fused = torch.randn(rows, FUSED_A_N, generator=gen, device="cuda") * 0.5
        if weights and "kv_a" in weights:
            hid = torch.randn(rows, HIDDEN, generator=gen, device="cuda")
            hid = hid * torch.rsqrt(hid.pow(2).mean(-1, keepdim=True) + 1e-6)
            hid = hid * weights["in_norm"].float()
            fused[:, Q_LORA:] = hid.bfloat16().float() @ weights["kv_a"].float().t()
        if outliers and rows >= 4:
            fused[0, Q_LORA : Q_LORA + KV_LORA] *= 300.0
            fused[1, Q_LORA : Q_LORA + KV_LORA] *= 1e-4
            fused[2, Q_LORA : Q_LORA + KV_LORA] = 0.0
            fused[3, Q_LORA + 5] = 1e4
        self.fused = fused.bfloat16()  # [rows, 2624]: q_lora | raw kv latent | raw k_pe
        self.k_nope_raw = self.fused[:, Q_LORA : Q_LORA + KV_LORA]
        self.k_pe = self.fused[:, Q_LORA + KV_LORA :]
        qb = (
            torch.randn(rows, HEADS * (NOPE + ROPE), generator=gen, device="cuda") * 2
        ).bfloat16()
        self.q_pe = qb.view(rows, HEADS, NOPE + ROPE)[..., NOPE:]
        qn = (
            torch.randn(HEADS, rows, KV_LORA, generator=gen, device="cuda") * 3
        ).bfloat16()
        self.q_nope = qn.transpose(0, 1)  # the absorb bmm's [rows, heads, 512] view
        if weights and "kv_norm" in weights:
            self.norm_w = weights["kv_norm"].contiguous()
        else:
            self.norm_w = (
                1.0 + 0.2 * torch.randn(KV_LORA, generator=gen, device="cuda")
            ).bfloat16()


def _locs(rows, gen, dtype, pad_rows=0):
    perm = torch.randperm(SLOTS - 1, generator=gen, device="cuda")[:rows] + 1
    if pad_rows:
        perm[-pad_rows:] = 0  # graph padding rows land on the reserved slot 0
    return perm.to(dtype)


def _stock_norm(x, w, eps):
    from sgl_kernel import rmsnorm

    return rmsnorm(x, w, eps)


def _stock(case, k_norm, positions, cache, locs, kv_u8):
    """The stock rope_cache body: FlashInfer RoPE + FP8, then the pool's KV append."""
    from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8
    from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton

    q, k_nope, k_rope = mla_quantize_and_rope_for_fp8(
        case.q_nope,
        case.q_pe,
        k_norm,
        case.k_pe,
        positions,
        cache,
        False,
        KV_LORA,
        ROPE,
    )
    set_mla_kv_buffer_triton(
        kv_u8, locs, k_nope.view(torch.uint8), k_rope.view(torch.uint8)
    )
    return q


def _bytes_equal(a, b):
    a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    return int((a8 != b8).sum())


@pytest.mark.parametrize("rows", ROWS)
def test_equivalence(model, cos_sin, rows):
    """q bytes and the whole cache, norm folded and not, both entry-trigger forms."""
    eps, _, layers = model
    gen = torch.Generator(device="cuda").manual_seed(1234 + rows)
    for li, (layer, weights) in enumerate(layers.items()):
        for dtype in (torch.int64, torch.int32):
            case = Case(rows, gen, weights, outliers=(li == 0))
            positions = torch.randint(
                0, cos_sin.shape[0], (rows,), generator=gen, device="cuda"
            ).to(dtype)
            locs = _locs(rows, gen, dtype, pad_rows=1 if rows >= 3 else 0)
            kv0 = torch.randint(
                0, 256, (SLOTS, 1, ROW), generator=gen, device="cuda", dtype=torch.int32
            ).to(torch.uint8)
            kv_ref = kv0.clone()
            k_norm = _stock_norm(case.k_nope_raw, case.norm_w, eps)
            q_ref = _stock(case, k_norm, positions, cos_sin, locs, kv_ref)
            for fold in (True, False):
                for entry in (False, True):
                    kv = kv0.clone()
                    q = lla.mla_kv_half(
                        case.q_nope,
                        case.q_pe,
                        case.k_nope_raw if fold else k_norm,
                        case.k_pe,
                        cos_sin,
                        positions,
                        locs,
                        kv.view(torch.float8_e4m3fn),  # as get_key_buffer returns it
                        norm_weight=case.norm_w if fold else None,
                        norm_eps=eps,
                        trigger_at_entry=entry,
                    )
                    torch.cuda.synchronize()
                    tag = (rows, layer, str(dtype), fold, entry)
                    assert q.is_contiguous() and q.shape == (rows, HEADS, ROW), tag
                    assert _bytes_equal(q, q_ref) == 0, tag
                    assert _bytes_equal(kv, kv_ref) == 0, tag


def _adversarial(rows, gen):
    """Rows whose per-lane sums of squares depend strongly on the association order."""
    x = torch.randn(rows, KV_LORA, generator=gen, device="cuda")
    lane_cols = torch.arange(32, device="cuda") * 8
    big = torch.exp2(
        torch.randint(3, 9, (rows, 1), generator=gen, device="cuda").float()
    )
    for col in (0, 3, 7):
        x[:, lane_cols + col] *= big
    x[:, 256 + lane_cols] *= big.sqrt()
    return x


def test_norm_reproduction(model):
    """The folded norm's bf16 output (``kv_half_norm``) against the stock norm and
    FlashInfer's public ``rmsnorm`` on 8192 strided rows per set."""
    import flashinfer.norm

    eps, _, layers = model
    gen = torch.Generator(device="cuda").manual_seed(77)
    rows = 8192
    sets = []
    for layer, weights in layers.items():
        sets.append((f"real-{layer}", Case(rows, gen, weights)))
    first = next(iter(layers.values()))
    adv = Case(rows, gen, first)
    adv.k_nope_raw.copy_(_adversarial(rows, gen).bfloat16())
    sets.append(("adversarial", adv))
    tiny = Case(rows, gen, first)
    tiny.k_nope_raw.copy_((tiny.k_nope_raw.float() * 1e-21).bfloat16())
    sets.append(("tiny", tiny))
    for name, c in sets:
        ref = _stock_norm(c.k_nope_raw, c.norm_w, eps)
        fi = flashinfer.norm.rmsnorm(c.k_nope_raw, c.norm_w, eps)
        out = lla.kv_half_norm(c.k_nope_raw, c.norm_w, eps)
        diff = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
        print(f"KV_HALF_NORM {name}: {diff} of {rows * KV_LORA} bf16 values differ")
        assert torch.equal(ref.view(torch.int16), fi.view(torch.int16)), name
        assert diff == 0, name


def test_fold_decision_here(model):
    """On this device the fold engages: CuTe DSL norms, and the probe is bitwise at
    the TP4 widths."""
    eps, _, layers = model
    w = next(iter(layers.values())).get("kv_norm")
    if w is None:
        w = (1.0 + 0.2 * torch.randn(KV_LORA, device="cuda")).bfloat16()
    decision = norm_backend.decide(
        lambda x: _stock_norm(x, w, eps),
        lambda x: lla.kv_half_norm(x, w, eps),
        ROWS,
        KV_LORA,
        row_stride=FUSED_A_N,
        col_offset=Q_LORA,
        device=torch.device("cuda"),
    )
    print(f"KV_HALF_FOLD {decision}")
    assert norm_backend.fi_norm_backend(torch.device("cuda")) == norm_backend.CUTE_DSL
    assert decision.fold, decision.reason


def test_fold_declines_on_cuda_norm():
    """``FLASHINFER_USE_CUDA_NORM=1`` (read at FlashInfer's import, so in a child
    process): the backend is CUDA and the fold does not engage."""
    code = textwrap.dedent(
        """
        import torch
        from sgl_nvidia_mega.glm5_3.ll import norm_backend
        dev = torch.device("cuda")
        assert norm_backend.fi_norm_backend(dev) == "cuda"
        d = norm_backend.decide(lambda x: x, lambda x: x, (1,), 512,
                                row_stride=2624, col_offset=2048, device=dev)
        assert not d.fold and "cuda" in d.reason, d
        print("OK", d.reason)
        """
    )
    env = dict(os.environ, FLASHINFER_USE_CUDA_NORM="1")
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stdout + res.stderr
    print("KV_HALF_CUDA_NORM", res.stdout.strip())


@pytest.mark.parametrize("rows", (6, 24))
def test_graph_replay(model, cos_sin, rows):
    """A CUDA graph of the folded kernel, replayed on 20 fresh inputs: the stock
    path's bytes each time."""
    eps, _, layers = model
    weights = next(iter(layers.values()))
    gen = torch.Generator(device="cuda").manual_seed(5)
    case = Case(rows, gen, weights)
    positions = torch.zeros(rows, dtype=torch.int64, device="cuda")
    locs = torch.zeros(rows, dtype=torch.int64, device="cuda")
    kv = torch.zeros(SLOTS, 1, ROW, dtype=torch.uint8, device="cuda")
    q_static = torch.empty(rows, HEADS, ROW, dtype=torch.float8_e4m3fn, device="cuda")

    def run():
        lla.mla_kv_half(
            case.q_nope,
            case.q_pe,
            case.k_nope_raw,
            case.k_pe,
            cos_sin,
            positions,
            locs,
            kv,
            norm_weight=case.norm_w,
            norm_eps=eps,
            q_out=q_static,
        )

    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
        with torch.cuda.graph(g, stream=s):
            run()
    torch.cuda.current_stream().wait_stream(s)
    for it in range(20):
        fresh = Case(rows, gen, weights)
        case.fused.copy_(fresh.fused)
        case.q_pe.copy_(fresh.q_pe)
        case.q_nope.copy_(fresh.q_nope)
        positions.copy_(
            torch.randint(0, cos_sin.shape[0], (rows,), generator=gen, device="cuda")
        )
        locs.copy_(_locs(rows, gen, torch.int64, pad_rows=1))
        kv_ref = kv.clone()
        g.replay()
        torch.cuda.synchronize()
        q_ref = _stock(
            case,
            _stock_norm(case.k_nope_raw, case.norm_w, eps),
            positions,
            cos_sin,
            locs,
            kv_ref,
        )
        torch.cuda.synchronize()
        assert _bytes_equal(q_static, q_ref) == 0, it
        assert _bytes_equal(kv, kv_ref) == 0, it
