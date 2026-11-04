# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.attention``'s fused-A forms (``csrc/gemm/fused_a_ll.cuh``) on one SM100-family
GPU (GB300), at the TP4 rank's shapes: the fused q_a / kv_a projection (2624 x 6144)
and q_b (4096 x 2048).

- The 1-16-row form, every option (pipelined MMA loop, deep stage ring), bit for bit
  SGLang's ``dsv3_fused_a_gemm`` JIT kernel, which stock runs there.
- The production forms of ``plan_forms`` at every TP4 width 1-48 against what stock
  runs: bitwise up to 16 rows; the split-K form at 17-24 rows within the named
  tolerance ``ll.attention.qkv_a_wide.max_rel`` of fp32, deterministic, writing no
  row past M (SGLang's own GEMM there is printed, ``FUSED_A_REL`` lines with ``-s``).
- The split-K form fits one wave on the device (the family's occupancy gate).
- Each production form captured in a CUDA graph behind a PDL producer that releases
  it at entry and writes its input late: every replay equals eager.

Weights: synthetic, or GLM-5.2 layer 5 when ``SGL_NVIDIA_MEGA_GLM_CKPT`` names the
checkpoint.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from sgl_nvidia_mega.glm5_3.ll import attention as lla

CKPT_ENV = "SGL_NVIDIA_MEGA_GLM_CKPT"
CKPT_LAYER = 5
SHAPES = {"fused_a": lla.QKV_A_SHAPE, "q_b": lla.Q_B_SHAPE}
TOL = lla.TOLERANCES["ll.attention.qkv_a_wide.max_rel"]


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import sglang.kernels.ops.gemm.dsv3_fused_a_gemm  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _ready(), reason="needs an SM100-family GPU and SGLang"
)


@pytest.fixture(scope="module")
def weights():
    """{"fused_a": [2624, 6144], "q_b": [4096, 2048]} bf16 on the GPU."""
    root = os.environ.get(CKPT_ENV)
    if root:
        from safetensors import safe_open

        index = json.loads((Path(root) / "model.safetensors.index.json").read_text())
        wmap = index["weight_map"]
        p = f"model.layers.{CKPT_LAYER}.self_attn."
        got = {}
        for key in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj"):
            name = p + key + ".weight"
            with safe_open(str(Path(root) / wmap[name]), framework="pt") as f:
                got[key] = f.get_tensor(name)
        fused = torch.cat([got["q_a_proj"], got["kv_a_proj_with_mqa"]], dim=0)
        q_b = got["q_b_proj"][: lla.Q_B_SHAPE[0]]  # rank 0's heads
        print(f"FUSED_A weights: GLM-5.2 layer {CKPT_LAYER}")
        return {"fused_a": fused.cuda().contiguous(), "q_b": q_b.cuda().contiguous()}
    gen = torch.Generator(device="cuda").manual_seed(1234)
    return {
        name: (torch.randn(*shape, device="cuda", generator=gen) * 0.02).bfloat16()
        for name, shape in SHAPES.items()
    }


def _x(m, k, gen):
    return torch.randn(m, k, device="cuda", generator=gen).bfloat16()


def _stock_ll(x, w):
    """What stock runs at 1-16 rows: SGLang's JIT fused-A kernel."""
    from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import dsv3_fused_a_gemm

    return dsv3_fused_a_gemm(x, w.T)


def _stock_wide(x, w):
    """What the layer's own GEMM runs above 16 rows (``ReplicatedLinear``, bf16):
    SGLang's CuTe DSL TGV GEMM where it is chosen, else cuBLAS."""
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
        cutedsl_bf16_gemm,
        use_cutedsl_bf16_gemm,
    )

    if use_cutedsl_bf16_gemm(x.shape[0], w.shape[0], w.shape[1]):
        return cutedsl_bf16_gemm(x, w)
    return torch.nn.functional.linear(x, w)


def _rel(out, exact):
    return (
        (out.float() - exact).abs().max() / exact.abs().max().clamp_min(1e-6)
    ).item()


def _production(x, w, name, rows):
    """``LLAttention``'s dispatch for ``name`` at ``rows`` (None: stock body)."""
    forms = lla.plan_forms(rows)
    form = forms.qkv_a if name == "fused_a" else forms.q_b
    if form is None:
        return None
    if form[0] == "ll":
        deep = form[2] if name == "fused_a" else False
        return lla.fused_a_ll(x, w, ilp=form[1], deep=deep)
    return lla.fused_a_wide(x, w, form[1])


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("ilp", [False, True])
@pytest.mark.parametrize("deep", [False, True])
def test_ll_bitwise(weights, name, ilp, deep):
    """Every option of the 1-16-row form at M 1-16: SGLang's kernel, bit for bit."""
    w = weights[name]
    gen = torch.Generator(device="cuda").manual_seed(7)
    for m in range(1, 17):
        x = _x(m, w.shape[1], gen)
        ref = _stock_ll(x, w)
        out = lla.fused_a_ll(x, w, ilp=ilp, deep=deep)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), (name, m, ilp, deep)


@pytest.mark.parametrize("name", list(SHAPES))
def test_production_forms(weights, name):
    """Every TP4 width 1-48: bitwise where the form is the 1-16-row one, within the
    named tolerance on the split-K form, the stock body elsewhere."""
    w = weights[name]
    gen = torch.Generator(device="cuda").manual_seed(11)
    served = {"bitwise": [], "tolerance": [], "stock": []}
    for m in range(1, lla.MAX_ROWS + 1):
        x = _x(m, w.shape[1], gen)
        out = _production(x, w, name, m)
        if out is None:
            served["stock"].append(m)
            continue
        if m <= lla.LL_MAX_ROWS:
            assert torch.equal(out, _stock_ll(x, w)), (name, m)
            served["bitwise"].append(m)
        else:
            exact = x.float() @ w.float().T
            err, stock_err = _rel(out, exact), _rel(_stock_wide(x, w), exact)
            print(
                f"FUSED_A_REL {name} rows={m} split_k={err:.3e} stock={stock_err:.3e}"
            )
            assert err <= TOL, (name, m, err)
            served["tolerance"].append(m)
    print(f"FUSED_A_FORMS {name} {served}")
    if name == "fused_a":
        assert served["tolerance"] == list(range(17, lla.QKV_A_WIDE_MAX_ROWS + 1))
    else:
        assert served["bitwise"] == list(range(*lla.Q_B_LL_ROWS)) + [lla.Q_B_LL_ROWS[1]]


def test_wide(weights):
    """The split-K form at M 1-48: within the tolerance of fp32, deterministic, no
    store past row M (a guard band below the output stays intact)."""
    w = weights["fused_a"]
    n, k = w.shape
    gen = torch.Generator(device="cuda").manual_seed(13)
    worst = 0.0
    for m in range(1, lla.WIDE_MAX_ROWS + 1):
        x = _x(m, k, gen)
        exact = x.float() @ w.float().T
        guard = torch.full((m + 8, n), -7.0, device="cuda", dtype=torch.bfloat16)
        lla.fused_a_wide(x, w, out=guard[:m])
        again = lla.fused_a_wide(x, w)
        torch.cuda.synchronize()
        out = guard[:m]
        err = _rel(out, exact)
        worst = max(worst, err)
        assert err <= TOL, (m, err)
        assert torch.equal(out, again), m
        assert bool((guard[m:] == -7.0).all()), m
        assert bool(torch.isfinite(out.float()).all()), m
    print(f"FUSED_A_REL_WORST split_k {worst:.3e}")


def test_wide_fits_one_wave():
    """The family's gate: one wave of 64 x 3 clusters (41 for 2624 outputs) fits."""
    wave = lla.QKV_A_SHAPE[0] // lla.QKV_A_WIDE[0]
    for m in range(17, lla.QKV_A_WIDE_MAX_ROWS + 1):
        active = lla.fused_a_wide_max_active_clusters(lla.QKV_A_SHAPE, m)
        print(f"FUSED_A_OCCUPANCY rows={m} clusters={active} wave={wave}")
        assert active >= wave, (m, active)


_PRODUCER_SRC = r"""
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <cstdint>

namespace sglang::glm_ll_test {

// Releases its PDL dependents at entry, spins window_ns, then writes dst: a
// dependent that read dst before its own griddepcontrol.wait sees the old bytes.
__global__ void pdl_producer_kernel(const uint16_t* src, uint16_t* dst, int64_t n, uint64_t window_ns) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint64_t start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < window_ns);
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

struct PdlProducer {
  static void run(const tvm::ffi::TensorView src, const tvm::ffi::TensorView dst, int64_t window_ns) {
    using namespace host;
    auto n = SymbolicSize{"numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({n}).with_dtype<bf16_t>().with_device(device).verify(src);
    TensorMatcher({n}).with_dtype<bf16_t>().with_device(device).verify(dst);
    LaunchKernel(dim3(16), dim3(256), device.unwrap())
        .enable_pdl(true)(
            pdl_producer_kernel,
            static_cast<const uint16_t*>(src.data_ptr()),
            static_cast<uint16_t*>(dst.data_ptr()),
            static_cast<int64_t>(n.unwrap()),
            static_cast<uint64_t>(window_ns));
  }
};

}  // namespace sglang::glm_ll_test
"""


@pytest.fixture(scope="module")
def producer(tmp_path_factory):
    from sglang.kernels.jit.utils import load_jit

    src = tmp_path_factory.mktemp("glm_ll_pdl") / "pdl_producer.cuh"
    src.write_text(_PRODUCER_SRC)
    return load_jit(
        "glm_ll_test_pdl_producer",
        cuda_files=[str(src)],
        cuda_wrappers=[("pdl_producer", "sglang::glm_ll_test::PdlProducer::run")],
    )


@pytest.mark.parametrize(
    "name,rows",
    [("fused_a", m) for m in (5, 6, 10, 12, 16, 18, 24)] + [("q_b", 10), ("q_b", 12)],
)
def test_graph_pdl(weights, producer, name, rows):
    """The production form in a CUDA graph behind a PDL producer that releases it at
    entry and writes its input 2 us later over NaN: 8 replays equal eager."""
    w = weights[name]
    gen = torch.Generator(device="cuda").manual_seed(31)
    src = _x(rows, w.shape[1], gen)
    x = src.clone()
    ref = _production(x, w, name, rows)

    def enqueue():
        x.fill_(float("nan"))
        producer.pdl_producer(src.view(-1), x.view(-1), 2000)
        return _production(x, w, name, rows)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        enqueue()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = enqueue()
    for replay in range(8):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref), (name, rows, replay)
