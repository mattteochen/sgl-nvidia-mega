# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.gemm``'s registered small-M BF16 GEMMs on one GB300 GPU (SM103, 152 SMs).

For every preferred role of ``gemm_plan`` (the TP4 o_proj and dense_down, the router,
the indexer's wk | weights_proj, the NextN eh_proj replicated and as the TP4 shard), at
every TP4 row count up to 97:

- the rows the plan serves run the planned kernel (split-K tactic or TGV), the others
  SGLang's GEMM for the module, bit for bit;
- FlashInfer's explicit-tactic split-K (fi-1, ``mm_bf16(..., tactic=)``) and the
  vendored copy return bit for bit the same output (needs fi-1's FlashInfer on the
  path; skipped otherwise);
- against cuBLAS on the same inputs: within ``gemm_plan.TOLERANCES``
  (``ll.gemm.bf16.rel``, ``ll.gemm.router.rel``: ``||y - y_cublas||_F / ||y_cublas||_F``),
  with an error against FP64 no larger than cuBLAS's (x 1.5 + slack), and the router's
  top-8 experts equal to cuBLAS's on >= 95 % of the rows; ``pytest -s`` prints the
  measured maxima as ``TOLERANCE`` lines;
- deterministic, and a captured graph replays bitwise like eager;
- in a captured graph the GEMM is launched with PDL behind a PDL producer, and lets a
  PDL consumer launch behind it (programmatic graph edges);
- a stress run: every split-K problem replayed thousands of times in one graph, each
  replay compared on the GPU with the first result (FlashInfer's kernel and the
  vendored copy with its synchronization fixes).

Weights are random (N(0, 1 / K)), inputs N(0, 1); ``SGL_NVIDIA_MEGA_GLM52_CKPT`` (a
GLM-5.2 checkpoint directory) adds rank-0 slices of the real BF16 weights.
"""

import json
import math
import os

import pytest
import torch
import torch.nn.functional as F

from sgl_nvidia_mega.glm5_3.ll import gemm
from sgl_nvidia_mega.glm5_3.ll import gemm_plan as plan

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("needs an SM100-family GPU", allow_module_level=True)

DEVICE_FACTS = gemm._device_facts(torch.cuda.current_device())
ON_PLAN_DEVICE = DEVICE_FACTS == plan.MEASURED_DEVICE
needs_plan_device = pytest.mark.skipif(
    not ON_PLAN_DEVICE,
    reason=f"the plan is for {plan.MEASURED_DEVICE}, this GPU is {DEVICE_FACTS}",
)
HAS_FI1 = gemm.flashinfer_tactic_api() is not None

# TP4 rows: draft decode 1-8, verify / draft extend 5-6 rows per request at 1-8
# requests (c30 / c40 up to 48 rows), the range edges, and past every range
ROWS = sorted(
    set(range(1, 17))
    | {
        18,
        20,
        24,
        25,
        30,
        32,
        33,
        36,
        40,
        42,
        45,
        47,
        48,
        49,
        54,
        60,
        63,
        64,
        65,
        72,
        80,
        90,
        96,
        97,
    }
)
ENTRIES = [e for e in plan.PLAN if e.preferred]
ROUTER_TINY_ROWS = 16
CKPT_ENV = "SGL_NVIDIA_MEGA_GLM52_CKPT"
STRESS_REPLAYS = int(os.environ.get("GLM_LL_GEMM_STRESS_REPLAYS", "20000"))
# set by the GPU job: the fi-1 vs vendored comparison must run, not skip
REQUIRE_FI1 = os.environ.get("GLM_LL_GEMM_REQUIRE_FI1") == "1"

MEASURED = {}  # tolerance name -> largest value seen


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    from sgl_nvidia_mega.glm5_3 import runtime

    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))
    monkeypatch.delenv(gemm.SPLITK_ENV, raising=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    gemm.reset()
    with torch.no_grad():  # as SGLang's forwards
        yield
    gemm.reset()


def teardown_module(module):
    for name, value in sorted(MEASURED.items()):
        print(
            f"TOLERANCE {name}: measured max {value:.3e} (bound {plan.TOLERANCES.get(name, float('nan')):.1e})"
        )


# ------------------------------------------------------------------ modules


class _CublasMethod:
    """Stand-in for SGLang's UnquantizedLinearMethod at a cuBLAS row count."""

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class _Linear(torch.nn.Module):
    """SGLang linear stand-in (Row/Column/ReplicatedLinear): weight without grad."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.quant_method = _CublasMethod()


def _cublas_fp32(x, w):
    try:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    except TypeError:
        return F.linear(x.float(), w.float())


class _Gate(torch.nn.Module):
    """MoEGate stand-in: an nn.Parameter weight (requires grad, as MoEGate's) and
    FP32 logits from cuBLAS."""

    tiny_router_gemm_max_tokens = ROUTER_TINY_ROWS

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        return _cublas_fp32(x, self.weight.detach())


def _random_weight(n, k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=g, dtype=torch.float32) / math.sqrt(k)
    return w.to(torch.bfloat16).cuda()


def _real_weight(role, shape):
    """Rank-0 slice of the checkpoint's BF16 weight for ``role``, or None."""
    root = os.environ.get(CKPT_ENV)
    index = os.path.join(root or "", "model.safetensors.index.json")
    if not root or not os.path.exists(index):
        return None
    from safetensors import safe_open

    with open(index) as f:
        weight_map = json.load(f)["weight_map"]
    n, k = shape

    def get(name, rows=None, cols=None):
        shard = weight_map.get(name)
        if shard is None:
            return None
        with safe_open(os.path.join(root, shard), framework="pt", device="cpu") as f:
            t = f.get_slice(name)
            t = t[
                slice(*rows) if rows else slice(None),
                slice(*cols) if cols else slice(None),
            ]
        return t if t.is_floating_point() else None

    mtp = 78
    parts = {
        "router": lambda: [get("model.layers.3.mlp.gate.weight")],
        "wk_weights_proj": lambda: [
            get("model.layers.0.self_attn.indexer.wk.weight"),
            get("model.layers.0.self_attn.indexer.weights_proj.weight"),
        ],
        "o_proj": lambda: [get("model.layers.3.self_attn.o_proj.weight", cols=(0, k))],
        "dense_down": lambda: [get("model.layers.0.mlp.down_proj.weight", cols=(0, k))],
        "eh_proj": lambda: [get(f"model.layers.{mtp}.eh_proj.weight", cols=(0, k))],
        # TP rank 0's vocab-parallel shard of the LM head
        "lm_head": lambda: [get("lm_head.weight", rows=(0, n))],
    }[role]()
    if any(p is None for p in parts):
        return None
    w = torch.cat(parts, 0).to(torch.bfloat16).contiguous()
    return w.cuda() if tuple(w.shape) == (n, k) else None


def _module(e: plan.Entry, *, real=False):
    n, k = e.shape
    w = _real_weight(e.role, e.shape) if real else None
    if real and w is None:
        pytest.skip(f"{CKPT_ENV}: no BF16 {e.role} weight of shape {e.shape}")
    if w is None:
        w = _random_weight(n, k, seed=sum(map(ord, e.role)) * 7919 + n + k)
    if e.role == "router":
        return _Gate(w)
    if e.role == "eh_proj" and e.shape == (6144, 12288):
        lin = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(w)
        return lin  # nn.Linear keeps requires_grad (the NextN eh_proj)
    return _Linear(w)


def _input(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)


def _rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _expected_kernels(e, min_m):
    out = {}
    for m in ROWS:
        r = plan.range_at(e, m)
        if r is not None and m >= min_m:
            out[m] = plan.kernel_name(r)
    return out


def _build(e, module, source):
    os.environ[gemm.SPLITK_ENV] = source
    try:
        gemm.reset()
        gemm.build(e.role, module, ROWS)
    finally:
        del os.environ[gemm.SPLITK_ENV]


def _ids(e):
    return f"{e.role}-{e.shape[0]}x{e.shape[1]}"


# ------------------------------------------------------------------ per role


@needs_plan_device
@pytest.mark.parametrize("real", [False, True], ids=["random", "ckpt"])
@pytest.mark.parametrize("e", ENTRIES, ids=_ids)
def test_role(e, real):
    module = _module(e, real=real)
    assert HAS_FI1 or not REQUIRE_FI1, (
        "FlashInfer's mm_bf16(..., tactic=) (fi-1) is not on the path"
    )
    n, k = e.shape
    min_m = gemm._min_rows(e.role, module)
    sources = ("flashinfer", "vendored") if HAS_FI1 else ("vendored",)
    outputs = {}
    for source in sources:
        _build(e, module, source)
        served = gemm.served_rows(e.role, n, k)
        assert {
            m: name.split(" [")[0] for m, name in served.items()
        } == _expected_kernels(e, min_m)
        assert all(
            name.endswith(f"[{source}]") for name in served.values() if name != plan.TGV
        )
        for m in ROWS:
            x = _input(m, k, seed=1000 + m)
            y = gemm.linear(e.role, module, x)
            assert torch.equal(y, gemm.linear(e.role, module, x)), (
                f"{source} M={m}: not deterministic"
            )
            outputs[(source, m)] = y
            if m not in served:
                assert torch.equal(y, gemm.stock(e.role, module, x)), (
                    f"M={m}: unserved row differs from stock"
                )
    if HAS_FI1:
        for m in ROWS:
            assert torch.equal(outputs[("flashinfer", m)], outputs[("vendored", m)]), (
                f"{_ids(e)} M={m}: fi-1 and the vendored split-K differ"
            )
    # numerics against cuBLAS and FP64
    w = module.weight.detach()
    name = "ll.gemm.router.rel" if e.role == "router" else "ll.gemm.bf16.rel"
    slack = 1e-6 if e.role == "router" else 2e-4
    agree = []
    for m in ROWS:
        if m not in _expected_kernels(e, min_m):
            continue
        x = _input(m, k, seed=1000 + m)
        y = outputs[(sources[-1], m)]
        ref = _cublas_fp32(x, w) if e.role == "router" else F.linear(x, w)
        exact = x.double() @ w.double().T
        rel = _rel(y, ref)
        MEASURED[name] = max(MEASURED.get(name, 0.0), rel)
        assert rel <= plan.TOLERANCES[name], f"{_ids(e)} M={m}: {name} {rel:.2e}"
        rel_new, rel_old = _rel(y, exact), _rel(ref, exact)
        assert rel_new <= 1.5 * rel_old + slack, (
            f"{_ids(e)} M={m}: error {rel_new:.2e} vs cuBLAS {rel_old:.2e}"
        )
        assert torch.isfinite(y).all()
        if e.role == "router":
            top = lambda t: torch.topk(t, 8, dim=-1).indices.sort(-1).values  # noqa: E731
            agree.append((top(y) == top(ref)).all(-1).float().mean().item())
    if agree:
        assert min(agree) >= 0.95, f"router top-8 agreement {min(agree):.3f}"


@needs_plan_device
@pytest.mark.parametrize(
    "e", [e for e in ENTRIES if e.ranges[0].kernel == plan.TGV], ids=_ids
)
def test_tgv_roles_run_sglangs_tgv(e):
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm

    module = _module(e)
    gemm.build(e.role, module, ROWS)
    for m, name in gemm.served_rows(e.role, *e.shape).items():
        x = _input(m, e.shape[1], seed=m)
        assert name == plan.TGV
        assert torch.equal(
            gemm.linear(e.role, module, x), cutedsl_bf16_gemm(x, module.weight.detach())
        )


# ------------------------------------------------------------------ graphs and PDL


def _graph_cases():
    out = []
    for e in ENTRIES:
        for m in (6, 24, 48, 96):
            r = plan.range_at(e, m)
            if r is not None and not (e.role == "router" and m <= ROUTER_TINY_ROWS):
                out.append((e, m))
    return out


@needs_plan_device
@pytest.mark.parametrize("source", ["flashinfer", "vendored"])
@pytest.mark.parametrize(
    "e, m",
    _graph_cases(),
    ids=lambda v: _ids(v) if isinstance(v, plan.Entry) else str(v),
)
def test_graph_replay_matches_eager(e, m, source):
    if source == "flashinfer" and not HAS_FI1:
        pytest.skip("FlashInfer without fi-1")
    if source == "flashinfer" and plan.range_at(e, m).kernel == plan.TGV:
        pytest.skip("TGV: no split-K source")
    module = _module(e)
    _build(e, module, source)
    x = _input(m, e.shape[1], seed=7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = gemm.linear(e.role, module, x)
    for i in range(3):
        x.copy_(_input(m, e.shape[1], seed=100 + i))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, gemm.linear(e.role, module, x.clone()))


def _graph_edges(graph):
    """(from-name, to-name, programmatic) of a captured graph's kernel edges, or None
    when this torch / cuda-python cannot read them."""
    from cuda.bindings import driver as cu

    try:
        handle = cu.CUgraph(graph.raw_cuda_graph())
    except Exception:  # noqa: BLE001 - no raw_cuda_graph, or the graph was not kept
        return None

    def ok(res):
        if int(res[0]) != 0:
            raise RuntimeError(res[0])
        return res[1:]

    get_edges = getattr(cu, "cuGraphGetEdges_v2", None) or cu.cuGraphGetEdges
    try:
        res = ok(get_edges(handle, 0))
        count = res[-1]
        frm, to, data, _ = ok(get_edges(handle, count))
    except Exception:  # noqa: BLE001
        return None

    def name(node):
        (kind,) = ok(cu.cuGraphNodeGetType(node))
        if kind != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            return str(kind)
        (params,) = ok(cu.cuGraphKernelNodeGetParams(node))
        for getter, handle_ in (("cuFuncGetName", "func"), ("cuKernelGetName", "kern")):
            try:
                (fname,) = ok(getattr(cu, getter)(getattr(params, handle_)))
                return fname.decode() if isinstance(fname, bytes) else str(fname)
            except Exception:  # noqa: BLE001
                continue
        return "kernel"

    programmatic = int(cu.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
    return [
        (name(a), name(b), int(d.type) == programmatic)
        for a, b, d in zip(frm, to, data)
    ]


@needs_plan_device
@pytest.mark.parametrize(
    "role, m",
    [
        ("router", 24),
        ("router", 64),
        ("wk_weights_proj", 6),
        ("wk_weights_proj", 48),
        ("o_proj", 48),
        ("dense_down", 6),
    ],
)
def test_pdl_edges_in_a_captured_graph(role, m):
    """PDL RMSNorm -> registered GEMM -> PDL RMSNorm (or a copy for the FP32 router):
    the GEMM is launched with PDL (programmatic edge from the norm) and its PDL
    successor too; replay equals eager."""
    import flashinfer

    e = next(e for e in ENTRIES if e.role == role)
    module = _module(e)
    gemm.build(role, module, [m])
    n, k = e.shape
    h = _input(m, k, seed=3)
    gamma = torch.ones(k, device="cuda", dtype=torch.bfloat16)
    xb = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
    gamma_out = torch.ones(n, device="cuda", dtype=torch.bfloat16)
    z = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    def chain():
        flashinfer.norm.rmsnorm(h, gamma, 1e-6, out=xb, enable_pdl=True)
        y = gemm.linear(role, module, xb)
        if y.dtype == torch.bfloat16:
            flashinfer.norm.rmsnorm(y, gamma_out, 1e-6, out=z, enable_pdl=True)
        return y

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        chain()
    torch.cuda.current_stream().wait_stream(side)
    try:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = chain()
    edges = _graph_edges(graph)
    graph.replay()
    torch.cuda.synchronize()
    ref = gemm.linear(role, module, flashinfer.norm.rmsnorm(h, gamma, 1e-6))
    assert torch.equal(y, ref)
    if edges is None:
        pytest.skip("graph edges not readable here (replay checked)")
    print(f"PDL {role} M={m}: {edges}")
    # norm -> GEMM (-> norm): one chain of kernels, every edge programmatic
    assert len(edges) == (2 if y.dtype == torch.bfloat16 else 1), edges
    assert all(p for _, _, p in edges), f"{role} M={m}: a non-PDL edge {edges}"
    kernel = "SplitK" if plan.range_at(e, m).kernel == plan.SPLITK else "Tgv"
    names = {n for a, b, _ in edges for n in (a, b)}
    if "kernel" not in names:  # names readable: the GEMM is the norm's PDL successor
        assert any(kernel in b for _, b, _ in edges), f"{role} M={m}: {edges}"


# ------------------------------------------------------------------ stress


@needs_plan_device
@pytest.mark.parametrize("source", ["flashinfer", "vendored"])
def test_split_k_stress(source):
    """Every split-K problem of the plan (router 17-96, wk 1-64 rows) in one graph,
    replayed many times; each replay is compared on the GPU with the first result.
    A synchronization race that drops or reads a partial early shows as a mismatch."""
    if source == "flashinfer" and not HAS_FI1:
        pytest.skip("FlashInfer without fi-1")
    os.environ[gemm.SPLITK_ENV] = source
    try:
        cases = []
        for e in ENTRIES:
            if e.ranges[0].kernel != plan.SPLITK:
                continue
            module = _module(e)
            rows = [
                m
                for m in range(1, 97)
                if plan.range_at(e, m) and m >= gemm._min_rows(e.role, module)
            ]
            gemm.build(e.role, module, rows)
            cases += [(e, module, m, _input(m, e.shape[1], seed=m)) for m in rows]
    finally:
        del os.environ[gemm.SPLITK_ENV]
    refs = [gemm.linear(e.role, mod, x) for e, mod, m, x in cases]
    bad = torch.zeros((), dtype=torch.int64, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for (e, mod, m, x), ref in zip(cases, refs):
            gemm.linear(e.role, mod, x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for (e, mod, m, x), ref in zip(cases, refs):
            y = gemm.linear(e.role, mod, x)
            bad.add_((y != ref).any().to(torch.int64))
    # a concurrent stream of other work perturbs the SM schedule
    noise = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    busy = torch.cuda.Stream()
    for i in range(STRESS_REPLAYS):
        if i % 50 == 0:
            with torch.cuda.stream(busy):
                noise = noise @ noise * 1e-3
        graph.replay()
    torch.cuda.synchronize()
    launches = STRESS_REPLAYS * len(cases)
    print(f"STRESS {source}: {launches} split-K launches, {bad.item()} mismatching")
    assert bad.item() == 0


# ------------------------------------------------------------------ rules


def test_auto_source_follows_fi1s_sync_fix_marker():
    """By default the split-K runs FlashInfer's kernel only when fi-1 marks it as
    carrying the synchronization fixes (SplitKTactic.SYNC_FIXED), else the vendored
    copy."""
    fi_gemm = gemm.flashinfer_tactic_api()
    fixed = fi_gemm is not None and gemm.flashinfer_splitk_sync_fixed(fi_gemm)
    print(
        f"SOURCE auto -> {gemm.splitk_source().name} (fi-1 {fi_gemm is not None}, fixed {fixed})"
    )
    assert gemm.splitk_source().name == ("flashinfer" if fixed else "vendored")


def test_plan_declines_on_another_gpu_and_runs_stock():
    e = next(e for e in ENTRIES if e.role == "o_proj")
    module = _module(e)
    gemm.build("o_proj", module, ROWS, sm=(10, 0), sms=148)
    assert gemm.served_rows("o_proj", *e.shape) == {}
    x = _input(48, e.shape[1], seed=1)
    assert torch.equal(
        gemm.linear("o_proj", module, x), gemm.stock("o_proj", module, x)
    )


@needs_plan_device
def test_unbuilt_planned_row_raises():
    e = next(e for e in ENTRIES if e.role == "dense_down")
    module = _module(e)
    gemm.build("dense_down", module, [6])
    gemm.linear("dense_down", module, _input(6, e.shape[1], seed=1))
    with pytest.raises(RuntimeError, match="not built"):
        gemm.linear("dense_down", module, _input(12, e.shape[1], seed=1))


@needs_plan_device
def test_build_under_capture_raises():
    e = next(e for e in ENTRIES if e.role == "router")
    module = _module(e)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="capture"):
        with torch.cuda.graph(graph):
            gemm.build("router", module, [24])


@needs_plan_device
def test_strided_input_raises():
    e = next(e for e in ENTRIES if e.role == "wk_weights_proj")
    module = _module(e)
    gemm.build("wk_weights_proj", module, [8])
    x = _input(8, 2 * e.shape[1], seed=1)[:, : e.shape[1]]
    with pytest.raises(ValueError, match="contiguous"):
        gemm.linear("wk_weights_proj", module, x)
