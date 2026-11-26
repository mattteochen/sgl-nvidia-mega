# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hop-family API used by the plugin's glue.

* ``unit_specs()``: every compile unit (kernel class, Constexpr configuration, example
  arguments, launch shape) with a cache key that names the class and every Constexpr
  value; the glue compiles them through its own compile cache.
* ``fenced_trigger_supported()``: the load-time check of the o_proj hop's start edge
  (PTX ISA 9.7.15.14 needs PTX ISA 9.2 and a PTX compiler of release 13.2 or newer); the
  plugin refuses slot ``hops`` when it fails.
* ``router_hoist_weights()``: the derived router weights of the router-hoisted o_proj
  hop.
* ``HopKernels``: the compiled units with the mailboxes of one tensor-parallel group,
  called with torch tensors; it chooses the fold form for a row count from the
  cooperative occupancy.
"""

import tempfile
from dataclasses import dataclass, field
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cu
from cutlass import Float32, Int32, Int64

from .. import launch as L
from .allreduce_norm import AllreduceNormHop
from .constants import CONSUMER_THREADS, ELEMS_PER_CTA, HIDDEN, WORLD
from .moe_fold import (
    MoeFoldArBsum,
    MoeFoldArNorm,
    MoeFoldArNormDn,
    RmsnormFromBlockSums,
)
from .o_proj import ROUTER_K, OProjArConsume, OProjArProduce, router_partial_bytes
from .transport import ErrorRecord, HopMailbox

MIN_PTX_ISA = (9, 2)
# The split fold serves 1..6 rows (the plain fold above): the rows it is tested at.
SPLIT_MAX_ROWS = 6
MIN_PTX_COMPILER = (13, 2)
# Opt-in fold forms, a comma list (unset / empty: the served units, unchanged):
#   split8dn  the deferred-norm fold runs its split form without clusters (unit
#             moe_fold_split_dn_nc) from SPLIT8DN_MIN_ROWS rows up to the rows whose
#             16 * rows + 8 CTAs are co-resident (4..8 on GB300); fewer rows keep the
#             served cluster split (T90a k2 foldbench, fold -> q_a: -0.2 us at 4 rows,
#             -1.3 at 7, -0.9 at 8, but +0.5 at 1..2 rows)
#   late8     the normed fold above the split limit (7..8 rows) triggers its dependent
#             after part 1
# Outputs are bitwise those of the served forms.
FOLD_FORM_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_FOLD_FORM"
FOLD_FORMS = ("split8dn", "late8")
SPLIT8_MAX_ROWS = 8
SPLIT8DN_MIN_ROWS = 4


def fold_form(raw=None) -> frozenset:
    """The opt-in fold forms of ``FOLD_FORM_ENV`` (or ``raw``)."""
    import os

    if raw is None:
        raw = os.environ.get(FOLD_FORM_ENV, "")
    toks = frozenset(x.strip().lower() for x in raw.split(",") if x.strip())
    bad = sorted(toks - set(FOLD_FORMS))
    if bad:
        raise ValueError(f"{FOLD_FORM_ENV}={raw!r}: unknown forms {bad} (of {FOLD_FORMS})")
    return toks


def fold_units(form=None) -> dict:
    """The fold units up to 8 rows (served names -> kernel objects) for the opt-in
    forms ``form`` (default: ``fold_form()``); the served objects when empty."""
    form = fold_form() if form is None else frozenset(form)
    units = {
        "moe_fold_split": MoeFoldArNorm(split=True),
        "moe_fold_plain": MoeFoldArNorm(split=False, trigger_late="late8" in form),
        "moe_fold_split_dn": MoeFoldArNormDn(split=True),
        "moe_fold_plain_dn": MoeFoldArNormDn(split=False),
    }
    if "split8dn" in form:
        units["moe_fold_split_dn_nc"] = MoeFoldArNormDn(split=True, nocluster=True)
    return units


@dataclass(frozen=True)
class UnitSpec:
    name: str
    obj: object
    args: tuple

    @property
    def constexprs(self) -> dict:
        """Every Constexpr configuration value of the unit (the compile-cache key
        besides the class)."""
        return {
            k: v
            for k, v in sorted(vars(self.obj).items())
            if isinstance(v, (bool, int, str))
        }

    @property
    def key(self) -> str:
        """Compile-cache key part: qualified class name + every Constexpr configuration
        value."""
        cls = type(self.obj)
        return f"{cls.__module__}.{cls.__qualname__}{sorted(self.constexprs.items())}"


def _z():
    return Int64(0)


def _args_ar():
    return (
        *[_z()] * 5,
        Float32(1e-5),
        Int32(1),
        *[_z()] * 8,
        Int32(0),
        _z(),
        Int32(0),
        cu.CUstream(0),
    )


def _args_produce():
    return (
        *[_z()] * 5,
        Int32(1),
        *[_z()] * 16,
        Int32(0),
        _z(),
        Int32(0),
        cu.CUstream(0),
    )


def _args_consume():
    return (
        *[_z()] * 4,
        Float32(1e-5),
        Int32(1),
        _z(),
        _z(),
        Float32(1.0),
        _z(),
        _z(),
        *[_z()] * 8,
        Int32(0),
        _z(),
        Int32(0),
        cu.CUstream(0),
    )


def _args_fold():
    return (
        *[_z()] * 9,
        Float32(1e-5),
        Int32(1),
        *[_z()] * 8,
        Int32(0),
        _z(),
        Int32(0),
        cu.CUstream(0),
    )


def _args_fold_dn():
    return (
        *[_z()] * 10,
        Int32(1),
        *[_z()] * 8,
        Int32(0),
        _z(),
        Int32(0),
        cu.CUstream(0),
    )


def _args_bsum():
    return (
        *[_z()] * 8,
        Int32(1),
        *[_z()] * 8,
        Int32(0),
        _z(),
        Int32(0),
        cu.CUstream(0),
    )


def _args_apply():
    return (*[_z()] * 4, Float32(1e-5), Int32(1), cu.CUstream(0))


def unit_specs(
    *, plain_allreduce: bool = False, allreduce_rmsnorm: bool = False, form=None
):
    """The compile units of the forward hops (instance A: up to 8 rows; instance B:
    9..16). ``plain_allreduce``: also the plain all-reduce (the draft's eh gather);
    ``allreduce_rmsnorm``: also the residual-free all-reduce + RMSNorm (the draft's
    row-parallel eh_proj sum with the draft layer's input norm); ``form``: the opt-in
    fold forms (default: ``fold_form()``)."""
    fu = fold_units(form)
    specs = [
        UnitSpec(
            "allreduce_norm", AllreduceNormHop(residual=True, norm=True), _args_ar()
        ),
        UnitSpec("o_proj_produce", OProjArProduce(), _args_produce()),
        UnitSpec("o_proj_consume", OProjArConsume(), _args_consume()),
        UnitSpec("o_proj_produce_router", OProjArProduce(router=True), _args_produce()),
        UnitSpec("o_proj_consume_router", OProjArConsume(router=True), _args_consume()),
        UnitSpec("moe_fold_split", fu["moe_fold_split"], _args_fold()),
        UnitSpec("moe_fold_plain", fu["moe_fold_plain"], _args_fold()),
        UnitSpec(
            "o_proj_produce_router16",
            OProjArProduce(router=True, max_rows=16),
            _args_produce(),
        ),
        UnitSpec(
            "o_proj_consume_router16",
            OProjArConsume(router=True, max_rows=16),
            _args_consume(),
        ),
        UnitSpec("moe_fold_block_sums16", MoeFoldArBsum(), _args_bsum()),
        UnitSpec("moe_fold_split_dn", fu["moe_fold_split_dn"], _args_fold_dn()),
        UnitSpec("moe_fold_plain_dn", fu["moe_fold_plain_dn"], _args_fold_dn()),
        UnitSpec("rmsnorm_from_block_sums16", RmsnormFromBlockSums(), _args_apply()),
        *[
            UnitSpec(name, fu[name], _args_fold_dn())
            for name in ("moe_fold_split_dn_nc",)
            if name in fu
        ],
    ]
    if plain_allreduce:
        specs.append(
            UnitSpec(
                "allreduce", AllreduceNormHop(residual=False, norm=False), _args_ar()
            )
        )
    if allreduce_rmsnorm:
        specs.append(
            UnitSpec(
                "allreduce_rmsnorm",
                AllreduceNormHop(residual=False, norm=True),
                _args_ar(),
            )
        )
    return specs


def _version(text, prefix):
    """(major, minor) of the first ``<prefix>X.Y`` in ``text``, else None."""
    i = text.find(prefix)
    if i < 0:
        return None
    head = text[i + len(prefix) :].lstrip().split(None, 1)
    parts = head[0].split(".") if head else []
    digits = ["".join(c for c in p if c.isdigit()) for p in parts[:2]]
    if len(digits) < 2 or not all(digits):
        return None
    return int(digits[0]), int(digits[1])


def fenced_trigger_supported() -> bool:
    """True iff the pinned DSL toolchain emits PTX ISA >= 9.2 and its PTX compiler is
    release >= 13.2: the condition under which a release fence before
    ``launch_dependents`` orders the dependent grid's start (PTX ISA 9.7.15.14). Checked
    on a compile of the o_proj producer whose PTX and cubin are kept in a temporary
    directory."""
    ver = cutlass.CUDA_VERSION
    if (ver.major, ver.minor) < MIN_PTX_COMPILER:
        return False
    with tempfile.TemporaryDirectory() as d:
        f = cute.compile(
            OProjArProduce(),
            *_args_produce(),
            options=f"--keep-ptx --keep-cubin --dump-dir={d}",
        )
        ptx = f.__ptx__ or ""
        cubin = f.__cubin__ or b""
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    isa = _version(ptx, ".version")
    comp = _version(cubin.decode("latin-1"), "Cuda compilation tools, release")
    return (
        isa is not None
        and comp is not None
        and isa >= MIN_PTX_ISA
        and comp >= MIN_PTX_COMPILER
    )


@torch.no_grad()
def router_hoist_weights(
    w_router: torch.Tensor, gamma: torch.Tensor, w_o: torch.Tensor, rank: int
):
    """``[Q_r | W'_slice]`` (256 x 2816) as fp32 split into bf16 hi + lo, with ``W' =
    W_router * diag(gamma)`` (exact in fp64), ``Q_r = W' W_o^(r)`` (fp64 product rounded
    to fp32) and ``W'_slice = W'[:, 768 r : 768 (r + 1)]``. ``w_o`` is this rank's
    o_proj shard [6144, 2048]."""
    slice_w = HIDDEN // WORLD
    wp = w_router.double() * gamma.double()[None, :]
    q = wp @ w_o.double()
    full = torch.cat([q, wp[:, slice_w * rank : slice_w * (rank + 1)]], dim=1).float()
    hi = full.to(torch.bfloat16)
    lo = (full - hi.float()).to(torch.bfloat16)
    assert hi.shape[1] == ROUTER_K
    return hi.contiguous(), lo.contiguous()


@dataclass
class HopKernels:
    """Compiled hop units + the mailboxes of one group. ``compiled`` maps unit names to
    compiled functions (from ``unit_specs``); ``mbox_a`` / ``mbox_b`` are the
    up-to-8-row and 9..16-row mailbox instances; ``rp_a`` / ``rp_b`` their
    router-partial mailboxes."""

    compiled: dict
    mbox_a: HopMailbox
    mbox_b: HopMailbox | None
    rp_a: Any
    rp_b: Any
    err: ErrorRecord
    eps: float
    rscale: float
    split_max_rows: int = 0
    # rows up to which the cluster-free deferred-norm split serves ("split8dn"; from
    # SPLIT8DN_MIN_ROWS rows; 0: not served)
    split8dn_rows: int = 0
    form: frozenset = frozenset()
    _peers: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        compiled,
        group_name,
        device,
        eps,
        rscale,
        device_index,
        *,
        instance_b=True,
        router=True,
        probe=None,
        barrier_group=None,
    ):
        """Collective (every rank of the group): mailbox instance A (up to 8 rows),
        instance B (9..16 rows) when ``instance_b``, their router-partial mailboxes when
        ``router``, and the split fold's row limit from the occupancy query when the
        split fold is compiled. ``probe``: a compiled split fold with its driver library
        (for the occupancy query) when the one in ``compiled`` came from a cache that
        does not keep it. ``barrier_group``: the mailboxes' barrier group (see
        ``HopMailbox``)."""
        rp_a = _router_mailbox(group_name, 8, device) if router else None
        rp_b = (
            _router_mailbox(group_name, 16, device) if router and instance_b else None
        )
        # its init quiesces every rank
        mbox_a = HopMailbox.create(group_name, 8, device, barrier_group=barrier_group)
        mbox_b = (
            HopMailbox.create(group_name, 16, device, barrier_group=barrier_group)
            if instance_b
            else None
        )
        self = cls(
            compiled, mbox_a, mbox_b, rp_a, rp_b, ErrorRecord(device), eps, rscale
        )
        if "moe_fold_split" in compiled:
            self.split_max_rows = self._split_rows(
                device_index, probe or compiled["moe_fold_split"]
            )
        self.form = fold_form()
        if "split8dn" in self.form and "moe_fold_split_dn_nc" in compiled:
            self.split8dn_rows = _split8_rows(device_index)
        return self

    def fold_limits(self):
        """What every rank must agree on: the split-fold row limit (served forms), or
        (it, the cluster-free deferred-norm split limit, the forms) with an opt-in
        form."""
        if not self.form:
            return self.split_max_rows
        return (self.split_max_rows, self.split8dn_rows, tuple(sorted(self.form)))

    def dn_fold_unit(self, rows: int) -> str:
        """The deferred-norm fold unit at ``rows`` (<= 8)."""
        if SPLIT8DN_MIN_ROWS <= rows <= self.split8dn_rows:
            return "moe_fold_split_dn_nc"
        if rows <= self.split_max_rows:
            return "moe_fold_split_dn"
        return "moe_fold_plain_dn"

    def reset(self):
        """Collective. Re-initialize the mailboxes and zero the router-partial words
        (their tags restart below the new row epochs, so no stale term can match)."""
        for mb, rp in ((self.mbox_a, self.rp_a), (self.mbox_b, self.rp_b)):
            if mb is not None:
                mb.init(
                    -1,
                    before_last_barrier=(lambda rp=rp: rp.zero())
                    if rp is not None
                    else None,
                )

    def _split_rows(self, device_index, unit):
        """Largest row count, at most ``SPLIT_MAX_ROWS`` (the rows the split fold is
        served and tested at), whose split fold (2 rows + 1 clusters of 8 CTAs) is
        co-resident."""
        h = L.kernel_handle(unit, device_index, _fold_smem())
        best = 0
        for rows in range(1, SPLIT_MAX_ROWS + 1):
            if (
                L.fits_cooperative(h, rows * 16 + 8, CONSUMER_THREADS)
                and L.max_active_clusters(h, rows * 16 + 8, CONSUMER_THREADS, 8)
                >= 2 * rows + 1
            ):
                best = rows
        return best

    def _rot(self, peers, rank):
        return [Int64(peers[(rank + i) % WORLD]) for i in range(WORLD)]

    def _tail(self, mb, site):
        return (
            Int32(mb.rank),
            Int64(self.err.device_ptr),
            Int32(site),
            cu.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    def mailbox_args(self, site=0):
        """Trailing launch arguments of a kernel that runs on mailbox instance A with
        this family's error record: the eight peer mailboxes in rotated order (this
        rank's first), the rank, the record's device pointer, the error site and the
        current stream. For kernels outside this module that share the instance (the
        draft model's hops), which must keep the instance's hop sequence the same on
        every rank."""
        mb = self.mbox_a
        return (*self._rot(mb.peers, mb.rank), *self._tail(mb, site))

    def allreduce_norm(self, partial, residual, gamma, out_res, out_norm, site=0):
        mb = self.mbox_a
        self.compiled["allreduce_norm"](
            _p(partial),
            _p(residual),
            _p(gamma),
            _p(out_res),
            _p(out_norm),
            Float32(self.eps),
            Int32(partial.shape[0]),
            *self._rot(mb.peers, mb.rank),
            *self._tail(mb, site),
        )

    def allreduce(self, partial, out, site=0):
        """Plain all-reduce of ``partial`` [rows <= 8, 6144] bf16 into ``out`` (the
        rank-order fp32 sum, one bf16 rounding; no residual, no norm), on mailbox
        instance A. Needs the plain unit (``unit_specs(plain_allreduce=True)``)."""
        if partial.shape[0] > 8:
            raise ValueError("the plain all-reduce serves at most 8 rows per call")
        mb = self.mbox_a
        z = Int64(0)
        self.compiled["allreduce"](
            _p(partial),
            z,
            z,
            _p(out),
            z,
            Float32(self.eps),
            Int32(partial.shape[0]),
            *self._rot(mb.peers, mb.rank),
            *self._tail(mb, site),
        )

    def allreduce_rmsnorm(self, partial, gamma, out_res, out_norm, site=0):
        """Residual-free all-reduce + RMSNorm of ``partial`` [rows <= 8, 6144] bf16:
        ``out_res`` = the rank-order fp32 sum rounded once to bf16, ``out_norm`` =
        RMSNorm(out_res) * gamma (the hop's norm arithmetic), on mailbox instance A.
        Needs ``unit_specs(allreduce_rmsnorm=True)``'s unit."""
        if partial.shape[0] > 8:
            raise ValueError("the all-reduce + RMSNorm serves at most 8 rows per call")
        mb = self.mbox_a
        self.compiled["allreduce_rmsnorm"](
            _p(partial),
            Int64(0),
            _p(gamma),
            _p(out_res),
            _p(out_norm),
            Float32(self.eps),
            Int32(partial.shape[0]),
            *self._rot(mb.peers, mb.rank),
            *self._tail(mb, site),
        )

    def o_proj(self, x, w_o, residual, gamma, out_res, out_norm, site=0):
        mb = self.mbox_a
        rows = Int32(x.shape[0])
        pr = self._rot(mb.peers, mb.rank)
        z = Int64(0)
        self.compiled["o_proj_produce"](
            _p(x), _p(w_o), z, z, z, rows, *pr, *[z] * WORLD, *self._tail(mb, site)
        )
        self.compiled["o_proj_consume"](
            _p(residual),
            _p(gamma),
            _p(out_res),
            _p(out_norm),
            Float32(self.eps),
            rows,
            z,
            z,
            Float32(self.rscale),
            z,
            z,
            *pr,
            *self._tail(mb, site),
        )

    def o_proj_router(
        self, x, w_o, qh, ql, bias, residual, gamma, out_res, out_norm, ids, wts, site=0
    ):
        rows = x.shape[0]
        big = rows > 8
        mb, rp = (self.mbox_b, self.rp_b) if big else (self.mbox_a, self.rp_a)
        assert mb is not None, "hop mailbox instance B (9..16 rows) was not built"
        sfx = "16" if big else ""
        pr = self._rot(mb.peers, mb.rank)
        self.compiled["o_proj_produce_router" + sfx](
            _p(x),
            _p(w_o),
            _p(qh),
            _p(ql),
            _p(residual),
            Int32(rows),
            *pr,
            *self._rot(rp.peers, mb.rank),
            *self._tail(mb, site),
        )
        self.compiled["o_proj_consume_router" + sfx](
            _p(residual),
            _p(gamma),
            _p(out_res),
            _p(out_norm),
            Float32(self.eps),
            Int32(rows),
            Int64(rp.local),
            _p(bias),
            Float32(self.rscale),
            _p(ids),
            _p(wts),
            *pr,
            *self._tail(mb, site),
        )

    def moe_fold(
        self,
        pq,
        shp,
        ids,
        wfin,
        g2s,
        residual,
        gamma,
        out_res,
        out_norm,
        bsum=None,
        site=0,
    ):
        """Fold + all-reduce + residual + RMSNorm. 9..16 rows: block-sum fold, then the
        norm apply (``bsum`` [rows, 8] fp32 scratch)."""
        rows = int(out_res.shape[0])
        if rows > 8:
            mb = self.mbox_b
            assert mb is not None, "hop mailbox instance B (9..16 rows) was not built"
            self.compiled["moe_fold_block_sums16"](
                _p(pq),
                _p(shp),
                _p(ids),
                _p(wfin),
                _p(g2s),
                _p(residual),
                _p(out_res),
                _p(bsum),
                Int32(rows),
                *self._rot(mb.peers, mb.rank),
                *self._tail(mb, site),
            )
            self.compiled["rmsnorm_from_block_sums16"](
                _p(out_res),
                _p(bsum),
                _p(gamma),
                _p(out_norm),
                Float32(self.eps),
                Int32(rows),
                cu.CUstream(torch.cuda.current_stream().cuda_stream),
            )
            return
        mb = self.mbox_a
        unit = "moe_fold_split" if rows <= self.split_max_rows else "moe_fold_plain"
        self.compiled[unit](
            _p(pq),
            _p(shp),
            _p(ids),
            _p(wfin),
            _p(g2s),
            _p(residual),
            _p(gamma),
            _p(out_res),
            _p(out_norm),
            Float32(self.eps),
            Int32(rows),
            *self._rot(mb.peers, mb.rank),
            *self._tail(mb, site),
        )

    def moe_fold_dn(
        self, pq, shp, ids, wfin, g2s, residual, gamma, out_res, out_u, bsum, site=0
    ):
        """Fold + all-reduce + residual with the RMSNorm deferred (up to 8 rows):
        ``out_u = bf16(x * gamma)`` and the block sums ``bsum`` [rows, 8] fp32 of the
        norm (``MoeFoldArNormDn``)."""
        rows = int(out_res.shape[0])
        assert rows <= 8, "the deferred-norm fold serves up to 8 rows"
        mb = self.mbox_a
        self.compiled[self.dn_fold_unit(rows)](
            _p(pq),
            _p(shp),
            _p(ids),
            _p(wfin),
            _p(g2s),
            _p(residual),
            _p(gamma),
            _p(out_res),
            _p(out_u),
            _p(bsum),
            Int32(rows),
            *self._rot(mb.peers, mb.rank),
            *self._tail(mb, site),
        )


def _p(t):
    return Int64(t.data_ptr() if t is not None else 0)


def _fold_smem():
    """Dynamic shared memory of the fold units (consumer block + y[8][768] + sh[768]
    fp32)."""
    return 21664 + 4 * (8 * ELEMS_PER_CTA + ELEMS_PER_CTA)


def _split8_rows(device_index) -> int:
    """Rows up to which the cluster-free split fold (``16 * rows + 8`` CTAs, at least
    one per SM: the unit launches) is co-resident."""
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    rows = [r for r in range(1, SPLIT8_MAX_ROWS + 1) if 16 * r + 8 <= sms]
    return max(rows, default=0)


@dataclass
class _RouterMailbox:
    storage: Any
    view: Any
    local: int
    peers: list

    def zero(self):
        self.view.zero_()


def _router_mailbox(group_name, max_rows, device):
    import torch.distributed._symmetric_memory as symm_mem

    n = router_partial_bytes(max_rows)
    st = symm_mem.empty(n + 4096, dtype=torch.uint8, device=device)
    h = symm_mem.rendezvous(st, group_name)
    peers = [int(p) + (-int(p)) % 4096 for p in h.buffer_ptrs]
    local = peers[int(h.rank)]
    off = local - st.data_ptr()
    view = st[off : off + n]
    view.zero_()
    return _RouterMailbox(st, view, local, peers)
