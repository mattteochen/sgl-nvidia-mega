# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""R7 probe: the plugin-owned hop buffer across two GB300 trays of one NVL72 rack
(8 ranks, W=8). Not on the TP4 critical path; it decides whether the owned buffer
can serve TP8 too.

1. Both allocators map NVLink multicast across the trays: torch symmetric memory
   (``rendezvous(...).multicast_ptr != 0``) and FlashInfer's ``McastGPUBuffer``
   (``get_multicast_ptr()``).
2. The W=8 one-shot hop on the owned buffer is bit-identical to FlashInfer's
   one-shot (on FlashInfer's own workspace) at 6, 12 and 18 rows with FlashInfer's
   geometry, and both allocators give the same bits. (The full hop suite at W=8,
   ``test_glm5_3_ll_hops.py``, runs in the same job.)
3. Fail-stop across trays: the last rank (second tray) never joins a hop; the other
   seven trap within the deadline with the error record (``--fail-stop`` mode, its
   own torchrun since a trapped rank loses its CUDA context).

Run on two trays: ``torchrun --nnodes 2 --nproc-per-node 4 ... -m pytest
test/kernels/test_glm5_3_ll_hops_two_tray.py`` and ``torchrun ...
test/kernels/test_glm5_3_ll_hops_two_tray.py --fail-stop``.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "8",
    reason="two trays: torchrun --nnodes 2 --nproc-per-node 4",
)

H = 6144
EPS = 1e-5
DEADLINE_S = 3.0


def _init():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=dev)
    return dev, dist.new_group(backend="gloo")


def log(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg, flush=True)


def _inputs(rows, dev, seed=0):
    rank = dist.get_rank()
    g = torch.Generator().manual_seed(seed)
    res = (torch.randn(rows, H, generator=g) * 0.6).to(dev, torch.bfloat16)
    gr = torch.Generator().manual_seed(seed * 31 + rank)
    x = (torch.randn(rows, H, generator=gr) * 0.25).to(dev, torch.bfloat16)
    gamma = (1 + 0.2 * torch.randn(H, generator=g)).to(dev, torch.bfloat16)
    return x, res, gamma


def _same(a, b) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_owned_buffer_across_trays():
    from flashinfer.comm import (
        AllReduceFusionPattern,
        Mapping,
        MNNVLAllReduceFusionWorkspace,
        allreduce_fusion,
    )
    from flashinfer.comm.mnnvl import TorchDistBackend

    from sgl_nvidia_mega.glm5_3.ll import hops, transport

    dev, cpu = _init()
    world, rank = dist.get_world_size(), dist.get_rank()
    fams = {}
    for alloc in ("symm_mem", "mcast"):
        fam = hops.LlHopFamily(
            device_group=dist.group.WORLD, cpu_group=cpu, eps=EPS, deadline_s=10.0,
            allocator=alloc, nvfp4=False,
        )  # fmt: skip
        fam.compile([48])
        buf = fam.buffer
        ptrs = [None] * world
        dist.all_gather_object(ptrs, (buf.mcast_ptr != 0, buf.allocator), group=cpu)
        log(
            f"[1] {alloc}: multicast mapped on every rank: {all(p[0] for p in ptrs)}; {buf.describe()}"
        )
        assert all(p[0] for p in ptrs)
        fams[alloc] = fam
    ws = MNNVLAllReduceFusionWorkspace(
        Mapping(world_size=world, rank=rank, gpus_per_node=4, tp_size=world),
        buffer_size_in_bytes=transport.buffer_bytes(world, 48, H),
        comm_backend=TorchDistBackend(group=cpu),
    )
    bad = []
    for rows in (6, 12, 18):
        x, res, gamma = _inputs(rows, dev, seed=rows)
        geo = hops.oneshot_geometry(rows, H, fams["symm_mem"].sm_count)
        launch = transport.Launch(*geo, unicast=False, max_carveout=True)
        outs = {
            a: transport.ar_rmsnorm(f.buffer, x, res, gamma, EPS, launch)
            for a, f in fams.items()
        }
        n, r = torch.empty_like(x), torch.empty_like(res)
        allreduce_fusion(
            input=x, workspace=ws, pattern=AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True, residual_out=r, norm_out=n, residual_in=res,
            rms_gamma=gamma, rms_eps=EPS, use_oneshot=True,
        )  # fmt: skip
        ok_fi = _same(outs["symm_mem"][0], n) and _same(outs["symm_mem"][1], r)
        ok_alloc = all(_same(a, b) for a, b in zip(outs["symm_mem"], outs["mcast"]))
        plan = hops.plain_plan(world, rows, H, fams["symm_mem"].sm_count)
        pn, pr = fams["symm_mem"].allreduce_norm(x, res, gamma, site=0)
        log(f"[2] M={rows:2d} geo {geo}: == FlashInfer one-shot {ok_fi}; symm_mem == mcast "
            f"{ok_alloc}; plan {plan.cluster}x{plan.block} residual exact {_same(pr, r)}")  # fmt: skip
        if not (ok_fi and ok_alloc and _same(pr, r)):
            bad.append(f"M={rows}")
    torch.cuda.synchronize()
    dist.barrier(group=cpu)
    assert not bad, bad


def fail_stop_worker() -> None:
    from sgl_nvidia_mega.glm5_3.ll import hops

    dev, cpu = _init()
    world, rank = dist.get_world_size(), dist.get_rank()
    fam = hops.LlHopFamily(
        device_group=dist.group.WORLD,
        cpu_group=cpu,
        eps=EPS,
        deadline_s=DEADLINE_S,
        nvfp4=False,
    )
    fam.compile([48])
    x, res, gamma = _inputs(6, dev)
    run = lambda: fam.allreduce_norm(x, res, gamma, site=4 * 10 + 1)  # noqa: E731
    run()
    torch.cuda.synchronize()
    dist.barrier(group=cpu)
    print(f"RANK{rank} HEALTHY", flush=True)
    if rank == world - 1:
        time.sleep(4 * DEADLINE_S + 20)
        os._exit(0)
    t0 = time.time()
    try:
        run()
        torch.cuda.synchronize()
        print(f"RANK{rank} NO-FAULT", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"RANK{rank} FAULTED after {time.time() - t0:.1f} s (deadline {DEADLINE_S} s): "
              f"{type(e).__name__}; record {fam.error_report()}", flush=True)  # fmt: skip
    os._exit(0)


if __name__ == "__main__":
    if "--fail-stop" in sys.argv:
        fail_stop_worker()
    sys.exit(pytest.main([__file__, "-q", "-s", "-p", "no:cacheprovider"]))
