# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-stop of the GLM low-latency hops on 4 SM100-family GPUs (GB300): when one
rank never joins a hop, the ranks that wait on it trap within the hop deadline
instead of hanging, and each leaves the host-mapped error record naming the wait,
its rank and the hop's site, readable after its CUDA context is lost.

One case per cross-GPU wait of the hop kernel: the one-shot poll (plain hop, 6
rows), the two-shot scatter / broadcast polls (plain hop, 40 rows) and the
finalize hop's one-shot poll (6 rows). Each case runs its own torchrun, since a
trapped rank leaves its CUDA context unusable.

Run with ``python test/kernels/test_glm5_3_ll_hops_fail_stop.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest
import torch

DEADLINE_S = 2.0
SITE = 4 * 41 + 1  # layer 41, the MLP-output hop
WORKER = textwrap.dedent(
    """
    import os, sys, time
    import torch, torch.distributed as dist
    from sgl_nvidia_mega.glm5_3.ll import hops

    case, deadline = sys.argv[1], float(sys.argv[2])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    cpu = dist.new_group(backend="gloo")
    fam = hops.LlHopFamily(device_group=dist.group.WORLD, cpu_group=cpu, eps=1e-5,
                           deadline_s=deadline, nvfp4=False)
    fam.compile([48])
    H = 6144
    rows = 40 if case == "two_shot" else 6
    g = torch.Generator().manual_seed(rank)
    x = torch.randn(rows, H, generator=g).to(dev, torch.bfloat16)
    res = torch.randn(rows, H, generator=g).to(dev, torch.bfloat16)
    gamma = torch.ones(H, device=dev, dtype=torch.bfloat16)
    site = int(sys.argv[3])
    if case == "finalize":
        p = rows * 8
        fin = hops.DeferredFinalize(
            torch.randn(p, H, generator=g).to(dev, torch.bfloat16),
            torch.rand(rows, 8, generator=g).to(dev),
            torch.randperm(p, generator=g).view(rows, 8).int().to(dev), None, 8)
        run = lambda: fam.finalize(fin, res, gamma, site=site)
    else:
        run = lambda: fam.allreduce_norm(x, res, gamma, site=site)
    run(); torch.cuda.synchronize()  # every rank joins once: built and healthy
    dist.barrier(group=cpu)
    print(f"RANK{rank} HEALTHY", flush=True)
    if rank == world - 1:
        time.sleep(4 * deadline + 20)  # never joins the second hop
        os._exit(0)
    t0 = time.time()
    try:
        run(); torch.cuda.synchronize()
        print(f"RANK{rank} NO-FAULT", flush=True)
    except Exception as e:
        rec = fam.error_report()
        print(f"RANK{rank} FAULTED after {time.time() - t0:.1f} s: {type(e).__name__}; "
              f"record {rec}", flush=True)
    os._exit(0)
    """
)
KINDS = {
    "one_shot": {"one-shot poll"},
    "finalize": {"one-shot poll"},
    "two_shot": {"two-shot scatter poll", "two-shot broadcast poll"},
}


def _gpu4() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and torch.cuda.device_count() >= 4
    )


@pytest.mark.skipif(not _gpu4(), reason="needs 4 SM100-family GPUs")
@pytest.mark.parametrize("case", sorted(KINDS))
def test_a_missing_rank_traps_its_hop_peers_in_time(case, tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    port = 29500 + (os.getpid() + 37 * len(case)) % 2000
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=4",
         f"--master_port={port}", str(script), case, str(DEADLINE_S), str(SITE)],
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    log = out.stdout + out.stderr
    print("\n".join(l for l in log.splitlines() if "RANK" in l))
    assert log.count("HEALTHY") == 4, log[-4000:]
    assert "NO-FAULT" not in log, log[-4000:]
    faulted = [l for l in log.splitlines() if "FAULTED after" in l]
    assert len(faulted) == 3, log[-4000:]
    for line in faulted:
        assert float(line.split("after ")[1].split(" s")[0]) < 4 * DEADLINE_S + 10, line
        rec = eval(line.split("record ", 1)[1])  # noqa: S307 - our own repr
        assert rec is not None, line
        assert rec["kind"] in KINDS[case], line
        assert rec["site"] == SITE and rec["layer"] == 41 and rec["hop"] == "mlp_output"
        assert rec["rank"] == int(line.split("RANK")[1].split()[0]), line
    assert time.time() - t0 < 600


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s", "-p", "no:cacheprovider"]))
