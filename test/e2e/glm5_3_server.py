# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.2 NVFP4 servers the GLM-5.3 / GLM-5.2 serving tests launch on GB300, one per
InferenceX AgentX point configuration (``SGLANG_NVIDIA_MEGAKERNEL_E2E_GLM_CONFIG``):

* ``tp4-416``: aggregated TP4, EAGLE 4/1/5 (verify steps of 5 rows a request), HiCache
* ``tp8-516``: aggregated TP8 on two trays, EAGLE 5/1/6, HiCache
* ``tp4-516``: the disaggregated decode role, TP4, EAGLE 5/1/6, run standalone

The megakernel switch is set per launch, so a test can launch the normal path too.
A two-tray configuration needs its second tray's ranks: with
``SGLANG_NVIDIA_MEGAKERNEL_E2E_PEER_DIR`` set (a directory both trays see), every
launch writes ``<n>.launch.json`` (argv and environment) there for a peer process on
the other tray to start with ``--node-rank 1``, and ``<n>.stop`` when it ends."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

MODEL = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH", "nvidia/GLM-5.2-NVFP4"
)
# An already running server, so the same checks can score the switch-off arm
URL = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_URL")
CONFIG = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GLM_CONFIG", "tp4-416")
# Kept server logs (default: temporary files)
LOG_DIR = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_LOG_DIR")
PEER_DIR = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_PEER_DIR")
NNODES = int(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_NNODES", "1"))
DIST_INIT_ADDR = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_DIST_INIT_ADDR", "")

# The fused draft's embedding front reads the replicated embedding table
ENV = {"SGLANG_ENABLE_EMBED_REPLICATION": "1"}
# Bitwise comparisons: the deterministic DSA top-k in the target and the draft
DETERMINISTIC_ENV = {"SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC": "1"}
DETERMINISTIC_ARGS = [
    "--dsa-topk-backend",
    "flashinfer",
    "--speculative-dsa-topk-backend",
    "flashinfer",
]
HICACHE = [
    "--enable-hierarchical-cache",
    "--hicache-size",
    "100",
    "--hicache-write-policy",
    "write_back",
    "--hicache-io-backend",
    "direct",
]

COMMON = [
    "--trust-remote-code",
    "--quantization",
    "modelopt_fp4",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-eagle-topk",
    "1",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--context-length",
    "65536",
    "--random-seed",
    "42",
    "--watchdog-timeout",
    "1800",
]

# Per configuration: (args, verify rows a request, the configuration uses HiCache)
CONFIGS = {
    "tp4-416": (
        [
            "--tp-size",
            "4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--bf16-gemm-backend",
            "cutedsl",
            "--mem-fraction-static",
            "0.83",
            "--speculative-num-steps",
            "4",
            "--speculative-num-draft-tokens",
            "5",
            "--max-running-requests",
            "8",
            "--cuda-graph-max-bs-decode",
            "8",
        ],
        5,
        True,
    ),
    "tp8-516": (
        [
            "--tp-size",
            "8",
            "--fp4-gemm-backend",
            "flashinfer_trtllm",
            "--mem-fraction-static",
            "0.8",
            "--speculative-num-steps",
            "5",
            "--speculative-num-draft-tokens",
            "6",
            "--max-running-requests",
            "4",
            "--cuda-graph-max-bs-decode",
            "4",
        ],
        6,
        True,
    ),
    "tp4-516": (
        [
            "--tp-size",
            "4",
            "--fp4-gemm-backend",
            "flashinfer_cutedsl",
            "--mem-fraction-static",
            "0.9",
            "--speculative-num-steps",
            "5",
            "--speculative-num-draft-tokens",
            "6",
            "--max-running-requests",
            "24",
            "--cuda-graph-max-bs-decode",
            "24",
        ],
        6,
        False,
    ),
}

_ARGS, ROWS_PER_REQUEST, HAS_HICACHE = CONFIGS[CONFIG]
EXTRA = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS", "").split()
ARGS = COMMON + _ARGS + (HICACHE if HAS_HICACHE else []) + EXTRA

_launches = 0


def node_args(rank: int = 0) -> list[str]:
    if NNODES == 1:
        return []
    return [
        "--nnodes",
        str(NNODES),
        "--node-rank",
        str(rank),
        "--dist-init-addr",
        DIST_INIT_ADDR,
    ]


@contextmanager
def server(name: str, args: list[str], env: dict[str, str], timeout: float):
    """Launch one server (and its peer ranks), yield (process, log path), stop it.

    ``env`` is the server's whole environment; the peer gets what differs from this
    process's (set and unset names)."""
    from sglang.test.test_utils import (
        DEFAULT_URL_FOR_TEST,
        popen_launch_server,
        terminate_and_kill_process_tree,
    )

    global _launches
    _launches += 1
    tag = f"{_launches:02d}-{name}"
    log_dir = Path(LOG_DIR or tempfile.mkdtemp(prefix="glm-dsa-e2e-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / f"{tag}.out", "w+")
    err = open(log_dir / f"{tag}.err", "w+")
    delta = {
        "set": {k: v for k, v in env.items() if os.environ.get(k) != v},
        "unset": sorted(set(os.environ) - set(env)),
    }
    if PEER_DIR:
        peer = Path(PEER_DIR)
        peer.mkdir(parents=True, exist_ok=True)
        (peer / f"{tag}.launch.json").write_text(
            json.dumps({"model": MODEL, "args": args + node_args(1), "env": delta})
        )
    process = None
    try:
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=timeout,
            other_args=args + node_args(0),
            env=env,
            return_stdout_stderr=(out, err),
        )
        yield process, Path(err.name)
    finally:
        if process is not None:
            terminate_and_kill_process_tree(process)
        if PEER_DIR:
            (Path(PEER_DIR) / f"{tag}.stop").touch()
        out.close()
        err.close()


def log_text(path: Path) -> str:
    """stderr and stdout of a launch (SGLang logs to stderr)."""
    text = path.read_text(errors="replace")
    other = path.with_suffix(".out")
    return text + (other.read_text(errors="replace") if other.exists() else "")
