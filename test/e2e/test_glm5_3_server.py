# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.2 NVFP4 with --enable-nvidia-megakernel on GB300, at one AgentX point
configuration (glm5_3_server.CONFIG): EAGLE verify steps of 5 or 6 rows a request.

``TestGlm53OnOff``: the switch-on server engages (its startup line and the decode
graphs it runs) and its GSM8K-200 accuracy stays in the switch-off server's band; the
switch-off server never engages.

``TestGlm53Parity``: with ``SGLANG_NVIDIA_MEGAKERNEL_SLOTS=""`` every covered width
composes its stock bodies; greedy outputs (token ids and their logprobs) must equal the
switch-off server's bitwise, at batch 1 and 2 (one and two requests' rows).

``TestGlm53HiCacheLoadBack`` (HiCache configurations): a prefix evicted to host comes
back layer by layer while later steps run. On each arm the greedy continuation of a
loaded-back prefix must equal, bitwise, the continuation of the same prefix served from
the device pool (the KV the steps read is the same; only where it came from differs),
and every load-back request must hit host. The switch-on outputs are not SGLang's
bitwise (the fused kernels have their own numerics), so the switch-on vs switch-off
agreement is printed, not asserted.

``SGLANG_NVIDIA_MEGAKERNEL_E2E_URL`` runs the GSM8K check against a server that is
already up, so the same check also scores the switch-off arm.
"""

from __future__ import annotations

import os
import re
import unittest
from types import SimpleNamespace

import requests
from glm5_3_server import (
    ARGS,
    CONFIG,
    DETERMINISTIC_ARGS,
    DETERMINISTIC_ENV,
    ENV,
    HAS_HICACHE,
    ROWS_PER_REQUEST,
    URL,
    log_text,
    server,
)
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase

from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
GSM8K_BOUND = float(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BOUND", "0.9"))
# switch-on vs switch-off GSM8K-200 accuracy (gross failures; L5 judges finer)
GSM8K_BAND = float(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BAND", "0.03"))
GSM8K_QUESTIONS = int(
    os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_QUESTIONS", "200")
)
UNITS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"
ACTIVE = "NVIDIA megakernel: active ("
GRAPHS = re.compile(
    r"NVIDIA megakernel: .+? runs the decode graphs of batch sizes \[([\d, ]*)\]"
)


def _env(**extra):
    # acceptance is real in every test here: the pin would hide a draft / verify mismatch
    env = {
        k: v
        for k, v in os.environ.items()
        if k != UNITS_ENV and not k.startswith("SGLANG_SIMULATE_ACC")
    }
    return {**env, **ENV, **extra}


def _gsm8k(url, parallel, questions=GSM8K_QUESTIONS):
    host, port = re.match(r"https?://([^:/]+):(\d+)", url).groups()
    args = SimpleNamespace(
        num_shots=5,
        data_path=os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K"),
        num_questions=questions,
        parallel=parallel,
        max_new_tokens=512,
        host=host,
        port=int(port),
        temperature=0.0,
    )
    metrics = run_eval_few_shot_gsm8k(args)
    print(f"[{CONFIG}] gsm8k parallel={parallel} {metrics=}")
    return metrics


def _generate(url, input_ids, max_new_tokens, logprobs=True):
    r = requests.post(
        url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": logprobs,
        },
        timeout=1800,
    )
    r.raise_for_status()
    return r.json()


def _greedy(out):
    """What two bitwise-equal decodes share: token ids, their logprobs, verify steps."""
    outs = out if isinstance(out, list) else [out]
    return [
        (
            o["output_ids"],
            [x[0] for x in o["meta_info"].get("output_token_logprobs") or []],
            o["meta_info"].get("spec_verify_ct"),
        )
        for o in outs
    ]


def _ids(rng_seed, n):
    import random

    rng = random.Random(rng_seed)
    return [rng.randrange(1000, 150000) for _ in range(n)]


def _assert_engaged(case, text, rows_per_request=ROWS_PER_REQUEST):
    case.assertIn(ACTIVE, text, "switch-on server did not engage")
    m = GRAPHS.findall(text)
    case.assertTrue(m, "no decode graph runs the megakernel")
    case.assertNotIn("Failed to build JIT module", text)
    print(
        f"[{CONFIG}] covered decode batch sizes {m[-1]} ({rows_per_request} rows a request)"
    )


@unittest.skipIf(URL, "launches its own servers")
class TestGlm53OnOff(CustomTestCase):
    def test_switch_on_engages_and_keeps_gsm8k(self):
        scores = {}
        for name, flag in (("on", [FLAG_CLI]), ("off", [])):
            with server(name, ARGS + flag, _env(), LAUNCH_TIMEOUT) as (_, log):
                scores[name] = _gsm8k(DEFAULT_URL_FOR_TEST, 4)["accuracy"]
                text = log_text(log)
            if flag:
                _assert_engaged(self, text)
            else:
                self.assertNotIn(ACTIVE, text, "switch-off server engaged")
            self.assertNotIn("Traceback", text)
        self.assertGreater(scores["on"], GSM8K_BOUND)
        self.assertLessEqual(
            abs(scores["on"] - scores["off"]), GSM8K_BAND, f"{scores=}"
        )


@unittest.skipUnless(
    URL, "set SGLANG_NVIDIA_MEGAKERNEL_E2E_URL to score a running server"
)
class TestRunningServerGsm8k(CustomTestCase):
    def test_gsm8k(self):
        for parallel in (1, 4):
            self.assertGreater(
                _gsm8k(URL, parallel)["accuracy"], GSM8K_BOUND, f"parallel {parallel}"
            )


@unittest.skipIf(URL, "launches its own servers")
class TestGlm53Parity(CustomTestCase):
    """Every covered width on its stock bodies is SGLang's step bitwise."""

    def _run(self, name, args, env):
        prompts = [_ids(7 + i, 512 + 97 * i) for i in range(2)]
        with server(name, args + DETERMINISTIC_ARGS, env, LAUNCH_TIMEOUT) as (_, log):
            got = {
                1: _greedy(_generate(DEFAULT_URL_FOR_TEST, prompts[0], 64)),
                2: _greedy(_generate(DEFAULT_URL_FOR_TEST, prompts, 64)),
            }
            return got, log_text(log)

    def test_stock_bodies_are_bitwise(self):
        det = _env(**DETERMINISTIC_ENV)
        off, _ = self._run("parity-off", ARGS, det)
        on, text = self._run("parity-on", ARGS + [FLAG_CLI], {**det, UNITS_ENV: ""})
        self.assertIn(ACTIVE, text, "parity mode must keep the covered widths")
        for batch in (1, 2):
            self.assertEqual(
                on[batch],
                off[batch],
                f"batch {batch}: stock bodies differ from SGLang's step",
            )


@unittest.skipIf(URL or not HAS_HICACHE, "launches its own HiCache servers")
class TestGlm53HiCacheLoadBack(CustomTestCase):
    # a device pool small enough that the fillers push the earlier prefixes to host
    DEVICE_TOKENS = int(
        os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_HC_DEVICE_TOKENS", "98304")
    )
    CASES = ((8192, 128), (32768, 512))
    FILL = 16384
    NEW = 48

    def _evict(self, seed):
        left, i = self.DEVICE_TOKENS + self.FILL, 0
        while left > 0:
            _generate(
                DEFAULT_URL_FOR_TEST,
                _ids(seed * 1000 + i, self.FILL),
                1,
                logprobs=False,
            )
            left, i = left - self.FILL, i + 1

    def _arm(self, name, flag):
        args = (
            ARGS
            + flag
            + DETERMINISTIC_ARGS
            + [
                "--enable-cache-report",
                "--max-total-tokens",
                str(self.DEVICE_TOKENS),
            ]
        )
        rows = {}
        with server(name, args, _env(**DETERMINISTIC_ENV), LAUNCH_TIMEOUT) as (_, log):
            for k, (p, n) in enumerate(self.CASES):
                prefix, tail = _ids(100 + k, p), _ids(200 + k, n)
                _generate(DEFAULT_URL_FOR_TEST, prefix, 1, logprobs=False)
                device = _generate(DEFAULT_URL_FOR_TEST, prefix + tail, self.NEW)
                self._evict(k)
                host = _generate(DEFAULT_URL_FOR_TEST, prefix + tail, self.NEW)
                details = host["meta_info"].get("cached_tokens_details") or {}
                rows[(p, n)] = (
                    _greedy(device),
                    _greedy(host),
                    int(details.get("host", 0) or 0),
                )
            text = log_text(log)
        return rows, text

    def test_load_back_equals_device_hit(self):
        arms = {}
        for name, flag in (("hicache-on", [FLAG_CLI]), ("hicache-off", [])):
            rows, text = self._arm(name, flag)
            if flag:
                _assert_engaged(self, text)
            self.assertNotIn("Traceback", text)
            for case, (device, host, host_tokens) in rows.items():
                self.assertGreater(
                    host_tokens,
                    0,
                    f"{name} {case}: the prefix did not come back from host",
                )
                self.assertEqual(
                    host, device, f"{name} {case}: load-back changed the greedy output"
                )
            arms[name] = rows
        for case in self.CASES:
            on, off = arms["hicache-on"][case][0][0], arms["hicache-off"][case][0][0]
            same = sum(a == b for a, b in zip(on[0], off[0]))
            print(
                f"[{CONFIG}] {case}: switch-on vs switch-off greedy tokens equal {same}/{len(off[0])}"
            )


if __name__ == "__main__":
    unittest.main()
