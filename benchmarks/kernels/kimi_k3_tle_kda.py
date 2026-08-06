#!/usr/bin/env python3
"""Benchmark vLLM Kimi K3 KDA prefill backends.

This benchmark intentionally measures eager CUDA-event latency. Profiler
reports are collected by ``tools/run_kimi_k3_tle_experiment.sh`` and are not
used as the latency source.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    chunk_kda_with_fused_gate,
)
from vllm.models.kimi_k3.nvidia.ops.tle_kda import (
    is_tle_kda_supported,
    tle_kda_prefill,
)

HEAD_DIM = 128
LOWER_BOUND = -3.0
BACKEND_CHECK_ATOL = 5e-2
BACKEND_CHECK_RTOL = 5e-2


def make_inputs(tokens: int, heads: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260806)
    return {
        "q": torch.randn(
            1, tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16,
            generator=generator,
        ),
        "k": torch.randn(
            1, tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16,
            generator=generator,
        ),
        "v": torch.randn(
            1, tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16,
            generator=generator,
        ),
        "raw_g": torch.randn(
            1, tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16,
            generator=generator,
        ),
        "raw_beta": torch.randn(
            1, tokens, heads, device=device, dtype=torch.bfloat16,
            generator=generator,
        ),
        "A_log": torch.randn(
            heads, device=device, dtype=torch.float32, generator=generator
        ),
        "dt_bias": torch.randn(
            heads, HEAD_DIM, device=device, dtype=torch.float32, generator=generator
        ),
        "initial_state": torch.randn(
            1, heads, HEAD_DIM, HEAD_DIM,
            device=device,
            dtype=torch.float32,
            generator=generator,
        ),
        "cu_seqlens": torch.tensor(
            [0, tokens], device=device, dtype=torch.int32
        ),
    }


def run_backend(inputs: dict[str, torch.Tensor], backend: str):
    if backend == "tle":
        if not is_tle_kda_supported(HEAD_DIM, torch.bfloat16, LOWER_BOUND):
            raise RuntimeError("This GPU does not satisfy the H20 TLE KDA contract")
        return tle_kda_prefill(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["raw_g"],
            beta=inputs["raw_beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            lower_bound=LOWER_BOUND,
            initial_state=inputs["initial_state"],
            cu_seqlens=inputs["cu_seqlens"],
        )
    if backend == "triton":
        return chunk_kda_with_fused_gate(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            raw_g=inputs["raw_g"],
            raw_beta=inputs["raw_beta"],
            A_log=inputs["A_log"],
            g_bias=inputs["dt_bias"],
            lower_bound=LOWER_BOUND,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inputs["cu_seqlens"],
        )
    raise ValueError(f"Unsupported backend: {backend}")


def benchmark(
    inputs: dict[str, torch.Tensor], backend: str, warmup: int, iterations: int
) -> list[float]:
    working_inputs = inputs.copy()
    working_v = inputs["v"].clone()
    working_inputs["v"] = working_v

    with torch.inference_mode():
        for _ in range(warmup):
            working_v.copy_(inputs["v"])
            run_backend(working_inputs, backend)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(iterations):
            working_v.copy_(inputs["v"])
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_backend(working_inputs, backend)
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
    return times_ms


def check_correctness(inputs: dict[str, torch.Tensor]) -> dict[str, float]:
    # The legacy Triton path updates v in-place, so isolate both calls.
    triton_inputs = inputs.copy()
    triton_inputs["v"] = inputs["v"].clone()
    tle_inputs = inputs.copy()
    tle_inputs["v"] = inputs["v"].clone()
    with torch.inference_mode():
        triton_out, triton_state = run_backend(triton_inputs, "triton")
        torch.cuda.synchronize()
        triton_out = triton_out.clone()
        triton_state = triton_state.clone()
        tle_out, tle_state = run_backend(tle_inputs, "tle")
        torch.cuda.synchronize()
        tle_out = tle_out.clone()
        tle_state = tle_state.clone()
    output_max_abs_diff = (tle_out.float() - triton_out.float()).abs().max().item()
    state_max_abs_diff = (tle_state.float() - triton_state.float()).abs().max().item()
    torch.testing.assert_close(
        tle_out,
        triton_out,
        atol=BACKEND_CHECK_ATOL,
        rtol=BACKEND_CHECK_RTOL,
    )
    torch.testing.assert_close(
        tle_state,
        triton_state,
        atol=BACKEND_CHECK_ATOL,
        rtol=BACKEND_CHECK_RTOL,
    )
    return {
        "atol": BACKEND_CHECK_ATOL,
        "rtol": BACKEND_CHECK_RTOL,
        "output_max_abs_diff": output_max_abs_diff,
        "state_max_abs_diff": state_max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("triton", "tle"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this benchmark on the H20 host")
    inputs = make_inputs(args.tokens, args.heads)
    correctness = None
    if args.check:
        correctness = check_correctness(inputs)
    times_ms = benchmark(inputs, args.backend, args.warmup, args.iterations)
    result = {
        "backend": args.backend,
        "tokens": args.tokens,
        "heads": args.heads,
        "head_dim": HEAD_DIM,
        "chunk_size": 16,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "min_ms": min(times_ms),
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "p95_ms": sorted(times_ms)[max(0, int(len(times_ms) * 0.95) - 1)],
        "times_ms": times_ms,
    }
    if correctness is not None:
        result["correctness"] = correctness
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
