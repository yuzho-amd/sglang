"""End-to-end perf A/B: aiter.fused_moe vs geak_fused_moe.

Times the FULL MoE forward (pre-quant + 2 GEMMs + SwiGLU + reduce) on
both paths, prints TFLOPs and speedup, and persists results to a JSON.

The aiter call uses QuantType.per_1x32 + Swiglu with unshuffled MXFP4
weights; that path is the same one sglang would dispatch when
SGLANG_USE_AITER=1 in production. The geak path uses our HIP kernel
for the two GEMM stages.

Run with:
    cd /sgl-workspace/sglang && \\
        python python/sglang/srt/layers/moe/geak_w4a8_hip/bench_geak_fused_moe.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch

from sglang.srt.layers.moe.geak_w4a8_hip.geak_fused_moe import geak_fused_moe


WARMUP = 50
ITERS = 200

CASES = [
    # M, K, inter, E, topk
    (64,  1024, 1024, 32, 4),
    (128, 1024, 1024, 32, 4),
    (256, 1024, 1024, 32, 4),
]


def _make_inputs(m, k, inter, E, topk, device="cuda", seed=0):
    torch.manual_seed(seed)
    h = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.5
    w1_bf = torch.randn((E, 2 * inter, k), device=device, dtype=torch.bfloat16) * 0.05
    w2_bf = torch.randn((E, k, inter), device=device, dtype=torch.bfloat16) * 0.05
    w1q, w1s = downcast_to_mxfp(w1_bf, torch.uint8, axis=-1)
    w2q, w2s = downcast_to_mxfp(w2_bf, torch.uint8, axis=-1)
    logits = torch.randn((m, E), dtype=torch.float16, device=device)
    probs = F.softmax(logits.float(), dim=-1)
    topk_w, topk_i = torch.topk(probs, topk, dim=-1)
    topk_w = topk_w.float()
    topk_i = topk_i.to(torch.int32)
    return dict(
        h=h, w1q=w1q, w1s=w1s, w2q=w2q, w2s=w2s,
        topk_w=topk_w, topk_i=topk_i,
        m=m, k=k, inter=inter, E=E, topk=topk,
    )


def _time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def main():
    if get_arch() != "gfx950":
        print("ERROR: bench requires CDNA4 (gfx950)")
        sys.exit(1)

    # IMPORTANT CONTEXT: aiter.fused_moe with unshuffled MXFP4 weights
    # (the format produced by downcast_to_mxfp / what our HIP kernel
    # accepts) emits a "preshuffle_off may produce incorrect results"
    # warning and indeed returns numerically wrong output (RMS ~0.18 vs
    # bf16 reference RMS ~0.25, max_abs ~2.4 on the 3 cases below).
    # The aiter timing therefore measures an INCORRECT kernel and is
    # included only as a relative baseline; geak's longer runtime
    # reflects a CORRECT pipeline (verified by test_geak_fused_moe.py).
    # For a fair perf comparison vs aiter, the bench would need to
    # shuffle weights for aiter via shuffle_weight_a16w4 +
    # shuffle_scale_a16w4 (matching sglang/srt/layers/quantization/
    # mxfp4.py:633-684); that's left as a follow-up.
    print("# NOTE: aiter column runs the un-shuffled MXFP4 path which")
    print("# emits a 'preshuffle_off may produce incorrect results'")
    print("# warning. geak column is numerically correct. See file")
    print("# header for the full caveat.")
    print()
    results = []
    print(f"{'M':>5s} {'K':>5s} {'I':>5s} {'E':>3s} {'topk':>4s} "
          f"{'aiter_ms':>10s} {'geak_ms':>10s} "
          f"{'aiter_TF':>10s} {'geak_TF':>10s} {'speedup':>9s}")
    print("-" * 90)

    for (m, k, inter, E, topk) in CASES:
        inp = _make_inputs(m, k, inter, E, topk)

        # Two GEMMs per token: stage1 (M*topk x K -> 2*inter) and
        # stage2 (M*topk x inter -> K). FLOPs counts both.
        flops_stage1 = 2.0 * m * topk * 2 * inter * k
        flops_stage2 = 2.0 * m * topk * k * inter
        flops = flops_stage1 + flops_stage2

        geak_fn = lambda: geak_fused_moe(
            hidden_states=inp["h"],
            w1=inp["w1q"], w2=inp["w2q"],
            topk_weight=inp["topk_w"], topk_ids=inp["topk_i"],
            w1_scale=inp["w1s"], w2_scale=inp["w2s"],
            a1_scale=None, a2_scale=None, bias1=None, bias2=None,
            activation="swiglu",
        )
        aiter_fn = lambda: fused_moe(
            hidden_states=inp["h"],
            w1=inp["w1q"].view(dtypes.fp4x2),
            w2=inp["w2q"].view(dtypes.fp4x2),
            topk_weight=inp["topk_w"], topk_ids=inp["topk_i"],
            quant_type=QuantType.per_1x32, activation=ActivationType.Swiglu,
            w1_scale=inp["w1s"].view(dtypes.fp8_e8m0),
            w2_scale=inp["w2s"].view(dtypes.fp8_e8m0),
        )

        # JIT compile / first-touch
        try:
            geak_fn()
        except Exception as exc:
            print(f"geak_fused_moe failed for {m}x{k}x{inter}: {exc}")
            continue
        try:
            aiter_fn()
        except Exception as exc:
            print(f"aiter.fused_moe failed for {m}x{k}x{inter}: {exc}; skipping")
            aiter_ms = float("nan")
            aiter_tf = float("nan")
            speedup = float("nan")
        else:
            torch.cuda.synchronize()
            aiter_ms = _time(aiter_fn)
            aiter_tf = flops / (aiter_ms * 1e-3) / 1e12
            speedup = None

        torch.cuda.synchronize()
        geak_ms = _time(geak_fn)
        geak_tf = flops / (geak_ms * 1e-3) / 1e12
        if speedup is None:
            speedup = aiter_ms / geak_ms

        results.append(dict(
            m=m, k=k, inter=inter, E=E, topk=topk,
            aiter_ms=aiter_ms, geak_ms=geak_ms,
            aiter_tflops=aiter_tf, geak_tflops=geak_tf,
            speedup=speedup,
        ))
        print(f"{m:5d} {k:5d} {inter:5d} {E:3d} {topk:4d} "
              f"{aiter_ms:10.4f} {geak_ms:10.4f} "
              f"{aiter_tf:10.2f} {geak_tf:10.2f} {speedup:8.3f}x")

    speedups = [r["speedup"] for r in results
                if r["speedup"] == r["speedup"]]  # filter NaN
    med = sorted(speedups)[len(speedups) // 2] if speedups else float("nan")

    out = dict(
        warmup=WARMUP, iters=ITERS,
        median_speedup_vs_aiter=med, results=results,
    )
    out_path = Path(__file__).resolve().parent / "bench.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nMedian speedup (geak / aiter): {med:.3f}x  -> {out_path}")


if __name__ == "__main__":
    main()
