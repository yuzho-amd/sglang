"""GEAK optimized HIP W4A8 MoE kernel integration for sglang.

This package wraps a custom HIP MoE GEMM kernel (FP8 e4m3 activation x
MXFP4 e2m1 weight + per-1x32 e8m0 microscale -> bf16) into sglang's MoE
forward path. It mirrors the orchestration of aiter.fused_moe_2stages
while substituting the two MoE GEMM stages (steps "b" and "d") with the
optimized HIP kernel. Steps a/c/e (pre-quant, swiglu+quant, post-reduce)
are reused from aiter helpers / torch.
"""

from sglang.srt.layers.moe.geak_w4a8_hip.geak_fused_moe import geak_fused_moe

__all__ = ["geak_fused_moe"]
