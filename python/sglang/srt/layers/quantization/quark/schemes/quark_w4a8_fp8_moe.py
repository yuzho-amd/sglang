# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.layers.quantization.utils import all_close_1d
from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_hip,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

__all__ = ["QuarkW4A8Fp8MoE"]

_is_fp8_fnuz = is_fp8_fnuz()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_use_geak_hip = get_bool_env_var("SGLANG_W4A8_HIP_GEAK") and _is_hip
_is_gfx950 = is_gfx95_supported()

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A8Fp8MoE(QuarkMoEScheme):
    """Quark MoE scheme: MXFP4 weights + static FP8 activations (W4A8).

    Backend: AITER Triton kernel ``moe_op_gemm_a8w4`` (the same kernel
    vLLM uses for GPT-OSS W4A8 on ROCm).

    Required runtime: ROCm gfx950 (MI355X) + ``SGLANG_USE_AITER=1``.

    Checkpoint contract (per-expert):
      - ``w13_weight``       : uint8 ``[E, 2*N, K/2]`` MXFP4 packed
      - ``w13_weight_scale`` : uint8 ``[E, 2*N, K/32]`` E8M0 block scales
      - ``w13_input_scale``  : fp32 ``[E]`` static per-tensor FP8 activation
                               scale, reduced to scalar at load time
      - ``w2_*``             : analogous with ``K, N`` swapped
    """

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if weight_qscheme != "per_group":
            raise ValueError(
                "For W4A8 Fused MoE layers, only per-group weight scales "
                f"are supported. Found {weight_qscheme}"
            )
        if input_qscheme != "per_tensor":
            raise ValueError(
                "For W4A8 Fused MoE layers, only per-tensor activation scales "
                f"are supported. Found {input_qscheme}"
            )
        if self.input_quant.get("is_dynamic"):
            raise ValueError(
                "For W4A8 Fused MoE layers, only static activation scales "
                "are supported."
            )

        if not _is_gfx950:
            logger.warning(
                "QuarkW4A8Fp8MoE targets the AITER Triton moe_gemm_a8w4 kernel, "
                "which requires gfx950 (MI355X). Current device is not gfx950; "
                "MoE forward will fail at runtime."
            )

        self.with_bias = False

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=_use_aiter,
            is_concat=True,
            is_packed=True,
        )

        extra_weight_attrs.update(
            {
                "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,
                "weight_padded": weight_padded,
            },
        )

        params_dtype = torch.uint8

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size // 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_down_dim,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_up_dim,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                (w2_down_dim * 2) // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        w13_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reduce static per-expert FP8 input scales to a single per-tensor
        # scalar (kernel only consumes a scalar). Warn on non-uniform scales.
        if layer.w13_input_scale is None or layer.w2_input_scale is None:
            raise ValueError(
                "W4A8 MoE requires static activation scales, but found None."
            )
        if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
            layer.w2_input_scale
        ):
            logger.warning(
                "Found input_scales that are not equal for W4A8 MoE layer. "
                "Using the maximum across experts for each stage."
            )
        a13 = layer.w13_input_scale.max().to(torch.float32)
        a2 = layer.w2_input_scale.max().to(torch.float32)
        if _is_fp8_fnuz:
            a13 = a13 * 2.0
            a2 = a2 * 2.0
        layer.w13_input_scale = torch.nn.Parameter(a13, requires_grad=False)
        layer.w2_input_scale = torch.nn.Parameter(a2, requires_grad=False)
        # Cache host-side scalars so the HIP MoE GEMM wrapper doesn't need a
        # D->H sync (.item()) inside the forward path — that sync is illegal
        # during CUDA/HIP graph capture (hipErrorStreamCaptureUnsupported).
        layer.w13_input_scale_value = float(a13.item())
        layer.w2_input_scale_value = float(a2.item())

        # Reorder weights into moe_gemm_a8w4's expected layout:
        #   weight: [E, K/2, N]   - kernel requires ``w.stride(-2) == 1``
        #                           (K on dim -2 must be the fast dim).
        #   scale : [E, K/32, N]  then swizzled via CDNA4_SCALE layout.
        #
        # The Quark checkpoint stores both as [E, N, K/2] / [E, N, K/32]
        # (row-major). We physically reorder the storage so that K is the
        # fast dim WITHOUT relying on a non-contiguous view (CUDA graph
        # capture + nn.Parameter prefer well-known strides). The trick is
        # to allocate a fresh tensor of shape ``[E, N, K/2]`` and copy the
        # row-major source into it via a permute target ``[E, K/2, N]``:
        # the resulting tensor has ``stride = (N*K/2, 1, K/2)`` which is
        # exactly what the kernel wants.
        from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
            swizzle_scales as swizzle_scales_gfx950,
        )

        def _to_k_fast(src: torch.Tensor) -> torch.Tensor:
            """``[E, N, K]`` row-major -> ``[E, K, N]`` with stride(-2)==1."""
            E, N, K = src.shape
            dst = torch.empty_strided(
                (E, K, N), (N * K, 1, K), dtype=src.dtype, device=src.device
            )
            dst.copy_(src.transpose(-2, -1))
            return dst

        # w13: [E, 2*N_inter, K/2] -> [E, K/2, 2*N_inter] (stride(-2)==1).
        w13 = _to_k_fast(layer.w13_weight.data)
        # w13_scale: [E, 2*N_inter, K/32] -> [E, K/32, 2*N_inter] -> swizzle.
        w13_scale = _to_k_fast(layer.w13_weight_scale.data)
        w13_scale = swizzle_scales_gfx950(w13_scale)

        # w2: [E, K (=hidden), N_inter/2] -> [E, N_inter/2, K].
        w2 = _to_k_fast(layer.w2_weight.data)
        w2_scale = _to_k_fast(layer.w2_weight_scale.data)
        w2_scale = swizzle_scales_gfx950(w2_scale)

        # Replace stored params; the kernel reads raw uint8 storage and
        # interprets it as packed MXFP4 (.view) at call time.
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        if hasattr(layer, "dispatcher"):
            # Keep the FP4 dtype hint on the dispatcher so SGLang's MoE plumbing
            # treats this layer as quantized; the dispatcher is bypassed at
            # apply_weights time because we call the AITER triton kernel
            # directly.
            try:
                layer.dispatcher.set_quant_config(
                    {"weight_dtype": torch.float4_e2m1fn_x2}
                )
            except Exception:  # noqa: BLE001
                pass

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        # The AITER triton kernel is invoked directly in apply_weights;
        # no MoeRunner is involved. We only stash the config for any
        # downstream consumers (e.g. activation override).
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        import aiter
        from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
        from aiter.ops.triton.moe.moe_routing.routing import routing
        from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        # SGLANG_W4A8_HIP_GEAK gate: swap aiter's triton moe_gemm_a8w4 for our
        # optimized HIP MoE GEMM (gfx950-only). The HIP kernel matches aiter's
        # call surface for the QuarkW4A8 subset: gather_indx / scatter_indx /
        # gammas / swizzle_mx_scale="CDNA4_SCALE". Pre-/post-quant + SwiGLU
        # are unchanged (still aiter's triton kernels).
        if _use_geak_hip:
            from sglang.srt.layers.moe.geak_w4a8_hip.moe_gemm_a8w4_hip_wrapper import (
                moe_gemm_a8w4_hip,
            )

            def _gemm(
                x, w, x_scales, w_scales, x_static_scale, quant_static_scale,
                bias, rdata, gather_indx, scatter_indx, gammas,
                swizzle_mx_scale, out_dtype, apply_swiglu, add_residual=False,
            ):
                # The geak HIP kernel doesn't fuse output quant, bias, or
                # SwiGLU; QuarkW4A8Fp8MoE doesn't ask for those either
                # (quant_static_scale/bias/apply_swiglu are always None/False
                # in this method's two call sites). Assert to catch drift.
                assert x_scales is None and quant_static_scale is None
                assert bias is None and not apply_swiglu
                return moe_gemm_a8w4_hip(
                    x, w, w_scales, x_static_scale, rdata,
                    out_dtype=out_dtype,
                    gather_indx=gather_indx, scatter_indx=scatter_indx,
                    gammas=gammas, swizzle_mx_scale=swizzle_mx_scale,
                )
        else:
            def _gemm(*args, **kwargs):
                return moe_gemm_a8w4(*args, **kwargs)

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        # SGLang's TopK has already computed topk_ids / topk_weights, but the
        # AITER triton kernel needs aiter's RoutingData (ExptData histogram +
        # gather/scatter index permutations + block_m). Rebuild from the raw
        # router_logits using aiter's own routing helper, which performs
        # softmax-then-topk-then-renormalize (identical math to SGLang's
        # ``TopK(renormalize=True)`` since softmax-then-topk has the same
        # indices as topk-then-softmax over the selected experts).
        router_logits = topk_output.router_logits
        n_expts_act = topk_output.topk_ids.shape[-1]
        routing_data, gather_idx, scatter_idx = routing(
            router_logits.to(torch.float32),
            n_expts_act,
            sm_first=False,  # softmax over topk values (= renormalize semantics)
        )

        # Static per-tensor FP8 quantization of input activations.
        # downcast_to_static_fp8 divides by scale and casts to fp8_e4m3.
        h_fp8 = downcast_to_static_fp8(
            hidden_states.view(-1, hidden_states.shape[-1]),
            layer.w13_input_scale,
        )

        # ------------------------------------------------------------------
        # Stage 1: x_fp8 [M, K] @ w13 [E, K/2, 2*N_inter]  ->  bf16 [M*topk, 2*N_inter]
        # ------------------------------------------------------------------
        # The kernel writes one row per (token, selected_expert) pair via
        # gather_idx; output preserves the [gate; up] stacked layout from the
        # weight, ready for silu_and_mul below.
        stage1_out = _gemm(
            h_fp8,
            layer.w13_weight,
            None,  # x_scales (no per-token MX scale, we are static FP8)
            layer.w13_weight_scale,
            layer.w13_input_scale_value if _use_geak_hip else layer.w13_input_scale,  # x_static_scale (per-tensor)
            None,  # quant_static_scale: no fused output requant
            None,  # bias (Qwen3.5 MoE has no expert bias)
            routing_data,
            gather_indx=gather_idx,
            scatter_indx=None,
            gammas=None,  # gate weights applied at stage2 scatter
            swizzle_mx_scale="CDNA4_SCALE",
            out_dtype=torch.bfloat16,
            apply_swiglu=False,
            add_residual=False,
        )

        # silu(gate) * up then re-quantize to FP8 with the static w2 scale.
        M_topk, two_N = stage1_out.shape
        N_inter = two_N // 2
        inter_bf16 = torch.empty(
            (M_topk, N_inter), dtype=stage1_out.dtype, device=stage1_out.device
        )
        aiter.silu_and_mul(inter_bf16, stage1_out)
        inter_fp8 = downcast_to_static_fp8(inter_bf16, layer.w2_input_scale)

        # ------------------------------------------------------------------
        # Stage 2: inter_fp8 [M*topk, N_inter] @ w2 [E, N_inter/2, K]  ->  bf16 [M, K]
        # ------------------------------------------------------------------
        # gammas = per-(token,expert) gate weight, applied during the scatter
        # reduction so we end up with one bf16 row per original token.
        out = _gemm(
            inter_fp8,
            layer.w2_weight,
            None,
            layer.w2_weight_scale,
            layer.w2_input_scale_value if _use_geak_hip else layer.w2_input_scale,
            None,
            None,
            routing_data,
            gather_indx=None,
            scatter_indx=scatter_idx,
            gammas=routing_data.gate_scal,
            swizzle_mx_scale="CDNA4_SCALE",
            out_dtype=hidden_states.dtype,
            apply_swiglu=False,
            add_residual=False,
        )

        return StandardCombineInput(hidden_states=out)
