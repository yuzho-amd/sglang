"""Python launcher for the HIP MoE W4A8 GEMM.

Extended to mirror aiter.ops.triton.moe.moe_op_gemm_a8w4.moe_gemm_a8w4 for the
features that QuarkW4A8Fp8MoE.apply_weights() actually needs end-to-end:
  - gather_indx        : per-row indirection for stage 1 (token routing input)
  - scatter_indx       : per-row indirection for stage 2 (token routing output)
  - gammas             : per-(token, expert) gate-weight scale at scatter time
  - swizzle_mx_scale   : "CDNA4_SCALE" or None  (kernel must unswizzle on the
                         fly when "CDNA4_SCALE")

Anything else (SwiGLU fusion, SPLIT_K, bias, x microscale) is still out of
scope — those are not used by QuarkW4A8Fp8MoE.

The HIP kernel ABI was extended accordingly. See
moe_op_gemm_a8w4_hip.hip::launch_moe_gemm_a8w4 for the new tensor args.
"""

from typing import Optional

import torch
import triton

from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
    allocate_output, get_kernel_config,
)
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton.moe.reduce import reduce_grouped

from .kernel_loader import get_extension


def moe_gemm_a8w4_hip(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    x_static_scale,
    routing_data: "RoutingData",
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    gather_indx: Optional[torch.Tensor] = None,
    scatter_indx: Optional[torch.Tensor] = None,
    gammas: Optional[torch.Tensor] = None,
    swizzle_mx_scale: Optional[str] = None,
):
    """W4A8 MoE GEMM (HIP).

    Matches the aiter-triton ``moe_gemm_a8w4`` for the
    QuarkW4A8Fp8MoE call subset:
      - FP8 e4m3 activation + per-tensor static scale
      - MXFP4 e2m1 weight + per-1x32 e8m0 scale
      - optional gather_indx / scatter_indx / gammas / swizzle_mx_scale
    """
    assert x.dtype == torch.float8_e4m3fn
    assert w.dtype == torch.uint8 and w_scales.dtype == torch.uint8
    assert w.stride(-2) == 1, "w must be K-fast (stride(-2)==1)"
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K = x.shape[-1]
    N = w.shape[-1]
    config = get_kernel_config(M, N, K, routing_data)

    y, y_final = allocate_output(
        x, w, out_dtype, 1, 1, routing_data, gather_indx, scatter_indx,
        config["block_m"], 1,
    )
    y = y.squeeze(0).contiguous()

    expt_data = routing_data.expt_data
    expt_hist = expt_data.hist.to(torch.int32).contiguous()
    expt_offs = expt_data.token_offs_raw.to(torch.int32).contiguous()
    expt_block_pid_map = expt_data.block_pid_map.to(torch.int32).contiguous()
    expt_offs_sum = (
        None
        if expt_data.token_offs_pad is None
        else expt_data.token_offs_pad[-1:].to(torch.int32).contiguous()
    )

    # Optional tensors passed to the kernel; use empty 1-element placeholders
    # when None so the C++ side doesn't need to handle null Tensors.
    _empty_i32 = torch.empty((0,), dtype=torch.int32, device=x.device)
    _empty_f32 = torch.empty((0,), dtype=torch.float32, device=x.device)

    block_m, block_n, block_k = config["block_m"], config["block_n"], config["block_k"]
    grid_m = routing_data.n_blocks(M, block_m)
    grid_n = triton.cdiv(N, block_n)

    swizzle_flag = 1 if swizzle_mx_scale == "CDNA4_SCALE" else 0
    n_expts_act = int(routing_data.n_expts_act)

    ext = get_extension()
    ext.launch_moe_gemm_a8w4(
        y, x, w, w_scales,
        float(x_static_scale.item() if torch.is_tensor(x_static_scale) else x_static_scale),
        expt_hist, expt_offs, expt_block_pid_map,
        expt_offs_sum if expt_offs_sum is not None else _empty_i32,
        gather_indx.to(torch.int32).contiguous() if gather_indx is not None else _empty_i32,
        scatter_indx.to(torch.int32).contiguous() if scatter_indx is not None else _empty_i32,
        gammas.to(torch.float32).contiguous() if gammas is not None else _empty_f32,
        int(routing_data.n_expts_tot), n_expts_act,
        int(M), int(N), int(K),
        int(grid_m), int(grid_n),
        int(block_m), int(block_n), int(block_k),
        int(swizzle_flag),
    )
    if scatter_indx is not None and y_final is not None:
        # Mirror aiter's moe_gemm_a8w4: post-kernel grouped reduction.
        # `y` (shape [M_in, N]) holds the per-(expert, token) outputs with
        # gammas already applied in-kernel. `reduce_grouped` sums them per
        # destination token using `scatter_indx`.
        group_indx = scatter_indx.view(-1, int(routing_data.n_expts_act))
        # reduce_grouped expects a 3D input [split_k, M, N]; we ran split_k=1.
        y3 = y.unsqueeze(0)
        y_final = reduce_grouped(
            y3, group_indx, y_final,
            apply_swiglu=False, alpha=1.0, limit=1.0,
            reduction_n=1, out_dtype=out_dtype, add_residual=False,
        )
        return y_final
    return y
