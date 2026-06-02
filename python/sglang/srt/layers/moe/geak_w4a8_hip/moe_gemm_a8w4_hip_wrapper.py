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


# Per-device cached empty placeholder tensors. The launch_moe_gemm_a8w4 ABI
# wants a real tensor for "feature off" slots (gather_indx, scatter_indx,
# gammas, expt_offs_sum) so we used to allocate `torch.empty((0,), ...)`
# per call. Allocating 4x per-call (one per layer = 320 allocs / token at 40
# layers, 2 GEMMs each) shows up at ~3us each in profilers. Hoist them.
_EMPTY_I32_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_EMPTY_F32_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _empty_placeholders(device: torch.device):
    key = (device.type, -1 if device.index is None else device.index)
    i32 = _EMPTY_I32_CACHE.get(key)
    if i32 is None:
        i32 = torch.empty((0,), dtype=torch.int32, device=device)
        _EMPTY_I32_CACHE[key] = i32
    f32 = _EMPTY_F32_CACHE.get(key)
    if f32 is None:
        f32 = torch.empty((0,), dtype=torch.float32, device=device)
        _EMPTY_F32_CACHE[key] = f32
    return i32, f32


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
    weight_is_shuffled: bool = False,
    gate_up_shuffled: bool = True,
):
    """W4A8 MoE GEMM (HIP).

    Matches the aiter-triton ``moe_gemm_a8w4`` for the
    QuarkW4A8Fp8MoE call subset:
      - FP8 e4m3 activation + per-tensor static scale
      - MXFP4 e2m1 weight + per-1x32 e8m0 scale
      - optional gather_indx / scatter_indx / gammas / swizzle_mx_scale

    Optional shuffled-weight fast path (opt6):
      - weight_is_shuffled=True asks the kernel to read W (+ w_scales) using
        aiter's `shuffle_weight_a16w4` / `shuffle_scale_a16w4` byte layout.
        Coalesces wave loads (~16x fewer cache-line transactions).
      - gate_up_shuffled selects between the two shuffle variants in
        aiter.ops.shuffle (gate_up=True is the stage-1 W13 layout that
        interleaves gate+up rows; False is the plain W2 layout).
      - Shape of `w` / `w_scales` is unchanged (shuffle is in-place byte
        permutation). The kernel knows it must reverse the permutation
        when reading.
    """
    assert x.dtype == torch.float8_e4m3fn
    assert w.dtype == torch.uint8 and w_scales.dtype == torch.uint8
    if not weight_is_shuffled:
        assert w.stride(-2) == 1, "w must be K-fast (stride(-2)==1) on the unshuffled path"
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
    # geak_fused_moe's _build_routing_data already produces int32 contiguous
    # tensors via torch.bincount (returns int32 when input is int32) and
    # torch.cumsum + arange. Avoid redundant .to(int32).contiguous() — they
    # are no-ops for the right dtype but still launch a kernel each.
    expt_hist = expt_data.hist
    if expt_hist.dtype != torch.int32 or not expt_hist.is_contiguous():
        expt_hist = expt_hist.to(torch.int32).contiguous()
    expt_offs = expt_data.token_offs_raw
    if expt_offs.dtype != torch.int32 or not expt_offs.is_contiguous():
        expt_offs = expt_offs.to(torch.int32).contiguous()
    expt_block_pid_map = expt_data.block_pid_map
    if expt_block_pid_map.dtype != torch.int32 or not expt_block_pid_map.is_contiguous():
        expt_block_pid_map = expt_block_pid_map.to(torch.int32).contiguous()
    expt_offs_sum = (
        None
        if expt_data.token_offs_pad is None
        else expt_data.token_offs_pad[-1:].to(torch.int32).contiguous()
    )

    # Optional tensors passed to the kernel; use cached empty placeholders
    # when None so the C++ side doesn't need to handle null Tensors and we
    # avoid per-call zero-byte allocations.
    _empty_i32, _empty_f32 = _empty_placeholders(x.device)

    block_m, block_n, block_k = config["block_m"], config["block_n"], config["block_k"]
    grid_m = routing_data.n_blocks(M, block_m)
    grid_n = triton.cdiv(N, block_n)

    swizzle_flag = 1 if swizzle_mx_scale == "CDNA4_SCALE" else 0
    n_expts_act = int(routing_data.n_expts_act)
    # weight_layout_flag: 0=unshuffled (default, K-fast), 1=aiter shuffle
    # gate_up=True (W13), 2=aiter shuffle gate_up=False (W2).
    if not weight_is_shuffled:
        weight_layout_flag = 0
    elif gate_up_shuffled:
        weight_layout_flag = 1
    else:
        weight_layout_flag = 2

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
        int(weight_layout_flag),
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
