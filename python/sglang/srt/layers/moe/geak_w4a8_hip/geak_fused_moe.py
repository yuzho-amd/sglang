"""GEAK fused MoE (W4A8 HIP) - mirrors aiter.fused_moe_2stages orchestration
with the two MoE GEMMs replaced by our optimized HIP kernel.

ORCHESTRATION MAP (vs. aiter.fused_moe.fused_moe_2stages, fused_moe.py:1141)
---------------------------------------------------------------------------
aiter's fused_moe_2stages does, in order:
  (a) PRE-REORDER / QUANT
        * moe_sorting(topk_ids, ...) -> sorted_ids, sorted_weights,
          sorted_expert_ids, num_valid_ids, moe_buf  (called inside
          fused_moe_ before fused_moe_2stages, see lines 298-308)
        * activation quant: either fused_dynamic_mxfp4_quant_moe_sort
          (per_1x32+bf16-A) OR hidden_states.to(fp8) +
          a1_scale=ones-or-empty (per_1x32+fp8-A, lines 1205-1218) OR
          quant_func (other quant_type)
  (b) STAGE-1 GEMM  (metadata.stage1)
        - dispatches to one of asm_stage1 / ck_moe_stage1 /
          cktile_moe_stage1 (chosen by get_2stage_cfgs)
        - input  : a1 (pre-quantized activation, MoE-sorted)
                   + w1 (E, 2*inter, model_dim)
                   + sorted_ids / sorted_expert_ids
                   + scales a1_scale, w1_scale
        - output : a2  (token_num, topk, 2*inter)  bf16
        - this is the FIRST MoE GEMM (gate_up_proj)
  (c) SiLU/SwiGLU + ACTIVATION QUANT
        - for per_1x32+fp8-A+swiglu+bf16-out: aiter writes the
          (bf16 SwiGLU) result into a2 then casts to fp8 with the
          inherited per-tensor a1_scale (lines 1320-1328)
        - for per_1x32+mxfp4-A: fused_dynamic_mxfp4_quant_moe_sort
        - for other types: quant_func
  (d) STAGE-2 GEMM  (metadata.stage2)
        - dispatches to ck_moe_stage2_fwd in our case
        - input  : a2 (post-act, quantized) + w2 + scales
        - output : moe_buf (token_num, model_dim) bf16
        - this is the SECOND MoE GEMM (down_proj)
  (e) POST-REORDER / REDUCE
        - the stage-2 kernel internally scatter-accumulates topk rows
          weighted by sorted_weights into moe_buf. No separate reduce
          call.

GEAK substitution:
  * (a), (c) and (e) are reimplemented here using simple torch ops +
    aiter helpers (downcast_to_static_fp8, swiglu). Behavior matches
    aiter's torch reference (torch_moe_stage1/stage2) so parity vs.
    aiter.fused_moe is held to the looser fp8/mxfp4 tolerance.
  * (b) and (d) call moe_gemm_a8w4_hip - our HIP kernel. It expects
    expert-sorted x of shape (M*topk, K) in FP8 e4m3 with a per-tensor
    static scale and MXFP4 weights stored as (E, K/2, N) uint8 + e8m0
    (E, K/32, N) uint8 scales.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# aiter helpers (reused from aiter; we do NOT modify aiter)
from aiter.ops.triton.moe.moe_routing.routing import RoutingData, ExptData

# Our HIP kernel (package-local import)
from sglang.srt.layers.moe.geak_w4a8_hip.moe_gemm_a8w4_hip_wrapper import (
    moe_gemm_a8w4_hip,
)

# aiter's fused FP8 static-scale downcast (single triton kernel, ~2x faster
# than the pure-torch (x*inv).clamp.to(fp8) chain we used before).
try:
    from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8 as _aiter_downcast_to_static_fp8
except ImportError:  # pragma: no cover
    _aiter_downcast_to_static_fp8 = None

# Per-device cached expert-id arange (used to expand bias[e] to per-row).
# Indexed lazily by device.index; arange itself is tiny but the redundant
# allocation per call shows up at ~5us each.
_expt_arange_cache: dict[tuple[str, int, int], torch.Tensor] = {}


def _get_expt_arange(E: int, device: torch.device) -> torch.Tensor:
    key = (device.type, -1 if device.index is None else device.index, E)
    t = _expt_arange_cache.get(key)
    if t is None:
        t = torch.arange(E, device=device, dtype=torch.long)
        _expt_arange_cache[key] = t
    return t


# --------------------------------------------------------------------------
# Weight layout helpers
# --------------------------------------------------------------------------
#
# sglang stores MXFP4 weights in (E, N, K/2) layout (last dim = packed K).
# Our HIP kernel expects (E, K/2, N) with stride(-2)==1. A simple
# transpose(-2,-1) achieves the right strides WITHOUT a copy.

# NOTE on caching: transpose(-2, -1) is a metadata-only no-copy
# operation, so caching the resulting view is not a perf win and can
# introduce hard-to-debug stale-pointer bugs when the underlying torch
# CUDA caching allocator reuses freed memory (a different tensor can
# end up with the same data_ptr). We just transpose on every call.


def _as_kn_layout(w: torch.Tensor) -> torch.Tensor:
    """Return a (E, K/2, N) view of an (E, N, K/2) uint8 MXFP4 tensor.

    No data is copied; the return value just has stride(-2)==1, which is
    what our HIP kernel checks for.
    """
    v = w.transpose(-2, -1)
    assert v.stride(-2) == 1, (
        f"unsupported weight layout: expected (E, N, K/2) row-major, "
        f"got shape {w.shape} stride {w.stride()}"
    )
    return v


def _scales_to_kn_layout(s: torch.Tensor) -> torch.Tensor:
    """Same idea for the (E, N, K/32) e8m0 scale tensor."""
    v = s.transpose(-2, -1)
    assert v.stride(-2) == 1, (
        f"unsupported scale layout: expected (E, N, K/32) row-major, "
        f"got shape {s.shape} stride {s.stride()}"
    )
    return v


# --------------------------------------------------------------------------
# Routing helper
# --------------------------------------------------------------------------

def _build_routing_data(
    topk_ids: torch.Tensor,
    n_expts_tot: int,
    n_expts_act: int,
    block_m: int = 32,
) -> tuple[RoutingData, torch.Tensor]:
    """Build aiter's RoutingData (+ a gather permutation) from topk_ids.

    Args:
      topk_ids: (M, topk) int32  expert assignments
      n_expts_tot, n_expts_act: total/active experts
      block_m: must be >= 32 (HIP kernel constraint)

    Returns:
      RoutingData ready to pass to moe_gemm_a8w4_hip, and a 1-D
      permutation `gather_indx` of length M*topk such that
      x_sorted[j] == hidden[gather_indx[j] // topk]  (expert-sorted).
    """
    assert block_m >= 32, f"HIP kernel requires block_m>=32, got {block_m}"
    device = topk_ids.device
    flat_e = topk_ids.to(torch.int32).reshape(-1)
    # Stable sort -> expert-sorted permutation
    sorted_e, perm = torch.sort(flat_e, stable=True)
    gather_indx = perm.to(torch.int32)

    hist = torch.bincount(flat_e.long(), minlength=n_expts_tot).to(torch.int32)
    token_offs_raw = torch.zeros(n_expts_tot + 1, dtype=torch.int32, device=device)
    torch.cumsum(hist, dim=0, out=token_offs_raw[1:])

    n_tiles = (hist + block_m - 1) // block_m
    token_offs_pad = torch.zeros(n_expts_tot + 1, dtype=torch.int32, device=device)
    torch.cumsum(n_tiles, dim=0, out=token_offs_pad[1:])

    # Block-pid map sizing: must match what aiter's
    # RoutingData.n_blocks(M*topk, block_m) returns (this is the worst-
    # case grid_m the kernel will dispatch). The closed form is
    # taken from aiter.compute_expt_data_torch. We size the buffer to
    # an upper bound that does NOT require a D->H sync; entries beyond
    # the truly used count stay -1 and the kernel treats them as
    # padding.
    n_gates = int(flat_e.numel())
    if n_gates <= n_expts_tot:
        worst_grid_m = n_gates
    else:
        worst_grid_m = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // block_m)
    # Upper bound on filled tiles (no sync). In the worst case every
    # expert gets ceil(n_gates / block_m / n_expts_tot) + 1 tiles, so
    # n_expts_tot * (...) entries. Cheaper: just worst_grid_m + n_expts_tot.
    bpm_size = max(worst_grid_m + n_expts_tot, 1)

    # Vectorized device-side fill of block_pid_map. For every slot pid
    # we look up which (expert, intra-expert-block) it corresponds to
    # using `token_offs_pad` (a small prefix-sum), and write the packed
    # (b << 16) | e value. Slots past the actually-used count get -1
    # via the `valid` mask.
    pos = torch.arange(bpm_size, dtype=torch.int32, device=device)
    e_idx = (
        torch.searchsorted(token_offs_pad, pos, right=True) - 1
    ).clamp(0, n_expts_tot - 1)
    b_idx = pos - token_offs_pad[e_idx]
    valid = b_idx < n_tiles[e_idx]
    block_pid_map = torch.where(
        valid,
        (b_idx << 16) | e_idx,
        torch.full_like(b_idx, -1),
    ).to(torch.int32)

    expt_data = ExptData(
        hist=hist,
        token_offs_raw=token_offs_raw,
        token_offs_pad=token_offs_pad,
        block_pid_map=block_pid_map,
    )
    rdata = RoutingData(
        block_m=block_m,
        gate_scal=None,  # not used by our HIP kernel
        expt_hist=hist,
        n_expts_tot=n_expts_tot,
        n_expts_act=n_expts_act,
        expt_data=expt_data,
    )
    return rdata, gather_indx


# --------------------------------------------------------------------------
# FP8 per-tensor static quant
# --------------------------------------------------------------------------

_FP8_E4M3_MAX = 448.0


def _to_fp8_static(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x (bf16/fp16) -> fp8 e4m3 by dividing by `scale` (per-tensor).

    Uses aiter's fused triton `downcast_to_static_fp8` when available — a
    single kernel vs the pure-torch (x*inv).clamp.to(fp8) chain (5
    kernels). Drops ~17us/call at decode shape; with 2 calls per layer
    and 40 layers, that's ~1.4ms saved per token step (~5% TPOT win
    by itself).
    """
    if _aiter_downcast_to_static_fp8 is not None and x.ndim == 2 and x.is_contiguous():
        return _aiter_downcast_to_static_fp8(x, scale)
    inv = (1.0 / scale.float()).to(x.device)
    y = (x.float() * inv).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return y.to(torch.float8_e4m3fn)


# --------------------------------------------------------------------------
# SwiGLU - matches aiter.fused_moe.swiglu
# --------------------------------------------------------------------------

def _swiglu_half_split(x_2n: torch.Tensor,
                       alpha: float = 1.702,
                       limit: float = 7.0) -> torch.Tensor:
    """SwiGLU with [gate; up] half-split along last dim, matching
    aiter.fused_moe.swiglu (bias 1 on up, alpha=1.702, clamp +-7).

    Compute in bf16 directly (no float() upcast) to avoid 2 extra D->D
    cast kernels per call. The clamp range +-7 is well within bf16's
    representable mantissa for sigmoid input, so numerical drift is
    bounded by ~1 ULP which is fine for the FP8 downcast immediately
    after.
    """
    inter = x_2n.shape[-1] // 2
    gate, up = x_2n[..., :inter], x_2n[..., inter:]
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    out_glu = gate * torch.sigmoid(alpha * gate)
    return out_glu * (up + 1.0)


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def geak_fused_moe(
    hidden_states: torch.Tensor,           # (M, K)  bf16
    w1: torch.Tensor,                       # (E, 2*inter, K/2) uint8 (MXFP4)
    w2: torch.Tensor,                       # (E, K, inter/2)   uint8 (MXFP4)
    topk_weight: torch.Tensor,              # (M, topk)  float32
    topk_ids: torch.Tensor,                 # (M, topk)  int32
    quant_type=None,                        # ignored - W4A8 fixed
    w1_scale: Optional[torch.Tensor] = None,  # (E, 2*inter, K/32)  uint8 e8m0
    w2_scale: Optional[torch.Tensor] = None,  # (E, K,       inter/32) uint8 e8m0
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,   # (E, 2*inter) bf16 - optional
    bias2: Optional[torch.Tensor] = None,   # (E, K)       bf16 - optional
    activation: str = "swiglu",
    expert_mask: Optional[torch.Tensor] = None,
    doweight_stage1: bool = False,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
) -> torch.Tensor:
    """Drop-in replacement for aiter.fused_moe in the W4A8 (FP8-A x
    MXFP4-W) + SwiGLU + per-1x32 e8m0 microscale path.

    Returns: (M, K) bf16 final MoE output.
    """
    assert activation.lower() in ("swiglu", "silu"), (
        f"only swiglu/silu activation supported, got {activation}"
    )
    assert expert_mask is None, "EP mask not yet plumbed in GEAK W4A8 HIP path"
    assert hidden_pad == 0 and intermediate_pad == 0, (
        "padded hidden_dim/intermediate_dim not supported"
    )
    assert hidden_states.dtype in (torch.bfloat16, torch.float16)
    M, K = hidden_states.shape
    topk = topk_ids.shape[-1]
    device = hidden_states.device
    out_dtype = hidden_states.dtype

    # ---- Weight layout massage (cached) ----------------------------------
    # sglang stores (E, N, K/2); HIP kernel wants (E, K/2, N) with
    # stride(-2)==1. transpose(-2,-1) gives that without a copy.
    w1_kn = _as_kn_layout(w1)            # (E, K/2, 2*inter)
    w2_kn = _as_kn_layout(w2)            # (E, inter/2, K)
    w1_scale_kn = _scales_to_kn_layout(w1_scale)  # (E, K/32, 2*inter)
    w2_scale_kn = _scales_to_kn_layout(w2_scale)  # (E, inter/32, K)

    E = w1.shape[0]
    twoN = w1.shape[1]
    inter = twoN // 2

    # ---- (a) Pre-reorder + activation FP8 quant --------------------------
    # Per-tensor static scale: use a1_scale if the caller supplied a
    # scalar; otherwise derive from |x|.max / 448. This matches the unit
    # test convention in workspace/test_moe_gemm_a8w4_hip.py.
    if a1_scale is not None and a1_scale.numel() == 1:
        x_scale1 = a1_scale.float().view(1).to(device)
    else:
        x_scale1 = (hidden_states.abs().max().float() / _FP8_E4M3_MAX).clamp_min(1e-12).view(1)

    # Build routing data + gather permutation.
    rdata, gather_indx = _build_routing_data(
        topk_ids, n_expts_tot=E, n_expts_act=topk, block_m=32
    )

    # Expand hidden -> (M*topk, K) expert-sorted FP8.
    src_token_idx = gather_indx.long() // topk
    hidden_sorted_bf = hidden_states.index_select(0, src_token_idx)  # (M*topk, K)
    x1_q = _to_fp8_static(hidden_sorted_bf, x_scale1)                # (M*topk, K) fp8

    # ---- (b) Stage-1 GEMM: x1_q @ w1 -> y1 (M*topk, 2*inter) -------------
    y1 = moe_gemm_a8w4_hip(
        x1_q, w1_kn, w1_scale_kn, x_scale1, rdata, out_dtype=out_dtype,
    )  # (M*topk, 2*inter) bf16

    # Optional bias1 (per-expert, per-channel). We need to add bias[e]
    # to rows assigned to expert e. expert id for sorted row j is given
    # by the routing histogram.
    #
    # `row_eid` only depends on the histogram, so compute once and reuse
    # for stage2 (saves ~70us at decode). arange(E) is module-cached.
    row_eid = None
    if bias1 is not None or bias2 is not None:
        row_eid = torch.repeat_interleave(
            _get_expt_arange(E, device),
            rdata.expt_data.hist.long(),
        )
    if bias1 is not None:
        y1 = y1 + bias1.to(out_dtype)[row_eid]

    # ---- (c) SwiGLU + activation FP8 quant -------------------------------
    # Skip the .float() upcast — bf16 is good enough for the clamp(+-7)
    # range and the FP8 downcast right after.
    if activation.lower() == "swiglu":
        y1_act = _swiglu_half_split(y1)                              # (M*topk, inter)
    else:
        gate, up = y1[..., :inter], y1[..., inter:]
        y1_act = torch.nn.functional.silu(gate) * up

    if a2_scale is not None and a2_scale.numel() == 1:
        x_scale2 = a2_scale.float().view(1).to(device)
    else:
        x_scale2 = (y1_act.abs().max().float() / _FP8_E4M3_MAX).clamp_min(1e-12).view(1)
    x2_q = _to_fp8_static(y1_act, x_scale2)  # (M*topk, inter) fp8

    # ---- (d) Stage-2 GEMM: x2_q @ w2 -> y2 (M*topk, K) -------------------
    y2 = moe_gemm_a8w4_hip(
        x2_q, w2_kn, w2_scale_kn, x_scale2, rdata, out_dtype=out_dtype,
    )  # (M*topk, K) bf16

    if bias2 is not None:
        # row_eid already built above if either bias is present.
        y2 = y2 + bias2.to(out_dtype)[row_eid]

    # ---- (e) Scatter + weighted reduce -----------------------------------
    # Scatter expert-sorted rows back to (M, topk, K), multiply by
    # routing weights, and sum over topk.
    y2_flat = torch.empty(
        (M * topk, K), dtype=out_dtype, device=device
    )
    y2_flat[gather_indx.long()] = y2
    y2_topk = y2_flat.view(M, topk, K)
    weights = topk_weight.to(y2_topk.dtype).view(M, topk, 1)
    out = (y2_topk * weights).sum(dim=1)
    return out
