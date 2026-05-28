// HIP MoE A8W4 GEMM (scaled-MFMA, K=128 per call).
//
// Activation: FP8 e4m3 OCP + per-tensor static scalar scale.
// Weight:     MXFP4 e2m1 packed uint8 (low nibble = even K, high nibble = odd K)
//             + e8m0 microscale per 32 K-elements.
// Output:     bf16.
//
// gfx950 scaled MFMA __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
// with A = FP8 E4M3 (cbsz=0), B = FP4 E2M1 (blgp=4).
//
// EMPIRICALLY-DETERMINED PER-LANE K LAYOUT (cbsz=0, blgp=4):
//   Wave of 64 lanes, lane_mn = lane % 16, lane_kg = lane / 16.
//
//   A (32 bytes per lane):
//     bytes 0..15  -> A[row=lane_mn, K = lane_kg*16 + [0,16)]
//     bytes 16..31 -> A[row=lane_mn, K = lane_kg*16 + 64 + [0,16)]
//
//   B (16 bytes per lane, padded to 32 with zeros):
//     bytes 0..7   -> B[K = lane_kg*32 + [0,16), col=lane_mn]  (2 fp4 per byte)
//     bytes 8..15  -> B[K = lane_kg*32 + [16,32), col=lane_mn]
//
//   Scale (one byte per A operand, one byte per B operand, per lane):
//     scale_a = byte 0 of the int32 (e8m0 exponent; 127 = 2^0 = 1.0)
//     scale_b = e8m0 byte for the 32-K block (lane covers exactly one 32-K block for B)
//
//   C output (4 fp32 per lane):
//     C[row = (lane/16)*4 + i, col = lane%16] = c[i]   (i in 0..3)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdint>

using int32x2_t  = int32_t __attribute__((ext_vector_type(2)));
using int32x4_t  = int32_t __attribute__((ext_vector_type(4)));
using int32x8_t  = int32_t __attribute__((ext_vector_type(8)));
using float4_t   = float   __attribute__((ext_vector_type(4)));
using float16v_t = float   __attribute__((ext_vector_type(16)));

__device__ __forceinline__
float4_t mfma_scale_f8_fp4(int32x8_t a, int32x8_t b, float4_t c,
                           int scale_a, int scale_b) {
#if defined(__gfx950__)
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c,
        /*cbsz*/0,    /*blgp*/4,
        /*opsel_a*/0, scale_a,
        /*opsel_b*/0, scale_b);
#else
    (void)a; (void)b; (void)scale_a; (void)scale_b;
    return c;
#endif
}

__device__ __forceinline__
float16v_t mfma_scale_f8_fp4_32x32(int32x8_t a, int32x8_t b, float16v_t c,
                                   int scale_a, int scale_b) {
#if defined(__gfx950__)
    return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a, b, c,
        /*cbsz*/0,    /*blgp*/4,
        /*opsel_a*/0, scale_a,
        /*opsel_b*/0, scale_b);
#else
    (void)a; (void)b; (void)scale_a; (void)scale_b;
    return c;
#endif
}

// XCD-swizzle: MI355X has 8 XCDs. Map blockIdx.x so consecutive pids stay
// on the same XCD (better L2 locality across (pid_m,pid_n) tiles).
template <int XCD>
__device__ __forceinline__
int xcd_swizzle_pid(int pid, int total_pid) {
    const int per_group = total_pid / XCD;
    const int extra     = total_pid - per_group * XCD;   // total_pid % XCD
    const int group     = pid - (pid / XCD) * XCD;       // pid % XCD
    const int local     = pid / XCD;
    const int min_ge    = (group < extra) ? group : extra;
    return group * per_group + min_ge + local;
}

__device__ __forceinline__
int32x4_t load_x4(const uint8_t* p) {
    return *reinterpret_cast<const int32x4_t*>(p);
}

// ---------------------------------------------------------------------------
// CDNA4 scale unswizzle helper.
//
// aiter pre-shuffles the W scale tensor by:
//   x = x.reshape(N//32, K//8, 4, 16, 2, 2, 1)
//   x = x.permute(0, 5, 3, 1, 4, 2, 6)    # within per-N-block
//   x = x.reshape(N, K)
//
// In the swizzled layout (shape `(E, K_scale*32, N/32)` after the trailing
// transpose), each 32-wide N-block has its `(SCALE_K * 32)` bytes laid out as
//   (sk_outer, c, d, e, f, g)  with strides (256, 64, 4, 2, 1, 1)
// so to access the unswizzled element (n, k_block) within that block:
//   sk_outer = k_block / 8
//   c        = k_block % 4
//   d        = n_inner % 16
//   e        = (k_block % 8) / 4
//   f        = n_inner / 16
//
// where n_block_idx = n / 32, n_inner = n % 32.
__device__ __forceinline__
int cdna4_swizzle_offset_in_block(int n_inner, int k_block) {
    const int sk_outer = k_block >> 3;
    const int c        = k_block & 3;
    const int d        = n_inner & 15;
    const int e        = (k_block >> 2) & 1;
    const int f        = n_inner >> 4;
    return sk_outer * 256 + c * 64 + d * 4 + e * 2 + f;
}

// Load one e8m0 scale byte (unswizzled view) for column n_col and k-block.
//   WSexp     = WScale + expt_id * stride_ws_e
//   stride_ws_n is the stride of the N-dimension of the swizzled tensor
//                (== K_scale * 32 in the typical layout).
__device__ __forceinline__
uint8_t load_scale_swizzled(const uint8_t* WSexp, int n_col, int k_block,
                            int64_t stride_ws_n) {
    const int n_block = n_col >> 5;
    const int n_inner = n_col & 31;
    const int64_t base = (int64_t)n_block * stride_ws_n;
    return WSexp[base + cdna4_swizzle_offset_in_block(n_inner, k_block)];
}
__device__ __forceinline__
int32x4_t load_x4_safe(const uint8_t* p, int valid_bytes) {
    if (valid_bytes >= 16) return *reinterpret_cast<const int32x4_t*>(p);
    uint8_t tmp[16] = {0};
    for (int i = 0; i < valid_bytes; ++i) tmp[i] = p[i];
    return *reinterpret_cast<const int32x4_t*>(tmp);
}

// ---------------------------------------------------------------------------
// aiter shuffle_weight_a16w4 layout (NLane=16, KLane=4, KPack=16).
//
//   gate_up=False (W2):  flat order [N0, K0, KLane, NLane, KPack]
//     off(n,k_pk) = n0*16*K_pk + k0*1024 + k_lane*256 + n_lane*16 + k_pack
//                   where n0=n>>4, n_lane=n&15,
//                         k0=k_pk>>6, k_lane=(k_pk>>4)&3, k_pack=k_pk&15
//
//   gate_up=True (W13): flat order [N0, 2(gu), K0, KLane, NLane, KPack]
//     n_full splits as (gu, n_half) with N_half = N_real/2
//     off(n,k_pk) = n0*32*K_pk + gu*16*K_pk + k0*1024 + k_lane*256 + n_lane*16 + k_pack
//                   where n0=n_half>>4, n_lane=n_half&15
//
// WLAYOUT: 1 = gate_up (W13), 2 = plain (W2).
// k_pk is required to be a multiple of KPack=16 (16-byte chunk start).
template <int WLAYOUT>
__device__ __forceinline__
int64_t shuffled_w_off(int n_col, int k_pk, int N_real, int K_pk) {
    const int k0     = k_pk >> 6;        // KLane*KPack = 64
    const int k_lane = (k_pk >> 4) & 3;
    // k_pack assumed 0 (chunk-aligned)
    if constexpr (WLAYOUT == 1) {
        const int N_half = N_real >> 1;
        const int gu     = (n_col >= N_half) ? 1 : 0;
        const int n_half = n_col - gu * N_half;
        const int n0     = n_half >> 4;
        const int n_lane = n_half & 15;
        return (int64_t)n0 * (int64_t)(32 * K_pk)
             + (int64_t)gu * (int64_t)(16 * K_pk)
             + (int64_t)k0 * 1024
             + (int64_t)k_lane * 256
             + (int64_t)n_lane * 16;
    } else {
        const int n0     = n_col >> 4;
        const int n_lane = n_col & 15;
        return (int64_t)n0 * (int64_t)(16 * K_pk)
             + (int64_t)k0 * 1024
             + (int64_t)k_lane * 256
             + (int64_t)n_lane * 16;
    }
}

// aiter shuffle_scale_a16w4 layout for e8m0 scale tensor.
//   per-expert layout: [N1, K1, K_Lane=4, N_Lane=16, K_Pack=2, N_Pack=2]
//     stride(N_Pack)=1, stride(K_Pack)=2, stride(N_Lane)=4, stride(K_Lane)=64,
//     stride(K1)=256, stride(N1)=32*K_groups
//
//   gate_up=True: n_full=(gu, n_half=(n1,n_lane)), n_pack=gu
//   gate_up=False: n_full=(n1, n_pack, n_lane) where n_pack=(n>>4)&1, n1=n>>5
//
//   k_scale -> (k1=k_scale>>3, k_pack=(k_scale>>2)&1, k_lane=k_scale&3)
template <int WLAYOUT>
__device__ __forceinline__
int64_t shuffled_s_off(int n_col, int k_scale, int N_real, int K_groups) {
    int n_pack, n1, n_lane;
    if constexpr (WLAYOUT == 1) {
        const int N_half = N_real >> 1;
        const int gu     = (n_col >= N_half) ? 1 : 0;
        const int n_half = n_col - gu * N_half;
        n_pack = gu;
        n1     = n_half >> 4;
        n_lane = n_half & 15;
    } else {
        n1     = n_col >> 5;
        n_pack = (n_col >> 4) & 1;
        n_lane = n_col & 15;
    }
    const int k_lane = k_scale & 3;
    const int k_pack = (k_scale >> 2) & 1;
    const int k1     = k_scale >> 3;
    return (int64_t)n1 * (int64_t)(32 * K_groups)
         + (int64_t)k1 * 256
         + (int64_t)k_lane * 64
         + (int64_t)n_lane * 4
         + (int64_t)k_pack * 2
         + (int64_t)n_pack;
}

// --------------------------------------------------------------------------
// 32x32x64 MFMA kernel for BLOCK_M == 32 (uses the 32x32x64 scaled MFMA).
//
// Per-lane operand layout (cbsz=0 fp8 A, blgp=4 fp4 B), wave of 64 lanes:
//   lane_mn = lane % 32, lane_kg = lane / 32  (kg in {0,1})
//   A (32 fp8 bytes per lane, split into 2 banks like the 16x16x128 layout):
//     bytes[0..15]  -> A[row = lane_mn, K = lane_kg*16 + [0,16)]      (bank 0 = K[0,32))
//     bytes[16..31] -> A[row = lane_mn, K = lane_kg*16 + 32 + [0,16)] (bank 1 = K[32,64))
//   B (16 packed bytes per lane = 32 fp4): B[K = lane_kg*32 + [0,32), col = lane_mn]
//   Scale: 1 e8m0 byte per A operand per lane and per B operand per lane
//          (each lane covers exactly one 32-K microscale block; B uses scale at
//           k_block = (K_base >> 5) + lane_kg).
//   C output: 16 fp32 per lane laid out as 4 groups of 4 rows each:
//      c[g*4 + i] -> row = g*8 + lane_kg*4 + i, col = lane_mn   (g in 0..3, i in 0..3)
//
// Layout per CTA: TPB = 128 = 2 waves. Each wave covers the FULL BLOCK_M=32
// rows and half of BLOCK_N. With BLOCK_N=128 each wave handles 64 N cols =
// NN_W = 2 32-wide N tiles per K iter -> only 2 MFMA calls per K iter per
// wave (versus 8 16x16x128 calls in the legacy layout).
// --------------------------------------------------------------------------
template <int BLOCK_N, int N_WAVES = 2,
          bool HAS_GATHER = false, bool HAS_GAMMAS = false, bool SWIZZLE_SCALE = false,
          int WEIGHT_LAYOUT = 0>
__global__ __launch_bounds__(N_WAVES * 64)
void moe_gemm_a8w4_mfma32_kernel(
    __hip_bfloat16* __restrict__ Y,
    const uint8_t* __restrict__ X,
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ WScale,
    float x_static_scale,
    const int32_t* __restrict__ ExptHist,
    const int32_t* __restrict__ ExptOffs,
    const int32_t* __restrict__ ExptBlockPidMap,
    const int32_t* __restrict__ GatherIndx,
    const float*   __restrict__ Gammas,
    int n_expts_act,
    int M, int N, int K,
    int64_t stride_x_m,
    int64_t stride_w_e, int64_t stride_w_n,
    int64_t stride_ws_e, int64_t stride_ws_n,
    int64_t stride_y_m, int64_t stride_y_n,
    int grid_n)
{
    constexpr int BLOCK_M = 32;
    static_assert(BLOCK_N % 32 == 0, "BLOCK_N must be multiple of 32 for the 32x32 path");
    static_assert((BLOCK_N / 32) % N_WAVES == 0, "BLOCK_N/32 must be multiple of N_WAVES");
    constexpr int NN_PER_WAVE = (BLOCK_N / N_WAVES) / 32;   // # 32-wide N-tiles per wave

    int pid = blockIdx.x;
    // XCD swizzle (MI355X has 8 XCDs). Re-map pids so consecutive linear pids
    // serve the same XCD's L2 partition.
    {
        constexpr int XCD = 8;
        const int total_pid = (int)(gridDim.x);
        if ((total_pid % XCD) == 0) {
            pid = xcd_swizzle_pid<XCD>(pid, total_pid);
        }
    }
    const int pid_m = pid / grid_n;
    const int pid_n = pid - pid_m * grid_n;

    const int32_t expt_data = ExptBlockPidMap[pid_m];
    const int expt_id  = expt_data & 0xFFFF;
    const int block_id = (expt_data >> 16) & 0xFFFF;
    if (expt_id == 0xFFFF) return;

    const int M_e = ExptHist[expt_id];
    if (M_e <= 0) return;
    const int start_m = ExptOffs[expt_id];

    const int tid = threadIdx.x;
    const int wave = tid >> 6;          // 0 or 1
    const int lane = tid & 63;
    const int lane_mn = lane & 31;      // 0..31
    const int lane_kg = lane >> 5;      // 0 or 1

    // M row that THIS lane provides for A
    const int local_m = block_id * BLOCK_M + lane_mn;
    const int clamped_local_m = (local_m < M_e) ? local_m : (M_e - 1);
    int x_row;
    if constexpr (HAS_GATHER) {
        // GatherIndx[start_m + offs_m] // n_expts_act
        int g = GatherIndx[start_m + clamped_local_m];
        x_row = g / n_expts_act;
    } else {
        x_row = start_m + clamped_local_m;
    }

    // N cols served by this wave: wave w -> [w*N_per_wave, (w+1)*N_per_wave)
    const int n_base = pid_n * BLOCK_N + wave * (BLOCK_N / N_WAVES);
    int n_cols[NN_PER_WAVE];
    bool n_valid[NN_PER_WAVE];
    #pragma unroll
    for (int t = 0; t < NN_PER_WAVE; ++t) {
        int nc = n_base + t * 32 + lane_mn;
        n_cols[t]  = (nc < N) ? nc : (N - 1);
        n_valid[t] = (nc < N);
    }

    const uint8_t* Wexp  = W      + (int64_t)expt_id * stride_w_e;
    const uint8_t* WSexp = WScale + (int64_t)expt_id * stride_ws_e;

    float16v_t acc[NN_PER_WAVE];
    #pragma unroll
    for (int t = 0; t < NN_PER_WAVE; ++t) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc[t][i] = 0.0f;
    }

    const uint8_t* X_lane_row = X + (int64_t)x_row * stride_x_m;

    const int K_pk     = K >> 1;
    const int K_groups = K >> 5;
    int64_t b_n_base[NN_PER_WAVE];
    int64_t s_n_base[NN_PER_WAVE];
    #pragma unroll
    for (int t = 0; t < NN_PER_WAVE; ++t) {
        if constexpr (WEIGHT_LAYOUT == 0) {
            b_n_base[t] = (int64_t)n_cols[t] * stride_w_n;
            s_n_base[t] = (int64_t)n_cols[t] * stride_ws_n;
        } else {
            // n-dependent part of shuffled offset (k-independent terms).
            // For W layout: n0*X*K_pk + gu*16*K_pk + n_lane*16
            // For S layout: n1*32*K_groups + n_lane*4 + n_pack
            int n_col = n_cols[t];
            if constexpr (WEIGHT_LAYOUT == 1) {
                const int N_half = N >> 1;
                const int gu     = (n_col >= N_half) ? 1 : 0;
                const int n_half = n_col - gu * N_half;
                const int n0     = n_half >> 4;
                const int n_lane = n_half & 15;
                b_n_base[t] = (int64_t)n0 * (int64_t)(32 * K_pk)
                            + (int64_t)gu * (int64_t)(16 * K_pk)
                            + (int64_t)n_lane * 16;
                const int n1   = n0;
                const int npk  = gu;
                s_n_base[t] = (int64_t)n1 * (int64_t)(32 * K_groups)
                            + (int64_t)n_lane * 4
                            + (int64_t)npk;
            } else {
                const int n0     = n_col >> 4;
                const int n_lane = n_col & 15;
                b_n_base[t] = (int64_t)n0 * (int64_t)(16 * K_pk)
                            + (int64_t)n_lane * 16;
                const int n1     = n_col >> 5;
                const int n_pack = (n_col >> 4) & 1;
                s_n_base[t] = (int64_t)n1 * (int64_t)(32 * K_groups)
                            + (int64_t)n_lane * 4
                            + (int64_t)n_pack;
            }
        }
    }

    const int num_k64 = (K + 63) / 64;
    // A layout per lane has two 16-byte banks: bank0 at K=lane_kg*16+[0,16),
    //                                          bank1 at K=lane_kg*16+32+[0,16).
    const int a_k0_bank0 = lane_kg * 16;
    const int a_k0_bank1 = lane_kg * 16 + 32;
    const int b_k0 = lane_kg * 32;            // K offset for B (contiguous 32 K)

    // Fast path: K divisible by 64.
    // Software pipeline: prefetch NEXT iter's A into a separate register so
    // its VMEM issues while current MFMA executes.  Only A is double-buffered
    // (B is large -> would blow up VGPR budget).
    if (K % 64 == 0) {
        const int scale_a = 127;

        auto load_a = [&](int K_base, int32x8_t& a_reg) {
            int32x4_t a0v = *reinterpret_cast<const int32x4_t*>(X_lane_row + K_base + a_k0_bank0);
            int32x4_t a1v = *reinterpret_cast<const int32x4_t*>(X_lane_row + K_base + a_k0_bank1);
            a_reg[0] = a0v[0]; a_reg[1] = a0v[1]; a_reg[2] = a0v[2]; a_reg[3] = a0v[3];
            a_reg[4] = a1v[0]; a_reg[5] = a1v[1]; a_reg[6] = a1v[2]; a_reg[7] = a1v[3];
        };

        int32x8_t a_cur, a_nxt;
        load_a(0, a_cur);

        #pragma unroll 2
        for (int kk = 0; kk < num_k64 - 1; ++kk) {
            const int K_base       = kk * 64;
            const int b_k          = K_base + b_k0;
            const int k_scale      = (K_base >> 5) + lane_kg;
            const int b_packed_off = b_k >> 1;
            // Shuffled-layout K decomposition (only used when WEIGHT_LAYOUT != 0)
            const int sh_k0     = b_packed_off >> 6;
            const int sh_klane  = (b_packed_off >> 4) & 3;
            const int sh_s_klane= k_scale & 3;
            const int sh_s_kpack= (k_scale >> 2) & 1;
            const int sh_s_k1   = k_scale >> 3;
            const int64_t sh_b_koff = (int64_t)sh_k0 * 1024 + (int64_t)sh_klane * 256;
            const int64_t sh_s_koff = (int64_t)sh_s_k1 * 256 + (int64_t)sh_s_klane * 64 + (int64_t)sh_s_kpack * 2;

            // Prefetch next A while we do current MFMAs
            load_a((kk + 1) * 64, a_nxt);

            #pragma unroll
            for (int t = 0; t < NN_PER_WAVE; ++t) {
                int32x4_t bv;
                int sb;
                if constexpr (WEIGHT_LAYOUT == 0) {
                    bv = *reinterpret_cast<const int32x4_t*>(
                                    Wexp + b_n_base[t] + b_packed_off);
                    if constexpr (SWIZZLE_SCALE) {
                        sb = (int)load_scale_swizzled(WSexp, n_cols[t], k_scale, stride_ws_n);
                    } else {
                        sb = (int)WSexp[s_n_base[t] + k_scale];
                    }
                } else {
                    bv = *reinterpret_cast<const int32x4_t*>(
                                    Wexp + b_n_base[t] + sh_b_koff);
                    sb = (int)WSexp[s_n_base[t] + sh_s_koff];
                }
                int32x8_t b_reg;
                b_reg[0] = bv[0]; b_reg[1] = bv[1]; b_reg[2] = bv[2]; b_reg[3] = bv[3];
                b_reg[4] = 0; b_reg[5] = 0; b_reg[6] = 0; b_reg[7] = 0;
                acc[t] = mfma_scale_f8_fp4_32x32(a_cur, b_reg, acc[t], scale_a, sb);
            }
            a_cur = a_nxt;
        }
        // Tail iter
        {
            const int K_base       = (num_k64 - 1) * 64;
            const int b_k          = K_base + b_k0;
            const int k_scale      = (K_base >> 5) + lane_kg;
            const int b_packed_off = b_k >> 1;
            const int sh_k0     = b_packed_off >> 6;
            const int sh_klane  = (b_packed_off >> 4) & 3;
            const int sh_s_klane= k_scale & 3;
            const int sh_s_kpack= (k_scale >> 2) & 1;
            const int sh_s_k1   = k_scale >> 3;
            const int64_t sh_b_koff = (int64_t)sh_k0 * 1024 + (int64_t)sh_klane * 256;
            const int64_t sh_s_koff = (int64_t)sh_s_k1 * 256 + (int64_t)sh_s_klane * 64 + (int64_t)sh_s_kpack * 2;
            #pragma unroll
            for (int t = 0; t < NN_PER_WAVE; ++t) {
                int32x4_t bv;
                int sb;
                if constexpr (WEIGHT_LAYOUT == 0) {
                    bv = *reinterpret_cast<const int32x4_t*>(
                                    Wexp + b_n_base[t] + b_packed_off);
                    if constexpr (SWIZZLE_SCALE) {
                        sb = (int)load_scale_swizzled(WSexp, n_cols[t], k_scale, stride_ws_n);
                    } else {
                        sb = (int)WSexp[s_n_base[t] + k_scale];
                    }
                } else {
                    bv = *reinterpret_cast<const int32x4_t*>(
                                    Wexp + b_n_base[t] + sh_b_koff);
                    sb = (int)WSexp[s_n_base[t] + sh_s_koff];
                }
                int32x8_t b_reg;
                b_reg[0] = bv[0]; b_reg[1] = bv[1]; b_reg[2] = bv[2]; b_reg[3] = bv[3];
                b_reg[4] = 0; b_reg[5] = 0; b_reg[6] = 0; b_reg[7] = 0;
                acc[t] = mfma_scale_f8_fp4_32x32(a_cur, b_reg, acc[t], scale_a, sb);
            }
        }
    } else {
        // Tail path: handle K not divisible by 64
        for (int kk = 0; kk < num_k64; ++kk) {
            const int K_base    = kk * 64;
            const int a_k_b0    = K_base + a_k0_bank0;
            const int a_k_b1    = K_base + a_k0_bank1;
            const int b_k       = K_base + b_k0;
            const int k_scale   = (K_base >> 5) + lane_kg;
            const int b_packed_off = b_k >> 1;

            int32x4_t a0v, a1v;
            if (a_k_b0 + 16 <= K) a0v = *reinterpret_cast<const int32x4_t*>(X_lane_row + a_k_b0);
            else { uint8_t tmp[16]={0}; int v=(a_k_b0<K)?(K-a_k_b0):0; if(v>16)v=16;
                   for(int i=0;i<v;++i) tmp[i]=X_lane_row[a_k_b0+i];
                   a0v = *reinterpret_cast<const int32x4_t*>(tmp); }
            if (a_k_b1 + 16 <= K) a1v = *reinterpret_cast<const int32x4_t*>(X_lane_row + a_k_b1);
            else { uint8_t tmp[16]={0}; int v=(a_k_b1<K)?(K-a_k_b1):0; if(v>16)v=16;
                   for(int i=0;i<v;++i) tmp[i]=X_lane_row[a_k_b1+i];
                   a1v = *reinterpret_cast<const int32x4_t*>(tmp); }
            int32x8_t a_reg;
            a_reg[0] = a0v[0]; a_reg[1] = a0v[1]; a_reg[2] = a0v[2]; a_reg[3] = a0v[3];
            a_reg[4] = a1v[0]; a_reg[5] = a1v[1]; a_reg[6] = a1v[2]; a_reg[7] = a1v[3];

            const int scale_a = 127;
            const bool b_full = (b_k + 32 <= K);
            const int sh_k0     = b_packed_off >> 6;
            const int sh_klane  = (b_packed_off >> 4) & 3;
            const int sh_s_klane= k_scale & 3;
            const int sh_s_kpack= (k_scale >> 2) & 1;
            const int sh_s_k1   = k_scale >> 3;
            const int64_t sh_b_koff = (int64_t)sh_k0 * 1024 + (int64_t)sh_klane * 256;
            const int64_t sh_s_koff = (int64_t)sh_s_k1 * 256 + (int64_t)sh_s_klane * 64 + (int64_t)sh_s_kpack * 2;
            #pragma unroll
            for (int t = 0; t < NN_PER_WAVE; ++t) {
                int32x8_t b_reg;
                int32x4_t bv;
                int64_t b_off = (WEIGHT_LAYOUT == 0)
                              ? (b_n_base[t] + b_packed_off)
                              : (b_n_base[t] + sh_b_koff);
                if (b_full) {
                    bv = *reinterpret_cast<const int32x4_t*>(Wexp + b_off);
                } else {
                    uint8_t tmp[16] = {0};
                    int valid_packed = ((b_k < K) ? ((K - b_k + 1) / 2) : 0);
                    if (valid_packed > 16) valid_packed = 16;
                    for (int i = 0; i < valid_packed; ++i) tmp[i] = Wexp[b_off + i];
                    bv = *reinterpret_cast<const int32x4_t*>(tmp);
                }
                b_reg[0] = bv[0]; b_reg[1] = bv[1]; b_reg[2] = bv[2]; b_reg[3] = bv[3];
                b_reg[4] = 0; b_reg[5] = 0; b_reg[6] = 0; b_reg[7] = 0;
                int scale_b;
                if (b_k < K) {
                    if constexpr (WEIGHT_LAYOUT == 0) {
                        if constexpr (SWIZZLE_SCALE) {
                            scale_b = (int)load_scale_swizzled(WSexp, n_cols[t], k_scale, stride_ws_n);
                        } else {
                            scale_b = (int)WSexp[s_n_base[t] + k_scale];
                        }
                    } else {
                        scale_b = (int)WSexp[s_n_base[t] + sh_s_koff];
                    }
                } else {
                    scale_b = 127;
                }
                acc[t] = mfma_scale_f8_fp4_32x32(a_reg, b_reg, acc[t], scale_a, scale_b);
            }
        }
    }

    // Epilogue: write 16 fp32 per lane.
    // c[g*4 + i] -> row = g*8 + lane_kg*4 + i, col = lane_mn  (g in 0..3, i in 0..3)
    const float sx = x_static_scale;
    #pragma unroll
    for (int t = 0; t < NN_PER_WAVE; ++t) {
        if (!n_valid[t]) continue;
        int n_g = n_cols[t];
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            int row_base = g * 8 + lane_kg * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int mm = row_base + i;
                int m_local = block_id * BLOCK_M + mm;
                if (m_local >= M_e) continue;
                int y_m = start_m + m_local;
                if (y_m >= M) continue;
                float out = acc[t][g * 4 + i] * sx;
                if constexpr (HAS_GAMMAS) {
                    out *= Gammas[y_m];
                }
                Y[(int64_t)y_m * stride_y_m + (int64_t)n_g * stride_y_n]
                    = __float2bfloat16(out);
            }
        }
    }
}

template <int BLOCK_M, int BLOCK_N,
          bool HAS_GATHER = false, bool HAS_GAMMAS = false, bool SWIZZLE_SCALE = false,
          int WEIGHT_LAYOUT = 0>
__global__ __launch_bounds__((BLOCK_M / 16) * 64)
void moe_gemm_a8w4_mfma_kernel(
    __hip_bfloat16* __restrict__ Y,
    const uint8_t* __restrict__ X,
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ WScale,
    float x_static_scale,
    const int32_t* __restrict__ ExptHist,
    const int32_t* __restrict__ ExptOffs,
    const int32_t* __restrict__ ExptBlockPidMap,
    const int32_t* __restrict__ GatherIndx,
    const float*   __restrict__ Gammas,
    int n_expts_act,
    int M, int N, int K,
    int64_t stride_x_m,
    int64_t stride_w_e, int64_t stride_w_n,
    int64_t stride_ws_e, int64_t stride_ws_n,
    int64_t stride_y_m, int64_t stride_y_n,
    int grid_n)
{
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be multiple of 16");
    static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be multiple of 16");
    constexpr int WAVES_M  = BLOCK_M / 16;
    constexpr int NN       = BLOCK_N / 16;

    int pid = blockIdx.x;
    {
        constexpr int XCD = 8;
        const int total_pid = (int)(gridDim.x);
        if ((total_pid % XCD) == 0) {
            pid = xcd_swizzle_pid<XCD>(pid, total_pid);
        }
    }
    const int pid_m = pid / grid_n;
    const int pid_n = pid - pid_m * grid_n;

    const int32_t expt_data = ExptBlockPidMap[pid_m];
    const int expt_id  = expt_data & 0xFFFF;
    const int block_id = (expt_data >> 16) & 0xFFFF;
    if (expt_id == 0xFFFF) return;

    const int M_e = ExptHist[expt_id];
    if (M_e <= 0) return;
    const int start_m = ExptOffs[expt_id];

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int lane_mn = lane & 15;
    const int lane_kg = lane >> 4;

    const int wave_m_off = wave * 16;
    const int local_m = block_id * BLOCK_M + wave_m_off + lane_mn;
    const int clamped_local_m = (local_m < M_e) ? local_m : (M_e - 1);
    int x_row;
    if constexpr (HAS_GATHER) {
        int g = GatherIndx[start_m + clamped_local_m];
        x_row = g / n_expts_act;
    } else {
        x_row = start_m + clamped_local_m;
    }

    const int n_base = pid_n * BLOCK_N;
    int n_cols[NN];
    bool n_valid[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) {
        int nc = n_base + t * 16 + lane_mn;
        n_cols[t]  = (nc < N) ? nc : (N - 1);
        n_valid[t] = (nc < N);
    }

    const uint8_t* Wexp  = W      + (int64_t)expt_id * stride_w_e;
    const uint8_t* WSexp = WScale + (int64_t)expt_id * stride_ws_e;

    float4_t acc[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) acc[t] = float4_t{0,0,0,0};

    const uint8_t* X_lane_row = X + (int64_t)x_row * stride_x_m;

    // Pre-compute per-N base offsets
    const int K_pk     = K >> 1;
    const int K_groups = K >> 5;
    int64_t b_n_base[NN];
    int64_t s_n_base[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) {
        if constexpr (WEIGHT_LAYOUT == 0) {
            b_n_base[t] = (int64_t)n_cols[t] * stride_w_n;
            s_n_base[t] = (int64_t)n_cols[t] * stride_ws_n;
        } else {
            int n_col = n_cols[t];
            if constexpr (WEIGHT_LAYOUT == 1) {
                const int N_half = N >> 1;
                const int gu     = (n_col >= N_half) ? 1 : 0;
                const int n_half = n_col - gu * N_half;
                const int n0     = n_half >> 4;
                const int n_lane = n_half & 15;
                b_n_base[t] = (int64_t)n0 * (int64_t)(32 * K_pk)
                            + (int64_t)gu * (int64_t)(16 * K_pk)
                            + (int64_t)n_lane * 16;
                s_n_base[t] = (int64_t)n0 * (int64_t)(32 * K_groups)
                            + (int64_t)n_lane * 4
                            + (int64_t)gu;
            } else {
                const int n0     = n_col >> 4;
                const int n_lane = n_col & 15;
                b_n_base[t] = (int64_t)n0 * (int64_t)(16 * K_pk)
                            + (int64_t)n_lane * 16;
                const int n1     = n_col >> 5;
                const int n_pack = (n_col >> 4) & 1;
                s_n_base[t] = (int64_t)n1 * (int64_t)(32 * K_groups)
                            + (int64_t)n_lane * 4
                            + (int64_t)n_pack;
            }
        }
    }

    const int num_k128 = (K + 127) / 128;
    const int a_k0_bank0 = lane_kg * 16;            // K offset for A bank 0
    const int a_k0_bank1 = lane_kg * 16 + 64;       // K offset for A bank 1
    const int b_k0_bank0 = lane_kg * 32;            // K offset for B bank 0
    const int b_k0_bank1 = lane_kg * 32 + 16;       // K offset for B bank 1

    // Fast path: K divisible by 128 -> no tail checks.
    if (K % 128 == 0) {
        for (int kk = 0; kk < num_k128; ++kk) {
            const int K_base = kk * 128;
            const int a_k_b0 = K_base + a_k0_bank0;
            const int a_k_b1 = K_base + a_k0_bank1;
            const int b_k_b0 = K_base + b_k0_bank0;
            const int k_scale_block = (K_base >> 5) + lane_kg;
            const int b_k_packed_off = b_k_b0 >> 1;
            // Shuffled K-decomposition (used only when WEIGHT_LAYOUT != 0)
            const int sh_k0     = b_k_packed_off >> 6;
            const int sh_klane  = (b_k_packed_off >> 4) & 3;
            const int sh_s_klane= k_scale_block & 3;
            const int sh_s_kpack= (k_scale_block >> 2) & 1;
            const int sh_s_k1   = k_scale_block >> 3;
            const int64_t sh_b_koff = (int64_t)sh_k0 * 1024 + (int64_t)sh_klane * 256;
            const int64_t sh_s_koff = (int64_t)sh_s_k1 * 256 + (int64_t)sh_s_klane * 64 + (int64_t)sh_s_kpack * 2;

            int32x4_t a0v = load_x4(X_lane_row + a_k_b0);
            int32x4_t a1v = load_x4(X_lane_row + a_k_b1);
            int32x8_t a_reg;
            a_reg[0] = a0v[0]; a_reg[1] = a0v[1]; a_reg[2] = a0v[2]; a_reg[3] = a0v[3];
            a_reg[4] = a1v[0]; a_reg[5] = a1v[1]; a_reg[6] = a1v[2]; a_reg[7] = a1v[3];

            const int scale_a = 127;

            #pragma unroll
            for (int t = 0; t < NN; ++t) {
                int32x4_t bv;
                int scale_b;
                if constexpr (WEIGHT_LAYOUT == 0) {
                    bv = load_x4(Wexp + b_n_base[t] + b_k_packed_off);
                    if constexpr (SWIZZLE_SCALE) {
                        scale_b = (int)load_scale_swizzled(WSexp, n_cols[t], k_scale_block, stride_ws_n);
                    } else {
                        scale_b = (int)WSexp[s_n_base[t] + k_scale_block];
                    }
                } else {
                    bv = load_x4(Wexp + b_n_base[t] + sh_b_koff);
                    scale_b = (int)WSexp[s_n_base[t] + sh_s_koff];
                }
                int32x8_t b_reg;
                b_reg[0] = bv[0]; b_reg[1] = bv[1]; b_reg[2] = bv[2]; b_reg[3] = bv[3];
                b_reg[4] = 0; b_reg[5] = 0; b_reg[6] = 0; b_reg[7] = 0;
                acc[t] = mfma_scale_f8_fp4(a_reg, b_reg, acc[t], scale_a, scale_b);
            }
        }
    } else {
        // Slow tail path
        for (int kk = 0; kk < num_k128; ++kk) {
            const int K_base = kk * 128;
            const int a_k_b0 = K_base + a_k0_bank0;
            const int a_k_b1 = K_base + a_k0_bank1;
            const int b_k_b0 = K_base + b_k0_bank0;
            const int b_k_b1 = K_base + b_k0_bank1;
            const int k_scale_block = (K_base >> 5) + lane_kg;

            int32x8_t a_reg;
            const bool a_b0_full = (a_k_b0 + 16 <= K);
            const bool a_b1_full = (a_k_b1 + 16 <= K);
            int32x4_t a0v, a1v;
            if (a_b0_full) a0v = load_x4(X_lane_row + a_k_b0);
            else a0v = load_x4_safe(X_lane_row + (a_k_b0 < K ? a_k_b0 : 0),
                                    (a_k_b0 < K) ? (K - a_k_b0) : 0);
            if (a_b1_full) a1v = load_x4(X_lane_row + a_k_b1);
            else a1v = load_x4_safe(X_lane_row + (a_k_b1 < K ? a_k_b1 : 0),
                                    (a_k_b1 < K) ? (K - a_k_b1) : 0);
            a_reg[0] = a0v[0]; a_reg[1] = a0v[1]; a_reg[2] = a0v[2]; a_reg[3] = a0v[3];
            a_reg[4] = a1v[0]; a_reg[5] = a1v[1]; a_reg[6] = a1v[2]; a_reg[7] = a1v[3];

            const int scale_a = 127;

            const int b_k_packed_off = b_k_b0 >> 1;
            const bool b_in_range = (b_k_b0 + 32 <= K);
            const int sh_k0     = b_k_packed_off >> 6;
            const int sh_klane  = (b_k_packed_off >> 4) & 3;
            const int sh_s_klane= k_scale_block & 3;
            const int sh_s_kpack= (k_scale_block >> 2) & 1;
            const int sh_s_k1   = k_scale_block >> 3;
            const int64_t sh_b_koff = (int64_t)sh_k0 * 1024 + (int64_t)sh_klane * 256;
            const int64_t sh_s_koff = (int64_t)sh_s_k1 * 256 + (int64_t)sh_s_klane * 64 + (int64_t)sh_s_kpack * 2;
            #pragma unroll
            for (int t = 0; t < NN; ++t) {
                int32x8_t b_reg;
                int64_t off = (WEIGHT_LAYOUT == 0)
                            ? (b_n_base[t] + b_k_packed_off)
                            : (b_n_base[t] + sh_b_koff);
                int32x4_t v;
                if (b_in_range) {
                    v = *reinterpret_cast<const int32x4_t*>(Wexp + off);
                } else {
                    uint8_t tmp[16] = {0};
                    int valid_packed = ((b_k_b0 < K) ? ((K - b_k_b0 + 1) / 2) : 0);
                    if (valid_packed > 16) valid_packed = 16;
                    for (int i = 0; i < valid_packed; ++i) tmp[i] = Wexp[off + i];
                    v = *reinterpret_cast<const int32x4_t*>(tmp);
                }
                b_reg[0] = v[0]; b_reg[1] = v[1]; b_reg[2] = v[2]; b_reg[3] = v[3];
                b_reg[4] = 0; b_reg[5] = 0; b_reg[6] = 0; b_reg[7] = 0;
                int scale_b;
                if (b_k_b0 < K) {
                    if constexpr (WEIGHT_LAYOUT == 0) {
                        if constexpr (SWIZZLE_SCALE) {
                            scale_b = (int)load_scale_swizzled(WSexp, n_cols[t], k_scale_block, stride_ws_n);
                        } else {
                            scale_b = (int)WSexp[s_n_base[t] + k_scale_block];
                        }
                    } else {
                        scale_b = (int)WSexp[s_n_base[t] + sh_s_koff];
                    }
                } else {
                    scale_b = 127;
                }
                acc[t] = mfma_scale_f8_fp4(a_reg, b_reg, acc[t], scale_a, scale_b);
            }
        }
    }

    // ---- Epilogue ----
    const float sx = x_static_scale;
    const int row_g = lane_kg * 4;
    #pragma unroll
    for (int t = 0; t < NN; ++t) {
        if (!n_valid[t]) continue;
        int n_g = n_cols[t];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int mm = wave_m_off + row_g + i;
            int m_local = block_id * BLOCK_M + mm;
            if (m_local >= M_e) continue;
            int y_m = start_m + m_local;
            if (y_m >= M) continue;
            float out = acc[t][i] * sx;
            if constexpr (HAS_GAMMAS) {
                out *= Gammas[y_m];
            }
            Y[(int64_t)y_m * stride_y_m + (int64_t)n_g * stride_y_n]
                = __float2bfloat16(out);
        }
    }
}

// --------------------------------------------------------------------------
// Launcher
// --------------------------------------------------------------------------

#define LAUNCH32_FULL(BN, NW, HG, HGA, SW, WL)                                                  \
    do {                                                                                        \
        constexpr int TPB = (NW) * 64;                                                          \
        dim3 grid(static_cast<unsigned>(grid_m * grid_n));                                      \
        dim3 block(TPB);                                                                        \
        moe_gemm_a8w4_mfma32_kernel<BN, NW, HG, HGA, SW, WL><<<grid, block, 0, stream>>>(       \
            reinterpret_cast<__hip_bfloat16*>(y.data_ptr()),                                    \
            reinterpret_cast<const uint8_t*>(x.data_ptr()),                                     \
            reinterpret_cast<const uint8_t*>(w.data_ptr()),                                     \
            reinterpret_cast<const uint8_t*>(w_scales.data_ptr()),                              \
            (float)x_static_scale,                                                              \
            expt_hist.data_ptr<int32_t>(),                                                      \
            expt_offs.data_ptr<int32_t>(),                                                      \
            expt_block_pid_map.data_ptr<int32_t>(),                                             \
            gather_ptr, gammas_ptr, (int)n_expts_act,                                           \
            (int)M, (int)N, (int)K,                                                             \
            x.stride(0),                                                                        \
            w.stride(0), w.stride(2),                                                           \
            w_scales.stride(0), w_scales.stride(2),                                             \
            y.stride(0), y.stride(1),                                                           \
            (int)grid_n);                                                                       \
    } while (0)

#define LAUNCH_FULL(BM, BN, HG, HGA, SW, WL)                                                    \
    do {                                                                                        \
        constexpr int TPB = (BM / 16) * 64;                                                     \
        dim3 grid(static_cast<unsigned>(grid_m * grid_n));                                      \
        dim3 block(TPB);                                                                        \
        moe_gemm_a8w4_mfma_kernel<BM, BN, HG, HGA, SW, WL><<<grid, block, 0, stream>>>(         \
            reinterpret_cast<__hip_bfloat16*>(y.data_ptr()),                                    \
            reinterpret_cast<const uint8_t*>(x.data_ptr()),                                     \
            reinterpret_cast<const uint8_t*>(w.data_ptr()),                                     \
            reinterpret_cast<const uint8_t*>(w_scales.data_ptr()),                              \
            (float)x_static_scale,                                                              \
            expt_hist.data_ptr<int32_t>(),                                                      \
            expt_offs.data_ptr<int32_t>(),                                                      \
            expt_block_pid_map.data_ptr<int32_t>(),                                             \
            gather_ptr, gammas_ptr, (int)n_expts_act,                                           \
            (int)M, (int)N, (int)K,                                                             \
            x.stride(0),                                                                        \
            w.stride(0), w.stride(2),                                                           \
            w_scales.stride(0), w_scales.stride(2),                                             \
            y.stride(0), y.stride(1),                                                           \
            (int)grid_n);                                                                       \
    } while (0)

// Dispatch over (HAS_GATHER, HAS_GAMMAS, SWIZZLE_SCALE) booleans.
// Backward-compat wrappers preserve the old single-set fast-path codegen for
// the no-gather / no-gammas / no-swizzle case (the perf-critical default).
#define DISPATCH_FLAGS_LAUNCH32_WL(BN, NW, WL)                                                  \
    do {                                                                                        \
        const int flag = ((int)has_gather << 2) | ((int)has_gammas << 1) | (int)has_swiz;       \
        switch (flag) {                                                                         \
            case 0: LAUNCH32_FULL(BN, NW, false, false, false, WL); break;                      \
            case 1: LAUNCH32_FULL(BN, NW, false, false, true , WL); break;                      \
            case 2: LAUNCH32_FULL(BN, NW, false, true , false, WL); break;                      \
            case 3: LAUNCH32_FULL(BN, NW, false, true , true , WL); break;                      \
            case 4: LAUNCH32_FULL(BN, NW, true , false, false, WL); break;                      \
            case 5: LAUNCH32_FULL(BN, NW, true , false, true , WL); break;                      \
            case 6: LAUNCH32_FULL(BN, NW, true , true , false, WL); break;                      \
            case 7: LAUNCH32_FULL(BN, NW, true , true , true , WL); break;                      \
        }                                                                                       \
    } while (0)

#define DISPATCH_FLAGS_LAUNCH32(BN, NW)                                                         \
    do {                                                                                        \
        switch (wlayout) {                                                                      \
            case 0:  DISPATCH_FLAGS_LAUNCH32_WL(BN, NW, 0); break;                              \
            case 1:  DISPATCH_FLAGS_LAUNCH32_WL(BN, NW, 1); break;                              \
            default: DISPATCH_FLAGS_LAUNCH32_WL(BN, NW, 2); break;                              \
        }                                                                                       \
    } while (0)

#define DISPATCH_FLAGS_LAUNCH_WL(BM, BN, WL)                                                    \
    do {                                                                                        \
        const int flag = ((int)has_gather << 2) | ((int)has_gammas << 1) | (int)has_swiz;       \
        switch (flag) {                                                                         \
            case 0: LAUNCH_FULL(BM, BN, false, false, false, WL); break;                        \
            case 1: LAUNCH_FULL(BM, BN, false, false, true , WL); break;                        \
            case 2: LAUNCH_FULL(BM, BN, false, true , false, WL); break;                        \
            case 3: LAUNCH_FULL(BM, BN, false, true , true , WL); break;                        \
            case 4: LAUNCH_FULL(BM, BN, true , false, false, WL); break;                        \
            case 5: LAUNCH_FULL(BM, BN, true , false, true , WL); break;                        \
            case 6: LAUNCH_FULL(BM, BN, true , true , false, WL); break;                        \
            case 7: LAUNCH_FULL(BM, BN, true , true , true , WL); break;                        \
        }                                                                                       \
    } while (0)

#define DISPATCH_FLAGS_LAUNCH(BM, BN)                                                           \
    do {                                                                                        \
        switch (wlayout) {                                                                      \
            case 0:  DISPATCH_FLAGS_LAUNCH_WL(BM, BN, 0); break;                                \
            case 1:  DISPATCH_FLAGS_LAUNCH_WL(BM, BN, 1); break;                                \
            default: DISPATCH_FLAGS_LAUNCH_WL(BM, BN, 2); break;                                \
        }                                                                                       \
    } while (0)

#define LAUNCH32(BN, NW) DISPATCH_FLAGS_LAUNCH32(BN, NW)
#define LAUNCH(BM, BN)   DISPATCH_FLAGS_LAUNCH(BM, BN)

extern "C" void launch_moe_gemm_a8w4(
    torch::Tensor y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor w_scales,
    double x_static_scale,
    torch::Tensor expt_hist,
    torch::Tensor expt_offs,
    torch::Tensor expt_block_pid_map,
    torch::Tensor expt_offs_sum,
    torch::Tensor gather_indx,
    torch::Tensor scatter_indx,
    torch::Tensor gammas,
    int64_t n_expts_tot, int64_t n_expts_act,
    int64_t M, int64_t N, int64_t K,
    int64_t grid_m, int64_t grid_n,
    int64_t block_m, int64_t block_n, int64_t block_k,
    int64_t swizzle_flag,
    int64_t weight_layout_flag)
{
    TORCH_CHECK(y.is_cuda() && x.is_cuda() && w.is_cuda(), "tensors must be on GPU");
    TORCH_CHECK(y.dtype() == at::kBFloat16, "y must be bf16");
    TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "x must be float8_e4m3fn");
    TORCH_CHECK(w.dtype() == at::kByte && w_scales.dtype() == at::kByte, "w/ws must be uint8");
    const int wlayout = (int)weight_layout_flag;
    if (wlayout == 0) {
        TORCH_CHECK(w.stride(1) == 1, "w.stride(-2) must be 1");
        TORCH_CHECK(w_scales.stride(1) == 1, "w_scales.stride(-2) must be 1");
    } else {
        // Shuffled layout: w/w_scales come as contiguous [E, N, K/2] / [E, N, K_groups].
        // The wrapper bug (N = w.shape[-1]) means it passed N = K/2 to us; recover
        // the true N from the tensor shape and reallocate Y accordingly.
        TORCH_CHECK(w.dim() == 3, "shuffled w must be 3D");
        const int64_t N_real   = w.size(1);
        const int64_t K_pk     = w.size(2);
        const int64_t K_real   = K_pk * 2;
        // Override N/K/grid_n with the true values
        N = N_real;
        K = K_real;
        // recompute grid_n
        grid_n = (N_real + block_n - 1) / block_n;
        // Resize Y to [M_y, N_real]. y was allocated [M_y, K/2] by the wrapper.
        const int64_t M_y = y.size(0);
        if (y.size(1) != N_real) {
            y.resize_({M_y, N_real});
        }
    }
    TORCH_CHECK(x.stride(1) == 1, "x.stride(-1) must be 1");
    TORCH_CHECK(grid_m > 0 && grid_n > 0, "empty grid");

    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    // Resolve optional-tensor pointers.  Empty tensors (numel==0) indicate
    // "feature not used" -- matches the wrapper's placeholder convention.
    const int32_t* gather_ptr =
        (gather_indx.numel() > 0) ? gather_indx.data_ptr<int32_t>() : nullptr;
    const float* gammas_ptr =
        (gammas.numel() > 0) ? gammas.data_ptr<float>() : nullptr;
    (void)expt_offs_sum;
    (void)scatter_indx;  // scatter (output indirection / reduce) is handled
                         // by the Python wrapper via reduce_grouped.
    const bool has_gather = (gather_ptr != nullptr);
    const bool has_gammas = (gammas_ptr != nullptr);
    const bool has_swiz   = (swizzle_flag != 0);

    if (block_m == 16) {
        if      (block_n == 16)  LAUNCH(16, 16);
        else if (block_n == 32)  LAUNCH(16, 32);
        else if (block_n == 64)  LAUNCH(16, 64);
        else if (block_n == 128) LAUNCH(16, 128);
        else if (block_n == 256) LAUNCH(16, 256);
        else if (block_n == 512) LAUNCH(16, 512);
        else TORCH_CHECK(false, "unsupported block_n for block_m=16: ", block_n);
    } else if (block_m == 32) {
        // BLOCK_M == 32: use 32x32x64 MFMA path when BLOCK_N is a multiple of 64.
        if      (block_n == 16)  LAUNCH  (32, 16);
        else if (block_n == 32)  LAUNCH32(32, 1);
        else if (block_n == 64)  LAUNCH32(64, 2);
        else if (block_n == 128) LAUNCH32(128, 2);
        else if (block_n == 256) LAUNCH32(256, 4);
        else if (block_n == 512) LAUNCH32(512, 4);
        else TORCH_CHECK(false, "unsupported block_n for block_m=32: ", block_n);
    } else if (block_m == 64) {
        if      (block_n == 16)  LAUNCH(64, 16);
        else if (block_n == 32)  LAUNCH(64, 32);
        else if (block_n == 64)  LAUNCH(64, 64);
        else if (block_n == 128) LAUNCH(64, 128);
        else if (block_n == 256) LAUNCH(64, 256);
        else if (block_n == 512) LAUNCH(64, 512);
        else TORCH_CHECK(false, "unsupported block_n for block_m=64: ", block_n);
    } else if (block_m == 128) {
        // BLOCK_M == 128 reuses the 16x16x128 MFMA path (WAVES_M = 8,
        // TPB = 512). Same set of BLOCK_N as block_m=64.
        if      (block_n == 16)  LAUNCH(128, 16);
        else if (block_n == 32)  LAUNCH(128, 32);
        else if (block_n == 64)  LAUNCH(128, 64);
        else if (block_n == 128) LAUNCH(128, 128);
        else if (block_n == 256) LAUNCH(128, 256);
        else if (block_n == 512) LAUNCH(128, 512);
        else TORCH_CHECK(false, "unsupported block_n for block_m=128: ", block_n);
    } else {
        TORCH_CHECK(false, "unsupported block_m: ", block_m);
    }
}

TORCH_LIBRARY(moe_op_gemm_a8w4_hip, m) {
    m.def("launch_moe_gemm_a8w4", &launch_moe_gemm_a8w4);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_moe_gemm_a8w4", &launch_moe_gemm_a8w4,
          "Minimal W4A8 MoE GEMM (HIP, scaled MFMA)");
}
