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

// --------------------------------------------------------------------------
// K-SPLIT kernel for BLOCK_M=16 path (Phase 4 architectural rewrite).
//
// Layout (vs the N-split path in moe_gemm_a8w4_mfma_kernel below):
//   - Same grid as the existing kernel (one CTA per (pid_m, pid_n) tile).
//   - TPB = NW * 64 waves; ALL waves cover the FULL (BLOCK_M=16, BLOCK_N)
//     output tile (so each wave has NN = BLOCK_N/16 16-wide N-tiles).
//   - Each wave w processes a disjoint slice of K: K-iters
//       [w * num_k128 / NW, (w+1) * num_k128 / NW).
//   - After the K-loop, each wave holds a partial sum acc[NN] in VGPR.
//   - Cross-wave reduction via __shared__ float s_partial[NW][BLOCK_M][BLOCK_N]
//     (16 KB for NW=2, BLOCK_M=16, BLOCK_N=128).
//   - All lanes participate in the final reduce+epilogue: each lane reads its
//     own write positions across NW slots, sums, scales, casts to bf16, writes.
//
// This produces bit-different but numerically-equivalent results to a serial
// K-reduce (FP non-associativity), so parity is checked with rtol=1e-3.
//
// LDS budget per CTA (BLOCK_M=16, BLOCK_N=128, NW=2):
//   s_partial:                  NW * BLOCK_M * BLOCK_N * 4 = 16 KB
//   s_x_rows (gather scratch):  64 B
//   ----------------------------------------------------------------
//   Total: ~16 KB  (<< 64 KB/CU MI355X budget; leaves room for LDS-B
//   in Step 2.)
//
// Requirements: BLOCK_M=16, K % (NW * 128) == 0, BLOCK_N % 16 == 0.
// Per-iter K-tile is 128 K-elements (matches the 16x16x128 MFMA).
// --------------------------------------------------------------------------
template <int BLOCK_N, int NW,
          bool HAS_GATHER = false, bool HAS_GAMMAS = false, bool SWIZZLE_SCALE = false,
          int WEIGHT_LAYOUT = 0>
__global__ __launch_bounds__(NW * 64)
void moe_gemm_a8w4_mfma_ksplit_kernel(
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
    constexpr int BLOCK_M = 16;
    static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be multiple of 16");
    static_assert(NW >= 2, "K-split kernel requires NW >= 2");
    constexpr int NN = BLOCK_N / 16;   // 16-wide N tiles (each wave covers all NN)

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
    const int wave = tid >> 6;       // wave id in [0, NW)
    const int lane = tid & 63;
    const int lane_mn = lane & 15;
    const int lane_kg = lane >> 4;

    // M row this lane provides for A. All waves use the SAME row mapping
    // (we are NOT splitting M across waves).
    const int block_m_off = block_id * BLOCK_M;
    const int local_m = block_m_off + lane_mn;
    const int clamped_local_m = (local_m < M_e) ? local_m : (M_e - 1);
    int x_row;
    if constexpr (HAS_GATHER) {
        int g = GatherIndx[start_m + clamped_local_m];
        x_row = g / n_expts_act;
    } else {
        x_row = start_m + clamped_local_m;
    }

    // N cols served by THIS lane. All waves cover the full BLOCK_N
    // (NN tiles per wave), so n_cols/n_valid are wave-independent.
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

    // Pre-compute per-N base offsets (N-dependent terms; K terms folded in
    // per iteration below).
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

    // K-loop range for THIS wave. K is assumed divisible by NW*128 (gated
    // by the launcher: K >= 1024, K % 128 == 0 for production stage1).
    const int num_k128       = K / 128;
    const int k_iters_per_w  = num_k128 / NW;
    const int kk_start       = wave * k_iters_per_w;
    const int kk_end         = (wave == NW - 1) ? num_k128 : (kk_start + k_iters_per_w);

    const int a_k0_bank0 = lane_kg * 16;            // K offset within a 128-K tile, bank 0
    const int a_k0_bank1 = lane_kg * 16 + 64;       // bank 1
    const int b_k0_bank0 = lane_kg * 32;

    // Per-wave K-loop. NO LDS-A/B yet (Step 2 will add LDS-B).
    const int scale_a = 127;
    for (int kk = kk_start; kk < kk_end; ++kk) {
        const int K_base = kk * 128;
        const int a_k_b0 = K_base + a_k0_bank0;
        const int a_k_b1 = K_base + a_k0_bank1;
        const int b_k_b0 = K_base + b_k0_bank0;
        const int k_scale_block = (K_base >> 5) + lane_kg;
        const int b_k_packed_off = b_k_b0 >> 1;
        const int sh_k0     = b_k_packed_off >> 6;
        const int sh_klane  = (b_k_packed_off >> 4) & 3;
        const int sh_s_klane= k_scale_block & 3;
        const int sh_s_kpack= (k_scale_block >> 2) & 1;
        const int sh_s_k1   = k_scale_block >> 3;
        const int64_t sh_b_koff = (int64_t)sh_k0 * 1024 + (int64_t)sh_klane * 256;
        const int64_t sh_s_koff = (int64_t)sh_s_k1 * 256 + (int64_t)sh_s_klane * 64 + (int64_t)sh_s_kpack * 2;

        // Direct HBM A load (no LDS in Step 1; both waves load disjoint K so
        // there is no A-sharing to exploit anyway in this layout).
        int32x4_t a0v = load_x4(X_lane_row + a_k_b0);
        int32x4_t a1v = load_x4(X_lane_row + a_k_b1);
        int32x8_t a_reg;
        a_reg[0] = a0v[0]; a_reg[1] = a0v[1]; a_reg[2] = a0v[2]; a_reg[3] = a0v[3];
        a_reg[4] = a1v[0]; a_reg[5] = a1v[1]; a_reg[6] = a1v[2]; a_reg[7] = a1v[3];

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

    // ---- Cross-wave reduction via LDS (asymmetric, NW=2 optimized) ----
    //
    // Compact reduction: only (NW-1) waves write their partials to LDS;
    // wave 0 keeps its acc in VGPR, reads (NW-1) partials, accumulates in
    // VGPR, then runs the standard single-wave epilogue. Wave-0-only does
    // the Y writeback; other waves exit after writing.
    //
    // LDS budget: (NW-1) * BLOCK_M * BLOCK_N * 4 = (NW-1) * 8 KB.
    //   NW=2 -> 8 KB (vs 16 KB symmetric, halves LDS pressure -> doubles
    //                  LDS-bound occupancy from 4 -> 8 CTAs/CU).
    constexpr int S_PARTIAL_SLOTS = NW - 1;
    constexpr int S_PARTIAL_ELTS  = S_PARTIAL_SLOTS * BLOCK_M * BLOCK_N;
    __shared__ float s_partial[S_PARTIAL_ELTS];

    const int row_g = lane_kg * 4;

    // Non-zero waves write their full acc to LDS slot (wave-1).
    if (wave > 0) {
        const int slot_off = (wave - 1) * (BLOCK_M * BLOCK_N);
        #pragma unroll
        for (int t = 0; t < NN; ++t) {
            int col_base = t * 16 + lane_mn;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = row_g + i;
                s_partial[slot_off + row * BLOCK_N + col_base] = acc[t][i];
            }
        }
    }
    __syncthreads();

    // Wave 0 reads back, accumulates, and writes Y. Other waves exit.
    if (wave == 0) {
        const float sx = x_static_scale;
        #pragma unroll
        for (int t = 0; t < NN; ++t) {
            if (!n_valid[t]) continue;
            int n_g = n_cols[t];
            int col_base = t * 16 + lane_mn;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int mm = row_g + i;
                int m_local = block_m_off + mm;
                if (m_local >= M_e) continue;
                int y_m = start_m + m_local;
                if (y_m >= M) continue;

                float sum = acc[t][i];
                #pragma unroll
                for (int w = 1; w < NW; ++w) {
                    sum += s_partial[(w - 1) * (BLOCK_M * BLOCK_N)
                                     + mm * BLOCK_N + col_base];
                }
                float out = sum * sx;
                if constexpr (HAS_GAMMAS) {
                    out *= Gammas[y_m];
                }
                Y[(int64_t)y_m * stride_y_m + (int64_t)n_g * stride_y_n]
                    = __float2bfloat16(out);
            }
        }
    }
}

// --------------------------------------------------------------------------
// COOPERATIVE LDS-B kernel for BLOCK_M=16, BLOCK_N=128 production path.
// (Phase 5 - the "aiter recipe": N-split + LDS-A + cooperative-LDS-B + NW=4)
//
// Layout:
//   - Same N-split skeleton as moe_gemm_a8w4_mfma_kernel (each wave covers
//     wave_m=0 of the BLOCK_M=16 rows + a wave_n slice of BLOCK_N).
//   - All NW waves cooperatively load the full (BLOCK_N=128, BLOCK_K=128)
//     B tile into a shared LDS buffer once per K-iter. Each wave then reads
//     its NN N-cols' 16-byte chunks from LDS for MFMA.
//   - LDS-A staging unchanged from Phase 2 (4 KB, double-buffered).
//   - LDS-B uses ONE single-buffer slot of 8 KB (sufficient at NW=2; Step 4
//     will double-buffer to 16 KB). Scale loads stay direct VGPR.
//   - HBM B-load count per CTA per K-iter is unchanged (still 512 16-byte
//     chunks total) BUT all NW waves now see the same B tile from LDS,
//     which removes per-wave HBM amplification (the same 8 KB serves NW
//     waves at NW=4) and lets adjacent threads coalesce intra-n-col k chunks.
//
// LDS layout per K-iter (single-buffer):
//   s_B[n_local * 4 + k_chunk] -> 16 packed B bytes covering
//     (n_col = pid_n*BLOCK_N + n_local, K = K_base + k_chunk*32 + [0,32))
//   where n_local in [0, BLOCK_N) and k_chunk in [0, 4).
//   Stride: 16 bytes per chunk; total 128 * 4 * 16 = 8 KB per K-iter.
//
// Cooperative load distribution (TPB = NW * 64):
//   chunks_per_thread = (BLOCK_N * 4) / TPB    (= 4 for NW=2; 2 for NW=4)
//   For each step s in [0, chunks_per_thread):
//     chunk_id = tid * chunks_per_thread + s
//     n_local  = chunk_id >> 2;  k_chunk = chunk_id & 3
//   This places intra-n_col k-chunks (which are contiguous 64 B in HBM for
//   WLAYOUT=0 or the same 1024 B sub-block for WLAYOUT=1) on adjacent
//   threads -> intra-warp coalesces 4 lanes into a 64-B burst.
//
// Constraints (matches the dispatch site):
//   BLOCK_M == 16, BLOCK_N == 128, NW in {2, 4}, K % 128 == 0.
// --------------------------------------------------------------------------
template <int BLOCK_N, int N_WAVES,
          bool HAS_GATHER = false, bool HAS_GAMMAS = false, bool SWIZZLE_SCALE = false,
          int WEIGHT_LAYOUT = 0>
__global__ __launch_bounds__(N_WAVES * 64)
void moe_gemm_a8w4_mfma_coopB_kernel(
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
    constexpr int BLOCK_M     = 16;
    constexpr int WAVES_M     = 1;                       // BLOCK_M=16 -> 1 wave per M-row
    constexpr int BLOCK_N_PW  = BLOCK_N / N_WAVES;       // 64 (NW=2) or 32 (NW=4)
    constexpr int NN          = BLOCK_N_PW / 16;         // 4 or 2
    constexpr int TPB         = N_WAVES * 64;
    constexpr int K_PER_ITER  = 128;                     // 16x16x128 MFMA
    constexpr int K_CHUNKS    = K_PER_ITER / 32;         // 4 chunks of 32 K-elements (=16 packed B bytes)
    constexpr int B_CHUNKS    = BLOCK_N * K_CHUNKS;      // total 16-byte chunks per K-iter
    constexpr int B_CHUNKS_PER_THREAD = B_CHUNKS / TPB;  // 4 (NW=2) or 2 (NW=4)
    static_assert(BLOCK_N % 16 == 0 && BLOCK_N >= 16, "BLOCK_N must be a multiple of 16");
    static_assert(N_WAVES >= 2 && (BLOCK_N % (16 * N_WAVES)) == 0, "BLOCK_N must split evenly across N_WAVES");
    static_assert(B_CHUNKS % TPB == 0, "B chunk count must divide TPB");

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

    const int tid     = threadIdx.x;
    const int wave    = tid >> 6;
    const int lane    = tid & 63;
    const int lane_mn = lane & 15;
    const int lane_kg = lane >> 4;
    const int wave_n  = wave;                            // WAVES_M=1
    const int wave_n_off = wave_n * BLOCK_N_PW;

    // N cols served by THIS lane / wave. Per-lane reads from LDS via n_local.
    const int n_base   = pid_n * BLOCK_N + wave_n_off;
    int n_cols[NN];
    int n_locals[NN];                                    // n_col - pid_n*BLOCK_N (LDS index)
    bool n_valid[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) {
        int n_local = wave_n_off + t * 16 + lane_mn;     // 0..127
        int nc      = pid_n * BLOCK_N + n_local;
        n_locals[t] = n_local;
        n_cols[t]   = (nc < N) ? nc : (N - 1);
        n_valid[t]  = (nc < N);
    }

    const uint8_t* Wexp  = W      + (int64_t)expt_id * stride_w_e;
    const uint8_t* WSexp = WScale + (int64_t)expt_id * stride_ws_e;

    float4_t acc[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) acc[t] = float4_t{0,0,0,0};

    // ---- Cooperative B-load address precomputation ----
    // Each thread is responsible for B_CHUNKS_PER_THREAD chunks per K-iter.
    // We precompute the per-thread N-side address ingredients once
    // (the K-side ingredients change per K-iter).
    //
    // Chunk mapping: chunk_id = tid * B_CHUNKS_PER_THREAD + s
    //                n_local_c = chunk_id >> 2
    //                k_chunk_c = chunk_id & 3
    int  bload_n_local_c[B_CHUNKS_PER_THREAD];
    int  bload_k_chunk_c[B_CHUNKS_PER_THREAD];
    int64_t bload_n_off_c[B_CHUNKS_PER_THREAD];          // N-side HBM offset (k-independent term)
    #pragma unroll
    for (int s = 0; s < B_CHUNKS_PER_THREAD; ++s) {
        int chunk_id = tid * B_CHUNKS_PER_THREAD + s;
        int n_local  = chunk_id >> 2;
        int k_chunk  = chunk_id & 3;
        int nc_c     = pid_n * BLOCK_N + n_local;
        int nc_c_cl  = (nc_c < N) ? nc_c : (N - 1);
        bload_n_local_c[s] = n_local;
        bload_k_chunk_c[s] = k_chunk;
        if constexpr (WEIGHT_LAYOUT == 0) {
            bload_n_off_c[s] = (int64_t)nc_c_cl * stride_w_n;
        } else if constexpr (WEIGHT_LAYOUT == 1) {
            const int N_half = N >> 1;
            const int gu     = (nc_c_cl >= N_half) ? 1 : 0;
            const int n_half = nc_c_cl - gu * N_half;
            const int n0     = n_half >> 4;
            const int n_lane_c = n_half & 15;
            const int K_pk   = K >> 1;
            bload_n_off_c[s] = (int64_t)n0 * (int64_t)(32 * K_pk)
                              + (int64_t)gu * (int64_t)(16 * K_pk)
                              + (int64_t)n_lane_c * 16;
        } else {
            const int n0     = nc_c_cl >> 4;
            const int n_lane_c = nc_c_cl & 15;
            const int K_pk   = K >> 1;
            bload_n_off_c[s] = (int64_t)n0 * (int64_t)(16 * K_pk)
                              + (int64_t)n_lane_c * 16;
        }
    }

    const uint8_t* X_lane_row;
    {
        const int block_m_off = block_id * BLOCK_M;
        const int local_m     = block_m_off + lane_mn;
        const int clamped_local_m = (local_m < M_e) ? local_m : (M_e - 1);
        int x_row;
        if constexpr (HAS_GATHER) {
            int g = GatherIndx[start_m + clamped_local_m];
            x_row = g / n_expts_act;
        } else {
            x_row = start_m + clamped_local_m;
        }
        X_lane_row = X + (int64_t)x_row * stride_x_m;
    }

    // Pre-compute per-N scale base offsets (N-side terms, used in MFMA loop).
    const int K_groups = K >> 5;
    int64_t s_n_base[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) {
        if constexpr (WEIGHT_LAYOUT == 0) {
            s_n_base[t] = (int64_t)n_cols[t] * stride_ws_n;
        } else if constexpr (WEIGHT_LAYOUT == 1) {
            const int N_half = N >> 1;
            const int gu     = (n_cols[t] >= N_half) ? 1 : 0;
            const int n_half = n_cols[t] - gu * N_half;
            const int n0     = n_half >> 4;
            const int n_lane = n_half & 15;
            s_n_base[t] = (int64_t)n0 * (int64_t)(32 * K_groups)
                        + (int64_t)n_lane * 4
                        + (int64_t)gu;
        } else {
            const int n1     = n_cols[t] >> 5;
            const int n_pack = (n_cols[t] >> 4) & 1;
            const int n_lane = n_cols[t] & 15;
            s_n_base[t] = (int64_t)n1 * (int64_t)(32 * K_groups)
                        + (int64_t)n_lane * 4
                        + (int64_t)n_pack;
        }
    }

    const int num_k128   = K / K_PER_ITER;
    const int a_k0_bank0 = lane_kg * 16;
    const int a_k0_bank1 = lane_kg * 16 + 64;

    // ---- LDS-A (ping-pong, 4 KB) -- identical to Phase 2 N-split path ----
    constexpr int S_A_ROWS       = BLOCK_M;              // 16
    constexpr int S_A_CKB        = 128;                  // bytes per row per K-iter
    constexpr int S_A_SLOT_BYTES = S_A_ROWS * S_A_CKB;   // 2 KB per slot
    constexpr int S_A_NBUF       = 2;
    constexpr int S_A_BYTES      = S_A_NBUF * S_A_SLOT_BYTES;
    __shared__ uint8_t s_A_storage[S_A_BYTES];
    __shared__ int     s_x_rows[S_A_ROWS];

    // ---- LDS-B (ping-pong, 8 KB per slot, 16 KB total) ----
    // Single barrier per K-iter: iter (k+1) stores into slot (k+1)&1 while
    // iter k still reads slot k&1 from MFMA. The barrier appears once at
    // the top of each iter and orders both the cooperative A store and the
    // cooperative B store before any wave reads from LDS.
    constexpr int S_B_SLOT_BYTES = BLOCK_N * K_CHUNKS * 16;   // 8 KB per slot
    constexpr int S_B_NBUF       = 2;
    constexpr int S_B_BYTES      = S_B_NBUF * S_B_SLOT_BYTES;
    __shared__ uint8_t s_B_storage[S_B_BYTES];

    // Per-CTA distinct-rows for A coop loader (handle short experts).
    const int block_m_off = block_id * BLOCK_M;
    int coop_rows_a;
    if (block_m_off + BLOCK_M <= M_e) {
        coop_rows_a = BLOCK_M;
    } else {
        coop_rows_a = (M_e > block_m_off) ? (M_e - block_m_off) : 1;
        if (coop_rows_a > BLOCK_M) coop_rows_a = BLOCK_M;
    }
    const int coop_lanes_a = coop_rows_a * 8;            // 8 16-byte chunks per A row

    // Precompute X-row table.
    if (tid < coop_rows_a) {
        int r_local_m = block_m_off + tid;
        int r_clamped = (r_local_m < M_e) ? r_local_m : (M_e - 1);
        int xr;
        if constexpr (HAS_GATHER) {
            int g = GatherIndx[start_m + r_clamped];
            xr = g / n_expts_act;
        } else {
            xr = start_m + r_clamped;
        }
        s_x_rows[tid] = xr;
    }
    __syncthreads();

    const int scale_a = 127;

    // ---- K-loop: fast path (K % 128 == 0; gated at dispatch) ----
    for (int kk = 0; kk < num_k128; ++kk) {
        const int K_base = kk * K_PER_ITER;

        // --- (1) Cooperative A load into LDS-A ping-pong slot ---
        const int a_slot      = kk & 1;
        const int a_slot_off  = a_slot * S_A_SLOT_BYTES;
        if (tid < coop_lanes_a) {
            int r     = tid >> 3;
            int k_off = (tid & 7) << 4;
            int xr    = s_x_rows[r];
            const uint8_t* src = X + (int64_t)xr * stride_x_m + K_base + k_off;
            int32x4_t v = load_x4(src);
            *reinterpret_cast<int32x4_t*>(&s_A_storage[a_slot_off + r * S_A_CKB + k_off]) = v;
        }

        // --- (2) Cooperative B load into LDS-B (single buffer) ---
        // Each thread loads B_CHUNKS_PER_THREAD = 512/TPB chunks per K-iter.
        const int b_slot     = kk & 1;
        const int b_slot_off = b_slot * S_B_SLOT_BYTES;
        #pragma unroll
        for (int s = 0; s < B_CHUNKS_PER_THREAD; ++s) {
            const int n_local_c = bload_n_local_c[s];
            const int k_chunk_c = bload_k_chunk_c[s];
            // HBM K-side offset
            int64_t k_off_hbm;
            if constexpr (WEIGHT_LAYOUT == 0) {
                // b_packed_off = (K_base + k_chunk*32) / 2 = K_base/2 + k_chunk*16
                k_off_hbm = (int64_t)((K_base >> 1) + k_chunk_c * 16);
            } else {
                // shuffled: sh_b_koff = sh_k0 * 1024 + sh_klane * 256
                // b_packed_off = K_base/2 + k_chunk*16
                int b_packed_off = (K_base >> 1) + k_chunk_c * 16;
                int sh_k0     = b_packed_off >> 6;
                int sh_klane  = (b_packed_off >> 4) & 3;
                k_off_hbm = (int64_t)sh_k0 * 1024 + (int64_t)sh_klane * 256;
            }
            int32x4_t bv = load_x4(Wexp + bload_n_off_c[s] + k_off_hbm);
            // LDS layout: [k_chunk][n_local] so adjacent MFMA lanes (which
            // step n_local by 1 within a NN tile) hit adjacent 16-byte LDS
            // addresses -> bank-stride 4 banks instead of 16, minimizes
            // ds_read_b128 conflicts.
            const int lds_off = b_slot_off + (k_chunk_c * BLOCK_N + n_local_c) * 16;
            *reinterpret_cast<int32x4_t*>(&s_B_storage[lds_off]) = bv;
        }

        // Single barrier per K-iter: A and B both visible to all waves.
        __syncthreads();

        // --- (3) Read A from LDS-A ---
        int32x8_t a_reg;
        {
            int read_row = (lane_mn < coop_rows_a) ? lane_mn : (coop_rows_a - 1);
            int32x4_t a0v = *reinterpret_cast<int32x4_t*>(
                                &s_A_storage[a_slot_off + read_row * S_A_CKB + a_k0_bank0]);
            int32x4_t a1v = *reinterpret_cast<int32x4_t*>(
                                &s_A_storage[a_slot_off + read_row * S_A_CKB + a_k0_bank1]);
            a_reg[0] = a0v[0]; a_reg[1] = a0v[1]; a_reg[2] = a0v[2]; a_reg[3] = a0v[3];
            a_reg[4] = a1v[0]; a_reg[5] = a1v[1]; a_reg[6] = a1v[2]; a_reg[7] = a1v[3];
        }

        // --- (4) MFMA loop: each NN tile reads its 16-byte B chunk from LDS ---
        const int k_scale_block = (K_base >> 5) + lane_kg;
        // Scale-side K decomposition (used only for shuffled WL).
        int sh_s_klane = k_scale_block & 3;
        int sh_s_kpack = (k_scale_block >> 2) & 1;
        int sh_s_k1    = k_scale_block >> 3;
        int64_t sh_s_koff = (int64_t)sh_s_k1 * 256 + (int64_t)sh_s_klane * 64 + (int64_t)sh_s_kpack * 2;

        #pragma unroll
        for (int t = 0; t < NN; ++t) {
            // B from LDS: per-lane chunk at (n_local[t], lane_kg) in current slot
            const int lds_off = b_slot_off + (lane_kg * BLOCK_N + n_locals[t]) * 16;
            int32x4_t bv = *reinterpret_cast<int32x4_t*>(&s_B_storage[lds_off]);

            // Scale: direct from HBM (small, kept in VGPR; LDS-stage later if helpful)
            int scale_b;
            if constexpr (WEIGHT_LAYOUT == 0) {
                if constexpr (SWIZZLE_SCALE) {
                    scale_b = (int)load_scale_swizzled(WSexp, n_cols[t], k_scale_block, stride_ws_n);
                } else {
                    scale_b = (int)WSexp[s_n_base[t] + k_scale_block];
                }
            } else {
                scale_b = (int)WSexp[s_n_base[t] + sh_s_koff];
            }
            int32x8_t b_reg;
            b_reg[0] = bv[0]; b_reg[1] = bv[1]; b_reg[2] = bv[2]; b_reg[3] = bv[3];
            b_reg[4] = 0; b_reg[5] = 0; b_reg[6] = 0; b_reg[7] = 0;
            acc[t] = mfma_scale_f8_fp4(a_reg, b_reg, acc[t], scale_a, scale_b);
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
            int mm = row_g + i;
            int m_local = block_m_off + mm;
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
// Phase 6: BLOCK_K=256 kernel for BLOCK_M=16, BLOCK_N=128 production path.
//
// Mirrors aiter Triton's BLOCK_K=256 recipe: each K-iter does TWO MFMA calls
// per NN tile per wave (one for K[0,128), one for K[128,256)), back-to-back
// so the instruction scheduler overlaps the second tile's VMEM with the
// first's MFMA. At K=2048 this halves the K-iter loop count (8 vs 16) and
// amortizes per-iter overhead (barrier, address arithmetic) over 2x more
// useful work per iter.
//
// VGPR-reduction choices (target: 60 -> closer to aiter's 32):
//   - int32 instead of int64 for `b_n_base[NN]` / `s_n_base[NN]` (per-N
//     offsets fit easily in int32 for stage1/stage2 -- see overflow check).
//   - Drop precomputed `n_valid[NN]` array; recompute inline in epilogue.
//   - Keep `n_cols[NN]` clamped (needed for scale-base recompute & epilogue).
//   - Reuse the Phase 2 LDS-A ping-pong staging but with 256 K-bytes/row
//     (= 4 KB per slot, 8 KB total -- still well under LDS budget).
//
// LDS budget (worst case N_WAVES=4):
//   LDS-A: 16 rows * 256 B * 2 slots = 8 KB
//   x-row table: 16 * 4 = 64 B
//   total: ~8.1 KB / CTA  (way under 64 KB / CU)
//
// Per-K-iter MFMA work: NN * 2 calls. Loop count = K / 256.
// Constraint: K % 256 == 0 (asserted at dispatch).
// --------------------------------------------------------------------------
template <int BLOCK_N, int N_WAVES,
          bool HAS_GATHER = false, bool HAS_GAMMAS = false, bool SWIZZLE_SCALE = false,
          int WEIGHT_LAYOUT = 0>
__global__ __launch_bounds__(N_WAVES * 64)
void moe_gemm_a8w4_mfma_bk256_kernel(
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
    constexpr int BLOCK_M    = 16;
    constexpr int BLOCK_N_PW = BLOCK_N / N_WAVES;
    constexpr int NN         = BLOCK_N_PW / 16;
    constexpr int TPB        = N_WAVES * 64;
    constexpr int K_PER_ITER = 256;   // BLOCK_K = 256 (TWO 16x16x128 MFMA per K-iter)
    static_assert(BLOCK_N % 16 == 0 && BLOCK_N >= 16, "BLOCK_N must be a multiple of 16");
    static_assert(N_WAVES >= 1 && (BLOCK_N % (16 * N_WAVES)) == 0,
                  "BLOCK_N must split evenly across N_WAVES");

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

    const int tid     = threadIdx.x;
    const int wave    = tid >> 6;
    const int lane    = tid & 63;
    const int lane_mn = lane & 15;
    const int lane_kg = lane >> 4;
    const int wave_n  = (N_WAVES == 1) ? 0 : wave;   // BLOCK_M=16 => WAVES_M=1
    const int wave_n_off = wave_n * BLOCK_N_PW;

    // N cols served by THIS lane. n_cols clamped to N-1 for safe addressing;
    // validity rechecked in epilogue (saves NN VGPRs vs caching n_valid[]).
    const int n_base = pid_n * BLOCK_N + wave_n_off;
    int n_cols[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) {
        int nc = n_base + t * 16 + lane_mn;
        n_cols[t] = (nc < N) ? nc : (N - 1);
    }

    const uint8_t* Wexp  = W      + (int64_t)expt_id * stride_w_e;
    const uint8_t* WSexp = WScale + (int64_t)expt_id * stride_ws_e;

    float4_t acc[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) acc[t] = float4_t{0,0,0,0};

    // ---- Per-N base offsets in int32 (VGPR-saving: was int64 in Phase 2) ----
    // Magnitude bound: for stage1 (N=1024, K=2048, K_pk=1024) WLAYOUT=1:
    //   n0*32*K_pk + gu*16*K_pk + n_lane*16 <= 32*32*1024 + 16*1024 + 16*15
    //   = ~1.06 MB << 2^31. For WLAYOUT=0: n_col*K_pk <= 1023*1024 ~ 1 MB.
    //   For stage2 (N=2048, K=512), bounds are even tighter.
    const int K_pk     = K >> 1;
    const int K_groups = K >> 5;
    int b_n_base[NN];
    int s_n_base[NN];
    #pragma unroll
    for (int t = 0; t < NN; ++t) {
        if constexpr (WEIGHT_LAYOUT == 0) {
            b_n_base[t] = (int)((int64_t)n_cols[t] * stride_w_n);
            s_n_base[t] = (int)((int64_t)n_cols[t] * stride_ws_n);
        } else if constexpr (WEIGHT_LAYOUT == 1) {
            const int N_half = N >> 1;
            const int gu     = (n_cols[t] >= N_half) ? 1 : 0;
            const int n_half = n_cols[t] - gu * N_half;
            const int n0     = n_half >> 4;
            const int n_lane = n_half & 15;
            b_n_base[t] = n0 * (32 * K_pk) + gu * (16 * K_pk) + n_lane * 16;
            s_n_base[t] = n0 * (32 * K_groups) + n_lane * 4 + gu;
        } else {
            const int n0     = n_cols[t] >> 4;
            const int n_lane = n_cols[t] & 15;
            b_n_base[t] = n0 * (16 * K_pk) + n_lane * 16;
            const int n1     = n_cols[t] >> 5;
            const int n_pack = (n_cols[t] >> 4) & 1;
            s_n_base[t] = n1 * (32 * K_groups) + n_lane * 4 + n_pack;
        }
    }

    // Per-lane K offsets inside one 128-K sub-iter.
    const int a_k0_bank0 = lane_kg * 16;
    const int a_k0_bank1 = lane_kg * 16 + 64;
    const int b_k0_bank0 = lane_kg * 32;    // (= b_k0_bank0 of the 128-K sub-iter)

    // ---- LDS-A ping-pong, 4 KB per slot, 8 KB total ----
    // Each K-iter holds (BLOCK_M=16) rows x (K_PER_ITER=256) bytes = 4 KB.
    // The cooperative loader fills the next slot while MFMA reads current.
    // Single __syncthreads() per K-iter is sufficient because the writer
    // targets the OTHER slot (kk^1) than the reader (kk).
    constexpr int S_A_ROWS       = BLOCK_M;
    constexpr int S_A_CKB        = K_PER_ITER;            // 256 B per row per K-iter
    constexpr int S_A_SLOT_BYTES = S_A_ROWS * S_A_CKB;    // 4 KB per slot
    constexpr int S_A_NBUF       = 2;
    constexpr int S_A_BYTES      = S_A_NBUF * S_A_SLOT_BYTES;
    __shared__ uint8_t s_A_storage[S_A_BYTES];
    __shared__ int     s_x_rows[S_A_ROWS];

    // Distinct-row count for the cooperative A loader.
    const int block_m_off = block_id * BLOCK_M;
    int coop_rows_a;
    if (block_m_off + BLOCK_M <= M_e) {
        coop_rows_a = BLOCK_M;
    } else {
        coop_rows_a = (M_e > block_m_off) ? (M_e - block_m_off) : 1;
        if (coop_rows_a > BLOCK_M) coop_rows_a = BLOCK_M;
    }
    // Each row has S_A_CKB / 16 = 16 16-byte chunks per K-iter.
    constexpr int A_CHUNKS_PER_ROW = S_A_CKB / 16;        // 16
    const int coop_lanes_a = coop_rows_a * A_CHUNKS_PER_ROW;

    if (tid < coop_rows_a) {
        int r_local_m = block_m_off + tid;
        int r_clamped = (r_local_m < M_e) ? r_local_m : (M_e - 1);
        int xr;
        if constexpr (HAS_GATHER) {
            int g = GatherIndx[start_m + r_clamped];
            xr = g / n_expts_act;
        } else {
            xr = start_m + r_clamped;
        }
        s_x_rows[tid] = xr;
    }
    __syncthreads();

    const int scale_a = 127;
    const int num_k256 = K / K_PER_ITER;

    // Lane-mn cap for safe LDS-A reads (matches the clamping the loader applied).
    const int read_row = (lane_mn < coop_rows_a) ? lane_mn : (coop_rows_a - 1);

    for (int kk = 0; kk < num_k256; ++kk) {
        const int K_base = kk * K_PER_ITER;   // K_base in {0, 256, 512, ...}

        // ---- (1) Cooperative A load into LDS-A ping-pong slot ----
        // 16 chunks per row * coop_rows_a rows = up to 256 chunks/CTA (at NW=4, TPB=256: 1/thread)
        const int a_slot     = kk & 1;
        const int a_slot_off = a_slot * S_A_SLOT_BYTES;
        if (tid < coop_lanes_a) {
            int r     = tid / A_CHUNKS_PER_ROW;          // row in [0, coop_rows_a)
            int k_off = (tid - r * A_CHUNKS_PER_ROW) << 4; // 0,16,32,...,240
            int xr    = s_x_rows[r];
            const uint8_t* src = X + (int64_t)xr * stride_x_m + K_base + k_off;
            int32x4_t v = load_x4(src);
            *reinterpret_cast<int32x4_t*>(&s_A_storage[a_slot_off + r * S_A_CKB + k_off]) = v;
        }
        // Coverage assertion: at NW=4 (TPB=256), coop_lanes_a <= 16*16=256 == TPB.
        // At NW=2 (TPB=128), coop_lanes_a can be up to 256 > TPB so we need
        // a serialized second pass. (Phase 6 dispatches NW=4 by default; the
        // NW=2 path is supported here for sweep flexibility.)
        if constexpr (TPB < S_A_ROWS * A_CHUNKS_PER_ROW) {
            const int OFF = TPB;
            if (tid + OFF < coop_lanes_a) {
                int idx   = tid + OFF;
                int r     = idx / A_CHUNKS_PER_ROW;
                int k_off = (idx - r * A_CHUNKS_PER_ROW) << 4;
                int xr    = s_x_rows[r];
                const uint8_t* src = X + (int64_t)xr * stride_x_m + K_base + k_off;
                int32x4_t v = load_x4(src);
                *reinterpret_cast<int32x4_t*>(&s_A_storage[a_slot_off + r * S_A_CKB + k_off]) = v;
            }
        }

        __syncthreads();

        // a_reg_lo / a_reg_hi: read just-in-time (separately in the two
        // sub-iter loops below) so the compiler can reuse the same A VGPRs
        // for lo and hi -> lower per-lane live state -> better occupancy.

        // ---- (3) MFMA loop: each NN tile fires TWO 16x16x128 MFMA back-to-back ----
        // Sub-iter 0 covers K[K_base, K_base+128), sub-iter 1 covers K[K_base+128, K_base+256).
        // For each sub-iter we issue: 1 B-load (16 packed B bytes per lane) + 1 scale byte
        // + 1 MFMA per NN tile. Inner unroll over t lets the scheduler interleave the
        // two sub-iters' VMEM and MFMA dispatches.
        const int b_k_packed_off_lo = (K_base + b_k0_bank0) >> 1;
        const int b_k_packed_off_hi = b_k_packed_off_lo + 64;   // +128 K-bytes -> +64 packed
        const int k_scale_block_lo = (K_base >> 5) + lane_kg;
        const int k_scale_block_hi = k_scale_block_lo + 4;      // +128 K = +4 32-K blocks

        // Shuffled K-decomposition (used only when WLAYOUT != 0).
        const int sh_k0_lo     = b_k_packed_off_lo >> 6;
        const int sh_klane_lo  = (b_k_packed_off_lo >> 4) & 3;
        const int sh_k0_hi     = b_k_packed_off_hi >> 6;
        const int sh_klane_hi  = (b_k_packed_off_hi >> 4) & 3;
        const int sh_s_klane_lo= k_scale_block_lo & 3;
        const int sh_s_kpack_lo= (k_scale_block_lo >> 2) & 1;
        const int sh_s_k1_lo   = k_scale_block_lo >> 3;
        const int sh_s_klane_hi= k_scale_block_hi & 3;
        const int sh_s_kpack_hi= (k_scale_block_hi >> 2) & 1;
        const int sh_s_k1_hi   = k_scale_block_hi >> 3;
        const int sh_b_koff_lo = sh_k0_lo * 1024 + sh_klane_lo * 256;
        const int sh_b_koff_hi = sh_k0_hi * 1024 + sh_klane_hi * 256;
        const int sh_s_koff_lo = sh_s_k1_lo * 256 + sh_s_klane_lo * 64 + sh_s_kpack_lo * 2;
        const int sh_s_koff_hi = sh_s_k1_hi * 256 + sh_s_klane_hi * 64 + sh_s_kpack_hi * 2;

        // Sub-iter LO: K[K_base, K_base+128). Reuse one A VGPR set.
        {
            int32x8_t a_reg_lo;
            int32x4_t a0v = *reinterpret_cast<int32x4_t*>(
                                &s_A_storage[a_slot_off + read_row * S_A_CKB + a_k0_bank0]);
            int32x4_t a1v = *reinterpret_cast<int32x4_t*>(
                                &s_A_storage[a_slot_off + read_row * S_A_CKB + a_k0_bank1]);
            a_reg_lo[0] = a0v[0]; a_reg_lo[1] = a0v[1]; a_reg_lo[2] = a0v[2]; a_reg_lo[3] = a0v[3];
            a_reg_lo[4] = a1v[0]; a_reg_lo[5] = a1v[1]; a_reg_lo[6] = a1v[2]; a_reg_lo[7] = a1v[3];
            #pragma unroll
            for (int t = 0; t < NN; ++t) {
                int32x4_t bv_lo;
                int scale_b_lo;
                if constexpr (WEIGHT_LAYOUT == 0) {
                    bv_lo = load_x4(Wexp + (int64_t)b_n_base[t] + b_k_packed_off_lo);
                    if constexpr (SWIZZLE_SCALE) {
                        scale_b_lo = (int)load_scale_swizzled(WSexp, n_cols[t], k_scale_block_lo, stride_ws_n);
                    } else {
                        scale_b_lo = (int)WSexp[s_n_base[t] + k_scale_block_lo];
                    }
                } else {
                    bv_lo = load_x4(Wexp + (int64_t)b_n_base[t] + sh_b_koff_lo);
                    scale_b_lo = (int)WSexp[s_n_base[t] + sh_s_koff_lo];
                }
                int32x8_t b_reg_lo;
                b_reg_lo[0] = bv_lo[0]; b_reg_lo[1] = bv_lo[1]; b_reg_lo[2] = bv_lo[2]; b_reg_lo[3] = bv_lo[3];
                b_reg_lo[4] = 0; b_reg_lo[5] = 0; b_reg_lo[6] = 0; b_reg_lo[7] = 0;
                acc[t] = mfma_scale_f8_fp4(a_reg_lo, b_reg_lo, acc[t], scale_a, scale_b_lo);
            }
        }
        // Sub-iter HI: K[K_base+128, K_base+256). New A VGPRs (compiler may reuse).
        {
            int32x8_t a_reg_hi;
            int32x4_t a2v = *reinterpret_cast<int32x4_t*>(
                                &s_A_storage[a_slot_off + read_row * S_A_CKB + a_k0_bank0 + 128]);
            int32x4_t a3v = *reinterpret_cast<int32x4_t*>(
                                &s_A_storage[a_slot_off + read_row * S_A_CKB + a_k0_bank1 + 128]);
            a_reg_hi[0] = a2v[0]; a_reg_hi[1] = a2v[1]; a_reg_hi[2] = a2v[2]; a_reg_hi[3] = a2v[3];
            a_reg_hi[4] = a3v[0]; a_reg_hi[5] = a3v[1]; a_reg_hi[6] = a3v[2]; a_reg_hi[7] = a3v[3];
            #pragma unroll
            for (int t = 0; t < NN; ++t) {
                int32x4_t bv_hi;
                int scale_b_hi;
                if constexpr (WEIGHT_LAYOUT == 0) {
                    bv_hi = load_x4(Wexp + (int64_t)b_n_base[t] + b_k_packed_off_hi);
                    if constexpr (SWIZZLE_SCALE) {
                        scale_b_hi = (int)load_scale_swizzled(WSexp, n_cols[t], k_scale_block_hi, stride_ws_n);
                    } else {
                        scale_b_hi = (int)WSexp[s_n_base[t] + k_scale_block_hi];
                    }
                } else {
                    bv_hi = load_x4(Wexp + (int64_t)b_n_base[t] + sh_b_koff_hi);
                    scale_b_hi = (int)WSexp[s_n_base[t] + sh_s_koff_hi];
                }
                int32x8_t b_reg_hi;
                b_reg_hi[0] = bv_hi[0]; b_reg_hi[1] = bv_hi[1]; b_reg_hi[2] = bv_hi[2]; b_reg_hi[3] = bv_hi[3];
                b_reg_hi[4] = 0; b_reg_hi[5] = 0; b_reg_hi[6] = 0; b_reg_hi[7] = 0;
                acc[t] = mfma_scale_f8_fp4(a_reg_hi, b_reg_hi, acc[t], scale_a, scale_b_hi);
            }
        }
    }

    // ---- Epilogue ----
    const float sx = x_static_scale;
    const int row_g = lane_kg * 4;
    #pragma unroll
    for (int t = 0; t < NN; ++t) {
        int nc_raw = n_base + t * 16 + lane_mn;
        if (nc_raw >= N) continue;                       // recompute valid (saves VGPRs)
        int n_g = n_cols[t];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int mm = row_g + i;
            int m_local = block_m_off + mm;
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


template <int BLOCK_M, int BLOCK_N, int N_WAVES = 1, bool LDS_A_ENABLE = (N_WAVES > 1),
          bool HAS_GATHER = false, bool HAS_GAMMAS = false, bool SWIZZLE_SCALE = false,
          int WEIGHT_LAYOUT = 0>
__global__ __launch_bounds__((BLOCK_M / 16) * N_WAVES * 64)
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
    static_assert(N_WAVES >= 1, "N_WAVES must be >= 1");
    static_assert((BLOCK_N % (16 * N_WAVES)) == 0,
                  "BLOCK_N must be multiple of 16*N_WAVES");
    constexpr int WAVES_M     = BLOCK_M / 16;
    constexpr int NN_TOTAL    = BLOCK_N / 16;          // total 16-wide N-tiles in the CTA
    constexpr int BLOCK_N_PW  = BLOCK_N / N_WAVES;     // N cols per wave
    constexpr int NN          = BLOCK_N_PW / 16;       // 16-wide N-tiles per wave

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

    // Wave decomposition: total waves = WAVES_M * N_WAVES.
    // Layout: wave_id = wave_m * N_WAVES + wave_n (row-major over (M, N)).
    const int wave_m = (N_WAVES == 1) ? wave : (wave / N_WAVES);
    const int wave_n = (N_WAVES == 1) ? 0    : (wave % N_WAVES);

    const int wave_m_off = wave_m * 16;
    const int wave_n_off = wave_n * BLOCK_N_PW;
    const int local_m = block_id * BLOCK_M + wave_m_off + lane_mn;
    const int clamped_local_m = (local_m < M_e) ? local_m : (M_e - 1);
    int x_row;
    if constexpr (HAS_GATHER) {
        int g = GatherIndx[start_m + clamped_local_m];
        x_row = g / n_expts_act;
    } else {
        x_row = start_m + clamped_local_m;
    }

    const int n_base = pid_n * BLOCK_N + wave_n_off;
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

    // ---- LDS-broadcast of A (Phase 2 of kernel optimization) ----
    //
    // When N_WAVES > 1, all waves in a CTA share the SAME BLOCK_M rows of A.
    // Without staging, each wave issues redundant HBM A loads (1.x VMEM
    // amplification per wave). Stage cooperatively into LDS, double-buffer
    // so only ONE __syncthreads() is needed per K-iter (load[i+1] writes
    // the OTHER slot, never clobbering slot[i] still being read).
    //
    // LDS budget per CTA (BLOCK_M=16): 2 slots * 16 rows * 128 K-bytes = 4 KB
    // (well under MI355X's 64 KB/CU). Gated by constexpr USE_LDS_A=(N_WAVES>1)
    // so BLOCK_M=32/64/128 NW=1 paths are byte-identical.
    constexpr bool USE_LDS_A     = LDS_A_ENABLE && (N_WAVES > 1);
    constexpr int  S_A_ROWS      = BLOCK_M;        // 16 in multi-wave path
    constexpr int  S_A_CKB       = 128;            // A bytes per row per K-tile
    constexpr int  S_A_SLOT_BYTES= USE_LDS_A ? (S_A_ROWS * S_A_CKB) : 1;
    constexpr int  S_A_NBUF      = USE_LDS_A ? 2 : 1;
    constexpr int  S_A_BYTES     = S_A_NBUF * S_A_SLOT_BYTES;
    __shared__ uint8_t s_A_storage[S_A_BYTES];

    // Per-row x_row table: 16 ints precomputed once per CTA so the
    // cooperative loader doesn't redo gather logic on every K-tile.
    constexpr int  S_XR_LEN = USE_LDS_A ? S_A_ROWS : 1;
    __shared__ int  s_x_rows[S_XR_LEN];

    // # distinct M rows in this CTA's slice (the rest are duplicates of row
    // coop_rows-1 from the clamping). Cooperative load only issues HBM
    // reads for these distinct rows, which keeps the per-K-iter VMEM issue
    // count small when M_e << BLOCK_M (e.g., M=32/stage2: M_e ~ 2).
    const int block_m_off = block_id * BLOCK_M;
    int coop_rows;
    if (block_m_off + BLOCK_M <= M_e) {
        coop_rows = BLOCK_M;
    } else {
        coop_rows = (M_e > block_m_off) ? (M_e - block_m_off) : 1;
        if (coop_rows > BLOCK_M) coop_rows = BLOCK_M;
    }
    const int coop_lanes = coop_rows * 8;

    if constexpr (USE_LDS_A) {
        if (tid < coop_rows) {
            int r_local_m = block_m_off + tid;
            int r_clamped = (r_local_m < M_e) ? r_local_m : (M_e - 1);
            int xr;
            if constexpr (HAS_GATHER) {
                int g = GatherIndx[start_m + r_clamped];
                xr = g / n_expts_act;
            } else {
                xr = start_m + r_clamped;
            }
            s_x_rows[tid] = xr;
        }
        __syncthreads();
    }

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

            int32x8_t a_reg;
            if constexpr (USE_LDS_A) {
                const int slot     = kk & 1;
                const int slot_off = slot * S_A_SLOT_BYTES;
                if (tid < coop_lanes) {
                    int r     = tid >> 3;                   // row in [0, coop_rows)
                    int k_off = (tid & 7) << 4;             // 0,16,...,112
                    int xr    = s_x_rows[r];
                    const uint8_t* src = X + (int64_t)xr * stride_x_m + K_base + k_off;
                    int32x4_t v = load_x4(src);
                    *reinterpret_cast<int32x4_t*>(&s_A_storage[slot_off + r * S_A_CKB + k_off]) = v;
                }
                __syncthreads();
                // MFMA lanes whose lane_mn >= coop_rows would read garbage
                // from s_A_storage; clamp the read row so they reuse the last
                // valid row (matches the clamping the loader applied).
                int read_row = (lane_mn < coop_rows) ? lane_mn : (coop_rows - 1);
                int32x4_t a0v = *reinterpret_cast<int32x4_t*>(&s_A_storage[slot_off + read_row * S_A_CKB + a_k0_bank0]);
                int32x4_t a1v = *reinterpret_cast<int32x4_t*>(&s_A_storage[slot_off + read_row * S_A_CKB + a_k0_bank1]);
                a_reg[0] = a0v[0]; a_reg[1] = a0v[1]; a_reg[2] = a0v[2]; a_reg[3] = a0v[3];
                a_reg[4] = a1v[0]; a_reg[5] = a1v[1]; a_reg[6] = a1v[2]; a_reg[7] = a1v[3];
            } else {
                int32x4_t a0v = load_x4(X_lane_row + a_k_b0);
                int32x4_t a1v = load_x4(X_lane_row + a_k_b1);
                a_reg[0] = a0v[0]; a_reg[1] = a0v[1]; a_reg[2] = a0v[2]; a_reg[3] = a0v[3];
                a_reg[4] = a1v[0]; a_reg[5] = a1v[1]; a_reg[6] = a1v[2]; a_reg[7] = a1v[3];
            }

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

#define LAUNCH_FULL_LDS(BM, BN, NW, LDS, HG, HGA, SW, WL)                                       \
    do {                                                                                        \
        constexpr int TPB = (BM / 16) * (NW) * 64;                                              \
        dim3 grid(static_cast<unsigned>(grid_m * grid_n));                                      \
        dim3 block(TPB);                                                                        \
        moe_gemm_a8w4_mfma_kernel<BM, BN, NW, LDS, HG, HGA, SW, WL><<<grid, block, 0, stream>>>(\
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

// Back-compat: LAUNCH_FULL forwards to LAUNCH_FULL_LDS using default LDS = (NW>1).
#define LAUNCH_FULL(BM, BN, NW, HG, HGA, SW, WL)                                                \
    LAUNCH_FULL_LDS(BM, BN, NW, ((NW) > 1), HG, HGA, SW, WL)

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

#define DISPATCH_FLAGS_LAUNCH_LDS_WL(BM, BN, NW, LDS, WL)                                       \
    do {                                                                                        \
        const int flag = ((int)has_gather << 2) | ((int)has_gammas << 1) | (int)has_swiz;       \
        switch (flag) {                                                                         \
            case 0: LAUNCH_FULL_LDS(BM, BN, NW, LDS, false, false, false, WL); break;           \
            case 1: LAUNCH_FULL_LDS(BM, BN, NW, LDS, false, false, true , WL); break;           \
            case 2: LAUNCH_FULL_LDS(BM, BN, NW, LDS, false, true , false, WL); break;           \
            case 3: LAUNCH_FULL_LDS(BM, BN, NW, LDS, false, true , true , WL); break;           \
            case 4: LAUNCH_FULL_LDS(BM, BN, NW, LDS, true , false, false, WL); break;           \
            case 5: LAUNCH_FULL_LDS(BM, BN, NW, LDS, true , false, true , WL); break;           \
            case 6: LAUNCH_FULL_LDS(BM, BN, NW, LDS, true , true , false, WL); break;           \
            case 7: LAUNCH_FULL_LDS(BM, BN, NW, LDS, true , true , true , WL); break;           \
        }                                                                                       \
    } while (0)

#define DISPATCH_FLAGS_LAUNCH_LDS(BM, BN, NW, LDS)                                              \
    do {                                                                                        \
        switch (wlayout) {                                                                      \
            case 0:  DISPATCH_FLAGS_LAUNCH_LDS_WL(BM, BN, NW, LDS, 0); break;                   \
            case 1:  DISPATCH_FLAGS_LAUNCH_LDS_WL(BM, BN, NW, LDS, 1); break;                   \
            default: DISPATCH_FLAGS_LAUNCH_LDS_WL(BM, BN, NW, LDS, 2); break;                   \
        }                                                                                       \
    } while (0)

// Back-compat (LDS default = NW>1).
#define DISPATCH_FLAGS_LAUNCH(BM, BN, NW)  DISPATCH_FLAGS_LAUNCH_LDS(BM, BN, NW, ((NW) > 1))

#define LAUNCH32(BN, NW)    DISPATCH_FLAGS_LAUNCH32(BN, NW)
#define LAUNCH(BM, BN)      DISPATCH_FLAGS_LAUNCH(BM, BN, 1)
#define LAUNCH_NW(BM, BN, NW) DISPATCH_FLAGS_LAUNCH(BM, BN, NW)

// ---- K-SPLIT launcher (Phase 4) ----
//   moe_gemm_a8w4_mfma_ksplit_kernel<BN, NW, HG, HGA, SW, WL>
//   TPB = NW * 64, BLOCK_M fixed to 16.
#define LAUNCH_KSPLIT_FULL(BN, NW, HG, HGA, SW, WL)                                             \
    do {                                                                                        \
        constexpr int TPB = (NW) * 64;                                                          \
        dim3 grid(static_cast<unsigned>(grid_m * grid_n));                                      \
        dim3 block(TPB);                                                                        \
        moe_gemm_a8w4_mfma_ksplit_kernel<BN, NW, HG, HGA, SW, WL><<<grid, block, 0, stream>>>(  \
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

#define DISPATCH_FLAGS_LAUNCH_KSPLIT_WL(BN, NW, WL)                                             \
    do {                                                                                        \
        const int flag = ((int)has_gather << 2) | ((int)has_gammas << 1) | (int)has_swiz;       \
        switch (flag) {                                                                         \
            case 0: LAUNCH_KSPLIT_FULL(BN, NW, false, false, false, WL); break;                 \
            case 1: LAUNCH_KSPLIT_FULL(BN, NW, false, false, true , WL); break;                 \
            case 2: LAUNCH_KSPLIT_FULL(BN, NW, false, true , false, WL); break;                 \
            case 3: LAUNCH_KSPLIT_FULL(BN, NW, false, true , true , WL); break;                 \
            case 4: LAUNCH_KSPLIT_FULL(BN, NW, true , false, false, WL); break;                 \
            case 5: LAUNCH_KSPLIT_FULL(BN, NW, true , false, true , WL); break;                 \
            case 6: LAUNCH_KSPLIT_FULL(BN, NW, true , true , false, WL); break;                 \
            case 7: LAUNCH_KSPLIT_FULL(BN, NW, true , true , true , WL); break;                 \
        }                                                                                       \
    } while (0)

#define DISPATCH_FLAGS_LAUNCH_KSPLIT(BN, NW)                                                    \
    do {                                                                                        \
        switch (wlayout) {                                                                      \
            case 0:  DISPATCH_FLAGS_LAUNCH_KSPLIT_WL(BN, NW, 0); break;                         \
            case 1:  DISPATCH_FLAGS_LAUNCH_KSPLIT_WL(BN, NW, 1); break;                         \
            default: DISPATCH_FLAGS_LAUNCH_KSPLIT_WL(BN, NW, 2); break;                         \
        }                                                                                       \
    } while (0)

// ---- COOPERATIVE-LDS-B launcher (Phase 5) ----
//   moe_gemm_a8w4_mfma_coopB_kernel<BN, NW, HG, HGA, SW, WL>
//   TPB = NW * 64. BLOCK_M fixed at 16.
#define LAUNCH_COOPB_FULL(BN, NW, HG, HGA, SW, WL)                                              \
    do {                                                                                        \
        constexpr int TPB = (NW) * 64;                                                          \
        dim3 grid(static_cast<unsigned>(grid_m * grid_n));                                      \
        dim3 block(TPB);                                                                        \
        moe_gemm_a8w4_mfma_coopB_kernel<BN, NW, HG, HGA, SW, WL><<<grid, block, 0, stream>>>(   \
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

#define DISPATCH_FLAGS_LAUNCH_COOPB_WL(BN, NW, WL)                                              \
    do {                                                                                        \
        const int flag = ((int)has_gather << 2) | ((int)has_gammas << 1) | (int)has_swiz;       \
        switch (flag) {                                                                         \
            case 0: LAUNCH_COOPB_FULL(BN, NW, false, false, false, WL); break;                  \
            case 1: LAUNCH_COOPB_FULL(BN, NW, false, false, true , WL); break;                  \
            case 2: LAUNCH_COOPB_FULL(BN, NW, false, true , false, WL); break;                  \
            case 3: LAUNCH_COOPB_FULL(BN, NW, false, true , true , WL); break;                  \
            case 4: LAUNCH_COOPB_FULL(BN, NW, true , false, false, WL); break;                  \
            case 5: LAUNCH_COOPB_FULL(BN, NW, true , false, true , WL); break;                  \
            case 6: LAUNCH_COOPB_FULL(BN, NW, true , true , false, WL); break;                  \
            case 7: LAUNCH_COOPB_FULL(BN, NW, true , true , true , WL); break;                  \
        }                                                                                       \
    } while (0)

#define DISPATCH_FLAGS_LAUNCH_COOPB(BN, NW)                                                     \
    do {                                                                                        \
        switch (wlayout) {                                                                      \
            case 0:  DISPATCH_FLAGS_LAUNCH_COOPB_WL(BN, NW, 0); break;                          \
            case 1:  DISPATCH_FLAGS_LAUNCH_COOPB_WL(BN, NW, 1); break;                          \
            default: DISPATCH_FLAGS_LAUNCH_COOPB_WL(BN, NW, 2); break;                          \
        }                                                                                       \
    } while (0)

// ---- BLOCK_K=256 launcher (Phase 6) ----
//   moe_gemm_a8w4_mfma_bk256_kernel<BN, NW, HG, HGA, SW, WL>
//   TPB = NW * 64. BLOCK_M fixed at 16. BLOCK_K fixed at 256.
#define LAUNCH_BK256_FULL(BN, NW, HG, HGA, SW, WL)                                              \
    do {                                                                                        \
        constexpr int TPB = (NW) * 64;                                                          \
        dim3 grid(static_cast<unsigned>(grid_m * grid_n));                                      \
        dim3 block(TPB);                                                                        \
        moe_gemm_a8w4_mfma_bk256_kernel<BN, NW, HG, HGA, SW, WL><<<grid, block, 0, stream>>>(   \
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

#define DISPATCH_FLAGS_LAUNCH_BK256_WL(BN, NW, WL)                                              \
    do {                                                                                        \
        const int flag = ((int)has_gather << 2) | ((int)has_gammas << 1) | (int)has_swiz;       \
        switch (flag) {                                                                         \
            case 0: LAUNCH_BK256_FULL(BN, NW, false, false, false, WL); break;                  \
            case 1: LAUNCH_BK256_FULL(BN, NW, false, false, true , WL); break;                  \
            case 2: LAUNCH_BK256_FULL(BN, NW, false, true , false, WL); break;                  \
            case 3: LAUNCH_BK256_FULL(BN, NW, false, true , true , WL); break;                  \
            case 4: LAUNCH_BK256_FULL(BN, NW, true , false, false, WL); break;                  \
            case 5: LAUNCH_BK256_FULL(BN, NW, true , false, true , WL); break;                  \
            case 6: LAUNCH_BK256_FULL(BN, NW, true , true , false, WL); break;                  \
            case 7: LAUNCH_BK256_FULL(BN, NW, true , true , true , WL); break;                  \
        }                                                                                       \
    } while (0)

#define DISPATCH_FLAGS_LAUNCH_BK256(BN, NW)                                                     \
    do {                                                                                        \
        switch (wlayout) {                                                                      \
            case 0:  DISPATCH_FLAGS_LAUNCH_BK256_WL(BN, NW, 0); break;                          \
            case 1:  DISPATCH_FLAGS_LAUNCH_BK256_WL(BN, NW, 1); break;                          \
            default: DISPATCH_FLAGS_LAUNCH_BK256_WL(BN, NW, 2); break;                          \
        }                                                                                       \
    } while (0)

// Runtime K-aware: enable LDS-broadcast A only when num_k128 >= 8 (K >= 1024).
// Stage-2 K=512 has only 4 K-iters so the per-iter __syncthreads overhead
// dominates the cooperative-load savings; fall back to direct HBM there.
#define LAUNCH_NW_KSEL(BM, BN, NW)                                                              \
    do {                                                                                        \
        const int _num_k128 = (int)((K + 127) / 128);                                           \
        if ((NW) > 1 && _num_k128 >= 8) {                                                       \
            DISPATCH_FLAGS_LAUNCH_LDS(BM, BN, NW, true);                                        \
        } else {                                                                                \
            DISPATCH_FLAGS_LAUNCH_LDS(BM, BN, NW, false);                                       \
        }                                                                                       \
    } while (0)

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
        // BLOCK_M=16 multi-wave: split BLOCK_N across N_WAVES so each CTA
        // gets ((BLOCK_M/16)*N_WAVES) waves instead of just 1. This lifts
        // occupancy 4x at M=256 (Phase 1 of the kernel optimization plan).
        // Phase 2: LDS-broadcast A for K >= 1024 (LAUNCH_NW_KSEL picks at
        // runtime). Stage-2 K=512 falls back to direct HBM A loads.
        //
        // Phase 4: NEW K-split path for the production stage1 case
        // (block_n==128, K>=1024, K divisible by NW*128). All other BLOCK_M=16
        // shapes (and stage2 K=512) fall through to the Phase 2 N-split path.
        if      (block_n == 16)  LAUNCH_NW(16, 16,  1);   // single 16x16 tile -> no split
        else if (block_n == 32)  LAUNCH_NW_KSEL(16, 32,  2);
        else if (block_n == 64)  LAUNCH_NW_KSEL(16, 64,  2);
        else if (block_n == 128) {
            // Phase 4 K-split path remains in source behind KSPLIT_ENABLE
            // (default off; see moe_gemm_a8w4_mfma_ksplit_kernel).
            //
            // Phase 5 cooperative-LDS-B path: N-split + LDS-A + cooperative
            // shared-LDS B-load. Dispatched when K >= 1024 and K % 128 == 0
            // (the production stage-1 shape). The NW knob controls per-CTA
            // wave count: NW=2 retains Phase 2 wave layout; NW=4 doubles the
            // concurrent waves per CTA (now affordable because LDS-B removes
            // the per-wave B-load amplification).
            constexpr bool KSPLIT_ENABLE = false;
            constexpr int  KS_NW         = 2;
            const bool ksplit_ok = KSPLIT_ENABLE && (K >= 1024) && ((K % (KS_NW * 128)) == 0);
            // Phase 6 BLOCK_K=256 path: halves K-iter count at K>=1024
            // (8 iters vs 16 at K=2048) and amortizes per-iter overhead
            // over 2x more useful work per iter. Only activated when
            // K>=1024 & K%256==0. Stage2 K=512 has only 2 BK=256 iters
            // (barriers don't amortize) so it falls through to the
            // Phase 5 / Phase 2 path automatically.
            constexpr bool BK256_ENABLE      = true;
            constexpr int  BK256_NW          = 4;        // matches aiter num_warps=4
            constexpr int  BK256_M_MIN       = 513;      // small-M (M<=512) stays on coopB
            const bool bk256_ok = BK256_ENABLE
                                   && (K >= 1024)
                                   && ((K % 256) == 0)
                                   && (M >= BK256_M_MIN);
            if (ksplit_ok) {
                DISPATCH_FLAGS_LAUNCH_KSPLIT(128, KS_NW);
            } else if (bk256_ok) {
                DISPATCH_FLAGS_LAUNCH_BK256(128, BK256_NW);
            } else {
                // Phase 5 cooperative-LDS-B path. Enabled only for SMALL M
                // (M_total <= 512) where the N-split-NW=4 layout dominates
                // the per-iter cooperative-LDS overhead. At larger M_total
                // BK256 takes over (above).
                //
                // M here is the post-routing M_total (= input M * topk).
                // Empirical per input-M (topk=8):
                //   M_in=1   (M_total=8)    stage1: ~24 us (vs 26 us Phase 2)   coopB-NW4 WIN
                //   M_in=32  (M_total=256)  stage1: ~44 us (vs 48 us Phase 2)   -8%
                constexpr bool COOPB_ENABLE      = true;
                constexpr int  COOPB_NW          = 4;
                constexpr int  COOPB_M_THRESHOLD = 512;
                const bool coopb_ok = COOPB_ENABLE
                                       && (K >= 1024)
                                       && ((K % 128) == 0)
                                       && (M <= COOPB_M_THRESHOLD);
                if (coopb_ok) {
                    DISPATCH_FLAGS_LAUNCH_COOPB(128, COOPB_NW);
                } else {
                    LAUNCH_NW_KSEL(16, 128, 2);   // Phase 2 active default
                }
            }
        }
        else if (block_n == 256) LAUNCH_NW_KSEL(16, 256, 2);
        else if (block_n == 512) LAUNCH_NW_KSEL(16, 512, 4);
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
