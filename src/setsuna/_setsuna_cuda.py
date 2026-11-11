"""SETSUNA's forward as two runtime-compiled CUDA kernels per window batch: the branch encoders with their head
prologues (CudaEncoders), then the pooled dual head (CudaHeadPool, further down).

A SRBranch encoder is a prologue (in_proj of the trace plus indicator, cell-type and time-position embeddings),
four (ConvBlock, LocalMHABlock, FFNBlock) layers and a final LayerNorm; the pooled path then applies the head
prologue (Linear(d, d) + exact GELU). With d = 24 features all of it is row work, except the ConvBlock's depthwise
convolution along time (15 taps) and the band attention (|m - n| <= half, two heads of 12 lanes). Through the
Triton kernels that work ran at a fraction of the fp32 rate (dot products zero-padded to 32/64/128, operands staged
per k step, ~100 launches per batch, every intermediate through device memory).

Here a thread block covers one window of one branch -- a thread per time step, the row in registers -- and one
launch covers every window and branch of a batch. The layer's weights sit in shared memory (all threads read one
address); the ConvBlock's LayerNorm rows and, per head, q/k/v share one row buffer. The row work is sequential fp32
FMA chains (reorder-level differences to the Triton kernels); the band attention runs on the tensor cores in the
fp16x3 arithmetic of the Triton kernel, its hi x lo correction products accumulated in fp16 (see SRC). H1 relRMS vs
the Triton path 1-4e-7; end to end against a float64 run of the models (24 ROIs x 157,889 frames) rate relRMS
4.6e-7 (Triton path 5.2e-7), signed bias +3.8e-7 (+4.8e-7); over 1450 ROIs the same 380,622 detected spikes, 17
moved by one fine bin (0.03 ms). RTX 5090, 512 windows of 384 frames: 4.0 ms (Triton path 8.1 ms)."""
from __future__ import annotations

import ctypes
import math
import struct
import sys

import torch

from . import _cuda_rt

D = 24
KS = 15
FF = 48
_HC_STRIDE = 28          # floats per row of the conv buffer (24 + pad: conflict-free float4 rows)
MAX_BRANCHES = 16        # the kernel's per-(branch, layer) band table
LAYER_FLOATS = 5920      # one layer's block (layout: the L_* offsets of SRC); a fifth block holds the rest
_ATT_OFF = 5908          # L_ATT: the fp16x3 attention constants of the layer

# ----------------------------------------------------------------------------------------------------------------
# The band attention in the arithmetic of _band_attn_fwd's fp16x3: q, k, v power-of-two scaled from the
# weight-derived bounds and split into fp16 hi + lo; S = (q_hi k_hi^T + (q_lo k_hi^T + q_hi k_lo^T)) * c over the
# same 64-key chunks; band mask; online softmax in base 2 (ex2.approx.ftz, row sums in the Triton kernel's order);
# P x 2^14 split likewise and P V as (p_hi v_hi + (p_lo v_hi + p_hi v_lo)) in fresh accumulators per chunk, folded
# in with acc * alpha + (.); the output divided by l * 2^(14 + ev) (div.full, as Triton emits it). Each warp owns
# the 32 query rows of its threads (two m16 tiles); q/k/v of one head sit in shared memory as fp16 hi|lo rows
# (64 bytes, 16-byte chunks XOR-swizzled so ldmatrix is conflict-free); the output returns through the warp's own
# q rows. Rows past L (a block is a whole number of warps) carry zeros and are never stored.
# Two departures from the Triton kernel's bits, fp32 level (against float64 over 24 ROIs: rate relRMS 4.64e-7 vs
# 4.58e-7, bias +3.78e-7 vs +3.80e-7): (1) the correction products (hi x lo, ~2^-11 of the main term) accumulate
# in fp16, which this GPU's tensor cores run at twice the fp32-accumulate rate; their lo operands are stored
# x 2^-LOSH (2^-6, exact) so no fp16 sum can overflow -- q, k, v, p x 2^14 are <= 2^14 by their scaling, a
# rounded lo part <= 4 and P's truncated one < 8, so a q k correction is <= 2 x 12 x 4 x 2^14 x 2^-6 = 24576 and a
# p v one <= 16 x 8 x 2^14 x 2^-6 = 32768, under fp16's 65504 -- and are scaled back in fp32 (exact) before they
# join the main terms; (2) P's hi part is its value truncated to 11 significant bits rather than rounded (the
# remainder is then exact in fp32 without unpacking hi). 512 windows of 384 frames: 4.33 -> 3.96-4.05 ms.
# ----------------------------------------------------------------------------------------------------------------
SRC = r"""
#define D 24
#define KS 15
#define HALFK 7
#define FF 48
#define HCS 28
#define LAYER_FLOATS 5920
#define L_CLNW 0
#define L_CLNB 24
#define L_WDWT 48
#define L_BDW 408
#define L_WPW 432
#define L_BPW 1008
#define L_ALNW 1032
#define L_ALNB 1056
#define L_WQKV 1080
#define L_BQKV 2808
#define L_WO 2880
#define L_BO 3456
#define L_FLNW 3480
#define L_FLNB 3504
#define L_W1 3528
#define L_B1 4680
#define L_W2T 4728
#define L_B2 5880
#define L_EPS 5904
#define L_ATT 5908
#define X_WIN 0
#define X_BIN 24
#define X_FINW 48
#define X_FINB 72
#define X_W0 96
#define X_B0 672
#define X_EPS 696
#define NEG_INF __int_as_float(0xff800000)
#ifndef MAXT
#define MAXT 384
#endif
#define MAX_BRANCHES 16

__device__ __forceinline__ float gelu_f(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

__device__ __forceinline__ void layer_norm(const float* x, float* out, const float* w, const float* b, float eps) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    #pragma unroll
    for (int k = 0; k < D; k += 4) { s0 += x[k]; s1 += x[k + 1]; s2 += x[k + 2]; s3 += x[k + 3]; }
    const float mu = ((s0 + s1) + (s2 + s3)) / (float)D;
    float xc[D];
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    #pragma unroll
    for (int k = 0; k < D; k += 4) {
        xc[k] = x[k] - mu; xc[k + 1] = x[k + 1] - mu; xc[k + 2] = x[k + 2] - mu; xc[k + 3] = x[k + 3] - mu;
        q0 = fmaf(xc[k], xc[k], q0); q1 = fmaf(xc[k + 1], xc[k + 1], q1);
        q2 = fmaf(xc[k + 2], xc[k + 2], q2); q3 = fmaf(xc[k + 3], xc[k + 3], q3);
    }
    const float rstd = 1.0f / sqrtf(((q0 + q1) + (q2 + q3)) / (float)D + eps);
    #pragma unroll
    for (int k = 0; k < D; ++k) out[k] = b[k] + xc[k] * (rstd * w[k]);
}

// ---- fp16 helpers (round to nearest even, as Triton's .to(float16))
__device__ __forceinline__ unsigned f2h2(float lo, float hi) {        // lo -> bits 0-15, hi -> bits 16-31
    unsigned r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float h2f_lo(unsigned v) {
    float f; asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, l; }" : "=f"(f) : "r"(v)); return f;
}
__device__ __forceinline__ float h2f_hi(unsigned v) {
    float f; asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, h; }" : "=f"(f) : "r"(v)); return f;
}
__device__ __forceinline__ float ex2(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ float div_full(float a, float b) {
    float y; asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y;
}
__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)) : "memory");
}
#ifndef LOSH
#define LOSH 6
#endif
#define LOMUL (1.0f / (float)(1 << LOSH))
#define LOBACK ((float)(1 << LOSH))
__device__ __forceinline__ void mma16816h(unsigned* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// (a, b) >= 0 -> hi = (a, b) truncated to 11 significant bits (exact in fp16 when >= 2^-14; below that the
// conversion rounds), lo = fp16((a - hi_a) 2^-LOSH, (b - hi_b) 2^-LOSH)
__device__ __forceinline__ void split2t(float a, float b, unsigned& hi, unsigned& lo) {
    const float ha = __int_as_float(__float_as_int(a) & 0xffffe000), hb = __int_as_float(__float_as_int(b) & 0xffffe000);
    hi = f2h2(ha, hb);
    lo = f2h2((a - ha) * LOMUL, (b - hb) * LOMUL);
}
// (a, b) -> packed hi = fp16(a, b), lo = fp16((a - hi_a) 2^-LOSH, (b - hi_b) 2^-LOSH)
__device__ __forceinline__ void split2s(float a, float b, unsigned& hi, unsigned& lo) {
    hi = f2h2(a, b);
    lo = f2h2((a - h2f_lo(hi)) * LOMUL, (b - h2f_hi(hi)) * LOMUL);
}
__device__ __forceinline__ void mma16816(float* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// One 64-byte row per time step: chunk 0 = hi lanes 0-7, 1 = hi 8-15, 2 = lo 0-7, 3 = lo 8-15, the chunks
// XOR-swizzled by (row >> 1) & 3 so the 8 rows of an ldmatrix hit 8 distinct 16-byte bank groups.
__device__ __forceinline__ unsigned char* chunk_ptr(unsigned char* base, int row, int c) {
    return base + row * 64 + ((c ^ ((row >> 1) & 3)) << 4);
}
__device__ __forceinline__ const unsigned char* chunk_ptr(const unsigned char* base, int row, int c) {
    return base + row * 64 + ((c ^ ((row >> 1) & 3)) << 4);
}
// 12 values (x mul, exact: a power of two) -> one q or k row, lanes 12-15 zero
__device__ __forceinline__ void store_split(unsigned char* base, int row, const float* v, float mul) {
    unsigned hi[8], lo[8];
    #pragma unroll
    for (int i = 0; i < 6; ++i) split2s(v[2 * i] * mul, v[2 * i + 1] * mul, hi[i], lo[i]);
    hi[6] = hi[7] = lo[6] = lo[7] = 0u;
    *reinterpret_cast<uint4*>(chunk_ptr(base, row, 0)) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
    *reinterpret_cast<uint4*>(chunk_ptr(base, row, 1)) = make_uint4(hi[4], hi[5], hi[6], hi[7]);
    *reinterpret_cast<uint4*>(chunk_ptr(base, row, 2)) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
    *reinterpret_cast<uint4*>(chunk_ptr(base, row, 3)) = make_uint4(lo[4], lo[5], lo[6], lo[7]);
}
// V rows for the packed P V products (attend_tile): chunk 0 = hi lanes 0-7, 1 = hi 8-11 | lo 0-3, 2 = lo 4-11,
// 3 = hi 8-11 | zero -- the 12 value lanes' hi and lo parts fill n tiles of 8 with one tile to spare
__device__ __forceinline__ void store_split_v(unsigned char* base, int row, const float* v, float mul) {
    unsigned hi[6], lo[6];
    #pragma unroll
    for (int i = 0; i < 6; ++i) split2s(v[2 * i] * mul, v[2 * i + 1] * mul, hi[i], lo[i]);
    *reinterpret_cast<uint4*>(chunk_ptr(base, row, 0)) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
    *reinterpret_cast<uint4*>(chunk_ptr(base, row, 1)) = make_uint4(hi[4], hi[5], lo[0], lo[1]);
    *reinterpret_cast<uint4*>(chunk_ptr(base, row, 2)) = make_uint4(lo[2], lo[3], lo[4], lo[5]);
    *reinterpret_cast<uint4*>(chunk_ptr(base, row, 3)) = make_uint4(hi[4], hi[5], 0u, 0u);
}
__device__ __forceinline__ void store_zero(unsigned char* base, int row) {
    uint4* p = reinterpret_cast<uint4*>(base + row * 64);
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    p[0] = z; p[1] = z; p[2] = z; p[3] = z;
}

// One m16 query tile (rows row0..row0+15) of one head; the result (fp32, 16 lanes) overwrites the tile's q rows.
__device__ __forceinline__ void attend_tile(unsigned char* qo, const unsigned char* kb, const unsigned char* vb,
                                            int row0, int L, int half, float qk_const, float out_div, int lane)
{
    const int g = lane >> 2, tig = lane & 3;
    unsigned qh[4], ql[4];
    {
        const int rr = row0 + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldsm_x4(qh, chunk_ptr(qo, rr, lane >> 4));
        ldsm_x4(ql, chunk_ptr(qo, rr, (lane >> 4) + 2));
    }
    float acc[2][4];
    #pragma unroll
    for (int d = 0; d < 2; ++d) { acc[d][0] = 0.0f; acc[d][1] = 0.0f; acc[d][2] = 0.0f; acc[d][3] = 0.0f; }
    float m_i[2] = {NEG_INF, NEG_INF}, l_i[2] = {0.0f, 0.0f};
    // the key chunks of the Triton program (64-row block) holding this tile; chunks outside the tile's own
    // band leave every row's state unchanged there (m, l, acc: exact identities), so they are skipped
    const int start_m = row0 & ~63;
    const int lo = max(start_m - half, 0) & ~63;
    const int hi = min(start_m + 64 + half, L);
    const int t_lo = row0 - half, t_hi = row0 + 15 + half;
    for (int n0 = lo; n0 < hi; n0 += 64) {
        if (n0 + 63 < t_lo || n0 > t_hi) continue;
        // 16-key blocks outside every row's band get p = 0 exactly: their products are skipped (adding the
        // zero products would not change any accumulator)
        bool need[4];
        #pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
            const int b0 = n0 + 16 * jp;
            need[jp] = b0 <= t_hi && b0 + 15 >= t_lo && b0 < L;
        }
        float s[8][4];
        #pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
            if (!need[jp]) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) { s[2 * jp][e] = NEG_INF; s[2 * jp + 1][e] = NEG_INF; }
                continue;
            }
            unsigned kh[4], kl[4];
            const int kk = n0 + 16 * jp + (lane & 7) + ((lane >> 4) & 1) * 8;
            const int ch = (lane >> 3) & 1;
            ldsm_x4(kh, chunk_ptr(kb, kk, ch));
            ldsm_x4(kl, chunk_ptr(kb, kk, ch + 2));
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                float hh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                unsigned sh[2] = {0u, 0u};
                mma16816h(sh, ql, kh[2 * t], kh[2 * t + 1]);
                mma16816h(sh, qh, kl[2 * t], kl[2 * t + 1]);
                mma16816(hh, qh, kh[2 * t], kh[2 * t + 1]);
                const float sc[4] = {h2f_lo(sh[0]) * LOBACK, h2f_hi(sh[0]) * LOBACK,
                                     h2f_lo(sh[1]) * LOBACK, h2f_hi(sh[1]) * LOBACK};
                #pragma unroll
                for (int e = 0; e < 4; ++e) s[2 * jp + t][e] = (hh[e] + sc[e]) * qk_const;
            }
        }
        if (!(n0 >= row0 + 15 - half && n0 + 63 <= row0 + half && n0 + 64 <= L)) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int key = n0 + 8 * j + tig * 2 + (e & 1);
                    const int row = row0 + g + (e >> 1) * 8;
                    if (!(key >= row - half && key <= row + half && key < L)) s[j][e] = NEG_INF;
                }
            }
        }
        float mx[2] = {NEG_INF, NEG_INF};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            mx[0] = fmaxf(mx[0], fmaxf(s[j][0], s[j][1]));
            mx[1] = fmaxf(mx[1], fmaxf(s[j][2], s[j][3]));
        }
        float msafe[2], alpha[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
            const float mnew = fmaxf(m_i[i], mx[i]);
            msafe[i] = mnew == NEG_INF ? 0.0f : mnew;
            alpha[i] = ex2(m_i[i] - msafe[i]);
            m_i[i] = mnew;
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) s[j][e] = ex2(s[j][e] - msafe[e >> 1]);
        }
        float ls[2];
        ls[0] = s[0][0] + s[0][1];
        ls[1] = s[0][2] + s[0][3];
        #pragma unroll
        for (int j = 1; j < 8; ++j) {
            ls[0] = (ls[0] + s[j][0]) + s[j][1];
            ls[1] = (ls[1] + s[j][2]) + s[j][3];
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float t2 = ls[i] + __shfl_xor_sync(0xffffffffu, ls[i], 2);
            const float t1 = t2 + __shfl_xor_sync(0xffffffffu, t2, 1);
            l_i[i] = fmaf(l_i[i], alpha[i], t1);
        }
        // A fragments of P (16 rows x 16 keys per k step): (row g, keys lo), (row g+8, keys lo), (g, hi), (g+8, hi)
        unsigned ph[4][4], pl[4][4];
        #pragma unroll
        for (int kj = 0; kj < 4; ++kj) {
            split2t(s[2 * kj][0] * 16384.0f, s[2 * kj][1] * 16384.0f, ph[kj][0], pl[kj][0]);
            split2t(s[2 * kj][2] * 16384.0f, s[2 * kj][3] * 16384.0f, ph[kj][1], pl[kj][1]);
            split2t(s[2 * kj + 1][0] * 16384.0f, s[2 * kj + 1][1] * 16384.0f, ph[kj][2], pl[kj][2]);
            split2t(s[2 * kj + 1][2] * 16384.0f, s[2 * kj + 1][3] * 16384.0f, ph[kj][3], pl[kj][3]);
        }
        // P V in five fresh accumulators per k step (V rows in store_split_v's layout):
        //   t0 = p_hi v_hi lanes 0-7          t1 = p_hi v_hi lanes 8-11 | p_hi v_lo lanes 0-3
        //   t2 = p_hi v_lo lanes 4-11         t3 = p_lo v_hi lanes 0-7   t4 = p_lo v_hi lanes 8-11 | 0
        // (the Triton kernel: p_hi v_hi, and p_lo v_hi then p_hi v_lo chained in one accumulator, over 16-lane
        // tiles with lanes 12-15 zero -- six products per k step)
        float t0[4], t1[4];
        unsigned u2[2] = {0u, 0u}, u3[2] = {0u, 0u}, u4[2] = {0u, 0u};
        #pragma unroll
        for (int e = 0; e < 4; ++e) { t0[e] = 0.0f; t1[e] = 0.0f; }
        #pragma unroll
        for (int kj = 0; kj < 4; ++kj) {
            if (!need[kj]) continue;
            unsigned va[4], vc[4];
            const int kr = n0 + 16 * kj + (lane & 7) + ((lane >> 3) & 1) * 8;
            ldsm_x4_t(va, chunk_ptr(vb, kr, lane >> 4));             // chunks 0, 1
            ldsm_x4_t(vc, chunk_ptr(vb, kr, (lane >> 4) + 2));       // chunks 2, 3
            mma16816(t0, ph[kj], va[0], va[1]);
            mma16816(t1, ph[kj], va[2], va[3]);
            mma16816h(u2, ph[kj], vc[0], vc[1]);
            mma16816h(u3, pl[kj], va[0], va[1]);
            mma16816h(u4, pl[kj], vc[2], vc[3]);
        }
        const float t2[4] = {h2f_lo(u2[0]) * LOBACK, h2f_hi(u2[0]) * LOBACK, h2f_lo(u2[1]) * LOBACK, h2f_hi(u2[1]) * LOBACK};
        const float t3[4] = {h2f_lo(u3[0]) * LOBACK, h2f_hi(u3[0]) * LOBACK, h2f_lo(u3[1]) * LOBACK, h2f_hi(u3[1]) * LOBACK};
        const float t4[4] = {h2f_lo(u4[0]) * LOBACK, h2f_hi(u4[0]) * LOBACK, h2f_lo(u4[1]) * LOBACK, h2f_hi(u4[1]) * LOBACK};
        // t1 holds v_hi lanes 8-11 in its columns 0-3 and the (scaled) v_lo lanes 0-3 in its columns 4-7: the
        // threads with tig >= 2 own columns 4-7
        if (tig >= 2) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) t1[e] *= LOBACK;
        }
        // back to the accumulator layout (acc[0]: value lanes 2tig, 2tig+1; acc[1]: lanes 8+2tig, 9+2tig for
        // tig < 2): p_hi v_lo of lanes 0-3 (t1, columns 4-7) and 4-11 (t2) sits in the thread two over
        const bool up = tig >= 2;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float cb0 = __shfl_xor_sync(0xffffffffu, up ? t1[e] : t2[e], 2);
            const float cb1 = __shfl_xor_sync(0xffffffffu, t2[e], 2);
            acc[0][e] = fmaf(acc[0][e], alpha[e >> 1], t0[e] + (t3[e] + cb0));
            if (!up) acc[1][e] = fmaf(acc[1][e], alpha[e >> 1], t1[e] + (t4[e] + cb1));
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float den = l_i[i] * out_div;
        float* orow = reinterpret_cast<float*>(qo + (row0 + g + 8 * i) * 64);
        #pragma unroll
        for (int d = 0; d < 2; ++d)
            *reinterpret_cast<float2*>(orow + d * 8 + tig * 2) =
                make_float2(div_full(acc[d][2 * i], den), div_full(acc[d][2 * i + 1], den));
    }
}

struct Halves { int v[MAX_BRANCHES * 4]; };

extern "C" __global__ void __launch_bounds__(MAXT, 1) setsuna_encoder(
        const float* __restrict__ trace,     // (B, L) window traces
        const float* __restrict__ ind,       // (nbr, B, D) indicator + nu + fps embedding per window
        const float* __restrict__ cell,      // (nbr, B, D) cell-type embedding per window
        const float* __restrict__ pos,       // (L, D) time position
        float* __restrict__ h1_out,          // (nbr, B*L, 32) head-prologue slab, 24 + 8 zero lanes
        const float* __restrict__ wall,      // (nbr, 5, LAYER_FLOATS): four layer blocks + the extra block
        int B, int L, int LP,                // LP: L rounded up to 64 (rows of the q/k/v buffers)
        const Halves halves)                 // band half-width of each (branch, layer) attention
{
    extern __shared__ __align__(16) unsigned char smraw[];
    float* w = reinterpret_cast<float*>(smraw);
    unsigned char* qo = smraw + LAYER_FLOATS * 4;
    unsigned char* kb = qo + LP * 64;
    unsigned char* vb = kb + LP * 64;
    float* hc = reinterpret_cast<float*>(qo);                     // conv rows, aliasing the attention buffers
    const int b = blockIdx.x, br = blockIdx.y, r = threadIdx.x, lane = r & 31, warp = r >> 5;
    const int NT = blockDim.x;
    const bool live = r < L;
    const float* wb = wall + (size_t)br * 5 * LAYER_FLOATS;
    const float* xw = wb + 4 * LAYER_FLOATS;
    float x[D];
    if (live) {
        const float t = trace[(size_t)b * L + r];
        const float* ip = ind + ((size_t)br * B + b) * D;
        const float* cp = cell + ((size_t)br * B + b) * D;
        const float* pp = pos + (size_t)r * D;
        #pragma unroll
        for (int c = 0; c < D; ++c)
            x[c] = ((fmaf(t, __ldg(xw + X_WIN + c), __ldg(xw + X_BIN + c)) + __ldg(ip + c)) + __ldg(cp + c)) + __ldg(pp + c);
    } else {
        #pragma unroll
        for (int c = 0; c < D; ++c) x[c] = 0.0f;
    }
    for (int layer = 0; layer < 4; ++layer) {
        __syncthreads();                                  // the previous layer's readers of w / the buffers are done
        {
            const float4* src = reinterpret_cast<const float4*>(wb + (size_t)layer * LAYER_FLOATS);
            float4* dst = reinterpret_cast<float4*>(w);
            for (int i = r; i < LAYER_FLOATS / 4; i += NT) dst[i] = src[i];
        }
        __syncthreads();
        float hn[D];
        // ---- ConvBlock: x = x + pw(gelu(dw(LN(x))))
        layer_norm(x, hn, w + L_CLNW, w + L_CLNB, w[L_EPS + 0]);
        {
            float4* hrow = reinterpret_cast<float4*>(hc + r * HCS);
            #pragma unroll
            for (int i = 0; i < D / 4; ++i) hrow[i] = make_float4(hn[i * 4], hn[i * 4 + 1], hn[i * 4 + 2], hn[i * 4 + 3]);
        }
        __syncthreads();
        {
            float acc[D];
            #pragma unroll
            for (int c = 0; c < D; ++c) acc[c] = 0.0f;
            for (int t = 0; t < KS; ++t) {
                const int rr = r + t - HALFK;
                if (rr >= 0 && rr < L) {
                    const float4* hp = reinterpret_cast<const float4*>(hc + rr * HCS);
                    const float4* wp = reinterpret_cast<const float4*>(w + L_WDWT + t * D);
                    #pragma unroll
                    for (int i = 0; i < D / 4; ++i) {
                        const float4 hv = hp[i];
                        const float4 wv = wp[i];
                        acc[i * 4 + 0] = fmaf(hv.x, wv.x, acc[i * 4 + 0]);
                        acc[i * 4 + 1] = fmaf(hv.y, wv.y, acc[i * 4 + 1]);
                        acc[i * 4 + 2] = fmaf(hv.z, wv.z, acc[i * 4 + 2]);
                        acc[i * 4 + 3] = fmaf(hv.w, wv.w, acc[i * 4 + 3]);
                    }
                }
            }
            float gg[D];
            #pragma unroll
            for (int c = 0; c < D; ++c) gg[c] = gelu_f(acc[c] + w[L_BDW + c]);
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                float a = 0.0f;
                #pragma unroll
                for (int k = 0; k < D; ++k) a = fmaf(gg[k], w[L_WPW + j * D + k], a);
                x[j] = x[j] + (a + w[L_BPW + j]);
            }
        }
        // ---- LocalMHABlock: x = x + attn(LN(x)) @ Wo^T + bo, two heads of 12 lanes, band |m - n| <= half
        const int half = halves.v[br * 4 + layer];
        const float q_mul = w[L_ATT + 0], k_mul = w[L_ATT + 1], v_mul = w[L_ATT + 2];
        const float qk_const = w[L_ATT + 3], out_div = w[L_ATT + 4];
        float o[D];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            layer_norm(x, hn, w + L_ALNW, w + L_ALNB, w[L_EPS + 1]);   // per head (same bits): fewer live registers
            float q[12], k[12], v[12];
            #pragma unroll
            for (int l = 0; l < 12; ++l) {
                float aq = 0.0f, ak = 0.0f, av = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < D; ++kk) {
                    aq = fmaf(hn[kk], w[L_WQKV + (h * 12 + l) * D + kk], aq);
                    ak = fmaf(hn[kk], w[L_WQKV + (24 + h * 12 + l) * D + kk], ak);
                    av = fmaf(hn[kk], w[L_WQKV + (48 + h * 12 + l) * D + kk], av);
                }
                q[l] = aq + w[L_BQKV + h * 12 + l];
                k[l] = ak + w[L_BQKV + 24 + h * 12 + l];
                v[l] = av + w[L_BQKV + 48 + h * 12 + l];
            }
            __syncthreads();                              // conv rows (h = 0) / the previous head (h = 1) consumed
            if (live) {
                store_split(qo, r, q, q_mul);
                store_split(kb, r, k, k_mul);
                store_split_v(vb, r, v, v_mul);
            } else {
                store_zero(qo, r); store_zero(kb, r); store_zero(vb, r);
            }
            for (int rr = NT + r; rr < LP; rr += NT) { store_zero(qo, rr); store_zero(kb, rr); store_zero(vb, rr); }
            __syncthreads();
            attend_tile(qo, kb, vb, warp * 32, L, half, qk_const, out_div, lane);
            attend_tile(qo, kb, vb, warp * 32 + 16, L, half, qk_const, out_div, lane);
            __syncwarp();
            {
                const float4* orow = reinterpret_cast<const float4*>(qo + r * 64);
                const float4 a0 = orow[0], a1 = orow[1], a2 = orow[2];
                o[h * 12 + 0] = a0.x; o[h * 12 + 1] = a0.y; o[h * 12 + 2] = a0.z; o[h * 12 + 3] = a0.w;
                o[h * 12 + 4] = a1.x; o[h * 12 + 5] = a1.y; o[h * 12 + 6] = a1.z; o[h * 12 + 7] = a1.w;
                o[h * 12 + 8] = a2.x; o[h * 12 + 9] = a2.y; o[h * 12 + 10] = a2.z; o[h * 12 + 11] = a2.w;
            }
        }
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            float a = 0.0f;
            #pragma unroll
            for (int k = 0; k < D; ++k) a = fmaf(o[k], w[L_WO + j * D + k], a);
            x[j] = x[j] + (a + w[L_BO + j]);
        }
        // ---- FFNBlock: x = x + lin2(gelu(lin1(LN(x))))
        layer_norm(x, hn, w + L_FLNW, w + L_FLNB, w[L_EPS + 2]);
        float y2[D];
        #pragma unroll
        for (int i = 0; i < D; ++i) y2[i] = 0.0f;
        for (int j = 0; j < FF; j += 4) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            #pragma unroll
            for (int k = 0; k < D; ++k) {
                a0 = fmaf(hn[k], w[L_W1 + (j + 0) * D + k], a0);
                a1 = fmaf(hn[k], w[L_W1 + (j + 1) * D + k], a1);
                a2 = fmaf(hn[k], w[L_W1 + (j + 2) * D + k], a2);
                a3 = fmaf(hn[k], w[L_W1 + (j + 3) * D + k], a3);
            }
            const float g0 = gelu_f(a0 + w[L_B1 + j]);
            const float g1 = gelu_f(a1 + w[L_B1 + j + 1]);
            const float g2 = gelu_f(a2 + w[L_B1 + j + 2]);
            const float g3 = gelu_f(a3 + w[L_B1 + j + 3]);
            #pragma unroll
            for (int i = 0; i < D; ++i) {
                float vv = fmaf(g0, w[L_W2T + (j + 0) * D + i], y2[i]);
                vv = fmaf(g1, w[L_W2T + (j + 1) * D + i], vv);
                vv = fmaf(g2, w[L_W2T + (j + 2) * D + i], vv);
                y2[i] = fmaf(g3, w[L_W2T + (j + 3) * D + i], vv);
            }
        }
        #pragma unroll
        for (int i = 0; i < D; ++i) x[i] = x[i] + (y2[i] + w[L_B2 + i]);
    }
    if (!live) return;
    // final LayerNorm + head prologue (sr_head[0] + exact GELU)
    float hn[D];
    layer_norm(x, hn, xw + X_FINW, xw + X_FINB, __ldg(xw + X_EPS));
    float4* h4 = reinterpret_cast<float4*>(h1_out + ((size_t)br * B * L + (size_t)b * L + r) * 32);
    #pragma unroll
    for (int i = 0; i < D / 4; ++i) {
        float vv[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = i * 4 + u;
            float a = 0.0f;
            #pragma unroll
            for (int k = 0; k < D; ++k) a = fmaf(hn[k], __ldg(xw + X_W0 + j * D + k), a);
            vv[u] = gelu_f(a + __ldg(xw + X_B0 + j));
        }
        h4[i] = make_float4(vv[0], vv[1], vv[2], vv[3]);
    }
    h4[6] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    h4[7] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}
"""


# ----------------------------------------------------------------------------------------------------------------
# The graphed batch's prologue in one kernel: every window of a ROI batch shares the noise level, frame rate,
# indicator and cell type, so each branch's embedding row -- (indicator embedding + nu projection) + fps projection,
# the nu projection a K = 1 linear layer (fma(w, log nu, b), as cuBLAS computes it) -- is computed once and
# broadcast to the batch's rows: one launch for the ~20 small torch kernels per batch, bitwise equal. It can also
# write each model's gate -- sigmoid(Linear(16->1)(GELU(Linear(2->16)([log nu, log fps])))) -- but the encoder
# does not ask for it: cuBLAS takes those two small dot products in an order that depends on the batch size, which
# this kernel does not reproduce (measured vs float64: rate relRMS 4.9e-7 instead of 4.6e-7), so the gates stay
# the models' own ops.
# ----------------------------------------------------------------------------------------------------------------
PROLOGUE_SRC = r"""
#define D 24
#define GH 16
__device__ __forceinline__ float gelu_f(float v) {
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}
extern "C" __global__ void setsuna_prologue(const float* __restrict__ nu, const float* __restrict__ fps,
                                            const float* __restrict__ emb, const float* __restrict__ fterm,
                                            const float* __restrict__ nuw, const float* __restrict__ nub,
                                            const float* __restrict__ gw1, const float* __restrict__ gb1,
                                            const float* __restrict__ gw2, const float* __restrict__ gb2,
                                            float* __restrict__ ind, float* __restrict__ gate, int B, int nbr, int nmod)
{
    __shared__ float gs[16];
    const float nv = nu[0], fv = fps[0];
    const float lnu = logf(nv < 1e-3f ? 1e-3f : nv);
    if (threadIdx.x < nmod) {
        const int i = threadIdx.x;
        const float lfps = logf(fv < 1e-3f ? 1e-3f : fv);
        float o = 0.0f;
        for (int j = 0; j < GH; ++j) {
            const float h = __fadd_rn(__fadd_rn(__fmul_rn(gw1[(i * GH + j) * 2], lnu),
                                                __fmul_rn(gw1[(i * GH + j) * 2 + 1], lfps)), gb1[i * GH + j]);
            o = fmaf(gw2[i * GH + j], gelu_f(h), o);
        }
        o = __fadd_rn(o, gb2[i]);
        gs[i] = 1.0f / (1.0f + expf(-o));
    }
    __syncthreads();
    for (int q = threadIdx.x; q < nmod * B; q += blockDim.x) gate[q] = gs[q / B];
    const int total = nbr * B * D;
    for (int q = threadIdx.x; q < total; q += blockDim.x) {
        const int c = q % D, t = (q / D) / B;
        const float nt = fmaf(nuw[t * D + c], lnu, nub[t * D + c]);
        ind[q] = (emb[t * D + c] + nt) + fterm[t * D + c];
    }
}
"""


def _gate_ok(m) -> bool:
    import torch.nn as nn
    g = getattr(m, "gate_mlp", None)
    return (not bool(getattr(m, "gate_rich", False)) and int(getattr(m, "n_corner", 0)) == 1
            and float(getattr(m, "gate_fps_bias", 0.0)) == 0.0 and isinstance(g, nn.Sequential) and len(g) == 3
            and isinstance(g[0], nn.Linear) and g[0].in_features == 2 and g[0].out_features == 16
            and g[0].bias is not None and isinstance(g[1], nn.GELU)
            and str(getattr(g[1], "approximate", "none")) == "none"
            and isinstance(g[2], nn.Linear) and g[2].in_features == 16 and g[2].out_features == 1
            and g[2].bias is not None)


def _t(x):
    return x.detach().float().reshape(-1)


def _branch_ok(br) -> bool:
    import torch.nn as nn
    if type(br).__name__ != "SRBranch" or not bool(getattr(br, "time_aware_pos", False)):
        return False
    blocks = list(getattr(br, "blocks", []))
    if len(blocks) != 12 or int(getattr(br, "d", 0)) != D:
        return False
    for i in range(4):
        cb, ab, fb = blocks[3 * i: 3 * i + 3]
        if type(cb).__name__ != "ConvBlock" or type(ab).__name__ != "LocalMHABlock" or type(fb).__name__ != "FFNBlock":
            return False
        ba = getattr(ab, "_band_attn", None)
        if ba is None or ba.D != 16 or ba.h != 2 or int(ba.dh) != 12:
            return False
        if ba.prec != "fp16x3" or ba.scale_exps is None:
            return False
        if int(cb.dw.kernel_size[0]) != KS or cb.dw.bias is None or cb.pw.bias is None:
            return False
        if int(fb.lin1.out_features) != FF or fb.lin1.bias is None or fb.lin2.bias is None:
            return False
    head = getattr(br, "sr_head", None)
    return (isinstance(getattr(br, "in_proj", None), nn.Linear) and br.in_proj.in_features == 1
            and isinstance(head, nn.Sequential) and isinstance(head[0], nn.Linear) and head[0].in_features == D
            and head[0].out_features == D and head[0].bias is not None and isinstance(head[1], nn.GELU)
            and str(getattr(head[1], "approximate", "none")) == "none")


def pack_branch(br) -> torch.Tensor:
    """(5, LAYER_FLOATS) float32: the four layer blocks and the extra block of one SRBranch (layout of SRC). The
    attention constants are the values _band_attention_triton passes its kernel: q/k/v multipliers,
    qk scale x log2 e x 2^-(eq+ek), 2^(14+ev)."""
    dev = br.in_proj.weight.device
    out = torch.zeros((5, LAYER_FLOATS), dtype=torch.float32, device=dev)
    blocks = list(br.blocks)

    def put(s, off, t):
        v = _t(t)
        out[s, off:off + v.numel()] = v

    with torch.no_grad():
        for l in range(4):
            cb, ab, fb = blocks[3 * l: 3 * l + 3]
            put(l, 0, cb.ln.weight); put(l, 24, cb.ln.bias)
            put(l, 48, cb.dw.weight.reshape(D, KS).t().contiguous())    # (tap, channel)
            put(l, 408, cb.dw.bias)
            put(l, 432, cb.pw.weight.reshape(D, D)); put(l, 1008, cb.pw.bias)
            put(l, 1032, ab.ln.weight); put(l, 1056, ab.ln.bias)
            put(l, 1080, ab.attn.in_proj_weight); put(l, 2808, ab.attn.in_proj_bias)   # (72, 24): q | k | v
            put(l, 2880, ab.attn.out_proj.weight); put(l, 3456, ab.attn.out_proj.bias)
            put(l, 3480, fb.ln.weight); put(l, 3504, fb.ln.bias)
            put(l, 3528, fb.lin1.weight); put(l, 4680, fb.lin1.bias)
            put(l, 4728, fb.lin2.weight.t().contiguous()); put(l, 5880, fb.lin2.bias)   # (hidden, out)
            out[l, 5904] = float(cb.ln.eps)
            out[l, 5905] = float(ab.ln.eps)
            out[l, 5906] = float(fb.ln.eps)
            ba = ab._band_attn
            eq, ek, ev = (int(e) for e in ba.scale_exps)
            out[l, _ATT_OFF + 0] = math.ldexp(1.0, eq)
            out[l, _ATT_OFF + 1] = math.ldexp(1.0, ek)
            out[l, _ATT_OFF + 2] = math.ldexp(1.0, ev)
            out[l, _ATT_OFF + 3] = math.ldexp(float(ba.scale) * 1.4426950408889634, -(eq + ek))
            out[l, _ATT_OFF + 4] = math.ldexp(1.0, 14 + ev)
        put(4, 0, br.in_proj.weight.reshape(-1)); put(4, 24, br.in_proj.bias)
        put(4, 48, br.ln.weight); put(4, 72, br.ln.bias)
        put(4, 96, br.sr_head[0].weight); put(4, 672, br.sr_head[0].bias)
        out[4, 696] = float(br.ln.eps)
    return out


class CudaEncoders:
    """The branch encoders of one ensemble (six SRBranch in the dual-gate models) and their head prologues."""

    MAX_THREADS = 384        # the register budget (<= 168 per thread) and the 99 KB of shared memory

    def __init__(self, models, device):
        self.branches = [br for m in models for br in (m.bulk, m.corner)]
        if not all(_branch_ok(br) for br in self.branches):
            raise _cuda_rt.CudaRtError("unexpected branch structure")
        if len(self.branches) > MAX_BRANCHES:
            raise _cuda_rt.CudaRtError(f"{len(self.branches)} branches")
        self.device = torch.device(device)
        self.wall = torch.stack([pack_branch(br) for br in self.branches]).contiguous()
        self.nbr = len(self.branches)
        self._kernels = {}
        self._consts = {}
        # the fused prologue (PROLOGUE_SRC): nu projections of the branches and the models' gate MLPs
        self._models = list(models)
        self.prologue_ok = False
        try:
            import os as _os
            if (_os.environ.get("JSCOPE_SETSUNA_CUDA_PROLOGUE", "1").strip().lower() not in ("0", "false", "no", "off")
                    and all(_gate_ok(m) for m in models)
                    and all(br.nu_proj.in_features == 1 and br.nu_proj.out_features == D
                            and br.nu_proj.bias is not None for br in self.branches)):
                with torch.no_grad():
                    self.nuw = torch.stack([_t(br.nu_proj.weight) for br in self.branches]).contiguous()
                    self.nub = torch.stack([_t(br.nu_proj.bias) for br in self.branches]).contiguous()
                    self.gw1 = torch.stack([_t(m.gate_mlp[0].weight) for m in models]).contiguous()   # (N, 16*2)
                    self.gb1 = torch.stack([_t(m.gate_mlp[0].bias) for m in models]).contiguous()
                    self.gw2 = torch.stack([_t(m.gate_mlp[2].weight) for m in models]).contiguous()
                    self.gb2 = torch.stack([_t(m.gate_mlp[2].bias) for m in models]).reshape(-1).contiguous()
                self.nmod = len(models)
                self.prologue_ok = self.nmod <= 16
        except Exception:
            self.prologue_ok = False

    @staticmethod
    def geometry(L: int) -> tuple[int, int]:
        """(threads per block, q/k/v buffer rows) for windows of L frames."""
        return (int(L) + 31) // 32 * 32, (int(L) + 63) // 64 * 64

    def supports(self, L: int) -> bool:
        return 16 <= int(L) and self.geometry(L)[0] <= self.MAX_THREADS

    def smem_bytes(self, L: int) -> int:
        nt, lp = self.geometry(L)
        return LAYER_FLOATS * 4 + max(3 * lp * 64, nt * _HC_STRIDE * 4)

    def kernel_for(self, nt: int):
        """The kernel compiled for blocks of ``nt`` threads (the register budget of __launch_bounds__)."""
        k = self._kernels.get(int(nt))
        if k is None:
            k = _cuda_rt.CudaKernel(SRC, "setsuna_encoder", device=self.device, options=(f"-DMAXT={int(nt)}",),
                                    max_dynamic_smem=99 * 1024)
            self._kernels[int(nt)] = k
        return k

    def prologue_inputs(self, br, x, indicator_id, nu, fps, cell_type):
        """(ind (B, D), cell (B, D)): the per-window embedding sums of SRBranch.features' prologue."""
        B = x.size(0)
        ind = br.indicator_emb(indicator_id)
        cell = br.cell_type_emb(cell_type if cell_type is not None
                                else torch.zeros(B, dtype=torch.long, device=x.device))
        if nu is not None:
            ind = ind + br.nu_proj(torch.log(nu.clamp(min=1e-3)).unsqueeze(-1))
        if fps is not None:
            ind = ind + br.fps_proj(torch.log(fps.clamp(min=1e-3)).unsqueeze(-1))
        return ind, cell

    def _session_consts(self, B, L, ids, ind_t, fps_t, ct_t, eff_window):
        """The prologue's parts that depend on the session only (``ids`` = indicator id, cell-type id, fps, which
        ``ind_t`` / ``ct_t`` / ``fps_t`` hold in every row): per branch the indicator embedding and the fps
        projection (rows of the same (B, .) operations prologue_inputs runs -- a gather and a K = 1 linear
        layer, the same values for any row count), the cell-type embedding rows, the time position table and
        the band table. Built on the first eager call for a batch shape (the graph capture's warm-up), None
        while a capture runs without them."""
        key = (int(B), int(L)) + tuple(ids) + (eff_window,)
        c = self._consts.get(key)
        if c is not None or torch.cuda.is_current_stream_capturing():
            return c
        dev = ind_t.device
        mod = sys.modules.get(type(self.branches[0]).__module__)
        lfps = torch.log(fps_t.clamp(min=1e-3)).unsqueeze(-1)
        cell = torch.empty((self.nbr, B, D), dtype=torch.float32, device=dev)
        emb, fterm = [], []
        halves = (ctypes.c_int32 * (MAX_BRANCHES * 4))()
        for t, br in enumerate(self.branches):
            emb.append(br.indicator_emb(ind_t).reshape(B, D)[0])
            fterm.append(br.fps_proj(lfps).reshape(B, D)[0])
            cell[t].copy_(br.cell_type_emb(ct_t if ct_t is not None
                                           else torch.zeros(B, dtype=torch.long, device=dev)).reshape(B, D))
            for i in range(4):
                ab = br.blocks[3 * i + 1]
                window = eff_window if eff_window is not None else ab.local_window
                halves[4 * t + i] = (int(window) // 2) if window is not None else L
        c = {"emb": torch.stack(emb).contiguous(), "fps": torch.stack(fterm).contiguous(), "cell": cell,
             "pos": getattr(mod, "time_aware_sinusoid")(L, D, fps_t.reshape(-1)[:1]).reshape(L, D).contiguous().float(),
             "halves": halves}
        self._consts[key] = c
        return c

    def _prologue_checked(self, device) -> bool:
        """The prologue kernel's gpu_kernel_check verdict (against prologue_inputs' torch ops, bit for bit)."""
        from setsuna._support import gpu_kernel_check as _gkc
        from . import gpu_checks as _gc
        return _gkc.use("setsuna.prologue", device, lambda c, dev: _gc.check_prologue(c, dev, self))

    def prologue_kernel(self):
        k = self._kernels.get("prologue")
        if k is None:
            k = _cuda_rt.CudaKernel(PROLOGUE_SRC, "setsuna_prologue", device=self.device)
            self._kernels["prologue"] = k
        return k

    def h1(self, xb, ind_t, nu_t, fps_t, ct_t, eff_window, H1, ids=None, gate_out=None):
        """Fill H1 (nbr, B*L, 32) -- the head-prologue slabs, 24 lanes + 8 zero -- for the window batch xb (B, L).
        ``ids`` = (indicator id, cell-type id, fps) when every row has these (the graphed ROI batches): the
        session's constant prologue terms come from _session_consts and each batch computes only the noise
        level's projection, ((indicator embedding + nu term) + fps term) as prologue_inputs adds them."""
        B, L = (int(v) for v in xb.shape)
        if not self.supports(L):
            raise _cuda_rt.CudaRtError(f"windows of {L} frames")
        dev = xb.device
        f32 = torch.float32
        cst = (self._session_consts(B, L, ids, ind_t, fps_t, ct_t, eff_window)
               if (ids is not None and nu_t is not None and fps_t is not None) else None)
        if cst is not None:
            ind = torch.empty((self.nbr, B, D), dtype=f32, device=dev)
            if self.prologue_ok and self._prologue_checked(nu_t.device):
                # the embedding rows in one kernel (every window of the batch shares nu and fps); the gates stay
                # the models' own ops unless asked for (their cuBLAS dot orders depend on the batch size)
                self.prologue_kernel().launch((1,), (1024,), 0,
                                              (nu_t, fps_t, cst["emb"], cst["fps"], self.nuw, self.nub, self.gw1,
                                               self.gb1, self.gw2, self.gb2, ind,
                                               gate_out if gate_out is not None else ind,
                                               (B, "i"), (self.nbr, "i"),
                                               (self.nmod if gate_out is not None else 0, "i")))
            else:
                lnu = torch.log(nu_t.clamp(min=1e-3)).unsqueeze(-1)
                for t, br in enumerate(self.branches):
                    torch.add(cst["emb"][t] + br.nu_proj(lnu).reshape(B, D), cst["fps"][t], out=ind[t])
            nt, lp = self.geometry(L)
            self.kernel_for(nt).launch((B, self.nbr), (nt,), self.smem_bytes(L),
                                       (xb.contiguous().float(), ind, cst["cell"], cst["pos"], H1, self.wall,
                                        (B, "i"), (L, "i"), (lp, "i"), cst["halves"]))
            return H1
        if gate_out is not None:                  # no session constants yet: the gates by the models' own ops
            for i, m in enumerate(self._models):
                gate_out[i].copy_(m.compute_gate(nu_t, fps_t, ind_t, ct_t))
        mod = sys.modules.get(type(self.branches[0]).__module__)
        pos = getattr(mod, "time_aware_sinusoid")(L, D, fps_t.reshape(-1)[:1]).reshape(L, D).contiguous().float()
        ind = torch.empty((self.nbr, B, D), dtype=f32, device=dev)
        cell = torch.empty((self.nbr, B, D), dtype=f32, device=dev)
        halves = (ctypes.c_int32 * (MAX_BRANCHES * 4))()
        for t, br in enumerate(self.branches):
            a, c = self.prologue_inputs(br, xb, ind_t, nu_t, fps_t, ct_t)
            ind[t].copy_(a.reshape(B, D))
            cell[t].copy_(c.reshape(B, D))
            for i in range(4):
                ab = br.blocks[3 * i + 1]
                window = eff_window if eff_window is not None else ab.local_window
                halves[4 * t + i] = (int(window) // 2) if window is not None else L
        nt, lp = self.geometry(L)
        self.kernel_for(nt).launch((B, self.nbr), (nt,), self.smem_bytes(L),
                                   (xb.contiguous().float(), ind, cell, pos, H1, self.wall, (B, "i"), (L, "i"),
                                    (lp, "i"), halves))
        return H1


# ----------------------------------------------------------------------------------------------------------------
# The pooled dual head (_head_pool_fwd's contract without the fine binary stream): per row (a frame of a window), per
# fine bin and per model, the bulk and corner count / binary logits (dot products of the 24 head features with the
# head weights, + bias), the gate mix, softplus / sigmoid in the Triton kernel's arithmetic (libdevice exp / log in
# flush-to-zero mode as Triton links them, the log(1 + e) rounding correction, div.full), summed into pooled bins of
# K = 32 or 64. A thread per fine bin holds its four heads' weights in registers; a warp covers 32 bins; the rows
# stream through shared memory (the model's bulk | corner features, read as broadcasts: one LDS.128 per 16 FMAs);
# each warp sums its 32 bins for 8 rows at a time by a shuffle transpose-reduce. Differences to the Triton kernel:
# the sums over models and bins run in another order (fp32 reorder level: relRMS 5-8e-8 on the pooled outputs),
# the dots skip the zero padding. RTX 5090, 512 windows of 384 frames: 1.43 ms (Triton 2.3 ms).
# ----------------------------------------------------------------------------------------------------------------
HEAD_SRC = r"""
#define KD 24
#ifndef NW
#define NW 4                              // warps per block = pooled bins per block (K = 32)
#endif
#define NT (32 * NW)
#ifndef MBINS
#define MBINS 512
#endif
#ifndef NMOD
#define NMOD 3
#endif
#ifndef RB
#define RB 256                            // rows per block
#endif
#ifndef RT
#define RT 64                             // rows per feature tile
#endif
#ifndef MINB
#define MINB 2
#endif
#ifndef KPOOL
#define KPOOL 32
#endif
#define WPP (KPOOL / 32)                  // warps per pooled bin (1 or 2)
#define NPOOL (MBINS / KPOOL)
#define HROW 48                           // floats per staged row: bulk 24 | corner 24
#define NTILE (RB / RT)
#ifndef WRITE_PF
#define WRITE_PF 0                        // also the fine binary stream: the ensemble sum per row and bin (out_pf)
#endif
#ifndef WRITE_PART
#define WRITE_PART 0                      // also each row's count sum over its first row_off[row] bins (out_part)
#endif
#ifndef WRITE_PFS
#define WRITE_PFS 0                       // (with WRITE_PF) the stream itself: rows in pairs, see setsuna_head_pool
#endif
#if WRITE_PFS
#define TILE_MAP(off) (rmap + (off))      // a feature tile's rows in the block's row map
#else
#define TILE_MAP(off) nullptr
#endif

__device__ __forceinline__ float div_full(float a, float b) {
    float y; asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y;
}
// IEEE float32 add and divide whatever -ftz says (the stream mode's writes match CudaSlabAdd's, built without it)
__device__ __forceinline__ float add_ieee(float a, float b) {
    float y; asm("add.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y;
}
__device__ __forceinline__ float div_ieee(float a, float b) {
    float y; asm("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y;
}
__device__ __forceinline__ float max_ieee(float a, float b) {     // subnormals kept (not max.ftz)
    float y; asm("max.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y;
}
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
#else
// before sm_80 (no cp.async): the same 16 bytes copied at once -- the staged tiles' values are the same, only the
// copy no longer overlaps the previous tile's work (every wait is followed by a __syncthreads, as with cp.async)
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    *reinterpret_cast<float4*>(smem) = __ldg(reinterpret_cast<const float4*>(gmem));
}
__device__ __forceinline__ void cp_async_commit() {}
__device__ __forceinline__ void cp_async_wait_all() {}
#endif

// rows [r0, r0 + RT) of model i -- WRITE_PFS: the rows rm[0 .. RT) (-1: none) -- bulk and corner features (24
// each) -> dst[RT][48], each row's gate -> gw[RT] and (WRITE_PART) its count split -> of[RT] (plain stores, visible
// after the next barrier); rows past the end clamp (WRITE_PFS: row 0 stands in for none)
__device__ __forceinline__ void load_rows(float* dst, float* gw, int* of, const float* h1, const float* gate,
                                          const int* row_off, int i, int rows, int r0, const int* rm, int L, int B) {
    if (threadIdx.x < RT) {
#if WRITE_PFS
        const int rq = rm[threadIdx.x], r = rq >= 0 ? rq : 0;
#else
        int r = r0 + threadIdx.x; r = r < rows ? r : rows - 1;
#endif
        gw[threadIdx.x] = __ldg(gate + (size_t)i * B + r / L);
#if WRITE_PART && WRITE_PFS
        of[threadIdx.x] = rq >= 0 ? __ldg(row_off + r) : 0;
#elif WRITE_PART
        of[threadIdx.x] = (r0 + (int)threadIdx.x < rows) ? __ldg(row_off + r) : 0;
#endif
    }
    for (int q = threadIdx.x; q < RT * 12; q += NT) {
        const int rr = q / 12, part = q % 12;           // 6 float4 of bulk, 6 of corner
#if WRITE_PFS
        const int r = rm[rr] >= 0 ? rm[rr] : 0;
#else
        int r = r0 + rr; r = r < rows ? r : rows - 1;
#endif
        const int br = 2 * i + (part >= 6 ? 1 : 0);
        const float* src = h1 + ((size_t)br * rows + r) * 32 + (part % 6) * 4;
        cp_async16(dst + rr * HROW + (part >= 6 ? 24 : 0) + (part % 6) * 4, src);
    }
    cp_async_commit();
}

// sum over the 32 lanes of 8 values per lane (one per row): returns, in lanes whose bits 4..2 select row q, the
// total of row q (lanes with (lane & 3) != 0 hold copies)
__device__ __forceinline__ float transpose_sum8(float* v, int lane) {
    // 16 <-> : keep 4 rows
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool up = (lane & 16) != 0;
        const float send = up ? v[j] : v[j + 4];
        const float got = __shfl_xor_sync(0xffffffffu, send, 16);
        v[j] = (up ? v[j + 4] : v[j]) + got;
    }
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const bool up = (lane & 8) != 0;
        const float send = up ? v[j] : v[j + 2];
        const float got = __shfl_xor_sync(0xffffffffu, send, 8);
        v[j] = (up ? v[j + 2] : v[j]) + got;
    }
    {
        const bool up = (lane & 4) != 0;
        const float send = up ? v[0] : v[1];
        const float got = __shfl_xor_sync(0xffffffffu, send, 4);
        v[0] = (up ? v[1] : v[0]) + got;
    }
    v[0] = v[0] + __shfl_xor_sync(0xffffffffu, v[0], 2);
    v[0] = v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);
    return v[0];
}

extern "C" __global__ void __launch_bounds__(NT, MINB) setsuna_head_pool(
        const float* __restrict__ h1,        // (2 * NMOD, rows, 32)
        const float* __restrict__ wt,        // (2 * NMOD, 2, 32, MBINS): _head_stack's WT (heads count, binary)
        const float* __restrict__ bias,      // (2 * NMOD, 2, MBINS)
        const float* __restrict__ gate,      // (NMOD, B)
        double* __restrict__ out_c,          // (rows, NPOOL)
        double* __restrict__ out_b,
        int rows, int L, int B,
        float* __restrict__ out_pf,          // WRITE_PF: (rows, MBINS)
        const int* __restrict__ row_off,     // WRITE_PART: (rows,)
        double* __restrict__ out_part,       // WRITE_PART: (rows, MBINS / NT), this block's bins' share
        const long long* __restrict__ sp,    // WRITE_PFS: the batch's stream geometry (below)
        float* __restrict__ edge_out)        // WRITE_PFS: (the batch's single rows, MBINS)
{
    // WRITE_PFS: the fine binary stream itself. Frame (region0 + r) * S + f is the average of the two regular windows
    // of the batch covering it, rows (r - 1) * W + S + f and r * W + f (region-local r >= 1): the first sp[6] blocks
    // take RB / 2 such pairs (local rows [0, RB/2) the earlier windows', [RB/2, RB) the later ones') and write
    // (a + b) / d -- a + b from region sp[5] on, where the flush window adds its share later. The other blocks take
    // the rows without a partner in the batch into edge_out, for the caller: window 0's first half (slots [0, S)),
    // the last regular window's second half ([S, 2S)) and the flush window's rows ([2S, 2S + W)). sp: [0] the
    // stream's address, [1] region0, [2] regular windows, [3] S, [4] W, [5] the first region under the flush window,
    // [6] pair blocks, [7] pairs, [8] single rows, [9] the flush window's first row, [10] d's float bits, [11] the
    // chunk maxima's address or 0: per frame and NT-bin chunk (this block's bins) the largest value written, NaN
    // where one is negative or NaN -- for the frames written final (before region sp[5]). The grid may hold more
    // blocks than a batch needs (one graph serves batches with and without the flush window): those leave at once.
#if WRITE_PFS
    __shared__ int rmap[RB];
    __shared__ int slot[RB];
    __shared__ float cred[(RB / 2) * NW];                // per pair and warp: the chunk maxima's partials
#endif
    __shared__ __align__(16) float ht[2][RT * HROW];
    __shared__ float gws[2][RT];
    __shared__ int ofs[2][RT];
    __shared__ float acc[RB * NW * 2];
#if WRITE_PF
    extern __shared__ float pfa[];                       // (RB, NT): the models' binary sum per row and bin
#endif
#if WRITE_PART
    __shared__ float pacc[RB * NW];                      // per row and warp: the models' count sum before the split
#endif
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int bin = blockIdx.x * NT + tid;               // this thread's fine bin
    const int row0 = blockIdx.y * RB;
#if WRITE_PFS
    if ((int)blockIdx.y >= (int)sp[6] && ((int)blockIdx.y - (int)sp[6]) * RB >= (int)sp[8]) return;
    for (int q = tid; q < RB; q += NT) {
        const int S = (int)sp[3], W = (int)sp[4], nreg = (int)sp[2], npb = (int)sp[6];
        int r = -1, s = 0;
        if ((int)blockIdx.y < npb) {
            s = (int)blockIdx.y * (RB / 2) + (q % (RB / 2));
            if (s < (int)sp[7]) {
                const int rloc = 1 + s / S, f = s - (rloc - 1) * S;
                r = q < RB / 2 ? (rloc - 1) * W + S + f : rloc * W + f;
            }
        } else {
            const int s1 = ((int)blockIdx.y - npb) * RB + q;
            if (s1 < (int)sp[8]) {
                s = nreg > 0 ? s1 : s1 + 2 * S;                // without regular windows only the flush window
                r = s < S ? s : (s < 2 * S ? (nreg - 1) * W + S + (s - S) : (int)sp[9] + (s - 2 * S));
            }
        }
        rmap[q] = r;
        slot[q] = s;
    }
    __syncthreads();
#endif
    for (int q = tid; q < RB * NW * 2; q += NT) acc[q] = 0.0f;
#if WRITE_PF
    for (int q = tid; q < RB * NT; q += NT) pfa[q] = 0.0f;
#endif
#if WRITE_PART
    for (int q = tid; q < RB * NW; q += NT) pacc[q] = 0.0f;
#endif
    // which row of an 8-row group this lane's reduced value belongs to (bits 4..2 of the lane)
    const int qrow = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    load_rows(ht[0], gws[0], ofs[0], h1, gate, row_off, 0, rows, row0, TILE_MAP(0), L, B);
    int buf = 0;
    for (int i = 0; i < NMOD; ++i) {
        float wbc[KD], wbb[KD], wcc[KD], wcb[KD];
        #pragma unroll
        for (int k = 0; k < KD; ++k) {
            wbc[k] = __ldg(wt + (((size_t)(2 * i) * 2 + 0) * 32 + k) * MBINS + bin);
            wbb[k] = __ldg(wt + (((size_t)(2 * i) * 2 + 1) * 32 + k) * MBINS + bin);
            wcc[k] = __ldg(wt + (((size_t)(2 * i + 1) * 2 + 0) * 32 + k) * MBINS + bin);
            wcb[k] = __ldg(wt + (((size_t)(2 * i + 1) * 2 + 1) * 32 + k) * MBINS + bin);
        }
        const float bbc = __ldg(bias + ((size_t)(2 * i) * 2 + 0) * MBINS + bin);
        const float bbb = __ldg(bias + ((size_t)(2 * i) * 2 + 1) * MBINS + bin);
        const float bcc = __ldg(bias + ((size_t)(2 * i + 1) * 2 + 0) * MBINS + bin);
        const float bcb = __ldg(bias + ((size_t)(2 * i + 1) * 2 + 1) * MBINS + bin);
        for (int t = 0; t < NTILE; ++t) {
            cp_async_wait_all();
            __syncthreads();
            {
                const int nt_ = i * NTILE + t + 1;
                if (nt_ < NMOD * NTILE)
                    load_rows(ht[buf ^ 1], gws[buf ^ 1], ofs[buf ^ 1], h1, gate, row_off, nt_ / NTILE, rows,
                              row0 + (nt_ % NTILE) * RT, TILE_MAP((nt_ % NTILE) * RT), L, B);
            }
            const float* hs = ht[buf];
            const float* gs = gws[buf];
            for (int g = 0; g < RT; g += 8) {
                float vc[8], vb[8];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const int rl = g + u;                                    // row within the tile
                    const float4* hp = reinterpret_cast<const float4*>(hs + rl * HROW);
                    float lbc = 0.0f, lbb = 0.0f, lcc = 0.0f, lcb = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < 6; ++q) {
                        const float4 a = hp[q];
                        lbc = fmaf(a.x, wbc[4 * q], lbc); lbb = fmaf(a.x, wbb[4 * q], lbb);
                        lbc = fmaf(a.y, wbc[4 * q + 1], lbc); lbb = fmaf(a.y, wbb[4 * q + 1], lbb);
                        lbc = fmaf(a.z, wbc[4 * q + 2], lbc); lbb = fmaf(a.z, wbb[4 * q + 2], lbb);
                        lbc = fmaf(a.w, wbc[4 * q + 3], lbc); lbb = fmaf(a.w, wbb[4 * q + 3], lbb);
                    }
                    #pragma unroll
                    for (int q = 0; q < 6; ++q) {
                        const float4 c = hp[6 + q];
                        lcc = fmaf(c.x, wcc[4 * q], lcc); lcb = fmaf(c.x, wcb[4 * q], lcb);
                        lcc = fmaf(c.y, wcc[4 * q + 1], lcc); lcb = fmaf(c.y, wcb[4 * q + 1], lcb);
                        lcc = fmaf(c.z, wcc[4 * q + 2], lcc); lcb = fmaf(c.z, wcb[4 * q + 2], lcb);
                        lcc = fmaf(c.w, wcc[4 * q + 3], lcc); lcb = fmaf(c.w, wcb[4 * q + 3], lcb);
                    }
                    lbc = lbc + bbc; lbb = lbb + bbb; lcc = lcc + bcc; lcb = lcb + bcb;
                    const float w = gs[rl];
                    const bool small = fabsf(w) < 0.5f;
                    const float coeff = small ? w : w - 1.0f;
                    const float mc = coeff * (lcc - lbc) + (small ? lbc : lcc);
                    const float em = expf(mc);
                    const float um = 1.0f + em;
                    const float sp = logf(um) - ((um - 1.0f) - em) * __fdividef(1.0f, um);
                    vc[u] = mc > 20.0f ? mc : sp;
                    const float mb = coeff * (lcb - lbb) + (small ? lbb : lcb);
                    vb[u] = div_full(1.0f, 1.0f + expf(-mb));
#if WRITE_PF
                    pfa[(t * RT + rl) * NT + tid] += vb[u];              // model by model, as _head_pool_fwd's acc_b
#endif
                }
#if WRITE_PART
                {
                    // rows split by a segment boundary are rare: the masked sums only for a group holding one
                    const int* of = ofs[buf];
                    int anyo = 0;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) anyo |= of[g + u];
                    if (anyo) {
                        float vm[8];
                        #pragma unroll
                        for (int u = 0; u < 8; ++u) vm[u] = bin < of[g + u] ? vc[u] : 0.0f;
                        const float sm = transpose_sum8(vm, lane);
                        if ((lane & 3) == 0) pacc[(t * RT + g + qrow) * NW + warp] += sm;
                    }
                }
#endif
                const float sc = transpose_sum8(vc, lane);
                const float sb = transpose_sum8(vb, lane);
                if ((lane & 3) == 0) {
                    const int ra = t * RT + g + qrow;                        // row within the block
                    acc[(ra * NW + warp) * 2 + 0] += sc;
                    acc[(ra * NW + warp) * 2 + 1] += sb;
                }
            }
            buf ^= 1;
        }
    }
    __syncthreads();
    // (row, this block's pooled bins): NW / WPP consecutive pooled bins per row
    for (int q = tid; q < RB * (NW / WPP); q += NT) {
        const int ra = q / (NW / WPP), w = q % (NW / WPP);
#if WRITE_PFS
        const int r = rmap[ra];
        if (r >= 0) {
#else
        const int r = row0 + ra;
        if (r < rows) {
#endif
            float c = acc[(ra * NW + w * WPP) * 2 + 0], b = acc[(ra * NW + w * WPP) * 2 + 1];
            #pragma unroll
            for (int u = 1; u < WPP; ++u) {
                c = c + acc[(ra * NW + w * WPP + u) * 2 + 0];
                b = b + acc[(ra * NW + w * WPP + u) * 2 + 1];
            }
            const int p = blockIdx.x * (NW / WPP) + w;
            out_c[(size_t)r * NPOOL + p] = (double)c;
            out_b[(size_t)r * NPOOL + p] = (double)b;
        }
    }
#if WRITE_PFS
    {
        float* stream = reinterpret_cast<float*>(sp[0]);
        float* cmax = reinterpret_cast<float*>(sp[11]);
        const int S = (int)sp[3], region0 = (int)sp[1], rtail = (int)sp[5];
        const float d = __int_as_float((int)sp[10]);
        if ((int)blockIdx.y < (int)sp[6]) {
            for (int q = 0; q < RB / 2; ++q) {            // a pair's NT bins per step: coalesced
                if (rmap[q] < 0) continue;
                const int rloc = 1 + slot[q] / S, f = slot[q] - (rloc - 1) * S;
                const float s = add_ieee(pfa[q * NT + tid], pfa[(RB / 2 + q) * NT + tid]);
                const bool fin = region0 + rloc < rtail;
                const float v = fin ? div_ieee(s, d) : s;
                stream[((size_t)(region0 + rloc) * S + f) * MBINS + bin] = v;
                if (cmax != nullptr && fin) {             // (uniform in the block)
                    float m = v;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) m = max_ieee(m, __shfl_xor_sync(0xffffffffu, m, off));
                    const unsigned bad = __ballot_sync(0xffffffffu, !(v >= 0.0f));
                    if (lane == 0) cred[q * NW + warp] = bad ? __int_as_float(0x7fc00000) : m;
                }
            }
            if (cmax != nullptr) {
                __syncthreads();
                if (tid < RB / 2 && rmap[tid] >= 0) {
                    const int rloc = 1 + slot[tid] / S, f = slot[tid] - (rloc - 1) * S;
                    if (region0 + rloc < rtail) {
                        float m = cred[tid * NW];
                        bool bad = m != m;
                        #pragma unroll
                        for (int w = 1; w < NW; ++w) {
                            const float x = cred[tid * NW + w];
                            bad = bad || x != x;
                            m = max_ieee(m, x);
                        }
                        cmax[((size_t)(region0 + rloc) * S + f) * (MBINS / NT) + blockIdx.x] =
                            bad ? __int_as_float(0x7fc00000) : m;
                    }
                }
            }
        } else {
            for (int q = 0; q < RB; ++q)
                if (rmap[q] >= 0) edge_out[(size_t)slot[q] * MBINS + bin] = pfa[q * NT + tid];
        }
    }
#elif WRITE_PF
    for (int ra = 0; ra < RB; ++ra) {                    // a row's NT bins per step: coalesced
        const int r = row0 + ra;
        if (r < rows) out_pf[(size_t)r * MBINS + bin] = pfa[ra * NT + tid];
    }
#endif
#if WRITE_PART
    for (int ra = tid; ra < RB; ra += NT) {
#if WRITE_PFS
        const int r = rmap[ra];
        if (r >= 0) {
#else
        const int r = row0 + ra;
        if (r < rows) {
#endif
            double s = 0.0;
            #pragma unroll
            for (int w = 0; w < NW; ++w) s += (double)pacc[ra * NW + w];
            out_part[(size_t)r * (MBINS / NT) + blockIdx.x] = s;
        }
    }
#endif
}
"""


class CudaHeadPool:
    """The pooled dual head of one ensemble (_head_pool_fwd, without WRITE_LF) as one CUDA kernel per batch."""

    NW = 4          # warps (x 32 fine bins) per block
    RB = 128        # rows per block
    MINB = 3        # blocks per SM: registers capped at 168 (192 unconstrained, two blocks); with 128-row
                    # blocks the last wave is shorter too -- 1.53 -> 1.43 ms per 512 windows, same arithmetic
    RB_PF = 32      # rows per block (and per staged tile) when the fine binary stream is written: its per-row sums
    RT_PF = 32      # over the models stay in shared memory (RB_PF x 128 floats, 16 KB; ~30 KB a block with the
                    # staged features, under the 48 KB default), three blocks per SM as without

    def __init__(self, WT, BIAS, n_models: int, M: int, device):
        """WT (2N, 2, DP, M), BIAS (2N, 2, M): _head_stack's tensors (24 real feature rows, the rest zero)."""
        T, H, DP, Mw = (int(v) for v in WT.shape)
        if T != 2 * int(n_models) or H != 2 or DP != 32 or Mw != int(M) or int(M) % (32 * self.NW):
            raise _cuda_rt.CudaRtError("unexpected head stack")
        if bool((WT[:, :, 24:, :] != 0).any()):
            raise _cuda_rt.CudaRtError("head features wider than 24")
        self.N, self.M = int(n_models), int(M)
        self.device = torch.device(device)
        self.wt = WT.float().contiguous()
        self.bias = BIAS.float().contiguous()
        self._kernels = {}

    def supports(self, K: int) -> bool:
        return int(K) in (32, 64)

    def kernel_for(self, K: int, pf: bool = False, part: bool = False, pfs: bool = False):
        key = (int(K), bool(pf), bool(part), bool(pfs))
        k = self._kernels.get(key)
        if k is None:
            rb = self.RB_PF if pf else self.RB
            rt = (f"-DRT={self.RT_PF}",) if pf else ()
            k = _cuda_rt.CudaKernel(HEAD_SRC, "setsuna_head_pool", device=self.device,
                                    options=("-ftz=true", f"-DNW={self.NW}", f"-DRB={rb}", f"-DMINB={self.MINB}",
                                             f"-DMBINS={self.M}",
                                             f"-DNMOD={self.N}", f"-DKPOOL={int(K)}", f"-DWRITE_PF={int(bool(pf))}",
                                             f"-DWRITE_PART={int(bool(part))}", f"-DWRITE_PFS={int(bool(pfs))}")
                                    + rt)
            self._kernels[key] = k
        return k

    def stream_blocks(self, B: int, W: int) -> tuple[int, int]:
        """The stream mode's (pair blocks, single-row blocks) for batches of B windows of W frames (at most B - 1
        regular windows with a flush window, or B without; S = W / 2): the grid holds the most either needs."""
        S = int(W) // 2
        half = self.RB_PF // 2
        return -(-max(0, int(B) - 1) * S // half), -(-(2 * S + int(W)) // self.RB_PF)

    def stream_geometry(self, stream_ptr: int, region0: int, n_reg: int, flush: bool, W: int, r_tail: int,
                        d: float, cmax_ptr: int = 0) -> list:
        """One batch's ``sp`` row (HEAD_SRC's WRITE_PFS): n_reg regular windows from region0 (their shared frames
        into the float32 stream at stream_ptr, divided by d before region r_tail), then the flush window if any;
        cmax_ptr (0: none) the stream's chunk maxima, (frames, M / (32 NW)) float32, for the frames written final."""
        S = int(W) // 2
        pairs = max(0, int(n_reg) - 1) * S
        singles = (2 * S if int(n_reg) > 0 else 0) + (int(W) if flush else 0)
        d_bits = struct.unpack("<i", struct.pack("<f", float(d)))[0]
        return [int(stream_ptr), int(region0), int(n_reg), S, int(W), int(r_tail), -(-pairs // (self.RB_PF // 2)),
                pairs, singles, int(n_reg) * int(W), d_bits, int(cmax_ptr)]

    def run(self, H1, G, out_c, out_b, rows: int, L: int, K: int, out_pf=None, row_off=None, out_part=None,
            sp=None, edge_out=None, blocks=None) -> None:
        """H1 (2N, rows, 32) fp32 contiguous, G (N, B) fp32 contiguous -> out_c / out_b (rows, M // K) float64.
        ``out_pf`` (rows, M) fp32: also the fine binary stream, the models' sum per bin in model order (as
        _head_pool_fwd's acc_b). ``row_off`` (rows,) int32 with ``out_pf``: ``out_part`` (rows, M // 128) float64
        gets each row's count sum over its first row_off bins, by 128-bin column block (the caller sums them).
        ``sp`` (int64 device tensor, see HEAD_SRC's WRITE_PFS) with ``edge_out`` and ``blocks`` (stream_blocks):
        the stream mode -- the windows' shared frames straight into the stream sp names, the other rows into
        ``edge_out`` (out_pf is then not written)."""
        B = int(G.shape[1])
        pfs = sp is not None
        pf = out_pf is not None or pfs
        part = pf and row_off is not None
        rb = self.RB_PF if pf else self.RB
        nt = 32 * self.NW
        grid_y = (int(blocks[0]) + int(blocks[1])) if pfs else (int(rows) + rb - 1) // rb
        self.kernel_for(K, pf, part, pfs).launch(
            (self.M // nt, grid_y), (nt,), (rb * nt * 4 if pf else 0),
            (H1, self.wt, self.bias, G, out_c, out_b, (int(rows), "i"), (int(L), "i"), (B, "i"),
             (out_pf if (pf and not pfs) else (0, "q")), (row_off if part else (0, "q")),
             (out_part if part else (0, "q")), (sp if pfs else (0, "q")), (edge_out if pfs else (0, "q"))))


# ----------------------------------------------------------------------------------------------------------------
# The pooled tail of one ROI in one pass: the overlap-add sums (float64, K-pooled) divided by the windows covering
# each frame x the models (the regular regions' counts, +1 under the flush window), the per-frame count rate summed
# in the order torch's float64 row sum takes for a power-of-two row (halves, repeatedly), and both pooled streams
# rounded once to float16 (cvt.rn.f16.f64: the correctly rounded value _f64_to_f16_rne computes with a dozen torch
# kernels). A thread per frame; ~25 torch kernels (0.35 ms per ROI) become one (~0.04 ms), bitwise equal.
# ----------------------------------------------------------------------------------------------------------------
TAIL_SRC = r"""
#ifndef PN
#define PN 16
#endif
__device__ __forceinline__ unsigned short f64_to_f16(double x) {
    unsigned short h; asm("cvt.rn.f16.f64 %0, %1;" : "=h"(h) : "d"(x)); return h;
}
extern "C" __global__ void setsuna_pooled_tail(const double* __restrict__ pred, const double* __restrict__ pred_b,
                                               float* __restrict__ rate, unsigned short* __restrict__ lam16,
                                               unsigned short* __restrict__ prob16,
                                               int n, int S, int n_regions, int s0, float n_models)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const int r = f / S;
    double cnt = 0.0;
    if (r < n_regions) cnt = (r == 0 || r == n_regions - 1) ? 1.0 : 2.0;
    if (s0 >= 0 && f >= s0) cnt += 1.0;
    const double div = cnt * (double)n_models;             // small integers: exact
    const double* p = pred + (size_t)f * PN;
    const double* q = pred_b + (size_t)f * PN;
    double v[PN];
    #pragma unroll
    for (int j = 0; j < PN; ++j) {
        v[j] = p[j] / div;
        lam16[(size_t)f * PN + j] = f64_to_f16(v[j]);
        prob16[(size_t)f * PN + j] = f64_to_f16(q[j] / div);
    }
    #pragma unroll
    for (int h = PN / 2; h >= 1; h /= 2) {
        #pragma unroll
        for (int j = 0; j < h; ++j) v[j] = v[j] + v[j + h];
    }
    rate[f] = (float)v[0];
}
"""


class CudaPooledTail:
    """_predict_fine_cuda_pooled's division + predict_fine_outputs' rate and float16 streams as one kernel."""

    def __init__(self, device):
        self.device = torch.device(device)
        self._kernels = {}

    @staticmethod
    def supports(Pn: int) -> bool:
        return int(Pn) in (1, 2, 4, 8, 16, 32)

    def run(self, pred, pred_b, n: int, Pn: int, S: int, n_regions: int, s0: int, n_models: int):
        """pred / pred_b (n * Pn,) float64 undivided sums -> (rate (n,) float32, lam16, prob16 (n * Pn,) float16)."""
        k = self._kernels.get(int(Pn))
        if k is None:
            k = _cuda_rt.CudaKernel(TAIL_SRC, "setsuna_pooled_tail", device=self.device, options=(f"-DPN={int(Pn)}",))
            self._kernels[int(Pn)] = k
        rate = torch.empty((int(n),), dtype=torch.float32, device=pred.device)
        lam16 = torch.empty((int(n) * int(Pn),), dtype=torch.float16, device=pred.device)
        prob16 = torch.empty_like(lam16)
        k.launch(((int(n) + 127) // 128,), (128,), 0,
                 (pred, pred_b, rate, lam16, prob16, (int(n), "i"), (int(S), "i"), (int(n_regions), "i"),
                  (int(s0), "i"), (float(n_models), "f")))
        return rate, lam16, prob16


# ----------------------------------------------------------------------------------------------------------------
# The fine binary stream's overlap-add in one pass: a region's value is the sum of the two half-windows covering it
# divided by the windows x models (IEEE float32, one rounding each: torch.add then div_, the same bits), written
# once -- where the torch path filled the stream with zeros, added, then divided in place (three passes).
# ----------------------------------------------------------------------------------------------------------------
SLAB_SRC = r"""
extern "C" __global__ void setsuna_slab_add_div(const float* __restrict__ a, long long sa, const float* __restrict__ b,
                                                 long long sb, float* __restrict__ out, long long so, long long cols4,
                                                 float d)
{
    // row blockIdx.y: out[r] = (a[r] + b[r]) / d, or the sum alone for d == 0; cols4 float4s a row (16-byte rows)
    const float4* ar = reinterpret_cast<const float4*>(a + (long long)blockIdx.y * sa);
    const float4* br = reinterpret_cast<const float4*>(b + (long long)blockIdx.y * sb);
    float4* orow = reinterpret_cast<float4*>(out + (long long)blockIdx.y * so);
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < cols4; k += (long long)gridDim.x * blockDim.x) {
        const float4 x = ar[k], y = br[k];
        float4 s = make_float4(__fadd_rn(x.x, y.x), __fadd_rn(x.y, y.y), __fadd_rn(x.z, y.z), __fadd_rn(x.w, y.w));
        if (d != 0.0f)
            s = make_float4(__fdiv_rn(s.x, d), __fdiv_rn(s.y, d), __fdiv_rn(s.z, d), __fdiv_rn(s.w, d));
        orow[k] = s;
    }
}
"""


class CudaSlabAdd:
    """setsuna_slab_add_div: out[r] = (a[r] + b[r]) / d over rows of float32 views (d = 0: the sum)."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.k = _cuda_rt.CudaKernel(SLAB_SRC, "setsuna_slab_add_div", device=self.device)

    @staticmethod
    def fits(*views) -> bool:
        """The kernel's layout: 2-D float32 views with unit column stride, rows and starts on 16-byte boundaries."""
        return all(v.dim() == 2 and v.dtype == torch.float32 and v.stride(1) == 1 and v.stride(0) % 4 == 0
                   and v.shape[1] % 4 == 0 and v.data_ptr() % 16 == 0 for v in views)

    def run(self, a, b, out, d: float) -> None:
        rows, cols = int(out.shape[0]), int(out.shape[1])
        if rows == 0:
            return
        cols4 = cols // 4
        self.k.launch((min(-(-cols4 // 256), 64), rows), (256,), 0,
                      (a, (int(a.stride(0)), "q"), b, (int(b.stride(0)), "q"), out, (int(out.stride(0)), "q"),
                       (cols4, "q"), (float(d), "f")))


# ----------------------------------------------------------------------------------------------------------------
# The spike paths' assembly around the head's stream mode in three kernels, for the ~15 torch ops per batch and ~35
# per ROI they replace (each ~1 us of work behind ~3 us of launch gap on the GPU): setsuna_oa_batch adds one batch's
# windows into the pooled sums, the boundary parts and the fine stream's edge regions; setsuna_fine_tail adds the
# flush window and divides the pooled sums and parts per frame (the rate, the per-frame binary mass and the float16
# streams with them); setsuna_fine_tail_pf does the fine stream's edge and tail regions. The same IEEE operations in
# the same order as the torch path (built without -ftz): the same bits.
# ----------------------------------------------------------------------------------------------------------------
FUSE_SRC = r"""
#ifndef PN
#define PN 1
#endif
__device__ __forceinline__ unsigned short f64_to_f16(double x) {
    unsigned short h; asm("cvt.rn.f16.f64 %0, %1;" : "=h"(h) : "d"(x)); return h;
}

// One batch's regular windows (n of them, the first at region j0; S frames a half-window, SP = S * PN pooled
// values, SM = S * M fine bins): pr / prb (B, 2 SP) float64 pooled rows, pp (B, 2 S) float64 parts or null, edges
// (2 S + W, M) float32 (the head's stream mode: window 0's first half, the last regular window's second half, ...)
// into sl / sl_b (regions, SP) and sl_p (regions, S), zero-initialized, and sl_f (regions, S M). A region shared by
// two windows of the batch is their sum; the batch's end regions are added to (the pooled sums and parts), and on
// the fine stream set to 0 + x on their first touch (region j0 + n; region j0 of the first batch).
extern "C" __global__ void setsuna_oa_batch(const double* __restrict__ pr, const double* __restrict__ prb,
                                            const double* __restrict__ pp, const float* __restrict__ edges,
                                            double* __restrict__ sl, double* __restrict__ sl_b, double* __restrict__ sl_p,
                                            float* __restrict__ sl_f, long long SP, long long S, long long SM,
                                            long long j0, long long n, int first)
{
    const long long A = (n - 1) * SP;
    const long long P = pp ? (n + 1) * S : 0;
    const long long total = A + 2 * SP + P + 2 * SM;
    for (long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x; q < total;
         q += (long long)gridDim.x * blockDim.x) {
        long long t = q;
        if (t < A) {                                        // region j0 + i, 1 <= i < n: windows i - 1 and i
            const long long i = 1 + t / SP, e = t - (i - 1) * SP;
            const long long a = (i - 1) * 2 * SP + SP + e, b = i * 2 * SP + e;
            sl[(j0 + i) * SP + e] = pr[a] + pr[b];
            sl_b[(j0 + i) * SP + e] = prb[a] + prb[b];
            continue;
        }
        t -= A;
        if (t < 2 * SP) {
            if (t < SP) {                                   // the last regular window's second half
                const long long d = (j0 + n) * SP + t, s = (n - 1) * 2 * SP + SP + t;
                sl[d] = sl[d] + pr[s];
                sl_b[d] = sl_b[d] + prb[s];
            } else {                                        // window 0's first half
                const long long e = t - SP, d = j0 * SP + e;
                sl[d] = sl[d] + pr[e];
                sl_b[d] = sl_b[d] + prb[e];
            }
            continue;
        }
        t -= 2 * SP;
        if (t < P) {
            if (t < (n - 1) * S) {
                const long long i = 1 + t / S, f = t - (i - 1) * S;
                sl_p[(j0 + i) * S + f] = pp[(i - 1) * 2 * S + S + f] + pp[i * 2 * S + f];
            } else if (t < n * S) {
                const long long f = t - (n - 1) * S, d = (j0 + n) * S + f;
                sl_p[d] = sl_p[d] + pp[(n - 1) * 2 * S + S + f];
            } else {
                const long long f = t - n * S, d = j0 * S + f;
                sl_p[d] = sl_p[d] + pp[f];
            }
            continue;
        }
        t -= P;
        if (t < SM) {
            sl_f[(j0 + n) * SM + t] = 0.0f + edges[SM + t];
        } else {
            const long long e = t - SM;
            sl_f[j0 * SM + e] = (first ? 0.0f : sl_f[j0 * SM + e]) + edges[e];
        }
    }
}

// Per frame: the flush window's pooled rows added (frames from s0; s0 < 0: none), then the pooled sums and parts
// divided by the windows covering the frame times the models ((1 or 2 regular) + 1 under the flush window; the
// frames past the regular windows have the flush window alone); the rate (the pooled row summed in halves, as
// torch's float64 row sum) to float32, and to float64 in rate64 when given; p_frame = the binary row sum x inv_m
// (torch divides by a host scalar through its reciprocal); lam16 / prob16 when given: the pooled values to float16.
extern "C" __global__ void setsuna_fine_tail(double* __restrict__ pred, double* __restrict__ pred_b,
                                             double* __restrict__ part, const double* __restrict__ fl_pr,
                                             const double* __restrict__ fl_prb, const double* __restrict__ fl_pp,
                                             float* __restrict__ rate32, double* __restrict__ rate64,
                                             double* __restrict__ p_frame, unsigned short* __restrict__ lam16,
                                             unsigned short* __restrict__ prob16, int n, int S, int n_regions, int s0,
                                             double n_models, double inv_m)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const int r = f / S;
    double cnt = 0.0;
    if (r < n_regions) cnt = (r == 0 || r == n_regions - 1) ? 1.0 : 2.0;
    const bool fl = s0 >= 0 && f >= s0;
    if (fl) cnt += 1.0;
    const double div = cnt * n_models;
    double v[PN], u[PN];
    #pragma unroll
    for (int j = 0; j < PN; ++j) {
        const size_t k = (size_t)f * PN + j;
        double a = pred[k], b = pred_b[k];
        if (fl) {
            a = a + fl_pr[(size_t)(f - s0) * PN + j];
            b = b + fl_prb[(size_t)(f - s0) * PN + j];
        }
        a = a / div;
        b = b / div;
        pred[k] = a;
        pred_b[k] = b;
        v[j] = a;
        u[j] = b;
        if (lam16) {
            lam16[k] = f64_to_f16(a);
            prob16[k] = f64_to_f16(b);
        }
    }
    if (part) {
        double c = part[f];
        if (fl) c = c + fl_pp[f - s0];
        part[f] = c / div;
    }
    #pragma unroll
    for (int h = PN / 2; h >= 1; h /= 2) {
        #pragma unroll
        for (int j = 0; j < h; ++j) { v[j] = v[j] + v[j + h]; u[j] = u[j] + u[j + h]; }
    }
    rate32[f] = (float)v[0];
    if (rate64) rate64[f] = v[0];
    p_frame[f] = u[0] * inv_m;
}

// The fine stream's frames still to divide, a block per frame: the first E edge regions (region i bw for i <= q =
// n_regular / bw, then n_regular), divided by their window count x the models; then the frames from r_first S on:
// the flush window's rows added (frames from s0; the frames past the regular windows start from 0), divided by
// (window count + 1 under the flush window) x the models. cmax (or null): the frames' chunk maxima as the head's
// stream mode writes them (chunks of blockDim.x bins, NaN where a value is negative or NaN).
extern "C" __global__ void setsuna_fine_tail_pf(float* __restrict__ pf, const float* __restrict__ fl_pf, int S, int M,
                                                int n_regions, int n_regular, int bw, int r_first, int E, int s0,
                                                float n_models, float* __restrict__ cmax)
{
    __shared__ float cred[32][32];                       // [chunk][warp]
    const int idx = blockIdx.x;
    int f;
    float d;
    bool fl = false, beyond = false;
    if (idx < E * S) {
        const int i = idx / S, q = n_regular / bw;
        const int region = i <= q ? i * bw : n_regular;
        f = region * S + (idx - i * S);
        d = ((region == 0 || region == n_regions - 1) ? 1.0f : 2.0f) * n_models;
    } else {
        f = r_first * S + (idx - E * S);
        const int r = f / S;
        float cnt = 0.0f;
        if (r < n_regions) cnt = (r == 0 || r == n_regions - 1) ? 1.0f : 2.0f;
        else beyond = true;
        fl = s0 >= 0 && f >= s0;
        if (fl) cnt += 1.0f;
        d = cnt * n_models;
    }
    float* row = pf + (size_t)f * M;
    const float* frow = fl ? fl_pf + (size_t)(f - s0) * M : nullptr;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = threadIdx.x; b < M; b += blockDim.x) {
        float v = beyond ? 0.0f : row[b];
        if (fl) v = v + frow[b];
        v = v / d;
        row[b] = v;
        if (cmax) {                                      // the chunk of blockDim.x bins this step covers
            float m = v;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
            const unsigned bad = __ballot_sync(0xffffffffu, !(v >= 0.0f));
            if (lane == 0) cred[b / blockDim.x][warp] = bad ? __int_as_float(0x7fc00000) : m;
        }
    }
    if (cmax) {
        __syncthreads();
        const int nc = M / blockDim.x, nw = blockDim.x / 32;
        if ((int)threadIdx.x < nc) {
            float m = cred[threadIdx.x][0];
            bool bad = m != m;
            for (int w = 1; w < nw; ++w) {
                const float x = cred[threadIdx.x][w];
                bad = bad || x != x;
                m = fmaxf(m, x);
            }
            cmax[(size_t)f * nc + threadIdx.x] = bad ? __int_as_float(0x7fc00000) : m;
        }
    }
}
"""


class CudaFineAssembly:
    """setsuna_oa_batch / setsuna_fine_tail / setsuna_fine_tail_pf (FUSE_SRC) for _predict_fine_cuda_pooled."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.oa = _cuda_rt.CudaKernel(FUSE_SRC, "setsuna_oa_batch", device=self.device)
        self.tail_pf = _cuda_rt.CudaKernel(FUSE_SRC, "setsuna_fine_tail_pf", device=self.device)
        self._tails = {}

    @staticmethod
    def supports(Pn: int) -> bool:
        return int(Pn) in (1, 2, 4, 8, 16, 32)

    def batch(self, pr, prb, pp, edges, sl, sl_b, sl_p, sl_f, S: int, Pn: int, M: int, j0: int, n: int) -> None:
        """One batch's n >= 1 regular windows from region j0 (see setsuna_oa_batch)."""
        SP, SM = int(S) * int(Pn), int(S) * int(M)
        total = (int(n) + 1) * SP + ((int(n) + 1) * int(S) if pp is not None else 0) + 2 * SM
        self.oa.launch((min(-(-total // 256), 1024),), (256,), 0,
                       (pr, prb, (pp if pp is not None else (0, "q")), edges, sl, sl_b,
                        (sl_p if sl_p is not None else (0, "q")), sl_f, (SP, "q"), (int(S), "q"), (SM, "q"),
                        (int(j0), "q"), (int(n), "q"), (1 if int(j0) == 0 else 0, "i")))

    def tail(self, pred, pred_b, part, fl_rows, pf, fl_pf, *, n: int, Pn: int, S: int, M: int, n_regions: int,
             n_regular: int, bw: int, s0: int, n_models: int, want16: bool, cmax=None):
        """The ROI's tail: fl_rows = (pr, prb, pp) of the flush window or None, fl_pf its fine rows (W, M); s0 its
        first frame (-1: none). Divides pred / pred_b / part / pf in place; returns (rate32, rate64 or None when
        Pn == 1, p_frame, lam16, prob16 or None, None). cmax: the stream's chunk maxima (chunks of 128 bins) for
        the frames divided here -- the head's stream mode writes the rest."""
        k = self._tails.get(int(Pn))
        if k is None:
            k = _cuda_rt.CudaKernel(FUSE_SRC, "setsuna_fine_tail", device=self.device, options=(f"-DPN={int(Pn)}",))
            self._tails[int(Pn)] = k
        dev = pred.device
        rate32 = torch.empty((int(n),), dtype=torch.float32, device=dev)
        rate64 = torch.empty((int(n),), dtype=torch.float64, device=dev) if int(Pn) > 1 else None
        p_frame = torch.empty((int(n),), dtype=torch.float64, device=dev)
        lam16 = torch.empty((int(n) * int(Pn),), dtype=torch.float16, device=dev) if want16 else None
        prob16 = torch.empty_like(lam16) if want16 else None
        q = (0, "q")
        fpr, fprb, fpp = fl_rows if fl_rows is not None else (None, None, None)
        k.launch(((int(n) + 127) // 128,), (128,), 0,
                 (pred, pred_b, (part if part is not None else q), (fpr if fpr is not None else q),
                  (fprb if fprb is not None else q), (fpp if fpp is not None else q), rate32,
                  (rate64 if rate64 is not None else q), p_frame, (lam16 if lam16 is not None else q),
                  (prob16 if prob16 is not None else q), (int(n), "i"), (int(S), "i"), (int(n_regions), "i"),
                  (int(s0), "i"), (float(n_models), "d"), (1.0 / float(M), "d")))
        if pf is not None:
            # the edge regions below the flush window's first region (all of them without one), then the tail
            edges = [i * int(bw) for i in range(int(n_regular) // int(bw) + 1)]
            if int(n_regular) % int(bw):
                edges.append(int(n_regular))
            r_first = (int(s0) // int(S)) if int(s0) >= 0 else int(n_regions)
            E = sum(1 for e in edges if e < r_first)
            blocks = E * int(S) + ((int(n) - r_first * int(S)) if int(s0) >= 0 else 0)
            if blocks > 0:
                self.tail_pf.launch((blocks,), (128,), 0,
                                    (pf, (fl_pf if fl_pf is not None else q), (int(S), "i"), (int(M), "i"),
                                     (int(n_regions), "i"), (int(n_regular), "i"), (int(bw), "i"), (r_first, "i"),
                                     (E, "i"), (int(s0), "i"), (float(n_models), "f"),
                                     (cmax if cmax is not None else q)))
        return rate32, rate64, p_frame, lam16, prob16


# ----------------------------------------------------------------------------------------------------------------
ODD_SRC = r"""
extern "C" __global__ void setsuna_odd_fine(const float* __restrict__ rows, long long rstride, float* __restrict__ pf,
                                            long long c0, long long nr, long long S, int M, long long n,
                                            long long n_regular, long long s_tail, float n_models)
{
    // a frame per block row (its windows and divisor once), the frame's M fine bins across the block
    const long long f0 = c0 * S;
    const long long f1 = min(n, (c0 + nr + 1) * S + 1);       // frames the batch's windows reach
    for (long long f = f0 + blockIdx.y; f < f1; f += gridDim.y) {
        long long ilo = f - 2 * S; ilo = ilo <= 0 ? 0 : (ilo + S - 1) / S;   // first regular window covering f
        long long ihi = f / S; if (ihi > n_regular - 1) ihi = n_regular - 1;  // last
        const long long blo = ilo > c0 ? ilo : c0, bhi = ihi < c0 + nr - 1 ? ihi : c0 + nr - 1;
        const bool prior = ilo < c0;                          // earlier batches' windows, else the zero start
        const bool fin = ihi <= c0 + nr - 1 && !(s_tail >= 0 && f >= s_tail);
        const float d = (float)(ihi - ilo + 1) * n_models;
        float* out = pf + f * (long long)M;
        const float* r0 = rows + (blo - c0) * rstride + (f - blo * S) * (long long)M;
        const int nwin = (int)(bhi - blo + 1);
        const long long step = rstride - S * (long long)M;     // the next window's row at the same frame
        for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < M; m += gridDim.x * blockDim.x) {
            float v = prior ? out[m] : 0.0f;
            const float* r = r0 + m;
            for (int k = 0; k < nwin; ++k) { v = __fadd_rn(v, *r); r += step; }
            if (fin) v = __fdiv_rn(v, d);
            out[m] = v;
        }
    }
}
"""


class CudaOddFine:
    """setsuna_odd_fine: one batch of odd windows into the fine binary stream (see ODD_SRC)."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.k = _cuda_rt.CudaKernel(ODD_SRC, "setsuna_odd_fine", device=self.device)

    def run(self, rows, pf, c0: int, nr: int, S: int, M: int, n: int, n_regular: int, s_tail: int,
            n_models: int) -> None:
        if nr <= 0:
            return
        f1 = min(int(n), (int(c0) + int(nr) + 1) * int(S) + 1)
        frames = f1 - int(c0) * int(S)
        bx = min(-(-int(M) // 256), 8)
        self.k.launch((bx, min(frames, 65535)), (256,), 0,
                      (rows, (int(rows.stride(0)), "q"), pf, (int(c0), "q"), (int(nr), "q"), (int(S), "q"),
                       (int(M), "i"), (int(n), "q"), (int(n_regular), "q"), (int(s_tail), "q"), (float(n_models), "f")))
