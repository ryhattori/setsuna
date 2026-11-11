"""The timing decoder's greedy matching pursuit as one CUDA kernel (NVRTC, see _cuda_rt).

``decode_readout.struct_decode`` picks, per segment, the first occurrence of the maximum of the matched-filter
response, stops when that maximum is <= 0 or the count is used up, and subtracts ``v * template`` over the clipped
window. ``setsuna._greedy_pursuit_batched_cuda`` did that for all segments of a ROI in lockstep with torch ops:
every iteration an argmax over every bin of every segment (87 x 936k float64 at 60 Hz, 0.75 ms), for as many
iterations as the busiest segment has spikes.

Here one thread block runs one segment's whole loop on the device. The block keeps the maximum of each run of
``B`` bins (with its first index) in shared memory: the segment's maximum is the best of those, ties going to the
smaller index -- the same first occurrence as a scan of every bin -- and after a subtraction only the runs the
window touched are scanned again. The subtraction is a float64 multiply then a subtract (``__dmul_rn`` and
``__dsub_rn``: two roundings, never a fused multiply-add), the arithmetic of the numpy loop and of the torch
lockstep version, so the picks are the same bins in the same order.

``pursuit_verified`` runs the same loop from a float32 matched filter (``matched_filter32``: a direct correlation,
which this GPU runs several times faster than float64 FFTs). Float32 alone moved a quarter of the picks (it cannot
resolve the top of a broad peak at the 0.03 ms grid) and sent 0.06 % of spikes to another peak. So the float32
response only bounds the float64 one: every bin starts as an UPPER BOUND, the response times (1 + rho) plus a small
absolute slack -- rho bounds the relative error of K fused multiply-adds of non-negative terms (bound_rho) -- and when
the segment's maximum is such a bound, every bin whose bound reaches a lower bound of that bin's value gets its exact
value -- the float64 correlation of the probabilities with the template, then the subtractions of the picks made so
far, in their order -- and the maximum is taken again, until it is an exact value. That value is the maximum of the
float64 response (no bound below it can hide a larger one), with the first occurrence among equals, so each pick is
the one the float64 loop makes, up to the ~1e-17 by which a float64 FFT and a direct float64 sum differ. On the
1450-ROI bench session a pick evaluates a median of ~40 bins exactly. A segment that would need more than CAND_CAP
evaluations for one pick is reported, and the caller decodes it with the float64 FFT; so is every segment when a
probability is negative or NaN (the relative bound needs non-negative terms).
"""
from __future__ import annotations

import threading

import numpy as np
import torch

from . import _cuda_rt

NT = 1024           # threads per block (one segment each)
# the verified pursuit's block size (one segment each): its picks are a chain of block-wide reductions, whose cost
# grows with the block; the values do not depend on it
VERIFIED_NT = int(__import__("os").environ.get("JSCOPE_SETSUNA_VERIFIED_NT", "256"))
RUN = 1024          # bins per run whose maximum the block keeps (a multiple of NT)
MAX_RUNS = 3072     # runs held in shared memory (12 bytes each, under the 48 KB default); longer segments use longer runs

PURSUIT_SRC = r"""
#ifndef NT
#define NT 1024
#endif
#define NW (NT / 32)
#define NEG_INF __longlong_as_double(0xfff0000000000000LL)

// [2026-11-07] An order-preserving integer key of a double: -0.0 folded onto +0.0, then the sign bit flipped for
// values >= 0 and every bit for values < 0, so that unsigned key order is the double order and equal keys are equal
// doubles, for every non-NaN value (NaN never reaches the pursuits: a negative or NaN probability sets *bad). Compared
// on the integer pipe: double compares run at 1/64 rate on GeForce parts and the reductions are chains of them.
__device__ __forceinline__ unsigned long long dkey(double v) {
    long long b = __double_as_longlong(v);
    if (b == (long long)0x8000000000000000LL) b = 0;
    return (unsigned long long)(b ^ ((b >> 63) | (long long)0x8000000000000000LL));
}

__device__ __forceinline__ bool better(double v, int i, double bv, int bi) {
    // (v, i) beats (bv, bi): a larger value, or the same value at a smaller index (the first occurrence)
    return v > bv || (v == bv && i < bi);
}

// better() on the keys, for the verified pursuit (whose values are never NaN); setsuna_pursuit, the float64 path
// that also takes the rows with NaN, keeps the double compares
__device__ __forceinline__ bool betterk(double v, int i, double bv, int bi) {
    const unsigned long long a = dkey(v), b = dkey(bv);
    return a > b || (a == b && i < bi);
}

__device__ __forceinline__ void warp_best(double& v, int& i) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        if (better(ov, oi, v, i)) { v = ov; i = oi; }
    }
}

__device__ __forceinline__ void warp_bestk(double& v, int& i) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        if (betterk(ov, oi, v, i)) { v = ov; i = oi; }
    }
}

// The best (value, index) over the block; every thread returns it. Uses red_v / red_i (NW entries).
__device__ __forceinline__ void block_best(double& v, int& i, double* red_v, int* red_i) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    warp_best(v, i);
    if (lane == 0) { red_v[warp] = v; red_i[warp] = i; }
    __syncthreads();
    if (warp == 0) {
        v = (lane < NW) ? red_v[lane] : NEG_INF;
        i = (lane < NW) ? red_i[lane] : 0x7fffffff;
        warp_best(v, i);
        if (lane == 0) { red_v[0] = v; red_i[0] = i; }
    }
    __syncthreads();
    v = red_v[0];
    i = red_i[0];
    __syncthreads();                      // red_v / red_i are reused by the next call
}

extern "C" __global__ void __launch_bounds__(NT)
setsuna_pursuit(double* __restrict__ work, long long row_stride, const long long* __restrict__ lens,
                const long long* __restrict__ counts, const double* __restrict__ tmpl, int hw, int run,
                long long* __restrict__ picks, int max_picks, int* __restrict__ n_picks)
{
    extern __shared__ double runs_v[];                 // n_runs maxima, then n_runs int indices
    __shared__ double red_v[NW];
    __shared__ int red_i[NW];
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    double* w = work + (size_t)row * (size_t)row_stride;
    const int L = (int)lens[row];
    const long long want = counts[row];
    const int n_runs = (L + run - 1) / run;
    int* runs_i = (int*)(runs_v + n_runs);

    // the maximum of every run: a warp per run, each lane scanning its bins in ascending order
    for (int r = warp; r < n_runs; r += NW) {
        double v = NEG_INF; int i = 0x7fffffff;
        const int lo = r * run, hi = min(L, lo + run);
        for (int j = lo + lane; j < hi; j += 32) { const double x = w[j]; if (better(x, j, v, i)) { v = x; i = j; } }
        warp_best(v, i);
        if (lane == 0) { runs_v[r] = v; runs_i[r] = i; }
    }
    __syncthreads();
    int n = 0;
    for (long long t = 0; t < want && n < max_picks; ++t) {
        double v = NEG_INF; int i = 0x7fffffff;
        for (int r = tid; r < n_runs; r += NT) {
            if (better(runs_v[r], runs_i[r], v, i)) { v = runs_v[r]; i = runs_i[r]; }
        }
        block_best(v, i, red_v, red_i);
        if (!(v > 0.0)) break;                         // the numpy loop stops when the maximum is <= 0
        if (tid == 0) picks[(size_t)row * (size_t)max_picks + n] = i;
        ++n;
        const int lo = max(0, i - hw), hi = min(L, i + hw + 1);
        for (int j = lo + tid; j < hi; j += NT) w[j] = __dsub_rn(w[j], __dmul_rn(tmpl[hw - (i - j)], v));
        __syncthreads();
        for (int r = lo / run; r <= (hi - 1) / run; ++r) {
            double rv = NEG_INF; int ri = 0x7fffffff;
            const int a = r * run, b = min(L, a + run);
            for (int j = a + tid; j < b; j += NT) { const double x = w[j]; if (better(x, j, rv, ri)) { rv = x; ri = j; } }
            block_best(rv, ri, red_v, red_i);
            if (tid == 0) { runs_v[r] = rv; runs_i[r] = ri; }
        }
        __syncthreads();
    }
    if (tid == 0) n_picks[row] = n;
}

#ifndef CAND_CAP
#define CAND_CAP 8192
#endif

// The matched filter in float32, directly: resp[row][j] = sum_d x[j - hw + d] * g[d], d = 0 .. 2hw, the products
// accumulated by one fused multiply-add each in increasing d (so |error| <= gamma_K * the exact value: every term is
// >= 0), x the row's probabilities (zero outside [0, L)). A thread computes CR consecutive outputs from a register
// window over a shared tile (CR odd: the tile reads of a warp hit 32 different banks). *bad is set when a probability
// is negative or NaN: the relative bound needs non-negative terms, and the verified pursuit then declines the rows.
// tile_v / tile_i (row stride tile_stride): each tile's largest response and its first bin -- the pursuit's run
// maxima, so it need not scan the response to start.
#ifndef CT
#define CT 128
#endif
#ifndef CR
#define CR 15
#endif
#define CTILE (CT * CR)
// CR consecutive outputs (the tile's base, base + 1, ...) from the staged tile xs (inputs from base - hw on) and the
// template gs: one fused multiply-add per tap in increasing tap order -- setsuna_mf32's arithmetic, shared with the
// verified pursuit's lazily computed runs so both give the same bits
__device__ __forceinline__ void mf_outputs(const float* xs, const float* gs, int K, int base, float* acc) {
    float win[CR];
    #pragma unroll
    for (int r = 0; r < CR; ++r) { acc[r] = 0.0f; win[r] = xs[base + r]; }
    int d = 0;
    for (; d + CR <= K; d += CR) {
        #pragma unroll
        for (int s = 0; s < CR; ++s) {
            const float gd = gs[d + s];
            #pragma unroll
            for (int r = 0; r < CR; ++r) acc[r] = fmaf(win[(s + r) % CR], gd, acc[r]);
            win[s] = xs[base + d + s + CR];            // the element entering the window replaces the one leaving
        }
    }
    for (; d < K; ++d) {                               // the last K % CR taps
        const float gd = gs[d];
        #pragma unroll
        for (int r = 0; r < CR; ++r) acc[r] = fmaf(xs[base + d + r], gd, acc[r]);
    }
}

extern "C" __global__ void __launch_bounds__(CT)
setsuna_mf32(const float* __restrict__ pf, const long long* __restrict__ seg0, const long long* __restrict__ lens,
             const float* __restrict__ g, int hw, float* __restrict__ resp, long long resp_stride,
             int* __restrict__ bad, float* __restrict__ tile_v, int* __restrict__ tile_i, long long tile_stride)
{
    __shared__ float s_tv[CT / 32];
    __shared__ int s_ti[CT / 32];
    extern __shared__ float sm[];                      // g[K], then the input tile [CTILE + K - 1 + CR]
    const int K = 2 * hw + 1;
    float* gs = sm;
    float* xs = sm + K;
    const int row = blockIdx.y;
    const int t0 = blockIdx.x * CTILE;
    const int L = (int)lens[row];
    if (t0 >= L) return;
    const float* x = pf + seg0[row];
    for (int q = threadIdx.x; q < K; q += CT) gs[q] = g[q];
    int neg = 0;
    for (int q = threadIdx.x; q < CTILE + K - 1 + CR; q += CT) {      // (+ CR: the window's last refill reads past)
        const int idx = t0 - hw + q;
        const float v = (idx >= 0 && idx < L) ? x[idx] : 0.0f;
        xs[q] = v;
        neg |= !(v >= 0.0f);
    }
    if (__syncthreads_or(neg) && threadIdx.x == 0) *bad = 1;
    const int base = threadIdx.x * CR;                 // this thread's outputs: t0 + base + r
    float acc[CR];
    mf_outputs(xs, gs, K, base, acc);
    float* out = resp + (size_t)row * (size_t)resp_stride;
    float bv = __int_as_float(0xff800000); int bi = 0x7fffffff;      // this thread's largest output, its first bin
    #pragma unroll
    for (int r = 0; r < CR; ++r) {
        const int j = t0 + base + r;
        if (j < L) {
            out[j] = acc[r];
            if (acc[r] > bv) { bv = acc[r]; bi = j; }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { s_tv[threadIdx.x >> 5] = bv; s_ti[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int q = 1; q < CT / 32; ++q)
            if (s_tv[q] > bv || (s_tv[q] == bv && s_ti[q] < bi)) { bv = s_tv[q]; bi = s_ti[q]; }
        tile_v[(size_t)row * (size_t)tile_stride + blockIdx.x] = bv;
        tile_i[(size_t)row * (size_t)tile_stride + blockIdx.x] = bi;
    }
}

// Per tile of CTILE outputs (the verified pursuit's runs), an upper bound of every float64 response in it: the
// largest probability under any of its windows times gsum (the float64 template's sum, taken a little larger) --
// the terms are non-negative, so no response, and no residual after subtractions, exceeds it. *bad as setsuna_mf32.
extern "C" __global__ void __launch_bounds__(256)
setsuna_tile_bounds(const float* __restrict__ pf, const long long* __restrict__ seg0, const long long* __restrict__ lens,
                    int hw, double gsum, double* __restrict__ tile_b, long long tile_stride, int* __restrict__ bad)
{
    __shared__ float red[8];
    const int row = blockIdx.y;
    const int t0 = blockIdx.x * CTILE;
    const int L = (int)lens[row];
    if (t0 >= L) return;
    const float* x = pf + seg0[row];
    const int lo = max(0, t0 - hw), hi = min(L, t0 + CTILE + hw);
    float m = 0.0f;
    int neg = 0;
    for (int q = lo + (int)threadIdx.x; q < hi; q += 256) {
        const float v = x[q];
        neg |= !(v >= 0.0f);
        m = fmaxf(m, v);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    if (__syncthreads_or(neg) && threadIdx.x == 0) *bad = 1;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int q = 1; q < 8; ++q) m = fmaxf(m, red[q]);
        tile_b[(size_t)row * (size_t)tile_stride + blockIdx.x] = (double)m * gsum;
    }
}

// setsuna_tile_bounds from the stream's per-chunk maxima (cmax[c]: the largest probability in bins [c chunk,
// (c + 1) chunk), NaN where one is negative or NaN -- the forward's stream writers keep them): a thread per tile, the
// maximum over the chunks covering its windows. They cover a superset of its bins: an upper bound again.
extern "C" __global__ void __launch_bounds__(256)
setsuna_tile_bounds_cmax(const float* __restrict__ cmax, long long chunk, const long long* __restrict__ seg0,
                         const long long* __restrict__ lens, int hw, double gsum, double* __restrict__ tile_b,
                         long long tile_stride, int n_tiles, int* __restrict__ bad)
{
    const int row = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int L = (int)lens[row];
    const int t0 = t * CTILE;
    if (t >= n_tiles || t0 >= L) return;
    const long long base = seg0[row];
    const long long lo = base + max(0, t0 - hw), hi = base + min(L, t0 + CTILE + hw);
    float m = 0.0f;
    int neg = 0;
    for (long long c = lo / chunk; c <= (hi - 1) / chunk; ++c) {
        const float v = cmax[c];
        neg |= !(v >= 0.0f);
        m = fmaxf(m, v);
    }
    if (neg) *bad = 1;
    tile_b[(size_t)row * (size_t)tile_stride + t] = (double)m * gsum;
}

// Per decoder segment s (a block each), the count mass over its native bins [s * sb, min((s + 1) * sb, n_fine)) from
// per-frame masses (frames of M bins): lf[g] the frame's mass, part[g] its mass before the segment boundary inside
// it (0 where none). The whole frames in a fixed order, then the frame split at the segment's start (its mass after
// the boundary) and the one split at its end (the mass before). n_fine = n_frames * M.
extern "C" __global__ void __launch_bounds__(256)
setsuna_segment_counts(const double* __restrict__ lf, const double* __restrict__ part, long long M, long long sb,
                       long long n_fine, double* __restrict__ out)
{
    __shared__ double red[256];
    const long long a = (long long)blockIdx.x * sb, b = min(a + sb, n_fine);
    const long long ga = (a + M - 1) / M, gb = b / M;             // the whole frames: [ga, gb)
    double acc = 0.0;
    for (long long g = ga + threadIdx.x; g < gb; g += 256) acc += lf[g];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double v = red[0];
        if (a % M) { const long long g = a / M; v += lf[g] - part[g]; }
        if (b % M) v += part[b / M];
        out[blockIdx.x] = v;
    }
}

// The bounds of run r's bins (resp (1 + rho) + errb), none of them exact yet: a lazily started run.
__device__ __forceinline__ void start_run(int r, int run, int L, const float* rs, double onep, double eb,
                                          double* w, unsigned char* ex) {
    const int a = r * run, b = min(L, a + run);
    for (int j = a + (int)threadIdx.x; j < b; j += NT) { w[j] = (double)rs[j] * onep + eb; ex[j] = 0; }
}

// ... when the responses are not computed ahead: run r's (CTILE bins) float32 responses first, by setsuna_mf32's
// arithmetic (mf_outputs, in the block's first CT threads), written to rs_w, then its bounds. Every thread calls it.
__device__ __forceinline__ void start_run_mf(int r, int L, const float* x, float* rs_w, const float* gs, float* xs,
                                             int K, int hw, double onep, double eb, double* w, unsigned char* ex) {
    const int a = r * CTILE;
    for (int q = (int)threadIdx.x; q < CTILE + K - 1 + CR; q += NT) {
        const int idx = a - hw + q;
        xs[q] = (idx >= 0 && idx < L) ? x[idx] : 0.0f;
    }
    __syncthreads();
    if (threadIdx.x < CT) {
        float acc[CR];
        const int base = (int)threadIdx.x * CR;
        mf_outputs(xs, gs, K, base, acc);
        #pragma unroll
        for (int q = 0; q < CR; ++q) {
            const int j = a + base + q;
            if (j < L) { rs_w[j] = acc[q]; w[j] = (double)acc[q] * onep + eb; ex[j] = 0; }
        }
    }
    __syncthreads();
}

// [2026-11-07] The verified pursuit also keeps the maximum (value, first index) of every chunk of CH bins of a started
// run (cm_v / cm_i, global, chunk c = bins [c CH, (c + 1) CH)): the candidate listing reads only the chunks whose
// maximum reaches the threshold, and a run's maximum is the best of its chunks' (betterk() is a total order: the same
// value and index as a scan of its bins). CH divides every run length (CTILE = 1920, multiples of 1024).
#define CH 64
// chunk c's maximum from w (a warp; every lane returns it)
__device__ __forceinline__ void chunk_best(const double* w, int c, int L, double& v, int& i) {
    const int lane = threadIdx.x & 31;
    const int j0 = c * CH + lane, j1 = j0 + 32;
    v = NEG_INF; i = 0x7fffffff;
    if (j0 < L) { v = w[j0]; i = j0; }
    if (j1 < L) { const double y = w[j1]; if (betterk(y, j1, v, i)) { v = y; i = j1; } }
    warp_bestk(v, i);
    v = __shfl_sync(0xffffffffu, v, 0);
    i = __shfl_sync(0xffffffffu, i, 0);
}

// run r's maximum from its chunks' (a warp; every lane returns it)
__device__ __forceinline__ void run_best_chunks(const double* cmv, const int* cmi, int r, int nch, int nchL,
                                                double& v, int& i) {
    const int lane = threadIdx.x & 31;
    const int c0 = r * nch, c1 = min(nchL, c0 + nch);
    v = NEG_INF; i = 0x7fffffff;
    for (int c = c0 + lane; c < c1; c += 32) { const double y = cmv[c]; const int k = cmi[c]; if (betterk(y, k, v, i)) { v = y; i = k; } }
    warp_bestk(v, i);
    v = __shfl_sync(0xffffffffu, v, 0);
    i = __shfl_sync(0xffffffffu, i, 0);
}

// The verified pursuit (see the module docstring). resp: the float32 response (row stride resp_stride); work: its
// float64 upper bounds (resp (1 + rho) + errb), a bin's exact float64 value once exact[] marks it; pf + seg0[row]:
// the segment's probabilities (float32, valid over lens[row]). *bad set: the bounds do not hold, every row is reported
// (n_picks -1) for the float64 path. tile_v / tile_i (setsuna_mf32's tile maxima, runs of its tile length): the run
// maxima to start from, and a run's bounds are written only once the pursuit touches it (started[]); null: every
// bound is written first, with the run maxima. tile_b (setsuna_tile_bounds; runs of CTILE bins, g32 the float32
// template): the responses are not computed ahead either -- a run starts with only its tile's bound, and the
// response of its bins (setsuna_mf32's bits, into resp) when the pursuit first needs them.
extern "C" __global__ void __launch_bounds__(NT)
setsuna_pursuit_verified(float* __restrict__ resp, long long resp_stride,
                         double* __restrict__ work, long long row_stride, const float* __restrict__ pf,
                         const long long* __restrict__ seg0, const long long* __restrict__ lens,
                         const long long* __restrict__ counts, const double* __restrict__ errb, double rho,
                         const double* __restrict__ tmpl, int hw, int run,
                         unsigned char* __restrict__ exact, long long exact_stride, int* __restrict__ cand,
                         long long* __restrict__ picks, double* __restrict__ pick_v, int max_picks,
                         int* __restrict__ n_picks, int* __restrict__ n_evals, const int* __restrict__ bad,
                         const float* __restrict__ tile_v, const int* __restrict__ tile_i, long long tile_stride,
                         const double* __restrict__ tile_b, const float* __restrict__ g32,
                         int* __restrict__ run_head, int* __restrict__ run_tail, long long rl_stride,
                         int* __restrict__ pick_next, double* __restrict__ chunk_v, int* __restrict__ chunk_i,
                         long long cm_stride, long long* __restrict__ prof)
{
#ifdef PROF
    long long _pt = 0, _ps[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#define PMARK(k) if (threadIdx.x == 0) { const long long _c = clock64(); _ps[k] += _c - _pt; _pt = _c; }
#define PSTART() if (threadIdx.x == 0) { _pt = clock64(); }
#else
#define PMARK(k)
#define PSTART()
#endif
    extern __shared__ double runs_v[];                 // n_runs maxima, n_runs indices, n_runs listed runs, started
                                                       // (tile_b: then the template and a staged input tile)
    __shared__ double red_v[NW];
    __shared__ int red_i[NW];
    __shared__ int s_nc, s_nr;
    // [2026-11-07] the runs' maxima per group of 32 runs (at most MAX_RUNS_V / 32 groups): the search for the largest
    // is one warp over the group maxima instead of the block over every run, and the candidate listing skips groups
    // below the threshold. betterk() is a total order (index breaks ties), so the same maximum and the same runs.
    __shared__ double s_gv[64];
    __shared__ int s_gi[64];
    __shared__ double s_bv;
    __shared__ int s_bi;
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (*bad) {
        if (tid == 0) { n_picks[row] = -1; n_evals[row] = 0; }
        return;
    }
    double* w = work + (size_t)row * (size_t)row_stride;
    const float* x = pf + seg0[row];
    unsigned char* ex = exact + (size_t)row * (size_t)exact_stride;
    int* cnd = cand + (size_t)row * CAND_CAP;
    long long* pk = picks + (size_t)row * (size_t)max_picks;
    double* pv = pick_v + (size_t)row * (size_t)max_picks;
    const int L = (int)lens[row];
    const long long want = counts[row];
    const double e2 = 2.0 * errb[row];
    const double onep = 1.0 + rho;
    const int n_runs = (L + run - 1) / run;
    int* runs_i = (int*)(runs_v + n_runs);
    int* rl = runs_i + n_runs;
    int* started = rl + n_runs;
    float* rs = resp + (size_t)row * (size_t)resp_stride;
    const double eb = errb[row];
    const bool mf = tile_b != nullptr;                 // responses computed as runs start (run == CTILE)
    const bool lazy = tile_v != nullptr || mf;
    const int K = 2 * hw + 1;
    float* gs = reinterpret_cast<float*>(started + n_runs);
    float* xs = gs + K;
    // the picks of each run as a list in pick order (head, tail, next), so an exact evaluation subtracts the earlier
    // picks within reach by walking the few runs around the bin instead of every pick
    int* rh = run_head + (size_t)row * (size_t)rl_stride;
    int* rt = run_tail + (size_t)row * (size_t)rl_stride;
    int* pnx = pick_next + (size_t)row * (size_t)max_picks;
    for (int r = tid; r < n_runs; r += NT) { rh[r] = -1; rt[r] = -1; }
    double* cmv = chunk_v + (size_t)row * (size_t)cm_stride;
    int* cmi = chunk_i + (size_t)row * (size_t)cm_stride;
    const int nch = run / CH, nchL = (L + CH - 1) / CH;   // chunks per run, in the segment

    if (mf) {
        // every run starts with its tile's bound (its bins' proxy index: the first); the template to shared memory
        const double* tb = tile_b + (size_t)row * (size_t)tile_stride;
        for (int r = tid; r < n_runs; r += NT) { runs_v[r] = tb[r]; runs_i[r] = r * run; started[r] = 0; }
        for (int q = tid; q < K; q += NT) gs[q] = g32[q];
    } else if (lazy) {
        // the run maxima from the matched filter's tiles: the bound is monotone in the response, so the largest bound
        // of a run is at its largest response, the first of equals
        const float* tv = tile_v + (size_t)row * (size_t)tile_stride;
        const int* ti = tile_i + (size_t)row * (size_t)tile_stride;
        for (int r = tid; r < n_runs; r += NT) { runs_v[r] = (double)tv[r] * onep + eb; runs_i[r] = ti[r]; started[r] = 0; }
    } else {
        // the bounds, and the maximum of every run: a warp per run, each lane scanning its bins in ascending order
        for (int r = warp; r < n_runs; r += NW) {
            double v = NEG_INF; int i = 0x7fffffff;
            const int lo = r * run, hi = min(L, lo + run);
            for (int j = lo + lane; j < hi; j += 32) {
                const double y = (double)rs[j] * onep + eb;
                w[j] = y;
                ex[j] = 0;
                if (betterk(y, j, v, i)) { v = y; i = j; }
            }
            warp_bestk(v, i);
            if (lane == 0) { runs_v[r] = v; runs_i[r] = i; started[r] = 1; }
        }
        __syncthreads();
        for (int c = warp; c < nchL; c += NW) {        // and every chunk's
            double v; int i;
            chunk_best(w, c, L, v, i);
            if (lane == 0) { cmv[c] = v; cmi[c] = i; }
        }
    }
    __syncthreads();
#define GROUP_REFRESH(g_) do { const int _g = (g_); const int _r = _g * 32 + lane;         double _v = (_r < n_runs) ? runs_v[_r] : NEG_INF; int _i = (_r < n_runs) ? runs_i[_r] : 0x7fffffff;         warp_bestk(_v, _i); if (lane == 0) { s_gv[_g] = _v; s_gi[_g] = _i; } } while (0)
    const int n_groups = (n_runs + 31) / 32;
    for (int g = warp; g < n_groups; g += NW) GROUP_REFRESH(g);
    __syncthreads();
    int n = 0, evals = 0, overflow = 0;
    for (long long t = 0; t < want && n < max_picks; ++t) {
        double v; int i;
        for (;;) {
            PSTART();
            if (warp == 0) {                           // the largest run maximum: one warp over the group maxima
                double gv = NEG_INF; int gi = 0x7fffffff;
                for (int g = lane; g < n_groups; g += 32) if (betterk(s_gv[g], s_gi[g], gv, gi)) { gv = s_gv[g]; gi = s_gi[g]; }
                warp_bestk(gv, gi);
                if (lane == 0) { s_bv = gv; s_bi = gi; }
            }
            __syncthreads();
            v = s_bv; i = s_bi;
            __syncthreads();                           // s_bv / s_bi are written again by the next search
            PMARK(0);
            if (mf && !started[i / run]) {
                // the largest bound is a tile's: its run's responses and bounds, its maximum, and the search again
                const int r = i / run;
                start_run_mf(r, L, x, rs, gs, xs, K, hw, onep, eb, w, ex);
                for (int c = r * nch + warp; c < min(nchL, (r + 1) * nch); c += NW) {
                    double cv; int ci;
                    chunk_best(w, c, L, cv, ci);
                    if (lane == 0) { cmv[c] = cv; cmi[c] = ci; }
                }
                __syncthreads();
                if (warp == 0) {
                    double rv; int ri;
                    run_best_chunks(cmv, cmi, r, nch, nchL, rv, ri);
                    if (lane == 0) { runs_v[r] = rv; runs_i[r] = ri; started[r] = 1; }
                    __syncwarp();
                    GROUP_REFRESH(r / 32);
                }
                __syncthreads();
                PMARK(8);                              // (slot 8: a top run started; not one of the five phases)
                continue;
            }
            if (started[i / run] && ex[i]) break;      // the largest bound is an exact value: the true maximum
            // a lower bound of the top bin's own value, so of the maximum: bins whose bound reaches it may hold it
            const double thr = v - (2.0 * rho * (double)rs[i] + e2);
            if (tid == 0) { s_nc = 0; s_nr = 0; }
            __syncthreads();
            for (int g = warp; g < n_groups; g += NW) {     // only the groups whose maximum reaches thr
                if (!(s_gv[g] >= thr)) continue;
                const int r = g * 32 + lane;
                if (r < n_runs && runs_v[r] >= thr) { const int k = atomicAdd(&s_nr, 1); rl[k] = r; }
            }
            __syncthreads();
            const int nr = s_nr;
            if (lazy) {                                // the listed runs' bounds, where not yet written
                for (int q = 0; q < nr; ++q) {
                    if (started[rl[q]]) continue;
                    if (mf) start_run_mf(rl[q], L, x, rs, gs, xs, K, hw, onep, eb, w, ex);
                    else start_run(rl[q], run, L, rs, onep, eb, w, ex);
                }
                __syncthreads();
                for (int q = 0; q < nr; ++q) {         // their chunks' maxima
                    const int r = rl[q];
                    if (started[r]) continue;
                    for (int c = r * nch + warp; c < min(nchL, (r + 1) * nch); c += NW) {
                        double cv; int ci;
                        chunk_best(w, c, L, cv, ci);
                        if (lane == 0) { cmv[c] = cv; cmi[c] = ci; }
                    }
                }
                __syncthreads();
                for (int q = tid; q < nr; q += NT) started[rl[q]] = 1;
                __syncthreads();
            }
            // the candidates (!ex[j] && w[j] >= thr): a warp per listed run, only in its chunks whose maximum
            // reaches thr
            for (int q = warp; q < nr; q += NW) {
                const int r = rl[q];
                const int c1 = min(nchL, (r + 1) * nch);
                for (int cb = r * nch; cb < c1; cb += 32) {
                    unsigned sel = __ballot_sync(0xffffffffu, cb + lane < c1 && cmv[cb + lane] >= thr);
                    while (sel) {
                        const int c = cb + __ffs(sel) - 1;
                        sel &= sel - 1;
                        const int j0 = c * CH + lane, j1 = j0 + 32;
                        const bool p0 = j0 < L && w[j0] >= thr && !ex[j0];
                        const bool p1 = j1 < L && w[j1] >= thr && !ex[j1];
                        const unsigned m0 = __ballot_sync(0xffffffffu, p0), m1 = __ballot_sync(0xffffffffu, p1);
                        if (m0 | m1) {
                            int base = 0;
                            if (lane == 0) base = atomicAdd(&s_nc, __popc(m0) + __popc(m1));
                            base = __shfl_sync(0xffffffffu, base, 0);
                            const unsigned lt = (1u << lane) - 1u;
                            if (p0) { const int k = base + __popc(m0 & lt); if (k < CAND_CAP) cnd[k] = j0; }
                            if (p1) { const int k = base + __popc(m0) + __popc(m1 & lt); if (k < CAND_CAP) cnd[k] = j1; }
                        }
                    }
                }
            }
            __syncthreads();
            PMARK(1);
            const int nc = s_nc;
            if (nc > CAND_CAP) { overflow = 1; break; }
            // exact values, a warp per bin: the float64 correlation, then the earlier picks' subtractions in order
            for (int c = warp; c < nc; c += NW) {
                const int j = cnd[c];
                double s = 0.0;
                for (int d = lane; d <= 2 * hw; d += 32) {
                    const int q = j - hw + d;
                    if (q >= 0 && q < L) s = fma((double)x[q], tmpl[d], s);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                if (lane == 0) {
                    // the earlier picks within the template's reach, subtracted in pick order: the runs around j hold
                    // every pick that can reach it; their lists (each in pick order) are merged by pick index
                    const int ra = max(0, j - hw) / run, rb = min(L - 1, j + hw) / run;
                    const int nrr = rb - ra + 1;
                    if (nrr <= 16) {
                        int cur[16];
                        for (int q = 0; q < nrr; ++q) cur[q] = rh[ra + q];
                        for (;;) {
                            int best = -1, bq = -1;
                            for (int q = 0; q < nrr; ++q)
                                if (cur[q] >= 0 && (best < 0 || cur[q] < best)) { best = cur[q]; bq = q; }
                            if (best < 0) break;
                            cur[bq] = pnx[best];
                            const long long dd = pk[best] - j;
                            if (dd >= -hw && dd <= hw) s = __dsub_rn(s, __dmul_rn(tmpl[hw - (int)dd], pv[best]));
                        }
                    } else {
                        for (int p = 0; p < n; ++p) {
                            const long long dd = pk[p] - j;
                            if (dd >= -hw && dd <= hw) s = __dsub_rn(s, __dmul_rn(tmpl[hw - (int)dd], pv[p]));
                        }
                    }
                    w[j] = s;
                    ex[j] = 1;
                }
            }
            __syncthreads();
            PMARK(2);
            evals += nc;
            // the listed runs' maxima again: a warp per run; only the chunks the listing read (maximum >= thr) can
            // hold a bin evaluated just now, and their maxima are taken again
            for (int q = warp; q < nr; q += NW) {
                const int r = rl[q];
                const int c1 = min(nchL, (r + 1) * nch);
                double rv = NEG_INF; int ri = 0x7fffffff;
                for (int cb = r * nch; cb < c1; cb += 32) {
                    double cv = NEG_INF; int ci = 0x7fffffff;
                    if (cb + lane < c1) { cv = cmv[cb + lane]; ci = cmi[cb + lane]; }
                    unsigned sel = __ballot_sync(0xffffffffu, cv >= thr);
                    while (sel) {
                        const int k = __ffs(sel) - 1;
                        sel &= sel - 1;
                        double bv; int bi;
                        chunk_best(w, cb + k, L, bv, bi);
                        if (lane == k) { cv = bv; ci = bi; cmv[cb + k] = bv; cmi[cb + k] = bi; }
                    }
                    if (betterk(cv, ci, rv, ri)) { rv = cv; ri = ci; }
                }
                warp_bestk(rv, ri);
                if (lane == 0) { runs_v[r] = rv; runs_i[r] = ri; }
            }
            __syncthreads();
            for (int q = warp; q < nr; q += NW) GROUP_REFRESH(rl[q] / 32);   // (a group listed twice: the same values)
            __syncthreads();
            PMARK(3);
#ifdef PROF
            if (threadIdx.x == 0) _ps[6] += 1;
#endif
        }
        if (overflow) break;
        if (!(v > 0.0)) break;                         // the numpy loop stops when the maximum is <= 0
        PSTART();
        if (tid == 0) {
            pk[n] = i; pv[n] = v;
            const int r = i / run;                     // append to its run's list
            pnx[n] = -1;
            if (rt[r] < 0) rh[r] = n; else pnx[rt[r]] = n;
            rt[r] = n;
        }
        ++n;                                           // (the lists are read by the next evaluations, after barriers)
        const int lo = max(0, i - hw), hi = min(L, i + hw + 1);
        unsigned int fresh = 0;                        // the window's runs started just now (bit r - lo / run)
        if (lazy) {                                    // the window's runs' bounds, where not yet written
            for (int r = lo / run; r <= (hi - 1) / run; ++r) {
                if (started[r]) continue;
                fresh |= 1u << ((r - lo / run) & 31);
                if (mf) start_run_mf(r, L, x, rs, gs, xs, K, hw, onep, eb, w, ex);
                else start_run(r, run, L, rs, onep, eb, w, ex);
            }
            __syncthreads();
            if (tid == 0) for (int r = lo / run; r <= (hi - 1) / run; ++r) started[r] = 1;
        }
        // the subtraction and the touched chunks' maxima, a warp per chunk: the chunks overlapping the window, and every
        // chunk of a run started just now (the window's runs are contiguous and those between its ends lie inside it)
        {
            const int r0 = lo / run, r1 = (hi - 1) / run;
            const int cs = (fresh & 1u) ? r0 * nch : lo / CH;
            const int ce = ((fresh >> ((r1 - r0) & 31)) & 1u) ? min(nchL, (r1 + 1) * nch) - 1 : (hi - 1) / CH;
            for (int c = cs + warp; c <= ce; c += NW) {
                const int j0 = c * CH + lane, j1 = j0 + 32;
                double cv = NEG_INF; int ci = 0x7fffffff;
                if (j0 < L) {
                    double y = w[j0];
                    if (j0 >= lo && j0 < hi) { y = __dsub_rn(y, __dmul_rn(tmpl[hw - (i - j0)], v)); w[j0] = y; }
                    cv = y; ci = j0;
                }
                if (j1 < L) {
                    double y = w[j1];
                    if (j1 >= lo && j1 < hi) { y = __dsub_rn(y, __dmul_rn(tmpl[hw - (i - j1)], v)); w[j1] = y; }
                    if (betterk(y, j1, cv, ci)) { cv = y; ci = j1; }
                }
                warp_bestk(cv, ci);
                if (lane == 0) { cmv[c] = cv; cmi[c] = ci; }
            }
            __syncthreads();
            if (warp == 0) {                           // the window's runs' maxima from their chunks, their groups'
                for (int r = r0; r <= r1; ++r) {
                    double rv; int ri;
                    run_best_chunks(cmv, cmi, r, nch, nchL, rv, ri);
                    if (lane == 0) { runs_v[r] = rv; runs_i[r] = ri; }
                    __syncwarp();
                    GROUP_REFRESH(r / 32);
                    __syncwarp();
                }
            }
            __syncthreads();
        }
        PMARK(4);
    }
    if (tid == 0) { n_picks[row] = overflow ? -1 : n; n_evals[row] = evals; }
#ifdef PROF
    if (tid == 0) for (int k = 0; k < 12; ++k) prof[(size_t)row * 12 + k] = _ps[k];
#endif
}
"""

_LOCK = threading.Lock()
_KERNELS: dict = {}
CAND_CAP = 8192     # exact evaluations one pick may need before the segment falls back to the float64 FFT
MAX_RUNS_V = 2048   # the verified kernel keeps a double and three ints per run in shared memory (40 KB)
SMEM_MAX = 48 * 1024  # shared memory a launch may use without the opt-in attribute (static + dynamic)
CHUNK = 64          # the verified kernel's chunk length (CH): divides CTILE and RUN
SMEM_STATIC_V = 2048  # setsuna_pursuit_verified's static shared arrays (reductions, group maxima: < 1 KB) + margin


def mf32_smem(hw: int) -> int:
    """setsuna_mf32's shared memory for a template of half-width ``hw`` (the template, a tile and its halo)."""
    K = 2 * int(hw) + 1
    return 4 * (K + MF_CT * MF_CR + K - 1 + MF_CR)


def verified_max_bins(hw: int) -> int:
    """The longest segment (native bins) the verified pursuit takes in one block with a template of half-width ``hw``:
    pursuit_from_bounds' shared memory -- 20 bytes per run of MF_CT * MF_CR bins plus the matched filter's tile --
    within SMEM_MAX, and at most MAX_RUNS_V runs. 0 when the template alone does not fit."""
    room = SMEM_MAX - SMEM_STATIC_V - mf32_smem(hw)
    if room <= 0:
        return 0
    return int(min(MAX_RUNS_V, room // 20)) * MF_CT * MF_CR


def mf32_fits(hw: int) -> bool:
    """The float32 matched filter can run with this template (wide templates -- above ~14 ms at 120 Hz, ~28 ms at
    60 Hz -- exceed the launch's shared memory; the decoder then takes the float64 FFT path, the same picks)."""
    return mf32_smem(hw) <= SMEM_MAX
MF_CT, MF_CR = 128, 15   # setsuna_mf32: threads per block, consecutive outputs per thread (odd)


_PURSUIT_PROF = __import__("os").environ.get("JSCOPE_SETSUNA_PURSUIT_PROF", "0") == "1"


def _kernel(device, name="setsuna_pursuit"):
    nt = VERIFIED_NT if name == "setsuna_pursuit_verified" else NT
    key = (device.type, device.index, name, nt)
    k = _KERNELS.get(key)
    if k is None:
        with _LOCK:
            k = _KERNELS.get(key)
            if k is None:
                k = _cuda_rt.CudaKernel(PURSUIT_SRC, name, device=device,
                                        options=(f"-DNT={nt}", f"-DCAND_CAP={CAND_CAP}", f"-DCT={MF_CT}",
                                                 f"-DCR={MF_CR}") + (("-DPROF",) if _PURSUIT_PROF else ()))
                _KERNELS[key] = k
    return k


def available(device, verified: bool = False) -> bool:
    """The kernel (``verified``: the verified pursuit's and the float32 matched filter's) compiles and loads on
    ``device`` (a CUDA device)."""
    try:
        if verified:
            _kernel(torch.device(device), "setsuna_pursuit_verified")
            _kernel(torch.device(device), "setsuna_mf32")
            _kernel(torch.device(device), "setsuna_tile_bounds")
        else:
            _kernel(torch.device(device), "setsuna_pursuit")
        return True
    except Exception:
        return False


def pursuit(work: torch.Tensor, lens, counts, hw: int, tmpl: torch.Tensor):
    """Greedy pursuit on each row of ``work`` (n_rows, >= max(lens)) float64 CUDA, a view whose rows may be
    strided (e.g. a slice of a larger buffer), modified in place. Row j is valid over ``lens[j]`` bins and asks for
    ``counts[j]`` picks. Returns the picks per row (int64 arrays, in pick order), as
    ``setsuna._greedy_pursuit_batched_cuda`` does."""
    n_rows = int(work.shape[0])
    if n_rows == 0:
        return []
    if work.dtype != torch.float64 or work.stride(1) != 1:
        raise ValueError("pursuit: work must be float64 with contiguous rows")
    device = work.device
    max_picks = int(max(counts)) if len(counts) else 0
    if max_picks <= 0:
        return [np.zeros(0, np.int64) for _ in range(n_rows)]
    lens_t = torch.as_tensor([int(x) for x in lens], dtype=torch.int64, device=device)
    cnt_t = torch.as_tensor([int(x) for x in counts], dtype=torch.int64, device=device)
    tmpl_t = tmpl.to(device=device, dtype=torch.float64).contiguous()
    L_max = int(max(lens))
    run = RUN
    while -(-L_max // run) > MAX_RUNS:
        run += RUN
    n_runs = -(-L_max // run)
    picks = torch.empty((n_rows, max_picks), dtype=torch.int64, device=device)
    n_picks = torch.empty(n_rows, dtype=torch.int32, device=device)
    smem = n_runs * 8 + n_runs * 4
    _kernel(device).launch((n_rows,), (NT,), smem,
                           (work, (int(work.stride(0)), "q"), lens_t, cnt_t, tmpl_t, (int(hw), "i"), (int(run), "i"),
                            picks, (max_picks, "i"), n_picks))
    n_h = n_picks.cpu().numpy()
    p_h = picks.cpu().numpy()
    return [p_h[j, :int(n_h[j])].astype(np.int64) for j in range(n_rows)]


def segment_counts(lam_frame: torch.Tensor, part: torch.Tensor, M: int, seg_bins: int, n_fine: int) -> torch.Tensor:
    """Per decoder segment (``seg_bins`` native bins, the last one partial) the count mass, float64 on the device
    (setsuna_segment_counts), from the per-frame masses ``lam_frame`` and ``part`` (per frame the mass before the
    segment boundary inside it) -- both (n_fine // M,) float64 CUDA. Needs seg_bins > M and n_fine a multiple of M."""
    device = lam_frame.device
    M, seg_bins, n_fine = int(M), int(seg_bins), int(n_fine)
    lf = lam_frame.reshape(-1).to(torch.float64).contiguous()
    pt = part.reshape(-1).to(device=device, dtype=torch.float64).contiguous()
    if seg_bins <= M or n_fine % M or lf.numel() != n_fine // M or pt.numel() != n_fine // M:
        raise ValueError("segment_counts: seg_bins > M and per-frame inputs covering n_fine = n_frames * M bins")
    n_seg = -(-n_fine // seg_bins)
    out = torch.empty(n_seg, dtype=torch.float64, device=device)
    if n_seg:
        _kernel(device, "setsuna_segment_counts").launch(
            (n_seg,), (256,), 0, (lf, pt, (M, "q"), (seg_bins, "q"), (n_fine, "q"), out))
    return out


def matched_filter32(pf: torch.Tensor, seg0, lens, hw: int, tmpl: torch.Tensor):
    """The float32 matched filter of each segment (setsuna_mf32): (response, info). The response is
    (n_rows, max(lens)) float32, row j over ``lens[j]`` bins of the float32 stream ``pf`` from ``seg0[j]``; each value
    is within gamma_K (K = 2hw + 1 taps, ~3.4e-5 at 60 Hz) of the float64 correlation with the float32 template,
    relatively (see bound_rho) -- when no probability is negative. info, for pursuit_verified: "bad" (int32, one
    element, on the device) set when one is (or is NaN), and the pursuit then leaves every row to the float64 path;
    "tile_v" / "tile_i" each tile's largest response and its first bin, the run maxima the pursuit starts from, for
    runs of "run" bins."""
    device = pf.device
    n_rows = len(lens)
    L_max = int(max(lens))
    ctile = MF_CT * MF_CR
    n_tiles = -(-L_max // ctile)
    resp = torch.empty((n_rows, L_max), dtype=torch.float32, device=device)
    bad = torch.zeros(1, dtype=torch.int32, device=device)
    tile_v = torch.empty((n_rows, n_tiles), dtype=torch.float32, device=device)
    tile_i = torch.empty((n_rows, n_tiles), dtype=torch.int32, device=device)
    seg_t = torch.as_tensor([int(v) for v in seg0], dtype=torch.int64, device=device)
    lens_t = torch.as_tensor([int(v) for v in lens], dtype=torch.int64, device=device)
    g32 = tmpl.to(device=device, dtype=torch.float32).contiguous()
    K = 2 * int(hw) + 1
    smem = 4 * (K + ctile + K - 1 + MF_CR)
    _kernel(device, "setsuna_mf32").launch(
        (n_tiles, n_rows), (MF_CT,), smem,
        (pf, seg_t, lens_t, g32, (int(hw), "i"), resp, (int(resp.stride(0)), "q"), bad, tile_v, tile_i,
         (n_tiles, "q")))
    return resp, {"bad": bad, "tile_v": tile_v, "tile_i": tile_i, "run": ctile}


def bound_rho(hw: int) -> float:
    """The relative bound of matched_filter32 against the float64 correlation with the float64 template: its K
    fused multiply-adds of non-negative terms err by at most gamma_K = K u / (1 - K u) of the value, and the float32
    template by u; 1 % larger."""
    u = 2.0 ** -24
    K = 2 * int(hw) + 1
    return 1.01 * (K * u / (1.0 - K * u) + 2.0 * u)


def pursuit_verified(resp: torch.Tensor, pf: torch.Tensor, seg0, lens, counts, errb: torch.Tensor, hw: int,
                     tmpl: torch.Tensor, stats: dict | None = None, rho: float = 0.0, mf: dict | None = None):
    """The verified pursuit (module docstring). ``resp`` (n_rows, >= max(lens)) float32 CUDA with contiguous rows (a
    strided view is fine): the float32 matched-filter response, whose error bin b of row j bounds by
    ``rho * resp[j, b] + errb[j]`` (rho: a relative bound, errb: an absolute one, per row); ``pf`` the
    float32 probability stream (1-D) whose row j starts at ``seg0[j]``. Returns the picks per row (int64 arrays, in
    pick order) as ``pursuit`` does, and None for a row that needed more exact evaluations than CAND_CAP for one pick
    (the caller decodes those rows in float64). ``mf``: matched_filter32's info for this response -- every row None
    when its flag is set; its tile maxima start the runs, and a run's bounds are written only when the pursuit
    touches it (the same picks). ``stats`` (optional dict) gets "evals": exact evaluations per row."""
    n_rows = int(resp.shape[0])
    if n_rows == 0:
        return []
    if resp.dtype != torch.float32 or resp.stride(1) != 1 or pf.dtype != torch.float32 or not pf.is_contiguous():
        raise ValueError("pursuit_verified: a float32 response with contiguous rows and a contiguous float32 stream")
    device = resp.device
    max_picks = int(max(counts)) if len(counts) else 0
    if max_picks <= 0:
        return [np.zeros(0, np.int64) for _ in range(n_rows)]
    lens_t = torch.as_tensor([int(v) for v in lens], dtype=torch.int64, device=device)
    cnt_t = torch.as_tensor([int(v) for v in counts], dtype=torch.int64, device=device)
    seg_t = torch.as_tensor([int(v) for v in seg0], dtype=torch.int64, device=device)
    err_t = errb.to(device=device, dtype=torch.float64).contiguous()
    tmpl_t = tmpl.to(device=device, dtype=torch.float64).contiguous()
    L_max = int(max(lens))
    lazy = mf is not None and -(-L_max // int(mf["run"])) <= MAX_RUNS_V
    if lazy:
        run = int(mf["run"])
        tiles = (mf["tile_v"], mf["tile_i"], (int(mf["tile_v"].stride(0)), "q"))
    else:
        run = RUN
        while -(-L_max // run) > MAX_RUNS_V:
            run += RUN
        tiles = ((0, "q"), (0, "q"), (0, "q"))                     # null: every bound written first
    n_runs = -(-L_max // run)
    bounds = torch.empty((n_rows, L_max), dtype=torch.float64, device=device)
    exact = torch.empty((n_rows, L_max), dtype=torch.uint8, device=device)     # cleared where a run's bounds go
    cand = torch.empty((n_rows, CAND_CAP), dtype=torch.int32, device=device)
    picks = torch.empty((n_rows, max_picks), dtype=torch.int64, device=device)
    pick_v = torch.empty((n_rows, max_picks), dtype=torch.float64, device=device)
    n_picks = torch.empty(n_rows, dtype=torch.int32, device=device)
    n_evals = torch.empty(n_rows, dtype=torch.int32, device=device)
    bad = mf["bad"] if mf is not None else torch.zeros(1, dtype=torch.int32, device=device)
    return _launch_pursuit(resp, pf, seg_t, lens_t, cnt_t, err_t, tmpl_t, hw, rho, run, L_max, max_picks, bad,
                           tiles + ((0, "q"), (0, "q")), n_runs * (8 + 4 + 4 + 4), stats)


def _launch_pursuit(resp, pf, seg_t, lens_t, cnt_t, err_t, tmpl_t, hw, rho, run, L_max, max_picks, bad, tail,
                    smem, stats):
    """setsuna_pursuit_verified over resp's rows; ``tail`` its last five arguments (the tile maxima or bounds)."""
    device = resp.device
    n_rows = int(resp.shape[0])
    bounds = torch.empty((n_rows, L_max), dtype=torch.float64, device=device)
    exact = torch.empty((n_rows, L_max), dtype=torch.uint8, device=device)     # cleared where a run's bounds go
    cand = torch.empty((n_rows, CAND_CAP), dtype=torch.int32, device=device)
    picks = torch.empty((n_rows, max_picks), dtype=torch.int64, device=device)
    pick_v = torch.empty((n_rows, max_picks), dtype=torch.float64, device=device)
    n_picks = torch.empty(n_rows, dtype=torch.int32, device=device)
    n_evals = torch.empty(n_rows, dtype=torch.int32, device=device)
    n_runs = -(-int(L_max) // int(run))
    run_head = torch.empty((n_rows, n_runs), dtype=torch.int32, device=device)
    run_tail = torch.empty((n_rows, n_runs), dtype=torch.int32, device=device)
    pick_next = torch.empty((n_rows, max(1, int(max_picks))), dtype=torch.int32, device=device)
    n_chunks = -(-int(L_max) // CHUNK)                    # the kernel's chunk maxima (written before they are read)
    chunk_v = torch.empty((n_rows, n_chunks), dtype=torch.float64, device=device)
    chunk_i = torch.empty((n_rows, n_chunks), dtype=torch.int32, device=device)
    prof = torch.zeros((n_rows, 12) if _PURSUIT_PROF else (1,), dtype=torch.int64, device=device)
    _kernel(device, "setsuna_pursuit_verified").launch(
        (n_rows,), (VERIFIED_NT,), smem,
        (resp, (int(resp.stride(0)), "q"), bounds, (int(bounds.stride(0)), "q"), pf, seg_t, lens_t, cnt_t, err_t,
         (float(rho), "d"), tmpl_t, (int(hw), "i"),
         (int(run), "i"), exact, (int(exact.stride(0)), "q"), cand, picks, pick_v, (max_picks, "i"), n_picks, n_evals,
         bad) + tuple(tail) + (run_head, run_tail, (n_runs, "q"), pick_next, chunk_v, chunk_i,
                                 (n_chunks, "q"), prof))
    n_h = n_picks.cpu().numpy()
    p_h = picks.cpu().numpy()
    if stats is not None:
        stats["evals"] = n_evals.cpu().numpy()
        if _PURSUIT_PROF:
            stats["prof"] = prof.cpu().numpy()
    return [None if int(n_h[j]) < 0 else p_h[j, :int(n_h[j])].astype(np.int64) for j in range(n_rows)]


def pursuit_from_bounds(pf: torch.Tensor, seg0, lens, counts, errb: torch.Tensor, hw: int, tmpl: torch.Tensor,
                        stats: dict | None = None, rho: float = 0.0, cmax=None):
    """The verified pursuit with the float32 matched filter computed only where the pursuit needs it: each run of
    CTILE bins starts from a bound of its tile (setsuna_tile_bounds: the largest probability under its windows
    times the template's sum) and gets its responses -- matched_filter32's bits -- when the pursuit first reaches
    it; the rest never is. The same picks as matched_filter32 + pursuit_verified, which the bounds only steer.
    Returns what pursuit_verified returns (every row None when a probability is negative or NaN), or None when a
    segment has more runs than the kernel holds (the caller then computes the matched filter ahead). ``cmax`` =
    (chunk, maxima): the stream's per-chunk maxima (predict_fine_outputs' ``prob_cmax``) -- the bounds from them
    (setsuna_tile_bounds_cmax) instead of from the stream: a little looser, the same picks."""
    n_rows = len(lens)
    if n_rows == 0:
        return []
    if pf.dtype != torch.float32 or not pf.is_contiguous():
        raise ValueError("pursuit_from_bounds: a contiguous float32 stream")
    device = pf.device
    max_picks = int(max(counts)) if len(counts) else 0
    if max_picks <= 0:
        return [np.zeros(0, np.int64) for _ in range(n_rows)]
    run = MF_CT * MF_CR
    L_max = int(max(lens))
    n_runs = -(-L_max // run)
    if n_runs > MAX_RUNS_V:
        return None
    seg_t = torch.as_tensor([int(v) for v in seg0], dtype=torch.int64, device=device)
    lens_t = torch.as_tensor([int(v) for v in lens], dtype=torch.int64, device=device)
    cnt_t = torch.as_tensor([int(v) for v in counts], dtype=torch.int64, device=device)
    err_t = errb.to(device=device, dtype=torch.float64).contiguous()
    tmpl_t = tmpl.to(device=device, dtype=torch.float64).contiguous()
    g32 = tmpl.to(device=device, dtype=torch.float32).contiguous()
    K = 2 * int(hw) + 1
    # the float64 template's sum, a little larger (its own rounding and the product's are ~1e-15)
    gsum = float(tmpl_t.sum()) * (1.0 + 1e-12)
    bad = torch.zeros(1, dtype=torch.int32, device=device)
    tile_b = torch.empty((n_rows, n_runs), dtype=torch.float64, device=device)
    if cmax is not None and int(cmax[1].numel()) * int(cmax[0]) >= int(pf.numel()) and cmax[1].device == device:
        _kernel(device, "setsuna_tile_bounds_cmax").launch(
            (-(-n_runs // 256), n_rows), (256,), 0,
            (cmax[1], (int(cmax[0]), "q"), seg_t, lens_t, (int(hw), "i"), (gsum, "d"), tile_b, (n_runs, "q"),
             (n_runs, "i"), bad))
    else:
        _kernel(device, "setsuna_tile_bounds").launch(
            (n_runs, n_rows), (256,), 0,
            (pf, seg_t, lens_t, (int(hw), "i"), (gsum, "d"), tile_b, (n_runs, "q"), bad))
    resp = torch.empty((n_rows, L_max), dtype=torch.float32, device=device)   # written where a run starts
    smem = n_runs * (8 + 4 + 4 + 4) + 4 * (K + run + K - 1 + MF_CR)
    if smem > SMEM_MAX - SMEM_STATIC_V:
        return None                       # the caller computes the matched filter ahead (pursuit_verified)
    return _launch_pursuit(resp, pf, seg_t, lens_t, cnt_t, err_t, tmpl_t, hw, rho, run, L_max, max_picks, bad,
                           ((0, "q"), (0, "q"), (n_runs, "q"), tile_b, g32), smem, stats)
