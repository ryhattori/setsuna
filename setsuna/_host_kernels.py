"""Compiled loops for SETSUNA's CPU path (numba), imported by the first CPU session that needs them.

gate_mix, gate_mix_bias, ensemble_overlap_add, finish_streams: the host path's tail after the heads
(setsuna._predict_host_fused), each float32 operation as torch / numpy round it, in one pass each instead of a pass (and
a fresh 80-400 MB array) per operation. No fastmath: every product, sum and quotient is rounded on its own, as in the
element-wise kernels.
"""
import numba
import numpy as np

from setsuna._support._openmp import one_launch_at_a_time


@one_launch_at_a_time
@numba.njit(cache=True, nogil=True, parallel=True)
def gate_mix(bc, cc, bb, cb, w, out_c, out_b):
    """BinarySRHybrid's gate mix of a window batch: out_c = (1 - w) * bc + w * cc and out_b = (1 - w) * bb + w * cb,
    row b weighted by w[b, 0] -- torch's `(1.0 - w) * bc + w * cc`, each product and the sum rounded to float32.
    out_c / out_b may be bc / bb themselves (element by element, in place)."""
    n_rows, n = bc.shape
    parts = max(1, min(64, n // 4096))              # row pieces, so a few long rows still use every thread
    step = n // parts
    for t in numba.prange(n_rows * parts):
        b = t // parts
        p = t % parts
        lo = p * step
        hi = n if p == parts - 1 else lo + step
        wb = w[b, 0]
        omw = np.float32(1.0) - wb
        for j in range(lo, hi):
            out_c[b, j] = omw * bc[b, j] + wb * cc[b, j]
            out_b[b, j] = omw * bb[b, j] + wb * cb[b, j]


@one_launch_at_a_time
@numba.njit(cache=True, nogil=True, parallel=True)
def gate_mix_bias(bc, cc, bb, cb, w, bias_bc, bias_cc, bias_bb, bias_cb, M, out_c, out_b):
    """gate_mix of heads computed without their Linear's bias: each head value first gets its bias -- bias_*[j % M]
    for column j of a (windows, L * M) row, one float32 add, what torch.addmm's result holds where the product is
    added to the broadcast bias after it is formed (setsuna._late_bias_ok) -- then the mix as gate_mix."""
    n_rows, n = bc.shape
    parts = max(1, min(64, n // 4096))
    step = n // parts
    for t in numba.prange(n_rows * parts):
        b = t // parts
        p = t % parts
        lo = p * step
        hi = n if p == parts - 1 else lo + step
        wb = w[b, 0]
        omw = np.float32(1.0) - wb
        m = lo % M
        for j in range(lo, hi):
            vbc = bc[b, j] + bias_bc[m]
            vcc = cc[b, j] + bias_cc[m]
            vbb = bb[b, j] + bias_bb[m]
            vcb = cb[b, j] + bias_cb[m]
            out_c[b, j] = omw * vbc + wb * vcc
            out_b[b, j] = omw * vbb + wb * vcb
            m += 1
            if m == M:
                m = 0


@one_launch_at_a_time
@numba.njit(cache=True, nogil=True, parallel=True)
def ensemble_overlap_add(S, Sb, P, Q, has_s, n_models, starts, W, M, pred, pred_b):
    """One window chunk of the overlap-add. Window i's ensemble mean is ((p_0 + ... + p_{n-2}) + p_{n-1}) / n --
    torch.stack(ps).mean(0): the models summed in order, then divided -- with the first n-1 models' sum in S / Sb
    (has_s) and the last model in P / Q; it is added to the M bins of each frame it covers, window after window, as
    `pred[s0*M:(s0+W)*M] += pr[i]` adds it. starts: this chunk's window starts (ascending); pred / pred_b hold the
    earlier chunks' sums."""
    nw = starts.shape[0]
    f0 = starts[0]
    f1 = starts[nw - 1] + W
    nmf = np.float32(n_models)
    for f in numba.prange(f0, f1):
        hi = np.searchsorted(starts, f, side="right")
        lo = np.searchsorted(starts, f - W, side="right")
        base = f * M
        for i in range(lo, hi):
            off = (f - starts[i]) * M
            for j in range(M):
                if has_s:
                    v = S[i, off + j] + P[i, off + j]
                    u = Sb[i, off + j] + Q[i, off + j]
                else:
                    v = P[i, off + j]
                    u = Q[i, off + j]
                pred[base + j] = pred[base + j] + v / nmf
                pred_b[base + j] = pred_b[base + j] + u / nmf


@numba.njit(cache=True, nogil=True)
def _leaf32(a, lo, n):
    """numpy's float32 pairwise sum of n <= 128 values a[lo:lo + n]: sequential from 0 below 8, else 8 accumulators
    and the rest after them (FLOAT_pairwise_sum)."""
    if n < 8:
        res = np.float32(0.0)
        for i in range(n):
            res += a[lo + i]
        return res
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    i = 8
    m = n - (n % 8)
    while i < m:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@numba.njit(cache=True, nogil=True)
def _pw32_stack(a, lo, n):
    """numpy's float32 pairwise sum of a[lo:lo + n] for any n: halves (the first a multiple of 8) down to 128 values,
    walked with an explicit stack, left before right (compiled recursion broke numba's disk cache elsewhere)."""
    los = np.empty(64, dtype=np.int64)
    ns = np.empty(64, dtype=np.int64)
    phase = np.zeros(64, dtype=np.int64)
    vals = np.empty(64, dtype=np.float32)
    vp = 0
    los[0] = lo
    ns[0] = n
    sp = 1
    while sp > 0:
        sp -= 1
        l0 = los[sp]
        m = ns[sp]
        if m <= 128:
            vals[vp] = _leaf32(a, l0, m)
            vp += 1
            continue
        if phase[sp] == 1:
            vals[vp - 2] = vals[vp - 2] + vals[vp - 1]
            vp -= 1
            continue
        n2 = m // 2
        n2 -= n2 % 8
        phase[sp] = 1
        sp += 1
        los[sp] = l0 + n2
        ns[sp] = m - n2
        phase[sp] = 0
        sp += 1
        los[sp] = l0
        ns[sp] = n2
        phase[sp] = 0
        sp += 1
    return vals[0]


@numba.njit(cache=True, nogil=True)
def _pw32(a, lo, n):
    """numpy's float32 pairwise sum of a[lo:lo + n] (np.sum's order within one inner-loop call): 128-value leaves
    under halvings; 256 / 512 / 1024 values (the model's bins per frame) written out, anything else by the stack."""
    if n <= 128:
        return _leaf32(a, lo, n)
    if n == 256:
        return _leaf32(a, lo, 128) + _leaf32(a, lo + 128, 128)
    if n == 512:
        return ((_leaf32(a, lo, 128) + _leaf32(a, lo + 128, 128))
                + (_leaf32(a, lo + 256, 128) + _leaf32(a, lo + 384, 128)))
    if n == 1024:
        return (((_leaf32(a, lo, 128) + _leaf32(a, lo + 128, 128))
                 + (_leaf32(a, lo + 256, 128) + _leaf32(a, lo + 384, 128)))
                + ((_leaf32(a, lo + 512, 128) + _leaf32(a, lo + 640, 128))
                   + (_leaf32(a, lo + 768, 128) + _leaf32(a, lo + 896, 128))))
    return _pw32_stack(a, lo, n)


@one_launch_at_a_time
@numba.njit(cache=True, nogil=True, parallel=True)
def finish_streams(pred, pred_b, starts, W, M, n_frames, k, rate, lam_pool, prob_pool):
    """After the overlap-add, frames 0..n_frames-1: each frame's M bins divided by the number of windows covering it
    (the host path's `pred /= cov`, in place), its rate (count_head_rate: np.sum over the M count bins) and, for
    k > 0, the sums over k bins of both streams (pool_fine) -- each numpy's pairwise float32 sum."""
    nk = M // k if k > 0 else 0
    for f in numba.prange(n_frames):
        hi = np.searchsorted(starts, f, side="right")
        lo = np.searchsorted(starts, f - W, side="right")
        c = np.float32(hi - lo)
        base = f * M
        for j in range(M):
            pred[base + j] = pred[base + j] / c
            pred_b[base + j] = pred_b[base + j] / c
        rate[f] = _pw32(pred, base, M)
        for q in range(nk):
            lam_pool[f * nk + q] = _pw32(pred, base + q * k, k)
            prob_pool[f * nk + q] = _pw32(pred_b, base + q * k, k)
