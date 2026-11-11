"""[C87 2026-08-05] Shared structured-decode readout -- ONE definition of `struct_decode`.

MOVED VERBATIM out of score_timing.py (it was defined there, and score_timing cannot be imported
by another scorer: it os.chdir()s, setdefaults MRES_MS=512 and imports models_binary_sr_mres,
which MONKEY-PATCHES BinarySRHybrid -- importing it from eval_binary_sr would silently change the
model eval_binary_sr is scoring).

The reason this file exists: eval_binary_sr's table-grid `f1_sub_*` must be OUR NATIVE decode, and
the native decode is this function. Copying it into eval_binary_sr would be the C76/C83 failure
shape (a parallel code path that drifts from the original). score_timing now imports from here, so
there is exactly one implementation and the headline jitter/F1 numbers and the grid columns cannot
disagree about what "our timing" means.

Env knobs are read at IMPORT time from the same variables score_timing used, so behaviour is
unchanged for every existing caller.
"""
from __future__ import annotations
import os
import numpy as np

_DECODE_OMP = os.environ.get("DECODE_OMP", "0") == "1"   # struct_decode variant: orthogonal MP (joint amplitude re-fit each step)
_DECODE_TMPL = None                                       # struct_decode matched template (empirical head impulse response); set below if DECODE_MATCHED=1
_DECODE_SEARCHSET = os.environ.get("DECODE_SEARCHSET", "0") == "1"  # (a) EXACT combinatorial decode over head-peak combos (binary-SR-style; low-fps only, falls back to MP if intractable)
SIG_MS = float(os.environ.get("SIG_MS", "3.0"))          # matched-filter width = training bin_sig_ms=3 (2026-07-16 sweep)


def struct_decode(pf, ffr, sig_bins, N=None, thr_q=None, max_n=None):
    """Matching-pursuit decode of a soft-ms head prob pf: matched-filter with the Gaussian bump (width sig_bins),
    then greedily pick the max, record the spike, SUBTRACT the scaled template. Stop at N (oracle) or when the
    matched-filter max drops below the thr_q-percentile. Returns spike times (s). (score_ladder_fps ~L418).
    Env variants (2026-07-16 decode-ablation): DECODE_OMP=1 -> orthogonal MP (re-solve ALL amplitudes vs pf each
    step by least-squares -> correct overlap/close-pair handling); DECODE_REFRAC_MS>0 -> hard min-ISI (forbid a
    pick within this many ms of an accepted spike). Templates are external if DECODE_TMPL is set (matched template)."""
    L = len(pf); sig = max(float(sig_bins), 0.5); hw = max(1, int(round(3 * sig)))
    tt = np.arange(-hw, hw + 1); tmpl = np.exp(-0.5 * (tt / sig) ** 2)
    if _DECODE_TMPL is not None:                                               # matched template (empirical head impulse response)
        tmpl = _DECODE_TMPL; hw = len(tmpl) // 2
    pf64 = pf.astype(np.float64)
    work = np.convolve(pf64, tmpl[::-1], mode="same")                          # matched-filter response
    thr = float(np.percentile(work, thr_q)) if thr_q is not None else 0.0
    cap = int(N) if N is not None else (int(max_n) if max_n else 100000)
    _refb = int(round(float(os.environ.get("DECODE_REFRAC_MS", "0")) / 1000.0 * ffr))   # refractory (bins); 0=off
    if _DECODE_SEARCHSET and N is not None and int(N) >= 1:                              # (a) EXACT combinatorial decode over head-peak combos (binary-SR-style)
        loc = np.where((work[1:-1] >= work[:-2]) & (work[1:-1] > work[2:]))[0] + 1        # candidate positions = matched-filter local maxima
        if len(loc) >= int(N):
            import math, itertools
            cand = loc[np.argsort(work[loc])[::-1][:min(len(loc), int(N) + 6)]]           # top N+6 candidate peaks (bound the combinatorics)
            if math.comb(len(cand), int(N)) <= 12000:                                     # only when tractable (low fps / few spikes); else fall back to MP
                best_c, best_s = None, -1e18
                for c in itertools.combinations(sorted(int(x) for x in cand), int(N)):
                    rc = np.zeros(L, dtype=np.float64)
                    for s in c:
                        a = max(0, s - hw); b = min(L, s + hw + 1); rc[a:b] += tmpl[hw - (s - a): hw + (b - s)]
                    sc = float(np.dot(rc, pf64) / (np.linalg.norm(rc) + 1e-9))            # corr of the N-spike template-sum with the head
                    if sc > best_s: best_s, best_c = sc, c
                if best_c is not None:
                    return np.sort(np.array(best_c, dtype=np.float64)) / ffr
    spikes = []; recon = np.zeros(L, dtype=np.float64) if _DECODE_OMP else None
    for _ in range(cap):
        i = int(np.argmax(work))
        if work[i] <= thr: break
        spikes.append(i)
        if _DECODE_OMP:                                                        # SPARSE orthogonal MP: re-fit only the overlap CLUSTER (dense L*k lstsq timed out)
            cl = sorted(s for s in spikes if abs(s - i) <= 2 * hw)             # spikes whose templates overlap the new one
            clo = max(0, cl[0] - hw); chi = min(L, cl[-1] + hw + 1)
            Dl = np.zeros((chi - clo, len(cl)), dtype=np.float64)
            for j, s in enumerate(cl):
                a = max(0, s - hw); b = min(L, s + hw + 1); Dl[a - clo:b - clo, j] = tmpl[hw - (s - a): hw + (b - s)]
            al, *_ = np.linalg.lstsq(Dl, pf64[clo:chi], rcond=None)
            recon[clo:chi] = Dl @ al                                           # local LS re-fit (clusters >hw apart -> exact)
            work = np.convolve(pf64 - recon, tmpl[::-1], mode="same")          # fresh residual matched-filter
        else:                                                                 # greedy MP: subtract scaled template
            lo = max(0, i - hw); hi = min(L, i + hw + 1)
            seg = tmpl[hw - (i - lo): hw + (hi - i)]
            work[lo:hi] -= work[i] * seg
        if _refb > 0:                                                         # hard refractory on ALL accepted spikes (survives OMP work reset)
            for s in spikes: work[max(0, s - _refb): min(L, s + _refb + 1)] = -1e18
    return np.sort(np.array(spikes, dtype=np.float64)) / ffr


#
# SWEEP RESULT (10 rates x 9 radii x 86 records / 39 neurons / 4 indicators): the radius tracks the
# TOLERANCE, not the fps -- best is ~5 ms at tol 5, ~8-12 at tol 10, ~12-20 at tol 25 -- and C107's
# claim that "the optimum grows as fps falls" is NOT supported (that came from 3 rates plus a
# tolerance confound). But the surface is FLAT: across the cells where `amp_top` is actually ROUTED
# (tol>=15 ms, eff>12 Hz) the mean gain of 12 ms over 5 ms is **+0.0024 F1 -- eleven times under the
# 0.027 seed floor** -- and it is NEGATIVE at 60 Hz (-0.0011 pooled, -0.0116 on 8f fold 0 alone).
#
# [!] I RAISED THIS TO 12 AND REVERTED IT. Shipping a default move on a +0.0024 mean is exactly the
# sub-floor-noise mistake this project keeps re-learning, and it would have forced a re-baseline of
# every published amp column for nothing. 5 ms is what every C107 measurement used. The real finding
# is only that the radius must not be tiny (<3 ms) or huge (>30 ms); inside 3-20 ms it does not matter.
#
# The optimum DOES climb steeply below ~8 Hz (16 ms at 8 Hz, 20 at 5, 30 at 3) where 5 ms costs
# -0.045..-0.072 -- but the router sends those rates to POOL, so it never bites. Set AMP_NMS_MS
# explicitly if you use amp_top off-route at low fps.
AMP_NMS_MS = float(os.environ.get("AMP_NMS_MS", "5.0"))


def _topk_nms(loc, score, ffr, N, nms_ms=0.0):
    """Top-N candidate positions by `score`, with an optional hard NMS radius. Returns times (s).

    ONE definition: `amptop_decode` below calls it and `peak_features.select_topk` delegates to it,
    so the deployed readout and the C106/C107 offline harness cannot drift (the C76/C83/C87 shape).
    """
    loc = np.asarray(loc); score = np.asarray(score, np.float64)
    if len(loc) == 0 or N is None or int(N) <= 0:
        return np.zeros(0, np.float64)
    order = np.argsort(score)[::-1]
    if nms_ms <= 0:
        sel = order[:int(N)]
    else:
        sep = max(1, int(round(nms_ms / 1000.0 * ffr)))
        sel = []; taken = []
        for j in order:
            p = loc[j]
            if taken and np.min(np.abs(np.asarray(taken) - p)) < sep:
                continue
            sel.append(j); taken.append(p)
            if len(sel) >= int(N):
                break
        sel = np.asarray(sel, int)
    return np.sort(np.asarray(loc, np.float64)[sel]) / ffr


def amptop_decode(pf, ffr, sig_bins, N, nms_ms=None, work=None):
    """[C107 2026-08-07] AMPLITUDE TOP-N: take the N strongest matched-filter LOCAL MAXIMA, NMS'd.

    The third routing arm. Same head, same candidate set and same budget as `struct_decode` -- the
    ONLY difference is the suppression: a hard radius instead of template subtraction, and hence no
    residual and no re-ranking. Deliberately trivial; the point is that it WINS where it wins.

    WHEN TO USE IT -- **>=20 Hz AND wide tolerance (~25 ms) ONLY.** Measured on 39 neurons / 86
    records / 4 indicators vs the routed struct/POOL incumbent: 20 Hz +0.0703 CI[+0.031,+0.098],
    60 Hz +0.0443 CI[+0.023,+0.059], both P=1.000. It LOSES at <=10 ms (-0.03..-0.08), loses at
    every tolerance at 8 Hz, and its jitter is 7-10 ms vs struct's 3.0-3.5. It is a DETECTION-mode
    readout exactly as DECODE_JOINT is -- never use it for timing, jitter or connectivity.

    [!] PEAK-RESTRICTED BY CONSTRUCTION, and that is precisely why it loses at tight tolerance:
    C108 measured that restricting `struct` to these same candidates costs f1@5 -0.05..-0.08,
    because the real MP searches the full residual and can place BETWEEN local maxima.

    Uses the PURE fine head (not the DECODE_JOINT count-modulated one) -- that is what was measured.
    `work` lets a caller pass a matched-filter response it already computed, to skip the convolution.
    """
    if work is None:
        tmpl, _hw = _mk_tmpl(sig_bins)
        work = np.convolve(np.asarray(pf, np.float64), tmpl[::-1], mode="same")
    loc = np.where((work[1:-1] >= work[:-2]) & (work[1:-1] > work[2:]))[0] + 1
    return _topk_nms(loc, work[loc], ffr, N, AMP_NMS_MS if nms_ms is None else nms_ms)


def poolmp_decode(pf, eff, n_frames, m, pool_m, sig_ms, N):
    """[C122 2026-08-08] MATCHING PURSUIT ON A POOLED GRID — pool FIRST, then MP.

    THE GAP THIS FILLS (user question): every decode tried so far either runs MP at the FULL m512
    resolution (`struct`, OMP, and the C65/C97 frame-constrained variants) or pools and then does a
    plain top-N peak-pick (`POOL`, via `peaks_topN`). **Pooling and THEN running MP was never tested.**

    WHY IT MIGHT MATTER: the head carries a `bin_sig_ms=3` smoothed target, so its real resolution is
    ~3 ms whatever `m` is. At 60 Hz a frame is 16.7 ms — about 5.6 resolvable slots — carried on 512
    sub-bins, i.e. ~90x oversampled; at 1 Hz it is ~333 slots on 512, nearly matched. So the
    oversampling is WORST at high fps, and C98 already noted the consequence for MP: adjacent atoms
    sit ~0.1 ms apart under a ~3 ms template, giving mutual coherence ~0.999, which is precisely the
    regime where greedy support recovery is weakest. Pooling to `pool_m` before MP cuts that coherence
    without changing the head.

    THE PREDICTABLE COST: a coarser grid gives up sub-bin placement, and C108 measured that
    peak-restricting `struct` costs f1@5 −0.05..−0.08. So expect this to help at loose tolerance and
    hurt at tight — the same axis as OMP and wider SIG_MS. Worth measuring because a slide and an
    escape look identical until both metrics are plotted.

    Uses the PURE fine head (no DECODE_JOINT count modulation), so the ONLY difference from `struct`
    is the grid. `pool_m` = sub-bins per FRAME after pooling (m itself = no pooling = plain struct).
    """
    pf = np.asarray(pf, np.float64)
    if int(pool_m) >= int(m):
        return struct_decode(pf, m * eff, sig_ms / 1000.0 * m * eff, N)
    pooled = pf.reshape(int(n_frames), int(pool_m), int(m) // int(pool_m)).sum(2).reshape(-1)
    ffr_p = float(pool_m) * float(eff)
    return struct_decode(pooled, ffr_p, sig_ms / 1000.0 * ffr_p, N)


def pool_m_for_ms(target_ms, eff, m):
    """[C125] pool_m giving a pooled bin of ~target_ms, snapped to a power of two dividing m.

    WHY THIS EXISTS: `pool_m` is bins per FRAME, so a FIXED pool_m has a bin width of
    (1000/fps)/pool_m -- 3.6 ms at 35 Hz but 25 ms at 5 Hz for pool_m=8. The head's resolution is
    ~3 ms at every rate (bin_sig_ms=3), so a fixed pool_m is badly mismatched at low fps. Holding the
    bin width CONSTANT IN MILLISECONDS is the principled parameterisation, and it is the same
    fps-adaptive-m point the m-ladder work already recorded.
    """
    want = 1000.0 / (max(float(target_ms), 1e-6) * max(float(eff), 1e-6))   # bins per frame
    best, bd = 1, 1e18
    pm = 1
    while pm <= int(m):
        if int(m) % pm == 0:
            d = abs(np.log(max(pm, 1e-9)) - np.log(max(want, 1e-9)))
            if d < bd: bd, best = d, pm
        pm *= 2
    return best


def budget_decode(pf, counts, ffr, m, sig_bins, N=None, halo=1, thr=0.0, soft=0.0):
    """[C97 2026-08-06] MP with GLOBAL search under a PER-FRAME COUNT BUDGET from the discretiser.

    THE POINT, and how it differs from `hybrid_decode` (which C65 refuted): hybrid CONFINES matching
    pursuit to [frame-halo, frame+halo] and so inherits the discretiser's FRAME CHOICE -- importing
    frame-blindness, which is what destroys tight-tolerance F1 (measured -0.099/-0.113 at f1@10).
    Here MP is NOT confined: every step still picks the GLOBAL argmax of the residual, exactly as
    `struct_decode` does. The discretiser contributes only a per-frame CAPACITY, so a burst cannot
    absorb the whole budget (the pathology that makes MP weak at loose tolerance) while placement
    stays globally free (the property that makes MP strong at tight tolerance).

    counts : per-frame integer counts from cascade_discretize (length n_frames)
    m      : sub-bins per frame, so frame(fine_bin i) = i // m
    halo   : a pick may be charged to any frame within +-halo that still has capacity. halo=0 would
             re-import frame-blindness at boundaries: a spike whose peak lands one bin across a frame
             edge would be refused even though the discretiser counted it next door.
    """
    L = len(pf); sig = max(float(sig_bins), 0.5); hw = max(1, int(round(3 * sig)))
    tt = np.arange(-hw, hw + 1); tmpl = np.exp(-0.5 * (tt / sig) ** 2)
    pf64 = pf.astype(np.float64)
    work = np.convolve(pf64, tmpl[::-1], mode="same")
    budget = np.asarray(counts, dtype=np.int64).copy()
    nfr = len(budget)
    cap = int(N) if N is not None else int(budget.sum())
    spikes = []
    while len(spikes) < cap:
        i = int(np.argmax(work))
        if work[i] <= thr:
            break
        f = min(nfr - 1, i // m)
        tgt = -1
        for d in range(0, halo + 1):                      # nearest frame with capacity, own frame first
            for ff in ((f,) if d == 0 else (f - d, f + d)):
                if 0 <= ff < nfr and budget[ff] > 0:
                    tgt = ff; break
            if tgt >= 0: break
        if tgt < 0:
            if soft > 0.0:
                # [C97-soft] SOFT mode: a frame over its expected count is DISCOURAGED, not forbidden.
                # The hard cap is what forbids MP from putting two picks in one frame when the sub-frame
                # evidence genuinely supports it -- the property that makes MP strong at tight tolerance.
                # Scaling the residual keeps that placement reachable when the evidence is overwhelming.
                work[f * m: (f + 1) * m] *= soft
                spikes.append(i)
                lo = max(0, i - hw); hi = min(L, i + hw + 1)
                work[lo:hi] -= work[i] * tmpl[hw - (i - lo): hw + (hi - i)]
                continue
            work[f * m: (f + 1) * m] = -1e18              # hard: forbid this frame, retry
            continue
        spikes.append(i); budget[tgt] -= 1
        lo = max(0, i - hw); hi = min(L, i + hw + 1)
        work[lo:hi] -= work[i] * tmpl[hw - (i - lo): hw + (hi - i)]
        if budget[tgt] == 0:                              # frame full -> close it
            work[tgt * m: (tgt + 1) * m] = -1e18
    return np.sort(np.array(spikes, dtype=np.float64)) / ffr


# ---------------------------------------------------------------------------

def _mk_tmpl(sig_bins):
    sig = max(float(sig_bins), 0.5); hw = max(1, int(round(3 * sig)))
    tt = np.arange(-hw, hw + 1)
    return np.exp(-0.5 * (tt / sig) ** 2), hw


def _ls_support(y, S, tmpl, hw, L):
    """LS amplitudes for atoms at sorted positions S, solved per NON-OVERLAPPING CLUSTER."""
    S = np.sort(np.asarray(S, dtype=int))
    if len(S) == 0: return S, np.zeros(0)
    amps = np.zeros(len(S)); groups = []; cur = [0]
    for k in range(1, len(S)):
        if S[k] - S[k - 1] <= 2 * hw:
            cur.append(k)
        else:
            groups.append(cur); cur = [k]      # NEW list -- rebinding, not mutating the one just stored
    groups.append(cur)
    for g in groups:
        idx = S[g]; lo = max(0, idx[0] - hw); hi = min(L, idx[-1] + hw + 1)
        Dl = np.zeros((hi - lo, len(g)))
        for j, s in enumerate(idx):
            a = max(0, s - hw); b = min(L, s + hw + 1)
            Dl[a - lo:b - lo, j] = tmpl[hw - (s - a): hw + (b - s)]
        sol, *_ = np.linalg.lstsq(Dl, y[lo:hi], rcond=None)
        amps[g] = sol
    return S, amps


def _recon(S, amps, tmpl, hw, L):
    r = np.zeros(L)
    for s, a in zip(S, amps):
        lo = max(0, s - hw); hi = min(L, s + hw + 1)
        r[lo:hi] += a * tmpl[hw - (s - lo): hw + (hi - s)]
    return r


def _peaks(v, k, hw):
    """k strongest LOCAL MAXIMA, min-separated by hw (a coherent dictionary otherwise returns k
    adjacent shifts of the same spike)."""
    if k <= 0: return np.zeros(0, int)
    loc = np.where((v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
    if len(loc) == 0: return np.zeros(0, int)
    loc = loc[np.argsort(v[loc])[::-1]]
    out = []
    for p in loc:
        if all(abs(p - q) > hw for q in out):
            out.append(int(p))
            if len(out) >= k: break
    return np.array(sorted(out), int)


def cosamp_decode(pf, ffr, sig_bins, N, n_iter=8):
    """CoSaMP: propose 2K, merge with current support, LS, PRUNE back to K. The prune is the point --
    unlike greedy MP it can UNDO an early pick, which is exactly the burst-shredding failure mode."""
    y = np.asarray(pf, np.float64); L = len(y); tmpl, hw = _mk_tmpl(sig_bins)
    K = max(1, int(N)); S = np.zeros(0, int); amps = np.zeros(0); resid = y.copy()
    for _ in range(n_iter):
        prop = _peaks(np.convolve(resid, tmpl[::-1], "same"), 2 * K, hw)
        S = np.union1d(S, prop)
        S, a = _ls_support(y, S, tmpl, hw, L)
        keep = np.argsort(np.abs(a))[::-1][:K]
        S, amps = _ls_support(y, S[np.sort(keep)], tmpl, hw, L)
        resid = y - _recon(S, amps, tmpl, hw, L)
    return np.sort(S.astype(np.float64)) / ffr


def sp_decode(pf, ffr, sig_bins, N, n_iter=8):
    """Subspace Pursuit: like CoSaMP but proposes K (not 2K) per step. Same prune-and-refit."""
    y = np.asarray(pf, np.float64); L = len(y); tmpl, hw = _mk_tmpl(sig_bins)
    K = max(1, int(N))
    S = _peaks(np.convolve(y, tmpl[::-1], "same"), K, hw)
    S, amps = _ls_support(y, S, tmpl, hw, L)
    for _ in range(n_iter):
        resid = y - _recon(S, amps, tmpl, hw, L)
        S2 = np.union1d(S, _peaks(np.convolve(resid, tmpl[::-1], "same"), K, hw))
        S2, a2 = _ls_support(y, S2, tmpl, hw, L)
        keep = np.sort(np.argsort(np.abs(a2))[::-1][:K])
        Snew, anew = _ls_support(y, S2[keep], tmpl, hw, L)
        if len(Snew) == len(S) and np.array_equal(Snew, S): break
        S, amps = Snew, anew
    return np.sort(S.astype(np.float64)) / ffr


def lasso_decode(pf, ffr, sig_bins, N, lam_frac=0.1, n_iter=60, reweight=0, tol=1e-4):
    """FISTA on the convolutional LASSO, then take the N strongest local maxima of the sparse code.
    reweight>0 runs that many extra reweighted-L1 passes (w = 1/(|x|+eps)), which is the standard
    remedy for a coherent dictionary smearing mass across adjacent shifts."""
    y = np.asarray(pf, np.float64); L = len(y); tmpl, hw = _mk_tmpl(sig_bins)
    A = lambda x: np.convolve(x, tmpl, "same")
    AT = lambda r: np.convolve(r, tmpl[::-1], "same")
    # [!] Lipschitz constant of A^T A for a CONVOLUTION is max|FFT(tmpl)|^2, NOT sum(tmpl^2).
    # The latter is ~sqrt(2*pi)*sig times too small, so the FISTA step overshoots and the iterate
    # blows up to NaN (caught on synthetic: n_pred=0 for every trial).
    Lc = float(np.max(np.abs(np.fft.rfft(tmpl, n=max(L, len(tmpl)))) ** 2)) * 1.05 + 1e-9
    lam0 = lam_frac * float(np.max(np.abs(AT(y))))
    w = np.ones(L); x = np.zeros(L)
    for rp in range(int(reweight) + 1):
        z = x.copy(); t = 1.0; lam = lam0          # warm start: keep x across reweight passes
        for _ in range(n_iter):
            x_prev = x
            g = AT(A(z) - y)
            xn = z - g / Lc
            thr = (lam * w) / Lc
            xn = np.sign(xn) * np.maximum(np.abs(xn) - thr, 0.0)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            z = xn + ((t - 1) / tn) * (xn - x)
            x, t = xn, tn
            nx = np.linalg.norm(x)
            if nx > 0 and np.linalg.norm(x - x_prev) / nx < tol:
                break                               # converged -- stop paying for iterations
        w = 1.0 / (np.abs(x) + 1e-3 * (np.max(np.abs(x)) + 1e-12))
    return np.sort(_peaks(np.abs(x), int(N), hw).astype(np.float64)) / ffr


def sbl_decode(pf, ffr, sig_bins, N, n_iter=60, cand_mult=6, prune=1e6, snr=20.0):
    """[C100 2026-08-06] Sparse Bayesian Learning (EM / MacKay updates) on the matched-filter peaks.

    WHY SBL AND NOT LASSO: our dictionary is shifted Gaussians on a sub-ms grid -- mutual coherence
    ~0.999. L1's support-recovery guarantees need incoherence, and empirically LASSO smears mass over
    adjacent shifts (measured: on bursty synthetic it could not even return N separated peaks, 18/30).
    SBL's global optimum is the sparsest solution INDEPENDENT of coherence (Wipf & Rao), which is the
    one theoretical reason to expect a convex-family method to work here at all.

    TRACTABILITY -- the reason this is not O(L^3): atoms further apart than 2*hw do not overlap, so the
    Gram is BLOCK-DIAGONAL and the SBL posterior FACTORISES over clusters exactly. We therefore run
    full SBL independently inside each cluster (typically 1-10 candidates) instead of once over ~1e5
    bins. The Gram itself is analytic -- the template autocorrelation evaluated at the position lags --
    so D is never materialised.
    """
    y = np.asarray(pf, np.float64); L = len(y)
    tmpl, hw = _mk_tmpl(sig_bins)
    corr = np.convolve(y, tmpl[::-1], "same")
    ac = np.correlate(tmpl, tmpl, "full")          # autocorrelation -> Gram entries by lag
    aclag = len(tmpl) - 1
    loc = np.where((corr[1:-1] >= corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
    if len(loc) == 0: return np.zeros(0)
    keep = min(len(loc), max(int(cand_mult * max(1, int(N))), 8))
    cand = loc[np.argsort(corr[loc])[::-1][:keep]]
    # [!] CANDIDATES ARE THE BINDING CONSTRAINT, NOT THE ALGORITHM. Restricted to matched-filter
    # LOCAL MAXIMA, a burst supplies fewer maxima than it contains spikes -- measured n_pred=21 of 30
    # on bursty synthetic, i.e. SBL could not select what was never offered. CoSaMP/SP avoid this by
    # re-proposing from the RESIDUAL each iteration. SBL has no residual loop, so densify instead:
    # add a regular sub-template grid wherever the correlation is high.
    if len(corr):
        thr = 0.25 * float(np.max(corr))
        step = max(1, hw // 2)
        dense = np.where(corr > thr)[0]
        if len(dense): cand = np.union1d(cand, dense[::step])
    cand = np.sort(np.unique(cand))
    beta = 1.0 / max((np.var(y) / max(snr, 1e-6)), 1e-12)      # noise precision from an SNR prior
    out_pos, out_amp = [], []
    grp, cur = [], [0]
    for k in range(1, len(cand)):
        if cand[k] - cand[k - 1] <= 2 * hw: cur.append(k)
        else: grp.append(cur); cur = [k]
    grp.append(cur)
    for g in grp:
        idx = cand[g]; M = len(idx)
        d = idx[:, None] - idx[None, :] + aclag
        G = ac[np.clip(d, 0, len(ac) - 1)]                     # analytic Gram, no dictionary needed
        b = corr[idx]
        alpha = np.ones(M)
        for _ in range(n_iter):
            try: Sig = np.linalg.inv(beta * G + np.diag(alpha))
            except np.linalg.LinAlgError: break
            mu = beta * (Sig @ b)
            gam = 1.0 - alpha * np.diag(Sig)                   # MacKay well-determinedness
            with np.errstate(divide="ignore", invalid="ignore"):
                a_new = np.where(mu ** 2 > 1e-30, gam / (mu ** 2), prune)
            a_new = np.clip(np.nan_to_num(a_new, nan=prune, posinf=prune), 1e-8, prune)
            if np.allclose(a_new, alpha, rtol=1e-3): alpha = a_new; break
            alpha = a_new
        live = alpha < prune * 0.5                             # surviving (relevant) atoms
        for j in np.where(live)[0]:
            out_pos.append(int(idx[j])); out_amp.append(abs(float(mu[j])))
    if not out_pos: return np.zeros(0)
    out_pos = np.array(out_pos); out_amp = np.array(out_amp)
    take = np.argsort(out_amp)[::-1][:max(1, int(N))]          # global top-N by evidence weight
    return np.sort(out_pos[take].astype(np.float64)) / ffr


def amp_decode(pf, ffr, sig_bins, N, n_iter=40, tau=1.4, track=None):
    """[C101 2026-08-06] Approximate Message Passing (unlearned) -- the algorithm AMP-Net unrolls.

    Implemented as the GATE on AMP-Net, not as a candidate. AMP's whole advantage is the ONSAGER
    correction, which makes the effective noise Gaussian so a scalar soft-threshold is near-optimal.
    That guarantee comes from state evolution, which assumes an i.i.d.-random measurement matrix.
    OURS IS A SQUARE CONVOLUTION with mutual coherence ~0.999 (delta = 1, maximally correlated
    columns) -- the regime where AMP is documented to diverge. If the unlearned base does not
    converge here, unrolling it into a trained network starts from a broken fixed point.

    track: optional list, appended with ||x||_2 per iteration so divergence is visible, not inferred.
    """
    y = np.asarray(pf, np.float64); L = len(y)
    tmpl, hw = _mk_tmpl(sig_bins)
    A  = lambda v: np.convolve(v, tmpl, "same")
    AT = lambda v: np.convolve(v, tmpl[::-1], "same")
    x = np.zeros(L); z = y.copy(); eta_prime_mean = 0.0
    for _ in range(n_iter):
        # Onsager-corrected residual
        z = y - A(x) + (1.0 / 1.0) * z * eta_prime_mean          # delta = L/L = 1
        theta = tau * np.linalg.norm(z) / np.sqrt(L)
        r = x + AT(z)
        x = np.sign(r) * np.maximum(np.abs(r) - theta, 0.0)      # soft threshold
        eta_prime_mean = float(np.mean(np.abs(r) > theta))       # <eta'> for the next Onsager term
        if track is not None: track.append(float(np.linalg.norm(x)))
        if not np.isfinite(x).all(): break
    if not np.isfinite(x).all(): return np.zeros(0)
    return np.sort(_peaks(np.abs(x), int(N), hw).astype(np.float64)) / ffr


def vamp_decode(pf, ffr, sig_bins, N, n_iter=30, snr=400.0, damp=0.85, track=None):
    """[C103 2026-08-06] Vector AMP. The member of the thresholding family designed for CORRELATED /
    ill-conditioned operators, i.e. exactly where plain AMP diverged here (||x|| 228 -> 5e69).

    VAMP alternates an LMMSE estimate against the operator with a separable denoiser, and rescales by
    the DIVERGENCE at each step so the input to the denoiser is (approximately) white. That whitening
    is the reason to expect it to beat LASSO/LISTA on a coherence-0.999 dictionary: the thresholding
    step never sees the correlation.

    Our operator is a CONVOLUTION, hence circulant, hence diagonal in the Fourier basis -- so the
    LMMSE step and its divergence are both CLOSED FORM in O(L log L):
        x1 = F^-1[ (conj(h)Y/s2 + g1 R1) / (|h|^2/s2 + g1) ],  alpha1 = mean( g1 / (|h|^2/s2 + g1) )
    [!] CAVEAT: VAMP's state evolution assumes RIGHT-ROTATIONALLY-INVARIANT A. A convolution's singular
    vectors are the deterministic Fourier basis, so the guarantee does not formally cover this -- which
    is why `damp` exists and why this is measured rather than assumed.
    """
    y = np.asarray(pf, np.float64); L = len(y)
    tmpl, hw = _mk_tmpl(sig_bins)
    h = np.fft.rfft(tmpl, n=L)                     # circulant spectrum of the dictionary
    Y = np.fft.rfft(y)
    s2 = max(np.var(y) / max(snr, 1e-9), 1e-12)    # noise variance
    H2 = np.abs(h) ** 2
    r1 = np.zeros(L); g1 = 1.0
    x2 = np.zeros(L)
    for _ in range(n_iter):
        # --- LMMSE against the operator (closed form via FFT) ---
        den = H2 / s2 + g1
        X1 = (np.conj(h) * Y / s2 + g1 * np.fft.rfft(r1)) / den
        x1 = np.fft.irfft(X1, n=L)
        a1 = float(np.mean(g1 / den))
        a1 = min(max(a1, 1e-6), 1 - 1e-6)
        r2 = (x1 - a1 * r1) / (1 - a1); g2 = g1 * (1 - a1) / a1
        # --- separable denoiser (soft threshold at the propagated noise scale) ---
        thr = np.sqrt(1.0 / max(g2, 1e-12))
        x2n = np.sign(r2) * np.maximum(np.abs(r2) - thr, 0.0)
        x2 = damp * x2n + (1 - damp) * x2
        a2 = float(np.mean(np.abs(r2) > thr)); a2 = min(max(a2, 1e-6), 1 - 1e-6)
        r1 = (x2 - a2 * r2) / (1 - a2); g1 = g2 * (1 - a2) / a2
        g1 = float(np.clip(g1, 1e-8, 1e12))
        if track is not None: track.append(float(np.linalg.norm(x2)))
        if not np.isfinite(x2).all(): return np.zeros(0)
    return np.sort(_peaks(np.abs(x2), int(N), hw).astype(np.float64)) / ffr


def cluster_merge(st, ext, amps_at, cnt_frames, eff, n_hat, radius_s):
    """[C138 2026-08-09] Cluster-merge of struct's OWN picks at radius beta*tol. st = struct_decode's
    n_hat picks (anchors); ext = struct_decode at an extended budget (prefix-stable, so st is exactly
    its first picks); amps_at(times)->matched-filter response; cnt_frames = per-frame COUNTS.
    beta->0 == struct BY CONSTRUCTION. Replicated 111 cells (4 ind x 5 folds x 2 bands): tol50
    +0.060 vs struct 111/111 positive; tol25 cm0.7 dominates pool8 on F1 AND jitter. See FANOUT_RECIPE C138."""
    import numpy as _np
    st = _np.asarray(st, _np.float64); ext = _np.asarray(ext, _np.float64)
    if len(st) == 0 or radius_s <= 0:
        return st
    def _k(lo, hi):
        f0, f1 = int(lo * eff), int(hi * eff) + 1
        return int(round(float(cnt_frames[max(0, f0):min(len(cnt_frames), f1)].sum())))
    aw = amps_at(st)
    cl = [[[float(st[0])], [float(aw[0])]]]
    for t, a in zip(st[1:], aw[1:]):
        c = cl[-1]
        if t - c[0][-1] <= radius_s and _k(c[0][0], t) < 2:
            c[0].append(float(t)); c[1].append(float(a))
        else:
            cl.append([[float(t)], [float(a)]])
    freed = n_hat - len(cl)
    mem = ext[~_np.isin(ext, st)]
    if len(mem):
        mw = amps_at(mem)
        cent = _np.array([_np.sum(_np.asarray(c[0]) * _np.asarray(c[1])) / max(_np.sum(c[1]), 1e-12) for c in cl])
        for i in _np.argsort(mw)[::-1]:
            d = _np.abs(cent - mem[i]); j = int(_np.argmin(d))
            if d[j] <= radius_s:
                cl[j][0].append(float(mem[i])); cl[j][1].append(float(mw[i]))
            elif freed > 0:
                cl.append([[float(mem[i])], [float(mw[i])]]); freed -= 1
                cent = _np.append(cent, mem[i])
    out = [float(_np.sum(_np.asarray(c[0]) * _np.asarray(c[1])) / max(_np.sum(c[1]), 1e-12)) for c in cl]
    return _np.sort(_np.asarray(out, _np.float64))


# ==================================================================================================================
#
# Why it exists (decoder_frame_budget, 12 rounds on 120 held-out 8f/6f/6s records + 118 untouched confirmation records):
# the greedy timing pursuit spends the 30-s budget on the binary head, which is FLAT inside a frame, so a confident
# frame receives a pick every ~9 ms (11 at 10 Hz) and weak count-head events receive none.  Counting per EVENT from
# the count head and placing inside frames fixes that: vs the timing decoder at gain 1.0, F1 +0.12 at 25 ms,
# +0.23 at 100 ms, +0.26 at one frame, +0.26 on per-frame counts, equal at 10 ms; vs the retired `msadapt` detection
# route +0.02 / +0.04 / +0.09 / +0.11.  Timing precision (jitter, F1 at 5 ms) is WORSE than the timing pursuit
# (mean |err| 5-12 vs 4-5 ms at 25 ms), so objective="timing" keeps `struct_decode`.
#
# Algorithm (gain multiplies the count head first; nothing else changes with gain):
#   1. frame mass lam_k = sum of the count head over the frame; mean prob p_k of the binary head over the frame.
#   2. EVENTS = runs of consecutive frames with lam_k > EVC_TAU0 * min(1, 10/fps), cut at 30-s segment edges.
#   3. picks per event n_e = floor(gain * mass_e + 1 - EVC_THR)  (an event needs >= 0.4 expected spikes for its first
#      pick; no largest-remainder top-up across events -- that top-up was measured to add false positives).
#   4. n_e is split over the event's frames by largest remainder of lam_k * max(p_k, 1e-6).
#   5. placement inside frame k: the binary head over [k*M - reach, (k+1)*M + reach], reach = EVC_REACH pooled bins,
#      pooled to bins of bin_ms_for(tol_ms, fps) (= min(tol, max(0.7 tol, frame/8)); default tolerance 25 ms);
#      greedy matching pursuit on the pooled vector with the msadapt template (sigma 0.5 bin, +-2 bins), then the
#      fine position = argmax of the 3 ms matched filter inside the chosen bin.  Frames are placed in ascending
#      order: fine bins within half a pooled bin of a pick already made by an earlier frame are zeroed first, so two
#      frames cannot claim one boundary peak.  A frame whose allocation exceeds its number of pooled bins falls back
#      to the fine 3 ms pursuit over its window.
# Implementation: device-agnostic torch (numpy arrays or CPU/CUDA tensors in, sorted float64 numpy seconds out).
# Frames that cannot interact (pick-frames more than two frames apart) are placed in parallel; frames inside a
# chain of near-neighbours are placed in chain order, which reproduces the sequential reference exactly.  The
# pooled pursuit runs in float64; the fine matched filter in float32 (its argmax inside one bin only).
# ==================================================================================================================
EVC_TAU0 = 0.15      # event threshold on the per-frame count mass at <= 10 Hz; scaled by min(1, 10/fps) above
EVC_THR = 0.4        # minimum expected spikes for an event's first pick: n_e = floor(mass + 1 - EVC_THR)
EVC_REACH = 0.5      # placement window reaches this many pooled bins beyond each frame edge
EVC_SIG_MS = 3.0     # fine matched filter (ms) for the position inside the chosen pooled bin
EVC_SEG_S = 30.0     # events are cut at these segment edges (same segmentation as the timing decoder)
EVC_BLOCK = 4096     # frames per placement block (bounds the device memory of the fine matched filter)
EVC_ALLOC_TOL_MS = 25.0   # the allocation super-frame is keyed to this tolerance's pooling bin, whatever tolerance is decoded
EVC_SUPERFRAME = os.environ.get("EVC_SUPERFRAME", "1").strip().lower() not in ("0", "false", "no", "off")
EVC_SPREAD = os.environ.get("EVC_SPREAD", "lr").strip().lower() or "lr"
EVC_SPREAD_MIN_FPS = float(os.environ.get("EVC_SPREAD_MIN_FPS", "30"))


def evc_superframe(eff):
    """Acquisition frames per allocation super-frame at frame rate `eff`: the smallest s with s * frame >=
    bin_ms_for(EVC_ALLOC_TOL_MS, eff); 1 when the frame is already at least the bin (<= ~57 Hz) or the rule is off."""
    import math
    from route_readout import bin_ms_for
    if not EVC_SUPERFRAME:
        return 1
    frame_ms = 1000.0 / float(eff); bin_ms = float(bin_ms_for(EVC_ALLOC_TOL_MS, float(eff)))
    return int(math.ceil(bin_ms / frame_ms - 1e-9)) if frame_ms < bin_ms else 1


def _evc_box3(x, w):
    """Three-pass box mean of half-width w (~Gaussian of sigma ~ w bins) via float64 cumulative sums: O(N) whatever the
    width.  Used for the fine position inside the chosen pooled bin; a true Gaussian conv1d cost 4e9-3e10 MACs per
    decode at 60-160 Hz on the m512 grid and pushed the CPU benchmark evals past their 8 h wall (2026-09-13)."""
    import torch
    import torch.nn.functional as F
    if w < 1:
        return x.to(torch.float64)
    y = x.to(torch.float64); k = 2 * w + 1
    for _ in range(3):
        c = torch.cumsum(F.pad(y, (w + 1, w)), 0)
        y = (c[k:] - c[:-k]) / k
    return y


def _evc_mp_fine(x, g, hw, n):
    """torch: greedy MP of `n` picks on one fine window x with Gaussian template g (half-width hw); pick indices."""
    import torch
    import torch.nn.functional as F
    work = F.conv1d(x.view(1, 1, -1).to(torch.float64), g.view(1, 1, -1).to(torch.float64), padding=hw).view(-1)
    L = int(work.numel()); picks = []
    for _ in range(int(n)):
        i = int(torch.argmax(work).item())
        if float(work[i]) <= 0.0:
            break
        picks.append(i)
        lo = max(0, i - hw); hi = min(L, i + hw + 1)
        work[lo:hi] -= work[i] * g[hw - (i - lo): hw + (hi - i)].to(torch.float64)
    return torch.tensor(picks, dtype=torch.long, device=x.device)


def evcount_decode(lam, pf, m, eff, tol_ms=25.0, gain=1.0, bin_ms=None, seg_s=EVC_SEG_S, sig_ms=EVC_SIG_MS,
                   device=None, _superframe=True, pfa=None, a_lam=1.0, a_p=1.0, split=None, _split_on=None):
    """Event-count detection decoder (see the block comment above).  lam / pf: the count / binary fine streams
    (n_frames * m,), numpy or torch (any device); m: sub-bins per frame; eff: frame rate (Hz).  Returns sorted
    spike times in seconds (numpy float64).  `_superframe` is internal: the super-frame recursion below clears it.

    [2026-09-20] Two OPTIONAL knobs, both inert at their defaults so the deployed column stays byte-identical
    (see `evcount_decode2` and route_readout's EVC_COMBINED arm):
      pfa   -- the stream whose per-frame mean sets the within-event split weight p_k. None (default) = `pf`,
               i.e. placement and allocation read the SAME stream, exactly as before. Passing the soft head here
               while `pf` carries the combined rate DECOUPLES the two: placement gets the count-scaled frame
               edges, the split keeps the deployed soft-head weighting.
      a_lam,
      a_p   -- exponents of the split weight, wa = la**a_lam * p_k**a_p. (1, 1) = deployed, and that branch keeps
               the original expression verbatim rather than raising to the power 1. (2, 0) is the lam^2 split,
               which is what a combined-rate `pf` induces implicitly (its frame mean is lam_k/M)."""
    import torch
    import torch.nn.functional as F
    from route_readout import bin_ms_for
    dev = device if device is not None else (lam.device if hasattr(lam, "device") else "cpu")
    lam_t = torch.as_tensor(lam, dtype=torch.float64, device=dev).reshape(-1)
    pf_t = torch.as_tensor(pf, dtype=torch.float64, device=dev).reshape(-1)
    pfa_t = pf_t if pfa is None else torch.as_tensor(pfa, dtype=torch.float64, device=dev).reshape(-1)
    M = int(m); eff = float(eff); ffr = eff * M
    # [C161] split mode, decided ONCE at the acquisition rate (the super-frame recursion below passes it down)
    if _split_on is None:
        _mode = (split or EVC_SPREAD)
        _split_on = _mode if (_mode in ("sys", "sys_long") and eff >= EVC_SPREAD_MIN_FPS) else "lr"
    s = evc_superframe(eff) if _superframe else 1
    if s > 1:
        if bin_ms is None:
            bin_ms = bin_ms_for(tol_ms, eff)
        L0 = int(min(lam_t.numel(), pf_t.numel())); Ls = -(-L0 // (M * s)) * (M * s)
        lam_t = F.pad(lam_t[:L0], (0, Ls - L0))
        # pad the allocation stream the same way; when it IS pf (the default) keep them the same object, and pass
        # pfa=None on down, so the default recursion is the pre-2026-09-20 call verbatim.
        pf_pad = F.pad(pf_t[:L0], (0, Ls - L0))
        pfa_t = pf_pad if pfa is None else F.pad(pfa_t[:L0], (0, Ls - L0))
        pf_t = pf_pad
        return evcount_decode(lam_t, pf_t, M * s, eff / s, tol_ms=tol_ms, gain=gain, bin_ms=bin_ms, seg_s=seg_s,
                              sig_ms=sig_ms, device=dev, _superframe=False,
                              pfa=(None if pfa is None else pfa_t), a_lam=a_lam, a_p=a_p, _split_on=_split_on)
    nf = int(min(lam_t.numel(), pf_t.numel()) // M)
    empty = np.zeros(0, np.float64)
    if nf < 1:
        return empty
    L = nf * M; lam_t = lam_t[:L]; pf_t = pf_t[:L]; pfa_t = pfa_t[:L]
    lam_raw = lam_t.view(nf, M).sum(1); lam_k = float(gain) * lam_raw; p_k = pfa_t.view(nf, M).mean(1)
    # ---- 2. events ----------------------------------------------------------------------------------------------
    tau = EVC_TAU0 * min(1.0, 10.0 / eff); seg_frames = max(1, int(round(seg_s * eff)))
    idx = torch.arange(nf, device=dev)
    active = lam_raw > tau
    prev = torch.zeros_like(active); prev[1:] = active[:-1]
    starts = active & ((~prev) | (idx % seg_frames == 0))
    n_ev = int(starts.sum().item())
    if n_ev == 0:
        return empty
    eid = torch.cumsum(starts.long(), 0) - 1
    a_idx = idx[active]; e = eid[active]; la = lam_k[active]
    if a_lam == 1.0 and a_p == 1.0:
        wa = la * p_k[active].clamp_min(1e-6)                              # deployed split, expression unchanged
    else:
        wa = (la ** a_lam) * (p_k[active].clamp_min(1e-6) ** a_p)
    # ---- 3. picks per event --------------------------------------------------------------------------------------
    mass = torch.zeros(n_ev, dtype=torch.float64, device=dev).index_add_(0, e, la)
    n_e = torch.floor(mass + (1.0 - EVC_THR)).long()
    # ---- 4. split over the event's frames (largest remainder of lam*p) ---------------------------------------------
    wsum = torch.zeros(n_ev, dtype=torch.float64, device=dev).index_add_(0, e, wa)
    q = n_e[e].to(torch.float64) * wa / wsum[e].clamp_min(1e-12)
    nfl = torch.floor(q); frac = q - nfl
    rem = n_e - torch.zeros(n_ev, dtype=torch.float64, device=dev).index_add_(0, e, nfl).long()
    key = e.to(torch.float64) + (1.0 - frac) * 0.999                    # within an event: descending fraction first
    order = torch.argsort(key); es = e[order]
    first = torch.full((n_ev,), int(order.numel()), dtype=torch.long, device=dev)
    first = first.scatter_reduce(0, es, torch.arange(int(order.numel()), device=dev), reduce="amin")
    posin = torch.arange(int(order.numel()), device=dev) - first[es]
    n_k = nfl.long(); n_k[order] = n_k[order] + (posin < rem[es]).long()
    if _split_on in ("sys", "sys_long"):
        # [C161] systematic split: active frames are in ascending frame order and grouped by event, so a running sum
        # minus the sum before the event's first frame is the within-event cumulative weight; frame k receives the number
        # of quantile points (j+1/2)/n_e that fall in its share (u_lo, u_hi] of the event's normalised weight.
        cw = torch.cumsum(wa, 0)
        first_a = torch.full((n_ev,), int(e.numel()), dtype=torch.long, device=dev)
        first_a = first_a.scatter_reduce(0, e, torch.arange(int(e.numel()), device=dev), reduce="amin")
        base = cw[first_a] - wa[first_a]
        den = wsum[e].clamp_min(1e-12); ne = n_e[e].to(torch.float64)
        u_hi = (cw - base[e]) / den; u_lo = (cw - wa - base[e]) / den
        n_sys = (torch.floor(ne * u_hi + 0.5) - torch.floor(ne * u_lo + 0.5)).clamp_min(0).long()
        if _split_on == "sys":
            n_k = n_sys
        else:
            len_e = torch.zeros(n_ev, dtype=torch.long, device=dev).index_add_(0, e, torch.ones_like(e))
            long_e = len_e >= torch.clamp(4 * n_e, min=8)
            n_k = torch.where(long_e[e], n_sys, n_k)
    has = n_k >= 1
    if not bool(has.any()):
        return empty
    ks = a_idx[has]; nks = n_k[has]                                       # ascending frame index
    # ---- 5. placement --------------------------------------------------------------------------------------------
    if bin_ms is None:
        bin_ms = bin_ms_for(tol_ms, eff)
    kk = max(1, min(M, int(round(float(bin_ms) / 1000.0 * ffr))))
    ext = int(round(EVC_REACH * kk)); Wf = M + 2 * ext; nb = -(-Wf // kk); Wp = nb * kk
    sig = max(sig_ms / 1000.0 * ffr, 0.5); hw = max(1, int(round(3 * sig)))
    g = torch.exp(-0.5 * (torch.arange(-hw, hw + 1, dtype=torch.float64, device=dev) / sig) ** 2)   # fallback MP only
    w_box = int(round(sig)) if sig >= 1.0 else 0                                                     # box3 ~ Gaussian(sigma)
    t5 = torch.exp(-0.5 * (torch.arange(-2, 3, dtype=torch.float64, device=dev) / 0.5) ** 2)   # msadapt template
    halo = ext + hw; r_mask = kk // 2
    # chains: pick-frames <= 2 frames apart can interact (window reach + mask dilation); place chain members in order
    gap = torch.ones_like(ks); gap[1:] = ks[1:] - ks[:-1]
    chain_start = gap > 2; chain_start[0] = True; chain = torch.cumsum(chain_start.long(), 0) - 1
    cfirst = torch.full((int(chain.max().item()) + 1,), int(ks.numel()), dtype=torch.long, device=dev)
    cfirst = cfirst.scatter_reduce(0, chain, torch.arange(int(ks.numel()), device=dev), reduce="amin")
    step = torch.arange(int(ks.numel()), device=dev) - cfirst[chain]
    PADT = ext + r_mask + 1                                               # taken mask padded once; picks written at +PADT
    taken = torch.zeros(L + 2 * PADT, dtype=torch.bool, device=dev)
    out_pos = []; ar_w = torch.arange(Wf, device=dev); ar_k = torch.arange(kk, device=dev); ar_m = torch.arange(Wf + 2 * r_mask, device=dev)
    blk_of = ks // EVC_BLOCK
    for b in torch.unique(blk_of).tolist():
        inb = blk_of == b
        f0 = b * EVC_BLOCK; f1 = min(nf, f0 + EVC_BLOCK)
        s0 = f0 * M - halo; s1 = f1 * M + halo
        pad_l = max(0, -s0); pad_r = max(0, s1 - L); lo_s = max(0, s0); hi_s = min(L, s1)
        seg = pf_t[lo_s:hi_s]
        if pad_l or pad_r:
            seg = F.pad(seg, (pad_l, pad_r))
        work = _evc_box3(seg, w_box)                                                   # O(N) smoothed stream for the in-bin position
        for s in torch.unique(step[inb]).tolist():
            sel = inb & (step == s); kb = ks[sel]; nbk = nks[sel]; Fb = int(kb.numel())
            gidx = ((kb - f0) * M + hw)[:, None] + ar_w[None, :]                   # (Fb, Wf) local fine indices
            # fine bins within r_mask of an earlier frame's pick are zeroed: gather the taken mask around each window
            tkw = taken[(kb * M - ext - r_mask + PADT)[:, None] + ar_m[None, :]]     # (Fb, Wf + 2 r_mask)
            if s > 0 and bool(tkw.any()):
                if r_mask > 0:
                    dil = F.max_pool1d(tkw.to(torch.float32).view(Fb, 1, -1), 2 * r_mask + 1, stride=1, padding=r_mask).view(Fb, -1)[:, r_mask:r_mask + Wf] > 0
                else:
                    dil = tkw
                win = seg[gidx] * (~dil).to(seg.dtype); winf = torch.where(dil, torch.full((Fb, Wf), -1e30, dtype=work.dtype, device=dev), work[gidx])
            else:
                win = seg[gidx]; winf = work[gidx]
            if Wp > Wf:
                win = F.pad(win, (0, Wp - Wf)); winf = F.pad(winf, (0, Wp - Wf), value=-1e30)
            pooled = win.view(Fb, nb, kk).sum(2)
            fb = nbk > nb                                                              # fallback: more picks than bins
            rows = torch.arange(Fb, device=dev)
            wp = F.conv1d(pooled.view(Fb, 1, nb), t5.view(1, 1, -1), padding=2).view(Fb, nb)
            for t in range(int(nbk.max().item())):
                alive = (nbk > t) & (~fb)
                if not bool(alive.any()):
                    break
                j = wp.argmax(1); peak = wp[rows, j]; valid = alive & (peak > 0)
                if bool(valid.any()):
                    lo = j * kk
                    sl = winf[rows[valid][:, None], (lo[valid][:, None] + ar_k[None, :]).clamp(max=Wp - 1)]
                    pos = lo[valid] + sl.argmax(1)
                    gpos = (kb[valid] * M - ext + pos).clamp(0, L - 1)
                    out_pos.append(gpos); taken[gpos + PADT] = True
                for d in range(-2, 3):
                    jj = j + d; ok = valid & (jj >= 0) & (jj < nb)
                    if bool(ok.any()):
                        wp[rows[ok], jj[ok]] -= peak[ok] * t5[d + 2]
            for i in torch.nonzero(fb).view(-1).tolist():
                picks = _evc_mp_fine(win[i, :Wf], g, hw, int(nbk[i]))
                if picks.numel():
                    gpos = (int(kb[i]) * M - ext + picks).clamp(0, L - 1); out_pos.append(gpos); taken[gpos + PADT] = True
    if not out_pos:
        return empty
    pos_all = torch.cat(out_pos).to(torch.float64)
    return np.sort((pos_all / ffr).cpu().numpy())


# ------------------------------------------------------------------------------------------------------------------
def combined_rate(cnt, pf, m, gamma=1.0):
    """Per-frame count-head mass x the soft head's within-frame profile, renormalised per frame.

    Frame sums equal the count head's EXACTLY by construction, so every frame-level and >= frame-window metric is
    unchanged; only the sub-frame shape differs. `gamma` sharpens the profile (gamma 4 is the shipped rate_fine
    recipe; gamma 1 is what the evcount placement arm uses -- inside a frame it is then the soft head times a
    scalar, so only the count-scaled frame EDGES of the placement window move). Frames whose soft mass is 0 get a
    uniform profile, matching t_gt_evc_decouple.norm_prof."""
    M = int(m)
    C = np.asarray(cnt, np.float64).reshape(-1, M)
    P = np.asarray(pf, np.float64).reshape(-1, M)
    Q = P if float(gamma) == 1.0 else P ** float(gamma)
    s = Q.sum(1, keepdims=True)
    prof = np.where(s > 0, Q / np.maximum(s, 1e-300), 1.0 / M)
    return (C.sum(1, keepdims=True) * prof).ravel()


def evcount_decode2(lam, pf, m, eff, pfa=None, a_lam=1.0, a_p=1.0, **kw):
    """`evcount_decode` with the allocation stream (`pfa`) and the split exponents exposed.

    Equivalences validated 45/45 by the source-transformed reference (t_gt_evc_decouple.py) and re-checked here by
    t_evc_combined_validate.py:
        evcount_decode2(lam, P,    pfa=P, a_lam=1, a_p=1) == evcount_decode(lam, P)      # deployed
        evcount_decode2(lam, prod, pfa=P, a_lam=2, a_p=0) == evcount_decode(lam, prod)   # combined-stream arm
    The second holds because a combined-rate stream's per-frame mean IS lam_k/M, so the deployed la*p_k weight
    already becomes a lam^2 split when `prod` is passed as the single stream. Splitting the arguments lets the
    placement edges and the split exponent be chosen INDEPENDENTLY, which is the whole point: the 50/100 ms cost
    of the combined stream is entirely the lam^2 split, not the placement."""
    return evcount_decode(lam, pf, m, eff, pfa=pfa, a_lam=a_lam, a_p=a_p, **kw)
