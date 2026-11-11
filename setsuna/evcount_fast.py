"""Event-count spike decoder, fast path.

The same algorithm as ``_vendor.decode_readout.evcount_decode`` (the reference): steps 1-4 -- per-frame
count mass and mean probability, events, picks per event, the largest-remainder split over frames --
run unchanged, in torch on whatever device the streams are on. Step 5, the placement inside each pick
frame, is where the reference spends its time: it smooths the whole binary stream block by block on
the device (three box passes over 2 M-element blocks, ~40 blocks and ~800 kernel launches per ROI on a
40-minute 60 Hz session) and then walks the blocks and chain steps with dozens of small tensor
operations and host synchronisations each, ~140-220 ms per ROI. Here only the pick frames' windows of
the binary stream are gathered (with the context the three box passes need), copied to the host in one
transfer (~1-2 MB per ROI), and everything else -- the box smoothing, the pooled matched filter, the
greedy picks, the fine position, the "taken" mask that stops two frames claiming one boundary peak,
the fine fallback pursuit when a frame's allocation exceeds its pooled bins -- runs in numba, visiting
the frames in ascending order, which is the order the reference reproduces.

What can differ from the reference: float64 sums are accumulated in a different order (the box means,
the 512-term pooled bins, the 5-tap and the fallback matched filters), so a fine position can move by
one bin where two candidates tie to ~1e-11 relative. The reference has that much noise itself: its box
smoothing is a difference of float64 prefix sums over 2 M elements computed by a CUDA scan whose
summation order changes call to call (30 of 30 repeated calls differ, max 2.5e-12), so two runs of the
reference on one stream can already disagree by one fine bin (0.03 ms at 60 Hz) at a near-tie.
Measured against it on the bench session: same spike counts on every ROI, positions equal except such
single-bin moves (see the commit). Frame-level inputs may be supplied directly (``lam_frame``,
``p_frame``) so the pooled forward can feed the decoder without a full-resolution count stream.

``JSCOPE_SETSUNA_EVC_HOST_SMOOTH=0`` keeps the box smoothing on the device (the reference's own per-block
ops, windows gathered afterwards) -- the previous fast path, ~4x slower per ROI.

With frame-level inputs on CUDA (the pooled forward's), steps 2-4 run on the host too (_events_split, after
one copy of the per-frame arrays) and the pick frames' windows come from one gather kernel: ~50 small device
ops and ~10 host synchronisations per ROI became ~8 and 2 -- in the SETSUNA loop each of those waited behind
the next ROI's forward. The same arithmetic in the same order, except that the per-event sums run in frame
order where torch's index_add_ on CUDA adds atomically (in no fixed order), and ties in the remainder order
keep frame order (a stable sort) where torch's argsort leaves them unspecified -- measured on the bench
session: the same spikes. ``JSCOPE_SETSUNA_EVC_HOST_STEPS=0`` keeps steps 2-4 in torch.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from setsuna._support._openmp import one_launch_at_a_time

try:  # numba is optional: without it decode() raises and the caller keeps the reference
    import numba as _nb
except Exception:  # pragma: no cover
    _nb = None

HOST_SMOOTH = os.environ.get("JSCOPE_SETSUNA_EVC_HOST_SMOOTH", "1").strip().lower() not in ("0", "false", "no", "off")
HOST_STEPS = os.environ.get("JSCOPE_SETSUNA_EVC_HOST_STEPS", "1").strip().lower() not in ("0", "false", "no", "off")

# the pick frames' windows of the binary stream, zero outside [0, Ld): out[i, j] = pf[ks[i] * M - off0 + j]
GATHER_SRC = r"""
extern "C" __global__ void evc_gather(const float* __restrict__ pf, const long long* __restrict__ ks, long long M,
                                      long long off0, long long Ld, int Wc, float* __restrict__ out)
{
    const int i = blockIdx.y;
    const long long base = ks[i] * M - off0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < Wc; j += gridDim.x * blockDim.x) {
        const long long g = base + j;
        out[(size_t)i * Wc + j] = (g >= 0 && g < Ld) ? pf[g] : 0.0f;
    }
}
"""
_GATHER: dict = {}
_PINNED: dict = {}
_PIN_LOCK = threading.Lock()


def available() -> bool:
    return _nb is not None


if _nb is not None:
    @one_launch_at_a_time
    @_nb.njit(cache=True, nogil=True, parallel=True)
    def _smooth_windows(ctx, lo_ok, hi_ok, w, Wf, out_w, out_wf):
        """The reference's three-pass box mean (half-width w, zero padding at the edges of the block it
        smooths), evaluated only on the pick frames' windows. ctx: (F, Wc) raw binary stream around each
        window, Wc = Wf + 6*w, zero beyond the ends of the stream; lo_ok/hi_ok: per frame, the range of
        context columns inside the block's padded array (everything outside is zero for every pass, as
        the reference's padding makes it). Writes the raw window (columns 3w..3w+Wf) to out_w and its
        smoothed version to out_wf, both float64. Frames are independent: one thread per frame."""
        F_ = ctx.shape[0]
        Wc = ctx.shape[1]
        k = 2 * w + 1
        inv_k = 1.0 / k
        for i in _nb.prange(F_):
            a = np.empty(Wc, np.float64)
            b = np.empty(Wc, np.float64)
            c = np.empty(Wc + 1, np.float64)
            lo = lo_ok[i]
            hi = hi_ok[i]
            for j in range(Wc):
                if j >= lo and j < hi:
                    a[j] = ctx[i, j]
                else:
                    a[j] = 0.0
            for j in range(Wf):
                out_w[i, j] = a[3 * w + j]
            if w < 1:
                for j in range(Wf):
                    out_wf[i, j] = a[3 * w + j]
                continue
            for _p in range(3):
                c[0] = 0.0
                for j in range(Wc):
                    c[j + 1] = c[j] + a[j]
                for j in range(Wc):
                    if j < lo or j >= hi:
                        b[j] = 0.0
                    else:
                        jl = j - w
                        if jl < 0:
                            jl = 0
                        jh = j + w + 1
                        if jh > Wc:
                            jh = Wc
                        b[j] = (c[jh] - c[jl]) * inv_k
                t = a
                a = b
                b = t
            for j in range(Wf):
                out_wf[i, j] = a[3 * w + j]

    @_nb.njit(cache=True, nogil=True)
    def _place(win, winf, kb, nbk, M, ext, kk, nb, Wp, r_mask, t5, g, hw, L):
        """Sequential placement over pick frames in ascending order. win/winf: (F, Wf) float64 windows of
        the binary stream and of its box-smoothed version, one per pick frame (Wf = M + 2*ext);
        kb: frame indices; nbk: picks per frame. Returns global fine-bin positions (int64)."""
        F_ = win.shape[0]
        Wf = win.shape[1]
        total = 0
        for i in range(F_):
            total += nbk[i]
        out = np.empty(total, np.int64)
        out_frame = np.empty(total, np.int64)
        n_out = 0
        w = np.empty(Wp, np.float64)
        wf = np.empty(Wp, np.float64)
        pooled = np.empty(nb, np.float64)
        wp = np.empty(nb, np.float64)
        work = np.empty(Wf, np.float64)
        for i in range(F_):
            k = kb[i]
            n_pick = nbk[i]
            base = k * M - ext
            for b in range(Wf):
                w[b] = win[i, b]
                wf[b] = winf[i, b]
            for b in range(Wf, Wp):
                w[b] = 0.0
                wf[b] = -1e30
            # the "taken" mask: fine bins within r_mask of a pick made by an earlier frame are zeroed
            # (frames more than two apart cannot interact; scan the recent picks only)
            j = n_out - 1
            while j >= 0 and out_frame[j] >= k - 2:
                p = out[j]
                lo = p - r_mask - base
                hi = p + r_mask - base
                if lo < 0:
                    lo = 0
                if hi > Wf - 1:
                    hi = Wf - 1
                for b in range(lo, hi + 1):
                    w[b] = 0.0
                    wf[b] = -1e30
                j -= 1
            if n_pick > nb:
                # fallback: greedy pursuit on the fine window with the Gaussian template (float64)
                for b in range(Wf):
                    s = 0.0
                    for d in range(-hw, hw + 1):
                        bb = b + d
                        if bb >= 0 and bb < Wf:
                            s += w[bb] * g[d + hw]
                    work[b] = s
                for t in range(n_pick):
                    im = 0
                    vm = work[0]
                    for b in range(1, Wf):
                        if work[b] > vm:
                            vm = work[b]
                            im = b
                    if vm <= 0.0:
                        break
                    lo = im - hw
                    if lo < 0:
                        lo = 0
                    hi = im + hw + 1
                    if hi > Wf:
                        hi = Wf
                    for b in range(lo, hi):
                        work[b] -= vm * g[hw - (im - b)]
                    gpos = base + im
                    if gpos < 0:
                        gpos = 0
                    if gpos > L - 1:
                        gpos = L - 1
                    out[n_out] = gpos
                    out_frame[n_out] = k
                    n_out += 1
                continue
            # pooled bins and the 5-tap matched filter (cross-correlation, zero outside)
            for jb in range(nb):
                s = 0.0
                for b in range(jb * kk, (jb + 1) * kk):
                    s += w[b]
                pooled[jb] = s
            for jb in range(nb):
                s = 0.0
                for d in range(5):
                    jj = jb + d - 2
                    if jj >= 0 and jj < nb:
                        s += pooled[jj] * t5[d]
                wp[jb] = s
            for t in range(n_pick):
                jm = 0
                pm = wp[0]
                for jb in range(1, nb):
                    if wp[jb] > pm:
                        pm = wp[jb]
                        jm = jb
                if pm <= 0.0:
                    break
                lo = jm * kk
                bm = lo
                vm = wf[lo]
                for b in range(lo + 1, lo + kk):
                    if wf[b] > vm:
                        vm = wf[b]
                        bm = b
                gpos = base + bm
                if gpos < 0:
                    gpos = 0
                if gpos > L - 1:
                    gpos = L - 1
                out[n_out] = gpos
                out_frame[n_out] = k
                n_out += 1
                for d in range(-2, 3):
                    jj = jm + d
                    if jj >= 0 and jj < nb:
                        wp[jj] -= pm * t5[d + 2]
        return out[:n_out]

    @_nb.njit(cache=True, nogil=True)
    def _events_split(lam_raw, p_k, wa_all, use_wa, gain, tau, seg_frames, thr_c):
        """Steps 2-4 of _decode_impl's torch ops on host arrays (finite per-frame count mass lam_raw and mean
        probability p_k): events (runs of frames with lam_raw > tau, cut every seg_frames frames), picks per
        event floor(gain * mass + thr_c), and the largest-remainder split of each event's picks over its frames
        by weight gain * lam * max(p, 1e-6) (use_wa: wa_all[f] instead). The same float64 operations; the
        per-event sums in frame order, the remainder order a stable sort. Returns (pick frames, picks each)."""
        nf = lam_raw.shape[0]
        a_idx = np.empty(nf, np.int64)
        e = np.empty(nf, np.int64)
        la = np.empty(nf, np.float64)
        wa = np.empty(nf, np.float64)
        n_act = 0
        n_ev = 0
        prev = False
        for f in range(nf):
            act = lam_raw[f] > tau
            if act:
                if (not prev) or (f % seg_frames == 0):
                    n_ev += 1
                a_idx[n_act] = f
                e[n_act] = n_ev - 1
                lk = gain * lam_raw[f]
                la[n_act] = lk
                if use_wa:
                    wa[n_act] = wa_all[f]
                else:
                    p = p_k[f]
                    if p < 1e-6:
                        p = 1e-6
                    wa[n_act] = lk * p
                n_act += 1
            prev = act
        if n_ev == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        mass = np.zeros(n_ev, np.float64)
        wsum = np.zeros(n_ev, np.float64)
        for i in range(n_act):
            mass[e[i]] += la[i]
        n_e = np.empty(n_ev, np.int64)
        for k in range(n_ev):
            n_e[k] = np.int64(np.floor(mass[k] + thr_c))
        for i in range(n_act):
            wsum[e[i]] += wa[i]
        nfl = np.empty(n_act, np.float64)
        key = np.empty(n_act, np.float64)
        sfl = np.zeros(n_ev, np.float64)
        for i in range(n_act):
            ws = wsum[e[i]]
            if ws < 1e-12:
                ws = 1e-12
            q = np.float64(n_e[e[i]]) * wa[i] / ws
            fl = np.floor(q)
            nfl[i] = fl
            key[i] = np.float64(e[i]) + (1.0 - (q - fl)) * 0.999
            sfl[e[i]] += fl
        rem = np.empty(n_ev, np.int64)
        for k in range(n_ev):
            rem[k] = n_e[k] - np.int64(sfl[k])
        order = np.argsort(key, kind="mergesort")
        first = np.full(n_ev, n_act, np.int64)
        for pos in range(n_act):
            ev = e[order[pos]]
            if pos < first[ev]:
                first[ev] = pos
        n_k = np.empty(n_act, np.int64)
        for i in range(n_act):
            n_k[i] = np.int64(nfl[i])
        for pos in range(n_act):
            i = order[pos]
            if pos - first[e[i]] < rem[e[i]]:
                n_k[i] += 1
        n_has = 0
        for i in range(n_act):
            if n_k[i] >= 1:
                n_has += 1
        ks = np.empty(n_has, np.int64)
        nks = np.empty(n_has, np.int64)
        j = 0
        for i in range(n_act):
            if n_k[i] >= 1:
                ks[j] = a_idx[i]
                nks[j] = n_k[i]
                j += 1
        return ks, nks

    @one_launch_at_a_time
    @_nb.njit(cache=True, nogil=True, parallel=True)
    def _combined_ctx(ctx, ks, M, off0, Ld, M0, c_f, u_f):
        """[C163] The combined rate on the gathered windows: bin g inside the stream -> float64(pf[g]) *
        c_f[g // M0] + u_f[g // M0], zero outside (as the torch path's where / multiply / add). Windows are
        independent: one thread per window (2026-11-06: the largest part of a dense ROI's decode at <= 12 Hz)."""
        F_ = ctx.shape[0]
        Wc = ctx.shape[1]
        out = np.empty((F_, Wc), np.float64)
        for i in _nb.prange(F_):
            base = ks[i] * M - off0
            for j in range(Wc):
                g = base + j
                if g >= 0 and g < Ld:
                    fr = g // M0
                    out[i, j] = np.float64(ctx[i, j]) * c_f[fr] + u_f[fr]
                else:
                    out[i, j] = 0.0
        return out


_SIDE_STREAMS: dict = {}


def _side_stream(dev):
    """One extra CUDA stream per device for the decoders' device work (see decode), at the highest priority: its
    short kernels are scheduled as soon as the running forward frees a slot instead of after it, so the host, which
    waits for them, gets back to queueing the next ROI sooner (the SETSUNA stage with detection spikes 24.0 ->
    20.7 s on a 1450-ROI session, same spikes)."""
    import torch
    key = (dev.type, dev.index)
    s = _SIDE_STREAMS.get(key)
    if s is None:
        s = torch.cuda.Stream(device=dev, priority=torch.cuda.Stream.priority_range()[1])
        _SIDE_STREAMS[key] = s
    return s


def decode(lam, pf, m: int, eff: float, tol_ms: float = 25.0, gain: float = 1.0, bin_ms=None,
           seg_s=None, sig_ms=None, lam_frame=None, p_frame=None, ready_event=None,
           combined: bool = False, a_lam: float = 1.0, a_p: float = 1.0) -> np.ndarray:
    """evcount_decode with the fast placement. ``lam`` / ``pf``: the count / binary fine streams as torch
    tensors (any device); ``lam`` may be None when ``lam_frame`` (per-frame count mass, float64) and
    ``p_frame`` (per-frame mean probability, float64) are given. Returns sorted spike times in seconds.

    ``ready_event``: a CUDA event recorded after the streams were produced. The device part of the
    decode then runs on a side stream that waits for that event only -- not for whatever the caller
    queued on the main stream afterwards (the next ROI's forward), which it overlaps. The result is
    the same either way: the streams are complete when the event has fired, and the host copies
    synchronise the side stream.

    [C163 2026-09-26] ``combined`` / ``a_lam`` / ``a_p``: the EVC_COMBINED arm (route_readout.combined_applies:
    pyramidal cells at <= 12 Hz). Placement reads the combined rate -- each frame's soft-head profile scaled to the
    count head's frame mass (decode_readout.combined_rate, gamma 1) -- instead of the soft head, and the within-event
    split weight is mass**a_lam * p**a_p (route_readout.evc_split_exponents). The reference is
    _vendor.decode_readout.evcount_decode2(lam, combined_rate(lam, pf), pfa=pf, a_lam, a_p). The defaults are the
    deployed decoder, unchanged."""
    import torch
    pf_t = torch.as_tensor(pf).reshape(-1)
    dev = pf_t.device
    if ready_event is not None and dev.type == "cuda":
        side = _side_stream(dev)
        with torch.cuda.stream(side):
            side.wait_event(ready_event)
            return _decode_impl(lam, pf_t, m, eff, tol_ms, gain, bin_ms, seg_s, sig_ms, lam_frame, p_frame,
                                combined, a_lam, a_p)
    return _decode_impl(lam, pf_t, m, eff, tol_ms, gain, bin_ms, seg_s, sig_ms, lam_frame, p_frame,
                        combined, a_lam, a_p)


def _decode_impl(lam, pf_t, m, eff, tol_ms, gain, bin_ms, seg_s, sig_ms, lam_frame, p_frame,
                 combined=False, a_lam=1.0, a_p=1.0) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from ._vendor import decode_readout as _D
    from ._vendor.route_readout import bin_ms_for
    if _nb is None:
        raise RuntimeError("numba is not available")
    seg_s = _D.EVC_SEG_S if seg_s is None else float(seg_s)
    sig_ms = _D.EVC_SIG_MS if sig_ms is None else float(sig_ms)
    dev = pf_t.device
    M = int(m)
    eff = float(eff)
    ffr = eff * M
    if lam_frame is not None:
        lam_raw = torch.as_tensor(lam_frame, dtype=torch.float64, device=dev).reshape(-1)
        p_k = torch.as_tensor(p_frame, dtype=torch.float64, device=dev).reshape(-1)
        nf = int(min(lam_raw.numel(), p_k.numel(), pf_t.numel() // M))
        lam_raw = lam_raw[:nf]
        p_k = p_k[:nf]
    else:
        lam_t = torch.as_tensor(lam, dtype=torch.float64, device=dev).reshape(-1)
        nf = int(min(lam_t.numel(), pf_t.numel()) // M)
        if nf < 1:
            return np.zeros(0, np.float64)
        lam_raw = lam_t[: nf * M].view(nf, M).sum(1)
        p_k = pf_t[: nf * M].to(torch.float64).view(nf, M).mean(1)
    empty = np.zeros(0, np.float64)
    if nf < 1:
        return empty
    comb_scale = None
    if combined:
        # [C163] the placement stream is the combined rate, bin b of acquisition frame f = pf[b] * mass[f] / soft
        # mass[f] (a frame without soft mass: mass[f] / M everywhere), built at the acquisition rate BEFORE the
        # super-frame merge exactly as the reference builds it -- only the gathered windows are scaled, never the
        # whole stream.
        M0 = M
        smass = p_k * M0
        comb_scale = (M0, torch.where(smass > 0, lam_raw / smass.clamp_min(1e-300), torch.zeros_like(lam_raw)),
                      torch.where(smass > 0, torch.zeros_like(lam_raw), lam_raw / M0))
    s = _D.evc_superframe(eff)
    if s > 1:
        if bin_ms is None:
            bin_ms = bin_ms_for(tol_ms, eff)
        nfs = -(-nf // s)
        if nfs * s > nf:
            lam_raw = F.pad(lam_raw, (0, nfs * s - nf))
            p_k = F.pad(p_k, (0, nfs * s - nf))
        lam_raw = lam_raw.view(nfs, s).sum(1)
        p_k = p_k.view(nfs, s).mean(1)
        # the binary stream's zero padding up to nfs * s frames is not built (a copy of the whole stream, 0.6 ms of
        # GPU per ROI at 60 Hz): the reads below take zeros past its end, which is what the padding held
        M = M * s
        eff = eff / s
        nf = nfs
    L = nf * M
    pf_t = pf_t[:L]
    Ld = int(pf_t.numel())                                # the stream's own bins; [Ld, L) reads as zeros
    if HOST_STEPS and HOST_SMOOTH and lam_frame is not None and dev.type == "cuda" and pf_t.is_contiguous():
        got = _decode_host(lam_raw, p_k, pf_t, M, eff, ffr, nf, L, Ld, tol_ms, gain, bin_ms, seg_s, sig_ms,
                           comb_scale, a_lam, a_p)
        if got is not None:
            return got
    lam_k = float(gain) * lam_raw
    # ---- 2. events (as the reference) ----
    tau = _D.EVC_TAU0 * min(1.0, 10.0 / eff)
    seg_frames = max(1, int(round(seg_s * eff)))
    idx = torch.arange(nf, device=dev)
    active = lam_raw > tau
    prev = torch.zeros_like(active)
    prev[1:] = active[:-1]
    starts = active & ((~prev) | (idx % seg_frames == 0))
    n_ev = int(starts.sum().item())
    if n_ev == 0:
        return empty
    eid = torch.cumsum(starts.long(), 0) - 1
    a_idx = idx[active]
    e = eid[active]
    la = lam_k[active]
    if a_lam == 1.0 and a_p == 1.0:
        wa = la * p_k[active].clamp_min(1e-6)
    else:                                                  # [C163] the reference's two-exponent split, verbatim
        wa = (la ** a_lam) * (p_k[active].clamp_min(1e-6) ** a_p)
    # ---- 3. picks per event ----
    mass = torch.zeros(n_ev, dtype=torch.float64, device=dev).index_add_(0, e, la)
    n_e = torch.floor(mass + (1.0 - _D.EVC_THR)).long()
    # ---- 4. split over the event's frames (largest remainder of lam*p) ----
    wsum = torch.zeros(n_ev, dtype=torch.float64, device=dev).index_add_(0, e, wa)
    q = n_e[e].to(torch.float64) * wa / wsum[e].clamp_min(1e-12)
    nfl = torch.floor(q)
    frac = q - nfl
    rem = n_e - torch.zeros(n_ev, dtype=torch.float64, device=dev).index_add_(0, e, nfl).long()
    key = e.to(torch.float64) + (1.0 - frac) * 0.999
    order = torch.argsort(key)
    es = e[order]
    first = torch.full((n_ev,), int(order.numel()), dtype=torch.long, device=dev)
    first = first.scatter_reduce(0, es, torch.arange(int(order.numel()), device=dev), reduce="amin")
    posin = torch.arange(int(order.numel()), device=dev) - first[es]
    n_k = nfl.long()
    n_k[order] = n_k[order] + (posin < rem[es]).long()
    has = n_k >= 1
    if not bool(has.any()):
        return empty
    ks = a_idx[has]
    nks = n_k[has]
    # ---- 5. placement ----
    kk, ext, Wf, nb, Wp, hw, g, w_box, t5, halo, r_mask = _placement_params(M, ffr, eff, tol_ms, bin_ms, sig_ms)
    if HOST_SMOOTH:
        # the pick frames' windows with the context of the three box passes: one gather, one copy;
        # the smoothing runs on the host (numba), block-padded exactly as the reference pads
        Wc = Wf + 6 * w_box
        w_start = ks * M - ext - 3 * w_box
        gi = w_start[:, None] + torch.arange(Wc, device=dev)[None, :]
        inside = ((gi >= 0) & (gi < Ld)) if comb_scale is not None else None   # [C163] before gi is clamped in place
        ctx = torch.where((gi >= 0) & (gi < Ld), pf_t[gi.clamp_(0, Ld - 1)],
                          torch.zeros((), dtype=pf_t.dtype, device=dev))
        if comb_scale is not None:                         # [C163] the combined rate on the gathered windows only
            M0, c_f, u_f = comb_scale
            fr = gi // M0                                  # gi is clamped to the stream now
            ctx = torch.where(inside, ctx.to(torch.float64) * c_f[fr] + u_f[fr],
                              torch.zeros((), dtype=torch.float64, device=dev))
        blk = ks // _D.EVC_BLOCK
        s0 = blk * (_D.EVC_BLOCK * M) - halo
        s1 = torch.clamp((blk + 1) * _D.EVC_BLOCK, max=nf) * M + halo
        meta = torch.stack([ks, nks, (s0 - w_start).clamp_(0, Wc), (s1 - w_start).clamp_(0, Wc)]).cpu().numpy()
        ctx_h = ctx.cpu().numpy()
        kb_all = np.ascontiguousarray(meta[0]).astype(np.int64)
        nk_all = np.ascontiguousarray(meta[1]).astype(np.int64)
        win = np.empty((kb_all.size, Wf), np.float64)
        winf = np.empty((kb_all.size, Wf), np.float64)
        _smooth_windows(ctx_h, np.ascontiguousarray(meta[2]).astype(np.int64),
                        np.ascontiguousarray(meta[3]).astype(np.int64), w_box, Wf, win, winf)
    else:
        # device-side smoothing per block as the reference, windows gathered, host-side pursuit
        ar_w = torch.arange(Wf, device=dev)
        wins, winfs = [], []
        blk_of = ks // _D.EVC_BLOCK
        for b in torch.unique(blk_of).tolist():
            inb = blk_of == b
            kb = ks[inb]
            f0 = b * _D.EVC_BLOCK
            f1 = min(nf, f0 + _D.EVC_BLOCK)
            s0 = f0 * M - halo
            s1 = f1 * M + halo
            pad_l = max(0, -s0)
            pad_r = max(0, s1 - Ld)
            lo_s = max(0, s0)
            hi_s = min(Ld, s1)
            seg = pf_t[lo_s:hi_s].to(torch.float64)
            if comb_scale is not None:                     # [C163] the combined rate over this block
                M0, c_f, u_f = comb_scale
                fr = torch.arange(lo_s, hi_s, device=dev) // M0
                seg = seg * c_f[fr] + u_f[fr]
            if pad_l or pad_r:
                seg = F.pad(seg, (pad_l, pad_r))
            work = _D._evc_box3(seg, w_box)
            gidx = ((kb - f0) * M + hw)[:, None] + ar_w[None, :]
            wins.append(seg[gidx])
            winfs.append(work[gidx])
        win = np.ascontiguousarray(torch.cat(wins).cpu().numpy())
        winf = np.ascontiguousarray(torch.cat(winfs).cpu().numpy())
        # frames of every block arrive in ascending order (unique() sorts; ks is ascending within a block)
        kb_all = ks.cpu().numpy().astype(np.int64)
        nk_all = nks.cpu().numpy().astype(np.int64)
    pos = _place(win, winf, kb_all, nk_all, M, ext, kk, nb, Wp, r_mask, t5, g, hw, L)
    if pos.size == 0:
        return empty
    # the reference divides on the device, where torch turns `tensor / scalar` into a multiply by
    # the reciprocal; do the same so the seconds are bit-identical to it
    return np.sort(pos.astype(np.float64) * (1.0 / ffr))


def _placement_params(M, ffr, eff, tol_ms, bin_ms, sig_ms):
    """Step 5's geometry: (pooled bin kk, reach ext, window Wf, pooled bins nb, Wp = nb kk, template half-width
    hw, Gaussian template g, box half-width w_box, 5-tap filter t5, halo, taken-mask radius)."""
    from ._vendor import decode_readout as _D
    from ._vendor.route_readout import bin_ms_for
    if bin_ms is None:
        bin_ms = bin_ms_for(tol_ms, eff)
    kk = max(1, min(M, int(round(float(bin_ms) / 1000.0 * ffr))))
    ext = int(round(_D.EVC_REACH * kk))
    Wf = M + 2 * ext
    nb = -(-Wf // kk)
    Wp = nb * kk
    sig = max(sig_ms / 1000.0 * ffr, 0.5)
    hw = max(1, int(round(3 * sig)))
    g = np.exp(-0.5 * (np.arange(-hw, hw + 1, dtype=np.float64) / sig) ** 2)
    w_box = int(round(sig)) if sig >= 1.0 else 0
    t5 = np.exp(-0.5 * (np.arange(-2, 3, dtype=np.float64) / 0.5) ** 2)
    halo = ext + hw
    r_mask = kk // 2
    return kk, ext, Wf, nb, Wp, hw, g, w_box, t5, halo, r_mask


def _pinned(slot: str, n: int, dtype):
    """A pinned host buffer of at least n elements, reused per slot (callers hold _PIN_LOCK)."""
    import torch
    b = _PINNED.get((slot, dtype))
    if b is None or b.numel() < n:
        b = torch.empty(max(int(n), 1 << 16), dtype=dtype, pin_memory=True)
        _PINNED[(slot, dtype)] = b
    return b


def _gather_kernel(dev):
    """evc_gather on ``dev``; raises when it does not compile or its gpu_kernel_check verdict says no (the caller
    then takes the torch ops)."""
    from setsuna._support import gpu_kernel_check as _gkc
    from . import gpu_checks as _gc
    k = _GATHER.get((dev.type, dev.index))
    if k is None:
        from . import _cuda_rt
        k = _GATHER[(dev.type, dev.index)] = _cuda_rt.CudaKernel(GATHER_SRC, "evc_gather", device=dev)
    if not _gkc.use("evcount.gather", dev, _gc.check_evc_gather):
        raise RuntimeError("evc_gather: off on this GPU (gpu_kernel_check)")
    return k


def _decode_host(lam_raw, pk, pf_t, M, eff, ffr, nf, L, Ld, tol_ms, gain, bin_ms, seg_s, sig_ms, comb_scale,
                 a_lam, a_p):
    """_decode_impl from step 2 on with the per-frame arrays on the host (see the module docstring): one copy of
    them, _events_split, the pick frames' windows in one gather and one copy, then the placement as before.
    Returns the spike times, or None to leave the decode to the torch ops (non-finite per-frame values, or no
    gather kernel)."""
    import torch
    from ._vendor import decode_readout as _D
    dev = pf_t.device
    try:
        gk = _gather_kernel(dev)
    except Exception:
        return None
    use_wa = not (a_lam == 1.0 and a_p == 1.0)
    parts = [lam_raw.reshape(-1), pk.reshape(-1)]
    if use_wa:                   # [C163] the two-exponent split weight on the device, with torch's pow as before
        parts.append(((float(gain) * lam_raw) ** a_lam) * (pk.clamp_min(1e-6) ** a_p))
    if comb_scale is not None:   # [C163] the combined-rate factors per acquisition frame
        parts += [comb_scale[1].reshape(-1), comb_scale[2].reshape(-1)]
    flat = torch.cat([t_.to(torch.float64) for t_ in parts])
    with _PIN_LOCK:
        hb = _pinned("frames", flat.numel(), torch.float64)
        hb[:flat.numel()].copy_(flat, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        ev.synchronize()
        arr = hb[:flat.numel()].numpy().copy()
    lam_h, p_h = arr[:nf], arr[nf:2 * nf]
    if not (np.isfinite(lam_h).all() and np.isfinite(p_h).all()):
        return None
    wa_h = arr[2 * nf:3 * nf] if use_wa else np.zeros(0, np.float64)
    tau = _D.EVC_TAU0 * min(1.0, 10.0 / eff)
    seg_frames = max(1, int(round(seg_s * eff)))
    ks, nks = _events_split(lam_h, p_h, wa_h, use_wa, float(gain), float(tau), int(seg_frames),
                            1.0 - _D.EVC_THR)
    if ks.size == 0:
        return np.zeros(0, np.float64)
    kk, ext, Wf, nb, Wp, hw, g, w_box, t5, halo, r_mask = _placement_params(M, ffr, eff, tol_ms, bin_ms, sig_ms)
    Wc = Wf + 6 * w_box
    off0 = ext + 3 * w_box
    w_start = ks * M - off0
    blk = ks // _D.EVC_BLOCK
    s0 = blk * (_D.EVC_BLOCK * M) - halo
    s1 = np.minimum((blk + 1) * _D.EVC_BLOCK, nf) * M + halo
    lo_ok = np.clip(s0 - w_start, 0, Wc).astype(np.int64)
    hi_ok = np.clip(s1 - w_start, 0, Wc).astype(np.int64)
    F_ = int(ks.size)
    with _PIN_LOCK:
        kb = _pinned("ks", F_, torch.int64)
        kb[:F_].copy_(torch.from_numpy(ks))
        ks_dev = kb[:F_].to(dev, non_blocking=True)
        ctx = torch.empty((F_, Wc), dtype=torch.float32, device=dev)
        gk.launch(((Wc + 255) // 256, F_), (256,), 0,
                  (pf_t, ks_dev, (int(M), "q"), (int(off0), "q"), (int(Ld), "q"), (int(Wc), "i"), ctx))
        cb = _pinned("ctx", F_ * Wc, torch.float32)
        cb[:F_ * Wc].view(F_, Wc).copy_(ctx, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        ev.synchronize()
        ctx_h = cb[:F_ * Wc].numpy().reshape(F_, Wc).copy()
    if comb_scale is not None:
        n0 = int(comb_scale[1].numel())
        o = (3 if use_wa else 2) * nf
        ctx_h = _combined_ctx(ctx_h, ks, int(M), int(off0), int(Ld), int(comb_scale[0]), arr[o:o + n0],
                              arr[o + n0:o + 2 * n0])
    win = np.empty((F_, Wf), np.float64)
    winf = np.empty((F_, Wf), np.float64)
    _smooth_windows(ctx_h, lo_ok, hi_ok, w_box, Wf, win, winf)
    pos = _place(win, winf, ks, nks, M, ext, kk, nb, Wp, r_mask, t5, g, hw, L)
    if pos.size == 0:
        return np.zeros(0, np.float64)
    # the reference divides on the device, where torch turns `tensor / scalar` into a multiply by
    # the reciprocal; do the same so the seconds are bit-identical to it
    return np.sort(pos.astype(np.float64) * (1.0 / ffr))
