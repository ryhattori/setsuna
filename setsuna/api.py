"""SETSUNA's public interface: fluorescence -> dF/F -> spike rates and spike times.

    import setsuna
    dff = setsuna.dff_from_fluorescence(F, Fneu, fps=30.0)            # (n_neurons, n_frames) fractional dF/F
    out = setsuna.infer(dff, fps=30.0, indicator="jGCaMP8s", tolerance_ms=10)
    out["rate"]            # (n_neurons, n_frames) expected spikes per frame

    out = setsuna.infer(dff, fps=30.0, smooth_ms=25)                  # recommended for rate analyses:
    out["rate_smooth"]     # (n_neurons, n_bins) spikes/s on a fine time grid, Gaussian-smoothed (sigma 25 ms)
    out["t_smooth"]        # (n_bins,) bin centres, seconds from the first frame
    setsuna.smooth_rate(out["count"], fps=30.0, bins_per_frame=out["bins_per_frame"], neurons=[0, 5])   # from a
                           # sub-frame count stream (infer(..., bins_per_frame=16)), for any neurons and time window
    out["spike_times"]     # list of 1-D arrays, spike times in seconds from the first frame

Everything below calls the inference engine (engine.py) unchanged; see docs/USER_GUIDE.md.
"""
from __future__ import annotations

from pathlib import Path

import os

import numpy as np

__all__ = ["dff_from_fluorescence", "infer", "smooth_rate", "Session", "INDICATORS", "CELL_TYPES", "SMOOTH_MS"]

#: Indicator names accepted by infer() (display spelling -> the name the models use). Any other indicator: "other".
INDICATORS = {
    "jGCaMP8s": "GCaMP8s", "jGCaMP8m": "GCaMP8m", "jGCaMP8f": "GCaMP8f", "jGCaMP7f": "GCaMP7f",
    "GCaMP6s": "GCaMP6s", "GCaMP6f": "GCaMP6f", "jRGECO1a": "jGECO1a", "other": "other",
}
#: Recommended Gaussian sigma (ms) for ``rate_smooth`` / smooth_rate(): the same at every frame rate.
SMOOTH_MS = 25.0
#: infer(smooth_ms=...) refuses to build an in-memory ``rate_smooth`` larger than this (GiB) unless told otherwise; see infer().
MAX_SMOOTH_GB = 2.0
#: Cell types accepted by infer(). "unknown" = probably pyramidal; "other" = a type not listed.
CELL_TYPES = ("pyramidal", "PV_FS", "SST", "VIP", "spinal_excitatory", "spinal_inhibitory", "unknown", "other")


def _maximin(x: np.ndarray, sigma: float, win: int) -> np.ndarray:
    from scipy import ndimage as ndi
    win = int(max(1, win))
    if win % 2 == 0:
        win += 1
    s = ndi.gaussian_filter1d(x, float(sigma), axis=1, mode="nearest") if sigma > 0 else x
    low = ndi.minimum_filter1d(s, win, axis=1, mode="nearest")
    return ndi.maximum_filter1d(low, win, axis=1, mode="nearest").astype(np.float32, copy=False)


def dff_from_fluorescence(F, Fneu=None, fps: float = 0.0, neuropil_coefficient: float = 0.7,
                          window_s: float = 60.0, baseline_percentile: float = 20.0) -> np.ndarray:
    """Fractional dF/F for SETSUNA from raw somatic fluorescence ``F`` and neuropil ``Fneu`` (n_neurons x n_frames).

        dF/F = (Fc - B_c(t)) / max(B_raw(t), 1),   Fc = F - neuropil_coefficient * Fneu

    B is a maximin baseline (Gaussian smoothing over max(5 frames, 1.5 s), then a rolling minimum and a rolling maximum
    over ``window_s``), so it follows slow drift and stays under calcium transients whatever a neuron's firing. The
    neuropil-corrected trace is in the numerator and the raw baseline in the denominator, which stays positive.
    Recordings shorter than two windows use a fixed percentile baseline instead. Pass ``Fneu=None`` (or a coefficient
    of 0) to skip neuropil subtraction. If you already have dF/F as a fraction (0.5 = 50 %), use it directly."""
    F = np.atleast_2d(np.asarray(F, np.float32))
    if Fneu is not None and neuropil_coefficient:
        Fc = F - np.float32(neuropil_coefficient) * np.atleast_2d(np.asarray(Fneu, np.float32))
    else:
        Fc = F
    fps = float(fps or 0.0)
    win = int(round(window_s * fps)) if fps > 0 else 0
    if win >= 11 and F.shape[1] >= 2 * win:
        sigma = min(max(5.0, round(1.5 * fps)), max(1.0, win / 4.0))
        b_c, b_raw = _maximin(Fc, sigma, win), _maximin(F, sigma, win)
    else:
        from scipy import ndimage as ndi
        b_c = np.percentile(ndi.gaussian_filter1d(Fc, 5.0, axis=1, mode="nearest"), baseline_percentile, axis=1, keepdims=True)
        b_raw = np.percentile(ndi.gaussian_filter1d(F, 5.0, axis=1, mode="nearest"), baseline_percentile, axis=1, keepdims=True)
    dff = (Fc - b_c) / np.maximum(b_raw, np.float32(1.0))
    dff[~np.isfinite(dff)] = 0.0
    return dff.astype(np.float32, copy=False)


class Session:
    """Models for one frame rate, indicator and cell type, loaded once and applied to many neurons.

    ``model_dir``: folder holding the checkpoints (default: $SETSUNA_MODEL_DIR, else ~/.setsuna/models).
    ``device``: "auto" (CUDA if available), "cuda" or "cpu"."""

    def __init__(self, fps: float, indicator: str = "jGCaMP8s", cell_type: str = "pyramidal",
                 model_dir: str | Path | None = None, device: str = "auto", batch_windows: int = 512):
        from . import engine
        if indicator not in INDICATORS and indicator not in INDICATORS.values():
            raise ValueError(f"indicator {indicator!r}: use one of {sorted(INDICATORS)}")
        if cell_type not in CELL_TYPES:
            raise ValueError(f"cell_type {cell_type!r}: use one of {CELL_TYPES}")
        self._e = engine
        if model_dir is None and not os.environ.get(engine.MODEL_DIR_ENV, "").strip():
            from . import models
            model_dir = models.ensure_models()          # downloads this version's models on first use
        self.fps = float(fps)
        self.s = engine.SetsunaSession(self.fps, indicator=INDICATORS.get(indicator, indicator), cell_type=cell_type,
                                       model_dir=model_dir, device=device, batch_windows=batch_windows)

    def infer_one(self, dff_row, tolerance_ms: float | None = 10.0, count_gain: float = 1.0, fine: bool = False) -> dict:
        """One neuron: ``rate`` (expected spikes per frame), ``nu`` (noise level), ``spike_times`` (s) when
        ``tolerance_ms`` is given, and the 512-bin ``count`` and ``prob`` streams when ``fine``."""
        r = self.s.infer_roi(np.asarray(dff_row, np.float32))
        out = {"rate": np.asarray(r["rate"], np.float32), "nu": float(r["nu"])}
        if tolerance_ms is not None:
            out["spike_times"] = np.asarray(self._e.decode_spike_times(
                r["lam"], r["prob"], self.s.M, self.fps, tol_ms=float(tolerance_ms), count_gain=float(count_gain),
                cell_type=self.s.decode_cell_type), np.float64)
        if fine:
            out["count"], out["prob"] = r["lam"], r["prob"]
        return out


def _smooth_bins_per_frame(fps: float, sigma_ms: float) -> int:
    """Bins per frame of ``rate_smooth``: the smallest power of two (<= 512) whose bin is at most a quarter of sigma."""
    frame_ms, k = 1000.0 / float(fps), 1
    while k < 512 and frame_ms / k > sigma_ms / 4.0:
        k *= 2
    return k


def _to_grid(blk: np.ndarray, src_k: int, k: int) -> np.ndarray:
    """Counts per bin at src_k bins per frame -> counts per bin at k bins per frame (both powers of two): sum-pooled when
    coarser, each bin spread evenly over its k/src_k sub-bins when finer. Totals are kept."""
    if src_k == k:
        return blk
    if src_k > k:
        r = src_k // k
        return blk.reshape(blk.shape[0], -1, r).sum(-1)
    r = k // src_k
    return np.repeat(blk, r, axis=1) / np.float32(r)


def _smooth_rate(counts: np.ndarray, src_k: int, k: int, fps: float, sigma_ms: float, device=None, out=None) -> np.ndarray:
    """Expected spikes per bin (n_neurons x n_frames*src_k) -> spikes/s on k bins per frame, Gaussian-smoothed (scipy's
    kernel: truncated at 4 sigma, edges reflected), in blocks of rows of ~256 MB, on a CUDA ``device`` when given. ``out``
    (n_neurons x n_frames*k float32, e.g. a np.memmap; may be ``counts`` itself when src_k == k) receives the result."""
    n = counts.shape[0]; T = counts.shape[1] // src_k
    sig = sigma_ms / 1000.0 * float(fps) * k
    scale = np.float32(float(fps) * k)              # expected spikes per bin -> spikes/s
    if out is None:
        out = np.empty((n, T * k), np.float32)
    rad = int(4.0 * sig + 0.5)
    step = int(max(1, (256 << 20) // max(1, 4 * T * max(k, src_k))))
    if device is not None and str(getattr(device, "type", device)).startswith("cuda") and 0 < rad < T * k:
        import torch
        x = np.arange(-rad, rad + 1, dtype=np.float64)
        w = np.exp(-0.5 * (x / sig) ** 2); w /= w.sum()
        wt = torch.as_tensor((w * scale).astype(np.float32), device=device).view(1, 1, -1)
        with torch.no_grad():
            for a in range(0, n, step):
                blk = _to_grid(np.asarray(counts[a:a + step], np.float32), src_k, k)
                xb = torch.as_tensor(np.ascontiguousarray(blk), device=device).unsqueeze(1)
                # scipy's "reflect" repeats the edge sample (d c b a | a b c d), unlike torch's
                xb = torch.cat([xb[..., :rad].flip(-1), xb, xb[..., -rad:].flip(-1)], -1)
                out[a:a + step] = torch.nn.functional.conv1d(xb, wt).squeeze(1).cpu().numpy()
        return out
    from scipy import ndimage as ndi
    for a in range(0, n, step):
        blk = _to_grid(np.asarray(counts[a:a + step], np.float32), src_k, k)
        out[a:a + step] = ndi.gaussian_filter1d(blk, sig, axis=1, mode="reflect") * scale
    return out


def smooth_rate(counts, fps: float, bins_per_frame: int = 1, smooth_ms: float = SMOOTH_MS, neurons=None,
                start_s: float | None = None, end_s: float | None = None, out_path: str | Path | None = None,
                device=None) -> dict:
    """A sub-frame count stream (n_neurons x n_frames*bins_per_frame, expected spikes per bin, e.g. infer(...,
    bins_per_frame=16)["count"]) -> the Gaussian-smoothed rate in spikes/s on k bins per frame (k = the smallest power of
    two whose bin is at most sigma/4): the stream is sum-pooled to k bins per frame, or each bin spread evenly over the
    finer bins when bins_per_frame < k, then smoothed with sigma ``smooth_ms``. With bins_per_frame >= k this is what
    infer(..., smooth_ms=...) returns as ``rate_smooth``; bins_per_frame=1 smooths the per-frame ``rate`` instead (no
    timing within the frame). For any subset: ``neurons`` (indices) and a time window ``start_s``..``end_s`` (seconds from
    the first frame; padded internally by the kernel width, so the values equal those of the whole recording). Memory
    scales with the subset. ``out_path`` writes the result to a .npy file and returns it memory-mapped. ``device``: a torch
    device to smooth on (default: CPU, scipy). Returns ``rate_smooth``, ``t_smooth`` (bin centres, seconds from the first
    frame), ``bins_per_frame`` (k) and ``smooth_ms``."""
    if not float(smooth_ms) > 0:
        raise ValueError("smooth_ms must be positive")
    b = int(bins_per_frame)
    if b < 1 or b > 512 or (b & (b - 1)):
        raise ValueError("bins_per_frame must be a power of two between 1 and 512")
    C = np.atleast_2d(np.asarray(counts))
    if neurons is not None:
        C = C[np.atleast_1d(neurons)]
    if C.shape[1] % b:
        raise ValueError("the stream's length is not a whole number of frames at bins_per_frame=%d" % b)
    n, T = C.shape[0], C.shape[1] // b
    fps = float(fps); k = _smooth_bins_per_frame(fps, float(smooth_ms))
    f0 = 0 if start_s is None else max(0, int(np.floor(float(start_s) * fps)))
    f1 = T if end_s is None else min(T, int(np.ceil(float(end_s) * fps)))
    if f1 <= f0:
        raise ValueError("empty time window")
    pad = int(np.ceil(4.0 * float(smooth_ms) / 1000.0 * fps)) + 1            # kernel half-width in frames (+1)
    a, e = max(0, f0 - pad), min(T, f1 + pad)
    src = C[:, a * b:e * b]
    if out_path is not None and (a, e) == (f0, f1):                         # whole recording: straight to disk
        from numpy.lib.format import open_memmap
        mm = open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=(n, (e - a) * k))
        _smooth_rate(src, b, k, fps, float(smooth_ms), device=device, out=mm)
        mm.flush(); del mm
        res = np.load(str(out_path), mmap_mode="r")
    else:
        res = _smooth_rate(src, b, k, fps, float(smooth_ms), device=device)[:, (f0 - a) * k:(f1 - a) * k]
        if out_path is not None:
            np.save(str(out_path), res); res = np.load(str(out_path), mmap_mode="r")
    t = (np.arange(f0 * k, f1 * k) + 0.5) / (fps * k)
    return {"rate_smooth": res, "t_smooth": t, "bins_per_frame": k, "smooth_ms": float(smooth_ms)}


def _infer_fast(sess, X, rows, tol, gain, rate, nu, spikes, progress=False, sub=None, sub_k=0):
    """Every row of X through the session the way JupiterScope's session runner does it (the paper's speed
    benchmark): on CUDA each row's forward is queued before the previous row is collected and decoded, the timing
    decoder decodes groups of rows per launch on a worker thread, the detection decoder runs on worker threads, and the
    per-row noise and z-score statistics are computed in blocks on a side thread. Same rates and spike times as
    Session.infer_one, row for row. With ``sub`` (n x n_frames*sub_k), each row's count stream is also sum-pooled to
    ``sub_k`` bins per frame on the device and stored there (the source of rate_smooth)."""
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor
    E, S = sess._e, sess.s
    fs, M, T = sess.fps, S.M, X.shape[1]
    want = tol is not None
    decoder, width = E._route.decoder_for(float(tol), fs) if want else (None, None)
    evc = want and decoder == "detection" and E.evcount_fast_ok()
    seg_bins = None
    if want and not evc:
        seg_bins = (E.timing_segment_bins(M, fs, n_frames=int(T), width_ms=width) if decoder == "timing"
                    else E.timing_segment_bins(M, fs, 30.0))
    pipelined = str(getattr(S.device, "type", "")) == "cuda"
    use_stats = getattr(S, "use_prep_stats", None)
    store = {} if (pipelined and use_stats is not None) else None
    stop = threading.Event()
    if use_stats is not None:
        use_stats(store)
    if store is not None:
        def _stats(rows_=np.asarray(rows, np.int64)):
            try:
                step = int(max(1, min(256, (32 << 20) // max(1, 8 * int(T)))))
                for a in range(0, rows_.size, step):
                    if stop.is_set():
                        return
                    blk = rows_[a:a + step]
                    for r, st in zip(blk, E.prep_stats_many(X[blk], fs)):
                        if st is not None:
                            store[int(r)] = st
            except Exception:
                pass
        threading.Thread(target=_stats, name="setsuna_row_stats", daemon=True).start()

    def collect(row, out):
        E.finish_outputs(out)
        rate[row] = out["rate"]; nu[row] = out["nu"]
        if sub is not None:
            sub[row] = np.asarray(out["lam_pooled"], np.float32)[:sub.shape[1]]

    def decode_st(out):
        return E.decode_spike_times(out.get("lam"), out["prob"], M, fs, tol_ms=float(tol), count_gain=float(gain),
                                    lam_frame=out.get("lam_frame"), p_frame=out.get("p_frame"),
                                    ready_event=out.get("done_event"), lam_part=out.get("lam_part"),
                                    prob_cmax=out.get("prob_cmax"), cell_type=S.decode_cell_type)

    pool, queue, qmax = None, [], 0
    if pipelined and evc:
        n_thr = int(os.environ.get("SETSUNA_DECODE_THREADS", "3"))
        if n_thr > 1:
            pool, qmax = ThreadPoolExecutor(n_thr, thread_name_prefix="setsuna_decode"), 2 * n_thr
    batch = pipelined and want and not evc and decoder == "timing" and E.decode_batch_ok(S.device)
    dk = 1
    if batch:
        mb = float(os.environ.get("SETSUNA_DECODE_BATCH_MB", "768"))
        dk = int(max(1, min(16, (mb * 2 ** 20) // max(1, int(T) * int(M) * 4))))
        if dk > 1:
            pool, qmax = ThreadPoolExecutor(1, thread_name_prefix="setsuna_timing"), 1

    def drain(limit):
        while queue and (len(queue) > limit or queue[0][1].done()):
            rr, fut = queue.pop(0)
            for r, st in zip(rr, fut.result()):
                spikes[r] = np.asarray(st, np.float64)

    def flush(group):
        for row, out in group:
            collect(row, out)
        if not want:
            return
        if dk > 1 and len(group) > 1:
            items = [dict(lam=o.get("lam"), prob=o["prob"], lam_frame=o.get("lam_frame"), lam_part=o.get("lam_part"),
                          prob_cmax=o.get("prob_cmax"), ready_event=o.get("done_event")) for _r, o in group]
            if pool is not None:
                queue.append(([r for r, _o in group], pool.submit(E.decode_timing_many, items, M, fs, width,
                                                                  count_gain=float(gain))))
                drain(qmax)
            else:
                for (r, _o), st in zip(group, E.decode_timing_many(items, M, fs, width, count_gain=float(gain))):
                    spikes[r] = np.asarray(st, np.float64)
        elif pool is not None:
            for r, o in group:
                queue.append(([r], pool.submit(lambda o=o: [decode_st(o)])))
            drain(qmax)
        else:
            for r, o in group:
                spikes[r] = np.asarray(decode_st(o), np.float64)

    pending = []
    try:
        for k, row in enumerate(rows):
            nxt = (int(rows[k + 1]), X[rows[k + 1]]) if k + 1 < len(rows) else None
            out = S.infer_roi_outputs(X[row], pool_k=(M // sub_k if sub is not None else None), need_full=want and not evc, full_device=True, key=int(row),
                                      prefetch=nxt, fine_prob=evc, defer_copy=pipelined, seg_bins=seg_bins)
            if pipelined:
                pending.append((int(row), out))
                if len(pending) >= 2 * dk:
                    flush(pending[:dk]); del pending[:dk]
            else:
                flush([(int(row), out)])
            if progress and (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(rows)} neurons", flush=True)
        while pending:
            flush(pending[:dk]); del pending[:dk]
        if pool is not None:
            drain(0)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        stop.set()
        if use_stats is not None:
            use_stats(None)


def infer(dff, fps: float, indicator: str = "jGCaMP8s", cell_type: str = "pyramidal", tolerance_ms: float | None = 10.0,
          count_gain: float = 1.0, model_dir: str | Path | None = None, device: str = "auto", fine: bool = False,
          progress: bool = False, batch_windows: int = 512, bins_per_frame: int | None = None,
          smooth_ms: float | None = None, rate_smooth_path: str | Path | None = None,
          max_smooth_gb: float | None = MAX_SMOOTH_GB) -> dict:
    """Infer spikes for every neuron of ``dff`` (n_neurons x n_frames, fractional dF/F at ``fps`` Hz).

    ``tolerance_ms`` is the spike-timing precision your analysis needs; it picks the decoder and its smoothing width
    (timing decoder below one frame and 35 ms, detection decoder otherwise). ``None`` skips spike times.
    ``count_gain`` scales the number of decoded spikes (1 = the count head's estimate).
    Returns ``rate`` (n_neurons x n_frames, expected spikes per frame; multiply by fps for spikes/s), ``nu`` (noise level
    per neuron), ``spike_times`` (list of arrays, seconds from the first frame) and, with ``fine``, the 512-bin streams.
    Rows that are all NaN are returned as NaN with no spikes.
    On a CUDA GPU the neurons are pipelined (one neuron's forward pass runs while the previous one is decoded); lower
    ``batch_windows`` if the GPU runs out of memory. ``fine=True`` processes one neuron at a time and is slower.
    ``bins_per_frame`` (a power of two up to 512, e.g. 16) returns the sub-frame streams sum-pooled to that many bins per
    frame: ``count`` (expected spikes per bin), ``prob`` and, with ``tolerance_ms``, ``spike_counts`` (decoded spikes per
    bin), each n_neurons x (n_frames * bins_per_frame). Sum-pooling keeps totals, so a stream saved at frame/16 can be
    pooled further to any coarser bin. It implies ``fine`` and runs one neuron at a time.
    ``smooth_ms`` (recommended for rate analyses: 25, i.e. SMOOTH_MS, at every frame rate; default None = off) also returns
    ``rate_smooth``: the model's sub-frame count stream (the count head's 512 bins per frame, sum-pooled to k bins per
    frame, k = the smallest power of two whose bin is at most sigma/4) smoothed with a Gaussian of that sigma, in spikes/s,
    and ``t_smooth``, the bin centres in seconds from the first frame. ``rate``, ``spike_times`` and the rest are returned
    as usual; frame totals equal ``rate``. For timing of single spikes use ``spike_times``. With smoothing on, ``rate`` at
    <= 10 Hz can differ from a run without it in the last float32 digit (the pooled GPU path sums in another order);
    spike times are identical. An in-memory ``rate_smooth`` above ``max_smooth_gb`` GiB (default 2) is refused before
    inference starts, with the size in the message: pass ``rate_smooth_path`` to write it to a .npy file (filled during
    inference and smoothed there block by block; returned memory-mapped), raise ``max_smooth_gb`` (None = no limit), or
    use bins_per_frame with smooth_rate() on subsets."""
    if bins_per_frame is not None:
        k = int(bins_per_frame)
        if k < 1 or k > 512 or (k & (k - 1)):
            raise ValueError("bins_per_frame must be a power of two between 1 and 512")
        fine = True
    X = np.atleast_2d(np.asarray(dff, np.float32))
    sess = Session(fps, indicator, cell_type, model_dir=model_dir, device=device, batch_windows=batch_windows)
    n, T = X.shape
    if smooth_ms is not None and not float(smooth_ms) > 0:
        raise ValueError("smooth_ms must be positive, or None to skip rate_smooth")
    sk = _smooth_bins_per_frame(fps, float(smooth_ms)) if smooth_ms is not None else 0
    if sk and rate_smooth_path is None and max_smooth_gb is not None:
        gb = n * T * sk * 4 / 2 ** 30
        if gb > float(max_smooth_gb):
            raise ValueError(
                f"rate_smooth would take {gb:.2f} GiB in memory ({n} neurons x {T} frames x {sk} bins per frame, float32), "
                f"above max_smooth_gb={max_smooth_gb}. Pass rate_smooth_path='file.npy' to write it to disk block by block, "
                f"raise max_smooth_gb (None = no limit), or smooth subsets with setsuna.smooth_rate(out['count'], fps, "
                f"bins_per_frame=..., neurons=..., start_s=..., end_s=...) from a run with bins_per_frame.")
    sub = None
    if sk:                         # the source stream: in memory, or the output file itself (smoothed there in place)
        if rate_smooth_path is not None:
            from numpy.lib.format import open_memmap
            sub = open_memmap(str(rate_smooth_path), mode="w+", dtype=np.float32, shape=(n, T * sk))
            sub[:] = np.nan
        else:
            sub = np.full((n, T * sk), np.nan, np.float32)
    rate = np.full((n, T), np.nan, np.float32); nu = np.full(n, np.nan); spikes = [np.zeros(0) for _ in range(n)]
    count, prob = ([None] * n, [None] * n) if fine else (None, None)
    if not fine:
        ok = np.isfinite(X).any(1)
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
        _infer_fast(sess, X, np.flatnonzero(ok), tolerance_ms, count_gain, rate, nu, spikes, progress, sub, sk)
        n = 0   # every row done
    for i in range(n):
        row = X[i]
        if not np.isfinite(row).any():
            continue
        row = np.where(np.isfinite(row), row, 0.0).astype(np.float32)
        o = sess.infer_one(row, tolerance_ms=tolerance_ms, count_gain=count_gain, fine=fine)
        rate[i], nu[i] = o["rate"], o["nu"]
        if sub is not None:
            M = int(sess.s.M)
            sub[i] = np.asarray(o["count"], np.float64)[:T * M].reshape(T, sk, M // sk).sum(-1).ravel()
        if tolerance_ms is not None:
            spikes[i] = o["spike_times"]
        if fine:
            count[i], prob[i] = o["count"], o["prob"]
        if progress and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} neurons", flush=True)
    out = {"rate": rate, "nu": nu, "fps": float(fps), "indicator": indicator, "cell_type": cell_type,
           "band": sess.s.band, "tolerance_ms": tolerance_ms}
    if tolerance_ms is not None:
        out["spike_times"] = spikes
    if sk:
        _smooth_rate(sub, sk, sk, fps, float(smooth_ms), device=sess.s.device, out=sub)        # in place, block by block
        if rate_smooth_path is not None:
            sub.flush(); del sub
            out["rate_smooth"] = np.load(str(rate_smooth_path), mmap_mode="r")
        else:
            out["rate_smooth"] = sub
        out["t_smooth"] = (np.arange(T * sk) + 0.5) / (float(fps) * sk)
        out["smooth_ms"], out["smooth_bins_per_frame"] = float(smooth_ms), sk
    if fine and bins_per_frame is None:
        out["count"], out["prob"], out["bins_per_frame"] = count, prob, sess.s.M
    elif fine:
        M, k = int(sess.s.M), int(bins_per_frame)
        def pool(v):
            return np.full(T * k, np.nan, np.float32) if v is None else                 np.asarray(v, np.float64)[:T * M].reshape(T, k, M // k).sum(-1).ravel().astype(np.float32)
        out["count"] = np.stack([pool(v) for v in count])
        out["prob"] = np.stack([pool(v) for v in prob])
        out["bins_per_frame"] = k
        if tolerance_ms is not None:
            edges = np.arange(T * k + 1) / (float(fps) * k)
            out["spike_counts"] = np.stack([np.histogram(st, edges)[0].astype(np.float32) for st in spikes])
    return out
