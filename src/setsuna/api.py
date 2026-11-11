"""SETSUNA's public interface: fluorescence -> dF/F -> spike rates and spike times.

    import setsuna
    dff = setsuna.dff_from_fluorescence(F, Fneu, fps=30.0)            # (n_neurons, n_frames) fractional dF/F
    out = setsuna.infer(dff, fps=30.0, indicator="jGCaMP8s", tolerance_ms=10)
    out["rate"]            # (n_neurons, n_frames) expected spikes per frame
    out["spike_times"]     # list of 1-D arrays, spike times in seconds from the first frame

Everything below calls the inference engine (engine.py) unchanged; see docs/USER_GUIDE.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["dff_from_fluorescence", "infer", "Session", "INDICATORS", "CELL_TYPES"]

#: Indicator names accepted by infer() (display spelling -> the name the models use). Any other indicator: "other".
INDICATORS = {
    "jGCaMP8s": "GCaMP8s", "jGCaMP8m": "GCaMP8m", "jGCaMP8f": "GCaMP8f", "jGCaMP7f": "GCaMP7f",
    "GCaMP6s": "GCaMP6s", "GCaMP6f": "GCaMP6f", "jRGECO1a": "jGECO1a", "other": "other",
}
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


def infer(dff, fps: float, indicator: str = "jGCaMP8s", cell_type: str = "pyramidal", tolerance_ms: float | None = 10.0,
          count_gain: float = 1.0, model_dir: str | Path | None = None, device: str = "auto", fine: bool = False,
          progress: bool = False) -> dict:
    """Infer spikes for every neuron of ``dff`` (n_neurons x n_frames, fractional dF/F at ``fps`` Hz).

    ``tolerance_ms`` is the spike-timing precision your analysis needs; it picks the decoder and its smoothing width
    (timing decoder below one frame and 35 ms, detection decoder otherwise). ``None`` skips spike times.
    ``count_gain`` scales the number of decoded spikes (1 = the count head's estimate).
    Returns ``rate`` (n_neurons x n_frames, expected spikes per frame; multiply by fps for spikes/s), ``nu`` (noise level
    per neuron), ``spike_times`` (list of arrays, seconds from the first frame) and, with ``fine``, the 512-bin streams.
    Rows that are all NaN are returned as NaN with no spikes."""
    X = np.atleast_2d(np.asarray(dff, np.float32))
    sess = Session(fps, indicator, cell_type, model_dir=model_dir, device=device)
    n, T = X.shape
    rate = np.full((n, T), np.nan, np.float32); nu = np.full(n, np.nan); spikes = [np.zeros(0) for _ in range(n)]
    count, prob = ([None] * n, [None] * n) if fine else (None, None)
    for i in range(n):
        row = X[i]
        if not np.isfinite(row).any():
            continue
        row = np.where(np.isfinite(row), row, 0.0).astype(np.float32)
        o = sess.infer_one(row, tolerance_ms=tolerance_ms, count_gain=count_gain, fine=fine)
        rate[i], nu[i] = o["rate"], o["nu"]
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
    if fine:
        out["count"], out["prob"], out["bins_per_frame"] = count, prob, sess.s.M
    return out
