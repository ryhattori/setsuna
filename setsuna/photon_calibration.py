"""SETSUNA's photon-noise count calibration ("option A"), without SETSUNA's model code (no torch).

Under photon-limited (shot) noise the pyramidal models over-count: on the C167 photon benchmark arm
(held out, GCaMP7f/8f/8m/8s) the predicted count at 0.5 photons per frame was 3.6-8.9x the true count, at
1 photon 1.8-4.1x, and within ~1.0-1.5x from ~5 photons up. Correlation and spike timing are not the
problem; the count is. This module estimates each ROI's photon level from its own dF/F and returns one
divisor ``c >= 1`` per ROI for the REPORTED count and rate streams (per-frame counts, the sub-frame count
stream, count summaries). Spike placement keeps the model's own count (user decision 2026-10-24: correcting
the placement count too cost F1@25 0.006-0.022 and 5-16 ms of jitter at <= 1 photon), and the probability
stream is untouched.

Estimator (``photon_level``, = spikeinf code_c167/plc_r2.lhat_mv with its intercept, the photon agent's
``lhat_mv2``): on the WHOLE ROI dF/F trace the model receives (fractional, baseline at 0), z = 1 + dF/F in
non-overlapping 8-frame blocks; per block the mean and half the variance of the first differences; 8 bins
between the 5th and 95th percentile of the block means, keeping bins with >= 5 blocks (>= 4 bins needed);
least squares through the bins' medians var = a + b * mean. Photon level L_hat = 1 / b (photons per frame at
the baseline, in dF/F units), shot-noise fraction frac_shot = b * mu0 / (a + b * mu0) at mu0 = the 5th
percentile of the block means. Needs >= 30 blocks (240 frames).

Rule (``count_calibration``): c = 1 when L_hat is not measurable, when frac_shot < 0.5 (the noise is not
shot-dominated: a Gaussian / read-noise trace never gets a correction -- on the Gaussian benchmark the rule
fired on 0.76% of rows and moved no score by more than 1e-4), or when L_hat >= L_ref (10); otherwise
c = exp(h), h = interp(log L_hat) in the isotonic table of the ROI's frame-rate bin (clamped at the table's
ends, h >= 0). One constant per ROI.

The constants (photon_calibration_<version>.json beside this file) are PROVISIONAL: fitted on the C167
pyramidal photon arm (GCaMP7f, GCaMP8f, GCaMP8m, GCaMP8s; 2026-10-23), indicator-free, per frame-rate bin.
GCaMP6f/6s, jRGECO1a and the interneuron / spinal / all-cell-types models are not covered, so SETSUNA
applies it to the pyramidal specialist / generalist sets only (SetsunaSession.count_calibration_applies).
A refit on the full C167 photon arm (with per-indicator tables where they beat the pooled one) is planned.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

#: The constants file shipped with this JupiterScope (beside this module).
CONSTANTS_FILE = "photon_calibration_optA_isorel_v1.json"
#: Blocks of this many frames, at least MIN_BLOCKS of them (240 frames), for the photon-level estimate.
BLOCK = 8
MIN_BLOCKS = 30

_CACHE: dict = {}


def constants_path() -> Path:
    return Path(__file__).resolve().parent / CONSTANTS_FILE


def load_constants(path=None) -> dict:
    """The calibration constants (cached per path): version, gate, L_ref, fps_bins, per-bin tables."""
    p = Path(path) if path else constants_path()
    key = str(p)
    if key not in _CACHE:
        c = json.loads(p.read_text(encoding="utf-8"))
        for name, t in c["tables"].items():
            x = np.asarray(t["log_lhat"], float)
            if x.size < 2 or x.size != len(t["h"]) or np.any(np.diff(x) < 0):
                raise ValueError(f"photon calibration table {name!r} in {p.name} is not an increasing knot list")
        _CACHE[key] = c
    return _CACHE[key]


def version(path=None) -> str:
    return str(load_constants(path).get("version", ""))


def photon_level(dff, block: int = BLOCK) -> tuple[float, float]:
    """(L_hat, frac_shot) of one ROI's whole fractional dF/F trace; (nan, nan) when it cannot be estimated
    (fewer than 30 blocks, non-finite values, too few populated bins, or a non-positive slope)."""
    z = 1.0 + np.asarray(dff, float)
    n = len(z) // block
    if n < MIN_BLOCKS or not np.all(np.isfinite(z[: n * block])):
        return math.nan, math.nan
    Z = z[: n * block].reshape(n, block)
    mu = Z.mean(1)
    var = np.var(np.diff(Z, axis=1), axis=1) / 2.0
    q = np.percentile(mu, [5, 95])
    if not q[1] > q[0]:
        return math.nan, math.nan
    edges = np.linspace(q[0], q[1], 9)
    idx = np.digitize(mu, edges)
    mx, vy = [], []
    for b in range(1, 9):
        s = idx == b
        if s.sum() >= 5:
            mx.append(np.median(mu[s]))
            vy.append(np.median(var[s]))
    if len(mx) < 4:
        return math.nan, math.nan
    slope, icpt = np.polyfit(mx, vy, 1)
    if not slope > 0:
        return math.nan, math.nan
    mu0 = q[0]
    return float(1.0 / slope), float(slope * mu0 / max(icpt + slope * mu0, 1e-12))


def fps_bin(fps: float, consts: dict) -> "str | None":
    """The constants' frame-rate bin for ``fps`` (lo < fps <= hi), or None."""
    f = float(fps)
    for lo, hi, name in consts["fps_bins"]:
        if float(lo) < f <= float(hi):
            return str(name)
    return None


def divisor(lhat: float, frac_shot: float, fps: float, consts: dict) -> tuple[float, str]:
    """(c, reason) for one ROI from its photon-level estimate; reason is "" when the correction applies."""
    if not (math.isfinite(lhat) and lhat > 0):
        return 1.0, "photon level not measurable"
    if not (math.isfinite(frac_shot) and frac_shot >= float(consts["gate"]["frac_shot_min"])):
        return 1.0, "noise not shot-dominated"
    if lhat >= float(consts["L_ref"]):
        return 1.0, "bright (photon level >= L_ref)"
    name = fps_bin(fps, consts)
    table = consts["tables"].get(name) if name is not None else None
    if table is None:
        return 1.0, "no table for this frame rate"
    h = float(np.interp(math.log(lhat), np.asarray(table["log_lhat"], float), np.asarray(table["h"], float)))
    h = max(0.0, h)
    return (float(math.exp(h)), "") if h > 0.0 else (1.0, "")


def count_calibration(dff, fps: float, consts: "dict | None" = None) -> dict:
    """{lhat, frac_shot, c, reason} for one ROI's whole fractional dF/F trace at frame rate ``fps``.
    Divide the REPORTED counts / rates by ``c`` (never the counts used to place spikes)."""
    consts = consts if consts is not None else load_constants()
    lhat, frac = photon_level(dff)
    c, reason = divisor(lhat, frac, fps, consts)
    return {"lhat": lhat, "frac_shot": frac, "c": c, "reason": reason}


def photon_level_many(rows) -> list:
    """photon_level of each row of the 2-D array ``rows`` (one ROI's whole trace per row), the same floats: the
    block means and variances, the percentiles, the bin edges, the bin assignment and the bins' medians are found
    for all rows at once (numpy's own reductions along each row, exact order statistics for the medians); only the
    least-squares fit stays per row (np.polyfit, as photon_level). For a few hundred rows at a time: the
    per-ROI version's ~20 small numpy calls held the GIL ~0.8 ms per ROI, next to SETSUNA's launch-bound forward."""
    rows = np.asarray(rows)
    R = int(rows.shape[0]) if rows.ndim == 2 else 0
    out = [(math.nan, math.nan)] * R
    if R == 0:
        return out
    n = int(rows.shape[1]) // BLOCK
    if n < MIN_BLOCKS:
        return out
    z = 1.0 + np.ascontiguousarray(rows[:, : n * BLOCK], dtype=float)
    ok = np.isfinite(z).all(axis=1)
    if not ok.all():
        z = z[ok]
    keep = np.flatnonzero(ok)
    if keep.size == 0:
        return out
    Z = z.reshape(keep.size, n, BLOCK)
    mu = Z.mean(2)
    var = np.var(np.diff(Z, axis=2), axis=2) / 2.0
    q = np.percentile(mu, [5, 95], axis=1)
    good = q[1] > q[0]
    sel = np.flatnonzero(good)
    if sel.size == 0:
        return out
    mu, var, q0, q1 = mu[sel], var[sel], q[0][sel], q[1][sel]
    m = sel.size
    edges = np.stack([np.linspace(a, b, 9) for a, b in zip(q0, q1)])
    # np.digitize(mu, edges) is np.searchsorted(edges, mu, side="right") for increasing edges
    idx = np.stack([np.searchsorted(e, v, side="right") for e, v in zip(edges, mu)])
    cnt = np.bincount((idx + 10 * np.arange(m)[:, None]).ravel(), minlength=10 * m).reshape(m, 10)
    start = np.concatenate([np.zeros((m, 1), cnt.dtype), np.cumsum(cnt, axis=1)[:, :-1]], axis=1)
    # each row's values grouped by bin, ascending within a bin: the bin is monotonic in mu, so mu sorted is grouped
    # already; var by (bin, rank of var)
    mu_s = np.sort(mu, axis=1)
    rank = np.empty(var.shape, np.int64)
    np.put_along_axis(rank, np.argsort(var, axis=1), np.arange(n, dtype=np.int64)[None, :].repeat(m, 0), axis=1)
    var_s = np.take_along_axis(var, np.argsort(idx.astype(np.int64) * n + rank, axis=1), axis=1)
    for j in range(sel.size):
        mx, vy = [], []
        for b in range(1, 9):
            c = int(cnt[j, b])
            if c >= 5:
                lo = int(start[j, b]) + (c - 1) // 2
                hi = int(start[j, b]) + c // 2
                # np.median: the middle value, or the mean of the two middle values
                if lo == hi:
                    mx.append(mu_s[j, lo])
                    vy.append(var_s[j, lo])
                else:
                    mx.append((mu_s[j, lo] + mu_s[j, hi]) / 2.0)
                    vy.append((var_s[j, lo] + var_s[j, hi]) / 2.0)
        if len(mx) < 4:
            continue
        slope, icpt = np.polyfit(mx, vy, 1)
        if not slope > 0:
            continue
        mu0 = q0[j]
        out[int(keep[sel[j]])] = (float(1.0 / slope), float(slope * mu0 / max(icpt + slope * mu0, 1e-12)))
    return out


def count_calibration_many(rows, fps: float, consts: "dict | None" = None) -> list:
    """count_calibration of each row of the 2-D array ``rows``, the same dicts (photon_level_many)."""
    consts = consts if consts is not None else load_constants()
    res = []
    for lhat, frac in photon_level_many(rows):
        c, reason = divisor(lhat, frac, fps, consts)
        res.append({"lhat": lhat, "frac_shot": frac, "c": c, "reason": reason})
    return res


def summary(consts: "dict | None" = None) -> dict:
    """What a run records about the constants it used (info.json / the session meta)."""
    consts = consts if consts is not None else load_constants()
    return {"version": str(consts.get("version", "")), "file": CONSTANTS_FILE,
            "status": str(consts.get("status", "")), "fitted_on": str(consts.get("fitted_on", "")),
            "L_ref": float(consts["L_ref"]), "frac_shot_min": float(consts["gate"]["frac_shot_min"])}
