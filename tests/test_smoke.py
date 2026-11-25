"""Smoke test: runs SETSUNA on a synthetic trace when the models are installed (skipped otherwise)."""
import numpy as np
import pytest

import setsuna
from setsuna import engine


def _models():
    d = engine.default_model_dir()
    return d.is_dir() and any(d.glob("*.pt"))


def test_dff_shape():
    rng = np.random.default_rng(0)
    F = 100 + rng.normal(0, 1, (3, 2000)).astype(np.float32)
    d = setsuna.dff_from_fluorescence(F, 0.5 * F, fps=10.0)
    assert d.shape == F.shape and np.isfinite(d).all()


@pytest.mark.skipif(not _models(), reason="models not installed")
def test_infer_runs():
    fps, T = 10.0, 3000
    rng = np.random.default_rng(1)
    t = np.arange(T) / fps
    spikes = np.sort(rng.uniform(0, T / fps, 60))
    k = np.exp(-np.clip(t[:, None] - spikes[None, :], 0, None) / 0.5) * (t[:, None] >= spikes[None, :])
    dff = (0.3 * k.sum(1) + rng.normal(0, 0.05, T)).astype(np.float32)[None]
    out = setsuna.infer(dff, fps, "jGCaMP8s", tolerance_ms=25, device="cpu")
    assert out["rate"].shape == dff.shape and len(out["spike_times"][0]) > 10


def test_smooth_grid():
    from setsuna.api import _smooth_bins_per_frame, _smooth_rate
    assert _smooth_bins_per_frame(30.0, 25.0) == 8 and _smooth_bins_per_frame(3.0, 25.0) == 64
    assert _smooth_bins_per_frame(1000.0, 25.0) == 1
    x = np.zeros((2, 500), np.float32); x[0, 250] = 1.0            # one expected spike, k = 8 at 30 Hz
    r = _smooth_rate(x, 8, 30.0, 25.0)
    assert r.shape == (2, 4000) and not r[1].any()
    assert np.isclose(r[0].sum() / (30.0 * 8), 1.0, atol=1e-4)      # spikes/s; the total is kept


@pytest.mark.skipif(not _models(), reason="models not installed")
def test_rate_smooth_totals():
    fps, T = 10.0, 3000
    rng = np.random.default_rng(2)
    dff = (0.05 * rng.normal(size=(2, T))).astype(np.float32); dff[:, 1000:1010] += 0.5
    out = setsuna.infer(dff, fps, "jGCaMP8s", tolerance_ms=None, device="cpu")
    k = out["smooth_bins_per_frame"]
    assert out["rate_smooth"].shape == (2, T * k) and out["t_smooth"].shape == (T * k,)
    assert np.allclose(out["rate_smooth"].sum(1) / (fps * k), out["rate"].sum(1), rtol=1e-3)
