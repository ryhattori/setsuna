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
    r = _smooth_rate(x, 1, 8, 30.0, 25.0)                          # per-frame source spread over 8 bins
    assert r.shape == (2, 4000) and not r[1].any()
    assert np.isclose(r[0].sum() / (30.0 * 8), 1.0, atol=1e-4)      # spikes/s; the total is kept


@pytest.mark.skipif(not _models(), reason="models not installed")
def test_rate_smooth_totals():
    fps, T = 10.0, 3000
    rng = np.random.default_rng(2)
    dff = (0.05 * rng.normal(size=(2, T))).astype(np.float32); dff[:, 1000:1010] += 0.5
    out = setsuna.infer(dff, fps, "jGCaMP8s", tolerance_ms=None, device="cpu", smooth_ms=25)
    k = out["smooth_bins_per_frame"]
    assert out["rate_smooth"].shape == (2, T * k) and out["t_smooth"].shape == (T * k,)
    assert np.allclose(out["rate_smooth"].sum(1) / (fps * k), out["rate"].sum(1), rtol=1e-3)


def test_smooth_rate_subsets():
    """smooth_rate on a subset of neurons / a time window equals the same slice of the whole recording; out_path matches."""
    import tempfile, os
    rng = np.random.default_rng(3)
    rate = rng.gamma(0.3, 0.2, (5, 900)).astype(np.float32)
    full = setsuna.smooth_rate(rate, 30.0, bins_per_frame=1, smooth_ms=25.0)
    k = full["bins_per_frame"]
    sub = setsuna.smooth_rate(rate, 30.0, bins_per_frame=1, smooth_ms=25.0, neurons=[1, 3], start_s=10.0, end_s=20.0)
    a, b = int(10.0 * 30) * k, int(20.0 * 30) * k
    assert np.allclose(sub["rate_smooth"], full["rate_smooth"][[1, 3], a:b], rtol=1e-5, atol=1e-6)
    assert np.allclose(sub["t_smooth"], full["t_smooth"][a:b])
    with tempfile.TemporaryDirectory() as d:
        disk = setsuna.smooth_rate(rate, 30.0, bins_per_frame=1, smooth_ms=25.0, out_path=os.path.join(d, "rs.npy"))
        assert np.allclose(np.asarray(disk["rate_smooth"]), full["rate_smooth"])
        del disk


@pytest.mark.skipif(not _models(), reason="models not installed")
def test_rate_smooth_size_guard():
    dff = np.zeros((4, 2000), np.float32)
    with pytest.raises(ValueError, match="GiB"):
        setsuna.infer(dff, 10.0, "jGCaMP8s", tolerance_ms=None, device="cpu", smooth_ms=25, max_smooth_gb=1e-6)
    out = setsuna.infer(dff, 10.0, "jGCaMP8s", tolerance_ms=None, device="cpu")
    assert "rate_smooth" not in out                                  # off by default


def test_smooth_rate_from_a_subframe_stream():
    """A stream at 16 bins per frame pooled to k = 8 (30 Hz, 25 ms) equals the same counts given at 8 bins per frame."""
    rng = np.random.default_rng(4)
    c16 = rng.gamma(0.3, 0.02, (3, 600 * 16)).astype(np.float32)
    c8 = c16.reshape(3, -1, 2).sum(-1)
    a = setsuna.smooth_rate(c16, 30.0, bins_per_frame=16)["rate_smooth"]
    b = setsuna.smooth_rate(c8, 30.0, bins_per_frame=8)["rate_smooth"]
    assert a.shape == (3, 600 * 8) and np.allclose(a, b, rtol=1e-5, atol=1e-6)
    assert np.allclose(a.sum(1) / (30.0 * 8), c16.sum(1), rtol=1e-3)


@pytest.mark.skipif(not _models(), reason="models not installed")
def test_rate_smooth_is_the_smoothed_subframe_stream():
    fps, T = 10.0, 2000
    rng = np.random.default_rng(5)
    dff = (0.05 * rng.normal(size=(2, T))).astype(np.float32); dff[:, 500:510] += 0.5
    a = setsuna.infer(dff, fps, "jGCaMP8s", tolerance_ms=None, device="cpu", smooth_ms=25)
    b = setsuna.infer(dff, fps, "jGCaMP8s", tolerance_ms=None, device="cpu", bins_per_frame=a["smooth_bins_per_frame"])
    want = setsuna.smooth_rate(b["count"], fps, bins_per_frame=b["bins_per_frame"])["rate_smooth"]
    assert np.allclose(a["rate_smooth"], want, rtol=1e-3, atol=1e-4)
