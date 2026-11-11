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
