# SETSUNA

**Spike Estimation with Temporal Super-resolution, Universal across Neuron types and Acquisition settings.**

SETSUNA infers spikes from two-photon calcium imaging on a time grid 512 times finer than the imaging frame. One network
design, conditioned on frame rate, noise level, indicator and cell type, covers seven calcium indicators (jGCaMP8s/m/f,
jGCaMP7f, GCaMP6s/f, jRGECO1a), frame rates from 1 to 120 Hz, cortical pyramidal neurons and interneurons and spinal-cord
neurons, and runs on indicators and cell types without ground truth.

```python
import setsuna
dff = setsuna.dff_from_fluorescence(F, Fneu, fps=30.0)          # raw + neuropil fluorescence -> fractional dF/F
out = setsuna.infer(dff, fps=30.0, indicator="jGCaMP8s", tolerance_ms=10)
out["rate"]          # expected spikes per frame
out["spike_times"]   # spike times (s), decoded at 10-ms precision
```

```
setsuna run --npz traces.npz --fps 30 --indicator jGCaMP8s --tolerance-ms 10 --out spikes.npz
```

## Install

    pip install git+https://github.com/ryhattori/setsuna.git

Requires Python >= 3.10 and PyTorch >= 2.2 (with CUDA for GPU inference). The trained models are distributed
separately: download them into `~/.setsuna/models` or set `SETSUNA_MODEL_DIR` (model download: to be added with the
release).

## Documentation

[docs/USER_GUIDE.md](docs/USER_GUIDE.md): preparing input, choosing the indicator, cell type and timing tolerance, and
reading the output.

SETSUNA is also built into JupiterScope, which runs it from a GUI after registration, segmentation and fluorescence
extraction.

## Citation

Hattori, R. Breaking the frame-rate limit of calcium imaging (2026).
