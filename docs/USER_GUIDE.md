# SETSUNA user guide

SETSUNA infers spikes from calcium-imaging traces on a time grid 512 times finer than the imaging frame. One network
design covers seven calcium indicators, frame rates from 1 to 120 Hz and the noise levels of real experiments; it is
told the frame rate, the noise level, the indicator and the cell type of each recording, and returns

* a **rate**: the expected number of spikes in every frame (or in every 1/512 of a frame), and
* **spike times**, decoded at the timing precision your analysis needs.

## 1. Install

    pip install git+https://github.com/ryhattori/setsuna.git

SETSUNA needs Python 3.10 or later, PyTorch 2.2 or later, NumPy, SciPy and Numba. Install PyTorch with CUDA support
first (see pytorch.org) to run on an NVIDIA GPU; SETSUNA also runs on the CPU, more slowly.

## 2. Get the models

Download the model files (link in the README) and unpack them into `~/.setsuna/models`, or anywhere else and point
`SETSUNA_MODEL_DIR` at that folder. Check with

    setsuna models

## 3. Prepare the input

SETSUNA reads **fractional dF/F** (0.5 means a 50% increase), one row per neuron, sampled at the frame rate.

If you have raw fluorescence and neuropil traces (for example Suite2p's `F.npy` and `Fneu.npy`), let SETSUNA compute
dF/F the way it was trained and benchmarked:

```python
import numpy as np, setsuna
F, Fneu = np.load("F.npy"), np.load("Fneu.npy")
iscell = np.load("iscell.npy")[:, 0].astype(bool)
dff = setsuna.dff_from_fluorescence(F[iscell], Fneu[iscell], fps=30.0, neuropil_coefficient=0.7)
```

The baseline is a maximin baseline over 60 s, which follows slow drift and stays under calcium transients whatever a
neuron's firing; the neuropil-corrected trace is divided by the raw fluorescence baseline, which stays positive. If you
already have dF/F from your own pipeline, use it directly, as long as it is fractional and not z-scored. SETSUNA
standardizes each trace itself and measures its noise level on the fractional trace.

**Frame rate.** Use the rate at which each neuron is sampled: for multi-plane imaging, the volume rate, not the
line or plane rate.

## 4. Run

```python
out = setsuna.infer(dff, fps=30.0, indicator="jGCaMP8s", cell_type="pyramidal", tolerance_ms=10)
out["rate"]          # (n_neurons, n_frames) expected spikes per frame; x fps = spikes per second
out["spike_times"]   # list of arrays, spike times in seconds from the first frame
out["nu"]            # noise level of each trace (the standardized noise level used by CASCADE)
```

or from the command line:

    setsuna run --npz traces.npz --fps 30 --indicator jGCaMP8s --tolerance-ms 10 --out spikes.npz

(`traces.npz` holds `F` and optionally `Fneu`; `--dff file.npy` takes dF/F directly.) The output file holds `rate`,
`nu`, and the spike times as `spike_neuron` and `spike_time`.

For many recordings at the same frame rate and indicator, load the models once:

```python
sess = setsuna.Session(fps=30.0, indicator="jGCaMP8s")
for trace in traces:
    o = sess.infer_one(trace, tolerance_ms=10)
```

## 5. Choosing the indicator and cell type

| `indicator` | |
|---|---|
| `jGCaMP8s`, `jGCaMP8m`, `jGCaMP8f`, `jGCaMP7f`, `GCaMP6s`, `GCaMP6f`, `jRGECO1a` | models trained with ground truth for that indicator |
| `other` | any other indicator: SETSUNA runs without an indicator identity |

| `cell_type` | |
|---|---|
| `pyramidal` | cortical pyramidal neurons |
| `PV_FS`, `SST`, `VIP` | cortical interneurons |
| `spinal_excitatory`, `spinal_inhibitory` | dorsal-horn neurons |
| `unknown` | cell type not known, probably pyramidal: runs the pyramidal models |
| `other` | a cell type not listed (striatal, cerebellar, ...): runs without a cell-type identity |

Choose the indicator you imaged even if it is not listed (use `other`), rather than the closest listed indicator:
SETSUNA was tested this way, with the indicator or cell type withheld from training, and loses little in most
populations.

## 6. Choosing the timing tolerance

`tolerance_ms` is the precision with which your analysis needs spike times: a spike-time correlation at 5 ms needs
5 ms; counting spikes per frame needs one frame. It selects the decoder:

* below one frame and below 35 ms: the **timing decoder**, which places the expected number of spikes at the peaks of the
  within-frame output, smoothed by min(0.65 x tolerance, 9 ms) (4.5 ms above 30 Hz);
* otherwise: the **detection decoder**, which finds events from the count output and places each spike at its frame's
  within-frame peak.

The rule was fitted by cross-validation on held-out neurons over every indicator, frame rate, noise level and tolerance
from 1 to 25 ms. Decode with the tolerance of the analysis you will run; decoding once at 1 ms and reusing the spikes
for a 25-ms analysis loses accuracy. `tolerance_ms=None` returns rates only. `count_gain` (default 1) scales the
number of decoded spikes, trading precision for recall.

## 7. Reading the output

* **Rate** is in expected spikes per frame. Absolute counts depend on how large a single-spike transient is in your
  preparation, so compare counts within rather than across preparations.
* **Spike times** are measured from the time stamp of the first frame. A scanning microscope reaches each neuron later
  than the frame's time stamp; for timing below a few milliseconds, add each neuron's scan-line delay (the time from the
  start of the frame to the line through the neuron) to its spike times.
* **Within-frame timing has limits.** It needs enough signal in each frame. At the lowest signal-to-noise ratios and at
  1 Hz, placement within the frame can fall to chance, and isolated spikes at the lowest frame rates are too rare to be
  placed reliably. Detection and rates remain informative there.
* `fine=True` returns the 512-bin count and within-frame streams for each neuron.

## 8. Frame rates and speed

Each indicator has three networks for overlapping frame-rate bands (about 1-10, 8-60 and 40-120 Hz; jRGECO1a to
25 Hz); SETSUNA picks the one for your frame rate and averages three independently trained copies. On one recent NVIDIA
GPU, an hour of recording from 10,000 neurons takes one to two minutes through the JupiterScope pipeline, which batches
neurons; `setsuna.infer` processes one neuron at a time and is slower. Lower `batch_windows` if the GPU runs out of memory.

## 9. Citing

Hattori, R. Breaking the frame-rate limit of calcium imaging (2026).
