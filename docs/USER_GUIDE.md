# SETSUNA user guide

SETSUNA infers spikes from calcium-imaging traces on a time grid 512 times finer than the imaging frame. One network
design serves frame rates from 1 to 120 Hz and the noise levels of real experiments, and generalizes across indicators and
cell types, including those without ground truth in training; it is told the frame rate, the noise level, the indicator
and the cell type of each recording, and returns

* a **rate**: the expected number of spikes in every frame (or in every 1/512 of a frame), optionally smoothed on a
  time grid finer than the frame, and
* **spike times**, decoded at the timing precision your analysis needs.

## Which output to use

* **Firing-rate analyses** (tuning, correlations, trial-averaged responses, population activity): the smoothed
  sub-frame rate, `out["rate_smooth"]` from `infer(..., smooth_ms=25)` (section 5).
* **Timing analyses** (when spikes occur, response onsets and latencies, synchrony between neurons, timing within the
  frame): the **spike times**, `out["spike_times"]`, with the tolerance set to the precision the analysis needs
  (section 7).
* **Frame-level analyses** (anything binned at the imaging frame): the per-frame `rate`.

## 1. Install

SETSUNA needs Python 3.10 or later, PyTorch 2.2 or later, NumPy, SciPy and Numba. It runs on an NVIDIA GPU or, much
more slowly, on the CPU.

**GPU setup (recommended).**

1. Install a recent NVIDIA driver (`nvidia-smi` should list your GPU).
2. In a fresh environment, install PyTorch built for CUDA before SETSUNA. The CUDA runtime comes with PyTorch; no
   separate CUDA toolkit is needed. For current GPUs, including the RTX 50 series, use the CUDA 12.8 build:

       conda create -n setsuna python=3.11
       conda activate setsuna
       pip install torch --index-url https://download.pytorch.org/whl/cu128

   (pytorch.org lists the command for other CUDA versions; the RTX 50 series needs CUDA 12.8 or later.)
3. Check that PyTorch sees the GPU:

       python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Then install SETSUNA:

    pip install git+https://github.com/ryhattori/setsuna.git

At the default batch size SETSUNA uses about 6-8 GB of GPU memory; pass `batch_windows=256` (or lower) to
`setsuna.infer` on a smaller GPU. Without a GPU, `device="cpu"` runs the same models.

**Tested on** Windows 11 with Python 3.11, PyTorch 2.11 (CUDA 12.8), NumPy 2.0-2.4, SciPy 1.17 and Numba 0.65-0.68,
on an NVIDIA GeForce RTX 4090 and on the CPU. No hardware beyond a standard computer is required; an NVIDIA GPU makes
large recordings fast. A fresh install took about 2 minutes on a university network (PyTorch with CUDA about 100 s,
SETSUNA and its other dependencies about 30 s).

## 2. Get the models

The trained models are published as an archive of a release of this repository (release v0.9.0 for version 0.9.0 of
the package, about 64 MB). They are downloaded automatically the first time you run SETSUNA. To install them
beforehand, for example on a cluster node without internet access, run

    setsuna download-models

which downloads the archive, checks it against the checksum built into the package and unpacks it into
`~/.setsuna/models` (`--model-dir` or `SETSUNA_MODEL_DIR` to put them elsewhere; set `SETSUNA_NO_DOWNLOAD=1` to turn
off the automatic download). When `SETSUNA_MODEL_DIR` is set, SETSUNA uses that folder as it is and does not download
into it; fill it with `setsuna download-models --model-dir <folder>`. Check with

    setsuna models

JupiterScope's model download installs the same release.

## 3. Demo

`examples/demo.ipynb` (a Jupyter notebook, with plots; `pip install matplotlib jupyter`) and `examples/demo.py` (the
same steps as a script) run SETSUNA on one jGCaMP8s neuron whose spikes were recorded electrically at the same time
(`examples/jGCaMP8s_ANM479572_cell08.npz`, 0.2 MB: raw and neuropil fluorescence, frame times and spike times from
DANDI:000168, CC BY 4.0; see `examples/DATA_LICENSE.md`). Both resample the 122-Hz recording to 30.5 Hz, computes dF/F
with `dff_from_fluorescence`, infers spikes at a 25-ms tolerance and compares them with the recorded spikes:

    cd examples
    jupyter notebook demo.ipynb      # or: python demo.py (GPU if available), python demo.py --cpu

Expected output (version 0.9.0):

    recording        : 170 s at 30.5 Hz, 127 recorded spikes
    SETSUNA rate     : 121.4 expected spikes in total (sum of out['rate'])
    SETSUNA spikes   : 121 inferred spike times (tolerance 25 ms)
    within +- 25 ms  : 95 of 127 recorded spikes matched (75%)
    within +-100 ms  : 105 of 127 recorded spikes matched (83%)
    run time         : ...

It runs in about 5 s on a GPU or a CPU, including loading the models (the models are downloaded first if they are not
installed). The neuron is part of the public ground truth SETSUNA was trained on, so the demo shows how to run the
software and what it returns, not held-out accuracy.

## 4. Prepare the input

SETSUNA reads **fractional dF/F** (0.5 means a 50% increase), one row per neuron, sampled at the frame rate.

If you have raw fluorescence and neuropil traces (for example Suite2p's `F.npy` and `Fneu.npy`), compute dF/F with
SETSUNA's function:

```python
import numpy as np, setsuna
F, Fneu = np.load("F.npy"), np.load("Fneu.npy")
iscell = np.load("iscell.npy")[:, 0].astype(bool)
dff = setsuna.dff_from_fluorescence(F[iscell], Fneu[iscell], fps=30.0, neuropil_coefficient=0.7)
```

The baseline is a maximin baseline over 60 s, which follows slow drift and stays under calcium transients whatever a
neuron's firing; the neuropil-corrected trace is divided by the raw fluorescence baseline, which stays positive even
for dim cells next to bright neuropil. If you already have dF/F from your own pipeline, use it directly, as long as it
is fractional and not z-scored.

**The size of dF/F matters for counts, not for timing.** SETSUNA standardizes each trace itself, but it reads the noise
level from the fractional dF/F, and the models were trained on dF/F of a typical size. Dividing by the
neuropil-corrected baseline instead of the raw one (a common convention) makes dF/F 2-4 times larger. On recordings
with simultaneous electrophysiology, doubling or halving dF/F changed the median correlation of the rate with the
recorded spikes by 0.03 or less, but scaled the number of inferred spikes by about 1.2-1.6 (doubling) or 0.7-0.9
(halving). Use one dF/F convention for
recordings whose rates you compare. Traces whose noise level is far below anything in training (dF/F several times
too small) are run as if at a floor of the training range (`setsuna.engine.NU_GUARD_K`), which prevents the large
over-counts such traces otherwise produce.

**Frame rate.** Use the rate at which each neuron is sampled: for multi-plane imaging, the volume rate, not the
line or plane rate.

## 5. Run

```python
out = setsuna.infer(dff, fps=30.0, indicator="jGCaMP8s", cell_type="pyramidal", tolerance_ms=10)
out["rate"]          # (n_neurons, n_frames) expected spikes per frame; x fps = spikes per second
out["spike_times"]   # list of arrays, spike times in seconds from the first frame
out["nu"]            # noise level of each trace (the standardized noise level used by CASCADE)
out["rate_smooth"]   # (n_neurons, n_bins) spikes per second on a sub-frame grid, Gaussian-smoothed (sigma 25 ms)
```

or from the command line:

    setsuna run --npz traces.npz --fps 30 --indicator jGCaMP8s --tolerance-ms 10 --out spikes.npz

(`traces.npz` holds `F` and optionally `Fneu`; `--dff file.npy` takes dF/F directly.) The output file holds `rate`,
`nu`, `rate_smooth` and `t_smooth`, and the spike times as `spike_neuron` and `spike_time`.

**Rates only.** `tolerance_ms=None` (command line: `--no-spikes`) skips the spike decoder and returns `rate` and `nu`.

**Smoothed sub-frame rate (recommended for firing-rate analyses).** `rate` is one value per frame, so on a time axis
it is a staircase. For firing-rate analyses, plots and trial averages, pass `smooth_ms=25` (command line:
`--smooth-ms 25`): `out["rate_smooth"]` is then the model's sub-frame count stream (the count head's 512 bins per frame, sum-pooled to k bins per frame, the
smallest power of two that makes a bin at most a quarter of the smoothing width) smoothed with a Gaussian of sigma
`smooth_ms`, in spikes/s; `out["t_smooth"]` holds the bin centres in seconds from the first frame. `rate`, `spike_times`
and the other outputs are returned as usual, and frame totals equal `rate`. We use 25 ms at every frame rate. The count
stream is nearly flat within each frame, so for the timing of single spikes use `spike_times`. In particular, for the
onset or latency of a response at low frame rates, average spike times across trials rather than `rate_smooth`, whose
rise starts early by up to a frame.

Smoothing is off by default only because of size: `rate_smooth` is k times larger than `rate` (k = 8 at 30 Hz,
128 at 2 Hz). For very large runs, pass `rate_smooth_path="rate_smooth.npy"` to write it to disk instead of memory.

`setsuna.smooth_rate` smooths a sub-frame count stream you already have, for any neurons and time window:

```python
out = setsuna.infer(dff, fps=30.0, bins_per_frame=16)          # the sub-frame count stream: out["count"]
r = setsuna.smooth_rate(out["count"], fps=30.0, bins_per_frame=16, smooth_ms=25,
                        neurons=[0, 5],              # which neurons (rows); omit for all
                        start_s=60, end_s=120)       # time window in seconds from the first frame; omit for all
r["rate_smooth"], r["t_smooth"]                      # spikes/s and bin times (s) for that subset only
```

The values are those of the whole recording cut to the window (the edges are smoothed with the data around them).

**Sub-frame streams.** `bins_per_frame=16` (command line: `--bins-per-frame 16`) also returns the count stream
(`count`, expected spikes per bin), the within-frame stream (`prob`) and, with a tolerance, the decoded spikes per bin
(`spike_counts`), each n_neurons x (n_frames x 16), sum-pooled from the model's 512 bins per frame. Sum-pooling keeps
totals, so a stream saved at frame/16 can be pooled further to any coarser bin; frame/16 is about as fine as the
models' timing supports below 60 Hz. This runs one neuron at a time and is slower than the default.

For many recordings at the same frame rate and indicator, load the models once:

```python
sess = setsuna.Session(fps=30.0, indicator="jGCaMP8s")
for trace in traces:
    o = sess.infer_one(trace, tolerance_ms=10)
```

## 6. Choosing the indicator and cell type

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

## 7. Choosing the timing tolerance

`tolerance_ms` is the precision with which your analysis needs spike times: a spike-time correlation at 5 ms needs
5 ms; counting spikes per frame needs one frame. You only choose the tolerance (default 10 ms); `infer` applies the
rule below itself, from the tolerance and the frame rate:

* below one frame and below 35 ms: the **timing decoder**, which places the expected number of spikes at the peaks of the
  within-frame output, smoothed by min(0.65 x tolerance, 9 ms) (4.5 ms above 30 Hz);
* otherwise: the **detection decoder**, which finds events from the count output and places each spike at its frame's
  within-frame peak.

The rule was fitted by cross-validation on held-out neurons over every indicator, frame rate, noise level and tolerance
from 1 to 25 ms. Decode with the tolerance of the analysis you will run; decoding once at 1 ms and reusing the spikes
for a 25-ms analysis loses accuracy. `tolerance_ms=None` returns rates only. `count_gain` (default 1) scales the
number of decoded spikes, trading precision for recall.

## 8. Reading the output

* **Rate** is in expected spikes per frame. Absolute counts depend on how large a single-spike transient is in your
  preparation, so compare counts within rather than across preparations.
* **Spike times** are measured from the time stamp of the first frame. A scanning microscope reaches each neuron later
  than the frame's time stamp; for timing below a few milliseconds, add each neuron's scan-line delay (the time from the
  start of the frame to the line through the neuron) to its spike times.
* **Within-frame timing needs enough signal in each frame.** In very noisy recordings, placement within the frame can
  fall to chance; detection and rates remain informative there.
* `fine=True` returns the full 512-bin count and within-frame streams for each neuron; `bins_per_frame` returns them
  pooled (section 5).

## 9. Frame rates and speed

Each indicator has three networks for overlapping frame-rate bands (about 1-10, 8-60 and 40-120 Hz; jRGECO1a to
25 Hz); SETSUNA picks the one for your frame rate and averages three independently trained copies. On a GPU,
`setsuna.infer` pipelines the neurons (one neuron's network pass runs while the previous neuron is decoded): on one
recent NVIDIA GPU an hour of recording from 10,000 neurons takes about a minute or two, and an hour of 3,000 mesoscope
neurons at 10.7 Hz about 17 s. Lower `batch_windows` if the GPU runs out of memory.

## 10. Citing

Hattori, R. Breaking the frame-rate limit of calcium imaging (2026).
