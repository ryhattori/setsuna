"""SETSUNA demo: one jGCaMP8s neuron with simultaneously recorded spikes.

Runs SETSUNA on a public ground-truth recording (examples/jGCaMP8s_ANM479572_cell08.npz; DANDI:000168, CC BY 4.0,
see examples/DATA_LICENSE.md), imaged at 122 Hz and resampled here to 30.5 Hz by keeping every fourth frame, and
compares the inferred spikes with the spikes recorded electrically. The neuron is part of the public ground truth that
SETSUNA was trained on, so this demo shows how to run the software and what it returns, not held-out accuracy.

    python examples/demo.py              # GPU if available, else CPU
    python examples/demo.py --cpu
"""
import argparse
import os
import time

import numpy as np

import setsuna


def matched(true_t, inferred_t, tol_s):
    """Recorded spikes with an inferred spike within +-tol_s, one-to-one, in time order."""
    used = np.zeros(len(inferred_t), bool)
    n = 0
    for t in true_t:
        j = np.flatnonzero(~used & (np.abs(inferred_t - t) <= tol_s))
        if j.size:
            used[j[np.argmin(np.abs(inferred_t[j] - t))]] = True
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cpu", action="store_true", help="run on the CPU")
    a = ap.parse_args()
    d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jGCaMP8s_ANM479572_cell08.npz"))
    step = 4                                                    # 122 Hz -> 30.5 Hz
    F, Fneu, t = d["F"][::step], d["Fneu"][::step], d["frame_times"][::step]
    fps = float(d["fps"]) / step
    spikes = d["spike_times"]
    spikes = spikes[(spikes >= t[0]) & (spikes < t[-1] + 1.0 / fps)] - t[0]

    t0 = time.time()
    dff = setsuna.dff_from_fluorescence(F[None], Fneu[None], fps=fps)
    out = setsuna.infer(dff, fps=fps, indicator="jGCaMP8s", cell_type="pyramidal", tolerance_ms=25, smooth_ms=25,
                        device="cpu" if a.cpu else "auto")
    run_s = time.time() - t0

    inferred = out["spike_times"][0]
    rate = out["rate"][0]
    print("recording        : %.0f s at %.1f Hz, %d recorded spikes" % (len(t) / fps, fps, len(spikes)))
    print("SETSUNA rate     : %.1f expected spikes in total (sum of out['rate'])" % rate.sum())
    print("SETSUNA spikes   : %d inferred spike times (tolerance 25 ms)" % len(inferred))
    for tol in (0.025, 0.1):
        m = matched(spikes, inferred, tol)
        print("within +-%3d ms  : %d of %d recorded spikes matched (%.0f%%)" % (tol * 1000, m, len(spikes),
                                                                          100.0 * m / max(len(spikes), 1)))
    # firing rate: the smoothed sub-frame rate against the recorded spikes smoothed with the same 25-ms Gaussian
    from scipy.ndimage import gaussian_filter1d
    ts, rsm = out["t_smooth"], out["rate_smooth"][0]
    dt = ts[1] - ts[0]
    true_sm = gaussian_filter1d(np.histogram(spikes, np.append(ts - dt / 2, ts[-1] + dt / 2))[0] / dt, 0.025 / dt)
    print("smoothed rate    : correlation %.3f with the recorded spikes (both 25-ms Gaussian)"
          % np.corrcoef(rsm, true_sm)[0, 1])
    print("run time         : %.1f s (including loading the models)" % run_s)


if __name__ == "__main__":
    main()
