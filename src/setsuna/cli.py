"""Command line:  setsuna run --fps 30 --indicator jGCaMP8s --dff traces.npy --out spikes.npz

Input (n_neurons x n_frames), one of
  --dff FILE            fractional dF/F (.npy, .csv, or .npz with key 'dff')
  --F FILE [--Fneu FILE]  raw and neuropil fluorescence (.npy, .csv); dF/F is computed as in setsuna.dff_from_fluorescence
  --npz FILE            .npz with keys 'F' and optionally 'Fneu' (e.g. converted from Suite2p's F.npy and Fneu.npy)
Output (.npz): rate (n_neurons x n_frames, expected spikes per frame), nu, and the spike times both as
spike_neuron / spike_time (flat arrays) and per neuron. `setsuna models` lists the installed checkpoints.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _load(path: str, key: str | None = None) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        z = np.load(p)
        return z[key] if key else z[z.files[0]]
    if p.suffix in (".csv", ".txt"):
        return np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None, ndmin=2)
    raise SystemExit(f"unsupported file type: {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="setsuna", description="SETSUNA spike inference")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="infer spikes")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--dff"); src.add_argument("--F"); src.add_argument("--npz")
    r.add_argument("--Fneu")
    r.add_argument("--fps", type=float, required=True, help="frame rate (Hz) of the recording")
    r.add_argument("--indicator", default="jGCaMP8s", help="jGCaMP8s, jGCaMP8m, jGCaMP8f, jGCaMP7f, GCaMP6s, GCaMP6f, "
                   "jRGECO1a, or other")
    r.add_argument("--cell-type", default="pyramidal", help="pyramidal, PV_FS, SST, VIP, spinal_excitatory, "
                   "spinal_inhibitory, unknown, or other")
    r.add_argument("--tolerance-ms", type=float, default=10.0, help="spike-timing precision the analysis needs (ms)")
    r.add_argument("--no-spikes", action="store_true", help="rates only")
    r.add_argument("--neuropil-coefficient", type=float, default=0.7)
    r.add_argument("--model-dir", default=None)
    r.add_argument("--device", default="auto")
    r.add_argument("--out", required=True)
    m = sub.add_parser("models", help="list installed checkpoints")
    m.add_argument("--model-dir", default=None)
    a = ap.parse_args(argv)

    import setsuna
    if a.cmd == "models":
        from setsuna import engine
        d = Path(a.model_dir) if a.model_dir else engine.default_model_dir()
        files = sorted(d.glob("*.pt")) if d.is_dir() else []
        print(f"{d}: {len(files)} checkpoints")
        return 0 if files else 1

    if a.dff:
        dff = _load(a.dff, "dff")
    else:
        if a.npz:
            z = np.load(a.npz); F = z["F"]; Fneu = z["Fneu"] if "Fneu" in z.files else None
        else:
            F = _load(a.F); Fneu = _load(a.Fneu) if a.Fneu else None
        dff = setsuna.dff_from_fluorescence(F, Fneu, fps=a.fps, neuropil_coefficient=a.neuropil_coefficient)
    t0 = time.time()
    out = setsuna.infer(dff, a.fps, a.indicator, a.cell_type, None if a.no_spikes else a.tolerance_ms,
                        model_dir=a.model_dir, device=a.device, progress=True)
    save = {"rate": out["rate"], "nu": out["nu"], "fps": a.fps, "indicator": a.indicator, "cell_type": a.cell_type,
            "band": out["band"]}
    if not a.no_spikes:
        st = out["spike_times"]
        save["spike_neuron"] = np.concatenate([np.full(len(s), i, np.int32) for i, s in enumerate(st)]) if st else np.zeros(0, np.int32)
        save["spike_time"] = np.concatenate(st) if st else np.zeros(0)
        save["tolerance_ms"] = a.tolerance_ms
    np.savez_compressed(a.out, **save)
    n_sp = len(save.get("spike_time", []))
    print(f"{dff.shape[0]} neurons, {dff.shape[1]} frames -> {a.out} ({n_sp} spikes; {time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
