"""SETSUNA spike inference for JupiterScope -- the sym24 deployment models.

Engine only (no Qt). Wraps the vendored BinarySRHybrid checkpoints:

  count head  -> per-fine-bin expected spike count (lambda; calibrated, sums are counts)
  binary head -> per-fine-bin spike probability
  both at m*fps resolution with m=512 fine bins per imaging frame.

The overlap-add inference loop is ported from spikeinf `eval_binary_sr.py` (:383-404),
which is the path every published number was scored through. It includes the tail-flush
fix (the older `predict_stream` in `synaptic_timing.py` silently zeroes the last frames
whenever (n - W) is not a multiple of W//2, and whole traces shorter than one window).

Checkpoint sets (in the folder default_model_dir() finds: $JUPITERSCOPE_SETSUNA_MODEL_DIR,
<runtime>/SETSUNA where the GUI's "Get SETSUNA models" installs them, or Pretrained_models/SETSUNA/):
  specialists : bsr_FOftDEPLOYSYM24C167_<IND>_<BAND>_s{1,2,3}.pt   (per indicator; GT-known)
  generalist  : bsr_GTUftDEPLOYSYM24C167_ALL_<BAND>_s{1,2,3}.pt    (indicator-agnostic set)
  interneurons: bsr_INabSYM24C167FIN_allin_<REGIME>_<BAND>_f0_s{1,2,3}.pt   (named indicators)
                bsr_INabSYM24C167GTU_allin_ALL_<BAND>_f0_s{1,2,3}.pt       ('Other' indicator)
  spinal cord : bsr_SPabC167_pooled_s1sp_GCaMP6s_<BAND>_fD_s{1,2,3}.pt      (excitatory + inhibitory)
  other / not-listed cell types: bsr_INabALLCTSYM24C167_<SET>_<BAND>_s{1,2,3}.pt  (all six cell types; SET =
                GCaMP6f, GCaMP8s, GCaMP6s or ALL; optional -- without them 'other' runs the pyramidal sets)

The pyramidal sets were trained pyramidal-only (cell_type id 0). Passing another cell-type id
conditions on an untrained embedding row -- it runs and returns garbage. The interneuron sets
are trained separately (PV/SST/VIP rows are trained there); do not point this module at
pyramidal weights for fast-spiking cells. The spinal set trains rows 4/5 (spinal excitatory /
inhibitory) and is the only family that may receive them. Two cell types are not named:
'unknown' (probably pyramidal) runs exactly the pyramidal models (row 0), and 'other' (a type not in
the list: striatal, midbrain, cerebellar, olfactory bulb ...) the all-cell-types set on its unknown
cell-type token (row 7), never a named row.
The C167 checkpoints trained with the blanking options (--ind_unknown_id 13 / --ct_unknown_id 7: the
generalist, the interneuron GT-unknown set, spinal, all-cell-types) carry 14-row indicator / 8-row
cell-type tables whose last row is a TRAINED unknown token; models are built with each checkpoint's own
table sizes and the zero-row surgeries leave such tables alone (same ids 13 / 7). 13 / 7-row files (the
specialists, the interneuron FIN set, older folders) still take an appended zero row.
Reported counts of the pyramidal families can be divided by a photon-noise calibration
(photon_calibration.py, applied by the caller to the reported streams only; see count_calibration_applies).

User guide: docs/USER_GUIDE.md in the SETSUNA repository.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

_PKG_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _PKG_DIR / "_vendor"
# The vendored spikeinf modules use flat absolute imports of each other
# (`from models_small_conformer_corner import ...`), so they must be importable as
# top-level modules.
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models_binary_sr import BinarySRHybrid  # noqa: E402  (vendored)
import decode_readout as _decode  # noqa: E402  (vendored: struct_decode)
import route_readout as _route  # noqa: E402  (vendored: decoder_for, choose_arm, bin_ms_for)
from . import _decode_cuda  # noqa: E402  (the timing decoder's pursuit kernel)

# The spike-time tolerance (ms) an analysis scores at picks the decoder and its width (route_readout.decoder_for).
DEFAULT_TOL_MS = float(_route.DEFAULT_TOL_MS)


# --------------------------------------------------------------------------- constants
# Mirrors spikeinf train_v3.py (verified 2026-08-26). The embedding table has 13 rows;
# all 13 were trained in the broad stage-1 the deployment models warm-start from.
INDICATOR_TO_ID = {
    "Cal520": 0, "GCaMP5k": 1, "GCaMP6f": 2, "GCaMP6s": 3, "GCaMP7f": 4,
    "GCaMP8f": 5, "GCaMP8m": 6, "GCaMP8s": 7, "OGB1": 8, "R-CaMP": 9,
    "XCaMPgf": 10, "jGECO1a": 11, "jRCaMP1a": 12,
}
NUM_INDICATORS = 13
NUM_CELL_TYPES = 7          # current embedding scheme; pyramidal=0 is valid under all schemes
CELLTYPE_TO_ID = {"pyramidal": 0, "PV_FS": 1, "SST": 2, "VIP": 3,
                  "spinal_excitatory": 4, "spinal_inhibitory": 5,   # = spikeinf train_mres CT ids
                  "unknown": 0,                # probably pyramidal: the pyramidal models, row 0, unchanged
                  "other": NUM_CELL_TYPES}     # not listed: the all-cell-types set's unknown token (row 7)
# Display label -> key, and the types the allin interneuron set serves: in the torch-free count_gain.py, beside
# the count-gain rule that reads them.
from .count_gain import (CELLTYPE_FROM_DISPLAY, INTERNEURON_TYPES, OTHER_CELL_TYPE,  # noqa: E402,F401
                         PYRAMIDAL_TYPES, SPINAL_TYPES, UNKNOWN_CELL_TYPE)
CELLTYPE_LABEL = {"pyramidal": "pyramidal", "PV_FS": "PV-FS", "SST": "SST", "VIP": "VIP",
                  "spinal_excitatory": "spinal excitatory", "spinal_inhibitory": "spinal inhibitory",
                  "unknown": "cell type unknown (probably pyramidal)",
                  "other": "cell type not listed"}

# Display names for GUIs. The red indicator's PROPER name is jRGECO1a; its internal key
# is the historical `jGECO1a`, baked into checkpoint filenames and the embedding map.
# Show the display name, never rename the internal key.
DISPLAY_NAME = {k: k for k in INDICATOR_TO_ID}
DISPLAY_NAME["jGECO1a"] = "jRGECO1a"
DISPLAY_NAME["other"] = "Other / novel (zero identity)"
INTERNAL_FROM_DISPLAY = {v: k for k, v in DISPLAY_NAME.items()}
# Accept common j-prefixed display spellings for the 8-series (jGCaMP8f == GCaMP8f).
for _k in ("GCaMP8f", "GCaMP8m", "GCaMP8s"):
    INTERNAL_FROM_DISPLAY["j" + _k] = _k
DISPLAY_NAME["GCaMP8f"], DISPLAY_NAME["GCaMP8m"], DISPLAY_NAME["GCaMP8s"] = (
    "jGCaMP8f", "jGCaMP8m", "jGCaMP8s")

# The seven indicators with GT-known specialist sets. jGECO1a has LOW/MID only.
CORE_INDICATORS = ("GCaMP6f", "GCaMP6s", "GCaMP7f", "GCaMP8f", "GCaMP8m", "GCaMP8s", "jGECO1a")
BANDS = ("LOW", "MID", "HIGH")
SEEDS = (1, 2, 3)

NU_K = 0.6745 * math.sqrt(2.0) * 100.0   # nu = NU_K * sigma / sqrt(fps)  (train_mres.py:220)

PV_FILE_FMT = "bsr_INabSYM24C167FIN_allin_{ind}_{band}_f0_s{seed}.pt"
PV_FLAVORS = ("GCaMP6f", "GCaMP8s")   # filename slots = ACQUISITION REGIMES, not specialisations
PV_GEN_FILE_FMT = "bsr_INabSYM24C168GTU_allin_ALL_{band}_f0_s{seed}.pt"
SP_FILE_FMT = "bsr_SPabC168_pooled_s1sp_GCaMP6s_{band}_fD_s{seed}.pt"
ALLCT_FILE_FMT = "bsr_INabALLCTSYM24C168_{ind}_{band}_s{seed}.pt"
ALLCT_INDICATORS = ("GCaMP6f", "GCaMP8s", "GCaMP6s")   # + "ALL" for 'Other' (or model_set 'generalist')

# Pyramidal / generalist band grids are NOT a constant: they differ per indicator (6s MID is
# 3-35 Hz, 7f MID 8-60 Hz) and are read from each checkpoint's own saved args at routing time
# (band_grids). The provenance string reports the grid of the band actually selected.
SPEC_FILE_FMT = "bsr_FOftDEPLOYSYM24C167_{ind}_{band}_s{seed}.pt"
GEN_FILE_FMT = "bsr_GTUftDEPLOYSYM24C168_ALL_{band}_s{seed}.pt"


#: Environment variable naming a folder that holds the SETSUNA checkpoints (it wins when set).
MODEL_DIR_ENV = "SETSUNA_MODEL_DIR"


def runtime_model_dir() -> Path:
    """<runtime>/SETSUNA, where the GUI's "Get SETSUNA models" installs the checkpoints. The
    runtime folder is the GUI's (user_folders.runtime_dir)."""
    return Path(os.environ.get("SETSUNA_HOME", Path.home() / ".setsuna")).expanduser() / "models"


def model_dir_candidates() -> list[Path]:
    """The folders searched for the checkpoints, in order: $JUPITERSCOPE_SETSUNA_MODEL_DIR, the
    runtime folder, and Pretrained_models/SETSUNA at the repo root (source checkouts made before
    the runtime folder was used)."""
    out = []
    env = os.environ.get(MODEL_DIR_ENV, "").strip()
    if env:
        out.append(Path(env).expanduser())
    out.append(runtime_model_dir())
    out.append(_PKG_DIR.parent.parent / "models")
    return out


#: The model families every installed set holds in full (each band x seed): the generalist, the interneurons'
#: generalist and the spinal cord set. The specialist sets differ by indicator, so they are not counted.
REQUIRED_FAMILIES = (GEN_FILE_FMT, PV_GEN_FILE_FMT, SP_FILE_FMT)


def required_model_names() -> list:
    """The checkpoint files a complete install holds (REQUIRED_FAMILIES)."""
    return [fmt.format(band=b, seed=s) for fmt in REQUIRED_FAMILIES for b in BANDS for s in SEEDS]


def generalist_installed(model_dir: Path) -> bool:
    """True when ``model_dir`` holds the generalist set, which the pyramidal and interneuron cell types can fall
    back to: enough to run, though a set from before the spinal cord models lacks those."""
    return all((Path(model_dir) / GEN_FILE_FMT.format(band=b, seed=s)).is_file()
               for b in BANDS for s in SEEDS)


def models_installed(model_dir: Path) -> bool:
    """True when ``model_dir`` holds the current model set in full (required_model_names): a folder installed
    before a family was added (the spinal cord models, 2026-10-12) is not complete."""
    return all((Path(model_dir) / name).is_file() for name in required_model_names())


#: Every family's file name format. Each carries the edition of the training run that produced it: the families
#: retrained for this release are C168 (no indicator or cell-type dropout; an unseen identity is an appended zero
#: row), the pyramidal specialists and the interneuron models with ground truth are unchanged from C167.
FILE_FMTS = (SPEC_FILE_FMT, GEN_FILE_FMT, PV_FILE_FMT, PV_GEN_FILE_FMT, SP_FILE_FMT, ALLCT_FILE_FMT)


def _edition(fmt: str) -> str:
    return re.search(r"C\d{3}(?!\d)", fmt).group(0)


EDITION_TAG = max((_edition(f) for f in FILE_FMTS), key=lambda t: int(t[1:]))   # the release edition: "C168"


def _earlier_edition_pattern(fmt: str) -> "re.Pattern":
    """A family's file name with any edition in place of its own (group 1: that edition's number)."""
    parts = re.split(r"\{\w+\}", fmt)
    body = r"[A-Za-z0-9]+".join(re.escape(p).replace(_edition(fmt), r"C(\d{3})") for p in parts)
    return re.compile(body)


def previous_release_files(model_dir) -> list:
    """The checkpoints in ``model_dir`` that an earlier release installed: a current family's file name with an
    earlier edition (bsr_GTUftDEPLOYSYM24C167_ALL_MID_s1.pt beside the C168 one), which this JupiterScope never
    loads. Anything else -- a fine-tuned or other checkpoint, a family no longer shipped -- is not one."""
    patterns = [(_earlier_edition_pattern(fmt), int(_edition(fmt)[1:])) for fmt in FILE_FMTS]
    folder = Path(model_dir)
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        for pattern, current in patterns:
            match = pattern.fullmatch(path.name)
            if match and int(match.group(1)) < current:
                out.append(path)
                break
    return out


def default_model_dir() -> Path:
    """The folder the checkpoints are loaded from: $JUPITERSCOPE_SETSUNA_MODEL_DIR when set, else
    the first searched folder that holds them in full, else the first that holds the generalist set (an older
    install still runs), else the runtime folder (where they get installed, so a missing-model error names the
    place to put them)."""
    candidates = model_dir_candidates()
    if os.environ.get(MODEL_DIR_ENV, "").strip():
        return candidates[0]
    return next((d for d in candidates if models_installed(d)),
                next((d for d in candidates if generalist_installed(d)), candidates[0]))


def missing_models_hint(model_dir) -> str:
    """How to get the checkpoints, appended to every missing-checkpoint error."""
    return (f"SETSUNA's models are not installed in {model_dir}. Download them (see the README) into "
            f"~/.setsuna/models, or set {MODEL_DIR_ENV} to a folder that holds them.")


# ------------------------------------------------------------------- trace preparation
def standardize_dff(dff: np.ndarray) -> np.ndarray:
    """Per-recording z-score of dF/F (verbatim port of spikeinf pipeline.standardize_dff)."""
    m = float(np.nanmedian(dff))
    s = float(np.nanstd(dff)) + 1e-6
    return ((dff - m) / s).astype(np.float32)


def prep_stats_many(rows, fps: float) -> list:
    """Per row of the 2-D float32 array ``rows``: (noise_level, the median and the std + 1e-6 that standardize_dff
    takes), the same floats -- numpy's reductions along each row of the block (np.median, np.nanstd with axis=1)
    give each row's 1-D values bit for bit; None for a row with a non-finite value (its own functions then)."""
    blk = np.ascontiguousarray(rows, dtype=np.float32)   # (a strided block reduces its rows in another order)
    if blk.ndim != 2 or blk.shape[1] < 2:
        return [None] * (int(blk.shape[0]) if blk.ndim else 0)
    nu = np.median(np.abs(np.diff(blk, axis=1)), axis=1) / np.sqrt(fps) * 100.0
    m = np.median(blk, axis=1)
    sd = np.nanstd(blk, axis=1)
    fin = np.isfinite(blk).all(axis=1)
    return [(float(nu[i]), float(m[i]), float(sd[i]) + 1e-6) if fin[i] else None for i in range(blk.shape[0])]


#: Lowest noise level of the training ladder (first rung, nu) per indicator and frame rate. Traces cleaner than this
#: were always brought up to it with added noise in training, so the models never saw a lower nu.
NU_TRAIN_FLOOR = {
    "GCaMP8f": {1: 26, 2: 19, 3: 16, 5: 12, 8: 9.5, 10: 8.5, 13: 7.5, 15: 7, 20: 6, 25: 5.5, 30: 5, 40: 4.5, 60: 3.5, 120: 2.5},
    "GCaMP8m": {1: 20, 2: 14, 3: 12, 5: 9, 8: 7, 10: 6.5, 13: 5.5, 15: 5, 20: 4.5, 25: 4, 30: 3.5, 40: 3.5, 60: 2.5, 120: 2},
    "GCaMP8s": {1: 21, 2: 15, 3: 12, 5: 9.5, 8: 7.5, 10: 6.5, 13: 6, 15: 5.5, 20: 5, 25: 4.5, 30: 4, 40: 3.5, 60: 3, 120: 2},
    "GCaMP7f": {1: 50, 2: 34, 3: 28, 5: 22, 8: 17, 10: 16, 13: 14, 15: 13, 20: 11, 25: 10, 30: 9, 40: 8, 60: 6.5, 120: 4.5},
    "GCaMP6f": {1: 7.5, 2: 5.5, 3: 4.5, 5: 3.5, 8: 3, 10: 2.5, 13: 2.5, 15: 2, 20: 2, 25: 1.5, 30: 1.5, 40: 1.5, 50: 1.5, 60: 1, 80: 1, 160: 1},
    "GCaMP6s": {1: 10, 2: 7, 3: 6, 5: 4.5, 8: 3.5, 10: 3.5, 13: 3, 15: 2.5, 20: 2.5, 25: 2, 30: 2, 40: 2, 50: 1.5, 60: 1.5, 80: 1.5, 160: 1},
    "jGECO1a": {1: 7.5, 2: 5.5, 3: 4.5, 5: 3.5, 8: 3, 10: 2.5, 15: 2, 30: 1.5},
}
#: The models are conditioned on max(nu, NU_GUARD_K * training floor). Far below the floor (dF/F much smaller than the
#: training data's, e.g. from a different baseline convention) a model can read noise as spikes: counts several-fold
#: too high and correlation near zero on single neurons. Measured on GENIE jGCaMP8 and Wei GCaMP6 recordings at 10 and
#: 30 Hz: 0.25 removes those failures and leaves traces at the training data's scale unchanged; 0.5 starts to move
#: normal traces. No noise is added; only the conditioning value is bounded.
NU_GUARD_K = 0.25


def nu_guard_floor(indicator: str, fps: float) -> float:
    """NU_GUARD_K x the training floor for ``indicator`` at ``fps`` (log-log interpolation, clamped at the ends); for an
    indicator outside the table ('other'), the lowest floor of any indicator."""
    tabs = [NU_TRAIN_FLOOR[indicator]] if indicator in NU_TRAIN_FLOOR else list(NU_TRAIN_FLOOR.values())
    lf = math.log(max(float(fps), 1e-3))
    vals = []
    for t in tabs:
        f = np.log(np.array(sorted(t), np.float64)); v = np.log(np.array([t[k] for k in sorted(t)], np.float64))
        vals.append(float(np.exp(np.interp(lf, f, v))))
    return NU_GUARD_K * min(vals)


def noise_level(dff: np.ndarray, fps: float) -> float:
    """Rupprecht 2021 standardized noise level in % * s^(-1/2).

    nu = median(|diff(dF/F)|) / sqrt(fps) * 100. MUST be computed on FRACTIONAL dF/F
    (not percent, not the standardized trace) -- both mistakes have shipped before.
    Trained coverage reaches ~NU_K*8/sqrt(fps) (~139 at 30 Hz), far beyond any usable
    recording, so no clamp is applied.
    """
    return float(np.median(np.abs(np.diff(dff))) / np.sqrt(fps) * 100.0)


# ------------------------------------------------------------------------ band routing
_GRID_CACHE: dict = {}


def ckpt_fps_grid(path: Path) -> tuple:
    """The fps grid a checkpoint was trained on, from its own saved args (cached per path)."""
    key = str(path)
    if key not in _GRID_CACHE:
        A = _torch_load(path, "cpu")["args"]
        if not isinstance(A, dict):
            A = vars(A)
        _GRID_CACHE[key] = tuple(sorted(float(x) for x in str(A["fps"]).split(",")))
    return _GRID_CACHE[key]


def band_grids(model_dir: Path, file_fmt: str, indicator: str = "ALL") -> dict:
    """{band: trained fps grid} for every band whose first-seed checkpoint exists in
    model_dir for this file family (e.g. jGECO1a specialists have no HIGH)."""
    out = {}
    for band in BANDS:
        p = Path(model_dir) / file_fmt.format(ind=indicator, band=band, seed=SEEDS[0])
        if p.exists():
            out[band] = ckpt_fps_grid(p)
    if not out:
        raise FileNotFoundError(f"no {file_fmt.format(ind=indicator, band='*', seed=SEEDS[0])} "
                                f"checkpoints in {model_dir}. {missing_models_hint(model_dir)}")
    return out


def pick_band(fps: float, grids: dict) -> str:
    """In-distribution routing: the band TRAINED at this frame rate.

    Rule: among the bands whose trained fps grid contains `fps`, take the highest; if no
    grid contains it (a gap, or beyond the top grid), the band with the nearest grid point
    in log-fps (ties -> higher). Parameter-free, monotone in fps, evaluated at the actual
    session rate, and the same philosophy as the PV set (pick_band_pv).

    [2026-09-06] This REPLACES the CV routing table (routing_f1rt25.json). Audited on the
    post-fix CV grids (7 indicators x 3 bands x 5 folds x 14 fps x nu grid), the table was
    a per-(indicator, fps) argmax whose median winning margin was 0.006 F1 -- 64/96 picks
    decided by < 0.01, only 11/96 above the 0.027 seed floor -- so it encoded seed noise as
    policy: non-monotone in fps (6 band flips for 6f/8f/8m, 2-3 of them downward), 31/96
    picks sent to a band whose training grid did not contain the fps (7f at 5 Hz -> HIGH,
    trained 40-120 Hz), and not reproducible from the current grids (stale vintage).
    Band choice is second-order: every GT-free rule sits within 0.1-0.3 seed floors of the
    per-metric oracle on F1, corr and jitter (C152: between-band spread at matched fps is
    0.004-0.014 F1 vs a much larger rate effect). This rule was the best of them on all
    three metrics (pooled f1_25 0.2339 vs table 0.2324, oracle 0.2357; smallest max
    per-pick regret), and the table beat it by more than the floor in 0/96 picks.
    Caveat: that audit used the corner_d=20 fan-out grids; the shipped sym24 models widen
    the MID grids down to 8 Hz for 7f/8f/8m/8s, which this rule follows automatically
    because it reads the grids from the shipped checkpoints themselves.
    """
    fps = float(fps)
    contains = [b for b in BANDS if b in grids and grids[b][0] <= fps <= grids[b][-1]]
    if contains:
        return contains[-1]                       # overlap (8-10 Hz, 40-60 Hz) -> higher band
    def dist(b):
        return min(abs(math.log(fps) - math.log(g)) for g in grids[b])
    return min(grids, key=lambda b: (round(dist(b), 9), -BANDS.index(b)))


def pv_band_grids(model_dir: Path) -> dict:
    """{flavor: {band: trained fps grid}} for every PV checkpoint present in model_dir."""
    out = {}
    for flav in PV_FLAVORS:
        g = {}
        for band in BANDS:
            p = Path(model_dir) / PV_FILE_FMT.format(ind=flav, band=band, seed=SEEDS[0])
            if p.exists():
                g[band] = ckpt_fps_grid(p)
        if g:
            out[flav] = g
    if not out:
        raise FileNotFoundError(f"no PV checkpoints ({PV_FILE_FMT}) in {model_dir}. "
                                f"{missing_models_hint(model_dir)}")
    return out


def pick_band_pv(indicator: str, fps: float, grids: dict) -> tuple[str, str]:
    """-> (grid_flavor, band) for PV, by the same IN-DISTRIBUTION rule as pick_band:
    the band trained at this frame rate, higher band on overlap, nearest log-fps on a gap.

    [2026-08-29] INDICATOR-AGNOSTIC ACROSS FLAVORS. The PV checkpoints are trained on
    IDENTICAL data -- all 41 PV cells across every PV-bearing indicator (PVFS_RECIPE par-17p:
    the stage-2 line has no --indicators; the `GCaMP6f`/`GCaMP8s` in the filename selects only
    the trained fps grid, w_ceil and sigma ladder). So the flavor is an ACQUISITION-REGIME
    label, not a sensor specialisation, and any indicator may be served by either flavor.
      * 6f and 8s keep same-flavor-first, so the two VALIDATED populations cannot regress.
      * every other indicator routes across all flavors, in-grid candidates first.

    [2026-09-06] Was nearest-log-fps, which is what pick_band replaced on the pyramidal side
    and which has the same defect here: its ties break toward the LOWER band by dict order,
    and jGCaMP8s at 13 Hz routed to LOW (trained 1-10 Hz) -- out of distribution. Measured on
    the PV parity grids, in-grid is at or above nearest-log-fps on every metric (corr gap to
    oracle 0.20 -> 0.04 seed floors, jitter 0.07 -> 0.02, F1 unchanged) and leaves 0 of 21
    picks out of distribution against 1. Four cells disagree; three improve, one (8s at 60 Hz,
    MID -> HIGH) costs 0.0065 F1 for 1.56 ms of jitter. Everything is far inside the 0.035 F1
    floor, so the case is principle and consistency, not accuracy.
    """
    if indicator in grids:                         # validated pair: same-flavor first
        return indicator, pick_band(fps, grids[indicator])
    cand = [(f, b, g) for f, bands in grids.items() for b, g in bands.items()]
    inb = [t for t in cand if t[2][0] <= fps <= t[2][-1]]
    if inb:                                        # in-distribution first, then higher band
        best = max(inb, key=lambda t: (BANDS.index(t[1]),
                                       -min(abs(math.log(fps) - math.log(x)) for x in t[2])))
        return best[0], best[1]
    best = min(cand, key=lambda t: (round(min(abs(math.log(fps) - math.log(x)) for x in t[2]), 9),
                                    -BANDS.index(t[1])))
    return best[0], best[1]


def resolve_checkpoints(model_dir: Path, model_set: str, indicator: str, band: str):
    """-> (paths[3], used_set). model_set: 'specialist' | 'generalist' | 'auto'.

    'auto' prefers the specialist when the indicator is core and all three seeds exist,
    else falls back to the generalist. jGECO1a has no HIGH specialists; 'auto' at HIGH
    resolves to the generalist rather than silently changing band.
    """
    model_dir = Path(model_dir)
    def _spec():
        return [model_dir / SPEC_FILE_FMT.format(ind=indicator, band=band, seed=s) for s in SEEDS]
    def _gen():
        return [model_dir / GEN_FILE_FMT.format(ind="ALL", band=band, seed=s) for s in SEEDS]

    if model_set == "specialist":
        paths = _spec()
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"SETSUNA specialist set incomplete for {indicator} {band}: missing {missing} "
                f"in {model_dir}. Use model_set='generalist' or install the specialist files. "
                f"{missing_models_hint(model_dir)}")
        return paths, "specialist"
    if model_set == "generalist":
        paths = _gen()
    else:  # auto
        paths = _spec()
        if indicator in CORE_INDICATORS and all(p.exists() for p in paths):
            return paths, "specialist"
        paths = _gen()
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"SETSUNA generalist set missing {missing} in {model_dir}. "
            f"{missing_models_hint(model_dir)}")
    return paths, "generalist"


# ------------------------------------------------------------------------ model loading
def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch without the kwarg
        return torch.load(path, map_location=device)


def table_rows(state_dict) -> tuple[int, int]:
    """(indicator rows, cell-type rows) of a checkpoint's embedding tables, read from the bulk branch's
    weights; (NUM_INDICATORS, NUM_CELL_TYPES) when the keys are absent. 14 / 8 = the C167 blanking recipes'
    trained unknown rows (indicator 13, cell type 7); 13 / 7 = no such row (the engine appends a zero row)."""
    def rows(key, default):
        w = state_dict.get(key) if hasattr(state_dict, "get") else None
        return int(w.shape[0]) if w is not None else default
    return (rows("bulk.indicator_emb.weight", NUM_INDICATORS),
            rows("bulk.cell_type_emb.weight", NUM_CELL_TYPES))


def load_ensemble(ckpt_paths, device) -> tuple[list, dict]:
    """Port of spikeinf synaptic_timing.load_ensemble (verified against the source).

    Each ensemble member is built from its OWN saved args; the shared decode/grid cfg
    (M, W, HEAD, DUAL) comes from the first checkpoint and members must agree on it.
    """
    ck0 = _torch_load(ckpt_paths[0], device)
    A = ck0["args"]
    if not isinstance(A, dict):
        A = vars(A)
    cfg = dict(
        M=int(A["m"]), W=int(A.get("W", A.get("w_ceil", 128))), HEAD=A["head"],
        WMODE=A.get("w_mode", "fixed"), WFLOOR=int(A.get("w_floor", 32)),
        WCEIL=int(A.get("w_ceil", A.get("W", 128))), TSEC=A.get("T"),
        DUAL=bool(A.get("dual_head", False)),
        HARDROUTE=bool(A.get("hard_route", False)),
        HC_FPS=float(str(A.get("hc_fps", "5,10")).split(",")[-1]),
        HC_NU=float(str(A.get("hc_nu", "16,30")).split(",")[0]), A=A)
    cfg["TABLE_ROWS"] = table_rows(ck0["model"])

    def _build(Ai, sd):
        n_ind, n_ct = table_rows(sd)
        return BinarySRHybrid(
            n_ind, n_ct, m=int(Ai["m"]), head_type=Ai["head"],
            W=int(Ai.get("W", Ai.get("w_ceil", 128))),
            bulk_d=int(Ai.get("bulk_d", 24)), bulk_ffn=int(Ai.get("bulk_ffn", 48)),
            bulk_n_heads=int(Ai.get("bulk_heads", 2)), bulk_layers=int(Ai.get("bulk_layers", 4)),
            corner_d=int(Ai.get("corner_d", 20)), corner_ffn=int(Ai.get("corner_ffn", 32)),
            corner_n_heads=int(Ai.get("corner_heads", 2)), corner_layers=int(Ai.get("corner_layers", 3)),
            n_corner=int(Ai.get("n_corner", 1)), gate_rich=bool(Ai.get("gate_rich", False)),
            dual=bool(Ai.get("dual_head", False)),
            bulk_backbone=(Ai.get("bulk_backbone")
                           or ("mamba" if Ai.get("bulk_mamba") else "ssm" if Ai.get("bulk_ssm") else "attn")),
            corner_backbone=Ai.get("corner_backbone"),
            gate_fps_bias=float(Ai.get("gate_fps_bias", 0.0)),
            gate_fps_ref=float(Ai.get("gate_fps_ref", 20.0))).to(device)

    models = []
    rows0 = None
    for cp in ckpt_paths:
        cki = ck0 if cp == ckpt_paths[0] else _torch_load(cp, device)
        Ai = cki["args"]
        if not isinstance(Ai, dict):
            Ai = vars(Ai)
        assert int(Ai["m"]) == cfg["M"] and Ai["head"] == cfg["HEAD"] \
            and bool(Ai.get("dual_head", False)) == cfg["DUAL"], \
            f"ensemble member {cp} mismatches base grid (M={cfg['M']}, HEAD={cfg['HEAD']}, DUAL={cfg['DUAL']})"
        rows = table_rows(cki["model"])
        if rows0 is None:
            rows0 = rows
        elif rows != rows0:   # a learned unknown token in some seeds and a zero row in others: refuse
            raise ValueError(f"ensemble member {cp} has (indicator, cell-type) table rows {rows}, "
                             f"the first member {rows0}; the seeds of one set must agree")
        m = _build(Ai, cki["model"])
        m.load_state_dict(cki["model"], strict=(os.environ.get("BSR_LOAD_STRICT", "1") == "1"))
        m.eval()
        models.append(m)
    return models, cfg


# Odd window lengths (W = 2 S + 1, about half the frame rates below the window ceiling) overlap-add each batch as
# three strided passes per stream instead of one slice add per window (2026-11-06: ~2,300 launches per ROI at
# 10.7 Hz, the host-bound part of the forward). Same bits. JSCOPE_SETSUNA_ODD_VEC=0 keeps the per-window loop.
_DEBUG_COVERAGE = os.environ.get("JSCOPE_SETSUNA_DEBUG_COVERAGE", "0").strip().lower() not in ("0", "false", "no", "off")
_ODD_VEC_ENABLED = os.environ.get("JSCOPE_SETSUNA_ODD_VEC", "1").strip().lower() not in ("0", "false", "no", "off")


def _odd_window_add(out, rows, c0: int, nr: int, S: int, X: int) -> None:
    """Overlap-add of ``nr`` regular windows of odd length W = 2S + 1 (stride S) into the flat ``out``, ``X`` values
    per frame: row i (window c0 + i, starting at frame (c0 + i) S) as three strided passes -- its last frame, its
    second half, its first half -- instead of one slice add per window. A bin receives its windows' values in
    increasing window order either way (the last frame of window r - 2, the second half of r - 1, the first half of
    r), so every bin sums the same values in the same order: bit for bit the per-window loop."""
    if nr <= 0:
        return
    w = S * X
    out.as_strided((nr, X), (w, 1), out.storage_offset() + (c0 + 2) * w).add_(rows[:nr, 2 * w:2 * w + X])
    out[(c0 + 1) * w:(c0 + 1 + nr) * w].view(nr, w).add_(rows[:nr, w:2 * w])
    out[c0 * w:(c0 + nr) * w].view(nr, w).add_(rows[:nr, :w])


def _odd_window_count(cov, c0: int, nr: int, S: int, X: int) -> None:
    """_odd_window_add of ones: the coverage count of the same windows."""
    if nr <= 0:
        return
    w = S * X
    cov.as_strided((nr, X), (w, 1), cov.storage_offset() + (c0 + 2) * w).add_(1)
    cov[(c0 + 1) * w:(c0 + 1 + nr) * w].view(nr, w).add_(1)
    cov[c0 * w:(c0 + nr) * w].view(nr, w).add_(1)


_ODD_FINE: dict = {}
_ODD_FINE_ENABLED = os.environ.get("JSCOPE_SETSUNA_ODD_FINE", "1").strip().lower() not in ("0", "false", "no", "off")


def _odd_fine_kernel(device):
    """_setsuna_cuda.CudaOddFine for this device once it has matched the torch path bit for bit (gpu_kernel_check),
    else None. JSCOPE_SETSUNA_ODD_FINE=0 keeps the torch path."""
    if not (_ODD_FINE_ENABLED and _ODD_VEC_ENABLED and device.type == "cuda"):
        return None
    key = (device.type, device.index)
    if key not in _ODD_FINE:
        k = None
        try:
            from . import _setsuna_cuda as _sc
            from setsuna._support import gpu_kernel_check as _gkc
            k = _sc.CudaOddFine(device)

            def _check(c, dev):
                if not _check_odd_fine(k, dev):
                    c.fail("odd-window fine stream")
            if not _gkc.use("setsuna.odd_fine", device, _check):
                k = None
        except Exception:
            k = None
        _ODD_FINE[key] = k
    return _ODD_FINE[key]


def _check_odd_fine(k, device) -> bool:
    """The kernel against _odd_window_add + _odd_window_count + the flush window + the division, on random rows of
    two batches and a flush window, bit for bit."""
    gen = torch.Generator(device=device)
    gen.manual_seed(20261107)
    for S, M, n_regular, bw, tail, nm in ((7, 64, 9, 4, True, 3), (5, 32, 6, 6, False, 3), (3, 16, 11, 5, True, 1)):
        W = 2 * S + 1
        n = (n_regular - 1) * S + W + (4 if tail else 0)
        s_tail = n - W if tail else -1
        rows_all = torch.rand((n_regular + 1, W * M), generator=gen, device=device, dtype=torch.float32) * 3
        ref = torch.zeros(n * M, dtype=torch.float32, device=device)
        cov = torch.zeros(n * M, dtype=torch.float32, device=device)
        got = torch.empty(n * M, dtype=torch.float32, device=device)
        cov_end = (n_regular + 1) * S + 1
        if cov_end < n:
            got[cov_end * M:].zero_()
        for c0 in range(0, n_regular, bw):
            nr = min(bw, n_regular - c0)
            rows = rows_all[c0:c0 + nr].contiguous()
            _odd_window_add(ref, rows, c0, nr, S, M)
            _odd_window_count(cov, c0, nr, S, M)
            k.run(rows, got, c0, nr, S, M, n, n_regular, s_tail, nm)
        if tail:
            ref[s_tail * M:n * M] += rows_all[n_regular]
            cov[s_tail * M:n * M] += 1
            got[s_tail * M:n * M] += rows_all[n_regular]
            got[s_tail * M:n * M].view(-1, M).div_(_odd_tail_counts(s_tail, n, S, n_regular, nm, device)[:, None])
        cov.mul_(float(nm))
        ref /= cov
        if not torch.equal(ref, got):
            return False
    torch.cuda.synchronize(device)
    return True


_ODD_TAIL_COUNTS: dict = {}


def _odd_tail_counts(s0: int, n: int, S: int, n_regular: int, n_models: int, device):
    """Per frame of the flush window (frames s0..n-1): (regular windows covering it + 1) x the models, float32 on the
    device -- cached per geometry: a pageable upload per ROI would wait for the device's queue to drain."""
    key = (int(s0), int(n), int(S), int(n_regular), int(n_models), str(device))
    t = _ODD_TAIL_COUNTS.get(key)
    if t is None:
        if len(_ODD_TAIL_COUNTS) > 64:
            _ODD_TAIL_COUNTS.clear()
        t = _ODD_TAIL_COUNTS[key] = _odd_tail_counts_host(s0, n, S, n_regular, n_models, device)
    return t


def _odd_tail_counts_host(s0: int, n: int, S: int, n_regular: int, n_models: int, device):
    f = np.arange(int(s0), int(n))
    lo = np.where(f - 2 * S <= 0, 0, (f - 2 * S + S - 1) // S)
    hi = np.minimum(n_regular - 1, f // S)
    cnt = np.maximum(0, hi - lo + 1) + 1
    return torch.as_tensor(cnt.astype(np.float32) * np.float32(n_models), device=device)


def w_of_fps(cfg: dict, fps: float) -> int:
    """Window length in frames: clamp(round(T*fps), w_floor, w_ceil) under mask mode."""
    if cfg["WMODE"] == "fixed" or cfg["TSEC"] is None:
        return cfg["W"]
    return int(min(cfg["WCEIL"], max(cfg["WFLOOR"], round(float(cfg["TSEC"]) * fps))))


def _extend_zero_identity(models) -> None:
    """Session-local surgery: append a ZERO row to every indicator-sized embedding table so an
    'other / novel' indicator conditions on the additive identity -- the deployment analog of
    the untrained row the GT-unknown recipe ships with. Settled by measurement
    (GTUNKNOWN_RECIPE.md §6, 2026-07-20 row ablation): zero ≈ random untrained row (differences
    ~18x below the seed floor), while kinetics-neighbor PROXIES do not substitute -- proxy-8f
    recovered +0.001 of the +0.016 identity effect and proxy-6f was WORSE than random; the
    embedding is not a smooth kinetics manifold. The identity channel feeds the COUNT head
    (calibration), not spike placement. Rows 0..12 are copied untouched, so named indicators
    are byte-identical through this surgery. A model whose tables already hold row 13 (the C167
    --ind_unknown_id 13 recipes: a TRAINED unknown-indicator token) is left alone."""
    import torch.nn as nn
    for m in models:
        if _has_row(m, "indicator_emb", OTHER_INDICATOR_ID):
            continue
        for parent in m.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Embedding) and child.num_embeddings == NUM_INDICATORS:
                    new = nn.Embedding(NUM_INDICATORS + 1, child.embedding_dim)
                    with torch.no_grad():
                        new.weight[:NUM_INDICATORS] = child.weight
                        new.weight[NUM_INDICATORS].zero_()
                    new = new.to(child.weight.device, dtype=child.weight.dtype)
                    setattr(parent, name, new)


OTHER_INDICATOR_ID = NUM_INDICATORS   # the appended zero row, or the C167 trained unknown-indicator token


def _has_row(model, table: str, row: int) -> bool:
    """True when the model's bulk branch ``table`` (indicator_emb / cell_type_emb) already has ``row``."""
    emb = getattr(getattr(model, "bulk", None), table, None)
    return emb is not None and int(emb.num_embeddings) > int(row)


def learned_unknown_rows(models) -> tuple[bool, bool]:
    """(indicator row 13, cell-type row 7) are TRAINED tokens in these freshly loaded models (C167 blanking
    recipes), as opposed to absent (the engine then appends zero rows). Call before the zero-row surgery."""
    return (bool(models) and all(_has_row(m, "indicator_emb", OTHER_INDICATOR_ID) for m in models),
            bool(models) and all(_has_row(m, "cell_type_emb", NUM_CELL_TYPES) for m in models))


def _extend_zero_cell_type(models) -> None:
    """Session-local surgery for a cell type the user cannot name: append a ZERO row to every
    cell-type-sized embedding table (the branches' cell_type_emb, and a rich gate's), the additive
    identity -- what _extend_zero_identity does for an unnamed indicator. Applied to the all-cell-types
    set only. Rows 0..NUM_CELL_TYPES-1 are copied untouched. A model whose tables already hold row 7 (the
    C167 --ct_unknown_id 7 recipes: a TRAINED unknown-cell-type token) is left alone."""
    import torch.nn as nn
    for m in models:
        if _has_row(m, "cell_type_emb", UNKNOWN_CELL_TYPE_ID):
            continue
        for parent in m.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Embedding) and child.num_embeddings == NUM_CELL_TYPES:
                    new = nn.Embedding(NUM_CELL_TYPES + 1, child.embedding_dim)
                    with torch.no_grad():
                        new.weight[:NUM_CELL_TYPES] = child.weight
                        new.weight[NUM_CELL_TYPES].zero_()
                    new = new.to(child.weight.device, dtype=child.weight.dtype)
                    setattr(parent, name, new)


UNKNOWN_CELL_TYPE_ID = NUM_CELL_TYPES   # the appended zero cell-type row, or the C167 trained unknown token
assert NUM_CELL_TYPES != NUM_INDICATORS   # the two surgeries tell the tables apart by their size


def allct_route(model_dir, indicator: str, fps: float, model_set: str = "auto"):
    """-> (set, band, grid, paths, note) for a cell type not listed ('other'): the all-cell-types set when its
    files are in ``model_dir``; else (None, None, None, None, note), and the caller runs the pyramidal sets.
    ``indicator`` is the internal key or 'other'. GCaMP6f / GCaMP8s / GCaMP6s take their own set (unless
    model_set is 'generalist'); the 'Other' indicator, model_set 'generalist' and every other indicator take
    the ALL set (with the indicator's own trained row, or the unknown-indicator token for 'Other')."""
    if indicator in ALLCT_INDICATORS and model_set != "generalist":
        flavor = indicator
    else:
        flavor = "ALL"
    absent = ("Cell type not listed: runs the pyramidal model -- the all-cell-types models are not installed "
              "(Get SETSUNA models installs them).")
    try:
        grids = band_grids(model_dir, ALLCT_FILE_FMT, flavor)
    except FileNotFoundError:
        return None, None, None, None, absent
    band = pick_band(fps, grids)
    paths = [Path(model_dir) / ALLCT_FILE_FMT.format(ind=flavor, band=band, seed=s) for s in SEEDS]
    if not all(p.exists() for p in paths):
        return None, None, None, None, absent
    return flavor, band, grids[band], paths, ""


def _dual_fast_path_ok(m) -> bool:
    """True when BinarySRHybrid's dual forward is exactly 'gate-mix bulk and corner' -- one
    corner, no count adapter -- so _predict_fine_cuda may run the two branches itself and mix
    them in one kernel. Anything else keeps the vendored forward."""
    return (getattr(m, "n_corner", None) == 1 and bool(getattr(m, "dual", False))
            and getattr(m, "count_adapter", None) is None
            and hasattr(m, "bulk") and hasattr(m, "corner")
            and callable(getattr(m, "compute_gate", None)))


_TRITON_ATTN_ENABLED = os.environ.get("JSCOPE_SETSUNA_TRITON_ATTN", "1").strip().lower() not in (
    "0", "false", "no", "off")
# Dot-product precision of the Triton kernel: "fp16x3" (the default: a 3-pass FP16 split of the fp32
# operands, power-of-two scaled from rigorous weight-derived bounds), "tf32x3" (3-pass TF32 emulation)
# or "ieee" (fp32 FMA). Measured on the whole SETSUNA forward against a float64 run of the unmodified
# models (24 ROIs x 157,889 frames, RTX 5090), rate relRMS / signed bias / ms per ROI:
#   ieee 3.9e-7 / -4.8e-8 / 28.0;  tf32x3 7.8e-7 / +7.6e-7 / 26.3;  fp16x3 4.9e-7 / +4.8e-7 / 21.1-21.9
# FP16 carries the same 11-bit significand as TF32 but its tensor cores run 2.1x the TF32 rate here
# (231 vs 108 TFLOP/s), so the split costs a third less while landing closer to float64 than tf32x3.
# (Earlier, on the attention output alone: SDPA 4.1e-7, ieee 3.1e-7, tf32x3 2.5e-7.)
# Plain "tf32" (1.5e-3) is deliberately not accepted here: measured on a 1450-ROI session it was
# no faster (97 s vs 92 s wall) and biased the outputs (+0.3 % total spike mass, relRMS 2.4e-3).
_TRITON_ATTN_PRECISION = os.environ.get("JSCOPE_SETSUNA_TRITON_ATTN_PRECISION", "fp16x3").strip().lower()
_ATTN_BLOCK_M = int(os.environ.get("JSCOPE_SETSUNA_ATTN_BLOCK_M", "64") or 64)
_ATTN_BLOCK_N = int(os.environ.get("JSCOPE_SETSUNA_ATTN_BLOCK_N", "64") or 64)
_ATTN_WARPS = int(os.environ.get("JSCOPE_SETSUNA_ATTN_WARPS", "4") or 4)

try:  # Triton is optional: without it the vendored nn.MultiheadAttention (SDPA) path is used
    import triton as _triton
    import triton.language as _tl
except Exception:  # pragma: no cover - depends on the environment
    _triton = None
    _tl = None

if _triton is not None:
    @_triton.jit
    def _band_attn_fwd(Q, K, V, O, qk_scale_log2e, L, HALF, H,
                       stride_qb, stride_qm, stride_qh, stride_kb, stride_kn, stride_kh,
                       stride_vb, stride_vn, stride_vh, stride_ob, stride_om, stride_oh,
                       q_mul, k_mul, v_mul, out_div,
                       BLOCK_M: _tl.constexpr, BLOCK_N: _tl.constexpr, D: _tl.constexpr, PREC: _tl.constexpr):
        """Flash-style attention forward in float32 with the band mask |m - n| <= HALF evaluated
        in-kernel: no mask tensor is read and key tiles entirely outside the band are skipped.
        Online softmax in base 2 (the scale carries log2 e), fp32 accumulators throughout.
        Program (m-block, b*H + h); (batch, head) are addressed with separate strides, so Q/K/V
        may be strided views of one packed projection and O any (batch, position, head) layout."""
        start_m = _tl.program_id(0) * BLOCK_M
        bh = _tl.program_id(1)
        b = bh // H
        hh = bh % H
        offs_m = start_m + _tl.arange(0, BLOCK_M)
        offs_d = _tl.arange(0, D)
        Q = Q + b * stride_qb + hh * stride_qh
        K = K + b * stride_kb + hh * stride_kh
        V = V + b * stride_vb + hh * stride_vh
        O = O + b * stride_ob + hh * stride_oh
        q = _tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :],
                     mask=offs_m[:, None] < L, other=0.0)
        if PREC == "fp16x3":
            # fp16 hi/lo split of the (power-of-two scaled, hence exact) operand
            qs = q * q_mul
            q_hi = qs.to(_tl.float16)
            q_lo = (qs - q_hi.to(_tl.float32)).to(_tl.float16)
        m_i = _tl.full([BLOCK_M], float("-inf"), _tl.float32)
        l_i = _tl.zeros([BLOCK_M], _tl.float32)
        acc = _tl.zeros([BLOCK_M, D], _tl.float32)
        lo = _tl.maximum(start_m - HALF, 0)
        lo = (lo // BLOCK_N) * BLOCK_N
        hi = _tl.minimum(start_m + BLOCK_M + HALF, L)
        for start_n in range(lo, hi, BLOCK_N):
            offs_n = start_n + _tl.arange(0, BLOCK_N)
            k = _tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :],
                         mask=offs_n[:, None] < L, other=0.0)
            if PREC == "fp16x3":
                ks = k * k_mul
                k_hi = ks.to(_tl.float16)
                k_lo = (ks - k_hi.to(_tl.float32)).to(_tl.float16)
                kt_hi = _tl.trans(k_hi)
                sc = _tl.dot(q_lo, kt_hi, out_dtype=_tl.float32)            # correction terms
                sc = _tl.dot(q_hi, _tl.trans(k_lo), sc, out_dtype=_tl.float32)
                # main term in its own accumulator, one round-to-nearest add; the host folds
                # 2^-(SQ+SK) into qk_scale_log2e
                qk = (_tl.dot(q_hi, kt_hi, out_dtype=_tl.float32) + sc) * qk_scale_log2e
            else:
                qk = _tl.dot(q, _tl.trans(k), input_precision=PREC) * qk_scale_log2e
            allowed = ((offs_n[None, :] >= offs_m[:, None] - HALF) & (offs_n[None, :] <= offs_m[:, None] + HALF)
                       & (offs_n[None, :] < L))
            qk = _tl.where(allowed, qk, float("-inf"))
            m_new = _tl.maximum(m_i, _tl.max(qk, 1))
            m_safe = _tl.where(m_new == float("-inf"), 0.0, m_new)   # rows with no allowed key yet
            alpha = _tl.math.exp2(m_i - m_safe)
            p = _tl.math.exp2(qk - m_safe[:, None])
            l_i = l_i * alpha + _tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = _tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :],
                         mask=offs_n[:, None] < L, other=0.0)
            if PREC == "fp16x3":
                ps = p * 16384.0                  # p in [0, 1]: 2^14 keeps the lo parts normal
                p_hi = ps.to(_tl.float16)
                p_lo = (ps - p_hi.to(_tl.float32)).to(_tl.float16)
                vs = v * v_mul
                v_hi = vs.to(_tl.float16)
                v_lo = (vs - v_hi.to(_tl.float32)).to(_tl.float16)
                # This key block's product in FRESH accumulators, added to acc by a round-to-nearest
                # add. Chaining the MMAs into acc across the key blocks accumulated the tensor
                # cores' truncation: rate +4.9e-6 biased vs float64, this form +4.8e-7.
                pc = _tl.dot(p_lo, v_hi, out_dtype=_tl.float32)
                pc = _tl.dot(p_hi, v_lo, pc, out_dtype=_tl.float32)
                acc = acc + (_tl.dot(p_hi, v_hi, out_dtype=_tl.float32) + pc)
            else:
                acc += _tl.dot(p, v, input_precision=PREC)
            m_i = m_new
        if PREC == "fp16x3":
            acc = acc / (l_i[:, None] * out_div)  # out_div = 2^(14+SV), exact
        else:
            acc = acc / l_i[:, None]
        _tl.store(O + offs_m[:, None] * stride_om + offs_d[None, :], acc,
                  mask=offs_m[:, None] < L)


def _tf32_ok(device) -> bool:
    try:
        return device.type == "cuda" and tuple(torch.cuda.get_device_capability(device)) >= (8, 0)
    except Exception:
        return False


def _pow2_scale_exponent(bound: float, target_log2: int = 14) -> int:
    """Largest S with bound * 2^S <= 2^target_log2 (FP16 max is ~2^16). bound <= 0 -> 0."""
    if not (bound > 0.0) or not math.isfinite(bound):
        return 0
    return int(math.floor(target_log2 - math.log2(bound)))


def _band_attention_triton(q, k, v, half: int, sm_scale: float, prec: str, heads: int = 1, out=None,
                           scale_exps=None):
    """q, k, v: float32 with a power-of-two head width D >= 16 (zero-padded head dim) in the last
    dim -- either (BH, L, D) contiguous (``heads`` = 1) or (B, L, heads*D) views of a packed
    projection (any strides; the last dim contiguous). ``out`` (same layout as q) is written and
    returned; by default a contiguous tensor of q's shape."""
    if heads == 1:
        BH, L, D = q.shape
        B = BH
        sq = (q.stride(0), q.stride(1), 0)
        sk = (k.stride(0), k.stride(1), 0)
        sv = (v.stride(0), v.stride(1), 0)
    else:
        B, L, HD = q.shape
        D = HD // int(heads)
        sq = (q.stride(0), q.stride(1), D)
        sk = (k.stride(0), k.stride(1), D)
        sv = (v.stride(0), v.stride(1), D)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device) if out is None else out
    so = (o.stride(0), o.stride(1), (D if heads > 1 else 0))
    grid = (_triton.cdiv(L, _ATTN_BLOCK_M), B * int(heads))
    qk_const = float(sm_scale) * 1.4426950408889634
    q_mul = k_mul = v_mul = out_div = 1.0
    if prec == "fp16x3":
        if scale_exps is None:                 # no weight-derived bound: take the data's own
            scale_exps = tuple(_pow2_scale_exponent(float(t.abs().max())) for t in (q, k, v))
        eq, ek, ev = (int(e) for e in scale_exps)
        q_mul, k_mul, v_mul = math.ldexp(1.0, eq), math.ldexp(1.0, ek), math.ldexp(1.0, ev)
        qk_const = math.ldexp(qk_const, -(eq + ek))
        out_div = math.ldexp(1.0, 14 + ev)
    _band_attn_fwd[grid](q, k, v, o, qk_const, int(L), int(half), int(heads),
                         sq[0], sq[1], sq[2], sk[0], sk[1], sk[2], sv[0], sv[1], sv[2], so[0], so[1], so[2],
                         q_mul, k_mul, v_mul, out_div,
                         BLOCK_M=_ATTN_BLOCK_M, BLOCK_N=_ATTN_BLOCK_N, D=D, PREC=prec, num_warps=_ATTN_WARPS)
    return o


class _BandAttention:
    """The projections of one nn.MultiheadAttention laid out for the Triton kernel.

    The packed in-projection is re-rowed as (3, heads, D) with D = 16 lanes per head, the extra
    lanes zero, so the GEMM writes the kernel's layout directly and pads nothing afterwards; the
    output projection gets zero columns for the same lanes. Zero rows/columns add exact zeros,
    so the values are those of the original projections. Not an nn.Module on purpose: the
    module tree and state_dict of the vendored model stay untouched."""

    def __init__(self, mha, prec: str, ln=None):
        E, h, dh = int(mha.embed_dim), int(mha.num_heads), int(mha.head_dim)
        D = 16
        while D < dh:
            D *= 2
        W = mha.in_proj_weight.detach()          # (3E, E)
        b = mha.in_proj_bias.detach()            # (3E,)
        Wp = W.new_zeros(3, h, D, E)
        bp = b.new_zeros(3, h, D)
        Wp[:, :, :dh, :] = W.view(3, h, dh, E)
        bp[:, :, :dh] = b.view(3, h, dh)
        Wo = mha.out_proj.weight.detach()        # (E, E) acting on (h, dh)
        Wop = Wo.new_zeros(E, h, D)
        Wop[:, :, :dh] = Wo.view(E, h, dh)
        self.Wp, self.bp = Wp.reshape(3 * h * D, E).contiguous(), bp.reshape(3 * h * D).contiguous()
        self.Wop = Wop.reshape(E, h * D).contiguous()
        self.bo = mha.out_proj.bias.detach() if mha.out_proj.bias is not None else None
        self.E, self.h, self.dh, self.D = E, h, dh, D
        self.scale = 1.0 / math.sqrt(dh)
        self.prec = prec
        self.scale_exps = None
        if prec == "fp16x3":
            # Rigorous magnitude bound of q, k, v: the block's LayerNorm output obeys
            # |z_i| <= sqrt(E - 1) |gamma_i| + |beta_i| (the normalised vector has zero mean and
            # squared norm E), so each projection row is bounded by its |w|-weighted sum plus |b|.
            if ln is None or getattr(ln, "weight", None) is None:
                # no bound available: the 3-pass TF32 kernel (ieee below compute capability 8.0: no TF32)
                self.prec = "tf32x3" if _tf32_ok(mha.in_proj_weight.device) else "ieee"
            else:
                g = ln.weight.detach().double().abs()
                zb = math.sqrt(max(E - 1, 1)) * g + (ln.bias.detach().double().abs()
                                                     if ln.bias is not None else 0.0)
                rows = W.double().abs() @ zb + b.double().abs()          # (3E,)
                bq, bk, bv = (float(rows[i * E:(i + 1) * E].max()) for i in range(3))
                self.scale_exps = (_pow2_scale_exponent(bq), _pow2_scale_exponent(bk),
                                   _pow2_scale_exponent(bv))

    def fused_block(self, x, half: int, ln, ffn=None):
        """The whole LocalMHABlock: LayerNorm + packed QKV projection (one kernel), band attention,
        out-projection + residual (one kernel) -- and with ``ffn`` (the following FFNBlock's
        _install_triton_ffn parameters) that block too, in the out-projection's kernel. Falls back to
        ln -> __call__ -> residual (then the FFN) for anything that is not a contiguous float32
        (B, L, E)."""
        if x.dtype != torch.float32 or x.dim() != 3 or x.shape[-1] != self.E or not x.is_contiguous():
            y = x + self(ln(x), half)
            return y if ffn is None else _ffn_apply(y, ffn)
        p = getattr(self, "_fused_params", None)
        if p is None:
            E, hd = self.E, self.h * self.D
            DP, OP, AP = _pad_pow2(E), _pad_pow2(3 * hd), _pad_pow2(hd)
            with torch.no_grad():
                p = (_pad_vec(ln.weight, DP), _pad_vec(ln.bias, DP), _pad_mat_t(self.Wp, DP, OP), _pad_vec(self.bp, OP),
                     _pad_mat_t(self.Wop, AP, DP),
                     _pad_vec(self.bo, DP) if self.bo is not None else torch.zeros(DP, device=x.device),
                     DP, OP, AP, float(ln.eps))
            self._fused_params = p
        lnw, lnb, wqkv, bqkv, wo, bo, DP, OP, AP, eps = p
        B, L, E = x.shape
        hd = self.h * self.D
        rows = B * L
        qkv = torch.empty((B, L, 3 * hd), dtype=torch.float32, device=x.device)
        _ln_proj_fwd[(_triton.cdiv(rows, 64),)](x, qkv, lnw, lnb, wqkv, bqkv, rows, eps,
                                                D=E, DP=DP, O=3 * hd, OP=OP, BL=64, num_warps=4)
        o = _band_attention_triton(qkv[..., :hd], qkv[..., hd:2 * hd], qkv[..., 2 * hd:],
                                   half, self.scale, self.prec, heads=self.h, scale_exps=self.scale_exps)
        out = torch.empty_like(x)
        if ffn is not None:
            flnw, flnb, w1t, b1, w2t, b2, fd, fDP, ff, FP, feps = ffn
            if fd == E and fDP == DP:
                # out-projection + residual and the FFN block in one pass (same operations: same bits)
                _proj_res_ffn_fwd[(_triton.cdiv(rows, 64),)](o, x, out, wo, bo, flnw, flnb, w1t, b1, w2t, b2,
                                                            rows, feps, AD=hd, AP=AP, D=E, DP=DP, FF=ff, FP=FP,
                                                            BL=64, num_warps=4)
                return out
        _proj_res_fwd[(_triton.cdiv(rows, 64),)](o, x, out, wo, bo, rows,
                                                 AD=hd, AP=AP, D=E, DP=DP, BL=64, num_warps=4)
        return out if ffn is None else _ffn_apply(out, ffn)

    def __call__(self, x, half: int):
        B, L, E = x.shape
        hd = self.h * self.D
        qkv = F.linear(x, self.Wp, self.bp).float()                  # (B, L, 3*h*D): [q | k | v]
        # Strided views straight into the kernel (batch/head strides), output in the (B, L, h*D)
        # layout the out-projection consumes: no permute copies.
        o = _band_attention_triton(qkv[..., :hd], qkv[..., hd:2 * hd], qkv[..., 2 * hd:],
                                   half, self.scale, self.prec, heads=self.h, scale_exps=self.scale_exps)
        return F.linear(o, self.Wop, self.bo)


def _triton_attention_selftest(prec: str, device) -> bool:
    """Compile and run the kernel once on a small problem and check it against SDPA. A failure
    (no Triton backend for this GPU, a compiler error) means the vendored path stays in place."""
    try:
        g = torch.Generator(device=device).manual_seed(0)
        B, h, L, dh = 4, 2, 96, 12
        q, k, v = (torch.randn(B, h, L, dh, device=device, generator=g) for _ in range(3))
        i = torch.arange(L, device=device)
        half = 40
        mask = torch.where((i[None, :] >= i[:, None] - half) & (i[None, :] <= i[:, None] + half), 0.0,
                           float("-inf")).to(torch.float32)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        qp, kp, vp = (F.pad(t, (0, 16 - dh)).reshape(B * h, L, 16).contiguous() for t in (q, k, v))
        out = _band_attention_triton(qp, kp, vp, half, 1.0 / math.sqrt(dh), prec)[:, :, :dh].view(B, h, L, dh)
        torch.cuda.synchronize(device)
        err = float((out - ref).abs().max()) / max(float(ref.abs().max()), 1e-30)
        return bool(torch.isfinite(out).all()) and err < 1e-4
    except Exception:
        return False


def attn_precision(device) -> str:
    """The attention kernel's dot-product precision on ``device``: JSCOPE_SETSUNA_TRITON_ATTN_PRECISION (fp16x3 by
    default). tf32x3 needs TF32 tensor cores (compute capability 8.0+); below, its request runs as ieee (fp32 FMA)."""
    prec = _TRITON_ATTN_PRECISION if _TRITON_ATTN_PRECISION in ("ieee", "tf32x3", "fp16x3") else "tf32x3"
    try:
        if prec == "tf32x3" and tuple(torch.cuda.get_device_capability(device)) < (8, 0):
            prec = "ieee"
    except Exception:
        pass
    return prec


def _install_triton_attention(models) -> int:
    """Session-local surgery: LocalMHABlock's attention through the Triton band kernel.

    Why: with fp32 inputs SDPA runs the cutlass mem-efficient kernel, 1.0 ms per call at
    B=512, L=384, 2 heads of 12 -- 48 calls, ~48 ms per ROI, half of the whole forward -- at a
    tenth of the GPU's fp32 rate (a generic 64x64 SIMT kernel, head dim padded internally, the
    (L, L) float mask read every call). The Triton kernel evaluates the band |m - n| <= half
    in-kernel, skips the key tiles entirely outside it, and keeps every accumulator in fp32:
    0.32 ms per call in IEEE fp32 (3.1x), 0.24 ms with 3-pass TF32 emulation.

    Precision, measured against a float64 softmax attention on real activations: relRMS SDPA
    4.1e-7, ieee 3.1e-7, tf32x3 2.5e-7 -- the kernel is closer to the exact result than the one
    it replaces. End to end vs the previous float32 outputs on 5 real ROIs (tf32x3): relRMS
    7-8e-7, per-ROI spike totals unchanged to 0.01, pooled float16 streams differ in
    0.04-0.15% of bins by one ulp, all 1196 decoded spikes at identical times. Pooled path
    0.100 -> 0.071 s per ROI. SDPA remains the path when Triton is missing,
    JSCOPE_SETSUNA_TRITON_ATTN=0, the block is not the plain batch-first MHA we recognise, or the
    kernel fails its self-test on this GPU (warned once). Returns the number of blocks patched.
    """
    import types
    import torch.nn as nn
    if _triton is None or not _TRITON_ATTN_ENABLED:
        return 0
    device = next((p.device for m in models for p in m.parameters()), None)
    if device is None or device.type != "cuda":
        return 0
    prec = attn_precision(device)
    # the kernel against the vendored block (SDPA) and a float64 one, once per process and GPU (gpu_kernel_check's
    # rule; failed -> the vendored attention, logged once)
    from setsuna._support import gpu_kernel_check as _gkc
    from . import gpu_checks as _gc
    if not _gkc.use("setsuna.attn", device, _gc.check_attn):
        return 0

    def forward(self, x, eff_window=None):
        window = eff_window if eff_window is not None else self.local_window
        half = (int(window) // 2) if window is not None else int(x.size(1))
        if _FUSED_BLOCKS_ENABLED:
            return self._band_attn.fused_block(x, half, self.ln)
        h = self.ln(x)
        return x + self.drop(self._band_attn(h, half))

    n = 0
    for mdl in models:
        for blk in mdl.modules():
            if type(blk).__name__ != "LocalMHABlock" or not isinstance(getattr(blk, "attn", None), nn.MultiheadAttention):
                continue
            mha = blk.attn
            if not (mha.batch_first and mha._qkv_same_embed_dim and mha.in_proj_bias is not None
                    and mha.bias_k is None and mha.bias_v is None and not mha.add_zero_attn
                    and mha.head_dim <= 128 and float(mha.dropout) == 0.0):
                continue
            object.__setattr__(blk, "_band_attn", _BandAttention(mha, prec, ln=getattr(blk, "ln", None)))
            blk.forward = types.MethodType(forward, blk)
            n += 1
    return n


_COMPILE_ENABLED = os.environ.get("JSCOPE_SETSUNA_COMPILE", "1").strip().lower() not in (
    "0", "false", "no", "off")
_compiled = {"ok": None, "fns": {}}


_PREWARM: dict = {"thread": None}


def prewarm_compile_async(needed: bool = True) -> None:
    """Start computing torch.compile's per-process cache key in the background.

    ``needed=False``: the caller knows this process's forward is the CUDA-kernel one, which never compiles --
    nothing starts, now or on later calls (a compile that is needed after all primes the key itself in
    _compile_available, as without a prewarm).

    The first ``torch.compile`` in a process makes inductor hash all 2,141 module files of
    the installed torch (``torch_key``; now set from ``jupiterscope.torch_key_cache`` when its
    on-disk cache exists). With a warm OS file cache that is under a second; after a
    pipeline has streamed 100+ GB of movie through the page cache it measured 13 s, all of
    it on the SETSUNA stage's critical path ("Loading SETSUNA models"). Calling this as
    early as possible lets that run under trace loading, dF/F and the checkpoint loads;
    ``_compile_available`` joins the thread, so the probe itself sees the cached key.
    Idempotent; a no-op when compile is disabled or Triton is missing.
    """
    if not needed:
        _PREWARM["skip"] = True
    if not needed or _PREWARM.get("skip") or not _COMPILE_ENABLED or _triton is None or _PREWARM["thread"] is not None:
        return

    def _work() -> None:
        try:
            import torch._dynamo  # noqa: F401  (1.4 s of imports, also off the critical path)
            from setsuna._support.torch_key_cache import prime_torch_key
            prime_torch_key()      # cached per torch install; else computed here and stored
        except Exception:
            pass

    t = threading.Thread(target=_work, name="jupiterscope_compile_prewarm", daemon=True)
    t.start()
    _PREWARM["thread"] = t


def _compile_available() -> bool:
    """torch.compile with the Triton (inductor) backend, probed once per process."""
    if _compiled["ok"] is None:
        ok = False
        if _COMPILE_ENABLED and _triton is not None:
            _t = _PREWARM["thread"]
            if _t is not None:
                _t.join()        # the cache key is being computed in the background; reuse it
            try:
                import torch._dynamo  # noqa: F401
                # the on-disk torch_key when no prewarm primed it (idempotent): otherwise this first compile
                # hashes torch's 2,141 source files -- 12 s after the registration has streamed the movie
                # through the page cache (the in-process runner's SETSUNA preload, which no longer follows
                # a torch.compile-using registration)
                from setsuna._support.torch_key_cache import prime_torch_key
                prime_torch_key()
                f = torch.compile(lambda a, b: a * b + 1.0, dynamic=False)
                x = torch.ones(8, device="cuda")
                ok = bool(torch.equal(f(x, x), x * x + 1.0))
            except Exception:
                ok = False
        _compiled["ok"] = ok
    return _compiled["ok"]


# Set (per thread) around a one-time self-test's reference forward: a session that has not probed torch.compile yet
# takes the eager functions there instead of probing (see _compile_ready).
_NO_PROBE = threading.local()


def _compile_ready() -> bool:
    """_compile_available(), except inside a self-test's reference forward (_NO_PROBE) of a session that has not
    probed yet: False there, without probing -- the reference runs the eager functions."""
    if _compiled["ok"] is None and getattr(_NO_PROBE, "on", False):
        return False
    return _compile_available()


def _fused(name, fn):
    """Compile ``fn`` once (shapes are static per session; each new shape compiles once more and
    is cached on disk by inductor). Falls back to the eager function if compilation fails.
    CUDA tensors only: the probe covers inductor's Triton backend, while CPU tensors would take its
    C++ backend (see _models_on_cuda)."""
    if _compiled["ok"] is None and getattr(_NO_PROBE, "on", False):
        return fn                           # a self-test's reference, before any probe: eager, not cached
    if name not in _compiled["fns"]:
        f = fn
        if _compile_available():
            try:
                f = torch.compile(fn, dynamic=False)
            except Exception:
                f = fn
        _compiled["fns"][name] = f
    return _compiled["fns"][name]


def _ln_fn(x, weight, bias, eps: float):
    x32 = x.float()
    var, mu = torch.var_mean(x32, dim=-1, unbiased=False, keepdim=True)
    y = torch.addcmul(bias, x32 - mu, torch.rsqrt(var + eps) * weight)
    return y if y.dtype == x.dtype else y.to(x.dtype)


_HOST_KERNELS: dict = {}


def _post_head_first(bc, cc, bb, cb, w):
    """Gate mix of bulk/corner logits, count softplus and binary sigmoid, for the first model."""
    c = torch.lerp(bc, cc, w)
    b = torch.lerp(bb, cb, w)
    return F.softplus(c), torch.sigmoid(b)


def _post_head_acc(bc, cc, bb, cb, w, acc, acc_b):
    """Same, accumulated onto the running ensemble sums (returns the new sums)."""
    c = torch.lerp(bc, cc, w)
    b = torch.lerp(bb, cb, w)
    return acc + F.softplus(c), acc_b + torch.sigmoid(b)

# ------------------------------------------------------------------------ fused FFN / attention prologue+epilogue (Triton)
_FUSED_BLOCKS_ENABLED = os.environ.get("JSCOPE_SETSUNA_FUSED_BLOCKS", "1").strip().lower() not in (
    "0", "false", "no", "off")

if _triton is not None:
    @_triton.jit
    def _ffn_fwd(X, OUT, LNW, LNB, W1T, B1, W2T, B2, n_rows, eps,
                 D: _tl.constexpr, DP: _tl.constexpr, FF: _tl.constexpr, FP: _tl.constexpr, BL: _tl.constexpr):
        """out = x + lin2(gelu(lin1(ln(x)))) on (rows, D) contiguous rows; parameters zero-padded
        (LNW/LNB (DP,), W1T (DP, FP), B1 (FP,), W2T (FP, DP), B2 (DP,))."""
        rows = _tl.program_id(0) * BL + _tl.arange(0, BL)
        rm = rows < n_rows
        offs_c = _tl.arange(0, DP)
        cmask = offs_c < D
        offs_f = _tl.arange(0, FP)
        m2 = rm[:, None] & cmask[None, :]
        x = _tl.load(X + rows[:, None] * D + offs_c[None, :], mask=m2, other=0.0)
        mu = _tl.sum(x, 1) / D
        xc = _tl.where(cmask[None, :], x - mu[:, None], 0.0)
        var = _tl.sum(xc * xc, 1) / D
        rstd = 1.0 / _tl.sqrt(var + eps)
        hn = _tl.load(LNB + offs_c)[None, :] + xc * (rstd[:, None] * _tl.load(LNW + offs_c)[None, :])
        w1 = _tl.load(W1T + offs_c[:, None] * FP + offs_f[None, :])
        h1 = _tl.dot(hn, w1, input_precision="ieee") + _tl.load(B1 + offs_f)[None, :]
        g = 0.5 * h1 * (1.0 + _libdevice.erf(h1 * 0.7071067811865476))
        w2 = _tl.load(W2T + offs_f[:, None] * DP + offs_c[None, :])
        y = _tl.dot(g, w2, input_precision="ieee") + _tl.load(B2 + offs_c)[None, :]
        _tl.store(OUT + rows[:, None] * D + offs_c[None, :], x + y, mask=m2)

    @_triton.jit
    def _ln_proj_fwd(X, OUT, LNW, LNB, WT, BV, n_rows, eps,
                     D: _tl.constexpr, DP: _tl.constexpr, O: _tl.constexpr, OP: _tl.constexpr, BL: _tl.constexpr):
        """out (rows, O) = ln(x) @ W^T + b on (rows, D) contiguous rows; WT (DP, OP), BV (OP,) zero-padded."""
        rows = _tl.program_id(0) * BL + _tl.arange(0, BL)
        rm = rows < n_rows
        offs_c = _tl.arange(0, DP)
        cmask = offs_c < D
        offs_o = _tl.arange(0, OP)
        x = _tl.load(X + rows[:, None] * D + offs_c[None, :], mask=rm[:, None] & cmask[None, :], other=0.0)
        mu = _tl.sum(x, 1) / D
        xc = _tl.where(cmask[None, :], x - mu[:, None], 0.0)
        var = _tl.sum(xc * xc, 1) / D
        rstd = 1.0 / _tl.sqrt(var + eps)
        hn = _tl.load(LNB + offs_c)[None, :] + xc * (rstd[:, None] * _tl.load(LNW + offs_c)[None, :])
        w = _tl.load(WT + offs_c[:, None] * OP + offs_o[None, :])
        y = _tl.dot(hn, w, input_precision="ieee") + _tl.load(BV + offs_o)[None, :]
        _tl.store(OUT + rows[:, None] * O + offs_o[None, :], y, mask=rm[:, None] & (offs_o < O)[None, :])

    @_triton.jit
    def _proj_gelu_fwd(X, OUT, WT, BV, n_rows,
                       D: _tl.constexpr, DP: _tl.constexpr, O: _tl.constexpr, OP: _tl.constexpr, BL: _tl.constexpr):
        """out (rows, OP) = gelu(x @ W^T + b) with every OP column written -- the zero-padded W/b give
        exact zeros in columns O..OP-1, so the output is directly a zero-padded head-stack slab;
        x (rows, D) contiguous; WT (DP, OP), BV (OP,) zero-padded."""
        rows = _tl.program_id(0) * BL + _tl.arange(0, BL)
        rm = rows < n_rows
        offs_c = _tl.arange(0, DP)
        offs_o = _tl.arange(0, OP)
        x = _tl.load(X + rows[:, None] * D + offs_c[None, :], mask=rm[:, None] & (offs_c < D)[None, :], other=0.0)
        w = _tl.load(WT + offs_c[:, None] * OP + offs_o[None, :])
        y = _tl.dot(x, w, input_precision="ieee") + _tl.load(BV + offs_o)[None, :]
        g = 0.5 * y * (1.0 + _libdevice.erf(y * 0.7071067811865476))
        _tl.store(OUT + rows[:, None] * OP + offs_o[None, :], g, mask=rm[:, None])

    @_triton.jit
    def _proj_res_fwd(A, X, OUT, WT, BV, n_rows,
                      AD: _tl.constexpr, AP: _tl.constexpr, D: _tl.constexpr, DP: _tl.constexpr, BL: _tl.constexpr):
        """out = x + a @ W^T + b; a (rows, AD), x/out (rows, D) contiguous; WT (AP, DP), BV (DP,) zero-padded."""
        rows = _tl.program_id(0) * BL + _tl.arange(0, BL)
        rm = rows < n_rows
        offs_a = _tl.arange(0, AP)
        offs_c = _tl.arange(0, DP)
        cmask = offs_c < D
        a = _tl.load(A + rows[:, None] * AD + offs_a[None, :], mask=rm[:, None] & (offs_a < AD)[None, :], other=0.0)
        w = _tl.load(WT + offs_a[:, None] * DP + offs_c[None, :])
        y = _tl.dot(a, w, input_precision="ieee") + _tl.load(BV + offs_c)[None, :]
        m2 = rm[:, None] & cmask[None, :]
        x = _tl.load(X + rows[:, None] * D + offs_c[None, :], mask=m2, other=0.0)
        _tl.store(OUT + rows[:, None] * D + offs_c[None, :], x + y, mask=m2)

    @_triton.jit
    def _proj_res_ffn_fwd(A, X, OUT, WOT, BO, LNW, LNB, W1T, B1, W2T, B2, n_rows, eps,
                          AD: _tl.constexpr, AP: _tl.constexpr, D: _tl.constexpr, DP: _tl.constexpr,
                          FF: _tl.constexpr, FP: _tl.constexpr, BL: _tl.constexpr):
        """out = ffn(x + a @ Wo^T + bo): _proj_res_fwd and _ffn_fwd in one pass -- the attention block's
        output row stays in registers instead of a round trip through memory. The same operations on the
        same operands as the two kernels (the row is masked to zeros where their load masked it): same bits."""
        rows = _tl.program_id(0) * BL + _tl.arange(0, BL)
        rm = rows < n_rows
        offs_a = _tl.arange(0, AP)
        offs_c = _tl.arange(0, DP)
        cmask = offs_c < D
        offs_f = _tl.arange(0, FP)
        m2 = rm[:, None] & cmask[None, :]
        a = _tl.load(A + rows[:, None] * AD + offs_a[None, :], mask=rm[:, None] & (offs_a < AD)[None, :], other=0.0)
        wo = _tl.load(WOT + offs_a[:, None] * DP + offs_c[None, :])
        y = _tl.dot(a, wo, input_precision="ieee") + _tl.load(BO + offs_c)[None, :]
        x = _tl.load(X + rows[:, None] * D + offs_c[None, :], mask=m2, other=0.0)
        x = _tl.where(m2, x + y, 0.0)
        mu = _tl.sum(x, 1) / D
        xc = _tl.where(cmask[None, :], x - mu[:, None], 0.0)
        var = _tl.sum(xc * xc, 1) / D
        rstd = 1.0 / _tl.sqrt(var + eps)
        hn = _tl.load(LNB + offs_c)[None, :] + xc * (rstd[:, None] * _tl.load(LNW + offs_c)[None, :])
        w1 = _tl.load(W1T + offs_c[:, None] * FP + offs_f[None, :])
        h1 = _tl.dot(hn, w1, input_precision="ieee") + _tl.load(B1 + offs_f)[None, :]
        g = 0.5 * h1 * (1.0 + _libdevice.erf(h1 * 0.7071067811865476))
        w2 = _tl.load(W2T + offs_f[:, None] * DP + offs_c[None, :])
        y2 = _tl.dot(g, w2, input_precision="ieee") + _tl.load(B2 + offs_c)[None, :]
        _tl.store(OUT + rows[:, None] * D + offs_c[None, :], x + y2, mask=m2)


def _pad_vec(v, n):
    out = torch.zeros(n, dtype=torch.float32, device=v.device)
    out[: v.numel()] = v.detach().float().reshape(-1)
    return out.contiguous()


def _pad_mat_t(w, rows_p, cols_p):
    """w (out, in) -> zero-padded transpose (in_p, out_p) so that y = a @ WT."""
    out = torch.zeros(rows_p, cols_p, dtype=torch.float32, device=w.device)
    out[: w.shape[1], : w.shape[0]] = w.detach().float().t()
    return out.contiguous()


def _models_on_cuda(models) -> bool:
    """True when the ensemble lives on a CUDA device -- the Triton kernels and the compiled blocks below exist only
    for that; installed on CPU models they fail at the first forward ("0 active drivers"). Compiled functions too:
    torch.compile of CPU tensors takes inductor's C++ backend, which needs a C++ compiler (MSVC's cl.exe on
    Windows, which user machines rarely have: "Compiler: cl is not found"), while _compile_available() probes the
    Triton backend on CUDA only."""
    device = next((p.device for m in models for p in m.parameters()), None)
    return device is not None and device.type == "cuda"


def _install_triton_ffn(models) -> int:
    """Session-local surgery: FFNBlock.forward through _ffn_fwd (one kernel). Returns blocks patched."""
    import types
    import torch.nn as nn
    if _triton is None or not _FUSED_BLOCKS_ENABLED or not _models_on_cuda(models):
        return 0
    from setsuna._support import gpu_kernel_check as _gkc
    from . import gpu_checks as _gc
    device = next(p.device for m in models for p in m.parameters())
    if not _gkc.use("setsuna.ffn", device, _gc.check_ffn):
        return 0
    n = 0
    for mdl in models:
        for blk in mdl.modules():
            params = _ffn_params(blk)
            if params is None:
                continue
            prev = blk.forward

            def forward(self, x, _p=params, _prev=prev):
                if x.dtype != torch.float32 or x.shape[-1] != _p[6] or not x.is_contiguous():
                    return _prev(x)
                return _ffn_apply(x, _p)

            blk.forward = types.MethodType(forward, blk)
            object.__setattr__(blk, "_jscope_ffn", params)     # the fused out-projection path takes the block over
            n += 1
    return n


def _ffn_params(blk):
    """_ffn_fwd's zero-padded parameters of an FFNBlock of the expected structure (None for anything else)."""
    import torch.nn as nn
    if not (type(blk).__name__ == "FFNBlock" and isinstance(getattr(blk, "ln", None), nn.LayerNorm)
            and isinstance(getattr(blk, "lin1", None), nn.Linear) and isinstance(getattr(blk, "lin2", None), nn.Linear)
            and blk.lin1.bias is not None and blk.lin2.bias is not None and blk.ln.elementwise_affine
            and float(getattr(getattr(blk, "drop", None), "p", 0.0) or 0.0) == 0.0 and not blk.training):
        return None
    d, ff = int(blk.lin1.in_features), int(blk.lin1.out_features)
    if blk.lin2.in_features != ff or blk.lin2.out_features != d or blk.ln.normalized_shape[0] != d:
        return None
    DP, FP = _pad_pow2(d), _pad_pow2(ff)
    with torch.no_grad():
        return (_pad_vec(blk.ln.weight, DP), _pad_vec(blk.ln.bias, DP), _pad_mat_t(blk.lin1.weight, DP, FP),
                _pad_vec(blk.lin1.bias, FP), _pad_mat_t(blk.lin2.weight, FP, DP), _pad_vec(blk.lin2.bias, DP),
                d, DP, ff, FP, float(blk.ln.eps))


def _ffn_apply(x, p):
    """The FFNBlock through _ffn_fwd on a contiguous float32 (..., d) tensor; p as _install_triton_ffn."""
    lnw, lnb, w1t, b1, w2t, b2, d, DP, ff, FP, eps = p
    out = torch.empty_like(x)
    rows = x.numel() // d
    # 32 rows per program: 10 % faster than 64 on the 512-window batch, same bits
    _ffn_fwd[(_triton.cdiv(rows, 32),)](x, out, lnw, lnb, w1t, b1, w2t, b2, rows, eps,
                                        D=d, DP=DP, FF=ff, FP=FP, BL=32, num_warps=4)
    return out


def _ffn_fn(x, ln_w, ln_b, eps: float, w1, b1, w2, b2):
    """FFNBlock.forward: residual + lin2(gelu(lin1(ln(x))))."""
    h = _ln_fn(x, ln_w, ln_b, eps)
    h = F.gelu(F.linear(h, w1, b1))
    return x + F.linear(h, w2, b2)


def _install_fused_ffn(models) -> int:
    """Session-local surgery: FFNBlock.forward through one compiled graph (layer norm, first
    linear, GELU, second linear, residual), so inductor fuses the elementwise work around the two
    GEMMs. Same operations in float32; measured 188 -> 125 us per block. Dropout is identity in
    eval and is not part of the fused function. Returns the number of blocks patched."""
    import types
    import torch.nn as nn
    # torch.compile is probed by the first forward that runs this block, not here: the CUDA-kernel forward
    # (encoders + head pool) never does, and the probe (a compile and a run, ~2 s of host work) held the
    # session build that long
    if not _COMPILE_ENABLED or _triton is None or not _models_on_cuda(models):
        return 0

    def forward(self, x):
        if not _compile_ready():
            return type(self).forward(self, x)      # the block's own forward, as when compile is unavailable
        return _fused("ffn", _ffn_fn)(x, self.ln.weight, self.ln.bias, float(self.ln.eps),
                                     self.lin1.weight, self.lin1.bias, self.lin2.weight, self.lin2.bias)

    n = 0
    for mdl in models:
        for blk in mdl.modules():
            if (type(blk).__name__ == "FFNBlock" and isinstance(getattr(blk, "ln", None), nn.LayerNorm)
                    and isinstance(getattr(blk, "lin1", None), nn.Linear) and isinstance(getattr(blk, "lin2", None), nn.Linear)
                    and float(getattr(getattr(blk, "drop", None), "p", 0.0) or 0.0) == 0.0 and not blk.training):
                blk.forward = types.MethodType(forward, blk)
                n += 1
    return n


def _install_fast_layernorm(models) -> int:
    """Session-local surgery: every nn.LayerNorm in the ensemble computes its statistics with
    torch.var_mean and applies them with one addcmul, instead of the native layer-norm kernel.

    The model normalises 24-wide rows -- 196,608 of them per call, 156 calls per ROI -- and
    PyTorch's fused kernel is at its worst there: 165 us per call against 86 us for
    var_mean + subtract + scale + addcmul, both memory-bound on the same 19 MB. Same
    arithmetic (biased variance, eps inside the square root, per-feature affine), float32
    throughout (inputs are up-cast, so the bf16 opt-in path keeps autocast's fp32 statistics);
    the difference is reduction order only: relRMS ~8e-8, no bias, at the block level.
    Set JSCOPE_SETSUNA_FAST_LN=0 to keep the native kernel (the reference for A/B runs).
    CUDA models only. CPU models keep the native kernel, the faster one there: 0.045 ms a call against 0.25 ms for
    this form (with its statistics from a compiled loop; torch.var_mean alone took ~4 ms), 78 calls per ROI -- a
    two-photon CPU pipeline (10,000 frames, 1472 ROIs) 399 -> 362 s, with the same reduction-order difference in
    its outputs (rate relRMS 2.7e-7, bias -4e-9). They were never compiled (see _models_on_cuda).
    Returns the number of modules patched.
    """
    import types
    import torch.nn as nn
    if os.environ.get("JSCOPE_SETSUNA_FAST_LN", "1").strip().lower() in ("0", "false", "no", "off"):
        return 0
    if not _models_on_cuda(models):
        return 0

    def forward(self, x):
        return _fused("ln", _ln_fn)(x, self.weight, self.bias, float(self.eps))

    n = 0
    for mdl in models:
        for mod in mdl.modules():
            if (isinstance(mod, nn.LayerNorm) and len(mod.normalized_shape) == 1
                    and mod.elementwise_affine and mod.weight is not None and mod.bias is not None):
                mod.forward = types.MethodType(forward, mod)
                n += 1
    return n


_FUSED_PROLOGUE_ENABLED = os.environ.get("JSCOPE_SETSUNA_FUSED_PROLOGUE", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _prologue_fn(x, w_in, b_in, ind, cell, pos):
    # the vendored prologue's expression, element for element: ((in_proj(x) + ind) + cell) + pos
    return ((x.unsqueeze(-1) * w_in + b_in) + ind) + cell + pos


def _install_fused_prologue(models) -> int:
    """Session-local surgery: SRBranch.features' embedding prologue -- in_proj (1 -> d), the
    indicator / nu / fps / cell-type embeddings and the time-aware sinusoid -- as one compiled
    elementwise kernel over (B, W, d) instead of a K=1 GEMV followed by three broadcast adds,
    each a pass over the 19 MB activation per branch and batch. The per-element expression order
    is the vendored one; the sinusoid is computed once per call at (1, W, d) instead of (B, W, d).
    Only branches with the expected layout (time-aware positions, dropout 0)
    are touched; the block loop and the final LayerNorm are the vendored ones. Returns the number
    of branches patched."""
    import sys as _sys
    import types
    import torch.nn as nn
    # torch.compile is probed by the first call of the patched features (see _install_fused_ffn)
    if not _FUSED_PROLOGUE_ENABLED or not _COMPILE_ENABLED or _triton is None or not _models_on_cuda(models):
        return 0
    n = 0
    for mdl in models:
        for br in (getattr(mdl, "bulk", None), getattr(mdl, "corner", None)):
            if br is None or type(br).__name__ != "SRBranch":
                continue
            mod = _sys.modules.get(type(br).__module__)
            local_mha = getattr(mod, "LocalMHABlock", None)
            fps_types = tuple(getattr(mod, "_FPS_BLOCK_TYPES", ()) or ())
            sinusoid = getattr(mod, "time_aware_sinusoid", None)
            in_proj = getattr(br, "in_proj", None)
            if not (isinstance(in_proj, nn.Linear) and in_proj.in_features == 1 and in_proj.bias is not None
                    and bool(getattr(br, "time_aware_pos", False)) and callable(sinusoid)
                    and local_mha is not None and isinstance(getattr(br, "blocks", None), nn.ModuleList)
                    and isinstance(getattr(br, "ln", None), nn.LayerNorm)
                    and float(getattr(getattr(br, "emb_drop", None), "p", 0.0) or 0.0) == 0.0 and not br.training):
                continue
            def features(self, x, indicator_id, nu, fps, cell_type, eff_window=None,
                         _lm=local_mha, _ft=fps_types, _sin=sinusoid):
                if not _compile_ready():          # the vendored features, as when compile is unavailable
                    return type(self).features(self, x, indicator_id, nu, fps, cell_type, eff_window)
                B = x.size(0)
                ind = self.indicator_emb(indicator_id).unsqueeze(1)
                cell = self.cell_type_emb(cell_type if cell_type is not None
                                          else torch.zeros(B, dtype=torch.long, device=x.device)).unsqueeze(1)
                if nu is not None:
                    ind = ind + self.nu_proj(torch.log(nu.clamp(min=1e-3)).unsqueeze(-1)).unsqueeze(1)
                if fps is not None:
                    ind = ind + self.fps_proj(torch.log(fps.clamp(min=1e-3)).unsqueeze(-1)).unsqueeze(1)
                if fps is None:                    # not the layout this surgery is for: vendored path
                    return type(self).features(self, x, indicator_id, nu, fps, cell_type, eff_window)
                # (1, W, d) for one fps value -- every row of the vendored (B, W, d) sinusoid is this
                # row. Computed on the device each call (a few tiny kernels): reading the fps value
                # back to the host to cache it would synchronise, which is illegal under CUDA graph
                # capture (measured: it invalidated the capture).
                pos = _sin(int(x.size(1)), int(self.d), fps.reshape(-1)[:1])
                h = _fused("prologue", _prologue_fn)(x, self.in_proj.weight.reshape(-1), self.in_proj.bias,
                                                     ind, cell, pos)
                blocks = self.blocks
                i, nb = 0, len(blocks)
                while i < nb:
                    blk = blocks[i]
                    if isinstance(blk, _lm):
                        # attention block + the FFN block after it, in one out-projection kernel
                        ba = getattr(blk, "_band_attn", None)
                        ffn = getattr(blocks[i + 1], "_jscope_ffn", None) if i + 1 < nb else None
                        if _FUSED_BLOCKS_ENABLED and ba is not None and ffn is not None:
                            window = eff_window if eff_window is not None else blk.local_window
                            half = (int(window) // 2) if window is not None else int(h.size(1))
                            h = ba.fused_block(h, half, blk.ln, ffn=ffn)
                            i += 2
                            continue
                        h = blk(h, eff_window=eff_window)
                    elif _ft and isinstance(blk, _ft):
                        h = blk(h, eff_window=eff_window, fps=fps)
                    else:
                        h = blk(h)
                    i += 1
                return self.ln(h)

            br.features = types.MethodType(features, br)
            n += 1
    return n


def _install_split_head(models) -> int:
    """Session-local surgery: emit a dual head's count and binary logits from two half-GEMMs
    instead of one GEMM followed by two strided slices.

    SRBranch.forward computes ``logits = sr_head(h)`` -> (B, L, 2m) and then takes
    ``logits[..., :m]`` and ``logits[..., m:]``; reshaping each strided slice to (B, L*m) is a
    402 MB copy at m=512, W=384, B=512, twice per branch, twelve times per ROI. The same two
    tensors come straight out of ``F.linear`` on the row halves of the last weight, and that is
    bit-identical -- an output element is the same 24-term dot product either way (torch.equal
    on all 3 models x 2 branches with real activations). 9.8 -> 8.9 ms per branch forward,
    ~11 ms per ROI. Only branches with the expected head (Linear, GELU, Linear(d, 2m)) are
    touched; anything else keeps the vendored forward. Returns the number of branches patched.
    """
    import types
    import torch.nn as nn

    def forward(self, x, indicator_id, nu=None, fps=None, cell_type=None, eff_window=None):
        h = self.features(x, indicator_id, nu, fps, cell_type, eff_window)   # (B, L, d)
        h1 = self.sr_head[1](self.sr_head[0](h))
        lin, m = self.sr_head[2], self.m
        c = F.linear(h1, lin.weight[:m], lin.bias[:m]).reshape(h.size(0), h.size(1) * m)
        b = F.linear(h1, lin.weight[m:], lin.bias[m:]).reshape(h.size(0), h.size(1) * m)
        return c, b                                                           # (count_logits, binary_logits)

    n = 0
    for mdl in models:
        for br in mdl.modules():
            if type(br).__name__ != "SRBranch" or not getattr(br, "dual", False):
                continue
            head = getattr(br, "sr_head", None)
            if not (isinstance(head, nn.Sequential) and len(head) == 3
                    and isinstance(head[0], nn.Linear) and isinstance(head[1], nn.GELU)
                    and isinstance(head[2], nn.Linear) and head[2].bias is not None
                    and head[2].out_features == 2 * int(br.m)):
                continue
            br.forward = types.MethodType(forward, br)
            n += 1
    return n


import contextlib


@contextlib.contextmanager
def _strict_fp32():
    """Evaluate the ensemble in true IEEE float32 for the duration of the call.

    Measured on the 1305-ROI / 157889-frame session (RTX 5090) against a full-fp32 run of the
    same weights: bf16 autocast made the forward SLOWER (0.265 vs 0.143 s per ROI -- this d=24
    model is memory-bound, so bf16 only added cast traffic and, with head_dim 12, pushed the
    attention onto the unfused fp32 math path) and moved the outputs by 2.5% relRMS with a
    systematic bias (+0.2% total expected spikes, +2.3% summed p(spike); single ROIs off by
    1-3%). TF32 in the matmuls/convs costs another 7e-4 relRMS (+0.05% spikes) for no speed.
    The GUI process sets the matmul precision to 'high' (TF32) for other stages, so the flags
    are forced here and restored on exit. The float32 run is what the weights define; nothing
    faster was as accurate, and nothing more accurate exists short of float64.
    """
    old_mm = torch.get_float32_matmul_precision()
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(old_mm)
        torch.backends.cudnn.allow_tf32 = old_cudnn


def resolve_device(choice: str = "auto") -> torch.device:
    """"auto" (CUDA when there is a GPU, else the CPU), or a torch device -- "gpu" (what the panels'
    Device row and run_extract_spikes's --spike-device hand over for CUDA) meaning "cuda"."""
    choice = str(choice or "auto").strip().lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if choice == "gpu" else choice)


# ---------------------------------------------------------------------------- inference
@torch.inference_mode()
@_strict_fp32()
def _predict_fine_host(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                 batch_windows: int = 512, autocast_bf16: bool = False):
    """Batched overlap-add over one standardized trace -> (lam, prob), both (n*M,) float32.

    Ported from eval_binary_sr.py:383-404 -- the scored path -- including the C4
    tail flush. Traces shorter than one window are reflect-padded and cropped back.
    """
    M = int(cfg["M"])
    n0 = len(xz)
    W = w_of_fps(cfg, eff)
    pad = 0
    if n0 < W:  # reflect-pad short traces; predict_stream's silent-zero case
        pad = W - n0
        xz = np.concatenate([xz, xz[-2::-1][:pad].astype(np.float32)]) if n0 > 1 \
            else np.full(W, float(xz[0]) if n0 else 0.0, np.float32)
        if len(xz) < W:  # extremely short trace; tile the reflection
            xz = np.resize(xz, W).astype(np.float32)
        pad = W - n0
    n = len(xz)

    eff_window = (W if cfg["WMODE"] == "mask" else None)
    route_mode = (("hard_corner" if (eff <= cfg["HC_FPS"] and nu >= cfg["HC_NU"]) else "hard_bulk")
                  if cfg["HARDROUTE"] else "soft")

    pred = np.zeros(n * M, np.float32)
    pred_b = np.zeros(n * M, np.float32)
    cov = np.zeros(n * M, np.float32)
    starts = list(range(0, n - W + 1, max(1, W // 2)))
    if starts and starts[-1] + W < n:   # [C4] flush the tail
        starts.append(n - W)

    use_ac = bool(autocast_bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    for c0 in range(0, len(starts), int(batch_windows)):
        chunk = starts[c0:c0 + int(batch_windows)]
        B = len(chunk)
        xb = torch.tensor(np.stack([xz[s0:s0 + W] for s0 in chunk]), device=device)
        nu_t = torch.full((B,), float(nu), device=device)
        fps_t = torch.full((B,), float(eff), device=device)
        ct_t = torch.full((B,), int(ct_id), dtype=torch.long, device=device)
        ind_t = torch.full((B,), int(ind_id), dtype=torch.long, device=device)
        prs, prbs = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
            for m in models:
                o = m(xb, ind_t, nu_t, fps_t, ct_t, eff_window=eff_window, route_mode=route_mode)
                if cfg["DUAL"]:
                    prs.append(F.softplus(o[0]).float())
                    prbs.append(torch.sigmoid(o[1]).float())
                else:
                    d = m.decode(o).float()
                    prs.append(d)
                    prbs.append(d)
        pr = torch.stack(prs).mean(0).reshape(B, -1).cpu().numpy()
        prb = torch.stack(prbs).mean(0).reshape(B, -1).cpu().numpy()
        for i, s0 in enumerate(chunk):
            pred[s0 * M:(s0 + W) * M] += pr[i]
            pred_b[s0 * M:(s0 + W) * M] += prb[i]
            cov[s0 * M:(s0 + W) * M] += 1
    assert cov.min() > 0, "internal: uncovered fine bins after tail flush"
    pred /= cov
    pred_b /= cov
    if pad:
        pred = pred[:n0 * M]
        pred_b = pred_b[:n0 * M]
    return pred, pred_b


# ------------------------------------------------------------------------ CPU: the fused host tail
# On a CPU the tail after the heads -- the gate mix, softplus / sigmoid, the ensemble mean, the overlap-add, the
# division by the coverage and the stored sums -- was a pass over the (windows, W*M) streams per operation, each into a
# fresh 80-400 MB array: 60 % of a CPU ROI once the LayerNorm statistics were compiled. _predict_host_fused runs the
# same operations with the same float32 roundings -- the head GEMMs and the activations as torch calls into reused
# buffers, the rest in compiled loops (_host_kernels) -- and never builds the full-resolution streams' divided copies
# unless they are asked for. JSCOPE_SETSUNA_HOST_FUSED=0 keeps _predict_fine_host.
_HOST_FUSED_ENABLED = os.environ.get("JSCOPE_SETSUNA_HOST_FUSED", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _host_tail_kernels():
    """The compiled host-tail loops (the _host_kernels module), or None without numba."""
    k = _HOST_KERNELS.get("tail")
    if k is None:
        try:
            from setsuna import _host_kernels as k
        except Exception:
            k = False
        _HOST_KERNELS["tail"] = k
    return k or None


def _host_fused_ok(models, cfg) -> bool:
    """Whether _predict_host_fused covers this ensemble: dual-head BinarySRHybrid models with one corner branch and no
    count adapter, each SR head Linear -> GELU -> Linear(d, 2m) with a bias. Anything else takes _predict_fine_host."""
    import torch.nn as nn
    if not (_HOST_FUSED_ENABLED and cfg.get("DUAL") and models) or _host_tail_kernels() is None:
        return False
    M = int(cfg["M"])
    for mdl in models:
        if (type(mdl).__name__ != "BinarySRHybrid" or int(getattr(mdl, "n_corner", -1)) != 1
                or not getattr(mdl, "dual", False) or getattr(mdl, "count_adapter", None) is not None
                or mdl.training):
            return False
        for br in (getattr(mdl, "bulk", None), getattr(mdl, "corner", None)):
            head = getattr(br, "sr_head", None)
            if (type(br).__name__ != "SRBranch" or not getattr(br, "dual", False) or int(getattr(br, "m", -1)) != M
                    or not (isinstance(head, nn.Sequential) and len(head) == 3 and isinstance(head[0], nn.Linear)
                            and isinstance(head[1], nn.GELU) and isinstance(head[2], nn.Linear)
                            and head[2].bias is not None and head[2].out_features == 2 * M)):
                return False
    return True


def _host_buf(bufs, name: str, numel: int) -> "torch.Tensor":
    """A flat float32 CPU buffer of numel elements: from the session's cache ``bufs`` (grown when too small, reused
    by the next ROI -- no fresh pages to fault in), or a new one without a cache."""
    if bufs is None:
        return torch.empty(numel, dtype=torch.float32)
    t = bufs.get(name)
    if t is None or t.numel() < numel:
        bufs.pop(name, None)
        t = torch.empty(numel, dtype=torch.float32)
        bufs[name] = t
    return t[:numel]


def _host_branch_features(br, xb, ind_t, nu_t, fps_t, ct_t, eff_window):
    """An SRBranch's head features on a window batch, (B, L, d): its encoder, the head's first Linear and GELU."""
    h = br.features(xb, ind_t, nu_t, fps_t, ct_t, eff_window)
    return br.sr_head[1](br.sr_head[0](h))


def _host_head_halves(br, h2, out_c, out_b, late_bias: bool = False) -> None:
    """An SRBranch's dual head on head features h2 ((rows, d)) into out_c / out_b ((rows, m) float32): the last
    Linear's row halves by addmm -- the call F.linear makes for a contiguous 3-D input, on the halves
    _install_split_head takes, so the same bits. ``late_bias``: by mm, without the bias, which the caller adds
    (gate_mix_bias; the same bits where _late_bias_ok) -- addmm first broadcasts the bias into the whole output, a pass
    over it."""
    lin, m = br.sr_head[2], int(br.m)
    if late_bias:
        torch.mm(h2, lin.weight[:m].t(), out=out_c)
        torch.mm(h2, lin.weight[m:].t(), out=out_b)
        return
    torch.addmm(lin.bias[:m], h2, lin.weight[:m].t(), out=out_c)
    torch.addmm(lin.bias[m:], h2, lin.weight[m:].t(), out=out_b)


# The CPU tail (heads, gate mix, softplus / sigmoid, overlap-add) over this many windows at a time, the encoders over
# the whole batch: 8 windows' four head buffers (25 MB) stay in the caches where the 512-window batch's (1.6 GB) went
# to memory and back for each pass -- a 158k-frame ROI 2.55 -> 2.23 s on 24 cores, and ~2.4 GB less memory. torch's
# element-wise kernels and MKL's products round by tensor size, so the streams differ from the whole-batch tail at
# float32 rounding level. JSCOPE_SETSUNA_CPU_TAIL_WINDOWS=0 runs the tail over the whole batch (the earlier bits).
_HOST_TAIL_WINDOWS = int(os.environ.get("JSCOPE_SETSUNA_CPU_TAIL_WINDOWS", "8").strip() or "8")


def _float16(a: np.ndarray) -> np.ndarray:
    """a.astype(np.float16) by torch's vectorised cast: the same rounding (to nearest, ties to even; overflow to
    inf), ~5x numpy's value-at-a-time cast, two of which per ROI were ~1 % of a CPU ROI."""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.float16).numpy()


def _late_bias_ok(models) -> bool:
    """Whether mm followed by a float32 add of the bias gives torch.addmm's bits for the ensemble's head halves on this
    CPU -- MKL adds the broadcast bias to the finished product (checked once per process on random rows; the heads'
    inner width is the model's d, whole in one pass of the product). False keeps the heads' addmm."""
    ok = _HOST_KERNELS.get("late_bias")
    if ok is None:
        ok = True
        gen = torch.Generator().manual_seed(0)
        with torch.inference_mode():
            for mdl in models[:1]:
                for br in (mdl.bulk, mdl.corner):
                    lin, m = br.sr_head[2], int(br.m)
                    h2 = torch.randn(4096, lin.in_features, generator=gen)
                    for sl in (slice(0, m), slice(m, 2 * m)):
                        ref = torch.addmm(lin.bias[sl], h2, lin.weight[sl].t())
                        got = torch.mm(h2, lin.weight[sl].t()) + lin.bias[sl]
                        ok = ok and bool(torch.equal(ref.view(torch.int32), got.view(torch.int32)))
        _HOST_KERNELS["late_bias"] = ok
    return bool(ok)


@torch.inference_mode()
@_strict_fp32()
def _predict_host_fused(models, cfg, xz, ind_id, ct_id, eff, nu, device, batch_windows: int = 512, pool_k=None,
                        full: bool = False, bufs=None) -> dict:
    """_predict_fine_host followed by the CPU branch of predict_fine_outputs (count_head_rate, pool_fine) -- the same
    windows, batches and float32 operations in the same order, with the tail fused (and, with _HOST_TAIL_WINDOWS,
    run a few windows at a time: the same operations on smaller tensors, the same values to float32 rounding; with
    0, the same bits).

    Per window batch: each model's branches' head features (_host_branch_features). Then per group of
    _HOST_TAIL_WINDOWS windows and model: the two branches' heads into four buffers (_host_head_halves; without their
    biases where _late_bias_ok), the gate mix (_host_kernels.gate_mix, or gate_mix_bias adding the biases first) into
    the ensemble sums (first model) or over the bulk heads (the others), softplus and sigmoid in place, the middle
    models added to the sums. After the last model one pass forms each window's ensemble mean and adds it to the
    frames it covers (ensemble_overlap_add); after the last batch one pass divides by the coverage and forms the rate
    and the pooled sums (finish_streams). ``bufs``: the session's buffer cache.
    Returns predict_fine_outputs' CPU dict: rate, lam_pooled / prob_pooled (float16) with ``pool_k``, lam / prob with
    ``full``."""
    K = _host_tail_kernels()
    M = int(cfg["M"])
    n0 = len(xz)
    W = w_of_fps(cfg, eff)
    pad = 0
    if n0 < W:  # reflect-pad short traces; predict_stream's silent-zero case (as _predict_fine_host)
        pad = W - n0
        xz = np.concatenate([xz, xz[-2::-1][:pad].astype(np.float32)]) if n0 > 1 \
            else np.full(W, float(xz[0]) if n0 else 0.0, np.float32)
        if len(xz) < W:  # extremely short trace; tile the reflection
            xz = np.resize(xz, W).astype(np.float32)
        pad = W - n0
    n = len(xz)
    eff_window = (W if cfg["WMODE"] == "mask" else None)
    starts = list(range(0, n - W + 1, max(1, W // 2)))
    if starts and starts[-1] + W < n:   # [C4] flush the tail
        starts.append(n - W)
    st = np.asarray(starts, np.int64)
    if full or bufs is None:
        pred = np.zeros((2, n * M), np.float32)                   # returned (lam, prob): not a reused buffer
    else:
        pb = _host_buf(bufs, "pred", 2 * n * M)
        pb.zero_()
        pred = pb.view(2, n * M).numpy()
    nm = len(models)
    N = W * M
    late = _late_bias_ok(models)
    biases = [tuple(br.sr_head[2].bias.detach()[sl].contiguous().numpy()
                    for br in (mdl.bulk, mdl.corner) for sl in (slice(0, M), slice(M, 2 * M)))
              for mdl in models] if late else None                 # per model: bulk count, bulk binary, corner ...
    for c0 in range(0, len(starts), int(batch_windows)):
        chunk = starts[c0:c0 + int(batch_windows)]
        B = len(chunk)
        xb = torch.tensor(np.stack([xz[s0:s0 + W] for s0 in chunk]), device=device)
        nu_t = torch.full((B,), float(nu), device=device)
        fps_t = torch.full((B,), float(eff), device=device)
        ct_t = torch.full((B,), int(ct_id), dtype=torch.long, device=device)
        ind_t = torch.full((B,), int(ind_id), dtype=torch.long, device=device)
        feats = [(_host_branch_features(mdl.bulk, xb, ind_t, nu_t, fps_t, ct_t, eff_window),
                  _host_branch_features(mdl.corner, xb, ind_t, nu_t, fps_t, ct_t, eff_window),
                  mdl.compute_gate(nu_t, fps_t, ind_t, ct_t).unsqueeze(-1).contiguous()) for mdl in models]
        sub = B if _HOST_TAIL_WINDOWS <= 0 else min(B, _HOST_TAIL_WINDOWS)
        for g0 in range(0, B, sub):
            g1 = min(B, g0 + sub)
            b = g1 - g0
            heads = _host_buf(bufs, "heads", 4 * b * N).view(4, b, N)  # bulk count / binary, corner count / binary
            acc = _host_buf(bufs, "acc", 2 * b * N).view(2, b, N)      # the ensemble sums: count, binary
            hn = [heads[i].numpy() for i in range(4)]
            for mi, mdl in enumerate(models):
                h_bulk, h_corner, w_all = feats[mi]
                for k, (br, h) in enumerate(((mdl.bulk, h_bulk), (mdl.corner, h_corner))):
                    _host_head_halves(br, h[g0:g1].reshape(b * W, h.shape[-1]),
                                      heads[2 * k].view(b * W, M), heads[2 * k + 1].view(b * W, M), late)
                w = w_all[g0:g1]
                dst = acc if mi == 0 else heads
                if late:
                    bbc, bbb, cbc, cbb = biases[mi]
                    K.gate_mix_bias(hn[0], hn[2], hn[1], hn[3], w.numpy(), bbc, cbc, bbb, cbb, M,
                                    dst[0].numpy(), dst[1].numpy())
                else:
                    K.gate_mix(hn[0], hn[2], hn[1], hn[3], w.numpy(), dst[0].numpy(), dst[1].numpy())
                torch._C._nn.softplus(dst[0], 1.0, 20.0, out=dst[0])   # F.softplus / torch.sigmoid, in place
                torch.sigmoid(dst[1], out=dst[1])
                if 0 < mi < nm - 1:
                    acc[0].add_(heads[0])
                    acc[1].add_(heads[1])
            if nm == 1:
                K.ensemble_overlap_add(acc[0].numpy(), acc[1].numpy(), acc[0].numpy(), acc[1].numpy(), False, 1,
                                       st[c0 + g0:c0 + g1], W, M, pred[0], pred[1])
            else:
                K.ensemble_overlap_add(acc[0].numpy(), acc[1].numpy(), hn[0], hn[1], True, nm,
                                       st[c0 + g0:c0 + g1], W, M, pred[0], pred[1])
    kp = int(pool_k) if pool_k else 0
    ks = kp if kp > 1 else 0
    nk = M // ks if ks else 0
    rate = np.empty(n0, np.float32)
    lam_pool = np.empty(n0 * nk, np.float32)
    prob_pool = np.empty(n0 * nk, np.float32)
    K.finish_streams(pred[0], pred[1], st, W, M, n0, ks, rate, lam_pool, prob_pool)
    out = {"rate": rate}
    if kp:
        if ks:
            out["lam_pooled"], out["prob_pooled"] = _float16(lam_pool), _float16(prob_pool)
        else:                                                     # pool_fine(stream, 1) is the stream itself
            out["lam_pooled"] = _float16(pred[0][:n0 * M])
            out["prob_pooled"] = _float16(pred[1][:n0 * M])
    if full:
        out["lam"], out["prob"] = pred[0][:n0 * M], pred[1][:n0 * M]
    return out


def _warm_host_kernels() -> None:
    """Compile the fused host tail's loops (or load them from numba's disk cache) for the types _predict_host_fused
    passes -- while a CPU session is built, not in its first ROI. Compiled, not run: the session may be built on a
    thread beside a stage's own parallel kernels, and under numba's workqueue threading layer (its fallback without
    an OpenMP runtime) two parallel kernels running at once end the process."""
    K = _host_tail_kernels()
    if K is None:
        return
    from numba import typeof

    def compile_for(kernel, *args):
        kernel.compile(tuple(typeof(x) for x in args))

    a = np.zeros((1, 16), np.float32)
    w = np.zeros((1, 1), np.float32)
    compile_for(K.gate_mix, a, a, a, a, w, a, a)
    bias = np.zeros(8, np.float32)
    compile_for(K.gate_mix_bias, a, a, a, a, w, bias, bias, bias, bias, 8, a, a)
    st = np.zeros(1, np.int64)
    pred = np.zeros((2, 16), np.float32)
    compile_for(K.ensemble_overlap_add, a, a, a, a, True, 3, st, 2, 8, pred[0], pred[1])
    one = np.zeros(2, np.float32)
    compile_for(K.finish_streams, pred[0], pred[1], st, 2, 8, 2, 4, one, np.zeros(4, np.float32),
                np.zeros(4, np.float32))


_GRAPHS_ENABLED = os.environ.get("JSCOPE_SETSUNA_CUDA_GRAPH", "1").strip().lower() not in (
    "0", "false", "no", "off")

# ------------------------------------------------------------------------ pooled tail (Triton)
_POOLED_TAIL_ENABLED = os.environ.get("JSCOPE_SETSUNA_POOLED_TAIL", "1").strip().lower() not in (
    "0", "false", "no", "off")
_POOL_TILE_ROWS = {64: 32, 128: 16, 256: 8, 512: 4}
_POOL_TILE_ROWS_ENV = os.environ.get("JSCOPE_SETSUNA_POOL_ROWS", "").strip()
# The K native bins of a pooled bin are summed in fp32 (a tree sum) and widened to float64 once, instead
# of widening every element first: GeForce runs fp64 at 1/64 of fp32, and the 400 M fp64 adds and
# conversions per 512-window batch were ~10 % of the head-pool kernel (0.6 ms per ROI). Measured against
# the fp64-first sums on 60 ROIs: rate relRMS 2.1e-8 (max 2.7e-8), 1.5e-5 / 4.1e-5 of the stored f16
# lambda / prob values move by one ulp, spike trains identical (counts and positions).
# JSCOPE_SETSUNA_POOL_SUM32=0 restores the fp64-first sums.
_POOL_SUM32 = os.environ.get("JSCOPE_SETSUNA_POOL_SUM32", "1").strip().lower() not in ("0", "false", "no", "off")
# Full-resolution streams (the timing decoder's input) from the pooled forward (see predict_fine_outputs).
_POOLED_FULL = os.environ.get("JSCOPE_SETSUNA_POOLED_FULL", "1").strip().lower() not in ("0", "false", "no", "off")
# The count head's softplus as log(1 + e) corrected by the rounding error of 1 + e (see _head_pool_fwd);
# JSCOPE_SETSUNA_SOFTPLUS=log1p keeps libdevice's log1p (a quarter of the kernel's time).
_SOFTPLUS_LOGCORR = os.environ.get("JSCOPE_SETSUNA_SOFTPLUS", "logcorr").strip().lower() != "log1p"

if _triton is not None:
    from triton.language.extra import libdevice as _libdevice

    @_triton.jit
    def _head_pool_fwd(H1, WT, BIAS, GATE, OUT_C, OUT_B, OUT_PF, OUT_LF, ROW_OFF, OUT_PART, n_rows, L,
                       s_h_t, s_h_r, s_w_t, s_w_h, s_w_k, s_b_t, s_b_h, s_g_n, s_o_r,
                       N: _tl.constexpr, DP: _tl.constexpr, M: _tl.constexpr, K: _tl.constexpr,
                       R: _tl.constexpr, C: _tl.constexpr, WRITE_PF: _tl.constexpr,
                       SUM32: _tl.constexpr = False, SP_LOGCORR: _tl.constexpr = False,
                       WRITE_LF: _tl.constexpr = False, WRITE_PART: _tl.constexpr = False):
        """Per row (one frame of one window) and per pooled bin: the sum over K native bins of the
        ensemble sum of softplus(lerp(bulk, corner, gate)) (count) and sigmoid(...) (binary), the
        dual heads evaluated on the fly. H1 (T = 2N branches, rows, DP): head features zero-padded
        from D to DP columns; WT (T, 2 heads, DP, M): head weights transposed and zero-padded;
        BIAS (T, 2, M); GATE (N, B) indexed by row // L. Head dot products: IEEE fp32 tl.dot.
        Outputs (rows, M // K) float64. WRITE_PART: OUT_PART[row] (float64) = the count sum over
        the row's first ROW_OFF[row] native bins (int32; 0 for none) -- the part of a frame before
        a decoder segment boundary, in place of the fine count stream."""
        pid = _tl.program_id(0)
        rows = pid * R + _tl.arange(0, R)
        rmask = rows < n_rows
        win = rows // L
        ks = _tl.arange(0, DP)
        part = _tl.zeros([R], dtype=_tl.float64)
        if WRITE_PART:
            offr = _tl.load(ROW_OFF + rows, mask=rmask, other=0)
        else:
            offr = _tl.zeros([R], dtype=_tl.int32)
        for cb in range(0, M, C):
            cols = cb + _tl.arange(0, C)
            acc_c = _tl.zeros([R, C], dtype=_tl.float32)
            acc_b = _tl.zeros([R, C], dtype=_tl.float32)
            for i in range(N):
                tb = 2 * i
                tc = tb + 1
                hb = _tl.load(H1 + tb * s_h_t + rows[:, None] * s_h_r + ks[None, :],
                              mask=rmask[:, None], other=0.0)                               # [R, DP]
                hc = _tl.load(H1 + tc * s_h_t + rows[:, None] * s_h_r + ks[None, :],
                              mask=rmask[:, None], other=0.0)
                wbc = _tl.load(WT + tb * s_w_t + ks[:, None] * s_w_k + cols[None, :])         # [DP, C]
                wbb = _tl.load(WT + tb * s_w_t + s_w_h + ks[:, None] * s_w_k + cols[None, :])
                wcc = _tl.load(WT + tc * s_w_t + ks[:, None] * s_w_k + cols[None, :])
                wcb = _tl.load(WT + tc * s_w_t + s_w_h + ks[:, None] * s_w_k + cols[None, :])
                lbc = _tl.dot(hb, wbc, input_precision="ieee") + _tl.load(BIAS + tb * s_b_t + cols)[None, :]
                lbb = _tl.dot(hb, wbb, input_precision="ieee") + _tl.load(BIAS + tb * s_b_t + s_b_h + cols)[None, :]
                lcc = _tl.dot(hc, wcc, input_precision="ieee") + _tl.load(BIAS + tc * s_b_t + cols)[None, :]
                lcb = _tl.dot(hc, wcb, input_precision="ieee") + _tl.load(BIAS + tc * s_b_t + s_b_h + cols)[None, :]
                w = _tl.load(GATE + i * s_g_n + win, mask=rmask, other=0.0)[:, None]
                small = _tl.abs(w) < 0.5
                coeff = _tl.where(small, w, w - 1.0)
                mc = coeff * (lcc - lbc) + _tl.where(small, lbc, lcc)
                mb = coeff * (lcb - lbb) + _tl.where(small, lbb, lcb)
                if SP_LOGCORR:
                    # softplus = log1p(e), e = exp(mc), as log(u) minus the rounding error of
                    # u = 1 + e: log1p(e) = log(u) - ((u - 1) - e) / u to first order in that error
                    # ((u - 1) - e is exact, |.| <= ulp(u)/2), within ~1 ulp like log1pf; the
                    # reciprocal may be approximate (the term it scales is <= 2^-24 relative).
                    # log1pf was a quarter of this kernel.
                    em = _libdevice.exp(mc)
                    um = 1.0 + em
                    acc_c += _tl.where(mc > 20.0, mc,
                                       _libdevice.log(um) - ((um - 1.0) - em) * _libdevice.fast_dividef(1.0, um))
                else:
                    acc_c += _tl.where(mc > 20.0, mc, _libdevice.log1p(_libdevice.exp(mc)))
                acc_b += 1.0 / (1.0 + _libdevice.exp(-mb))
            if WRITE_PF:
                # the fp32 ensemble sum of the binary head per fine bin (the full path's acc_b),
                # for consumers that need sub-frame positions (the spike-time decoders)
                _tl.store(OUT_PF + rows[:, None] * M + cols[None, :], acc_b, mask=rmask[:, None])
            if WRITE_LF:
                # the same for the count head (the full path's acc): the timing decoder's per-segment counts
                _tl.store(OUT_LF + rows[:, None] * M + cols[None, :], acc_c, mask=rmask[:, None])
            if WRITE_PART:
                part += _tl.sum(_tl.where(cols[None, :] < offr[:, None], acc_c.to(_tl.float64), 0.0), axis=1)
            if SUM32:
                pc = _tl.sum(_tl.reshape(acc_c, [R, C // K, K]), axis=2).to(_tl.float64)
                pb = _tl.sum(_tl.reshape(acc_b, [R, C // K, K]), axis=2).to(_tl.float64)
            else:
                pc = _tl.sum(_tl.reshape(acc_c.to(_tl.float64), [R, C // K, K]), axis=2)
                pb = _tl.sum(_tl.reshape(acc_b.to(_tl.float64), [R, C // K, K]), axis=2)
            pcols = (cb // K) + _tl.arange(0, C // K)
            optr = rows[:, None] * s_o_r + pcols[None, :]
            _tl.store(OUT_C + optr, pc, mask=rmask[:, None])
            _tl.store(OUT_B + optr, pb, mask=rmask[:, None])
        if WRITE_PART:
            _tl.store(OUT_PART + rows, part, mask=rmask)
# ------------------------------------------------------------------------ fused ConvBlock (Triton)
_FUSED_CONV_ENABLED = os.environ.get("JSCOPE_SETSUNA_FUSED_CONV", "1").strip().lower() not in (
    "0", "false", "no", "off")

if _triton is not None:
    @_triton.jit
    def _convblock_fwd(H, X, OUT, WDW, BDW, WPWT, BPW, L,
                       s_hb, s_hl, s_xb, s_xl, s_ob, s_ol,
                       D: _tl.constexpr, DP: _tl.constexpr, KS: _tl.constexpr, HALF: _tl.constexpr,
                       BL: _tl.constexpr):
        """out = x + pw(gelu(dw(h))) for one window (program 1) and BL positions (program 0), h = ln(x)
        computed beforehand. H/X/OUT (B, L, D) with strides; parameters zero-padded to DP lanes:
        WDW (DP, KS), BDW (DP,), WPWT (DP_in, DP_out), BPW (DP,). Conv zero padding outside [0, L)."""
        b = _tl.program_id(1)
        rows = _tl.program_id(0) * BL + _tl.arange(0, BL)
        offs_c = _tl.arange(0, DP)
        cmask = offs_c < D
        hb = H + b * s_hb
        acc = _tl.zeros([BL, DP], dtype=_tl.float32)
        for t in _tl.static_range(KS):
            r = rows + (t - HALF)
            rm = (r >= 0) & (r < L)
            ht = _tl.load(hb + r[:, None] * s_hl + offs_c[None, :],
                          mask=rm[:, None] & cmask[None, :], other=0.0)              # [BL, DP]
            wt = _tl.load(WDW + offs_c * KS + t)                                       # [DP]
            acc += ht * wt[None, :]
        acc += _tl.load(BDW + offs_c)[None, :]
        g = 0.5 * acc * (1.0 + _libdevice.erf(acc * 0.7071067811865476))
        wpw = _tl.load(WPWT + offs_c[:, None] * DP + offs_c[None, :])                  # [DP_in, DP_out]
        y = _tl.dot(g, wpw, input_precision="ieee") + _tl.load(BPW + offs_c)[None, :]
        om = (rows < L)[:, None] & cmask[None, :]
        xo = _tl.load(X + b * s_xb + rows[:, None] * s_xl + offs_c[None, :], mask=om, other=0.0)
        _tl.store(OUT + b * s_ob + rows[:, None] * s_ol + offs_c[None, :], xo + y, mask=om)

def _plain_convblock(blk) -> bool:
    """A ConvBlock of the expected structure -- LayerNorm(d), depthwise Conv1d(d, d, k, padding=(k-1)//2, groups=d,
    zero padding), Conv1d(d, d, 1), dropout 0, eval mode: the only kind the fused forwards replace."""
    import torch.nn as nn
    if type(blk).__name__ != "ConvBlock":
        return False
    ln, dw, pw = getattr(blk, "ln", None), getattr(blk, "dw", None), getattr(blk, "pw", None)
    return bool(isinstance(ln, nn.LayerNorm) and len(ln.normalized_shape) == 1 and ln.elementwise_affine
                and isinstance(dw, nn.Conv1d) and isinstance(pw, nn.Conv1d)
                and dw.groups == dw.in_channels == dw.out_channels == ln.normalized_shape[0]
                and dw.stride == (1,) and dw.dilation == (1,) and dw.padding == ((dw.kernel_size[0] - 1) // 2,)
                and dw.kernel_size[0] % 2 == 1 and dw.bias is not None and dw.padding_mode == "zeros"
                and pw.kernel_size == (1,) and pw.groups == 1 and pw.bias is not None
                and pw.in_channels == pw.out_channels == dw.out_channels
                and float(getattr(getattr(blk, "drop", None), "p", 0.0) or 0.0) == 0.0 and not blk.training)


def _install_fused_convblock(models) -> int:
    """Session-local surgery: ConvBlock.forward through _convblock_fwd. Only blocks with the
    expected structure (_plain_convblock) are touched; parameters are zero-padded once per block.
    Returns the number of blocks patched."""
    import types
    if _triton is None or not _FUSED_CONV_ENABLED or not _models_on_cuda(models):
        return 0
    from setsuna._support import gpu_kernel_check as _gkc
    from . import gpu_checks as _gc
    device = next(p.device for m in models for p in m.parameters())
    if not _gkc.use("setsuna.convblock", device, _gc.check_convblock):
        return 0
    n = 0
    for mdl in models:
        for blk in mdl.modules():
            if not _plain_convblock(blk):
                continue
            params = _convblock_params(blk)

            def forward(self, x, _p=params):
                return _convblock_apply(self, x, _p)

            blk.forward = types.MethodType(forward, blk)
            n += 1
    return n


def _convblock_params(blk):
    """_convblock_fwd's zero-padded parameters of a plain ConvBlock (_plain_convblock)."""
    ln, dw, pw = blk.ln, blk.dw, blk.pw
    d = int(ln.normalized_shape[0])
    ks = int(dw.kernel_size[0])
    DP = _pad_pow2(d)
    dev = ln.weight.device
    with torch.no_grad():
        wdw = torch.zeros(DP, ks, device=dev); wdw[:d] = dw.weight.float().view(d, ks)
        bdw = torch.zeros(DP, device=dev); bdw[:d] = dw.bias.float()
        wpwt = torch.zeros(DP, DP, device=dev); wpwt[:d, :d] = pw.weight.float().view(d, d).t()
        bpw = torch.zeros(DP, device=dev); bpw[:d] = pw.bias.float()
    return (wdw.contiguous(), bdw.contiguous(), wpwt.contiguous(), bpw.contiguous(), d, DP, ks, (ks - 1) // 2)


def _convblock_apply(blk, x, p):
    """ConvBlock.forward through _convblock_fwd (p: _convblock_params); the vendored forward for anything else."""
    wdw, bdw, wpwt, bpw, d, DP, ks, half = p
    if x.dtype != torch.float32 or x.dim() != 3 or x.shape[-1] != d or x.stride(-1) != 1:
        return type(blk).forward(blk, x)                # vendored path for anything unexpected
    h = blk.ln(x)                                       # the fused LayerNorm, one pass
    if h.stride(-1) != 1:
        h = h.contiguous()
    B, L, _ = x.shape
    out = torch.empty_like(x)
    BL = 64
    _convblock_fwd[(_triton.cdiv(L, BL), B)](
        h, x, out, wdw, bdw, wpwt, bpw, int(L),
        h.stride(0), h.stride(1), x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        D=d, DP=DP, KS=ks, HALF=half, BL=BL, num_warps=4)
    return out


# CPU models: the conv blocks' depthwise conv in its channels-last form (_install_host_convblock);
# JSCOPE_SETSUNA_HOST_CONV=0 keeps the vendored forward
_HOST_CONV_ENABLED = os.environ.get("JSCOPE_SETSUNA_HOST_CONV", "1").strip().lower() not in ("0", "false", "no", "off")


def _host_convblock_forward(blk, x):
    """The vendored ConvBlock.forward with its depthwise conv as a conv2d over the channels-last (B, d, 1, L) view of the
    normalised input: oneDNN's NHWC depthwise kernel, ~10x the Conv1d on the transposed input at the model's shapes,
    and the same bits. Its output is made contiguous where the Conv1d's output lies before GELU, whose bits depend on
    the layout; GELU, the pointwise Conv1d and the residual as the vendored forward."""
    h = blk.ln(x)
    dw = blk.dw
    d = F.conv2d(h.transpose(1, 2).unsqueeze(2), dw.weight.unsqueeze(2), dw.bias, padding=(0, dw.padding[0]),
                 groups=dw.groups).squeeze(2).contiguous()
    return x + blk.drop(blk.pw(F.gelu(d)).transpose(1, 2))


def _install_host_convblock(models) -> int:
    """Session-local surgery for CPU models: ConvBlock.forward through _host_convblock_forward -- a CPU ROI runs 24
    conv blocks, 0.88 -> 0.34 ms each on a two-photon batch (52 windows x 384 frames). Blocks of the expected
    structure only (_plain_convblock); each checks its first call against the vendored forward bit for bit and keeps
    the vendored forward if they differ. Returns the number of blocks patched."""
    import types
    if not _HOST_CONV_ENABLED or not models or _models_on_cuda(models):
        return 0
    n = 0
    for mdl in models:
        for blk in mdl.modules():
            if not _plain_convblock(blk):
                continue
            state: dict = {}

            def forward(self, x, _st=state):
                if _st.get("ok") is False or x.dim() != 3 or x.dtype != torch.float32:
                    return type(self).forward(self, x)
                y = _host_convblock_forward(self, x)
                if _st.get("ok") is None:                       # the first call: against the vendored forward
                    ref = type(self).forward(self, x)
                    _st["ok"] = bool(torch.equal(ref.view(torch.int32), y.view(torch.int32)))
                    if not _st["ok"]:
                        return ref
                return y

            blk.forward = types.MethodType(forward, blk)
            n += 1
    return n


def _branch_head_ok(br) -> bool:
    import torch.nn as nn
    head = getattr(br, "sr_head", None)
    return (type(br).__name__ == "SRBranch" and bool(getattr(br, "dual", False))
            and isinstance(head, nn.Sequential) and len(head) == 3
            and isinstance(head[0], nn.Linear) and isinstance(head[1], nn.GELU)
            and isinstance(head[2], nn.Linear) and head[2].bias is not None
            and head[2].out_features == 2 * int(br.m))


def _gpu_ok(name: str, device) -> bool:
    """gpu_kernel_check's verdict for one of SETSUNA's kernels whose check needs no model (gpu_checks.CHECKS)."""
    from setsuna._support import gpu_kernel_check as _gkc
    from . import gpu_checks as _gc
    return _gkc.use(name, device, _gc.CHECKS[name])


def _pooled_tail_ok(models, cfg, use_ac: bool) -> bool:
    """The Triton pooled tail applies: kernel available and enabled, fp32 forward, every model is the
    plain gate-mixed dual (bulk + one corner, no count adapter) with the expected head."""
    if _triton is None or not _POOLED_TAIL_ENABLED or use_ac or not bool(cfg.get("DUAL", False)):
        return False
    if not all(_dual_fast_path_ok(m) for m in models):
        return False
    if not all(_branch_head_ok(m.bulk) and _branch_head_ok(m.corner) for m in models):
        return False
    d = {int(br.sr_head[2].in_features) for m in models for br in (m.bulk, m.corner)}
    mm = {int(br.m) for m in models for br in (m.bulk, m.corner)}
    if not (len(d) == 1 and len(mm) == 1 and mm == {int(cfg["M"])}):
        return False
    device = next((p.device for m in models for p in m.parameters()), None)
    return device is not None and device.type == "cuda" and _gpu_ok("setsuna.pooled_head", device)


def _head_stack(models):
    """(WT (2N, 2, M, D), BIAS (2N, 2, M)) float32 device tensors of the dual heads, built once per
    ensemble: index [2*model + branch, head] with head 0 = count, 1 = binary."""
    cache = getattr(models[0], "_jscope_head_stack", None)
    if cache is None:
        Ws, Bs = [], []
        for m in models:
            for br in (m.bulk, m.corner):
                lin = br.sr_head[2]
                mm = int(br.m)
                Ws.append(torch.stack([lin.weight[:mm], lin.weight[mm:]]))     # (2, M, D)
                Bs.append(torch.stack([lin.bias[:mm], lin.bias[mm:]]))         # (2, M)
        W = torch.stack(Ws).float()                                            # (T, 2, M, D)
        D = int(W.shape[-1])
        DP = _pad_pow2(D)
        WT = torch.zeros((W.shape[0], 2, DP, W.shape[2]), dtype=torch.float32, device=W.device)
        WT[:, :, :D, :] = W.permute(0, 1, 3, 2)
        cache = (WT.contiguous(), torch.stack(Bs).contiguous().float(), D, DP)
        models[0]._jscope_head_stack = cache
    return cache


def _pad_pow2(d: int) -> int:
    p = 16
    while p < int(d):
        p *= 2
    return p


def _pool_tile(pool_k: int) -> tuple[int, int, int]:
    """(rows, columns, warps) of one program: 64 x 64 x 8 measured fastest (3.4 ms per 512-window
    batch at M=512, D=24, 3 models); JSCOPE_SETSUNA_POOL_ROWS overrides the rows for experiments."""
    R = int(_POOL_TILE_ROWS_ENV) if _POOL_TILE_ROWS_ENV.isdigit() else 64
    return R, 64, 8


# The head prologue (sr_head[0] + exact GELU) as one kernel per branch writing straight into the
# padded head stack: drops a cuBLAS GEMM + GELU pair per branch and the stack / pad copies of the
# stack (~260 MB per 512-window batch). fp32 FMA dot with zero-padded K; reorder-level differences.
_HEAD_PROLOGUE_FUSED = os.environ.get("JSCOPE_SETSUNA_HEAD_PROLOGUE", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _head_prologue_params(models, D: int, DP: int):
    """Per branch, in _head_stack order: sr_head[0] as a zero-padded transposed weight (DP, DP) and
    bias (DP,) for _proj_gelu_fwd; None when any branch's prologue is not Linear(d, d) -> exact GELU
    (the eager prologue is used then). Built once per ensemble."""
    cache = getattr(models[0], "_jscope_head_prologue", None)
    if cache is None:
        import torch.nn as nn
        params = []
        for m in models:
            for br in (m.bulk, m.corner):
                head = getattr(br, "sr_head", None)
                lin = head[0] if isinstance(head, nn.Sequential) and len(head) >= 2 else None
                act = head[1] if lin is not None else None
                if not (isinstance(lin, nn.Linear) and lin.in_features == D and lin.out_features == D
                        and lin.bias is not None and isinstance(act, nn.GELU)
                        and str(getattr(act, "approximate", "none")) == "none"):
                    params = None
                    break
                with torch.no_grad():
                    params.append((_pad_mat_t(lin.weight, DP, DP), _pad_vec(lin.bias, DP)))
            if params is None:
                break
        cache = (params,)
        models[0]._jscope_head_prologue = cache
    return cache[0]


# The six branch encoders and their head prologues as ONE runtime-compiled CUDA kernel per window batch
# (_setsuna_cuda.CudaEncoders): a thread block per (window, branch), a thread per time step, every
# intermediate on the SM, the band attention on the tensor cores in the fp16x3 arithmetic of
# _band_attn_fwd (its correction products accumulated in fp16, see _setsuna_cuda). Through the Triton
# kernels the same encoders cost 7-8 ms per 512-window batch, mostly row work at a fraction of the fp32
# rate and the round trips between ~100 launches; the kernel takes 4.0 ms. fp32 reorder-level
# differences (H1 relRMS 1-4e-7 vs the Triton path). Needs the fp16x3
# attention, windows of <= 384 frames and a GPU with m16n8k16 tensor cores (sm_80+); anything else, a
# failed compile or a failed self-test keeps the Triton path. JSCOPE_SETSUNA_CUDA_ENCODER=0 opts out.
_CUDA_ENCODER_ENABLED = os.environ.get("JSCOPE_SETSUNA_CUDA_ENCODER", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _cuda_encoder_for(models, xb, ind_t, nu_t, fps_t, ct_t, eff_window, D: int, DP: int):
    """The ensemble's _setsuna_cuda.CudaEncoders when it serves windows of this length, else None. Built once per
    ensemble; checked against the Triton path once per window length (eagerly: never during a capture)."""
    st = getattr(models[0], "_jscope_cuda_enc", None)
    if st is None:
        st = {"enc": None, "ok": {}}
        if _CUDA_ENCODER_ENABLED and xb.device.type == "cuda" and _triton is not None and D == 24 and DP == 32:
            try:
                from . import _setsuna_cuda
                st["enc"] = _setsuna_cuda.CudaEncoders(models, xb.device)
            except Exception:
                st["enc"] = None
        models[0]._jscope_cuda_enc = st
    enc = st["enc"]
    L = int(xb.shape[1])
    if enc is None or not enc.supports(L):
        return None
    ok = st["ok"].get(L)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        from setsuna._support import gpu_kernel_check as _gkc
        from . import gpu_checks as _gc
        ok = _gkc.use(f"setsuna.encoder.L{L}", xb.device,
                      lambda c, dev: _gc.check_encoder(c, dev, enc, models, xb, ind_t, nu_t, fps_t, ct_t, eff_window))
        st["ok"][L] = ok
    return enc if ok else None


def _cuda_encoder_selftest(enc, models, xb, ind_t, nu_t, fps_t, ct_t, eff_window, D: int, DP: int) -> bool:
    """Four windows of a fixed pseudo-random trace through both paths: finite, relRMS < 2e-5 (measured 1-4e-7)."""
    try:
        n = min(4, int(xb.shape[0]))
        L = int(xb.shape[1])
        g = torch.Generator(device=xb.device).manual_seed(1234)
        x = torch.randn((n, L), generator=g, device=xb.device, dtype=torch.float32)
        args = (ind_t[:n], nu_t[:n], fps_t[:n], ct_t[:n])
        P0 = _head_prologue_params(models, D, DP)
        branches = [br for m in models for br in (m.bulk, m.corner)]
        ref = torch.empty((len(branches), n * L, DP), dtype=torch.float32, device=xb.device)
        # the reference forward without a torch.compile probe (eager when none has run): the kernel path never
        # compiles, and the probe cost ~2 s here; compiled and eager references differ by ~1e-7, 1/100 of the bound
        _NO_PROBE.on = True
        try:
            for t, br in enumerate(branches):
                h = br.features(x, *args, eff_window).reshape(-1, D).contiguous()
                w0t, b0 = P0[t]
                _proj_gelu_fwd[(_triton.cdiv(int(h.shape[0]), 64),)](
                    h, ref[t], w0t, b0, int(h.shape[0]), D=D, DP=DP, O=D, OP=DP, BL=64, num_warps=4)
        finally:
            _NO_PROBE.on = False
        out = torch.full_like(ref, float("nan"))
        enc.h1(x, *args, eff_window, out)
        a, b = out.double(), ref.double()
        rel = float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30))
        return bool(torch.isfinite(a).all()) and rel < 2e-5
    except Exception:
        return False


# The pooled dual head as a runtime-compiled CUDA kernel (_setsuna_cuda.CudaHeadPool): a thread per fine bin with its
# heads' weights in registers, the rows' features broadcast from shared memory -- the Triton kernel staged the
# weights through shared memory per k step and ran at a third of the fp32 rate. 512 windows of 384 frames: 1.5 ms vs
# 2.3 ms; pooled outputs relRMS 5-8e-8 vs the Triton kernel (the model / bin sums run in another order). Pool
# factors 32 and 64; anything else keeps the Triton kernel, as does a failed compile or self-test (per variant).
# JSCOPE_SETSUNA_CUDA_HEAD_POOL=0 opts out.
_CUDA_HEAD_POOL_ENABLED = os.environ.get("JSCOPE_SETSUNA_CUDA_HEAD_POOL", "1").strip().lower() not in (
    "0", "false", "no", "off")
# ... with the spike decoders' fine binary stream (and the timing decoder's boundary parts) too: the models' sum per
# bin in the Triton kernel's model order, each head's dot in another order (fine stream relRMS ~6e-8); the head's
# time with the stream 4.1 -> 2.6 ms per ROI on the 1450-ROI 60.9 Hz bench session, where 17 of 633,494 timing
# spikes and 1 of 380,622 detection spikes moved by one fine bin (0.03 ms). JSCOPE_SETSUNA_CUDA_HEAD_PF=0 keeps the
# Triton kernel there.
_CUDA_HEAD_PF_ENABLED = os.environ.get("JSCOPE_SETSUNA_CUDA_HEAD_PF", "1").strip().lower() not in (
    "0", "false", "no", "off")


_CUDA_TAIL_ENABLED = os.environ.get("JSCOPE_SETSUNA_CUDA_TAIL", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _cuda_tail_state(models, device):
    """{"t": a _setsuna_cuda.CudaPooledTail or None, "ok": None until its first-use check, then bool}."""
    st = getattr(models[0], "_jscope_cuda_tail", None)
    if st is None:
        st = {"t": None, "ok": None}
        if _CUDA_TAIL_ENABLED and device.type == "cuda":
            try:
                from . import _setsuna_cuda
                st["t"] = _setsuna_cuda.CudaPooledTail(device)
            except Exception:
                st["t"] = None
        models[0]._jscope_cuda_tail = st
    return st


# The fine binary stream's overlap-add as one pass per batch (_setsuna_cuda.CudaSlabAdd: a region's two half-windows
# added and divided, the same float32 operations as torch.add then div_), zeroing only the regions that accumulate
# across batches: 0.58 + 0.41 ms of adds and divisions and a 308 MB zero fill per ROI become one ~0.35 ms pass, the
# same bits. JSCOPE_SETSUNA_PF_SLAB=0 keeps the torch ops.
_PF_SLAB_ENABLED = os.environ.get("JSCOPE_SETSUNA_PF_SLAB", "1").strip().lower() not in ("0", "false", "no", "off")
_PF_SLAB: dict = {}


def _cuda_slab(device):
    """The device's CudaSlabAdd, or None (disabled, not CUDA, or it did not compile)."""
    if not _PF_SLAB_ENABLED or device.type != "cuda":
        return None
    key = (device.type, device.index)
    if key not in _PF_SLAB:
        try:
            from . import _setsuna_cuda
            _PF_SLAB[key] = _setsuna_cuda.CudaSlabAdd(device) if _gpu_ok("setsuna.slab", device) else None
        except Exception:
            _PF_SLAB[key] = None
    return _PF_SLAB[key]


# Around the head's stream mode the rest of the spike paths' assembly runs as three kernels
# (_setsuna_cuda.CudaFineAssembly): each batch's overlap-add of the pooled sums, the boundary parts and the fine
# stream's edge regions, and the ROI's tail (the flush window, the divisions, the rate and the per-frame binary
# mass) -- ~15 torch ops per batch and ~35 per ROI, each ~1 us of work behind ~3 us of launch gap on the GPU. The
# same float operations in the same order; checked once per configuration against the torch ops on the ROI
# (predict_fine_outputs). JSCOPE_SETSUNA_FINE_FUSED=0 keeps the torch ops.
_FINE_FUSED_ENABLED = os.environ.get("JSCOPE_SETSUNA_FINE_FUSED", "1").strip().lower() not in (
    "0", "false", "no", "off")
_FINE_ASM: dict = {}
_FINE_FUSED_OK: dict = {}


def _fine_assembly(device):
    """The device's CudaFineAssembly, or None (disabled, not CUDA, or it did not compile)."""
    if not _FINE_FUSED_ENABLED or device.type != "cuda":
        return None
    key = (device.type, device.index)
    if key not in _FINE_ASM:
        try:
            from . import _setsuna_cuda
            _FINE_ASM[key] = _setsuna_cuda.CudaFineAssembly(device)
        except Exception:
            _FINE_ASM[key] = None
    return _FINE_ASM[key]


# JSCOPE_SETSUNA_CMAX=0: the timing decoder's tile bounds from the stream itself, not from its writers' chunk maxima
_CMAX_ENABLED = os.environ.get("JSCOPE_SETSUNA_CMAX", "1").strip().lower() not in ("0", "false", "no", "off")


def _chunk_max(pf, chunk: int):
    """The fine stream's per-chunk maxima as its writers keep them: the largest value of each chunk of ``chunk``
    bins, NaN where one is negative or NaN."""
    x = pf.reshape(-1, int(chunk))
    return torch.where(((x < 0) | torch.isnan(x)).any(dim=1), torch.full((), float("nan"), device=x.device),
                       x.amax(dim=1))


def _bits_equal(a, b) -> bool:
    """Same shape and the same bits (NaN and signed zeros included)."""
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = {torch.float64: torch.int64, torch.float32: torch.int32, torch.float16: torch.int16}.get(a.dtype)
    a, b = a.contiguous(), b.contiguous()
    return bool(torch.equal(a.view(iv), b.view(iv))) if iv is not None else bool(torch.equal(a, b))


_ROW_OFFSETS: dict = {}


def _row_offsets(n: int, n0: int, M: int, sb: int, device):
    """Per frame g of an n-frame trace (n0 real frames) the offset of the timing decoder's segment boundary inside it
    (native bins; 0: none, or at the frame's first bin): boundaries at s * sb, s >= 1, inside the real frames.
    int32 on the device and read-only: every ROI of a session shares one (built once per geometry, with an event
    the using stream waits on)."""
    key = (int(n), int(n0), int(M), int(sb), str(device))
    hit = _ROW_OFFSETS.get(key)
    if hit is None:
        g0 = torch.arange(n, dtype=torch.int64, device=device) * M
        b = torch.div(g0 + (M - 1), sb, rounding_mode="floor") * sb           # the last boundary <= the frame's end
        o = torch.where((b > g0) & (b < n0 * M), b - g0,
                        torch.zeros((), dtype=torch.int64, device=device)).to(torch.int32)
        if len(_ROW_OFFSETS) >= 16:
            return o                                   # (never dropped once cached: another stream may read it)
        ev = torch.cuda.Event()
        ev.record()
        hit = _ROW_OFFSETS[key] = (o, ev, torch.cuda.current_stream(device).cuda_stream)
    cur = torch.cuda.current_stream(device)
    if cur.cuda_stream != hit[2] and not torch.cuda.is_current_stream_capturing():
        cur.wait_event(hit[1])
    return hit[0]


def _cuda_head_pool_state(models, WT, BIAS, M: int, device):
    """{"hp": the ensemble's CudaHeadPool or None, "ok": {pool factor: self-test result}}, built once."""
    st = getattr(models[0], "_jscope_cuda_hp", None)
    if st is None:
        st = {"hp": None, "ok": {}}
        if _CUDA_HEAD_POOL_ENABLED and device.type == "cuda":
            try:
                from . import _setsuna_cuda
                st["hp"] = _setsuna_cuda.CudaHeadPool(WT, BIAS, len(models), M, device)
            except Exception:
                st["hp"] = None
        models[0]._jscope_cuda_hp = st
    return st


# The fine binary stream straight from the CUDA head (_setsuna_cuda.CudaHeadPool's stream mode): the head kernel
# takes each frame shared by two windows of a batch as a pair of rows and writes their sum, divided, into the stream
# (the same float32 operations as CudaSlabAdd); only the rows without a partner in the batch come back, for the
# overlap-add's edges. The (rows, M) per-batch stream and the slab pass over it go. Checked once per pool factor,
# boundary-part flag and window length on a synthetic batch (_check_head_stream). JSCOPE_SETSUNA_PF_STREAM=0 keeps the
# per-batch stream and the slab pass.
_PF_STREAM_ENABLED = os.environ.get("JSCOPE_SETSUNA_PF_STREAM", "1").strip().lower() not in ("0", "false", "no", "off")
_STREAM_SP_LEN = 12                      # int64s per batch: _setsuna_cuda.CudaHeadPool.stream_geometry


def _head_stream_ok(models, cfg, device, pool_k: int, want_part: bool, W: int) -> bool:
    """Whether the CUDA head writes the fine binary stream itself (see _PF_STREAM_ENABLED) for pool factor pool_k,
    with the timing decoder's boundary parts or not, and windows of W frames."""
    if not (_PF_STREAM_ENABLED and _CUDA_HEAD_PF_ENABLED and device.type == "cuda") or int(W) < 2 or int(W) % 2:
        return False
    WT, BIAS, D, DP = _head_stack(models)
    M = int(cfg["M"])
    kk = min(int(pool_k), 64)
    st = _cuda_head_pool_state(models, WT, BIAS, M, device)
    hp = st["hp"]
    if hp is None or int(DP) != 32 or not hp.supports(kk):
        return False
    key = ("stream", kk, bool(want_part), int(W))
    ok = st["ok"].get(key)
    if ok is None:
        # once per process, GPU and configuration (gpu_kernel_check): the stream mode's outputs against the plain
        # CUDA head's rows and the torch overlap-add, bit for bit
        from setsuna._support import gpu_kernel_check as _gkc

        def _check(c, dev):
            if not _check_head_stream(hp, WT, BIAS, len(models), M, int(DP), int(pool_k), bool(want_part), int(W),
                                      dev):
                c.fail("stream mode outputs")
        ok = _gkc.use(f"setsuna.head_stream.k{kk}{'.part' if want_part else ''}.W{int(W)}", device, _check)
        st["ok"][key] = ok
    return ok


def _check_head_stream(hp, WT, BIAS, n_models: int, M: int, DP: int, pool_k: int, want_part: bool, W: int,
                       device) -> bool:
    """The stream mode on synthetic batches (interior regions divided or not, with and without the flush window,
    a flush window alone): the plain CUDA head against the Triton kernel as _ensemble_batch_pooled's self-test
    (relRMS < 1e-5), then the stream mode's regions, edge rows, pooled sums and boundary parts against the plain
    CUDA head's rows bit for bit."""
    kk = min(int(pool_k), 64)
    S = W // 2
    d = float(np.float32(2.0) * np.float32(n_models))
    d_dev = torch.full((), d, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(20260926)

    def _same(a, b):
        iv = torch.int64 if a.dtype == torch.float64 else torch.int32
        return a.shape == b.shape and torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))

    for n_reg, flush, region0, r_tail in ((4, False, 2, 1 << 30), (3, True, 0, 2), (0, True, 5, 0)):
        B = n_reg + int(flush)
        rows = B * W
        H1 = torch.randn((2 * n_models, rows, DP), generator=gen, device=device) * 0.5
        H1[:, :, 24:] = 0.0
        G = torch.rand((n_models, B), generator=gen, device=device)
        ro = p2 = None
        if want_part:
            ro = torch.where(torch.rand(rows, generator=gen, device=device) < 0.25,
                             torch.randint(1, M, (rows,), generator=gen, device=device),
                             torch.zeros((), dtype=torch.int64, device=device)).to(torch.int32)
            p2 = torch.empty((rows, M // (32 * hp.NW)), dtype=torch.float64, device=device)
        oc = torch.empty((rows, M // kk), dtype=torch.float64, device=device)
        ob = torch.empty_like(oc)
        opf = torch.empty((rows, M), dtype=torch.float32, device=device)
        hp.run(H1, G, oc, ob, rows, W, kk, out_pf=opf, row_off=ro, out_part=p2)
        tc, tb, tpf = torch.empty_like(oc), torch.empty_like(ob), torch.empty_like(opf)
        tpart = torch.empty(rows, dtype=torch.float64, device=device) if want_part else tc
        _head_pool_triton(H1, WT, BIAS, G, tc, tb, tpf, tb, ro, tpart, rows, W, n_models, DP, M, int(pool_k), kk,
                          True, False, want_part)
        pairs = [(oc, tc), (ob, tb), (opf, tpf)] + ([(p2.sum(dim=1), tpart)] if want_part else [])
        for a, b in pairs:
            rel = float((a.double() - b.double()).pow(2).mean().sqrt()
                        / b.double().pow(2).mean().sqrt().clamp_min(1e-300))
            if not (bool(torch.isfinite(a).all()) and rel < 1e-5):
                return False
        n_st = region0 + max(n_reg, 1) + 1
        stream = torch.full((n_st * S * M,), float("nan"), dtype=torch.float32, device=device)
        edges = torch.full((2 * S + W, M), float("nan"), dtype=torch.float32, device=device)
        chunk = 32 * hp.NW
        cm = torch.full((n_st * S * (M // chunk),), float("nan"), dtype=torch.float32, device=device)
        sp = torch.tensor(hp.stream_geometry(stream.data_ptr(), region0, n_reg, flush, W, r_tail, d, cm.data_ptr()),
                          dtype=torch.int64, device=device)
        sc, sb = torch.empty_like(oc), torch.empty_like(ob)
        sp2 = torch.empty_like(p2) if want_part else None
        hp.run(H1, G, sc, sb, rows, W, kk, row_off=ro, out_part=sp2, sp=sp, edge_out=edges,
               blocks=hp.stream_blocks(B, W))
        v = opf.view(B, W, M)
        exp = torch.full_like(stream, float("nan"))
        exp_cm = torch.full_like(cm, float("nan"))
        for rl in range(1, n_reg):
            s = torch.add(v[rl - 1, S:], v[rl, :S])
            if region0 + rl < r_tail:
                s = s / d_dev
                exp_cm[(region0 + rl) * S * (M // chunk):(region0 + rl + 1) * S * (M // chunk)] = _chunk_max(s, chunk)
            exp[(region0 + rl) * S * M:(region0 + rl + 1) * S * M] = s.reshape(-1)
        exp_e = torch.full_like(edges, float("nan"))
        if n_reg > 0:
            exp_e[:S] = v[0, :S]
            exp_e[S:2 * S] = v[n_reg - 1, S:]
        if flush:
            exp_e[2 * S:] = v[n_reg]
        if not (_same(stream, exp) and _same(edges, exp_e) and _same(sc, oc) and _same(sb, ob)
                and (not want_part or _same(sp2, p2)) and _same(cm, exp_cm)):
            return False
    return True


def _head_pool_triton(H1, WT, BIAS, G, out_c, out_b, out_pf, out_lf, row_off, out_part, rows: int, L: int,
                      n_models: int, DP: int, M: int, k: int, kk: int, want_pf: bool, want_lf: bool,
                      want_part: bool) -> None:
    """_head_pool_fwd over one batch (_ensemble_batch_pooled's outputs; unused outputs name a live tensor)."""
    R, C, warps = _pool_tile(k)
    grid = (_triton.cdiv(rows, R),)
    _head_pool_fwd[grid](H1, WT, BIAS, G, out_c, out_b, out_pf, out_lf,
                         (row_off if want_part else out_c), out_part, rows, L,
                         H1.stride(0), H1.stride(1), WT.stride(0), WT.stride(1), WT.stride(2),
                         BIAS.stride(0), BIAS.stride(1), G.stride(0), out_c.stride(0),
                         N=n_models, DP=DP, M=M, K=kk, R=R, C=C, WRITE_PF=bool(want_pf), SUM32=_POOL_SUM32,
                         SP_LOGCORR=_SOFTPLUS_LOGCORR, WRITE_LF=bool(want_lf), WRITE_PART=bool(want_part),
                         num_warps=warps)


def _ensemble_batch_pooled(models, cfg, xb, ind_t, nu_t, fps_t, ct_t, eff_window, pool_k: int,
                           want_pf: bool = False, ids=None, want_lf: bool = False, row_off=None,
                           stream_sp=None):
    """One batch of windows through the ensemble -> per-window K-pooled SUMS over models of the
    count rate and the spike probability, (B, W*M/K) float64, via _head_pool_fwd. With ``want_pf``
    also the fp32 ensemble-summed binary stream per window, (B, W*M) -- the full path's ``prb`` --
    and with ``want_lf`` (which needs ``want_pf``) the count stream likewise, the full path's ``pr``.
    ``row_off`` ((B*W,) int32, with ``want_pf``): one more output, (B, W) float64, each row's count
    sum over its first row_off native bins (_head_pool_fwd's WRITE_PART). ``stream_sp`` (with
    ``want_pf``, not ``want_lf``; the caller checked _head_stream_ok): the CUDA head's stream mode --
    the frames two windows of the batch share go straight into the fine stream the (11,) int64 device
    tensor describes (_setsuna_cuda.CudaHeadPool.stream_geometry), and the third output holds the
    other rows instead, (W + 2 * (W // 2), M) float32: window 0's first half, the last regular
    window's second half, then the flush window."""
    B = int(xb.shape[0])
    WT, BIAS, D, DP = _head_stack(models)
    P0 = _head_prologue_params(models, D, DP) if (_HEAD_PROLOGUE_FUSED and _gpu_ok("setsuna.head_prologue",
                                                                                  xb.device)) else None
    h1s, gates = [], []
    H1 = None
    enc = _cuda_encoder_for(models, xb, ind_t, nu_t, fps_t, ct_t, eff_window, D, DP) if P0 is not None else None
    if enc is not None:
        H1 = torch.empty((2 * len(models), B * int(xb.shape[1]), DP), dtype=torch.float32, device=xb.device)
        enc.h1(xb, ind_t, nu_t, fps_t, ct_t, eff_window, H1, ids=ids)
        gates = [m.compute_gate(nu_t, fps_t, ind_t, ct_t) for m in models]              # (B,) each
    t = 0
    for m in (models if enc is None else ()):
        for br in (m.bulk, m.corner):
            h = br.features(xb, ind_t, nu_t, fps_t, ct_t, eff_window)          # (B, L, d)
            if P0 is not None:
                # sr_head[0] + GELU straight into slab t of the padded head stack
                h2 = h.reshape(-1, D)
                if not h2.is_contiguous():
                    h2 = h2.contiguous()
                if H1 is None:
                    H1 = torch.empty((2 * len(models), int(h2.shape[0]), DP), dtype=torch.float32,
                                     device=xb.device)
                w0t, b0 = P0[t]
                _proj_gelu_fwd[(_triton.cdiv(int(h2.shape[0]), 64),)](
                    h2, H1[t], w0t, b0, int(h2.shape[0]), D=D, DP=DP, O=D, OP=DP, BL=64, num_warps=4)
            else:
                h1 = br.sr_head[1](br.sr_head[0](h))                            # (B, L, D)
                h1s.append(h1.reshape(-1, h1.shape[-1]))
            t += 1
        gates.append(m.compute_gate(nu_t, fps_t, ind_t, ct_t))                 # (B,)
    if H1 is None:
        H1 = torch.stack(h1s)                                                   # (2N, B*L, D)
        if DP != D:
            H1 = F.pad(H1, (0, DP - D))                                         # zero columns D..DP-1
    G = torch.stack(gates).contiguous().float()                                 # (N, B)
    M = int(cfg["M"])
    rows = int(H1.shape[1])
    L = rows // B
    k = int(pool_k)
    kk = min(k, 64)                      # the kernel pools by kk; coarser factors finish below
    Pk = M // kk
    out_c = torch.empty((rows, Pk), dtype=torch.float64, device=xb.device)
    out_b = torch.empty((rows, Pk), dtype=torch.float64, device=xb.device)
    out_pf = torch.empty(((L + 2 * (L // 2)) if stream_sp is not None else rows, M), dtype=torch.float32,
                         device=xb.device) if want_pf else out_b
    out_lf = torch.empty((rows, M), dtype=torch.float32, device=xb.device) if want_lf else out_b
    want_part = row_off is not None
    out_part = torch.empty(rows, dtype=torch.float64, device=xb.device) if want_part else out_c
    # The CUDA kernel also writes the fine binary stream and the boundary parts (JSCOPE_SETSUNA_CUDA_HEAD_PF=0: the
    # Triton kernel does, as before); the fine count stream (want_lf) stays with the Triton kernel.
    hp_st = (None if want_lf or ((want_pf or want_part) and not _CUDA_HEAD_PF_ENABLED)
             else _cuda_head_pool_state(models, WT, BIAS, M, xb.device))
    hp = hp_st["hp"] if hp_st is not None else None
    hp_ok = None
    variant = (kk, bool(want_pf), want_part)
    if hp is not None and H1.is_contiguous() and int(H1.shape[-1]) == 32 and hp.supports(kk):
        # (the stream mode was checked once, by _head_stream_ok)
        hp_ok = True if stream_sp is not None else hp_st["ok"].get(variant)
        if hp_ok is None and torch.cuda.is_current_stream_capturing():
            hp = None                        # the one-time check runs eagerly (the warm-up before a capture)
        elif hp_ok is None:
            # once per process, GPU and variant: against the Triton kernel and a float64 head (gpu_kernel_check)
            from setsuna._support import gpu_kernel_check as _gkc
            from . import gpu_checks as _gc
            _kk, _pf, _part = variant
            hp_ok = _gkc.use(_gc.head_pool_name(_kk, _pf, _part), xb.device,
                             lambda c, dev: _gc.check_head_pool(c, dev, hp, WT, BIAS, len(models), M, _kk, _pf,
                                                                _part, k))
            hp_st["ok"][variant] = hp_ok
            if hp_ok:
                # this ensemble's first batch of the variant keeps the Triton kernel's values, as before the check
                # moved into gpu_kernel_check (the same outputs as before; later batches take the CUDA kernel)
                hp_ok = False
    else:
        hp = None
    if stream_sp is not None and (hp is None or not want_pf or want_lf):
        raise RuntimeError("SETSUNA: the head's stream mode needs the CUDA head-pool kernel")

    def _run_hp(oc, ob, opf, opart):
        # the parts come by 128-bin column block; their float64 sum per row
        p2 = torch.empty((rows, M // (32 * hp.NW)), dtype=torch.float64, device=xb.device) if want_part else None
        if stream_sp is not None:
            hp.run(H1, G, oc, ob, rows, L, kk, row_off=(row_off if want_part else None), out_part=p2,
                   sp=stream_sp, edge_out=opf, blocks=hp.stream_blocks(B, L))
        else:
            hp.run(H1, G, oc, ob, rows, L, kk, out_pf=(opf if want_pf else None),
                   row_off=(row_off if want_part else None), out_part=p2)
        if want_part:
            torch.sum(p2, dim=1, out=opart)

    if hp is not None and hp_ok:
        _run_hp(out_c, out_b, out_pf, out_part)
    else:
        _head_pool_triton(H1, WT, BIAS, G, out_c, out_b, out_pf, out_lf, row_off, out_part, rows, L, len(models), DP,
                          M, k, kk, want_pf, want_lf, want_part)
    if k > kk:                           # e.g. rate-only (k = M): sum the 64-bin partials in float64
        out_c = out_c.view(rows, M // k, k // kk).sum(dim=2)
        out_b = out_b.view(rows, M // k, k // kk).sum(dim=2)
    Pn = M // k
    if want_lf:
        res = (out_c.reshape(B, L * Pn), out_b.reshape(B, L * Pn), out_pf.view(B, L * M), out_lf.view(B, L * M))
    elif want_pf:
        res = (out_c.reshape(B, L * Pn), out_b.reshape(B, L * Pn),
               out_pf if stream_sp is not None else out_pf.view(B, L * M))
    else:
        res = (out_c.reshape(B, L * Pn), out_b.reshape(B, L * Pn))
    return (*res, out_part.view(B, L)) if want_part else res


_TRACE_PIN = {"bufs": [None, None], "events": [None, None], "i": 0}
_SP_PIN = {"bufs": [None, None], "events": [None, None], "i": 0}


def _upload_trace(xz: np.ndarray, device) -> "torch.Tensor":
    """xz (n,) float32 host -> device, through a pinned staging buffer and a non-blocking copy, so the
    upload queues behind whatever the stream is running instead of blocking the host until the
    stream drains (which a pageable copy does -- measured: torch.tensor(a, device=cuda) returned
    only after the queued kernels finished)."""
    return _upload_pinned(np.ascontiguousarray(xz, dtype=np.float32), device, _TRACE_PIN, 1 << 18)


def _upload_pinned(a: np.ndarray, device, ring: dict, min_numel: int) -> "torch.Tensor":
    """a (contiguous) host -> device (flat) through the ring's pinned staging buffers and a non-blocking copy. Two
    staging buffers alternate; one is rewritten only after the event recorded behind its previous copy has fired."""
    n = int(a.size)
    i = ring["i"]
    ring["i"] = 1 - i
    ev = ring["events"][i]
    if ev is not None:
        ev.synchronize()
    src = torch.from_numpy(a.reshape(-1))
    buf = ring["bufs"][i]
    if buf is None or buf.numel() < n or buf.dtype != src.dtype:
        buf = torch.empty(max(n, int(min_numel)), dtype=src.dtype, pin_memory=True)
        ring["bufs"][i] = buf
    buf[:n].copy_(src)
    out = buf[:n].to(device, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    ring["events"][i] = ev
    return out


# The pooled forward's per-ROI upload (the trace, in the stream mode with the stream geometry) on a copy stream of
# its own, into one of three persistent device buffers. In the compute stream the copy sat between one ROI's last
# kernels and the next ROI's first, and the switch to the copy engine and back idled the GPU ~50-75 us per ROI
# (profiled, WDDM). Now the compute stream waits only for the copy's event, recorded well before it gets there; a
# buffer is rewritten three uploads later, after the event recorded behind its last use (_upload_ring's release).
# The same bytes. JSCOPE_SETSUNA_UPLOAD_STREAM=0 keeps the copy in the compute stream.
_UPLOAD_STREAM_ENABLED = os.environ.get("JSCOPE_SETSUNA_UPLOAD_STREAM", "1").strip().lower() not in (
    "0", "false", "no", "off")
_UP_RING: dict = {}


def _upload_ring(a: np.ndarray, device):
    """a (contiguous) host array -> (its bytes on the device as a flat uint8 tensor, release). The copy runs on the
    ring's own stream; the current stream waits for it. Call release() once the tensor's uses are queued on the
    current stream: its buffer is rewritten three uploads later, after them."""
    key = (device.type, device.index)
    st = _UP_RING.get(key)
    if st is None:
        st = _UP_RING[key] = {"stream": torch.cuda.Stream(device=device), "dev": [None] * 3, "used": [None] * 3,
                              "k": 0, "pin": {"bufs": [None, None], "events": [None, None], "i": 0}}
    cur = torch.cuda.current_stream(device)
    up = st["stream"]
    k = st["k"]
    st["k"] = (k + 1) % 3
    src = torch.from_numpy(np.ascontiguousarray(a).reshape(-1).view(np.uint8))
    n = int(src.numel())
    ring = st["pin"]
    i = ring["i"]
    ring["i"] = 1 - i
    if ring["events"][i] is not None:
        ring["events"][i].synchronize()             # this staging buffer's previous copy is done
    hb = ring["bufs"][i]
    if hb is None or hb.numel() < n:
        hb = ring["bufs"][i] = torch.empty(max(n, 1 << 22), dtype=torch.uint8, pin_memory=True)
    hb[:n].copy_(src)
    used = st["used"][k]
    db = st["dev"][k]
    if used == "pending" or db is None or db.numel() < n:
        # a release that never came (an exception mid-ROI), or a longer trace: wait until nothing uses the buffer
        torch.cuda.synchronize(device)
        used = None
        if db is None or db.numel() < n:
            db = st["dev"][k] = torch.empty(max(n, 1 << 22), dtype=torch.uint8, device=device)
    if used is not None:
        up.wait_event(used)                         # its uses three uploads ago are done
    with torch.cuda.stream(up):
        db[:n].copy_(hb[:n], non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(up)
    ring["events"][i] = ready
    cur.wait_event(ready)
    st["used"][k] = "pending"

    def release(slot=k, stream=cur):
        ev = torch.cuda.Event()
        ev.record(stream)
        st["used"][slot] = ev

    return db[:n], release


@torch.inference_mode()
@_strict_fp32()
def _predict_fine_cuda_pooled(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                              batch_windows: int = 512, pool_k: int = 32, fine_prob: bool = False,
                              defer_div: bool = False, fine_lam: bool = False, seg_bins=None, extras=None,
                              want16: bool = False, _no_fuse: bool = False, _cap=None):
    """_predict_fine_cuda with the pooled tail: the ensemble emits K-pooled float64 sums per window
    (_ensemble_batch_pooled, graph-replayed) and the overlap-add, coverage division and ensemble
    mean run on those in float64. Returns (lam_pooled, prob_pooled) float64 DEVICE tensors of
    length n0 * M / K -- the exact linear reductions of the fine streams, rounded once by the
    caller (rate to float32, pooled streams to float16). With ``fine_prob`` a third tensor: the
    full-resolution probability stream (n0 * M, float32) assembled from the kernel's fp32 binary
    sums with the same fp32 slab adds and division as _predict_fine_cuda -- for the spike-time
    decoders, which need sub-frame positions from the binary head only. ``fine_lam`` (with
    ``fine_prob``) adds a fourth: the count stream the same way -- with the third, what the timing
    decoder reads, in place of _predict_fine_cuda's (pred, pred_b). ``seg_bins`` (with ``fine_prob``,
    instead of ``fine_lam``; > M): the fourth is (n0,) float64 instead, per frame the count mass
    before the decoder segment boundary inside it (0 where none: boundaries at multiples of
    seg_bins native bins), overlap-added and divided as the pooled count sums -- with the per-frame
    mass, the timing decoder's per-segment counts without the 2 x 308 MB fine count stream.
    ``defer_div``: on the common path (half-overlapping windows, no padding, no fine stream) return
    the UNdivided sums and ``(n, Pn, S, n_regions, flush-window start or -1, n_models)`` for
    _setsuna_cuda.CudaPooledTail, which divides them together with the caller's rate and float16
    conversions. ``extras`` (a dict): on the fused assembly (_FINE_FUSED_ENABLED) filled with what
    the caller would otherwise compute from the returned sums -- ``rate32``, ``rate64`` (None when
    each frame has one pooled value: the first tensor itself), ``p_frame``, and with ``want16``
    ``lam16`` / ``prob16`` -- plus ``key`` for predict_fine_outputs' one-time check."""
    M = int(cfg["M"])
    k = int(pool_k)
    Pn = M // k
    n0 = len(xz)
    want_part = bool(fine_prob) and not fine_lam and seg_bins is not None
    if want_part and int(seg_bins) <= M:
        raise ValueError("_predict_fine_cuda_pooled: seg_bins must exceed M (one boundary per frame at most)")
    W = w_of_fps(cfg, eff)
    pad = 0
    if n0 < W:
        pad = W - n0
        xz = np.concatenate([xz, xz[-2::-1][:pad].astype(np.float32)]) if n0 > 1 \
            else np.full(W, float(xz[0]) if n0 else 0.0, np.float32)
    if len(xz) < W:
        xz = np.resize(xz, W).astype(np.float32)
        pad = W - n0
    n = len(xz)
    eff_window = (W if cfg["WMODE"] == "mask" else None)
    route_mode = (("hard_corner" if (eff <= cfg["HC_FPS"] and nu >= cfg["HC_NU"]) else "hard_bulk")
                  if cfg["HARDROUTE"] else "soft")
    stride = max(1, W // 2)
    starts = list(range(0, n - W + 1, stride))
    has_tail = bool(starts) and starts[-1] + W < n
    if has_tail:
        starts.append(n - W)
    n_regular = len(starts) - (1 if has_tail else 0)
    slabbed = (W % 2 == 0) and (stride * 2 == W) and n_regular > 0
    S = stride
    dt = torch.float64
    n_models = len(models)
    pred = torch.zeros(n * Pn, dtype=dt, device=device)
    pred_b = torch.zeros(n * Pn, dtype=dt, device=device)
    cov = None
    SP = S * Pn
    # the fine binary stream (fp32, as _predict_fine_cuda's pred_b) when requested
    Mb = M
    # its overlap-add as one pass per batch (_cuda_slab): the stream is then written, not accumulated, except the
    # regions noted in pf_edges, zeroed below
    slab = _cuda_slab(device) if (fine_prob and slabbed) else None
    # or the head kernel writes those regions itself (_head_stream_ok) and returns only the batch's edge rows
    stream = bool(fine_prob) and slabbed and not fine_lam and _head_stream_ok(models, cfg, device, k, want_part, W)
    # and the rest of the assembly as three kernels (_fine_assembly)
    asm = _fine_assembly(device) if (stream and not _no_fuse) else None
    fused = asm is not None and asm.supports(Pn)
    pf_written = slab is not None or stream
    # odd W: the fine stream's regular windows in one kernel per batch (setsuna_odd_fine), not the torch passes
    odd_fine = (_odd_fine_kernel(device) if (fine_prob and not fine_lam and not slabbed and W == 2 * S + 1
                                             and n_regular > 0) else None)
    pf = (torch.empty if (pf_written or odd_fine is not None) else torch.zeros)(n * Mb, dtype=torch.float32,
                                                                                device=device) \
        if fine_prob else None
    if odd_fine is not None and (n_regular + 1) * S + 1 < n:
        pf[((n_regular + 1) * S + 1) * Mb:].zero_()       # frames only the flush window reaches
    pf_edges = []                                    # regions accumulated with +=: zeroed first, divided at the end
    pf_div = float(np.float32(2.0) * np.float32(n_models))     # an interior region's divisor, as reg_count_f * N
    r_tail = ((n - W) // S) if has_tail else (n_regular + 1)   # the first region the flush window reaches
    sp_dev = None
    # with the timing decoder's boundary parts, the fine stream's per-chunk maxima from its writers (the head's stream
    # mode and the fused tail): the decoder's tile bounds read them instead of the 323 MB stream (_decode_cuda's
    # setsuna_tile_bounds_cmax)
    cmax = None
    if stream:
        # each batch's stream geometry goes up with the trace, in one pinned copy (the graphs copy their row in
        # before a replay)
        hp = models[0]._jscope_cuda_hp["hp"]
        cmax_chunk = 32 * hp.NW
        if fused and want_part and _CMAX_ENABLED and Mb % cmax_chunk == 0:
            cmax = torch.empty(n * (Mb // cmax_chunk), dtype=torch.float32, device=device)
        bw = int(batch_windows)
        geo = []
        for c0 in range(0, len(starts), bw):
            nb = min(bw, len(starts) - c0)
            fl = has_tail and c0 + nb == len(starts)
            geo.append(hp.stream_geometry(pf.data_ptr(), c0, nb - int(fl), fl, W, r_tail, pf_div,
                                          cmax.data_ptr() if cmax is not None else 0))
        geo = np.asarray(geo, dtype=np.int64)
    if stream and _cap is not None:
        # _pooled_whole's capture: the trace in its static device buffer; the geometry (it names this capture's own
        # stream buffers) copied from a pinned buffer the graph keeps
        gh = _cap["geo_host"]
        gh[: geo.size].copy_(torch.from_numpy(geo.reshape(-1)))
        sp_dev = gh[: geo.size].to(device, non_blocking=True).view(len(geo), -1)
        xz_dev, up_release = _cap["xz_dev"], None
    elif _cap is not None:
        xz_dev, up_release = _cap["xz_dev"], None
    elif stream:
        off = (4 * n + 7) // 8 * 8
        host = np.empty(off + geo.nbytes, dtype=np.uint8)
        host[:4 * n].view(np.float32)[:] = xz
        host[off:].view(np.int64)[:] = geo.reshape(-1)
        if _UPLOAD_STREAM_ENABLED:
            both, up_release = _upload_ring(host, device)
        else:
            both, up_release = _upload_pinned(host, device, _SP_PIN, 1 << 20), None
        xz_dev = both[:4 * n].view(torch.float32)
        sp_dev = both[off:off + geo.nbytes].view(torch.int64).view(len(geo), -1)
    elif _UPLOAD_STREAM_ENABLED:
        both, up_release = _upload_ring(np.ascontiguousarray(xz, dtype=np.float32), device)
        xz_dev = both.view(torch.float32)
    else:
        # the trace goes up once (pinned, non-blocking); the window batches are strided views of it,
        # copied into the graph's static input on the device -- no per-batch np.stack + upload
        xz_dev, up_release = _upload_trace(xz, device), None
    win_all = xz_dev.unfold(0, W, stride)                        # (n_regular, W): row i = xz[starts[i]:+W]
    lf = torch.zeros(n * Mb, dtype=torch.float32, device=device) if (fine_prob and fine_lam) else None
    part = o_win = o_dev = None
    if want_part:
        # per frame the offset of the segment boundary inside it (_row_offsets: once per geometry)
        o_dev = _row_offsets(n, n0, M, int(seg_bins), device)
        o_win = o_dev.unfold(0, W, stride)                                    # (n_regular, W), as win_all
        part = torch.zeros(n, dtype=dt, device=device)
    cov_f = None
    if slabbed:
        n_regions = n_regular + 1
        sl = pred[: n_regions * SP].view(n_regions, SP)
        sl_b = pred_b[: n_regions * SP].view(n_regions, SP)
        sl_p = part[: n_regions * S].view(n_regions, S) if part is not None else None
        if not fused:                                 # (the fused kernels count the windows themselves)
            reg_count = torch.full((n_regions,), 2.0, dtype=dt, device=device)
            # fill_ on the end slices, not item assignment: `t[0] = 1.0` copies the scalar from pageable host
            # memory, and a pageable host->device copy first synchronises the stream -- every ROI the host
            # stalled here until the previous ROI's forward had finished, then left the GPU idle while it
            # queued this one (~0.2 ms per ROI). Same values.
            reg_count[:1].fill_(1.0)
            reg_count[-1:].fill_(1.0)
        if pf is not None:
            sl_f = pf[: n_regions * S * Mb].view(n_regions, S * Mb)
            sl_l = lf[: n_regions * S * Mb].view(n_regions, S * Mb) if lf is not None else None
            if not fused:
                reg_count_f = torch.full((n_regions,), 2.0, dtype=torch.float32, device=device)
                reg_count_f[:1].fill_(1.0)
                reg_count_f[-1:].fill_(1.0)
            if pf_written and not fused:
                pf[n_regions * S * Mb:].zero_()          # past the regular windows: the flush window's alone
    else:
        cov = torch.zeros(n * Pn, dtype=dt, device=device)
        if pf is not None and odd_fine is None:
            cov_f = torch.zeros(n * Mb, dtype=torch.float32, device=device)
    tail_range = None
    flush = None                  # fused: the flush window's (first frame, (count, binary, part) rows, fine rows)
    for c0 in range(0, len(starts), int(batch_windows)):
        chunk = starts[c0:c0 + int(batch_windows)]
        B = len(chunk)
        n_reg_chunk = B - (1 if (has_tail and c0 + B == len(starts)) else 0)
        if n_reg_chunk == B:
            xb = win_all[c0:c0 + B]
        elif n_reg_chunk > 0:
            xb = torch.cat([win_all[c0:c0 + n_reg_chunk], xz_dev[n - W:].unsqueeze(0)])
        else:
            xb = xz_dev[n - W:].unsqueeze(0)
        pfl = pp = None
        if lf is not None:
            pr, prb, pff, pfl = _ensemble_batch_graphed(models, cfg, xb, nu, ind_id, ct_id, eff, eff_window,
                                                        route_mode, False, device, pool_k=k, want_pf=True,
                                                        want_lf=True, nu_static=(_cap["nu_t"] if _cap is not None else None))
        elif part is not None:
            # the rows' boundary offsets, window by window as xb
            if n_reg_chunk == B:
                ro = o_win[c0:c0 + B]
            elif n_reg_chunk > 0:
                ro = torch.cat([o_win[c0:c0 + n_reg_chunk], o_dev[n - W:].unsqueeze(0)])
            else:
                ro = o_dev[n - W:].unsqueeze(0)
            pr, prb, pff, pp = _ensemble_batch_graphed(models, cfg, xb, nu, ind_id, ct_id, eff, eff_window,
                                                       route_mode, False, device, pool_k=k, want_pf=True,
                                                       row_off=ro,
                                                       stream_sp=(sp_dev[c0 // int(batch_windows)] if stream
                                                                  else None), nu_static=(_cap["nu_t"] if _cap is not None else None))
        elif pf is not None:
            pr, prb, pff = _ensemble_batch_graphed(models, cfg, xb, nu, ind_id, ct_id, eff, eff_window,
                                                   route_mode, False, device, pool_k=k, want_pf=True,
                                                   stream_sp=(sp_dev[c0 // int(batch_windows)] if stream else None),
                                                   nu_static=(_cap["nu_t"] if _cap is not None else None))
        else:
            pr, prb = _ensemble_batch_graphed(models, cfg, xb, nu, ind_id, ct_id, eff, eff_window,
                                              route_mode, False, device, pool_k=k, nu_static=(_cap["nu_t"] if _cap is not None else None))
            pff = None
        n_reg_here = B - (1 if (has_tail and c0 + B == len(starts)) else 0)
        if fused:
            if n_reg_here > 0:
                asm.batch(pr, prb, pp, pff, pred, pred_b, part, pf, S, Pn, Mb, c0, n_reg_here)
            if n_reg_here < B:
                s0 = chunk[-1]
                flush = (s0, (pr[B - 1], prb[B - 1], pp[B - 1] if pp is not None else None), pff[2 * S:])
                tail_range = (s0 * Pn, (s0 + W) * Pn)
            continue
        if slabbed and n_reg_here > 0:
            j0 = c0
            if pff is not None and pf_written:
                # the regions this batch alone covers are written (and divided) below; the two at its ends take
                # += from this batch and a neighbour: zeroed on their first touch, divided at the end
                if j0 == 0:
                    sl_f[0].zero_()
                    pf_edges.append(0)
                sl_f[j0 + n_reg_here].zero_()
                pf_edges.append(j0 + n_reg_here)
            if n_reg_here > 1:
                torch.add(pr[:n_reg_here - 1, SP:], pr[1:n_reg_here, :SP], out=sl[j0 + 1: j0 + n_reg_here])
                torch.add(prb[:n_reg_here - 1, SP:], prb[1:n_reg_here, :SP], out=sl_b[j0 + 1: j0 + n_reg_here])
                if stream:
                    pass                         # the head kernel wrote them (added and divided as below)
                elif pff is not None and slab is not None:
                    # interior regions (count 2) before the flush window: added and divided in one pass; from
                    # r_tail on only added (the flush window's division follows at the end)
                    r_div = max(j0 + 1, min(j0 + n_reg_here, r_tail))
                    for lo, hi, d in ((j0 + 1, r_div, pf_div), (r_div, j0 + n_reg_here, 0.0)):
                        if hi > lo:
                            a, bb, o = pff[lo - j0 - 1: hi - j0 - 1, S * Mb:], pff[lo - j0: hi - j0, :S * Mb], sl_f[lo:hi]
                            if slab.fits(a, bb, o):
                                slab.run(a, bb, o, d)
                            else:
                                torch.add(a, bb, out=o)
                                if d:                    # (a device scalar: no host copy)
                                    o.div_(torch.full((), d, dtype=torch.float32, device=device))
                elif pff is not None:
                    torch.add(pff[:n_reg_here - 1, S * Mb:], pff[1:n_reg_here, :S * Mb],
                              out=sl_f[j0 + 1: j0 + n_reg_here])
                if pfl is not None:
                    torch.add(pfl[:n_reg_here - 1, S * Mb:], pfl[1:n_reg_here, :S * Mb],
                              out=sl_l[j0 + 1: j0 + n_reg_here])
                if pp is not None:
                    torch.add(pp[:n_reg_here - 1, S:], pp[1:n_reg_here, :S], out=sl_p[j0 + 1: j0 + n_reg_here])
            sl[j0 + n_reg_here] += pr[n_reg_here - 1, SP:]
            sl_b[j0 + n_reg_here] += prb[n_reg_here - 1, SP:]
            sl[j0] += pr[0, :SP]
            sl_b[j0] += prb[0, :SP]
            if pff is not None:                      # (the stream mode's edge rows: see _ensemble_batch_pooled)
                sl_f[j0 + n_reg_here] += pff[S:2 * S].view(-1) if stream else pff[n_reg_here - 1, S * Mb:]
                sl_f[j0] += pff[:S].view(-1) if stream else pff[0, :S * Mb]
            if pfl is not None:
                sl_l[j0 + n_reg_here] += pfl[n_reg_here - 1, S * Mb:]
                sl_l[j0] += pfl[0, :S * Mb]
            if pp is not None:
                sl_p[j0 + n_reg_here] += pp[n_reg_here - 1, S:]
                sl_p[j0] += pp[0, :S]
        elif _ODD_VEC_ENABLED and W == 2 * S + 1:
            # odd W: the per-window adds below as three strided passes per stream (_odd_window_add; same bits)
            _odd_window_add(pred, pr, c0, n_reg_here, S, Pn)
            _odd_window_add(pred_b, prb, c0, n_reg_here, S, Pn)
            _odd_window_count(cov, c0, n_reg_here, S, Pn)
            if pff is not None and odd_fine is not None:
                odd_fine.run(pff, pf, c0, n_reg_here, S, Mb, n, n_regular, (n - W) if has_tail else -1, n_models)
            elif pff is not None:
                _odd_window_add(pf, pff, c0, n_reg_here, S, Mb)
                _odd_window_count(cov_f, c0, n_reg_here, S, Mb)
            if pfl is not None:
                _odd_window_add(lf, pfl, c0, n_reg_here, S, Mb)
            if pp is not None:
                _odd_window_add(part, pp, c0, n_reg_here, S, 1)
        else:
            for i in range(n_reg_here):
                s0 = chunk[i]
                pred[s0 * Pn:(s0 + W) * Pn] += pr[i]
                pred_b[s0 * Pn:(s0 + W) * Pn] += prb[i]
                cov[s0 * Pn:(s0 + W) * Pn] += 1
                if pff is not None:
                    pf[s0 * Mb:(s0 + W) * Mb] += pff[i]
                    cov_f[s0 * Mb:(s0 + W) * Mb] += 1
                if pfl is not None:
                    lf[s0 * Mb:(s0 + W) * Mb] += pfl[i]
                if pp is not None:
                    part[s0:s0 + W] += pp[i]
        if n_reg_here < B:
            s0 = chunk[-1]
            pred[s0 * Pn:(s0 + W) * Pn] += pr[B - 1]
            pred_b[s0 * Pn:(s0 + W) * Pn] += prb[B - 1]
            if cov is not None:
                cov[s0 * Pn:(s0 + W) * Pn] += 1
            if pff is not None:
                pf[s0 * Mb:(s0 + W) * Mb] += pff[2 * S:].view(-1) if stream else pff[B - 1]
                if cov_f is not None:
                    cov_f[s0 * Mb:(s0 + W) * Mb] += 1
            if pfl is not None:
                lf[s0 * Mb:(s0 + W) * Mb] += pfl[B - 1]
            if pp is not None:
                part[s0:s0 + W] += pp[B - 1]
            tail_range = (s0 * Pn, (s0 + W) * Pn)
    if up_release is not None:
        up_release()                                 # the trace's and geometry's uses are queued
    if defer_div and slabbed and not pad and pf is None:
        return pred, pred_b, (n, Pn, S, n_regions, (tail_range[0] // Pn) if tail_range is not None else -1,
                              n_models)
    if fused:
        rate32, rate64, p_frame, lam16, prob16 = asm.tail(
            pred, pred_b, part, flush[1] if flush is not None else None, pf, flush[2] if flush is not None else None,
            n=n, Pn=Pn, S=S, M=Mb, n_regions=n_regions, n_regular=n_regular, bw=int(batch_windows),
            s0=flush[0] if flush is not None else -1, n_models=n_models, want16=bool(want16), cmax=cmax)
        if extras is not None:
            extras.update(rate32=rate32[:n0], rate64=rate64[:n0] if rate64 is not None else None,
                          p_frame=p_frame[:n0], lam16=lam16[:n0 * Pn] if lam16 is not None else None,
                          prob16=prob16[:n0 * Pn] if prob16 is not None else None,
                          prob_cmax=(cmax_chunk, cmax[: n0 * (Mb // cmax_chunk)]) if cmax is not None else None,
                          key=(str(device), Pn, part is not None, bool(want16)))
    elif slabbed:
        divisor = reg_count * float(n_models)
        if tail_range is None:
            sl.div_(divisor[:, None])
            sl_b.div_(divisor[:, None])
            if pf is not None and pf_written:  # the interior regions were divided as they were written
                rcf = reg_count_f * float(n_models)
                for e in pf_edges:
                    sl_f[e].div_(rcf[e])
            elif pf is not None:
                sl_f.div_((reg_count_f * float(n_models))[:, None])
            if lf is not None:
                sl_l.div_((reg_count_f * float(n_models))[:, None])
            if part is not None:
                sl_p.div_(divisor[:, None])
        else:
            t0, t1 = tail_range
            r_first = t0 // SP
            if r_first > 0:
                sl[:r_first].div_(divisor[:r_first, None])
                sl_b[:r_first].div_(divisor[:r_first, None])
                if pf is not None and pf_written:   # interior regions before r_first: divided when written
                    rcf = reg_count_f * float(n_models)
                    for e in pf_edges:
                        if e < r_first:
                            sl_f[e].div_(rcf[e])
                elif pf is not None:
                    sl_f[:r_first].div_((reg_count_f * float(n_models))[:r_first, None])
                if lf is not None:
                    sl_l[:r_first].div_((reg_count_f * float(n_models))[:r_first, None])
                if part is not None:
                    sl_p[:r_first].div_(divisor[:r_first, None])
            base = r_first * SP
            per_bin = torch.zeros(n * Pn - base, dtype=dt, device=device)
            for r in range(r_first, n_regions):
                per_bin[(r - r_first) * SP:(r - r_first + 1) * SP] = reg_count[r]
            per_bin[t0 - base:] += 1.0
            per_bin.mul_(float(n_models))
            pred[base:].div_(per_bin)
            pred_b[base:].div_(per_bin)
            if part is not None:             # per frame: every pooled bin of a frame has the frame's divisor
                part[r_first * S:].div_(per_bin.view(-1, Pn)[:, 0])
            if pf is not None:               # the same region counts, on the fine grid, in fp32
                base_f = r_first * S * Mb
                per_bin_f = torch.zeros(n * Mb - base_f, dtype=torch.float32, device=device)
                for r in range(r_first, n_regions):
                    per_bin_f[(r - r_first) * S * Mb:(r - r_first + 1) * S * Mb] = reg_count_f[r]
                per_bin_f[(t0 // Pn) * Mb - base_f:] += 1.0
                per_bin_f.mul_(float(n_models))
                pf[base_f:].div_(per_bin_f)
                if lf is not None:
                    lf[base_f:].div_(per_bin_f)
    else:
        # every bin is covered by construction (the regular windows and the tail flush). The check reads the minimum
        # back to the host -- a synchronisation that made every ROI wait for its own forward before the next could be
        # queued (2026-11-06: ~2 ms per ROI at 10.7 Hz) -- so it runs only with JSCOPE_SETSUNA_DEBUG_COVERAGE=1.
        if _DEBUG_COVERAGE:
            assert float(cov.min()) > 0, "internal: uncovered pooled bins after tail flush"
        cov.mul_(float(n_models))
        pred /= cov
        pred_b /= cov
        if pf is not None and odd_fine is not None:
            if tail_range is not None:                # the kernel left the flush window's frames undivided
                s0t = n - W
                pf[s0t * Mb:n * Mb].view(-1, Mb).div_(_odd_tail_counts(s0t, n, S, n_regular, n_models, device)[:, None])
        elif pf is not None:
            cov_f.mul_(float(n_models))
            pf /= cov_f
            if lf is not None:
                lf /= cov_f
        if part is not None:
            part /= cov.view(n, Pn)[:, 0]
    if pad:
        pred = pred[:n0 * Pn]
        pred_b = pred_b[:n0 * Pn]
        if pf is not None:
            pf = pf[:n0 * Mb]
        if lf is not None:
            lf = lf[:n0 * Mb]
        if part is not None:
            part = part[:n0]
    if lf is not None:
        return pred, pred_b, pf, lf
    if part is not None:
        return pred, pred_b, pf, part
    if fine_prob:
        return pred, pred_b, pf
    return pred, pred_b


def _ensemble_batch(models, cfg, xb, ind_t, nu_t, fps_t, ct_t, eff_window, route_mode, use_ac):
    """One batch of windows through the ensemble -> per-window SUMS over models of the count rate
    and the spike probability, (B, W*M) float32. (A float64 pooled-domain variant was measured
    2026-09-04 and rejected: exact float64 reductions run at a quarter of the fp32 bandwidth on
    this GPU, so pooling per model cost more than the fp32 accumulation it replaced.)"""
    fast_dual = bool(cfg["DUAL"]) and all(_dual_fast_path_ok(m) for m in models)
    B = xb.shape[0]
    acc = acc_b = None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
        for m in models:
            if fast_dual:
                # BinarySRHybrid.forward's dual path with the gate mix as ONE kernel:
                # (1-w)*bulk + w*corner == lerp(bulk, corner, w), written over the bulk logits.
                bc, bb = m.bulk(xb, ind_t, nu_t, fps_t, ct_t, eff_window)
                cc, cb = m.corner(xb, ind_t, nu_t, fps_t, ct_t, eff_window)
                w = m.compute_gate(nu_t, fps_t, ind_t, ct_t).unsqueeze(-1).to(bc.dtype)
                if acc is None:
                    pr, prb = _fused("post_head_first", _post_head_first)(bc, cc, bb, cb, w)
                    pr, prb = pr.float(), prb.float()
                else:
                    acc, acc_b = _fused("post_head_acc", _post_head_acc)(bc, cc, bb, cb, w, acc, acc_b)
                    del bc, cc, bb, cb
                    continue
                del bc, cc, bb, cb
            elif cfg["DUAL"]:
                o = m(xb, ind_t, nu_t, fps_t, ct_t, eff_window=eff_window, route_mode=route_mode)
                pr = F.softplus(o[0]).float()
                prb = torch.sigmoid(o[1]).float()
            else:
                o = m(xb, ind_t, nu_t, fps_t, ct_t, eff_window=eff_window, route_mode=route_mode)
                pr = m.decode(o).float()
                prb = pr
            if acc is None:
                acc = pr
                acc_b = prb if prb is not pr else prb.clone()
            else:
                acc.add_(pr)
                acc_b.add_(prb)
    return acc.reshape(B, -1), acc_b.reshape(B, -1)


def _ensemble_batch_graphed(models, cfg, xb, nu, ind_id, ct_id, eff, eff_window, route_mode,
                            use_ac, device, pool_k=None, want_pf: bool = False, want_lf: bool = False,
                            row_off=None, stream_sp=None, nu_static=None):
    """_ensemble_batch through a CUDA graph captured once per batch shape and replayed.

    Bit-identical to the eager call by construction -- the same kernels run in the same order on
    the same data; only the ~600 kernel launches per batch stop paying their CPU cost. Static
    inputs are the window batch and the noise level (per ROI), ``row_off`` ((B, W) int32, the
    pooled path's per-row count split, see _ensemble_batch_pooled) and ``stream_sp`` (the head's
    stream geometry, likewise) when given; fps, cell type and
    indicator are baked into the key. Graphs sharing one memory pool are replayed strictly in
    sequence. Runs eagerly under the bf16 opt-in (autocast's cast cache and capture do not mix), when
    JSCOPE_SETSUNA_CUDA_GRAPH=0, or after a capture failure (warned once, then eager for the session).
    """
    B, W = xb.shape
    want_part = row_off is not None
    if nu_static is not None:
        # inside the per-ROI forward's own capture (_pooled_whole): eager, with the noise level from its static
        # input -- the kernels _capture_ensemble_batch records, in the same order
        nu_t = nu_static[:B]
    elif _GRAPHS_ENABLED and not use_ac and device.type == "cuda":
        cache = getattr(models[0], "_jscope_graphs", None)
        if cache is None:
            cache = {"pool": torch.cuda.graph_pool_handle(), "graphs": {}, "disabled": False}
            models[0]._jscope_graphs = cache
        if not cache["disabled"]:
            key = (B, W, int(ind_id), int(ct_id), float(eff), eff_window, route_mode, str(device),
                   len(models), (None if pool_k is None else int(pool_k)), bool(want_pf), bool(want_lf),
                   want_part, stream_sp is not None)
            entry = cache["graphs"].get(key)
            if entry is None:
                try:
                    entry = _capture_ensemble_batch(models, cfg, B, W, ind_id, ct_id, eff, eff_window,
                                                    route_mode, device, cache["pool"], pool_k=pool_k,
                                                    want_pf=want_pf, want_lf=want_lf, want_part=want_part,
                                                    want_stream=stream_sp is not None)
                    cache["graphs"][key] = entry
                except Exception as exc:          # pragma: no cover - depends on the driver
                    cache["disabled"] = True
                    import warnings
                    warnings.warn("SETSUNA: CUDA graph capture failed (%s); running eagerly." % exc)
                    entry = None
            if entry is not None:
                graph, xb_s, nu_s, _keep, out, off_s, sp_s = entry
                xb_s.copy_(xb)
                nu_s.fill_(float(nu))
                if want_part:
                    off_s.view(B, W).copy_(row_off)
                if stream_sp is not None:
                    sp_s.copy_(stream_sp)
                graph.replay()
                return out
    if nu_static is None:
        nu_t = torch.full((B,), float(nu), device=device)
    fps_t = torch.full((B,), float(eff), device=device)
    ct_t = torch.full((B,), int(ct_id), dtype=torch.long, device=device)
    ind_t = torch.full((B,), int(ind_id), dtype=torch.long, device=device)
    if pool_k is not None:
        return _ensemble_batch_pooled(models, cfg, xb, ind_t, nu_t, fps_t, ct_t, eff_window, int(pool_k),
                                      want_pf=want_pf, ids=(int(ind_id), int(ct_id), float(eff)), want_lf=want_lf,
                                      row_off=(row_off.reshape(-1).contiguous() if want_part else None),
                                      stream_sp=stream_sp)
    return _ensemble_batch(models, cfg, xb, ind_t, nu_t, fps_t, ct_t, eff_window, route_mode, use_ac)


def _capture_ensemble_batch(models, cfg, B, W, ind_id, ct_id, eff, eff_window, route_mode,
                            device, pool, pool_k=None, want_pf: bool = False, want_lf: bool = False,
                            want_part: bool = False, want_stream: bool = False):
    xb_s = torch.zeros(B, W, dtype=torch.float32, device=device)
    nu_s = torch.ones(B, dtype=torch.float32, device=device)
    off_s = torch.zeros(B * W, dtype=torch.int32, device=device) if want_part else None
    # the head's stream geometry: zeros (no rows: every block of the head kernel leaves) until a replay's copy
    sp_s = torch.zeros(_STREAM_SP_LEN, dtype=torch.int64, device=device) if want_stream else None
    fps_t = torch.full((B,), float(eff), device=device)
    ct_t = torch.full((B,), int(ct_id), dtype=torch.long, device=device)
    ind_t = torch.full((B,), int(ind_id), dtype=torch.long, device=device)
    if pool_k is not None:
        def _run():
            return _ensemble_batch_pooled(models, cfg, xb_s, ind_t, nu_s, fps_t, ct_t, eff_window, int(pool_k),
                                          want_pf=want_pf, ids=(int(ind_id), int(ct_id), float(eff)),
                                          want_lf=want_lf, row_off=off_s, stream_sp=sp_s)
    else:
        def _run():
            return _ensemble_batch(models, cfg, xb_s, ind_t, nu_s, fps_t, ct_t, eff_window, route_mode, False)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):                       # warm-up: lazy caches, allocator state, Triton JIT
        for _ in range(3):
            _run()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    # thread_local: another thread's cudaHostRegister/Unregister (a prefetched next session's TIFF
    # decode pinning its blocks) must not invalidate this capture; the capturing thread's own
    # unsafe calls still do. capture_graph: no device-wide synchronize before it -- the in-process
    # runner's preload captures here under a registration pass and waited for that pass's queue.
    from setsuna._support import frame_stream as _fs_cap
    with _fs_cap.capture_graph(graph, pool=pool, capture_error_mode="thread_local"):
        out = _run()
    return graph, xb_s, nu_s, (fps_t, ct_t, ind_t), out, off_s, sp_s


@torch.inference_mode()
@_strict_fp32()
def _predict_fine_cuda(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                       batch_windows: int = 512, autocast_bf16: bool = False):
    """CUDA core of predict_fine: batched ensemble forward + overlap-add on the device.
    Returns (lam, prob) as float32 DEVICE tensors, already trimmed to n0*M.

    Precision: float32 throughout the forward (see _strict_fp32; bf16 autocast is opt-in and was
    both slower and biased here). With fp32 inputs and head_dim 12, SDPA selects the fused
    memory-efficient kernel on its own. The per-batch ensemble computation is replayed from a
    captured CUDA graph (bit-identical; see _ensemble_batch_graphed).

    Same overlap-add as the host path -- each output bin receives its window halves in the same
    order, then one division -- with three float32 reorderings accepted 2026-09-04 for speed:
    the gate mix runs as lerp(bulk, corner, w) instead of (1-w)*bulk + w*corner, the ensemble is
    summed in place and divided once together with the coverage count instead of averaged
    first, and the regions inside a batch are written in one pass. Measured against the host
    path on real ROIs: relRMS ~1e-7, no signed bias. The C4 tail-flush window and odd W keep
    per-window slice adds with a per-bin coverage tensor, in order.
    """
    M = int(cfg["M"])
    n0 = len(xz)
    W = w_of_fps(cfg, eff)
    pad = 0
    if n0 < W:  # reflect-pad short traces; predict_stream's silent-zero case
        pad = W - n0
        xz = np.concatenate([xz, xz[-2::-1][:pad].astype(np.float32)]) if n0 > 1 \
            else np.full(W, float(xz[0]) if n0 else 0.0, np.float32)
    if len(xz) < W:  # extremely short trace; tile the reflection
        xz = np.resize(xz, W).astype(np.float32)
        pad = W - n0
    n = len(xz)

    eff_window = (W if cfg["WMODE"] == "mask" else None)
    route_mode = (("hard_corner" if (eff <= cfg["HC_FPS"] and nu >= cfg["HC_NU"]) else "hard_bulk")
                  if cfg["HARDROUTE"] else "soft")

    stride = max(1, W // 2)
    starts = list(range(0, n - W + 1, stride))
    has_tail = bool(starts) and starts[-1] + W < n
    if has_tail:   # [C4] flush the tail
        starts.append(n - W)
    n_regular = len(starts) - (1 if has_tail else 0)
    slabbed = (W % 2 == 0) and (stride * 2 == W) and n_regular > 0
    S = stride

    Mb = M
    dt = torch.float32
    n_models = len(models)
    pred = torch.zeros(n * Mb, dtype=dt, device=device)
    pred_b = torch.zeros(n * Mb, dtype=dt, device=device)
    cov = None
    if slabbed:
        n_regions = n_regular + 1                       # regular windows cover (R+1) half-slabs
        sl = pred[: n_regions * S * Mb].view(n_regions, S * Mb)
        sl_b = pred_b[: n_regions * S * Mb].view(n_regions, S * Mb)
        # windows covering each region: one for the first and the last, two inside
        reg_count = torch.full((n_regions,), 2.0, dtype=dt, device=device)
        # fill_ on the end slices, not item assignment: `t[0] = 1.0` copies the scalar from pageable host
        # memory, and a pageable host->device copy first synchronises the stream -- every ROI the host
        # stalled here until the previous ROI's forward had finished, then left the GPU idle while it
        # queued this one (~0.2 ms per ROI). Same values.
        reg_count[:1].fill_(1.0)
        reg_count[-1:].fill_(1.0)
    else:
        cov = torch.zeros(n * Mb, dtype=dt, device=device)
    tail_range = None

    use_ac = bool(autocast_bf16) and torch.cuda.is_bf16_supported()
    # The trace goes up once (pinned, non-blocking) and the window batches are strided views of it, as in
    # _predict_fine_cuda_pooled: a pageable upload per batch (torch.tensor(np.stack(...), device=...)) returned
    # only once the stream had drained, so a caller could never queue this ROI behind the previous one's work.
    # The windows hold the same values.
    xz_dev = _upload_trace(xz, device)
    win_all = xz_dev.unfold(0, W, stride)                   # (n_regular, W): row i = xz[starts[i]:+W]
    for c0 in range(0, len(starts), int(batch_windows)):
        chunk = starts[c0:c0 + int(batch_windows)]
        B = len(chunk)
        n_reg_chunk = B - (1 if (has_tail and c0 + B == len(starts)) else 0)
        if n_reg_chunk == B:
            xb = win_all[c0:c0 + B]
        elif n_reg_chunk > 0:
            xb = torch.cat([win_all[c0:c0 + n_reg_chunk], xz_dev[n - W:].unsqueeze(0)])
        else:
            xb = xz_dev[n - W:].unsqueeze(0)
        pr, prb = _ensemble_batch_graphed(models, cfg, xb, nu, ind_id, ct_id, eff, eff_window,
                                          route_mode, use_ac, device)
        # pr, prb: (B, W*Mb) sums over models; graph outputs are consumed before the next replay

        n_reg_here = B - (1 if (has_tail and c0 + B == len(starts)) else 0)
        if slabbed and n_reg_here > 0:
            j0 = c0
            # Regions strictly inside this batch receive exactly two halves, both from this
            # batch -- window j-1's second half and window j's first half -- so they are
            # written in one pass. The batch's first and last regions are shared with the
            # neighbouring batch and accumulate, in the same per-bin order as the host loop.
            if n_reg_here > 1:
                torch.add(pr[:n_reg_here - 1, S * Mb:], pr[1:n_reg_here, :S * Mb],
                          out=sl[j0 + 1: j0 + n_reg_here])
                torch.add(prb[:n_reg_here - 1, S * Mb:], prb[1:n_reg_here, :S * Mb],
                          out=sl_b[j0 + 1: j0 + n_reg_here])
            sl[j0 + n_reg_here] += pr[n_reg_here - 1, S * Mb:]
            sl_b[j0 + n_reg_here] += prb[n_reg_here - 1, S * Mb:]
            sl[j0] += pr[0, : S * Mb]
            sl_b[j0] += prb[0, : S * Mb]
        elif _ODD_VEC_ENABLED and W == 2 * S + 1:
            _odd_window_add(pred, pr, c0, n_reg_here, S, Mb)   # odd W (_odd_window_add; same bits)
            _odd_window_add(pred_b, prb, c0, n_reg_here, S, Mb)
            _odd_window_count(cov, c0, n_reg_here, S, Mb)
        else:
            for i in range(n_reg_here):
                s0 = chunk[i]
                pred[s0 * Mb:(s0 + W) * Mb] += pr[i]
                pred_b[s0 * Mb:(s0 + W) * Mb] += prb[i]
                cov[s0 * Mb:(s0 + W) * Mb] += 1
        if n_reg_here < B:                                  # the tail-flush window, last of all
            s0 = chunk[-1]
            pred[s0 * Mb:(s0 + W) * Mb] += pr[B - 1]
            pred_b[s0 * Mb:(s0 + W) * Mb] += prb[B - 1]
            if cov is not None:
                cov[s0 * Mb:(s0 + W) * Mb] += 1
            tail_range = (s0 * Mb, (s0 + W) * Mb)

    # Every bin holds the SUM over models and over the windows covering it; divide once by
    # (windows x models). In the slabbed layout the count is constant per region, except
    # where the tail window adds one more, so only the last regions need a per-bin divisor.
    if slabbed:
        divisor = reg_count * float(n_models)
        if tail_range is None:
            sl.div_(divisor[:, None])
            sl_b.div_(divisor[:, None])
        else:
            t0, t1 = tail_range
            r_first = t0 // (S * Mb)                      # first region the tail touches
            if r_first > 0:
                sl[:r_first].div_(divisor[:r_first, None])
                sl_b[:r_first].div_(divisor[:r_first, None])
            base = r_first * S * Mb
            per_bin = torch.zeros(n * Mb - base, dtype=dt, device=device)
            for r in range(r_first, n_regions):            # region counts, then +1 under the tail
                per_bin[(r - r_first) * S * Mb:(r - r_first + 1) * S * Mb] = reg_count[r]
            per_bin[t0 - base:] += 1.0
            per_bin.mul_(float(n_models))
            pred[base:].div_(per_bin)
            pred_b[base:].div_(per_bin)
    else:
        if _DEBUG_COVERAGE:                    # (a host read-back: see _predict_fine_cuda_pooled)
            assert float(cov.min()) > 0, "internal: uncovered fine bins after tail flush"
        cov.mul_(float(n_models))
        pred /= cov
        pred_b /= cov
    if pad:
        pred = pred[:n0 * Mb]
        pred_b = pred_b[:n0 * Mb]
    return pred, pred_b


@torch.inference_mode()
def predict_fine(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                 batch_windows: int = 512, autocast_bf16: bool = False):
    """Batched overlap-add over one standardized trace -> (lam, prob), both (n*M,) float32.

    Same contract as _predict_fine_host, which is the original implementation and still runs on
    CPU devices. On CUDA the overlap-add is accumulated ON THE DEVICE (see _predict_fine_cuda)
    and the two finished streams are copied back once -- bit-identical to the host path,
    verified with array_equal on real ROIs.
    """
    if device.type != "cuda":
        return _predict_fine_host(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                  batch_windows=batch_windows, autocast_bf16=autocast_bf16)
    pred, pred_b = _predict_fine_cuda(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                      batch_windows=batch_windows, autocast_bf16=autocast_bf16)
    return pred.cpu().numpy(), pred_b.cpu().numpy()


def _f64_to_f16_rne(x: "torch.Tensor") -> "torch.Tensor":
    """float64 -> float16 with a single round-to-nearest-even, on any device.

    torch has no direct float64->float16 kernel: ``x.to(torch.float16)`` rounds to float32
    first and to float16 second, and that double rounding is wrong whenever the float32 step
    lands exactly on a float16 midpoint. With the earlier bf16 forward the pooled SETSUNA sums
    hit that case in ~1% of elements -- their bf16-derived leading bits often sat on a midpoint
    with only a tiny positive tail from quiet bins above it -- and always in the same direction
    (rarer with the float32 forward, same hazard). Rounding the
    intermediate to ODD instead (the sticky-bit trick) makes the second rounding exact; float32
    carries 13 bits more than float16 needs, so the standard result applies. Verified
    bit-identical to numpy's direct cast on 15.5M values incl. ties, near-ties and float16
    subnormals, and on real pooled streams.
    """
    if x.dtype != torch.float64:
        raise TypeError("_f64_to_f16_rne expects float64, got %s" % (x.dtype,))
    y = x.to(torch.float32)
    yd = y.to(torch.float64)
    make_odd = (yd != x) & ((y.contiguous().view(torch.int32) & 1) == 0)
    toward = torch.where(x > yd, torch.full_like(y, float("inf")), torch.full_like(y, float("-inf")))
    y = torch.where(make_odd, torch.nextafter(y, toward), y)
    return y.to(torch.float16)


_WHOLE_GRAPH_ENABLED = os.environ.get("JSCOPE_SETSUNA_WHOLE_GRAPH", "1").strip().lower() not in (
    "0", "false", "no", "off")
_WHOLE_MAX_ENTRIES = 2


def _clone_out(res, ex):
    rc = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in res)
    ec = {}
    for k, v in ex.items():
        if isinstance(v, torch.Tensor):
            ec[k] = v.clone()
        elif k == "prob_cmax" and v is not None:
            ec[k] = (v[0], v[1].clone())
        else:
            ec[k] = v
    return rc, ec


def _outs_equal(a_res, a_ex, b_res, b_ex) -> bool:
    if len(a_res) != len(b_res) or set(a_ex) != set(b_ex):
        return False
    for x, y in zip(a_res, b_res):
        if isinstance(x, torch.Tensor) != isinstance(y, torch.Tensor):
            return False
        if isinstance(x, torch.Tensor) and not _bits_equal(x, y):
            return False
    for k in a_ex:
        x, y = a_ex[k], b_ex[k]
        if k == "prob_cmax" and x is not None:
            if y is None or x[0] != y[0]:
                return False
            x, y = x[1], y[1]
        if isinstance(x, torch.Tensor):
            if not isinstance(y, torch.Tensor) or not _bits_equal(x, y):
                return False
        elif x != y:
            return False
    return True


def _pooled_whole(models, cfg, xz, ind_id, ct_id, eff, nu, device, bw: int, k_tail: int, fine_prob: bool,
                  fine_lam: bool, part_sb, want16: bool, no_fuse: bool):
    """(res, extras) of _predict_fine_cuda_pooled for this ROI through the geometry's whole-forward graph, or None
    (run it eagerly: graphs off, a padded trace, this geometry's first ROI or no room for another graph)."""
    if not (_GRAPHS_ENABLED and device.type == "cuda"):
        return None
    cache = getattr(models[0], "_jscope_graphs", None)
    if cache is not None and cache.get("disabled"):
        return None
    n = int(len(xz))
    W = w_of_fps(cfg, eff)
    if n < W:
        return None
    route_mode = (("hard_corner" if (eff <= cfg["HC_FPS"] and nu >= cfg["HC_NU"]) else "hard_bulk")
                  if cfg["HARDROUTE"] else "soft")
    st = getattr(models[0], "_jscope_whole", None)
    if st is None:
        st = models[0]._jscope_whole = {}
    key = (n, int(W), int(bw), int(k_tail), bool(fine_prob), bool(fine_lam), part_sb, bool(want16), bool(no_fuse),
           int(ind_id), int(ct_id), float(eff), route_mode, str(device), len(models))
    ent = st.get(key)
    kw = dict(batch_windows=bw, pool_k=k_tail, fine_prob=fine_prob, fine_lam=fine_lam, seg_bins=part_sb,
              want16=want16, _no_fuse=no_fuse)
    if ent is None:
        st[key] = "seen"                                    # the first ROI: eager
        return None
    if ent == "seen":
        if sum(1 for v in st.values() if isinstance(v, dict)) >= _WHOLE_MAX_ENTRIES:
            st[key] = "off"
            return None
        ex_e: dict = {}
        res_e = _predict_fine_cuda_pooled(models, cfg, xz, ind_id, ct_id, eff, nu, device, extras=ex_e, **kw)
        ok = False
        try:
            ent = _capture_whole(models, cfg, n, ind_id, ct_id, eff, nu, device, bw, kw)
            _replay_whole(ent, xz, nu, device)
            ok = _outs_equal(res_e, ex_e, ent["res"], ent["ex"])
            if not ok:
                import warnings
                warnings.warn("SETSUNA: whole-forward graph differs from the eager forward; per-batch graphs.")
        except Exception as exc:                            # pragma: no cover - depends on the driver
            import warnings
            warnings.warn("SETSUNA: whole-forward graph capture failed (%s); per-batch graphs." % exc)
        st[key] = ent if ok else "off"
        return res_e, ex_e
    if ent == "off":
        return None
    _replay_whole(ent, xz, nu, device)
    return _clone_out(ent["res"], ent["ex"])


def _capture_whole(models, cfg, n, ind_id, ct_id, eff, nu, device, bw, kw):
    W = w_of_fps(cfg, eff)
    stride = max(1, W // 2)
    n_reg = len(range(0, n - W + 1, stride))
    n_win = n_reg + (1 if (n_reg and (n_reg - 1) * stride + W < n) else 0)
    n_batches = -(-n_win // bw)
    cap = {"xz_dev": torch.zeros(n, dtype=torch.float32, device=device),
           "nu_t": torch.ones(bw, dtype=torch.float32, device=device),
           "geo_host": torch.zeros(max(1, n_batches) * _STREAM_SP_LEN, dtype=torch.int64, pin_memory=True)}
    cache = getattr(models[0], "_jscope_graphs", None)
    if cache is None:
        cache = models[0]._jscope_graphs = {"pool": torch.cuda.graph_pool_handle(), "graphs": {}, "disabled": False}
    dummy = np.zeros(n, np.float32)
    # warm-up outside the capture (lazy caches, Triton JIT for the eager ensemble), then nothing pending
    _predict_fine_cuda_pooled(models, cfg, dummy, ind_id, ct_id, eff, nu, device, extras={}, _cap=cap, **kw)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    from setsuna._support import frame_stream as _fs_cap
    ex: dict = {}
    with _fs_cap.capture_graph(graph, pool=cache["pool"], capture_error_mode="thread_local"):
        res = _predict_fine_cuda_pooled(models, cfg, dummy, ind_id, ct_id, eff, nu, device, extras=ex, _cap=cap,
                                        **kw)
    return {"graph": graph, "cap": cap, "res": res, "ex": ex}


def _replay_whole(ent, xz, nu, device):
    cap = ent["cap"]
    if _UPLOAD_STREAM_ENABLED:
        both, release = _upload_ring(np.ascontiguousarray(xz, dtype=np.float32), device)
        cap["xz_dev"].copy_(both.view(torch.float32))
        release()
    else:
        cap["xz_dev"].copy_(_upload_trace(xz, device))
    cap["nu_t"].fill_(float(nu))
    ent["graph"].replay()


@torch.inference_mode()
def predict_fine_outputs(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                         batch_windows: int = 512, autocast_bf16: bool = False,
                         pool_k=None, need_full: bool = True, full_device: bool = False,
                         fine_prob: bool = False, defer_copy: bool = False, seg_bins=None, host_bufs=None) -> dict:
    """predict_fine plus the reductions the run actually stores, done where the data already is.

    Returns a dict: ``rate`` (n_frames,) float32 always; ``lam_pooled`` / ``prob_pooled``
    (n_frames*M/pool_k,) float16 when ``pool_k`` is given (mass-preserving SUM over pool_k
    native bins, as pool_fine); ``lam`` / ``prob`` full-resolution float32 only when
    ``need_full`` (the spike-time decoder is the one consumer). ``seg_bins`` (the timing
    decoder's segment length in native bins, timing_segment_bins; with ``need_full`` and
    ``full_device``): on the pooled CUDA path ``lam`` is replaced by ``lam_frame`` (per-frame
    count mass, float64) and ``lam_part`` = (seg_bins, per-frame mass before the segment
    boundary inside the frame, float64), from which decode_spike_times forms the per-segment
    counts -- the fine count stream is never built (other paths still return ``lam``). With the fused
    assembly also ``prob_cmax`` = (chunk, per-chunk maxima of ``prob``, NaN where a value is negative
    or NaN), from which decode_spike_times takes the timing decoder's tile bounds.

    Why: the full streams are 2 x 308 MB per ROI. Copying them to the host and pooling them
    there was ~0.1 s of a 0.33 s ROI once the overlap-add had moved to the device; the pooled
    forms are 32x smaller and the rate 512x. On CUDA the reductions accumulate in FLOAT64 --
    exact for these sums -- and are rounded ONCE: ``rate`` to float32 by the cast, the pooled
    streams to float16 by ``_f64_to_f16_rne`` (torch's own cast double-rounds through float32
    and missed the correctly rounded value in ~1% of elements, one-sidedly). So the stored
    values are the correctly rounded exact sums. The host branch below keeps the original
    float32 pairwise sums, which are themselves inexact (~1e-7 relative on ``rate``, a last-ulp
    miss in ~0.8% of the pooled float16 values), so the two branches are close but not
    bit-identical. The forward itself runs in float32 (since 2026-09-04; the earlier bf16
    autocast was slower and biased -- see _strict_fp32); measured, not assumed.

    ``defer_copy`` (CUDA): ``rate`` and the pooled streams stay on the device as
    ``rate_dev`` / ``lam_pooled_dev`` / ``prob_pooled_dev`` with a ``done_event`` recorded after
    them, and nothing synchronises -- the caller queues the next ROI's forward first and then
    completes this one with finish_outputs() while the GPU is busy (see run_setsuna_deconvolution).

    ``host_bufs`` (CPU): the session's buffer cache for _predict_host_fused, which runs this branch's operations
    fused where it covers the ensemble (the same bits).
    """
    M = int(cfg["M"])
    n_frames = int(len(xz))
    if device.type != "cuda" and (not pool_k or M % int(pool_k) == 0) and _host_fused_ok(models, cfg):
        return _predict_host_fused(models, cfg, xz, ind_id, ct_id, eff, nu, device, batch_windows=batch_windows,
                                   pool_k=pool_k, full=bool(need_full or fine_prob), bufs=host_bufs)
    if device.type != "cuda":
        lam, prob = _predict_fine_host(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                       batch_windows=batch_windows, autocast_bf16=autocast_bf16)
        out = {"rate": count_head_rate(lam, n_frames, M)}
        if pool_k:
            out["lam_pooled"] = pool_fine(lam, int(pool_k)).astype(np.float16)
            out["prob_pooled"] = pool_fine(prob, int(pool_k)).astype(np.float16)
        if need_full or fine_prob:
            out["lam"], out["prob"] = lam, prob
        return out
    k_tail = int(pool_k) if pool_k else M
    # Full-resolution requests (the timing decoder) come from the pooled forward too, which then also writes both fine
    # streams: 37 -> ~20 ms per ROI on the 1450-ROI bench session. Its fp32 sums run in another order than
    # _predict_fine_cuda's (relRMS ~2e-7 on the streams); JSCOPE_SETSUNA_POOLED_FULL=0 keeps the full-resolution forward.
    pooled_full = bool(need_full) and _POOLED_FULL
    if ((not need_full or pooled_full) and M % k_tail == 0 and 4 <= k_tail <= M
            and _pooled_tail_ok(models, cfg, bool(autocast_bf16) and torch.cuda.is_bf16_supported())):
        # Pooled tail: the fine count grid is never materialised (see _predict_fine_cuda_pooled);
        # with fine_prob the binary stream is, for the event-count spike decoder, and for a
        # full-resolution request both streams are.
        if need_full:
            fine_prob = True
        # the timing decoder's counts from per-frame sums and boundary parts instead of the fine count stream
        part_sb = (int(seg_bins) if (need_full and full_device and seg_bins is not None and int(seg_bins) > M)
                   else None)
        Pn = M // k_tail
        ft = _cuda_tail_state(models, device) if (pool_k and not fine_prob) else None
        use_ft = ft is not None and ft["t"] is not None and ft["ok"] is not False and ft["t"].supports(Pn)
        ex: dict = {}
        fkey = (str(device), Pn, part_sb is not None, bool(pool_k))
        res = None
        if _WHOLE_GRAPH_ENABLED and not use_ft:
            got = _pooled_whole(models, cfg, xz, ind_id, ct_id, eff, nu, device, int(batch_windows), k_tail,
                                bool(fine_prob), bool(need_full) and part_sb is None, part_sb, bool(pool_k),
                                _FINE_FUSED_OK.get(fkey) is False)
            if got is not None:
                res, ex = got
        if res is None:
            res = _predict_fine_cuda_pooled(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                            batch_windows=batch_windows, pool_k=k_tail, fine_prob=bool(fine_prob),
                                            defer_div=use_ft, fine_lam=bool(need_full) and part_sb is None,
                                            seg_bins=part_sb, extras=ex, want16=bool(pool_k),
                                            _no_fuse=_FINE_FUSED_OK.get(fkey) is False)
        if "rate32" in ex and _FINE_FUSED_OK.get(ex["key"]) is None:
            # first use of the fused assembly in this configuration: against the torch ops on the same ROI (the
            # forward is deterministic), every output bit for bit -- gpu_kernel_check's rule (a), once per process,
            # GPU and configuration (the verdict and its time go to the run metadata)
            from setsuna._support import gpu_kernel_check as _gkc
            ref = None

            def _check_asm(c, dev):
                nonlocal ref
                ref = _predict_fine_cuda_pooled(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                                batch_windows=batch_windows, pool_k=k_tail, fine_prob=True,
                                                seg_bins=part_sb, _no_fuse=True)
                r64 = ref[0].view(n_frames, Pn).sum(dim=1) if Pn > 1 else ref[0]
                rp = (ref[1].view(n_frames, Pn).sum(dim=1) if Pn > 1 else ref[1]) / float(M)
                same = (len(ref) == len(res) and all(_bits_equal(a, b) for a, b in zip(res, ref))
                        and (ex.get("prob_cmax") is None or _bits_equal(ex["prob_cmax"][1],
                                                                          _chunk_max(ref[2], ex["prob_cmax"][0])))
                        and _bits_equal(ex["rate32"], r64.to(torch.float32))
                        and _bits_equal(ex["rate64"] if ex["rate64"] is not None else res[0], r64)
                        and _bits_equal(ex["p_frame"], rp)
                        and (not pool_k or (_bits_equal(ex["lam16"], _f64_to_f16_rne(ref[0]))
                                            and _bits_equal(ex["prob16"], _f64_to_f16_rne(ref[1])))))
                if not same:
                    c.fail("fused assembly outputs")
            _k = ex["key"]
            ok = _gkc.use(f"setsuna.fine_asm.P{_k[1]}{'.part' if _k[2] else ''}{'.f16' if _k[3] else ''}", device,
                          _check_asm)
            _FINE_FUSED_OK[ex["key"]] = bool(ok)
            if not ok:
                if ref is None:
                    ref = _predict_fine_cuda_pooled(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                                    batch_windows=batch_windows, pool_k=k_tail, fine_prob=True,
                                                    seg_bins=part_sb, _no_fuse=True)
                res, ex = ref, {}
        fused = None
        if use_ft and len(res) == 3 and isinstance(res[2], tuple) and int(res[2][0]) == n_frames:
            _n, _pn, _S, _nreg, _s0, _nm = res[2]
            fused = ft["t"].run(res[0], res[1], _n, _pn, _S, _nreg, _s0, _nm)
            if ft["ok"] is None:
                # first use: against the torch tail on the same ROI (the forward is deterministic), bit for bit --
                # gpu_kernel_check's rule (a), once per process, GPU and pooled bins per frame
                from setsuna._support import gpu_kernel_check as _gkc
                ref = None

                def _check_tail(c, dev):
                    nonlocal ref
                    ref = _predict_fine_cuda_pooled(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                                    batch_windows=batch_windows, pool_k=k_tail, fine_prob=False)
                    r64 = ref[0].view(n_frames, Pn).sum(dim=1) if Pn > 1 else ref[0]
                    c.equal(fused[0], r64.to(torch.float32), "rate")
                    c.equal(fused[1], _f64_to_f16_rne(ref[0]), "lambda f16")
                    c.equal(fused[2], _f64_to_f16_rne(ref[1]), "prob f16")
                ok = _gkc.use(f"setsuna.tail.P{Pn}", device, _check_tail)
                ft["ok"] = bool(ok)
                if not ok:
                    if ref is None:
                        ref = _predict_fine_cuda_pooled(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                                        batch_windows=batch_windows, pool_k=k_tail, fine_prob=False)
                    res, fused = ref, None
        if fused is not None:
            rate32, pooled16 = fused[0], (fused[1], fused[2])
        else:
            if len(res) == 3 and isinstance(res[2], tuple):   # deferred but not taken: divide as usual
                res = _predict_fine_cuda_pooled(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                                batch_windows=batch_windows, pool_k=k_tail,
                                                fine_prob=bool(fine_prob),
                                                fine_lam=bool(need_full) and part_sb is None, seg_bins=part_sb)
            pc, pb = res[0], res[1]
            if "rate32" in ex:                               # the fused assembly's tail computed them
                rate64 = ex["rate64"] if ex["rate64"] is not None else pc
                rate32 = ex["rate32"]
                pooled16 = (ex["lam16"], ex["prob16"]) if pool_k else None
            else:
                rate64 = pc.view(n_frames, Pn).sum(dim=1) if Pn > 1 else pc
                rate32 = rate64.to(torch.float32)
                pooled16 = (_f64_to_f16_rne(pc), _f64_to_f16_rne(pb)) if pool_k else None
        if defer_copy:
            out = {"rate_dev": rate32}
            if pooled16 is not None:
                out["lam_pooled_dev"], out["prob_pooled_dev"] = pooled16
        else:
            out = {"rate": rate32.cpu().numpy()}
            if pooled16 is not None:
                out["lam_pooled"] = pooled16[0].cpu().numpy()
                out["prob_pooled"] = pooled16[1].cpu().numpy()
        if fine_prob:                                              # (never the fused tail)
            out["prob"] = res[2]                                   # fine binary stream, device fp32
            out["lam_frame"] = rate64                              # per-frame count mass, float64
            out["p_frame"] = ex["p_frame"] if "p_frame" in ex else (
                (pb.view(n_frames, Pn).sum(dim=1) if Pn > 1 else pb) / float(M))
        if need_full and part_sb is not None:                      # counts from lam_frame + the boundary parts
            out["lam_part"] = (part_sb, res[3])
            if ex.get("prob_cmax") is not None:                    # the stream's chunk maxima, for the tile bounds
                out["prob_cmax"] = ex["prob_cmax"]
        elif need_full:                                            # both fine streams, device fp32
            out["lam"] = res[3]
            if not full_device:
                out["lam"], out["prob"] = out["lam"].cpu().numpy(), out["prob"].cpu().numpy()
        if defer_copy:
            ev = torch.cuda.Event()
            ev.record()                                            # everything above is queued before it
            out["done_event"] = ev
        return out
    pred, pred_b = _predict_fine_cuda(models, cfg, xz, ind_id, ct_id, eff, nu, device,
                                      batch_windows=batch_windows, autocast_bf16=autocast_bf16)
    rate32 = torch.sum(pred.view(n_frames, M), dim=1, dtype=torch.float64).to(torch.float32)
    pooled16 = None
    if pool_k:
        k = int(pool_k)
        pooled16 = (_f64_to_f16_rne(torch.sum(pred.view(-1, k), dim=1, dtype=torch.float64)),
                    _f64_to_f16_rne(torch.sum(pred_b.view(-1, k), dim=1, dtype=torch.float64)))
    # defer_copy as on the pooled path: rate and the pooled streams stay on the device behind a done_event, so the
    # caller is not held until this ROI's forward has finished (finish_outputs copies them)
    defer = bool(defer_copy) and pred.is_cuda
    if defer:
        out = {"rate_dev": rate32}
        if pooled16 is not None:
            out["lam_pooled_dev"], out["prob_pooled_dev"] = pooled16
    else:
        out = {"rate": rate32.cpu().numpy()}
        if pooled16 is not None:
            out["lam_pooled"] = pooled16[0].cpu().numpy()
            out["prob_pooled"] = pooled16[1].cpu().numpy()
    if need_full:
        if full_device:      # the spike decoder consumes them on the device; no copy back
            out["lam"], out["prob"] = pred, pred_b
        else:
            out["lam"], out["prob"] = pred.cpu().numpy(), pred_b.cpu().numpy()
    if defer:
        ev = torch.cuda.Event()
        ev.record()                                                # everything above is queued before it
        out["done_event"] = ev
    return out


_D2H_BUFS: dict = {}                    # finish_outputs' pinned staging buffers, by (output, shape, dtype)
_D2H_LOCK = threading.Lock()


def finish_outputs(out: dict) -> dict:
    """Complete a predict_fine_outputs(..., defer_copy=True) result in place: the deferred
    ``rate`` / ``lam_pooled`` / ``prob_pooled`` cross to the host now. The copies run on a side
    stream that waits for the result's own ``done_event`` only, so a forward the caller has
    queued on the main stream since then is not waited for. A no-op for results that were not
    deferred, so callers can treat both kinds alike."""
    keys = (("rate_dev", "rate"), ("lam_pooled_dev", "lam_pooled"), ("prob_pooled_dev", "prob_pooled"))
    if not any(k in out for k, _ in keys):
        return out
    ev = out.pop("done_event", None)
    dev = next(out[k].device for k, _ in keys if k in out)
    if ev is not None and dev.type == "cuda":
        from .evcount_fast import _side_stream
        side = _side_stream(dev)
        # into pinned buffers, not pageable memory: a pageable copy is staged through the host in chunks, and the
        # compute stream sat idle ~30 us per ROI next to it (profiled; same values)
        with _D2H_LOCK:
            staged = []
            with torch.cuda.stream(side):
                side.wait_event(ev)
                for dev_key, key in keys:
                    t = out.pop(dev_key, None)
                    if t is not None:
                        hb = _D2H_BUFS.get((key, tuple(t.shape), t.dtype))
                        if hb is None:
                            hb = _D2H_BUFS[(key, tuple(t.shape), t.dtype)] = torch.empty(
                                t.shape, dtype=t.dtype, pin_memory=True)
                        hb.copy_(t, non_blocking=True)
                        staged.append((key, hb, t))
                done = torch.cuda.Event()
                done.record(side)
            done.synchronize()
            for key, hb, _t in staged:
                out[key] = hb.numpy().copy()
        out["done_event"] = ev                  # the spike decoder waits on it too
        return out
    for dev_key, key in keys:
        t = out.pop(dev_key, None)
        if t is not None:
            out[key] = t.cpu().numpy()
    if ev is not None:
        out["done_event"] = ev
    return out


# ------------------------------------------------------------------------- output tiers
def auto_fine_bin_ms(fps: float) -> float:
    """Saved bin width (ms) that `auto` resolves to -- frame / recommended_fine_divisor.

    Identical by construction to what the GUI picker recommends for the same frame
    rate, so the two default paths cannot drift apart. See recommended_fine_divisor
    for the measurement behind it.

    [2026-09-06] This REPLACES the old frame/10-clamped-to-[1,16]ms rule, which was
    wrong in two ways. Its divisor sat on the COARSE side of the measured 10.5-15x SR
    factor (the docstring called frame/10 "slightly conservative", which had the
    direction backwards -- at 10 Hz it stored 9.84 ms bins for a model resolving
    6.56 ms, ~9% added in quadrature), and its 16 ms ceiling forced frame/64 at 1 Hz,
    far finer than the model resolves there. The spike-time decoder always runs on the
    full m-grid regardless of what is saved -- this only sets what lands on disk."""
    return 1000.0 / (max(float(fps), 1e-9) * recommended_fine_divisor(fps))


# The count-gain and saved-bin rules live in torch-free modules so the GUI can apply them without torch.
from .count_gain import COUNT_GAIN_AUTO, COUNT_GAIN_BY_FPS, recommended_count_gain, resolve_count_gain  # noqa: E402,F401
from .fine_bins import MIN_USEFUL_BIN_MS, RECOMMENDED_FINE_DIVISOR, recommended_fine_divisor  # noqa: E402,F401


def fine_pool_factor(M: int, fps: float, target_bin_ms: float | None) -> int:
    """Sum-pool factor k (a power-of-two divisor of M) giving ~target_bin_ms bins.
    None or 0 -> 1 (full resolution); negative -> auto, which is exactly the GUI's
    recommended frame/D for this fps (see recommended_fine_divisor)."""
    if isinstance(target_bin_ms, str):
        t = target_bin_ms.strip().lower()
        if t.startswith("frame/"):      # exact bins-per-frame, fps-free (the GUI picker)
            D = max(1, min(int(M), int(t.split("/", 1)[1])))
            k = M // D if M % D == 0 else 2 ** int(round(math.log2(max(1.0, M / D))))
            return int(max(1, min(M, k)))
        target_bin_ms = -1.0 if t in ("auto", "") else float(t)
    if target_bin_ms is not None and float(target_bin_ms) < 0:
        # Auto: take the divisor directly rather than round-tripping through a bin
        # width and back, so it is exactly the GUI picker's frame/D -- no float
        # rounding, and no clamp of its own to disagree with.
        return int(max(1, M // max(1, min(int(M), recommended_fine_divisor(fps)))))
    if not target_bin_ms:
        return 1
    full_bin_ms = 1000.0 / (M * fps)
    k = target_bin_ms / full_bin_ms
    k = 2 ** int(round(math.log2(max(1.0, min(float(M), k)))))
    return int(max(1, min(M, k)))


def pool_fine(stream: np.ndarray, k: int) -> np.ndarray:
    """Mass-preserving sum-binning of a fine stream by factor k (k | M).
    This is the eval's own reduction (C55 rebins lambda this way; the pooled-MP arm
    pools the probability stream this way). Never smooth-then-decimate for storage."""
    if k <= 1:
        return stream
    return stream.reshape(-1, k).sum(axis=1)


def count_head_rate(lam: np.ndarray, n_frames: int, M: int) -> np.ndarray:
    """Per-frame expected spike counts (the OASIS-shaped default output)."""
    return lam.reshape(n_frames, M).sum(axis=1)


def _decode_knobs_active() -> bool:
    """True when one of decode_readout's import-time env variants is on (orthogonal MP, the
    combinatorial search set, a refractory period, an external matched template). The CUDA
    decoder reproduces only the default greedy pursuit, so those cases stay on the vendored code."""
    return bool(getattr(_decode, "_DECODE_OMP", False) or getattr(_decode, "_DECODE_SEARCHSET", False)
                or getattr(_decode, "_DECODE_TMPL", None) is not None
                or float(os.environ.get("DECODE_REFRAC_MS", "0") or 0) > 0)


@torch.inference_mode()
def _greedy_pursuit_cuda(work, n_hat, hw, tmpl):
    """decode_readout.struct_decode's greedy loop on one float64 device segment, in place.

    Same decisions as the numpy loop: first-occurrence argmax, stop when the maximum is <= 0,
    subtract ``work[i] * template`` over the clipped window (float64 multiply, then subtract --
    the same two roundings). Returns the picked bin indices (Python ints, in pick order)."""
    L = work.numel()
    picks = []
    for _ in range(int(n_hat)):
        i = torch.argmax(work)
        both = torch.stack([i.to(torch.float64), work[i]]).cpu()      # one sync per pick
        i = int(both[0].item())
        v = float(both[1].item())
        if v <= 0.0:
            break
        picks.append(i)
        lo = max(0, i - hw)
        hi = min(L, i + hw + 1)
        work[lo:hi].sub_(tmpl[hw - (i - lo): hw + (hi - i)] * v)
    return picks


@torch.inference_mode()
def _greedy_pursuit_batched_cuda(work, lens, n_hat, hw, tmpl):
    """_greedy_pursuit_cuda for all segments of a ROI at once, in lockstep. The fallback of the pursuit
    kernel (_decode_cuda.pursuit, same picks; JSCOPE_SETSUNA_PURSUIT_KERNEL=0), which replaced it: every
    iteration here scans every bin of every segment.

    ``work`` is (n_seg, L) float64 with row j valid over ``lens[j]`` bins; ``n_hat[j]`` picks are
    requested per row. Iteration t takes every row's first-occurrence argmax (torch.argmax
    over dim 1 breaks ties on the smallest index, as numpy does), subtracts ``v * template``
    over the clipped window through one index_add_ -- each in-range bin receives exactly one
    contribution, so the arithmetic is the sequential loop's ``work[lo:hi] -= v * tmpl`` bin
    for bin -- and records the pick. A row whose maximum is <= 0, or whose count is used up,
    contributes exact zeros from then on, which is the sequential loop's break: nothing
    changes and nothing is recorded. The cost therefore scales with max(n_hat), not with the
    total number of spikes, and there is one device->host copy per ROI instead of one per
    spike. Returns a list of int arrays (picks per row, in pick order)."""
    n_seg, L = work.shape
    device = work.device
    n_iter = int(max(n_hat)) if len(n_hat) else 0
    if n_iter <= 0:
        return [np.zeros(0, np.int64) for _ in range(n_seg)]
    lens_t = torch.as_tensor(lens, dtype=torch.int64, device=device)
    cnt_t = torch.as_tensor([int(x) for x in n_hat], dtype=torch.int64, device=device)
    rows = torch.arange(n_seg, device=device)
    # bins past a row's own length do not exist in the numpy segment: keep them out of argmax
    cols = torch.arange(L, device=device)
    work.masked_fill_(cols[None, :] >= lens_t[:, None], float("-inf"))
    offs = torch.arange(-hw, hw + 1, device=device)
    zero = torch.zeros((), dtype=work.dtype, device=device)
    picks = torch.empty(n_iter, n_seg, dtype=torch.int64, device=device)
    valid = torch.empty(n_iter, n_seg, dtype=torch.bool, device=device)
    flat = work.view(-1)
    for t in range(n_iter):
        i = torch.argmax(work, dim=1)                              # (n_seg,) first occurrence
        v = work[rows, i]                                          # (n_seg,)
        ok = (v > 0) & (cnt_t > t)
        picks[t] = i
        valid[t] = ok
        idx = i[:, None] + offs[None, :]                           # (n_seg, 2hw+1)
        inside = (idx >= 0) & (idx < L)
        contrib = tmpl[None, :] * torch.where(ok, v, zero)[:, None]  # v*tmpl, or exact zeros
        contrib = torch.where(inside, contrib, zero)
        gidx = (rows[:, None] * L + idx.clamp_(0, L - 1)).view(-1)
        flat.index_add_(0, gidx, -contrib.view(-1))               # unique in-range bins; clamped duplicates add 0
    picks = picks.cpu().numpy()
    valid = valid.cpu().numpy()
    return [picks[valid[:, j], j].astype(np.int64) for j in range(n_seg)]


_PURSUIT_KERNEL_ENABLED = os.environ.get("JSCOPE_SETSUNA_PURSUIT_KERNEL", "1").strip().lower() not in (
    "0", "false", "no", "off")
# JSCOPE_SETSUNA_FFT64=1: the timing decoder's matched filter as float64 FFTs throughout (see _decode_spike_times_cuda)
_FFT64_ONLY = os.environ.get("JSCOPE_SETSUNA_FFT64", "0").strip().lower() not in ("0", "false", "no", "off")
# The verified pursuit computes the float32 matched filter only for the runs it reaches, starting every run from a
# bound of its tile (_decode_cuda.pursuit_from_bounds); JSCOPE_SETSUNA_MF_LAZY=0 computes it everywhere first. The
# same picks either way.
_MF_LAZY = os.environ.get("JSCOPE_SETSUNA_MF_LAZY", "1").strip().lower() not in ("0", "false", "no", "off")


def _pursuit_kernel_ok(device) -> bool:
    """The timing decoder's pursuit kernel (_decode_cuda) is enabled and loads on ``device``; otherwise the lockstep
    torch loop (_greedy_pursuit_batched_cuda) runs, with the same picks. JSCOPE_SETSUNA_PURSUIT_KERNEL=0 forces it."""
    if not _PURSUIT_KERNEL_ENABLED:
        return False
    try:
        return _decode_cuda.available(device) and _gpu_ok("decode.pursuit", device)
    except Exception:
        return False


TIMING_MIN_SEGMENT_S = 30.0


def timing_segment_bins(M: int, fps: float, segment_s: "float | None" = None, n_frames: "int | None" = None,
                        width_ms: "float | None" = None) -> int:
    """The timing decoder's segment length in native bins -- what predict_fine_outputs(seg_bins=...) needs to hand
    the decoder its per-segment counts. An explicit ``segment_s`` is used as given. By default (``n_frames`` and
    ``width_ms`` of the decode) a record is cut into as few equal segments as the GPU decoder's fast path holds for
    this template (_decode_cuda.verified_max_bins; ~3.8M bins, 12 min at 10 Hz, ~45 s at 120 Hz at the rule's width),
    never shorter than 30 s, and a record under that length is decoded whole. [2026-11-06] Every segment boundary
    cost ~0.001 F1 on the C167 GT-known records (30 s segments -0.005 against the whole record, 243 records, 1-25 ms);
    the boundaries depend on the record only, so the CPU and GPU decoders cut identically."""
    ffr = M * fps
    if segment_s is not None:
        return max(1, int(round(float(segment_s) * ffr)))
    if n_frames is None or width_ms is None:
        raise ValueError("timing_segment_bins: give segment_s, or n_frames and width_ms of the decode")
    sig = max(float(width_ms) / 1000.0 * ffr, 0.5)
    hw = max(1, int(round(3 * sig)))
    max_bins = max(int(_decode_cuda.verified_max_bins(hw)), int(round(TIMING_MIN_SEGMENT_S * ffr)), int(M) + 1)
    n_fine = max(1, int(n_frames) * int(M))
    n_seg = -(-n_fine // max_bins)
    return max(int(M) + 1, -(-n_fine // n_seg))


def _segment_count_sums(lam_frame, lam_part, M: int, seg_bins: int, n_fine: int) -> "torch.Tensor":
    """The timing decoder's per-segment count mass (float64, on lam_frame's device) without the fine count stream:
    from the per-frame masses ``lam_frame`` and ``lam_part`` = (seg_bins, per frame the mass before the segment
    boundary inside it), as predict_fine_outputs(seg_bins=...) returns them -- the whole frames of a segment plus the
    two split ones. These are the forward's float64 sums of its float32 values, where the fine stream first rounded
    each overlap-added value to float32: on the 1450-ROI bench session they differ from the sums over that stream
    by <= 1.7e-7 (1.4e-8 relative) and all 126,150 counts N = round(gain * sum) are the same (the nearest sum to a
    rounding tie was 6.4e-6 away)."""
    sb, part = lam_part
    if int(sb) != int(seg_bins):
        raise ValueError("decode_spike_times: the counts' boundary parts were made for %d-bin segments, the decoder's "
                         "are %d (see timing_segment_bins)" % (int(sb), int(seg_bins)))
    if lam_frame is None:
        raise ValueError("decode_spike_times: lam_part needs lam_frame")
    lf = torch.as_tensor(lam_frame).reshape(-1)
    pt = torch.as_tensor(part).reshape(-1)
    if lf.is_cuda and _pursuit_kernel_ok(lf.device):
        return _decode_cuda.segment_counts(lf, pt, M, seg_bins, n_fine)
    # the same sums with torch ops (host tensors, or without the kernels)
    lf = lf.to(torch.float64)
    pt = pt.to(device=lf.device, dtype=torch.float64)
    M, sb, n_fine = int(M), int(seg_bins), int(n_fine)
    n_frames = n_fine // M
    if sb <= M or n_fine % M or lf.numel() != n_frames or pt.numel() != n_frames:
        raise ValueError("decode_spike_times: per-frame counts need seg_bins > M and n_frames * M bins")
    out = []
    for a in range(0, n_fine, sb):
        b = min(a + sb, n_fine)
        v = lf[-(-a // M): b // M].sum()
        if a % M:
            v = v + (lf[a // M] - pt[a // M])
        if b % M:
            v = v + pt[b // M]
        out.append(v)
    return torch.stack(out) if out else torch.zeros(0, dtype=torch.float64, device=lf.device)


@torch.inference_mode()
def _decode_spike_times_cuda(lam, prob, M: int, fps: float, sig_ms: float, objective: str,
                             tol_ms: float, segment_s: "float | None",
                             count_gain: float = 1.0, lam_frame=None, lam_part=None,
                             prob_cmax=None, seg_bins=None) -> np.ndarray:
    """decode_spike_times on CUDA tensors: the same segmented greedy matching pursuit as the
    numpy path, with the matched filter as a batched float64 FFT convolution and the pursuit on
    the device (one thread block per segment, _decode_cuda), so the full-resolution streams never
    cross to the host. 2026-09-26, 1450-ROI 60.9 Hz session: the decode ~60 -> ~18 ms per ROI
    (pursuit 14 -> 1 ms, the all-zero mask no longer computed for the padding of the last
    segment, 22 ms), 633,494 spikes identical; then ~2.4 ms with the matched filter in float32 that
    the pursuit verifies from its tile maxima (_decode_cuda.matched_filter32 / pursuit_verified;
    JSCOPE_SETSUNA_FFT64=1 keeps float64 FFTs), same spikes again.

    Equivalence to the vendored decoder: on real streams the spike trains are identical
    (measured 2026-09-04 on 145 ROIs of a 60.9 Hz session, both objectives, ~67k spikes each,
    zero differences). The only arithmetic that differs is the matched filter -- FFT instead of
    np.convolve, 2e-17 absolute apart on unit-scale probabilities, which is also the level at
    which numpy's BLAS-backed correlate is defined -- and the per-segment count N = round(sum
    lambda), summed in float64 rather than numpy's float32 pairwise sum (identical on all 2610
    test segments; can only differ when the sum sits within ~1e-6 of a half-integer). Template,
    window clipping, first-occurrence argmax, the <= 0 stop and the time formula are the same
    operations on the same values. Where the direct convolution is EXACTLY zero (every
    probability under the window zero) the FFT result is forced to zero as well, so all-zero
    regions stop the pursuit exactly as in numpy.

    What can differ, and only on synthetic inputs: exact ties (mirror-symmetric bumps, constant
    plateaus) are broken by 1e-17 rounding rather than by index order, and a residual that the
    direct arithmetic leaves at exactly zero (an isolated single nonzero bin) is +-1e-17 here,
    which matters only once the count exceeds the number of real peaks. Sigmoid outputs have
    neither property: the measured floor of real streams is ~5e-4 with bin-to-bin variation.
    """
    device = prob.device
    ffr = M * fps
    n_fine = int(prob.numel())
    if seg_bins is None:
        seg_bins = max(1, int(round(segment_s * ffr)))
    seg_bins = int(seg_bins)
    arm = _route.choose_arm(fps, tol_ms=tol_ms, objective=objective)
    n_seg = -(-n_fine // seg_bins)
    if n_seg == 0:
        return np.zeros(0, np.float64)
    if lam is None:
        # per-segment counts from the forward's per-frame sums and boundary parts (no fine count stream)
        sums = _segment_count_sums(lam_frame, lam_part, M, seg_bins, n_fine).to(device)
    else:
        # per-segment counts from the COUNT head, summed in float64 (the numpy path sums float32) straight from the
        # float32 stream: whole segments as rows, a partial last one on its own
        lam_flat = lam.reshape(-1)
        n_whole = n_fine // seg_bins
        sums = torch.empty(n_seg, dtype=torch.float64, device=device)
        if n_whole:
            torch.sum(lam_flat[: n_whole * seg_bins].view(n_whole, seg_bins), dim=1, dtype=torch.float64,
                      out=sums[:n_whole])
        if n_seg > n_whole:
            torch.sum(lam_flat[n_whole * seg_bins:], dim=0, dtype=torch.float64, out=sums[n_whole])
    # count_gain scales the count head BEFORE rounding, so the two paths round the same value.
    n_hat = torch.round(float(count_gain) * sums).to(torch.int64).cpu().tolist()
    todo = [si for si in range(n_seg) if n_hat[si] >= 1]
    if not todo:
        return np.zeros(0, np.float64)

    if arm == "msadapt":
        bin_ms = _route.bin_ms_for(tol_ms, fps)
        k = int(fine_pool_factor(M, fps, bin_ms))
        ffr_eff = ffr / k
    else:
        k = 1
        ffr_eff = ffr
    sig = max(float(sig_ms / 1000.0 * ffr_eff), 0.5)
    hw = max(1, int(round(3 * sig)))
    tt = torch.arange(-hw, hw + 1, dtype=torch.float64, device=device)
    tmpl = torch.exp(-0.5 * (tt / sig) ** 2)

    # segment signals (pooled for msadapt: zero-pad each segment to a multiple of k, sum groups)
    seg_len_full = -(-seg_bins // k) * k if k > 1 else seg_bins
    L_seg = seg_len_full // k                                # pooled length of a full segment
    nfft = 1
    while nfft < L_seg + 2 * hw:
        nfft <<= 1
    T = None                                                 # the template's spectrum, for the float64 FFT path

    def _segment_signal(si):
        a = si * seg_bins
        b = min(n_fine, a + seg_bins)
        pf = prob[a:b].to(torch.float64)
        if k > 1:
            rem = pf.numel() % k
            if rem:
                pf = F.pad(pf, (0, k - rem))
            pf = pf.view(-1, k).sum(dim=1)
        return pf

    n_todo = len(todo)
    counts = [n_hat[si] for si in todo]
    # The pursuit kernel (_decode_cuda) runs each chunk's segments in place on the FFT output; without it the
    # lockstep torch loop takes them all at the end (the same picks).
    use_kernel = _pursuit_kernel_ok(device)
    work = None if use_kernel else torch.empty(n_todo, L_seg, dtype=torch.float64, device=device)
    lens = [0] * n_todo
    picks = []
    flat = prob.reshape(-1)
    # The verified path (_decode_cuda): the matched filter in float32 by direct correlation (setsuna_mf32), which
    # bounds the float64 response bin by bin -- its fused multiply-adds of non-negative terms err by at most
    # rho = gamma_K + 2u of the value (bound_rho; 3.4e-5 at 60 Hz, 2.7e-6 measured) -- and the pursuit computes the
    # float64 value exactly wherever it decides a pick (pursuit_verified). The same picks as the float64 FFT path up
    # to float64 rounding (~1e-17): identical trains on the 1450-ROI bench session. A negative or NaN probability
    # (never a sigmoid output) sends the chunk to the float64 path.
    verified = (k == 1 and use_kernel and not _FFT64_ONLY and flat.dtype == torch.float32 and flat.is_contiguous()
                and _decode_cuda.mf32_fits(hw)
                and _decode_cuda.available(device, verified=True) and _gpu_ok("decode.verified", device))
    if verified:
        rho = _decode_cuda.bound_rho(hw)
        # the absolute part of the bound: float64 roundings of the bounds and of the subtractions (~1e-11 at worst)
        slack = torch.full((128,), 1e-9, dtype=torch.float64, device=device)

    def _fill(batch, idx, c0):
        """The segments' probabilities into ``batch`` (float64) in one copy per chunk (a partial last segment on its
        own), zero beyond each segment's end: the batch the per-segment copies built. Sets ``lens``; returns whether a
        segment holds an exact zero (a device tensor)."""
        batch[:, L_seg:].zero_()
        if k == 1:
            partial = (idx[-1] + 1) * seg_bins > n_fine
            n_full = len(idx) - int(partial)
            has_zero = torch.zeros((), dtype=torch.bool, device=device)
            if n_full:
                src = flat[: (n_fine // seg_bins) * seg_bins].view(-1, seg_bins)
                if idx[n_full - 1] - idx[0] == n_full - 1:
                    batch[:n_full, :L_seg].copy_(src[idx[0]: idx[0] + n_full])
                else:
                    batch[:n_full, :L_seg].copy_(src[torch.as_tensor(idx[:n_full], device=device)])
                lens[c0: c0 + n_full] = [seg_bins] * n_full
                has_zero = has_zero | (batch[:n_full, :L_seg] == 0).any()
            if partial:
                a = idx[-1] * seg_bins
                ln = n_fine - a
                batch[n_full, :ln].copy_(flat[a:])
                batch[n_full, ln:L_seg].zero_()
                lens[c0 + n_full] = ln
                has_zero = has_zero | (batch[n_full, :ln] == 0).any()
            return has_zero
        batch[:, :L_seg].zero_()
        has_zero = torch.zeros((), dtype=torch.bool, device=device)
        for j, si in enumerate(idx):
            x = _segment_signal(si)
            lens[c0 + j] = int(x.numel())
            batch[j, : lens[c0 + j]] = x
            has_zero = has_zero | (batch[j, : lens[c0 + j]] == 0).any()
        return has_zero

    # the verified path takes a ROI's segments in one launch (a block each), up to 128 at a time
    # segments per batch, and no more bins than 128 (verified) / 32 thirty-second segments at 120 Hz (the buffers' size
    # before segments followed the fast path's length)
    chunk = max(1, min(128 if verified else 32, ((128 if verified else 32) * 1843200) // max(1, seg_bins)))
    for c0 in range(0, n_todo, chunk):
        idx = todo[c0:c0 + chunk]
        if verified:
            seg0 = [si * seg_bins for si in idx]
            lens[c0: c0 + len(idx)] = [min(seg_bins, n_fine - a) for a in seg0]
            got = None
            if _MF_LAZY:
                # the matched filter computed only where the pursuit reaches (the same picks)
                got = _decode_cuda.pursuit_from_bounds(flat, seg0, lens[c0: c0 + len(idx)], counts[c0: c0 + len(idx)],
                                                       slack[: len(idx)], hw, tmpl, rho=rho, cmax=prob_cmax)
            if got is None:
                resp, mf = _decode_cuda.matched_filter32(flat, seg0, lens[c0: c0 + len(idx)], hw, tmpl)
                got = _decode_cuda.pursuit_verified(resp, flat, seg0, lens[c0: c0 + len(idx)],
                                                    counts[c0: c0 + len(idx)], slack[: len(idx)], hw, tmpl, rho=rho,
                                                    mf=mf)
                del resp
            if all(g is not None for g in got):
                picks.extend(got)
                continue
            # a segment needed more exact evaluations than the kernel holds: this chunk in float64
        if T is None:
            T = torch.fft.rfft(tmpl, n=nfft)
        batch = torch.empty(len(idx), nfft, dtype=torch.float64, device=device)
        has_zero = _fill(batch, idx, c0)
        wk = torch.fft.irfft(torch.fft.rfft(batch, n=nfft) * T, n=nfft)[:, hw: hw + L_seg]
        # The direct convolution is EXACTLY zero wherever every probability under the template
        # window is zero, and the greedy loop stops on a maximum <= 0; the FFT leaves +-1e-17
        # there instead, which could be "picked" once the real peaks are used up. A rolling
        # max over the same window finds those bins exactly (probabilities are >= 0). Real
        # streams (sigmoid outputs) have no zeros at all, so the check usually skips it. Only
        # zeros inside a segment count: the bins past a segment's end are never picked, and the
        # zeros that pad a partial last segment made every ROI pay for the mask (22 ms at 60 Hz).
        if bool(has_zero):
            allzero = F.max_pool1d(batch[:, None, :L_seg], 2 * hw + 1, stride=1, padding=hw)[:, 0, :] == 0
            wk = torch.where(allzero, torch.zeros((), dtype=wk.dtype, device=device), wk)
        if work is None:
            picks.extend(_decode_cuda.pursuit(wk, lens[c0: c0 + len(idx)], counts[c0: c0 + len(idx)], hw, tmpl))
        else:
            work[c0:c0 + len(idx)] = wk
    if work is not None:
        picks = _greedy_pursuit_batched_cuda(work, lens, counts, hw, tmpl)
    out = []
    for j, si in enumerate(todo):
        if picks[j].size:
            t = np.sort(picks[j].astype(np.float64)) / ffr_eff
            out.append(t + (si * seg_bins) / ffr)
    if not out:
        return np.zeros(0, np.float64)
    return np.sort(np.concatenate(out))


# Several ROIs' timing decodes in shared pursuit launches (decode_timing_many). JSCOPE_SETSUNA_DECODE_BATCH=0 decodes one
# ROI at a time.
_DECODE_BATCH_ENABLED = os.environ.get("JSCOPE_SETSUNA_DECODE_BATCH", "1").strip().lower() not in (
    "0", "false", "no", "off")


def decode_batch_ok(device) -> bool:
    """decode_timing_many runs its shared launches on this device (else it decodes the ROIs one at a time)."""
    try:
        return (_DECODE_BATCH_ENABLED and str(getattr(device, "type", device)) == "cuda" and not _FFT64_ONLY
                and not _decode_knobs_active() and _pursuit_kernel_ok(device)
                and _decode_cuda.available(device, verified=True) and _gpu_ok("decode.verified", device))
    except Exception:
        return False


@torch.inference_mode()
def decode_timing_many(items, M: int, fps: float, width_ms: float, count_gain: float = 1.0,
                       segment_s: "float | None" = None) -> list:
    """decode_timing for several ROIs at once -- ``items``: dicts with ``prob`` (the fine binary stream, float32 CUDA)
    and ``lam`` or (``lam_frame``, ``lam_part``), optionally ``prob_cmax`` and ``ready_event`` -- returning their spike
    times in order. Each ROI is cut into its own segments exactly as decode_timing cuts it and every segment is decoded
    on its own (one block each); the ROIs' streams are concatenated on the device so their segments share launches,
    which a record decoded as a few long segments otherwise leaves mostly empty. The same spikes as decode_timing ROI
    by ROI: the same counts, segments, template and verified pursuit; a chunk the verified kernel cannot finish sends
    its ROIs to decode_timing."""
    items = list(items)
    if not items:
        return []
    single = lambda it: decode_timing(it.get("lam"), it["prob"], M, fps, width_ms, segment_s=segment_s,
                                      count_gain=count_gain, lam_frame=it.get("lam_frame"),
                                      ready_event=it.get("ready_event"), lam_part=it.get("lam_part"),
                                      prob_cmax=it.get("prob_cmax"))
    prob0 = items[0]["prob"]
    if (len(items) == 1 or not torch.is_tensor(prob0) or not prob0.is_cuda or not decode_batch_ok(prob0.device)
            or any((not torch.is_tensor(it["prob"])) or it["prob"].dtype != torch.float32 for it in items)):
        return [single(it) for it in items]
    device = prob0.device
    from .evcount_fast import _side_stream
    side = _side_stream(device)
    with torch.cuda.stream(side):
        for it in items:
            if it.get("ready_event") is not None:
                side.wait_event(it["ready_event"])
        return _decode_timing_many_on(items, M, fps, width_ms, count_gain, segment_s, device, single)


def _decode_timing_many_on(items, M, fps, width_ms, count_gain, segment_s, device, single):
    ffr = M * fps
    sig = max(float(width_ms) / 1000.0 * ffr, 0.5)
    hw = max(1, int(round(3 * sig)))
    if not _decode_cuda.mf32_fits(hw):
        return [single(it) for it in items]
    tt = torch.arange(-hw, hw + 1, dtype=torch.float64, device=device)
    tmpl = torch.exp(-0.5 * (tt / sig) ** 2)
    n_fines, seg_bs, sums = [], [], []
    for it in items:
        prob = it["prob"]
        n_fine = int(prob.numel())
        sb = int(timing_segment_bins(M, fps, segment_s, n_frames=n_fine // int(M), width_ms=width_ms))
        n_seg = -(-n_fine // sb)
        lam = it.get("lam")
        if lam is None:
            sm = _segment_count_sums(it["lam_frame"], it["lam_part"], M, sb, n_fine).to(device)
        else:                                            # as _decode_spike_times_cuda: float64 sums of the float32 stream
            lam_flat = lam.reshape(-1)
            n_whole = n_fine // sb
            sm = torch.empty(n_seg, dtype=torch.float64, device=device)
            if n_whole:
                torch.sum(lam_flat[: n_whole * sb].view(n_whole, sb), dim=1, dtype=torch.float64, out=sm[:n_whole])
            if n_seg > n_whole:
                torch.sum(lam_flat[n_whole * sb:], dim=0, dtype=torch.float64, out=sm[n_whole])
        n_fines.append(n_fine); seg_bs.append(sb); sums.append(sm)
    n_hat_all = torch.round(float(count_gain) * torch.cat(sums)).to(torch.int64).cpu().tolist()
    base = np.concatenate([[0], np.cumsum(n_fines)]).astype(np.int64)
    segs = []                                            # (roi, si, start in the concatenated stream, length, count)
    pos = 0
    for r, (n_fine, sb) in enumerate(zip(n_fines, seg_bs)):
        n_seg = -(-n_fine // sb)
        for si in range(n_seg):
            n = n_hat_all[pos + si]
            if n >= 1:
                segs.append((r, si, int(base[r]) + si * sb, min(sb, n_fine - si * sb), int(n)))
        pos += n_seg
    picks = {}
    fallback = set()
    if segs:
        flat = torch.cat([it["prob"].reshape(-1) for it in items]) if len(items) > 1 else items[0]["prob"].reshape(-1)
        flat = flat.contiguous()
        cms = [it.get("prob_cmax") for it in items]
        cmax = None
        if _MF_LAZY and all(c is not None for c in cms):
            ch = int(cms[0][0])
            if all(int(c[0]) == ch and n % ch == 0 and int(c[1].numel()) * ch >= n for c, n in zip(cms, n_fines)):
                cmax = (ch, torch.cat([c[1][: n // ch] for c, n in zip(cms, n_fines)]))
        rho = _decode_cuda.bound_rho(hw)
        slack = torch.full((128,), 1e-9, dtype=torch.float64, device=device)
        L_max = max(sg[3] for sg in segs)
        chunk = max(1, min(128, (128 * 1843200) // max(1, L_max)))
        for c0 in range(0, len(segs), chunk):
            part = segs[c0:c0 + chunk]
            seg0 = [sg[2] for sg in part]; lens = [sg[3] for sg in part]; cnts = [sg[4] for sg in part]
            got = None
            if _MF_LAZY:
                got = _decode_cuda.pursuit_from_bounds(flat, seg0, lens, cnts, slack[: len(part)], hw, tmpl, rho=rho,
                                                       cmax=cmax)
            if got is None:
                resp, mf = _decode_cuda.matched_filter32(flat, seg0, lens, hw, tmpl)
                got = _decode_cuda.pursuit_verified(resp, flat, seg0, lens, cnts, slack[: len(part)], hw, tmpl,
                                                    rho=rho, mf=mf)
                del resp
            if all(g is not None for g in got):
                for sg, g in zip(part, got):
                    picks[(sg[0], sg[1])] = g
            else:
                fallback.update(sg[0] for sg in part)
        del flat
    out = []
    for r, it in enumerate(items):
        if r in fallback:
            out.append(single(it))
            continue
        sb = seg_bs[r]
        ts = []
        for si in range(-(-n_fines[r] // sb)):
            g = picks.get((r, si))
            if g is not None and g.size:
                ts.append(np.sort(g.astype(np.float64)) / ffr + (si * sb) / ffr)
        out.append(np.sort(np.concatenate(ts)) if ts else np.zeros(0, np.float64))
    return out


_EVC_FAST_ENABLED = os.environ.get("JSCOPE_SETSUNA_EVC_FAST", "1").strip().lower() not in (
    "0", "false", "no", "off")


def evcount_fast_ok() -> bool:
    """The fast event-count decoder (evcount_fast, numba) is available and enabled -- then the
    detection objective can run on the pooled forward with the fine binary stream only."""
    if not _EVC_FAST_ENABLED:
        return False
    try:
        from . import evcount_fast as _evf
        return bool(_evf.available())
    except ImportError:
        return False


def uses_detection(tol_ms, fps: float) -> bool:
    """True when decode_spike_times serves tolerance ``tol_ms`` (None = DEFAULT_TOL_MS) at ``fps`` with the
    detection decoder (which runs on the pooled forward with the fine binary stream only)."""
    return bool(_route.use_detection(DEFAULT_TOL_MS if tol_ms is None else float(tol_ms), float(fps)))


def decode_spike_times(lam: np.ndarray, prob: np.ndarray, M: int, fps: float,
                       tol_ms: float = DEFAULT_TOL_MS, segment_s: "float | None" = None,
                       count_gain: float = 1.0, lam_frame=None, p_frame=None,
                       ready_event=None, lam_part=None, cell_type=None, prob_cmax=None) -> np.ndarray:
    """Discrete spike times (s) for an analysis at spike-time tolerance ``tol_ms`` (default 10 ms).

    The tolerance picks the decoder (route_readout.decoder_for):
      * tol >= one frame (1000 / fps ms) or tol >= 35 ms -> the DETECTION decoder (decode_detection): the count
        head defines events (runs of frames with mass above a threshold) and how many spikes each holds; the
        binary head decides which frame inside the event and where inside the frame.
      * otherwise -> the TIMING decoder (decode_timing): per segment N = round(gain * sum lambda) and greedy
        matching pursuit places N spikes on the binary head's full sub-frame grid with a Gaussian of width
        w = min(0.65 * tol, 9 ms at <= 30 Hz | 4.5 ms above). Segments (timing_segment_bins): the longest the GPU
        decoder's fast path takes, at least 30 s -- a record under that length is decoded whole.
    The detection decoder cuts events at 30 s segments. ``segment_s`` sets one length for both.

    CUDA tensors (the streams as predict_fine_outputs leaves them on the device) are decoded on the GPU unless one
    of decode_readout's env variants is active; numpy arrays take the vendored reference path.

    count_gain scales the count head before either decoder: a PRECISION/RECALL control. 1.0 (default) is the
    calibrated count. Below 1.0 fewer, better-supported spikes; above 1.0 more spikes into weaker evidence. For the
    timing decoder the train at gain g is a prefix of the train at any larger gain; for the detection decoder the
    trade is monotone per event but not a strict prefix. Rate, lambda and prob are untouched.

    ``ready_event`` (CUDA event recorded after the streams were produced): the device work runs on a side stream
    that waits for that event only, so it overlaps whatever the caller has queued since (the next ROI's forward).
    ``lam_frame`` / ``p_frame`` (per-frame count mass and mean probability) replace a full-resolution count stream
    for the detection decoder; ``lam_part`` (with ``lam_frame``) gives the timing decoder its per-segment counts
    (see _segment_count_sums); ``prob_cmax`` (predict_fine_outputs' chunk maxima of ``prob``) its tile bounds.
    ``cell_type``: the detection decoder places spikes on the combined rate for pyramidal cells at <= 12 Hz
    (route_readout.combined_applies).
    """
    which, width_ms = _route.decoder_for(tol_ms, fps)
    if which == "detection":
        return decode_detection(lam, prob, M, fps, segment_s=30.0 if segment_s is None else segment_s,
                                count_gain=count_gain,
                                lam_frame=lam_frame, p_frame=p_frame, ready_event=ready_event, cell_type=cell_type)
    return decode_timing(lam, prob, M, fps, width_ms, segment_s=segment_s, count_gain=count_gain,
                         lam_frame=lam_frame, ready_event=ready_event, lam_part=lam_part, prob_cmax=prob_cmax)


def decode_detection(lam, prob, M: int, fps: float, segment_s: float = 30.0, count_gain: float = 1.0,
                     lam_frame=None, p_frame=None, ready_event=None, cell_type=None) -> np.ndarray:
    """The detection (event-count) decoder (vendored decode_readout.evcount_decode), in its 25 ms configuration
    (route_readout.EVENT_TOL_MS: placement bin and split exponents) with the 3 ms placement filter. numpy or
    CPU/CUDA tensors; tensors take the fast path (evcount_fast: same algorithm, the box smoothing and per-frame
    pursuit in numba on gathered windows; JSCOPE_SETSUNA_EVC_FAST=0 keeps the reference)."""
    ev_tol = float(_route.EVENT_TOL_MS)
    sig_ms = float(_decode.EVC_SIG_MS)
    comb = bool(_route.combined_applies(float(fps), cell_type))
    a_lam, a_p = _route.evc_split_exponents(ev_tol, float(fps)) if comb else (1.0, 1.0)
    if _EVC_FAST_ENABLED and torch.is_tensor(prob):
        try:
            from . import evcount_fast as _evf
            if _evf.available():
                return _evf.decode(lam, prob, int(M), float(fps), tol_ms=ev_tol,
                                   gain=float(count_gain), seg_s=float(segment_s), sig_ms=sig_ms,
                                   lam_frame=lam_frame, p_frame=p_frame, ready_event=ready_event,
                                   combined=comb, a_lam=a_lam, a_p=a_p)
        except ImportError:
            pass
    if lam is None:
        raise ValueError("decode_detection: the reference event-count decoder needs the fine count "
                         "stream (lam); frame-level inputs need the fast decoder (numba)")
    if comb:
        _l = lam.detach().cpu().numpy() if torch.is_tensor(lam) else np.asarray(lam)
        _p = prob.detach().cpu().numpy() if torch.is_tensor(prob) else np.asarray(prob)
        prod = _decode.combined_rate(_l, _p, int(M), gamma=float(_route.EVC_COMBINED_GAMMA))
        return _decode.evcount_decode2(_l, prod, int(M), float(fps), pfa=_p, a_lam=a_lam, a_p=a_p,
                                       tol_ms=ev_tol, gain=float(count_gain), seg_s=float(segment_s),
                                       sig_ms=sig_ms)
    return _decode.evcount_decode(lam, prob, int(M), float(fps), tol_ms=ev_tol,
                                  gain=float(count_gain), seg_s=float(segment_s), sig_ms=sig_ms)


def decode_timing(lam, prob, M: int, fps: float, width_ms: float, segment_s: "float | None" = None,
                  count_gain: float = 1.0, lam_frame=None, ready_event=None, lam_part=None,
                  prob_cmax=None) -> np.ndarray:
    """The timing decoder: per segment N = round(gain * sum lambda), then greedy matching pursuit
    (decode_readout.struct_decode) places N spikes on the full sub-frame binary head with a Gaussian of
    ``width_ms`` (decode_spike_times picks it from the tolerance)."""
    sig_ms = float(width_ms)
    objective, tol_ms = "timing", float(_route.EVENT_TOL_MS)   # choose_arm: 'timing' is always the full-grid struct arm
    if lam is None and lam_part is None:
        raise ValueError("decode_timing: needs the fine count stream (lam) or the "
                         "per-frame counts (lam_frame, lam_part)")
    n_fine = int(prob.numel()) if torch.is_tensor(prob) else int(len(prob))
    seg_bins = timing_segment_bins(M, fps, segment_s, n_frames=n_fine // int(M), width_ms=sig_ms)
    if torch.is_tensor(prob):
        if prob.is_cuda and not _decode_knobs_active():
            if ready_event is not None:
                # on the decoder side stream behind this ROI's own event, as the event-count decoder does: its FFTs
                # and pursuit overlap the next ROI's forward instead of queueing behind it. The decode ends with
                # host copies of its results, so nothing it reads or allocates outlives the call on that stream.
                from .evcount_fast import _side_stream
                side = _side_stream(prob.device)
                with torch.cuda.stream(side):
                    side.wait_event(ready_event)
                    return _decode_spike_times_cuda(lam, prob, M, fps, sig_ms, objective, tol_ms,
                                                    None, count_gain, lam_frame=lam_frame, lam_part=lam_part,
                                                    prob_cmax=prob_cmax, seg_bins=seg_bins)
            return _decode_spike_times_cuda(lam, prob, M, fps, sig_ms, objective, tol_ms,
                                            None, count_gain, lam_frame=lam_frame, lam_part=lam_part,
                                            prob_cmax=prob_cmax, seg_bins=seg_bins)
        if lam is not None:
            lam = lam.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()
    ffr = M * fps
    arm = _route.choose_arm(fps, tol_ms=tol_ms, objective=objective)
    seg_sums = (None if lam is not None
                else _segment_count_sums(lam_frame, lam_part, M, seg_bins, n_fine).cpu().numpy())
    out = []
    for a in range(0, n_fine, seg_bins):
        b = min(n_fine, a + seg_bins)
        n_hat = int(round(count_gain * float(lam[a:b].sum() if seg_sums is None else seg_sums[a // seg_bins])))
        if n_hat < 1:
            continue
        pf = prob[a:b]
        if arm == "msadapt":
            bin_ms = _route.bin_ms_for(tol_ms, fps)
            k = fine_pool_factor(M, fps, bin_ms)
            pf64 = pf.astype(np.float64)
            rem = pf64.size % k
            if rem:
                # pool_fine reshapes by k, but this is a SEGMENT slice, not the whole
                # stream: its length is round(segment_s * M * fps), which is not a
                # multiple of k at a non-integer frame rate, and the final segment is a
                # remainder in any case. Zero-pad the tail -- sum-binning is
                # mass-preserving, so the pad adds nothing, cannot attract a spike, and
                # (being at the end) shifts no time.
                pf64 = np.concatenate([pf64, np.zeros(k - rem, dtype=np.float64)])
            pf_d = pool_fine(pf64, k)
            ffr_d = ffr / k
            t = _decode.struct_decode(pf_d, ffr_d, sig_ms / 1000.0 * ffr_d, N=n_hat)
        else:
            t = _decode.struct_decode(pf, ffr, sig_ms / 1000.0 * ffr, N=n_hat)
        out.append(np.asarray(t, np.float64) + a / ffr)
    if not out:
        return np.zeros(0, np.float64)
    return np.sort(np.concatenate(out))


# ----------------------------------------------------------------------- session runner
# Opt-in (the batch runner sets it): keep the prepared ensemble of the last SetsunaSession alive
# in this process so the next session with the same checkpoints and device skips the load.
# Off by default so an interactive GUI process does not pin GPU memory between runs.
REUSE_ENSEMBLE = False
_ENSEMBLE_CACHE: dict = {}


def release_ensemble() -> bool:
    """Drop the cached ensemble (and its CUDA graphs) and give the memory back to the driver.
    Returns True when something was released."""
    had = bool(_ENSEMBLE_CACHE)
    _ENSEMBLE_CACHE.clear()
    if had:
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return had


class SetsunaSession:
    """Load once, run many ROIs. The GUI keeps one per (indicator, band, model_set).

    Takes fractional dF/F over the RAW-F baseline (maximin_dff_from_fluorescence), not the neuropil-corrected one.
    """

    def __init__(self, fps: float, indicator: str = "GCaMP8s", model_set: str = "auto",
                 model_dir: Path | str | None = None, cell_type: str = "pyramidal",
                 device: str = "auto", batch_windows: int = 512, autocast_bf16: bool = False):
        # batch_windows=512, not 256: measured on a 1305-ROI / 157889-frame session
        # (RTX 5090) at 0.914 -> 0.750 s per ROI, 19.9 -> 16.3 min for the session, with
        # BIT-IDENTICAL output -- overlap-add is per window, so batching only changes how
        # many are in flight. Peak VRAM 3.30 -> 4.15 GiB. The curve is flat past 512
        # (1024 and 2048 buy nothing and cost 6.65 GiB), so this is the knee. Lower it on
        # a GPU that is tight on memory.
        indicator = INTERNAL_FROM_DISPLAY.get(indicator, indicator)
        self.is_other = str(indicator).strip().lower().startswith("other")
        if self.is_other:
            indicator = "other"
            if model_set == "specialist":
                raise ValueError("'Other / novel indicator' has no specialist set; it runs the "
                                 "generalist on its unknown-indicator row.")
            model_set = "generalist"
        elif indicator not in INDICATOR_TO_ID:
            raise ValueError(f"unknown indicator {indicator!r}; expected one of {sorted(INDICATOR_TO_ID)}"
                             " or 'other'")
        cell_type = CELLTYPE_FROM_DISPLAY.get(str(cell_type).strip(), cell_type)
        if cell_type not in CELLTYPE_TO_ID:
            raise ValueError(
                f"unsupported cell_type {cell_type!r}: one of {sorted(CELLTYPE_TO_ID)}. "
                "Anything else has an untrained embedding row -- the output would be "
                "plausible-looking garbage rather than an error.")
        self.cell_type = cell_type
        self.fps = float(fps)
        self.indicator = indicator
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.cell_type_note = ""
        allct = None
        if self.cell_type == OTHER_CELL_TYPE:
            allct, band, grid, paths, self.cell_type_note = allct_route(
                self.model_dir, indicator, self.fps, model_set)
        serve = ("pyramidal" if self.cell_type in PYRAMIDAL_TYPES or (self.cell_type == OTHER_CELL_TYPE and allct is None)
                 else self.cell_type)
        self.decode_cell_type = serve
        if allct is not None:
            self.allct_flavor = allct
            self.band, self.band_grid = band, grid
            used = "allct_gen" if allct == "ALL" else "allct"
        elif self.cell_type in SPINAL_TYPES:
            sp_grids = band_grids(self.model_dir, SP_FILE_FMT)
            self.band = pick_band(self.fps, sp_grids)
            self.band_grid = sp_grids[self.band]
            paths = [self.model_dir / SP_FILE_FMT.format(band=self.band, seed=s) for s in SEEDS]
            used = "spinal"
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"spinal set missing {missing} in {self.model_dir}. "
                                        f"{missing_models_hint(self.model_dir)}")
        elif self.cell_type in INTERNEURON_TYPES:
            gen_grids = None
            if model_set == "generalist":
                try:
                    gen_grids = band_grids(self.model_dir, PV_GEN_FILE_FMT)
                except FileNotFoundError:
                    gen_grids = None
            if gen_grids:
                self.pv_flavor = "ALL"
                self.band = pick_band(self.fps, gen_grids)
                self.band_grid = gen_grids[self.band]
                paths = [self.model_dir / PV_GEN_FILE_FMT.format(band=self.band, seed=s) for s in SEEDS]
                used = "pv_gen"
            else:
                pv_grids = pv_band_grids(self.model_dir)
                self.pv_flavor, self.band = pick_band_pv(
                    "other" if self.is_other else indicator, self.fps, pv_grids)
                self.band_grid = pv_grids[self.pv_flavor][self.band]
                paths = [self.model_dir / PV_FILE_FMT.format(ind=self.pv_flavor, band=self.band, seed=s)
                         for s in SEEDS]
                used = "pv_fin"
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"interneuron (allin) set missing {missing} in {self.model_dir}. "
                                        f"{missing_models_hint(self.model_dir)}")
        else:
            # Route on the grids of the family that will actually be loaded: the specialist
            # files when a core indicator has them (and the set allows it), else the ALL set.
            spec_ok = (not self.is_other and indicator in CORE_INDICATORS and model_set != "generalist"
                       and (self.model_dir / SPEC_FILE_FMT.format(ind=indicator, band="LOW", seed=SEEDS[0])).exists())
            grids = (band_grids(self.model_dir, SPEC_FILE_FMT, indicator) if spec_ok
                     else band_grids(self.model_dir, GEN_FILE_FMT))
            self.band = pick_band(self.fps, grids)
            self.band_grid = grids[self.band]
            paths, used = resolve_checkpoints(self.model_dir, model_set,
                                              ("GCaMP8s" if self.is_other else indicator), self.band)
        self.model_set = used
        self.ckpt_names = [p.name for p in paths]
        self.device = resolve_device(device)
        _key = (tuple(str(p) for p in paths), str(self.device), allct is not None)
        _hit = _ENSEMBLE_CACHE.get(_key) if REUSE_ENSEMBLE else None
        if _hit is not None:
            # Batch runner: the prepared ensemble (and its CUDA graphs) of the previous session is
            # reused; the model load + install steps (~5 s) are skipped. Same models, same outputs.
            self.models, self.cfg = _hit
        else:
            self.models, self.cfg = load_ensemble(paths, self.device)
            self.cfg = dict(self.cfg, LEARNED_UNKNOWN=learned_unknown_rows(self.models))
            _extend_zero_identity(self.models)   # appends row 13 = zeros; rows 0..12 untouched
            if allct is not None:
                _extend_zero_cell_type(self.models)   # appends cell-type row 7 = zeros; rows 0..6 untouched
            _install_split_head(self.models)     # bit-identical; drops 12 x 402 MB slice copies per ROI
            _install_fast_layernorm(self.models)  # CUDA: var_mean + addcmul, ~2x the native kernel there
            _install_triton_attention(self.models) # fp32 band attention kernel, ~3x SDPA here
            _install_fused_ffn(self.models)        # compiled FFN block when torch.compile works
            _install_triton_ffn(self.models)       # ... or one Triton kernel per FFN block
            _install_fused_convblock(self.models)  # one Triton kernel per conformer conv block
            _install_host_convblock(self.models)   # CPU: the depthwise conv channels-last, the same bits
            _install_fused_prologue(self.models)   # embedding prologue as one compiled kernel
            if REUSE_ENSEMBLE:
                _ENSEMBLE_CACHE.clear()          # one resident ensemble at a time
                _ENSEMBLE_CACHE[_key] = (self.models, self.cfg)
        self.M = int(self.cfg["M"])
        self.ind_id = OTHER_INDICATOR_ID if self.is_other else INDICATOR_TO_ID[indicator]
        self.nu_min = nu_guard_floor("other" if self.is_other else indicator, self.fps)   # see NU_GUARD_K
        # 1/2/3 TRAINED in the allin set, 4/5 in the spinal set, 0-5 in the all-cell-types set ('other': row 7,
        # the C167 trained token, or an appended zero row on a 7-row file)
        self.ct_id = UNKNOWN_CELL_TYPE_ID if allct is not None else CELLTYPE_TO_ID[serve]
        # which unknown row this session conditions on: True = the checkpoint's TRAINED token, False = zero row
        _lu = (self.cfg.get("LEARNED_UNKNOWN") if isinstance(self.cfg, dict) else None) or (False, False)
        self.indicator_unknown_learned = bool(self.is_other and _lu[0])
        self.cell_type_unknown_learned = bool(allct is not None and _lu[1])
        self.count_calibration_applies = bool(self.cell_type in PYRAMIDAL_TYPES
                                              and used in ("specialist", "generalist"))
        self.batch_windows = int(batch_windows)
        # bf16 autocast is OFF by default since 2026-09-04: on this memory-bound d=24 model it
        # was slower (0.265 vs 0.143 s per ROI, RTX 5090) AND moved the outputs 2.5% relRMS
        # from the float32 model with a systematic bias (+0.2% total spikes). _strict_fp32
        # documents the measurement; the flag stays for experiments only.
        self.autocast_bf16 = bool(autocast_bf16)
        self._prep_pool = None          # one worker for the next ROI's CPU preparation (lazy)
        self._prep_cache = None         # (key, future) from the last prefetch
        self._prep_stats = {}           # key -> (nu, median, std + 1e-6), see use_prep_stats
        self._host_bufs = {}            # CPU: _predict_host_fused's buffers, reused ROI after ROI
        if self.device.type != "cuda" and _host_fused_ok(self.models, self.cfg):
            try:
                _warm_host_kernels()
            except Exception:
                pass

    def _prep(self, dff_row, key=None):
        """The per-ROI CPU work that precedes the GPU: noise level on fractional dF/F, z-score. With ``key`` in the
        statistics the caller handed over (use_prep_stats), standardize_dff's expression on them."""
        row = np.asarray(dff_row, np.float32)
        given = getattr(self, "_prep_stats", None)
        st = given.pop(key, None) if (key is not None and given) else None
        if st is not None:
            nu, m, s = st
            return max(nu, self.nu_min), ((row - m) / s).astype(np.float32)
        return max(noise_level(row, self.fps), self.nu_min), standardize_dff(row)

    def use_prep_stats(self, store=None) -> None:
        """Have _prep take a ROI's statistics from ``store`` (a dict key -> (nu, median, std + 1e-6) that the caller
        fills, e.g. from a thread, with prep_stats_many: the same floats) when its key is there; None = compute them
        per ROI again. For many short ROIs: one at a time, np.nanmedian / np.nanstd held the GIL ~0.25 ms per ROI
        next to the launch-bound forward."""
        self._prep_stats = store if store is not None else {}

    def _prep_for(self, dff_row, key, prefetch):
        """This ROI's (nu, xz), from the prefetch cache when ``key`` matches, else computed now;
        then queue ``prefetch`` = (next_key, next_row) on the worker so it overlaps this ROI's GPU
        work. Numbers are identical either way (same numpy functions on the same rows)."""
        cached = self._prep_cache
        self._prep_cache = None
        if cached is not None and key is not None and cached[0] == key:
            nu, xz = cached[1].result()
        else:
            nu, xz = self._prep(dff_row, key)
        if prefetch is not None:
            if self._prep_pool is None:
                from concurrent.futures import ThreadPoolExecutor
                self._prep_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jupiterscope_setsuna_prep")
            nk, nrow = prefetch
            self._prep_cache = (nk, self._prep_pool.submit(self._prep, nrow, nk))
        return nu, xz

    @property
    def provenance(self) -> str:
        """One factual line naming the model actually selected, for the GUI/CLI to display.

        PROVENANCE ONLY, by decision (user 2026-08-29): which set, which band and the fps range
        it was trained on, how many seeds. Deliberately NO validation status, NO per-sensor cell
        counts and NO performance numbers -- a quality-looking annotation next to a model name
        gets read as a score. The PV sets are labelled by ACQUISITION REGIME, never as an
        indicator specialist, because they are not one (identical training data across flavors).
        """
        if self.model_set in ("allct", "allct_gen"):
            grid = self.band_grid
            what = ("all-cell-types generalist model" if self.model_set == "allct_gen"
                    else f"all-cell-types model ({DISPLAY_NAME.get(self.allct_flavor, self.allct_flavor)})")
        elif self.cell_type in SPINAL_TYPES:
            grid = self.band_grid
            what = "pooled spinal cord model"
        elif self.cell_type in INTERNEURON_TYPES:
            grid = self.band_grid
            what = ("pooled interneuron generalist model" if self.model_set == "pv_gen"
                    else "pooled interneuron model")
        else:
            grid = self.band_grid
            what = ("generalist model" if self.model_set == "generalist"
                    else f"{DISPLAY_NAME.get(self.indicator, self.indicator)} specialist")
        rng = f" ({min(grid):g}-{max(grid):g} Hz)" if grid else ""
        ct = CELLTYPE_LABEL.get(self.cell_type, self.cell_type)
        # ASCII separator on purpose: this string reaches Windows consoles and log files whose
        # default encoding is cp1252, where a middle dot mojibakes (seen in validation output).
        if self.cell_type in (UNKNOWN_CELL_TYPE, OTHER_CELL_TYPE) and self.model_set not in ("allct", "allct_gen"):
            what += " (pyramidal)"
        return f"{ct} | {what} | band {self.band}{rng} | {len(self.ckpt_names)} seeds"

    def infer_roi(self, dff_row: np.ndarray):
        """fractional dF/F (1D) -> dict(rate, lam_fine, prob_fine, nu). NaN-safe rows
        must be filtered by the caller (mirrors the OASIS valid-rows convention)."""
        dff_row = np.asarray(dff_row, np.float32)
        nu = max(noise_level(dff_row, self.fps), self.nu_min)   # BEFORE standardizing, fractional units; guarded
        xz = standardize_dff(dff_row)
        lam, prob = predict_fine(self.models, self.cfg, xz, self.ind_id, self.ct_id,
                                 self.fps, nu, self.device,
                                 batch_windows=self.batch_windows,
                                 autocast_bf16=self.autocast_bf16)
        return {"lam": lam, "prob": prob, "nu": nu,
                "rate": count_head_rate(lam, len(dff_row), self.M)}

    def infer_roi_outputs(self, dff_row: np.ndarray, pool_k=None, need_full: bool = True,
                          full_device: bool = False, key=None, prefetch=None,
                          fine_prob: bool = False, defer_copy: bool = False, seg_bins=None) -> dict:
        """infer_roi, but returning only what the caller will store (see predict_fine_outputs).
        Same nu / standardisation as infer_roi; adds ``nu`` to the dict. ``key`` identifies this
        row and ``prefetch`` = (next_key, next_row) lets the next row's CPU preparation run on a
        worker thread under this row's GPU work (see _prep_for). ``defer_copy`` leaves the host
        copies to finish_outputs() so the caller can overlap them with the next row. ``seg_bins``:
        see predict_fine_outputs (the timing decoder's counts without the fine count stream)."""
        nu, xz = self._prep_for(dff_row, key, prefetch)
        dff_row = np.asarray(dff_row, np.float32)
        out = predict_fine_outputs(self.models, self.cfg, xz, self.ind_id, self.ct_id,
                                   self.fps, nu, self.device,
                                   batch_windows=self.batch_windows,
                                   autocast_bf16=self.autocast_bf16,
                                   pool_k=pool_k, need_full=need_full, full_device=full_device,
                                   fine_prob=fine_prob, defer_copy=defer_copy, seg_bins=seg_bins,
                                   host_bufs=self._host_bufs)
        out["nu"] = nu
        return out
