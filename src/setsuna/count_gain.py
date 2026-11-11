"""SETSUNA's recommended spike-count gain, without SETSUNA's model code.

Kept apart from setsuna.py (which imports torch at import time) like fine_bins.py: the main window applies the rule
while it opens a field, and the Full Pipeline panel while it opens, and importing SETSUNA there loaded torch on the GUI
thread (~1.6 s). The cell-type tables the rule reads live here for the same reason. setsuna.py re-exports these
names, so the GUI and the engine read the same rule.
"""
from __future__ import annotations

# Display label -> key. PV/SST/VIP are all TRAINED rows in the allin interneuron set;
# on the pyramidal set only row 0 is trained, which is why the two sets never mix.
CELLTYPE_FROM_DISPLAY = {
    "Pyramidal": "pyramidal", "pyramidal": "pyramidal",
    "Fast-spiking interneuron (PV)": "PV_FS", "fast-spiking": "PV_FS", "PV": "PV_FS", "pv_fs": "PV_FS",
    "SST interneuron": "SST", "SST": "SST", "sst": "SST",
    "VIP interneuron": "VIP", "VIP": "VIP", "vip": "VIP",
    "Spinal cord excitatory": "spinal_excitatory", "spinal_excitatory": "spinal_excitatory",
    "spinal excitatory": "spinal_excitatory", "SPINAL_EXC": "spinal_excitatory",
    "Spinal cord inhibitory": "spinal_inhibitory", "spinal_inhibitory": "spinal_inhibitory",
    "spinal inhibitory": "spinal_inhibitory", "SPINAL_INH": "spinal_inhibitory",
    "Unknown (probably pyramidal)": "unknown", "unknown": "unknown", "Unknown": "unknown",
    "Other / not listed (striatal, midbrain, cerebellar, olfactory bulb…)": "other",
    "Other / not listed": "other", "other": "other", "Other": "other", "not listed": "other",
}
INTERNEURON_TYPES = ("PV_FS", "SST", "VIP")   # served by the allin set; 0 = the pyramidal sets
SPINAL_TYPES = ("spinal_excitatory", "spinal_inhibitory")   # served by the spinal set (rows 4/5)
UNKNOWN_CELL_TYPE = "unknown"   # "Unknown (probably pyramidal)": the indicator's pyramidal models, exactly
OTHER_CELL_TYPE = "other"       # "Other / not listed": the all-cell-types set on its unknown cell-type token
PYRAMIDAL_TYPES = ("pyramidal", UNKNOWN_CELL_TYPE)   # served by the pyramidal sets (row 0)

# [!] PYRAMIDAL ONLY -- measured on DS29-DS32 (pyramidal V1). See recommended_count_gain.
# Count-gain recommendation: 1.0 at every frame rate, the calibrated count. The 2026-09-06 table
# (1.75 / 1.50 / 1.25) optimised F1 at 10/25 ms, which are sub-frame below ~20 Hz: the pursuit tiles a
# confident frame with a pick every ~9 ms, so a larger gain bought chance matches, not detection.
# Re-scored at frame-scale criteria the optimum is flat ~1.0 (2026-09-12; see recommended_count_gain).
COUNT_GAIN_BY_FPS = ((1e9, 1.0),)
# "Use the recommendation for THIS run": resolved by resolve_count_gain against the frame rate
# and cell type the run actually has. The default everywhere a gain enters the pipeline (both
# panels, the extract CLI, the full-pipeline CONFIG), so a headless or batch run decodes at the
# same operating point the GUI shows -- and each session of a batch at its own frame rate's.
COUNT_GAIN_AUTO = -1.0


def recommended_count_gain(fps: float, cell_type: str = "pyramidal") -> float:
    """Spike-count multiplier for this frame rate: 1.0 at every rate, the calibrated count.

    [2026-09-12] The earlier fps table (1.75 below 8 Hz / 1.50 to 15 Hz / 1.25 above) was fitted on
    F1 at 10 and 25 ms. Those tolerances are sub-frame below ~20 Hz, and the pursuit places a pick
    every ~9 ms across a confident frame, so at low rates a larger gain bought 10-ms matches by
    chance rather than by detection (predicted/true counts 1.5-2.7x). Re-scored at frame-scale
    criteria -- one-frame tolerance and per-frame counts, 120 held-out 8f/6f/6s records -- the best
    gain is flat at ~1.0 for every frame rate (fitted 0.92 at one frame, 1.13 at per-frame counts),
    and the old table loses 0.023 F1 at one frame against 0.004 for a flat 1.0. With 1.0 the
    predicted count is within ~10% of truth above 3 Hz.

    Cell type is accepted for API stability; every type returns 1.0. A user who wants more recall
    at a known precision cost can raise the gain by hand: the pursuit is greedy, so the train at
    gain g is a prefix of the train at any larger gain.
    """
    if CELLTYPE_FROM_DISPLAY.get(str(cell_type).strip(), cell_type) in INTERNEURON_TYPES + SPINAL_TYPES + (OTHER_CELL_TYPE,):
        return 1.0
    f = float(fps)
    for hi, g in COUNT_GAIN_BY_FPS:
        if f < hi:
            return g
    return COUNT_GAIN_BY_FPS[-1][1]


def resolve_count_gain(gain: float, fps: float, cell_type: str = "pyramidal") -> float:
    """COUNT_GAIN_AUTO (any negative value) -> recommended_count_gain(fps, cell_type); any
    value >= 0 is used exactly as given, so 1.0 still reproduces the calibrated count."""
    g = float(gain)
    return float(recommended_count_gain(fps, cell_type)) if g < 0 else g
