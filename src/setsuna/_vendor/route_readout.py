"""[C109/C127 2026-08-08] THE HARD ROUTER: pick a readout arm from (eff, tolerance). One definition.

    from route_readout import choose_arm, decode_routed
    arm = choose_arm(eff_hz, tol_ms=25, objective="detection")
    spikes, arm = decode_routed(pf, cnt, ffr, eff, n_frames, n_hat, tol_ms=25)

WHY A ROUTER: no single decode wins everywhere and the gaps are large (up to +0.14 F1). Measured on
8 cells (4 indicators x 2 folds) x 5 rates x 3 tolerances, nu 8.

    tol  5 ms : struct        (greedy MP on m512) -- wins 5/5 rates
    tol 10 ms : poolOMP32     (pool to 32 bins/frame, then ORTHOGONAL MP) -- +0.040 vs struct,
                              POSITIVE IN 39/40 (cell x rate)
    tol 25 ms : ms16          (fps-adaptive: pool to a ~16 ms bin, then greedy MP) for eff >= 12 Hz
                poolMP8       (pool to 8 bins/frame, then greedy MP) below 12 Hz

FOUR DESIGN RULES, each of which cost something to learn:
 1. THE ROUTER NEVER SEES GROUND TRUTH -- inputs are `eff` (known at deploy) and the tolerance the
    application requires. An auditable 2-D lookup, deliberately not the learned per-record router C106
    refuted.
 2. FIT SEAMS, NOT PER-CELL WINNERS -- one threshold across all rates, so a noisy cell cannot bend it.
 3. STRUCT IS THE DEFAULT; switching needs the 0.027 seed floor.
 4. TIMING / JITTER / CONNECTIVITY ALWAYS ROUTE TO STRUCT (`objective="timing"`), whatever F1 says --
    every other arm runs 5-11 ms jitter against struct's ~3.3. A constraint, not a measurement.

RETIRED: `amp_top` (C121, won 0/30 cells); `POOL` (C127, displaced everywhere by the pooled-MP family);
fixed pool_m as the tol-25 arm (C126, superseded by constant-ms).

[!] THE TWO LOOSE TOLERANCES CANNOT SHARE AN ARM: ms16 - poolOMP32 is -0.032 at tol 10 and +0.030 at
tol 25. At 10 ms you need placement precision AND overlap handling; at 25 ms placement stops mattering
and coarse binning maximises detection.

[!] Tolerance is sampled at 5/10/25 ms only, so the tolerance thresholds interpolate three points; the
eff seam is the better-measured axis. Scope: 4 GCaMP indicators x 2 folds, 1 seed, nu 8, pyramidal.
Re-measure before trusting it on interneurons, another noise level, or jGECO.
"""
from __future__ import annotations
import os
import numpy as np

EFF_SEAM_HZ = float(os.environ.get("ROUTE_EFF_SEAM", "12.0"))   # POOL below, struct/amp_top above
TOL_TIGHT_MS = float(os.environ.get("ROUTE_TOL_TIGHT", "8.0"))  # at or below this, struct everywhere
TOL_TIGHT_PYR_MS = float(os.environ.get("ROUTE_TOL_TIGHT_PYR", "10.0"))


def tight_ms(cell_type=None):
    """The struct/event-count seam for a cell type: 10 ms for 'pyramidal', 8 ms for interneurons and unnamed callers."""
    if cell_type is not None and str(cell_type).strip().lower() == "pyramidal":
        return TOL_TIGHT_PYR_MS
    return TOL_TIGHT_MS
TOL_WIDE_MS = float(os.environ.get("ROUTE_TOL_WIDE", "15.0"))   # at or above this, the wide-tol rules
EFF_POOL_LO_HZ = float(os.environ.get("ROUTE_EFF_POOL_LO", "4.0"))  # legacy; POOL retired at C127
# [C128] The wide-tolerance bin is TOLERANCE-adaptive as well as fps-adaptive: bin_ms = ALPHA * tol_ms.
# Measured best bin/tol for the plain-MP family: 0.80 (tol 5), 0.80 (tol 10), 0.64 (tol 25) -> ALPHA 0.7
# sits inside all three. At tol 25 this gives 17.5 ms, i.e. the `ms16` that was hard-coded before, so
# shipped behaviour barely moves -- the point is that the arm now EXTRAPOLATES to tolerances that were
# never measured instead of being pinned to one number.
# [!] 3 tolerance points, one indicator, one fold. A reasonable parameterisation, not a law.
ROUTE_BIN_ALPHA = float(os.environ.get("ROUTE_BIN_ALPHA", "0.7"))
# [C130] ...and a FRAME term, capped by the tolerance:  bin = min(tol, max(ALPHA*tol, BETA*frame_ms))
# WHY: a pure ALPHA*tol bin is too FINE at low fps. The per-rate oracle at tol 25 wants 25 ms at 5 Hz
# but ALPHA*25 = 17.5, which is where the router lost (4 cells below floor, worst -0.046, all at 5 Hz).
# A bare max(ALPHA*tol, BETA*frame) fixes tol 25 but then over-coarsens at tol 10 (-0.012), because the
# frame term does not know the tolerance. Capping it AT the tolerance fixes both.
# MEASURED vs the per-rate oracle (8f f0, full pool_m sweep): tol 25 gap +0.0000 (EXACT), tol 10 gap
# -0.0014 (vs -0.0007 for the pure alpha rule). beta in 0.10-0.15 is flat; 0.125 is the midpoint.
ROUTE_BIN_BETA = float(os.environ.get("ROUTE_BIN_BETA", "0.125"))

EVC_COMBINED = os.environ.get("EVC_COMBINED", "0").strip().lower() not in ("0", "false", "no", "off")
EVC_COMBINED_GAMMA = float(os.environ.get("EVC_COMBINED_GAMMA", "1.0"))   # placement profile sharpening; 1 = shape unchanged
EVC_SPLIT_TOL_ANY_MS = float(os.environ.get("EVC_SPLIT_TOL_ANY_MS", "10.0"))  # lam^2 at or below this tol...
EVC_SPLIT_TOL_ANY_MAX_HZ = float(os.environ.get("EVC_SPLIT_TOL_ANY_MAX_HZ", "30.0"))
EVC_SPLIT_TOL_MS = float(os.environ.get("EVC_SPLIT_TOL_MS", "25.0"))          # ...and at or below this tol when
EVC_SPLIT_EFF_HZ = float(os.environ.get("EVC_SPLIT_EFF_HZ", "10.0"))          #    the rate is at or below this
EVC_COMBINED_PYR = os.environ.get("EVC_COMBINED_PYR", "1").strip().lower() not in ("0", "false", "no", "off")
EVC_COMBINED_PYR_MAX_HZ = float(os.environ.get("EVC_COMBINED_PYR_MAX_HZ", "12.0"))


def combined_applies(eff, cell_type=None):
    """True when decode_routed's event-count arm runs EVC_COMBINED for this call (see the C163 block above)."""
    if EVC_COMBINED:
        return True
    return bool(EVC_COMBINED_PYR and cell_type is not None and str(cell_type).strip().lower() == "pyramidal"
                and float(eff) <= EVC_COMBINED_PYR_MAX_HZ)


def evc_split_exponents(tol_ms, eff):
    """(a_lam, a_p) for the within-event split of the EVC_COMBINED arm.

    lam^2 where it was measured to win -- tol <= 10 ms at <= 30 Hz, or tol <= 25 ms at <= 10 Hz -- and the
    deployed lam*pbar elsewhere (50/100 ms everywhere, 25 ms at 20-30 Hz, and ANY tolerance above 30 Hz, where the
    2026-09-21 benchmark measured lam^2 as a loss). Returns the deployed (1, 1) for every other cell, so changing the
    flag's scope is a one-line change here."""
    tol = float(tol_ms); e = float(eff)
    if (tol <= EVC_SPLIT_TOL_ANY_MS and e <= EVC_SPLIT_TOL_ANY_MAX_HZ) or (tol <= EVC_SPLIT_TOL_MS and e <= EVC_SPLIT_EFF_HZ):
        return 2.0, 0.0
    return 1.0, 1.0


def bin_ms_for(tol_ms, eff_hz):
    """The pooling bin the `msadapt` arm asks for. fps- AND tolerance-adaptive."""
    tol = float(tol_ms); frame = 1000.0 / max(float(eff_hz), 1e-9)
    return min(tol, max(ROUTE_BIN_ALPHA * tol, ROUTE_BIN_BETA * frame))


def choose_arm(eff_hz, tol_ms=25.0, objective="detection", cell_type=None):
    """Return 'struct' | 'msadapt'.  TWO arms. That is the whole router.

        objective='timing'  -> struct     (jitter / connectivity / latency, always)
        tol <= 8 ms         -> struct     (greedy MP on the full m512 grid)
        tol >  8 ms         -> msadapt    (pool to a bin of ALPHA*tol ms, then greedy MP)

    [C129 2026-08-08] COLLAPSED FROM FOUR ARMS TO TWO after the user asked whether so many were needed.
    They were not. Measured cost of this router vs the 4-arm one (`struct` / `poolOMP32` / `msadapt` /
    `poolMP8`), over 80 (cell x rate x tolerance) combinations on 4 indicators x 2 folds:
        mean -0.0056, below the 0.027 seed floor in 8/80 cells, worst -0.046.
    In exchange it drops two arms, BOTH seams, and an entire second mechanism (orthogonal MP, with its
    least-squares refits and a mutable process-wide flag).

    **THE FINDING THAT MADE IT POSSIBLE: OMP's tol-10 advantage was mostly the POOLING, not the refit.**
    C127 measured poolOMP32 at +0.0397 over `struct` (39/40 cells) and I attributed that to the joint
    refit. Against `msadapt` -- pooled plain MP at the RIGHT bin -- the same arm buys only **+0.0072
    (4/40 above floor)**. The correct comparison was never against `struct`; almost all of the gain was
    available from pooling alone. C127's number stands, its attribution does not.

    WHERE THE 2-ARM ROUTER LOSES, so it is not a surprise later: (1) tol 25 at 5 Hz, 4 cells below floor,
    worst -0.046 -- ALPHA*25 = 17.5 ms there but the optimum is ~25 ms, i.e. the best ALPHA drifts up at
    low rates (a fixed 0.7 is a compromise, and ALPHA=1.0 at 5 Hz is exactly the retired `poolMP8`);
    (2) tol 10, scattered small losses, 4 cells. Everywhere else it is EXACTLY the 4-arm router.
    A 3-arm middle option was measured and is WORSE than this (-0.0071, worst -0.081): `poolMP8` is poor
    at tol 10 low-fps, so keeping it as a low-rate arm hurts more than it helps.

    RETIRED: `POOL` (C127), `amp_top` (C121), `poolOMP32` and `poolMP8` (here). All stay callable in
    `decode_routed` so older grids can be reproduced; none are selected.
    """
    if objective != "detection":
        return "struct"
    return "struct" if float(tol_ms) <= tight_ms(cell_type) else "evcount"


def decode_routed(pf, cnt, ffr, eff, n_frames, n_hat, tol_ms=25.0, objective="detection",
                  m=512, pool_m=16, sig_ms=3.0, nms_ms=2.5, min_frac=0.15, cell_type=None):
    """Dispatch to the routed arm. Mirrors score_timing's decode sites.

    [!] EVERY arm choose_arm can emit MUST have a branch here. When C127 rebuilt the router to emit
    'poolOMP32' / 'ms16' / 'poolMP8', this function still only knew 'struct' / 'amp_top' / POOL, so
    every wide-tolerance call would have SILENTLY fallen through to the POOL branch -- the router
    saying one thing and the decode doing another, which is the whole failure mode this module exists
    to prevent. The assertion at the end makes that unmissable rather than silent.
    """
    import synaptic_timing as S
    import decode_readout as DR
    pf = np.asarray(pf)
    if len(pf) != int(n_frames) * int(m):
        raise ValueError("decode_routed: len(pf)=%d but n_frames*m=%d*%d=%d. `pf` is the fine head at "
                         "m sub-bins per frame; n_frames must be len(pf)//m."
                         % (len(pf), n_frames, m, int(n_frames) * int(m)))
    if int(m) % int(pool_m):
        raise ValueError("decode_routed: pool_m=%d must divide m=%d" % (pool_m, m))
    arm = choose_arm(eff, tol_ms, objective, cell_type=cell_type)
    sig_b = sig_ms / 1000.0 * ffr

    if arm == "struct":                      # greedy MP on the full m512 grid
        return DR.struct_decode(pf, ffr, sig_b, n_hat), arm
    if arm == "poolMP8":                     # pool to 8 bins/frame, then greedy MP
        return DR.poolmp_decode(pf, eff, n_frames, m, 8, sig_ms, n_hat), arm
    if arm == "evcount":
        if combined_applies(eff, cell_type):
            prod = DR.combined_rate(cnt, pf, m, gamma=EVC_COMBINED_GAMMA)
            a_lam, a_p = evc_split_exponents(tol_ms, eff)
            # placement from the combined rate, split weight still from the SOFT head (pfa=pf) unless the
            # routed exponents ask otherwise -- the decoupling that removes the 50/100 ms cost.
            return DR.evcount_decode2(cnt, prod, m, eff, pfa=pf, a_lam=a_lam, a_p=a_p,
                                      tol_ms=tol_ms, gain=1.0), arm
        return DR.evcount_decode(cnt, pf, m, eff, tol_ms=tol_ms, gain=1.0), arm
    if arm == "msadapt":                     # fps- AND tolerance-adaptive: bin = ALPHA * tol, then greedy MP
        pm = DR.pool_m_for_ms(bin_ms_for(tol_ms, eff), eff, m)
        return DR.poolmp_decode(pf, eff, n_frames, m, pm, sig_ms, n_hat), arm
    if arm == "poolOMP32":                   # pool to 32 bins/frame, then ORTHOGONAL MP
        _prev = DR._DECODE_OMP
        try:
            DR._DECODE_OMP = True            # poolmp_decode -> struct_decode reads this module global
            return DR.poolmp_decode(pf, eff, n_frames, m, 32, sig_ms, n_hat), arm
        finally:
            DR._DECODE_OMP = _prev           # restore: this global is process-wide
    if arm == "amp_top":                     # retired by C121; branch kept so old callers still work
        return DR.amptop_decode(pf, ffr, sig_b, n_hat), arm
    if arm == "POOL":                        # retired by C127; kept to reproduce pre-C127 grids
        cw = np.clip(cnt / (np.percentile(cnt, 99) + 1e-9), 0.0, 3.0)
        pooled = (pf * (0.25 + 0.75 * cw)).reshape(n_frames, pool_m, m // pool_m).sum(2).reshape(-1)
        return S.peaks_topN(pooled, n_hat, pool_m * eff, nms_ms=nms_ms, min_frac=min_frac), arm
    raise AssertionError("decode_routed: choose_arm returned %r with no dispatch branch. "
                         "Every arm the router can emit must be handled here." % (arm,))


DEFAULT_TOL_MS = 10.0
DETECT_TOL_MS = 35.0
WIDTH_FRAC = 0.65
WIDTH_CAP_MS = 9.0
WIDTH_CAP_HI_MS = 4.5
WIDTH_CAP_FPS = 30.0
EVENT_TOL_MS = 25.0


def use_detection(tol_ms, eff_hz):
    """True when the detection (event-count) decoder serves an analysis at tolerance ``tol_ms`` and frame rate ``eff_hz``."""
    tol = float(tol_ms)
    return tol >= 1000.0 / max(float(eff_hz), 1e-9) or tol >= DETECT_TOL_MS


def timing_width_ms(tol_ms, eff_hz):
    """The timing decoder's Gaussian width (ms) for an analysis at tolerance ``tol_ms``."""
    cap = WIDTH_CAP_MS if float(eff_hz) <= WIDTH_CAP_FPS else WIDTH_CAP_HI_MS
    return min(WIDTH_FRAC * float(tol_ms), cap)


def decoder_for(tol_ms, eff_hz):
    """('detection', None) or ('timing', width_ms) for an analysis at tolerance ``tol_ms`` (ms)."""
    if not (np.isfinite(float(tol_ms)) and float(tol_ms) > 0):
        raise ValueError("decoder_for: the tolerance must be a positive number of ms, got %r" % (tol_ms,))
    if use_detection(tol_ms, eff_hz):
        return "detection", None
    return "timing", timing_width_ms(tol_ms, eff_hz)


def route_table(effs=(3, 5, 8.1, 10, 15, 20, 24.4, 30.5, 40.7, 60.1), tols=(5, 10, 25)):
    return {float(e): {float(t): choose_arm(e, t) for t in tols} for e in effs}


if __name__ == "__main__":
    T = route_table()
    print("  eff(Hz)  " + "".join("tol%-8g" % t for t in (5, 10, 25)))
    for e, r in T.items():
        print("  %-8.1f " % e + "".join("%-11s" % r[t] for t in (5, 10, 25)))
