"""SETSUNA's recommended saved bin width, without SETSUNA's model code.

Kept apart from setsuna.py (which imports torch at import time) so the GUI can label its
fine-bin picker while building a window without loading torch; setsuna.py re-exports these
names, so both default paths read the same rule.
"""
from __future__ import annotations

RECOMMENDED_FINE_DIVISOR = 16   # bins per frame; flat across the measured 1-35 Hz range
MIN_USEFUL_BIN_MS = 1.0         # below this the probability stream is provably empty


def recommended_fine_divisor(fps: float) -> int:
    """Bins per frame (frame/D) to SAVE at this frame rate -- the GUI default.

    Saving is sum-pooling, and sum-pooling is associative: pooling to frame/D and then
    to an analysis window W gives the same numbers as pooling straight to W whenever the
    bin divides W (worst case one bin of boundary quantisation). So a saved stream is
    lossless for any analysis coarser than its bin, and the only question is how fine an
    analysis the model can support: bin <= the model's own timing error.

    That criterion is FLAT at frame/16 over the whole measured range. From the published
    nested-band headline (sr_loo2_final.json) the SR factor is 10.5-15x the frame at
    5-35 Hz, needing D >= 10.5-15 -> 16; reading the deployed-routing sweep at peak
    capture instead gives D >= 5.1-9.5 -> 8-16 over 1-35 Hz. frame/16 is the safe side of
    both, so there is nothing to gain from an fps-dependent divisor below ~60 Hz.

    The one genuine fps dependence is the high end: above ~62.5 Hz frame/16 falls under
    MIN_USEFUL_BIN_MS, where the probability stream is band-limited by its 3 ms-sigma
    training kernel (variance above 500 Hz = 3.6e-4), so D steps down to keep the bin at
    or above 1 ms: frame/8 at 100-120 Hz, frame/4 at 160 Hz.

    Read at PEAK CAPTURE, not at the finest bin: MP keeps only the spikes it can place,
    so the timing MAD is conditional on capture and improves monotonically with
    resolution while the capture fraction collapses -- it has no interior optimum and
    must not be used on its own to pick a resolution.
    """
    frame_ms = 1000.0 / max(float(fps), 1e-9)
    d = int(RECOMMENDED_FINE_DIVISOR)
    while d > 1 and frame_ms / d < MIN_USEFUL_BIN_MS:
        d //= 2
    return d


FINE_STREAM_ITEMSIZE = 2        # the sub-frame streams are saved as float16


def fine_stream_bytes(n_rois: int, n_frames: int, divisor: int) -> int:
    """Size on disk of ONE saved sub-frame stream (lambda or prob): a float16 per ROI per bin."""
    return int(n_rois) * int(n_frames) * int(divisor) * FINE_STREAM_ITEMSIZE


def _size(nbytes: float) -> str:
    return "%.1f GB" % (nbytes / 1e9) if nbytes >= 1e9 else "%d MB" % max(1, round(nbytes / 1e6))


def fine_stream_size_text(divisor: int, fps: float = 0.0, n_rois: int = 0, n_frames: int = 0,
                          compact: bool = False) -> str:
    """What one stream costs at frame/``divisor``, for the label beside its checkbox: for this session when its
    ROI and frame counts are known, else per 1,000 ROIs per hour at ``fps``, else per 1,000 ROIs per 100,000
    frames. (A long 2p session -- 1611 ROIs, 157,889 frames at 61 Hz -- is 8.1 GB per stream at frame/16.)
    ``compact``: the short form for the main window's narrow panel ("~8.1 GB", "~7.0 GB/1k ROIs/h")."""
    if n_rois > 0 and n_frames > 0:
        size = _size(fine_stream_bytes(n_rois, n_frames, divisor))
        return "~" + size if compact else "~%s for this session" % size
    if fps > 0:
        size = _size(fine_stream_bytes(1000, fps * 3600.0, divisor))
        return "~%s/1k ROIs/h" % size if compact else "~%s per 1,000 ROIs per hour" % size
    size = _size(fine_stream_bytes(1000, 100000, divisor))
    return "~%s/1k ROIs/100k frames" % size if compact else "~%s per 1,000 ROIs per 100,000 frames" % size
