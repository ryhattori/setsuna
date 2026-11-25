"""SETSUNA: spike inference with temporal super-resolution for calcium imaging.

    import setsuna
    dff = setsuna.dff_from_fluorescence(F, Fneu, fps=30.0)
    out = setsuna.infer(dff, fps=30.0, indicator="jGCaMP8s", tolerance_ms=10)

See docs/USER_GUIDE.md.
"""
from .api import CELL_TYPES, INDICATORS, SMOOTH_MS, Session, dff_from_fluorescence, infer, smooth_rate

__version__ = "0.9.0"
__all__ = ["dff_from_fluorescence", "infer", "smooth_rate", "Session", "INDICATORS", "CELL_TYPES", "SMOOTH_MS", "__version__"]
