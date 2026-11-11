"""SETSUNA: spike inference with temporal super-resolution for calcium imaging.

    import setsuna
    dff = setsuna.dff_from_fluorescence(F, Fneu, fps=30.0)
    out = setsuna.infer(dff, fps=30.0, indicator="jGCaMP8s", tolerance_ms=10)

See docs/USER_GUIDE.md.
"""
from .api import CELL_TYPES, INDICATORS, Session, dff_from_fluorescence, infer

__version__ = "0.1.0"
__all__ = ["dff_from_fluorescence", "infer", "Session", "INDICATORS", "CELL_TYPES", "__version__"]
