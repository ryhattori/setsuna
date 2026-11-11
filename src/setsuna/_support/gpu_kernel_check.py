"""Per-session checks of JupiterScope's own GPU kernels (NVRTC and Triton), one rule for every pipeline.

A custom kernel stands in for a PyTorch operation (cuDNN, cuBLAS, an ATen kernel, an eager chain) or for a CPU
computation. Whether it gives that path's values depends on the GPU: the kernel's own arithmetic is fixed, but the
library it replaces picks per architecture (a cuDNN engine, a cuBLAS order, PyTorch's sm_86 vs sm_120 build), and
a kernel may not compile at all on an older GPU. So before first use in each process, on each device, a kernel's
check runs it beside the path it replaces on representative inputs, and:

(a) every output equal to the bit                                     -> ``bitwise``: the kernel runs;
(b) else, each output's error against a float64 reference no larger
    than the replaced path's own error on the same inputs, within
    ERR_SLACK (and NaN/inf where, and only where, that path has them) -> ``checked``: the kernel runs;
(c) else (or the check raised: no compile, a launch error)            -> ``failed``: the replaced path runs.

The verdict is cached per process, device and kernel, logged once (``[gpu-check] ...``) and kept for the run
metadata (report()). Checks run lazily from the kernels' loaders -- never on import -- and from the pipelines'
warm-up / standby processes (prewarm()), and cost a few milliseconds each.

JSCOPE_GPU_KERNEL_CHECK (one switch for every pipeline):
    unset / 1   the rule above;
    strict      bitwise kernels only (a ``checked`` verdict runs the replaced path);
    0 / off     no custom kernel anywhere (PyTorch's or the CPU's path for all).
The kernels' own switches (JSCOPE_JUNO_NVRTC=0, J2P_REG_FUSE=0, JSCOPE_DWCONV_NVRTC=0, ...) keep working on top.
JSCOPE_GPU_KERNEL_CHECK_FAIL=<name>[,<name>...] (``*`` for all): those checks report ``failed`` (fallback tests).

How a module uses it::

    from setsuna._support import gpu_kernel_check as gkc

    def _check_blur(c, device):          # c: a Comparison
        x = ...
        with gkc.forced("juno.blur", False):
            ref = reference_path(x)              # the replaced path, as the pipeline runs it
        got = kernel_path(x)                     # inside the check the kernel is allowed (forced True)
        c.compare(got, ref, ref64=float64_path(x), what="blur r=6")

    if gkc.use("juno.blur", dev, _check_blur):   # before the kernel's first launch on dev
        ...
"""
from __future__ import annotations

import contextlib
import os
import threading
import time

ENV = "JSCOPE_GPU_KERNEL_CHECK"
FAIL_ENV = "JSCOPE_GPU_KERNEL_CHECK_FAIL"
BITWISE, CHECKED, FAILED = "bitwise", "checked", "failed"
#: Rule (b)'s slack: the kernel's maximum and RMS error against the float64 reference may exceed the replaced path's
#: by this factor, plus SLACK_ULPS units in the last place of the output type at the reference's largest magnitude
#: (so a path that happens to be exact on the check's inputs does not reject a kernel that rounds once more).
ERR_SLACK = 1.25
SLACK_ULPS = 0.5

_LOCK = threading.RLock()
_VERDICTS: dict = {}          # (name, device key) -> record
_LOCAL = threading.local()    # .forced: {name: bool}; .active: [names being checked on this thread]
_LOG = print


def mode() -> str:
    """'default', 'strict' (bitwise kernels only) or 'off' (no custom kernels), from JSCOPE_GPU_KERNEL_CHECK."""
    v = os.environ.get(ENV, "").strip().lower()
    if v in ("strict", "bitwise", "exact"):
        return "strict"
    if v in ("0", "off", "false", "no", "none", "torch"):
        return "off"
    return "default"


def _forced_map() -> dict:
    m = getattr(_LOCAL, "forced", None)
    if m is None:
        m = _LOCAL.forced = {}
    return m


@contextlib.contextmanager
def forced(name: str, on: bool):
    """Within the block, on this thread, use(name, ...) answers ``on`` without a check (a check function's way to
    run the replaced path, or the kernel, through the module's own dispatch)."""
    m = _forced_map()
    had, old = name in m, m.get(name)
    m[name] = bool(on)
    try:
        yield
    finally:
        if had:
            m[name] = old
        else:
            m.pop(name, None)


def device_key(device):
    """'cuda:<index>' for a CUDA device (index None: the current one), str(device) otherwise."""
    try:
        import torch
        dev = torch.device(device) if not isinstance(device, torch.device) else device
    except Exception:
        return str(device)
    if dev.type == "cuda":
        index = dev.index
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{int(index)}"
    return str(dev)


def _describe(device):
    """(GPU name, 'M.m' compute capability) of a CUDA device, ('', '') otherwise."""
    try:
        import torch
        dev = torch.device(device_key(device))
        if dev.type != "cuda":
            return "", ""
        cc = torch.cuda.get_device_capability(dev)
        return torch.cuda.get_device_name(dev), f"{cc[0]}.{cc[1]}"
    except Exception:
        return "", ""


def _simulated_failure(name: str) -> bool:
    v = os.environ.get(FAIL_ENV, "")
    names = {s.strip() for s in v.split(",") if s.strip()}
    return "*" in names or name in names or any(n.endswith("*") and name.startswith(n[:-1]) for n in names)


def _capturing() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


# ----------------------------------------------------------------------------------------------------------------------
# comparisons
# ----------------------------------------------------------------------------------------------------------------------
def _eps(dtype) -> float:
    import torch
    if dtype.is_floating_point:
        return float(torch.finfo(dtype).eps)
    return 0.0


class Comparison:
    """What a check found, output by output: compare() for floating outputs (bitwise, else rule (b)), equal() for
    outputs that must match exactly (indices, counts, integer images)."""

    def __init__(self):
        self.bitwise = True
        self.ok = True
        self.notes: list = []

    def fail(self, what: str):
        self.ok = self.bitwise = False
        self.notes.append(f"{what}: failed")

    def equal(self, got, ref, what: str = "") -> bool:
        """``got`` must equal ``ref`` exactly (same shape; NaN equal to NaN)."""
        same = _bitwise(got, ref)
        if not same:
            self.fail(what or "output")
        return same

    def compare(self, got, ref, ref64=None, what: str = "", slack: float = ERR_SLACK, ulps: float = SLACK_ULPS) -> str:
        """One output of the kernel (``got``) against the replaced path's (``ref``) on the same inputs: 'bitwise'
        when equal to the bit; else, given the float64 reference ``ref64``, 'checked' when the kernel's maximum and
        RMS errors against it are within ``slack`` times the replaced path's (+ ``ulps`` units in the last place of
        the output type at ref64's largest magnitude) and both have NaN / inf at the same places; else 'failed'."""
        if _bitwise(got, ref):
            return BITWISE
        self.bitwise = False
        verdict, note = _within(got, ref, ref64, slack, ulps)
        self.notes.append(f"{what or 'output'}: {note}")
        if verdict != CHECKED:
            self.ok = False
        return verdict

    def verdict(self) -> str:
        if not self.ok:
            return FAILED
        return BITWISE if self.bitwise else CHECKED


def _as_tensor(x):
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return torch.from_numpy(np.asarray(x))
    return torch.as_tensor(x)


def _bitwise(got, ref) -> bool:
    import torch
    if got is None or ref is None:
        return got is None and ref is None
    g, r = _as_tensor(got), _as_tensor(ref)
    if g.shape != r.shape or g.dtype != r.dtype:
        return False
    g, r = g.detach().cpu().contiguous(), r.detach().cpu().contiguous()
    if g.dtype == torch.bool:
        return bool(torch.equal(g, r))
    if g.is_complex():
        g, r = torch.view_as_real(g), torch.view_as_real(r)
    if g.dtype.is_floating_point:
        # the bits, so -0.0 vs 0.0 and NaN payloads count, and NaN equals NaN
        view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[g.element_size()]
        return bool(torch.equal(g.view(view), r.view(view)))
    return bool(torch.equal(g, r))


def _within(got, ref, ref64, slack, ulps):
    """Rule (b) for one output: (verdict, note)."""
    import torch
    if got is None or ref is None:
        return FAILED, "missing output"
    g, r = _as_tensor(got).detach().cpu(), _as_tensor(ref).detach().cpu()
    if g.shape != r.shape:
        return FAILED, f"shape {tuple(g.shape)} vs {tuple(r.shape)}"
    if ref64 is None:
        return FAILED, "differs (no float64 reference)"
    t = _as_tensor(ref64).detach().cpu()
    if t.shape != g.shape:
        return FAILED, f"reference shape {tuple(t.shape)} vs {tuple(g.shape)}"
    if g.is_complex() or r.is_complex() or t.is_complex():
        g, r, t = (torch.view_as_real(v.to(torch.complex128)) if v.is_complex() else v for v in (g, r, t))
    eps = max(_eps(g.dtype) if g.dtype.is_floating_point else 0.0,
              _eps(r.dtype) if r.dtype.is_floating_point else 0.0) or 2.0 ** -23
    g, r, t = g.double(), r.double(), t.double()
    fin_g, fin_r = torch.isfinite(g), torch.isfinite(r)
    if not torch.equal(fin_g, fin_r):
        return FAILED, "non-finite values at other places"
    nf = ~fin_g
    if bool(nf.any()) and not _bitwise(g[nf], r[nf]):
        return FAILED, "non-finite values differ"
    fin = fin_g & torch.isfinite(t)
    if not bool(fin.any()):
        return CHECKED, "non-finite only"
    dg, dr = (g - t)[fin].abs(), (r - t)[fin].abs()
    scale = float(t[fin].abs().max())
    floor = ulps * eps * scale
    gmax, rmax = float(dg.max()), float(dr.max())
    grms, rrms = float(dg.pow(2).mean().sqrt()), float(dr.pow(2).mean().sqrt())
    ok = gmax <= slack * rmax + floor and grms <= slack * rrms + floor
    note = f"max err {gmax:.3g} vs {rmax:.3g}, rms {grms:.3g} vs {rrms:.3g}"
    return (CHECKED if ok else FAILED), note


# ----------------------------------------------------------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------------------------------------------------------
def verdict(name: str, device, check, *, quiet: bool = False) -> str:
    """The verdict for kernel ``name`` on ``device``: ``check(comparison, device)`` runs once per process and
    device (other threads wait for it), its verdict cached and logged once. 'bitwise', 'checked' or 'failed';
    'off' when JSCOPE_GPU_KERNEL_CHECK turns custom kernels off (no check runs then). While a stream is being
    captured into a CUDA graph no check can run: an uncached kernel answers 'failed' for that call only."""
    if mode() == "off":
        return "off"
    key = (name, device_key(device))
    rec = _VERDICTS.get(key)
    if rec is not None:
        return rec["verdict"]
    if _capturing():
        return FAILED
    with _LOCK:
        rec = _VERDICTS.get(key)
        if rec is not None:
            return rec["verdict"]
        gpu, cc = _describe(device)
        t0 = time.perf_counter()
        comp = Comparison()
        error = None
        if _simulated_failure(name):
            comp.fail("simulated (JSCOPE_GPU_KERNEL_CHECK_FAIL)")
        else:
            active = getattr(_LOCAL, "active", None)
            if active is None:
                active = _LOCAL.active = []
            active.append(name)
            try:
                with forced(name, True):
                    check(comp, device)
                _sync(device)
            except Exception as exc:  # noqa: BLE001 - any failure of the kernel's check means its fallback
                error = f"{type(exc).__name__}: {exc}"
                comp.fail(error.splitlines()[0][:200])
            finally:
                active.pop()
        v = comp.verdict()
        seconds = time.perf_counter() - t0
        rec = {"verdict": v, "gpu": gpu, "cc": cc, "check_s": round(seconds, 4),
               "notes": "; ".join(comp.notes)[:500]}
        _VERDICTS[key] = rec
        if not quiet or v == FAILED:
            where = f"{gpu} (cc {cc})" if gpu else key[1]
            what = {BITWISE: "bitwise", CHECKED: "checked (within the replaced path's error)",
                    FAILED: "failed -> the replaced path"}[v]
            tail = f"; {rec['notes']}" if v != BITWISE and rec["notes"] else ""
            _LOG(f"[gpu-check] {name} on {where}: {what} ({seconds * 1000:.0f} ms){tail}", flush=True)
        return v


def _sync(device):
    try:
        import torch
        if device_key(device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(device_key(device)))
    except Exception:
        pass


def use(name: str, device, check) -> bool:
    """Whether kernel ``name`` may run on ``device`` now: forced() on this thread decides first; else its verdict
    (verdict()) is 'bitwise', or 'checked' outside JSCOPE_GPU_KERNEL_CHECK=strict."""
    f = _forced_map().get(name)
    if f is not None:
        return f
    v = verdict(name, device, check)
    if v == BITWISE:
        return True
    return v == CHECKED and mode() == "default"


def enabled() -> bool:
    """False when JSCOPE_GPU_KERNEL_CHECK turns every custom kernel off."""
    return mode() != "off"


def checking(name: str | None = None) -> bool:
    """Whether this thread is inside a check (of ``name``, or of any kernel)."""
    active = getattr(_LOCAL, "active", None) or ()
    return bool(active) if name is None else name in active


def cached(name: str, device):
    """The cached verdict of ``name`` on ``device`` (None: not checked yet)."""
    rec = _VERDICTS.get((name, device_key(device)))
    return None if rec is None else rec["verdict"]


def report(prefix: str = "", device=None) -> dict:
    """{kernel name: {'verdict', 'used', 'gpu', 'cc', 'check_s'(, 'notes')}} of the checks this process ran (names
    starting with ``prefix``; on ``device`` only when given) -- for the run metadata. 'used': whether the kernel
    runs under the current JSCOPE_GPU_KERNEL_CHECK mode."""
    want = None if device is None else device_key(device)
    out = {}
    for (name, dev), rec in sorted(_VERDICTS.items()):
        if not name.startswith(prefix) or (want is not None and dev != want):
            continue
        r = {"verdict": rec["verdict"],
             "used": rec["verdict"] == BITWISE or (rec["verdict"] == CHECKED and mode() == "default"),
             "gpu": rec["gpu"], "cc": rec["cc"], "check_s": rec["check_s"]}
        if rec["verdict"] != BITWISE and rec["notes"]:
            r["notes"] = rec["notes"]
        out[name if want is not None or len({d for (n, d) in _VERDICTS if n == name}) == 1 else f"{name}@{dev}"] = r
    return out


def summary(prefix: str = "", device=None) -> str:
    """One line for a log: 'mode default; juno.blur bitwise, juno.pcg checked, ...' (empty: none checked)."""
    rep = report(prefix, device)
    if not rep:
        return ""
    return f"mode {mode()}; " + ", ".join(f"{k} {v['verdict']}" for k, v in rep.items())


#: Each pipeline's prewarm(device): runs every check of that pipeline's kernels (the warm-up / standby processes, the
#: GPU report). "module:function", imported on first use.
PIPELINES = {
    "reg": "jupiterscope.registration.gpu_checks:prewarm",
    "xsession": "jupiterscope.registration.gpu_checks:prewarm_cross_session",
    "seg": "jupiterscope.segmentation.gpu_checks:prewarm",
    "extract": "jupiterscope.pipeline.extract_gpu_checks:prewarm",
    "setsuna": "jupiterscope.spike_inference.gpu_checks:prewarm",
}


def prewarm(pipelines=None, device="cuda") -> dict:
    """Run the checks of ``pipelines`` (names of PIPELINES; None: all) on ``device`` now, so that no check falls
    inside a run's timed or captured work; returns report(). A pipeline whose checks cannot be imported or raise is
    logged and skipped (its kernels check themselves on first use then)."""
    import importlib
    if mode() == "off":
        return {}
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
    except Exception:
        return {}
    for p in (PIPELINES if pipelines is None else pipelines):
        target = PIPELINES.get(p)
        if target is None:
            continue
        mod, _, fn = target.partition(":")
        try:
            getattr(importlib.import_module(mod), fn)(device)
        except Exception as exc:  # noqa: BLE001
            _LOG(f"[gpu-check] {p}: prewarm failed ({type(exc).__name__}: {exc})", flush=True)
    return report()


def _reset():
    """Forget every verdict (tests)."""
    with _LOCK:
        _VERDICTS.clear()
