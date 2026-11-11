"""Give numba its thread-safe OpenMP threading layer on Windows.

numba's parallel kernels run on MS OpenMP (vcomp140.dll), which numba's wheel does not ship: Windows has it only with
the Visual C++ redistributable. Without it numba falls back to its workqueue layer, which ends the process when two
threads launch parallel kernels at the same moment (a background warm-up beside a stage, for instance). Several
packages in the environment ship their own vcomp140.dll; loading one first (by full path, as scikit-learn does)
makes it the one numba's OpenMP module binds to.
"""

import os
import sys
import threading

# Packages that ship vcomp140.dll, and the folder it is in.
_CARRIERS = (("sklearn", ".libs"), ("shiboken6", ""), ("PySide6", ""))


def ensure_vcomp() -> "str | None":
    """Load vcomp140.dll if Windows finds none of its own; the path loaded, "system" when Windows has one, or None
    (not Windows, or no copy found -- numba then uses workqueue)."""
    if sys.platform != "win32":
        return None
    import ctypes

    try:
        ctypes.WinDLL("vcomp140.dll")
        return "system"
    except OSError:
        pass
    from importlib.util import find_spec

    for package, sub in _CARRIERS:
        try:
            spec = find_spec(package)                    # finds the package without importing it
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for root in spec.submodule_search_locations:
            path = os.path.join(root, sub, "vcomp140.dll")
            if os.path.isfile(path):
                try:
                    ctypes.WinDLL(os.path.abspath(path))
                    return path
                except OSError:
                    continue
    return None


def cpu_only_defaults(environ=None, driver_present=None) -> bool:
    """On a Windows PC without NVIDIA's driver -- no nvcuda.dll in System32, so no CUDA device can exist -- set
    KMP_BLOCKTIME=1 and OMP_WAIT_POLICY=PASSIVE where unset: what the GUI gives its child processes on a CPU-only
    machine (manual_classify_neuron_rois_pyqt._cpu_only_child_env), here for every process that imports the package
    -- the headless pipeline, a batch or a script -- before torch's and numba's OpenMP runtimes read them. Intel's
    runtime (torch) then sleeps 1 ms after a parallel region instead of spinning 200 ms, and Microsoft's (numba)
    at once (KMP_BLOCKTIME keeps the Intel runtime at 1 ms): a CPU-only two-photon run 397 -> 354 s, one-photon
    460 -> 397 s, the same outputs. Nothing once torch is imported (its runtime may have read the defaults already),
    with JSCOPE_CPU_OMP_DEFAULTS=0, or on other systems. ``environ`` / ``driver_present`` for tests. True when it
    set something."""
    env = os.environ if environ is None else environ
    if environ is None and (sys.platform != "win32" or "torch" in sys.modules):
        return False
    if str(env.get("JSCOPE_CPU_OMP_DEFAULTS", "1")).strip().lower() in ("0", "false", "no", "off"):
        return False
    if driver_present is None:
        system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        driver_present = os.path.isfile(os.path.join(system32, "nvcuda.dll"))
    if driver_present:
        return False
    changed = False
    for key, value in (("KMP_BLOCKTIME", "1"), ("OMP_WAIT_POLICY", "PASSIVE")):
        if key not in env:
            env[key] = value
            changed = True
    return changed


# ---------------------------------------------------------------------------------------------------------------------
# Where numba has no thread-safe threading layer (macOS: pip's numba brings its OpenMP layer but not the libomp it
# loads, and no TBB is published for Apple Silicon), its parallel kernels run on workqueue, which ends the process
# when two threads launch them at the same moment. There each parallel kernel is launched under one lock.

_LAUNCH_LOCK = threading.RLock()
_WORKQUEUE_ONLY = None


def workqueue_only() -> bool:
    """Whether numba's parallel kernels would run on its workqueue layer here: when $NUMBA_THREADING_LAYER asks for
    it, never otherwise on Windows (ensure_vcomp gives it OpenMP), elsewhere when neither its OpenMP nor its TBB
    layer loads."""
    global _WORKQUEUE_ONLY
    if _WORKQUEUE_ONLY is None:
        if os.environ.get("NUMBA_THREADING_LAYER", "").strip().lower() == "workqueue":
            _WORKQUEUE_ONLY = True
        elif sys.platform == "win32":
            _WORKQUEUE_ONLY = False
        else:
            import importlib

            _WORKQUEUE_ONLY = True
            for layer in ("omppool", "tbbpool"):
                try:
                    importlib.import_module(f"numba.np.ufunc.{layer}")
                    _WORKQUEUE_ONLY = False
                    break
                except Exception:
                    continue
    return _WORKQUEUE_ONLY


class _OneAtATime:
    """A numba parallel kernel launched under the one lock (one_launch_at_a_time); its attributes are the kernel's."""

    def __init__(self, kernel):
        self.__wrapped__ = kernel

    def __call__(self, *args, **kwargs):
        with _LAUNCH_LOCK:
            return self.__wrapped__(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__wrapped__, name)


def one_launch_at_a_time(kernel):
    """For a numba kernel compiled with parallel=True: the kernel itself where numba has a thread-safe threading layer
    (Windows, Linux with libgomp), else (workqueue_only: macOS) the kernel launched under one lock, so a thread
    launching one while another runs waits for it instead of ending the process. Python launches every parallel
    kernel in the package (none is called from another compiled function, where the lock could not be taken)."""
    return kernel if not workqueue_only() else _OneAtATime(kernel)
