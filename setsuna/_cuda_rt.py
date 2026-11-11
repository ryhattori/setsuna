"""Runtime-compiled CUDA kernels for SETSUNA, without a CUDA toolkit: NVRTC (the DLL PyTorch ships) compiles the
source to a cubin for this GPU, the driver API loads it, and launches go on PyTorch's current stream, so they are
ordered with torch work and captured by CUDA graphs like any other kernel. Compiled images are cached on disk by
source + options + GPU, so only the first use compiles.

    k = CudaKernel(src, "name", options=("-default-device",))
    k.launch(grid, block, shared_bytes, (tensor_a, tensor_b, (int(n), "i"), (1.5, "f")))

Arguments: tensors (their data pointers), (value, code) pairs with code "i" (int32), "q" (int64), "f" (float32),
"d" (float64), or
ctypes Structure / Array instances (by-value struct parameters, laid out as the kernel declares them).
Any failure raises CudaRtError; callers keep their Triton path then."""
from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import threading
from pathlib import Path

import torch


class CudaRtError(RuntimeError):
    pass


_LOCK = threading.Lock()
_LIBS: dict = {}


def _nvrtc():
    lib = _LIBS.get("nvrtc")
    if lib is None:
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        names = sorted(torch_lib.glob("nvrtc64_*.dll")) if os.name == "nt" else sorted(torch_lib.glob("libnvrtc*.so*"))
        names = [n for n in names if ".alt." not in n.name] or names
        if not names:
            raise CudaRtError("no NVRTC library next to torch")
        lib = ctypes.CDLL(str(names[0]))
        lib.nvrtcGetErrorString.restype = ctypes.c_char_p
        _LIBS["nvrtc"] = lib
    return lib


def _cuda():
    lib = _LIBS.get("cuda")
    if lib is None:
        lib = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        lib.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _LIBS["cuda"] = lib
    return lib


def _cuda_held():
    """The driver through ctypes.PyDLL, whose calls keep the interpreter lock: for launches, which return at once,
    so that a thread does not have to win the lock back from busy threads after each one."""
    lib = _LIBS.get("cuda_held")
    if lib is None:
        lib = ctypes.PyDLL("nvcuda.dll") if os.name == "nt" else ctypes.PyDLL("libcuda.so.1")
        lib.cuLaunchKernel.argtypes = _cuda().cuLaunchKernel.argtypes
        _LIBS["cuda_held"] = lib
    return lib


def _make_current(device):
    """Makes ``device``'s primary context current on this thread when none is. PyTorch makes it current on its own
    CUDA calls, and a thread whose PyTorch work so far came from its caches has made none."""
    lib = _cuda_held()
    ctx = ctypes.c_void_p()
    _check_cu(lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    if ctx.value:
        return
    dev = ctypes.c_int()
    _check_cu(lib.cuDeviceGet(ctypes.byref(dev), device.index if device.index is not None else torch.cuda.current_device()),
              "cuDeviceGet")
    _check_cu(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    try:
        _check_cu(lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    finally:
        lib.cuDevicePrimaryCtxRelease(dev)        # PyTorch's own reference keeps the context


def _check_nvrtc(res, what):
    if res != 0:
        raise CudaRtError(f"{what}: {_nvrtc().nvrtcGetErrorString(res).decode(errors='replace')}")


def _check_cu(res, what):
    if res != 0:
        s = ctypes.c_char_p()
        _cuda().cuGetErrorString(res, ctypes.byref(s))
        raise CudaRtError(f"{what}: CUDA error {res} ({(s.value or b'?').decode(errors='replace')})")


def _cache_dir() -> Path:
    base = os.environ.get("JSCOPE_CUDA_RT_CACHE") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                                                                "jupiterscope", "cuda_rt")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compile_cubin(src: str, name: str, arch: str, options=()) -> bytes:
    """NVRTC: CUDA C++ source -> a loadable image for ``arch``, cached on disk: a cubin for a real architecture
    ("sm_120"), PTX (null-terminated; the driver compiles it for the GPU on load) for a virtual one ("compute_75")."""
    opts = [f"--gpu-architecture={arch}"] + list(options)
    ptx = arch.startswith("compute_")
    key = hashlib.sha256(("\0".join([src, name] + opts) + "\0v1").encode()).hexdigest()[:32]
    f = _cache_dir() / f"{name}_{arch}_{key}.{'ptx' if ptx else 'cubin'}"
    if f.exists():
        return f.read_bytes()
    lib = _nvrtc()
    prog = ctypes.c_void_p()
    _check_nvrtc(lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), (name + ".cu").encode(), 0, None, None),
                 "nvrtcCreateProgram")
    try:
        c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        res = lib.nvrtcCompileProgram(prog, len(opts), c_opts)
        if res != 0:
            n = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(n.value + 1)
            lib.nvrtcGetProgramLog(prog, log)
            raise CudaRtError(f"NVRTC compile of {name} failed:\n{log.value.decode(errors='replace')}")
        n = ctypes.c_size_t()
        if ptx:
            _check_nvrtc(lib.nvrtcGetPTXSize(prog, ctypes.byref(n)), "nvrtcGetPTXSize")
            buf = ctypes.create_string_buffer(n.value)
            _check_nvrtc(lib.nvrtcGetPTX(prog, buf), "nvrtcGetPTX")
        else:
            _check_nvrtc(lib.nvrtcGetCUBINSize(prog, ctypes.byref(n)), "nvrtcGetCUBINSize")
            buf = ctypes.create_string_buffer(n.value)
            _check_nvrtc(lib.nvrtcGetCUBIN(prog, buf), "nvrtcGetCUBIN")
        img = buf.raw
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
    tmp = f.with_suffix(".tmp%d" % os.getpid())
    tmp.write_bytes(img)
    os.replace(tmp, f)
    return img


def supported_archs() -> tuple:
    """The architectures (as 75, 80, ..., 120) this NVRTC compiles for; () when it cannot say."""
    got = _LIBS.get("archs")
    if got is None:
        try:
            lib = _nvrtc()
            n = ctypes.c_int()
            _check_nvrtc(lib.nvrtcGetNumSupportedArchs(ctypes.byref(n)), "nvrtcGetNumSupportedArchs")
            arr = (ctypes.c_int * n.value)()
            _check_nvrtc(lib.nvrtcGetSupportedArchs(arr), "nvrtcGetSupportedArchs")
            got = tuple(sorted(int(v) for v in arr))
        except Exception:
            got = ()
        _LIBS["archs"] = got
    return got


#: JSCOPE_CUDA_RT_ARCH=compute_75 (a test setting): every kernel compiled as PTX for that virtual architecture, which
#: the driver compiles for the GPU on load -- an older architecture's code paths (__CUDA_ARCH__) run on a newer GPU.
ARCH_ENV = "JSCOPE_CUDA_RT_ARCH"


def device_arch(device) -> str:
    """NVRTC's target for ``device``: its own architecture ("sm_89") when NVRTC knows it; else PTX for the newest
    architecture NVRTC knows below it ("compute_120" on a cc 12.1 GPU with NVRTC 12.8), which the driver compiles for
    the GPU on load. JSCOPE_CUDA_RT_ARCH overrides it (tests)."""
    forced = os.environ.get(ARCH_ENV, "").strip()
    if forced:
        return forced
    major, minor = torch.cuda.get_device_capability(device)
    cc = major * 10 + minor
    archs = supported_archs()
    if not archs or cc in archs:
        return f"sm_{major}{minor}"
    lower = [a for a in archs if a <= cc]
    if not lower:
        return f"sm_{major}{minor}"            # (older than NVRTC knows: the compile fails, callers fall back)
    return f"compute_{lower[-1]}"


def device_cc(device) -> int:
    """The compute capability the kernels are compiled for, as 75, 89, 120: JSCOPE_CUDA_RT_ARCH's when set, else the
    device's -- for sources that pick a variant by architecture before compiling."""
    forced = os.environ.get(ARCH_ENV, "").strip()
    if forced:
        return int(forced.split("_")[-1])
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


_ARG_TYPES = {"i": ctypes.c_int32, "q": ctypes.c_int64, "f": ctypes.c_float, "d": ctypes.c_double}
# (absent from torch builds without CUDA -- macOS's -- where no kernel is ever launched)
_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


class CudaKernel:
    """One kernel of a runtime-compiled module, loaded in the current CUDA context of ``device``."""

    def __init__(self, src: str, name: str, device=None, options=(), max_dynamic_smem: int = 0):
        dev = torch.device("cuda" if device is None else device)
        with torch.cuda.device(dev):
            torch.cuda.current_stream(dev)
            # PyTorch's primary context, current on this thread even when PyTorch has made no CUDA call here
            _make_current(dev if dev.index is not None else torch.device("cuda", torch.cuda.current_device()))
            img = compile_cubin(src, name, device_arch(dev), options)
            cu = _cuda()
            self._mod = ctypes.c_void_p()
            self._img = ctypes.create_string_buffer(img, len(img))
            _check_cu(cu.cuModuleLoadData(ctypes.byref(self._mod), self._img), f"cuModuleLoadData({name})")
            self._fn = ctypes.c_void_p()
            _check_cu(cu.cuModuleGetFunction(ctypes.byref(self._fn), self._mod, name.encode()),
                      f"cuModuleGetFunction({name})")
            if max_dynamic_smem > 48 * 1024:
                # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
                _check_cu(cu.cuFuncSetAttribute(self._fn, 8, int(max_dynamic_smem)), "cuFuncSetAttribute")
        self.name = name
        self.device = dev
        self._what = f"cuLaunchKernel({name})"

    def launch(self, grid, block, shared_bytes: int, args, stream=None, hold=False) -> None:
        """``hold`` keeps the interpreter lock through the launch call."""
        holders = []
        params = (ctypes.c_void_p * len(args))()
        for k, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                h = ctypes.c_void_p(a.data_ptr())
            elif isinstance(a, (ctypes.Structure, ctypes.Array)):
                h = a                                           # a by-value struct / array parameter
            else:
                v, code = a
                h = _ARG_TYPES[code](v)
            holders.append(h)
            params[k] = ctypes.addressof(h)
        if stream is not None:
            handle = stream.cuda_stream
        else:                                                   # the current stream's handle, as current_stream gives it
            index = self.device.index
            handle = _current_raw_stream(index if index is not None else torch.cuda.current_device())
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        if hold:
            _make_current(self.device)
        _check_cu((_cuda_held() if hold else _cuda()).cuLaunchKernel(
            self._fn, g[0], g[1], g[2], b[0], b[1], b[2], int(shared_bytes), ctypes.c_void_p(handle), params, None),
            self._what)
