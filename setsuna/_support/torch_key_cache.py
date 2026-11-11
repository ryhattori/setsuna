"""Skip torch's per-process source hash (``torch._inductor.codecache.torch_key``) by caching it on disk.

Every process that uses ``torch.compile`` first computes ``torch_key()``: a SHA-256 over every module
file of the installed torch package (2,141 files, 42 MB for torch 2.11) that keys inductor's on-disk
caches to the torch build. With a warm OS file cache that is 0.6-2 s; on a cold one (first run after
other work, after a Defender signature update, after a pipeline streamed a lot of data) it is 12-16 s
-- and it sits on the critical path of a registration's first pass and of SETSUNA's model load.

torch exposes ``torch_key.set(value)`` to prepopulate that cache. :func:`prime_torch_key` computes the
key once per torch install, stores it in ``torch_key/`` of the runtime folder in the home folder (see
user_folders), keyed by a fingerprint of the install (version, git revision, path, and the size and
mtime of torch's own package files that change with any reinstall), and on later processes sets it
without touching the 2,141 files. The
value is exactly what torch would compute, so the compiled-kernel caches are the same ones; a torch
upgrade or reinstall changes the fingerprint and the key is recomputed. ``JSCOPE_TORCH_KEY_CACHE=0``
disables this.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_STATE = {"primed": False, "how": "", "seconds": 0.0}


def _enabled() -> bool:
    return os.environ.get("JSCOPE_TORCH_KEY_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def _cache_dir() -> Path:
    from setsuna._support.user_folders import runtime_dir
    return runtime_dir(beside_package=False) / "torch_key"


def _fingerprint(torch_mod) -> str:
    """A short hex id of the torch install: version, git revision, location, and the (size, mtime) of
    a handful of its own files. Any upgrade or reinstall rewrites those files."""
    import hashlib
    root = Path(torch_mod.__file__).resolve().parent
    h = hashlib.sha256()
    h.update(str(torch_mod.__version__).encode())
    h.update(str(getattr(getattr(torch_mod, "version", None), "git_version", "")).encode())
    h.update(str(root).lower().encode())
    for rel in ("__init__.py", "_inductor/codecache.py", "_inductor/__init__.py", "_dynamo/__init__.py",
                "version.py", "_C.cp311-win_amd64.pyd", "lib/torch_cpu.dll", "lib/torch_python.dll"):
        p = root / rel
        try:
            st = p.stat()
            h.update(f"{rel}:{st.st_size}:{st.st_mtime_ns}".encode())
        except OSError:
            h.update(f"{rel}:missing".encode())
    return h.hexdigest()[:24]


def prime_torch_key() -> str:
    """Set inductor's ``torch_key`` from the on-disk cache, or compute and store it. Returns how the
    key was obtained: 'cached', 'computed', 'already-set', 'disabled' or 'unavailable'. Idempotent."""
    if _STATE["primed"]:
        return str(_STATE["how"])
    _STATE["primed"] = True
    t0 = time.perf_counter()
    if not _enabled():
        _STATE["how"] = "disabled"
        return "disabled"
    try:
        import torch
        from torch._inductor import codecache as cc
    except Exception:
        _STATE["how"] = "unavailable"
        return "unavailable"
    key_fn = getattr(cc, "torch_key", None)
    setter = getattr(key_fn, "set", None)
    if key_fn is None or setter is None:
        _STATE["how"] = "unavailable"
        return "unavailable"
    try:
        path = _cache_dir() / f"{_fingerprint(torch)}.bin"
    except Exception:
        _STATE["how"] = "unavailable"
        return "unavailable"
    how = "computed"
    try:
        cached = path.read_bytes() if path.is_file() else b""
        if len(cached) == 32:                       # a SHA-256 digest
            try:
                setter(bytes(cached))
                how = "cached"
            except AssertionError:                  # torch already computed it in this process
                how = "already-set"
        if how == "computed":
            key = bytes(key_fn())
            if len(key) == 32:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_bytes(key)
                os.replace(tmp, path)
    except Exception:
        how = "unavailable"
    _STATE["how"] = how
    _STATE["seconds"] = time.perf_counter() - t0
    return how
