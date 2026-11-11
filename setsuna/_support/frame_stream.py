"""Stream a large int16 frame binary through pinned host memory to the GPU at disk speed.

Every heavy stage streams the same object -- the registered (or raw) movie, an
``(n_frames, Ly, Lx)`` int16 binary of tens of GB -- and until 2026-09-04 each did it the
same slow way: ``np.array(memmap[s:e])`` (a page-fault-driven copy) followed by a pageable
host->device copy (a second copy, synchronous). Measured on the 53 GB session binary of a
1305-ROI recording (RTX 5090, NVMe, file in page cache): 1.2-1.6 GB/s. A direct
``readinto`` of the same bytes into a pinned buffer followed by a non-blocking copy runs
at 5.5-6.5 GB/s on one thread and 7.7 GB/s with two reader threads -- the disk's own
sequential ceiling. The frames delivered are the same bytes, so a consumer's arithmetic is
untouched; only how the bytes travel changes.

    for start, stop, chunk in stream_frames(path, (n, Ly, Lx), ranges, device):
        ...  # chunk: (stop - start, Ly, Lx) int16 on `device`, in range order

``ranges`` is the list of (start, stop) frame ranges to deliver, in order (any chunking the
consumer wants). ``n_readers`` reader threads keep ``depth`` pinned buffers ahead; a buffer
is refilled only after the host->device copy that consumed it has completed (a CUDA event
per slot). With ``device=None`` the pinned host tensor itself is yielded instead (the
consumer must not keep it past the next iteration).

``write_frames`` is the matching output side: sequential ``write()`` of contiguous frame
blocks from (pinned) host memory, which runs well above memmap assignment.
"""
from __future__ import annotations

import collections
import os
import sys
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator, Sequence

import numpy as np

try:  # torch is optional at import time so numpy-only callers can import the writer
    import torch
except Exception:  # pragma: no cover - depends on the environment
    torch = None


def _backing_memmap(arr):
    """The ROOT np.memmap behind ``arr`` (the one created by ``np.memmap(...)``, whose
    ``.base`` is the mmap object), reached through the ``.base`` chain of views, or None.

    Only a C-contiguous view whose bytes lie inside that memmap qualifies -- the frame
    ranges the readers fetch must be laid out in the file exactly as in ``arr``. A sliced
    memmap keeps its parent's ``offset`` attribute, so offsets are always measured from
    the root's data pointer, never read from a view.
    """
    if arr is None or not isinstance(arr, np.ndarray) or not arr.flags["C_CONTIGUOUS"]:
        return None
    root = None
    node = arr
    for _ in range(32):
        if isinstance(node, np.memmap) and getattr(node, "filename", None):
            root = node
        nxt = getattr(node, "base", None)
        if not isinstance(nxt, np.ndarray):
            break
        node = nxt
    if root is None:
        return None
    p_arr = int(arr.__array_interface__["data"][0])
    p_root = int(root.__array_interface__["data"][0])
    if p_arr < p_root or p_arr + arr.nbytes > p_root + root.nbytes:
        return None
    return root


def frames_path(arr) -> "str | None":
    """The backing file of a numpy memmap -- or of a C-contiguous view of one (a slice,
    an ``np.asarray``/``reshape`` of a memmap keeps it as ``.base``). None for in-memory
    arrays, so callers fall back to their memory paths."""
    base = _backing_memmap(arr)
    return str(base.filename) if base is not None else None


def memmap_offset(arr) -> int:
    """Byte offset of ``arr``'s first element in its backing file (0 when not file-backed):
    the memmap's own offset plus the view's displacement from the memmap's start."""
    base = _backing_memmap(arr)
    if base is None:
        return int(getattr(arr, "offset", 0) or 0)
    p_arr = int(arr.__array_interface__["data"][0])
    p_base = int(base.__array_interface__["data"][0])
    return int(getattr(base, "offset", 0) or 0) + (p_arr - p_base)


def frames_view_ok(b) -> bool:
    """``b`` is a (T, H, W) stack whose frames are each C-contiguous at one constant stride of at least a frame
    -- a TIFF's pages viewed in place (phase_a_io's zero-copy decode). FrameChunks keeps such blocks as they are
    and the pinned streamer uploads them with one pitched DMA per range."""
    return (isinstance(b, np.ndarray) and b.ndim == 3 and b.shape[0] > 0
            and b.strides[2] == b.itemsize and b.strides[1] == b.shape[2] * b.itemsize
            and b.strides[0] >= b.shape[1] * b.shape[2] * b.itemsize and b.strides[0] % b.itemsize == 0)


def frames_view_or_contiguous(b):
    """``b`` itself when C-contiguous or a frames view (:func:`frames_view_ok`), else a contiguous copy."""
    b = np.asarray(b)
    if b.flags["C_CONTIGUOUS"] or frames_view_ok(b):
        return b
    return np.ascontiguousarray(b)


class FrameChunks:
    """A ``(T, H, W)`` frame stack held as consecutive in-memory blocks (one per decoded TIFF)
    without ever concatenating them -- peak RAM is the movie once, not twice.

    Read-only, memmap-like along the frame axis: ``.shape``, ``.dtype``, ``.ndim``, ``len()``,
    ``x[i]``, ``x[a:b]`` (a view when the range lies in one block, else a copy), ``x[idx]`` with
    an index array, and ``x[a:b, ...]`` tuples. ``copy_range_into`` fills a caller buffer.
    """

    def __init__(self, blocks: Sequence[np.ndarray]):
        blocks = [frames_view_or_contiguous(b) for b in blocks if b.shape[0] > 0]
        if not blocks:
            raise ValueError("FrameChunks needs at least one non-empty block")
        if len({b.shape[1:] for b in blocks}) != 1 or len({b.dtype for b in blocks}) != 1:
            raise ValueError("FrameChunks blocks must share frame shape and dtype")
        self.blocks = blocks
        self.frame_shape = tuple(int(v) for v in blocks[0].shape[1:])
        self.dtype = blocks[0].dtype
        self.starts = np.cumsum([0] + [int(b.shape[0]) for b in blocks])
        self.shape = (int(self.starts[-1]),) + self.frame_shape
        self.ndim = 1 + len(self.frame_shape)
        self.nbytes = int(sum(b.nbytes for b in blocks))

    # An optional copy of the movie's last frames on the GPU: ``(first frame, device tensor)``. The in-process
    # re-warp keeps it for the extraction, whose stream_frames then hands those frames out on the device
    # instead of uploading them again; their host blocks are filled by device->host copies after the pass,
    # and a host read of any of those frames waits for them (``_tail_ready``).
    device_tail = None
    _tail_start = None
    _tail_ready = None
    _tail_go = None

    def attach_device_tail(self, start: int, tensor, ready: "threading.Event",
                           go: "threading.Event | None" = None) -> None:
        """Frames ``[start, len)`` are also held in ``tensor`` on the device; their host blocks are
        valid once ``ready`` is set (``go``, when given, starts the copies that fill them)."""
        self._tail_start = int(start)
        self._tail_ready = ready
        self._tail_go = go
        self.device_tail = (int(start), tensor)

    def _await_tail(self) -> None:
        go = self._tail_go
        if go is not None:
            go.set()
        self._tail_ready.wait()

    def release_device_tail(self) -> None:
        """Drop the device copy of the tail (its consumers are done with it); waits for the host
        copies that still read it."""
        tail, self.device_tail = self.device_tail, None
        if self._tail_ready is not None:
            self._await_tail()
        del tail

    def __len__(self) -> int:
        return self.shape[0]

    def _block_of(self, k: int) -> int:
        return int(np.searchsorted(self.starts, k, side="right") - 1)

    def _block(self, i: int) -> np.ndarray:
        b = self.blocks[i]
        if b is None:
            raise RuntimeError(f"FrameChunks: frames {int(self.starts[i])}-{int(self.starts[i + 1])} were released")
        ts = self._tail_start
        if ts is not None and int(self.starts[i + 1]) > ts:
            ready = self._tail_ready
            if ready is not None and not ready.is_set():
                self._await_tail()             # the block's frames are still on their way from the device
        return b

    def release_upto(self, frame: int, sink=None) -> int:
        """Drop every block whose frames all lie below ``frame`` (their memory is freed once
        no other reference holds them, or handed to ``sink.give(block)`` for reuse).
        Returns the number of blocks released."""
        n = 0
        for i in range(len(self.blocks)):
            if self.blocks[i] is None:
                continue
            if int(self.starts[i + 1]) <= int(frame):
                b = self.blocks[i]
                self.blocks[i] = None
                if sink is not None:
                    sink.give(b)
                n += 1
            else:
                break
        return n

    def copy_range_into(self, start: int, stop: int, dst) -> None:
        """Copy frames ``[start, stop)`` into ``dst`` (an array of shape ``(stop-start, *frame_shape)``)."""
        i = self._block_of(start)
        pos = start
        while pos < stop:
            b = self._block(i)
            b0 = int(self.starts[i])
            a, e = pos - b0, min(stop - b0, b.shape[0])
            dst[pos - start: pos - start + (e - a)] = b[a:e]
            pos = b0 + e
            i += 1

    def copy_range_from(self, start: int, stop: int, src) -> None:
        """Overwrite frames ``[start, stop)`` with ``src`` (shape ``(stop-start, *frame_shape)``)."""
        i = self._block_of(start)
        pos = start
        while pos < stop:
            b = self._block(i)
            b0 = int(self.starts[i])
            a, e = pos - b0, min(stop - b0, b.shape[0])
            b[a:e] = src[pos - start: pos - start + (e - a)]
            pos = b0 + e
            i += 1

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                raise IndexError("FrameChunks assignment needs a unit-step slice")
            if stop > start:
                self.copy_range_from(start, stop, np.asarray(value))
            return
        if isinstance(key, (int, np.integer)):
            k = int(key) + (self.shape[0] if int(key) < 0 else 0)
            i = self._block_of(k)
            self._block(i)[k - int(self.starts[i])] = value
            return
        raise IndexError("FrameChunks assignment supports an int or a unit-step slice")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            first, rest = key[0], key[1:]
            out = self[first]
            return out[(slice(None),) + rest] if rest else out
        n = self.shape[0]
        if isinstance(key, (int, np.integer)):
            k = int(key) + (n if int(key) < 0 else 0)
            if not 0 <= k < n:
                raise IndexError(k)
            i = self._block_of(k)
            return self._block(i)[k - int(self.starts[i])]
        if isinstance(key, slice):
            start, stop, step = key.indices(n)
            if step != 1:
                return self[np.arange(start, stop, step)]
            if stop <= start:
                return np.empty((0,) + self.frame_shape, self.dtype)
            i = self._block_of(start)
            if stop <= int(self.starts[i + 1]):          # one block -> a view, no copy
                b0 = int(self.starts[i])
                return self._block(i)[start - b0: stop - b0]
            out = np.empty((stop - start,) + self.frame_shape, self.dtype)
            self.copy_range_into(start, stop, out)
            return out
        idx = np.asarray(key)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64).ravel()
        idx = np.where(idx < 0, idx + n, idx)
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise IndexError("frame index out of range")
        out = np.empty((idx.size,) + self.frame_shape, self.dtype)
        bi = np.searchsorted(self.starts, idx, side="right") - 1
        for j in range(idx.size):
            i = int(bi[j])
            out[j] = self._block(i)[int(idx[j]) - int(self.starts[i])]
        return out


class GrowingFrameChunks(FrameChunks):
    """A :class:`FrameChunks` whose blocks arrive one by one from a producer (the TIFF decode) while
    consumers already read it: the per-block frame counts are known up front (probed from the files),
    ``shape`` is final from the start, and a read of frames not decoded yet waits for them. The producer
    calls ``put(i, block)`` for each block and ``fail(exc)`` if it cannot finish, which wakes and fails
    every waiting reader. ``expect_pinned``: every block is registered (cudaHostRegister) before a DMA
    reads it, so the direct DMA streamer may be used before all blocks exist -- by the producer before
    handing it over, or with ``pin_on_read`` by the pinned streamer right before its first copy (on the
    thread that launches the GPU work, between launches; see _stream_frames_pinned)."""

    def __init__(self, counts, frame_shape, dtype, expect_pinned: bool = False, pin_on_read: bool = False):
        counts = [int(c) for c in counts]
        if not counts or min(counts) <= 0:
            raise ValueError("GrowingFrameChunks needs positive block frame counts")
        self.blocks = [None] * len(counts)
        self.frame_shape = tuple(int(v) for v in frame_shape)
        self.dtype = np.dtype(dtype)
        self.starts = np.cumsum([0] + counts)
        self.shape = (int(self.starts[-1]),) + self.frame_shape
        self.ndim = 1 + len(self.frame_shape)
        self.nbytes = int(self.shape[0]) * int(np.prod(self.frame_shape)) * int(self.dtype.itemsize)
        self.expect_pinned = bool(expect_pinned)
        self.pin_on_read = bool(pin_on_read) and self.expect_pinned
        self._pin_cursor = 0
        self._produced = [False] * len(counts)
        self._cv = threading.Condition()
        self._error = None

    def put(self, i: int, block) -> None:
        i = int(i)
        block = frames_view_or_contiguous(block)
        want = (int(self.starts[i + 1] - self.starts[i]),) + self.frame_shape
        if tuple(block.shape) != want or block.dtype != self.dtype:
            exc = ValueError(f"GrowingFrameChunks: block {i} is {tuple(block.shape)} {block.dtype}, "
                             f"expected {want} {self.dtype}")
            self.fail(exc)
            raise exc
        with self._cv:
            self.blocks[i] = block
            self._produced[i] = True
            self._cv.notify_all()

    def pin_next(self) -> bool:
        """With ``pin_on_read``: register the lowest produced block that is not registered yet (one per
        call) -- for a consumer that calls it while its GPU work is queued and it would only wait. False
        when there is none (the streamer still registers any block it reaches unregistered)."""
        if not self.pin_on_read:
            return False
        i = self._pin_cursor
        while i < len(self.blocks) and self._produced[i]:
            b = self.blocks[i]
            i += 1
            self._pin_cursor = i
            if b is not None and not is_pinned_array(b):
                pin_host_array(b)
                return True
        return False

    def fail(self, exc: BaseException) -> None:
        with self._cv:
            if self._error is None:
                self._error = exc
            self._cv.notify_all()

    def complete(self) -> bool:
        return all(self._produced)

    def _block(self, i: int) -> np.ndarray:
        if not self._produced[i]:
            with self._cv:
                while not self._produced[i]:
                    if self._error is not None:
                        raise RuntimeError("the decode feeding this movie failed") from self._error
                    self._cv.wait()
        return super()._block(i)

    def release_upto(self, frame: int, sink=None) -> int:
        n = 0
        for i in range(len(self.blocks)):
            if not self._produced[i]:
                break
            if self.blocks[i] is None:
                continue
            if int(self.starts[i + 1]) <= int(frame):
                b = self.blocks[i]
                self.blocks[i] = None
                if sink is not None:
                    sink.give(b)
                n += 1
            else:
                break
        return n


# ---------------------------------------------------------------------------------------
# In-RAM movies published under their binary path (the in-process pipeline): a stage that
# would memmap ``image_postreg.bin`` gets the registered frames straight from memory
# instead, while the file is still being written in the background. Fused per-bin-size
# segmentation results computed during the re-warp travel with the movie. Deferred jobs
# (cache saves that need the finished file) run once the background writer is done.
# ---------------------------------------------------------------------------------------
_PUBLISHED: dict = {}
_DEFERRED: list = []
# Facts about the session being processed that a hook may need (register_lean notes the frame rate
# and frame count after Phase A) and objects one stage prepared for a later one (the in-process
# runner's SETSUNA preload). Plain dicts: nothing here is required by any stage.
SESSION: dict = {}
PRELOADED: dict = {}
# Held by background threads for CUDA work that must not overlap a CUDA-graph capture on the main thread
# (module loads, device synchronisation, their own captures); registration's per-batch graph captures
# only try-acquire it and retry on a later batch.
class DerivedFrames:
    """A (T, H, W) int16 movie computed from another one whenever it is read, never stored -- adaptive
    estimation's high-pass frames, a one-photon movie minus its still pattern. :func:`stream_frames` hands its
    frames out through ``device_ranges``; reads (``x[i]``, ``x[a:b]``, ``x[index array]``, ``x[i, ...]``) return
    host arrays (``frames``). Subclasses set ``shape`` and ``device`` and implement ``device_ranges``."""

    dtype = np.dtype(np.int16)
    ndim = 3
    shape: tuple = (0, 0, 0)
    device = None
    #: The movie whose frame i alone gives this one's frame i (a pass that has read frames [0, i) of this one is done
    #: with them in that movie too, and may release its blocks), or None.
    frame_source = None

    def __len__(self) -> int:
        return int(self.shape[0])

    @property
    def frame_shape(self) -> tuple:
        return tuple(int(v) for v in self.shape[1:])

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def device_ranges(self, ranges, device, recycle: bool = False):
        """``(start, stop, int16 tensor)`` per frame range, in order, on ``device`` -- stream_frames' contract."""
        raise NotImplementedError

    def frames(self, idx) -> np.ndarray:
        """The frames at ``idx`` (any order, repeats allowed) as a (len(idx), H, W) array, computed from the runs
        of consecutive frames among them."""
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if len(idx) == 0:
            return np.empty((0,) + self.frame_shape, self.dtype)
        uniq, inv = np.unique(idx, return_inverse=True)
        cut = np.flatnonzero(np.diff(uniq) != 1) + 1
        starts, stops = np.concatenate([[0], cut]), np.concatenate([cut, [len(uniq)]])
        ranges = [(int(uniq[s]), int(uniq[e - 1]) + 1) for s, e in zip(starts, stops)]
        vals = np.empty((len(uniq),) + self.frame_shape, self.dtype)
        pos = 0
        for a, b, x in self.device_ranges(ranges, self.device):
            vals[pos:pos + (b - a)] = x.cpu().numpy()
            pos += b - a
        return vals if len(uniq) == len(idx) and np.array_equal(uniq, idx) else vals[inv.reshape(-1)]

    def __getitem__(self, key):
        n = int(self.shape[0])
        if isinstance(key, tuple):
            first, rest = (key[0], tuple(key[1:])) if key else (slice(None), ())
            got = self[first]
            if not rest:
                return got
            return got[rest] if isinstance(first, (int, np.integer)) else got[(slice(None),) + rest]
        if isinstance(key, (int, np.integer)):
            k = int(key) + (n if int(key) < 0 else 0)
            if not 0 <= k < n:
                raise IndexError(f"frame {int(key)} is out of range for {n} frames")
            return self.frames([k])[0]
        if isinstance(key, slice):
            return self.frames(np.arange(*key.indices(n)))
        idx = np.asarray(key)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64)
        return self.frames(np.where(idx < 0, idx + n, idx))

    def __array__(self, dtype=None, copy=None):
        out = self.frames(np.arange(int(self.shape[0])))
        return out if dtype is None else out.astype(dtype)


def in_memory_source(src) -> bool:
    """``src`` is a movie :func:`stream_frames` reads without a file: a FrameChunks, an ndarray or DerivedFrames."""
    return isinstance(src, (FrameChunks, np.ndarray, DerivedFrames))


def stream_source(src, ranges, device, recycle: bool = True):
    """:func:`stream_frames` of the int16 movie ``src``: a file-backed memmap read from its file, anything else
    as it is."""
    path = frames_path(src) if getattr(src, "dtype", None) == np.int16 else None
    return stream_frames(path if path is not None else src, tuple(int(v) for v in src.shape), ranges, device,
                         offset=memmap_offset(src) if path is not None else 0, recycle=recycle)


GRAPH_LOCK = threading.Lock()

_CAPTURE_STREAMS: dict = {}
_CAPTURE_STREAMS_LOCK = threading.Lock()


def capture_stream(device=None):
    """The stream the calling thread captures CUDA graphs on: one per thread and device, created by the CUDA
    runtime (cudaStreamCreateWithFlags, non-blocking) and never taken from torch's stream pool.

    torch.cuda.Stream() hands out the pool's 32 streams per device round-robin, and torch.cuda.graph's default
    capture stream is one of them, shared by every capture in the process. Once a process had made enough
    streams -- a warm runner's second run -- the frame streamer's new copy stream could BE the stream a preload
    thread was capturing on: its uploads went into that capture and its event sync failed ("operation not
    permitted on an event last recorded in a capturing stream"); work that nothing synchronized would have gone
    into the other thread's graph unnoticed. A stream from outside the pool is never handed out to anyone else,
    and one per thread keeps two threads' captures apart as well. Falls back to a pooled stream when the runtime
    cannot be reached."""
    import torch
    dev = torch.device(device) if device is not None else torch.device("cuda")
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (threading.get_ident(), int(index))
    with _CAPTURE_STREAMS_LOCK:
        stream = _CAPTURE_STREAMS.get(key)
        if stream is None:
            stream = _runtime_stream(int(index))
            _CAPTURE_STREAMS[key] = stream
    return stream


def _runtime_stream(index: int):
    """A new non-blocking stream on device ``index`` from the CUDA runtime, as a torch ExternalStream (kept for
    the life of the process); a pooled torch stream if the runtime is not reachable."""
    import torch
    lib = _cudart()
    if lib is not None:
        try:
            import ctypes
            create = lib.cudaStreamCreateWithFlags
            create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
            create.restype = ctypes.c_int
            handle = ctypes.c_void_p()
            with torch.cuda.device(index):
                rc = create(ctypes.byref(handle), ctypes.c_uint(1))          # 1 = cudaStreamNonBlocking
            if rc == 0 and handle.value:
                return torch.cuda.ExternalStream(handle.value, device=torch.device("cuda", index))
            lib.cudaGetLastError()
        except Exception:
            pass
    return torch.cuda.Stream(device=index)


class capture_graph:
    """``torch.cuda.graph`` without the work its ``__enter__`` does before capturing (torch 2.11): a device-wide
    ``torch.cuda.synchronize()``, ``empty_cache()`` and the pinned-host cache flush. The capture records the same
    kernels with the same launch parameters, so the replays compute the same bits. What goes: waiting for every
    other stream's queued work -- a preload capturing on a background thread waited for the registration pass's
    whole queue (up to 24 batches) with GRAPH_LOCK held -- and handing every cached device and pinned-host block
    back to the driver, which the next allocations then had to cudaMalloc / cudaHostAlloc again. Same arguments
    as ``torch.cuda.graph``. Captures whose private pool may need room on a nearly full device (beside the
    registration's device tail) keep ``torch.cuda.graph``: its empty_cache returns the cached blocks first."""

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode: str = "global"):
        import torch
        if stream is None:
            stream = capture_stream()               # not torch's pooled default capture stream (see capture_stream)
        self._inner = torch.cuda.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode)

    def __enter__(self):
        g = self._inner
        try:
            ctx, cg, pool, mode = g.stream_ctx, g.cuda_graph, g.pool, g.capture_error_mode
        except AttributeError:          # a torch whose graph keeps other attributes: its own enter
            return g.__enter__()
        ctx.__enter__()
        try:
            cg.capture_begin(*pool, capture_error_mode=mode)
        except BaseException:
            ctx.__exit__(None, None, None)
            raise
        return None

    def __exit__(self, *args):
        return self._inner.__exit__(*args)


def note(**facts) -> None:
    """Record session facts for hooks (``SESSION.update``)."""
    SESSION.update({k: v for k, v in facts.items() if v is not None})


class DeviceHold:
    """Device tensors kept referenced until a CUDA event recorded after their last use on another
    stream has completed -- used instead of ``Tensor.record_stream`` wherever another thread may be
    capturing a CUDA graph. The caching allocator counts a stream use recorded (and a block freed)
    while ANY thread of the process is capturing as part of that capture and drops it when the
    capture ends, so the block was handed out again before the other stream had read it: a re-warp
    output block still queued for its device->host copy came back holding the next batch's float32
    frames while the runner's SETSUNA preload captured its graphs. Holding the reference involves
    no allocator bookkeeping. ``add`` also releases what has completed; ``drain`` waits for the rest
    (and runs when the holder is collected, so nothing is let go early)."""

    def __init__(self) -> None:
        self._q: "collections.deque" = collections.deque()

    def add(self, ev, *tensors) -> None:
        self._q.append((ev, tensors))
        self.release_done()

    def release_done(self) -> None:
        q = self._q
        while q and q[0][0].query():       # events of one stream complete in order
            q.popleft()

    def drain(self) -> None:
        q = self._q
        while q:
            q[0][0].synchronize()
            q.popleft()

    def __del__(self) -> None:
        try:
            self.drain()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------
# Pinned host memory for the in-RAM movies. cudaHostRegister on the decoded TIFF blocks (4 ms per
# GB measured on Windows) turns every later host<->device copy of those bytes into a DMA (16-17
# GB/s here) instead of a memcpy into a staging slot plus the copy (or a 4 GB/s pageable copy):
# the registration passes read the raw movie without reader threads, the re-warp writes its
# output straight into the RAM copy, and the extraction streams the published movie the same
# way. JSCOPE_PIN_RAW=0 disables; a failed registration leaves the block pageable and the callers
# keep their previous paths.
#
# Registered memory is never freed while it is registered: the registry holds each owning array
# until drain_unpinned finds nothing else referencing it, unregisters the range and only then lets
# the array go. Until 2026-09-26 a finalizer on the owner only noted the range for a later drain
# while numpy freed the memory at once, and CUDA went on treating those addresses as page-locked:
# host allocations placed over them (a dropped 100 MB temporary was enough) failed their copies
# ("invalid argument", "out of memory") or faulted unrelated kernels ("illegal instruction",
# "invalid program counter"; nvlddmkm events 13/153 in the System log), and unregistering after
# the free did not repair it. Holding the array costs no memory: its pages stayed locked, out of
# the free pool, until the unregistration either way.
# ---------------------------------------------------------------------------------------
_PIN_ENABLED = os.environ.get("JSCOPE_PIN_RAW", "1").strip().lower() not in ("0", "false", "no", "off")
_PIN_LOCK = threading.Lock()
_PIN_RANGES: dict = {}          # owner data pointer -> _Pinned (registered with CUDA; the owner is held)
_CUDART: dict = {"lib": None, "tried": False}
_HOST_NOT_REGISTERED = 713      # cudaErrorHostMemoryNotRegistered


def _cudart():
    """The CUDA runtime torch ships (host registration is not exposed by torch itself)."""
    if _CUDART["tried"]:
        return _CUDART["lib"]
    _CUDART["tried"] = True
    try:
        import ctypes
        import glob
        if torch is None or not torch.cuda.is_available():
            return None
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        cands = glob.glob(os.path.join(lib_dir, "cudart64_*.dll")) + glob.glob(os.path.join(lib_dir, "libcudart.so*"))
        if not cands:
            return None
        lib = ctypes.CDLL(cands[0])
        lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        lib.cudaHostRegister.restype = ctypes.c_int
        lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        lib.cudaHostUnregister.restype = ctypes.c_int
        lib.cudaGetLastError.restype = ctypes.c_int
        _CUDART["lib"] = lib
    except Exception:
        _CUDART["lib"] = None
    return _CUDART["lib"]


def _owner(arr):
    """The ndarray that owns the memory behind ``arr`` (views resolve through their bases)."""
    o = arr
    while isinstance(o, np.ndarray) and not o.flags["OWNDATA"] and isinstance(o.base, np.ndarray):
        o = o.base
    return o


class _Pinned:
    """A registry entry: the array that owns a registered range, held until the range is unregistered."""
    __slots__ = ("owner",)

    def __init__(self, owner) -> None:
        self.owner = owner


def _owner_refs(entry: _Pinned) -> int:
    return sys.getrefcount(entry.owner)


# What _owner_refs reads for an array that only its registry entry references (measured the same way,
# so the interpreter's own temporary references cancel out).
_UNUSED_REFS = _owner_refs(_Pinned(np.empty(1)))


def _capturing() -> bool:
    """This thread's current stream is capturing a CUDA graph (or its state cannot be read). A
    cudaHostRegister/Unregister on the capturing thread invalidates the capture -- seen when the
    garbage collector unregistered: the segmentation's tile graph and SETSUNA's ensemble graph
    were both lost."""
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return True


def drain_unpinned() -> int:
    """Unregister the ranges whose arrays nothing but the registry references any more, then let the
    arrays go: their memory is freed after its unregistration. Measured here, a cudaHostUnregister
    waits until the device has finished ALL queued work (a copy still reading the range included) and
    holds up other threads' launches meanwhile, as it always did. Called before a new registration,
    at every stage start of the in-process runner, when a movie is released and when a recycler
    closes -- never from the garbage collector, and a no-op while this thread captures a CUDA graph.
    An array kept only by a reference cycle waits for the cyclic collector. Returns the number of
    ranges released."""
    lib = _CUDART["lib"]
    if not _PIN_RANGES or lib is None or _capturing():
        return 0
    with _PIN_LOCK:
        idle = [ptr for ptr, e in _PIN_RANGES.items() if _owner_refs(e) <= _UNUSED_REFS]
        unused = [(ptr, _PIN_RANGES.pop(ptr)) for ptr in idle]     # (no loop variable left holding one)
    if not unused:
        return 0
    import ctypes
    n, kept = len(unused), []
    while unused:
        ptr, e = unused.pop()
        try:
            rc = lib.cudaHostUnregister(ctypes.c_void_p(ptr))
        except Exception:
            rc = -1
        if rc != 0:
            try:
                lib.cudaGetLastError()
            except Exception:
                pass
            if rc != _HOST_NOT_REGISTERED:
                kept.append((ptr, e))   # may still be registered: the memory stays until a later drain
        del e                           # freed one by one: a free holds the GIL (~26 ms per GB here)
    if kept:
        with _PIN_LOCK:
            _PIN_RANGES.update(kept)
    return n - len(kept)


def pin_host_array(arr) -> bool:
    """Register the memory behind ``arr`` with CUDA (page-locked; copies become DMAs). Returns
    True when the array is pinned afterwards. No-op without CUDA, with JSCOPE_PIN_RAW=0 or while this
    thread captures a CUDA graph. The array owning the memory is held until :func:`drain_unpinned`
    releases it, so dropping ``arr`` -- a temporary included -- never frees registered memory."""
    if not _PIN_ENABLED or torch is None:
        return False
    owner = _owner(arr)
    if not isinstance(owner, np.ndarray) or not owner.flags["C_CONTIGUOUS"] or owner.nbytes == 0:
        return False
    ptr = int(owner.ctypes.data)
    if ptr in _PIN_RANGES:
        return True
    lib = _cudart()
    if lib is None or _capturing():
        return False
    import ctypes
    drain_unpinned()                   # arrays nothing uses any more: released before more memory is locked
    with _PIN_LOCK:
        if ptr in _PIN_RANGES:
            return True
        rc = lib.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(int(owner.nbytes)), 0)
        if rc != 0:
            try:
                lib.cudaGetLastError()
            except Exception:
                pass
            return False
        _PIN_RANGES[ptr] = _Pinned(owner)
    return True


_PINNED_PREALLOC: dict = {}
# prealloc_pinned registers touched pageable memory instead of allocating pinned memory (JSCOPE_PREALLOC_REGISTER=0)
_PREALLOC_REGISTER = os.environ.get("JSCOPE_PREALLOC_REGISTER", "1").strip().lower() not in ("0", "false", "no", "off")


def prealloc_pinned(key: str, nbytes: int) -> None:
    """Start allocating ``nbytes`` of page-locked host memory on a worker thread, for a later
    ``take_pinned(key, n)``. A pinned allocation holds the CUDA driver while it pins (~0.25 s per GB), so
    the launching thread's calls wait meanwhile: start it where that thread is not launching (registration
    starts the binned stack's buffer under its CPU-only slow-affine fit). A second call for a key still
    pending is a no-op."""
    if torch is None or int(nbytes) <= 0 or key in _PINNED_PREALLOC:
        return
    try:
        if not torch.cuda.is_available():
            return
    except Exception:
        return

    def _alloc(n=int(nbytes)):
        try:
            if _PREALLOC_REGISTER:
                # pages first touched here (no driver held), then registered: cudaHostRegister of resident
                # memory holds the driver ~2.5 ms per GB, cudaHostAlloc ~0.25 s per GB (it faults and pins)
                buf = np.empty(n, dtype=np.uint8)
                buf.fill(0)
                if pin_host_array(buf):
                    return torch.from_numpy(buf)
            return torch.empty(n, dtype=torch.uint8, pin_memory=True)
        except Exception:
            return None

    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jscope_pinned_prealloc")
    _PINNED_PREALLOC[key] = (int(nbytes), ex.submit(_alloc))
    ex.shutdown(wait=False)


def take_pinned(key: str, nbytes: int):
    """The future of a ``prealloc_pinned(key)`` buffer (a uint8 tensor, None on failure) of at least
    ``nbytes`` bytes, removed from the table; None when there is none or it is too small."""
    item = _PINNED_PREALLOC.pop(key, None)
    if item is None or item[0] < int(nbytes):
        return None
    return item[1]


def is_pinned_array(arr) -> bool:
    """True when ``arr`` lies in memory registered by :func:`pin_host_array`."""
    if not _PIN_RANGES:
        return False
    owner = _owner(arr)
    return isinstance(owner, np.ndarray) and int(owner.ctypes.data) in _PIN_RANGES


def all_blocks_pinned(chunks) -> bool:
    """Every live block of a :class:`FrameChunks` is registered (released blocks are ignored)."""
    blocks = getattr(chunks, "blocks", None)
    if not isinstance(blocks, list):
        return False
    live = [b for b in blocks if b is not None]
    if isinstance(chunks, GrowingFrameChunks) and chunks.expect_pinned:
        if chunks.pin_on_read:
            return True          # the pinned streamer registers each block before its first copy
        return all(is_pinned_array(b) for b in live)       # blocks still to come arrive pinned
    return bool(live) and all(is_pinned_array(b) for b in live)


def _norm_path(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def publish_movie(path, frames, *, binned: "dict | None" = None, file_ready=None) -> None:
    """Register the in-RAM registered movie ``frames`` (FrameChunks or ndarray) for ``path``.
    ``binned`` maps a temporal bin size to the fused binned-stack result for that size."""
    _PUBLISHED[_norm_path(path)] = {"frames": frames, "binned": dict(binned or {})}
    _PUBLISHED_READY[_norm_path(path)] = file_ready



def published_movie(path):
    """The in-RAM movie published for ``path`` (FrameChunks/ndarray) or None."""
    entry = _PUBLISHED.get(_norm_path(path))
    return None if entry is None else entry["frames"]


def published_binned(path, bin_size: int):
    """The fused binned-stack result published for ``path`` at ``bin_size``, or None."""
    entry = _PUBLISHED.get(_norm_path(path))
    return None if entry is None else entry["binned"].get(int(bin_size))


_PUBLISHED_READY: dict = {}


def wait_for_file(path) -> bool:
    """Block until the binary at ``path`` is complete on disk when a published in-RAM movie is
    still being written for it (the publisher's ``file_ready`` callable, e.g. the writer join);
    a no-op otherwise. For stages that must map the FILE (the activity segmentation reshapes
    and materialises the whole stack). Returns True when it had to wait."""
    fn = _PUBLISHED_READY.get(_norm_path(path))
    if fn is None:
        return False
    fn()
    return True


def unpublish_movie(path) -> None:
    _PUBLISHED.pop(_norm_path(path), None)
    _PUBLISHED_READY.pop(_norm_path(path), None)
    drain_unpinned()



def open_movie(path, dtype, shape, mode: str = "r"):
    """The published in-RAM movie for ``path`` when there is one, else a numpy memmap."""
    frames = published_movie(path)
    if frames is not None:
        return frames
    return np.memmap(str(path), dtype=dtype, mode=mode, shape=tuple(int(v) for v in shape))


_HOOKS: dict = {}


def on(event: str, fn) -> None:
    """Register a one-shot callback for ``event`` (e.g. ``"extraction_done"``, fired by the
    extraction step between fluorescence extraction and spike inference)."""
    _HOOKS.setdefault(str(event), []).append(fn)


def fire(event: str) -> int:
    fns = _HOOKS.pop(str(event), [])
    for fn in fns:
        fn()
    return len(fns)


def off(event: str, fn=None) -> int:
    """Drop callbacks registered with ``on`` that have not fired: ``fn`` only, or all of ``event``'s.
    Returns how many were dropped. A run drops its own when it is over: a hook whose event that run
    never fired (a pass its settings do not run) would otherwise fire in the NEXT run -- the next
    session of a batch -- with the previous run's arguments."""
    fns = _HOOKS.get(str(event))
    if not fns:
        return 0
    if fn is None:
        n = len(fns)
        fns.clear()
        return n
    try:
        fns.remove(fn)
    except ValueError:
        return 0
    return 1


def defer(job) -> None:
    """Queue a callable to run when the in-process runner has finished the background
    writes (``run_deferred``). Outside the runner, callers must not rely on it."""
    _DEFERRED.append(job)


def run_deferred() -> int:
    n = 0
    while _DEFERRED:
        job = _DEFERRED.pop(0)
        job()
        n += 1
    return n


class BlockRecycler:
    """Hands out destination arrays for the in-RAM copy of a movie without page-faulting on
    the copier's critical path.

    Windows faults freshly allocated memory in at ~1-2 GB/s on first touch, which made a
    copier thread the pacing item of the registration re-warp. Instead the destinations are
    carved out of buffers that are already resident: the raw movie's blocks as they are
    released during its last pass (``FrameChunks.release_upto(..., sink=recycler)``), and,
    when no such buffer is available, spare blocks a helper thread allocates and touches
    ahead of time. ``copy_in(src)`` returns a resident copy of ``src``.
    """

    def __init__(self, block_shape=None, dtype=np.int16, prefault_threads: int = 0, spare_depth: int = 8,
                 pin: bool = False):
        self.pin = bool(pin)               # register fresh blocks too (carved ones inherit the raw's)
        self._lock = threading.Lock()
        self._pool: list = []            # resident buffers (flat views) to carve from
        self._cur = None
        self._off = 0
        self.dtype = np.dtype(dtype)
        self.block_shape = tuple(int(v) for v in block_shape) if block_shape is not None else None
        self._spares: "queue.Queue" = queue.Queue(maxsize=max(1, int(spare_depth)))
        self._stop = threading.Event()
        self._threads = []
        self.n_reused = 0
        self.n_fresh = 0
        if prefault_threads > 0 and self.block_shape is not None:
            for i in range(int(prefault_threads)):
                t = threading.Thread(target=self._prefault, name=f"jupiterscope_prefault_{i}", daemon=True)
                t.start()
                self._threads.append(t)

    def _prefault(self) -> None:
        while not self._stop.is_set():
            arr = np.empty(self.block_shape, dtype=self.dtype)
            arr.fill(0)                              # touch every page off the critical path
            while not self._stop.is_set():
                try:
                    self._spares.put(arr, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def give(self, buffer: np.ndarray) -> None:
        """Offer a resident buffer (any shape, same dtype) for reuse. A frames view (a TIFF read in place)
        offers the whole file buffer it lies in."""
        if buffer is None:
            return
        if not buffer.flags["C_CONTIGUOUS"] and frames_view_ok(buffer):
            owner = _owner(buffer)
            if not isinstance(owner, np.ndarray) or not owner.flags["C_CONTIGUOUS"] \
                    or owner.nbytes % self.dtype.itemsize:
                return
            buffer = owner.reshape(-1).view(np.uint8).view(self.dtype)
        if buffer.dtype != self.dtype or not buffer.flags["C_CONTIGUOUS"]:
            return
        with self._lock:
            self._pool.append(buffer.reshape(-1))

    def take(self, shape, phase=None) -> np.ndarray:
        """A resident block of ``shape``. With ``phase`` (bytes) its address is placed congruent to ``phase``
        modulo the disk sector -- pass the file offset its first byte will have, and an unbuffered writer
        DMAs the block's aligned interior without a staging copy (see _DirectWriter.write)."""
        shape = tuple(int(v) for v in shape)
        n = int(np.prod(shape))
        isz = self.dtype.itemsize
        with self._lock:
            while True:
                if self._cur is not None:
                    skip = 0
                    if phase is not None:
                        gap = (int(phase) - (int(self._cur.ctypes.data) + self._off * isz)) % _SECTOR
                        skip = gap // isz if gap % isz == 0 else 0
                    if self._off + skip + n <= self._cur.shape[0]:
                        out = self._cur[self._off + skip:self._off + skip + n].reshape(shape)
                        self._off += skip + n
                        self.n_reused += 1
                        return out
                if self._pool:
                    self._cur = self._pool.pop(0)
                    self._off = 0
                    continue
                break
        if self.block_shape is not None and shape == self.block_shape and phase is None:
            try:
                arr = self._spares.get_nowait()
                self.n_fresh += 1
                return arr
            except queue.Empty:
                pass
        self.n_fresh += 1
        if phase is not None and int(phase) % isz == 0:
            raw = np.empty(n * isz + _SECTOR, dtype=np.uint8)
            off = (int(phase) - int(raw.ctypes.data)) % _SECTOR
            arr = raw[off:off + n * isz].view(self.dtype).reshape(shape)
        else:
            arr = np.empty(shape, dtype=self.dtype)
        if self.pin:
            arr.fill(0)                    # resident before it is registered
            pin_host_array(arr)
        return arr

    def copy_in(self, src) -> np.ndarray:
        # One memcpy thread is enough: a 100-frame batch copies in ~8 ms against a ~9 ms
        # re-warp batch, and splitting it over four threads only contended with the DMA traffic
        # (measured 2.9 s vs 2.6 s for the 30k-frame subset re-warp).
        dst = self.take(src.shape)
        np.copyto(dst, src)
        return dst

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        with self._lock:
            self._pool.clear()
            self._cur = None
        drain_unpinned()               # buffers nothing was carved from die here


def _is_path(source) -> bool:
    return isinstance(source, (str, bytes, os.PathLike))


def stream_frames(path, shape, ranges: Sequence[tuple[int, int]], device=None, *,
                  n_readers: int = 2, depth: int = 3, offset: int = 0,
                  dtype=np.int16, recycle: bool = False) -> Iterator[tuple[int, int, "torch.Tensor"]]:
    """Yield ``(start, stop, tensor)`` for each requested frame range, in order.

    ``path`` is a file (raw int16 frames at ``offset``) or an in-memory source -- a
    :class:`FrameChunks` or a ``(T, H, W)`` ndarray -- in which case the reader threads
    memcpy into the pinned slots instead of reading the file.

    ``tensor`` is on ``device`` (a fresh device tensor per range; the allocator recycles
    them) or, with ``device=None``, a view of the pinned host slot for that range -- valid
    until the next item is requested.

    ``depth`` pinned host slots circulate through a free-slot queue: a reader thread takes
    a slot, fills it with one range, and hands the slot index back through its future; the
    consumer enqueues the host->device copy, records an event on it and returns the slot
    to the queue at once (the next reader waits on that event before overwriting the
    slot). At most ``depth - 1`` ranges are in flight, so a slot is always available and
    the readers never starve the consumer.

    ``recycle``: the consumer is done with a device tensor -- on the host and on the device --
    once it requests the next range. The tensors then come from the copy stream's pool and
    go back to it (a DeviceHold) once the consumer's stream has passed an event recorded at
    that request, so a copy never waits on the consumer's kernels; otherwise each tensor is
    allocated on the consumer's stream and its copy waits for that stream to pass the
    allocation (safe for any consumer; the copy-bound re-warp lost 0.35 s to those waits).
    """
    if torch is None:
        raise RuntimeError("stream_frames needs torch")
    n_frames, ly, lx = (int(v) for v in shape)
    ranges = [(int(a), int(b)) for a, b in ranges]
    if not ranges:
        return
    if isinstance(path, DerivedFrames):
        # computed from another movie as they are read: nothing stored to copy
        yield from path.device_ranges(ranges, device, recycle=recycle)
        return
    mem_src = None if _is_path(path) else path
    if (device is not None and torch.device(device).type == "cuda" and isinstance(mem_src, FrameChunks)
            and all_blocks_pinned(mem_src)
            and os.environ.get("JSCOPE_STREAM_DIRECT", "1").strip().lower() not in ("0", "false", "no", "off")):
        yield from _stream_frames_pinned(mem_src, ranges, device, dtype, recycle=recycle)
        return
    itemsize = np.dtype(dtype).itemsize
    frame_bytes = ly * lx * itemsize
    max_len = max(b - a for a, b in ranges)
    # JSCOPE_STREAM_READERS: reader threads (each memcpy's or reads one range into a pinned slot);
    # the slot depth follows so that ``n_readers`` ranges can be in flight. The consumer sees
    # the same ranges in the same order whatever the count.
    _env_readers = os.environ.get("JSCOPE_STREAM_READERS", "").strip()
    if _env_readers.isdigit() and int(_env_readers) > 0:
        n_readers = int(_env_readers)
        depth = max(int(depth), n_readers + 1)
    depth = max(2, int(depth))
    n_readers = max(1, min(int(n_readers), depth - 1))
    pin = device is not None and torch.device(device).type == "cuda"
    # The host->device copy of range k+1 goes on its own CUDA stream, so it runs on the copy
    # engine while the consumer's kernels for range k are still executing on the compute
    # stream (on one stream the copy would queue behind them: ~2 ms of a ~8 ms batch at
    # 100 frames of 244x866). The compute stream waits on the copy's event before the tensor
    # is handed out (where the tensor itself comes from: see ``recycle``). Same bytes, same
    # order. JSCOPE_STREAM_COPY_STREAM=0 restores the single-stream copy.
    copy_stream = None
    if pin and os.environ.get("JSCOPE_STREAM_COPY_STREAM", "1").strip().lower() not in ("0", "false", "no", "off"):
        copy_stream = torch.cuda.Stream(device=torch.device(device))
    slots = [torch.empty((max_len, ly, lx), dtype=getattr(torch, np.dtype(dtype).name),
                         pin_memory=pin) for _ in range(depth)]
    views = [memoryview(s.numpy()).cast("B") for s in slots]
    free_events: list["torch.cuda.Event | None"] = [None] * depth
    free_q: "queue.Queue[int]" = queue.Queue()
    for i in range(depth):
        free_q.put(i)
    _STOP = -1

    _dbg = bool(int(os.environ.get("JSCOPE_FRAME_STREAM_DEBUG", "0") or 0))
    _st = {"slot_wait": 0.0, "event_wait": 0.0, "read": 0.0, "bytes": 0, "jobs": 0, "read_max": 0.0}
    _t_start = time.perf_counter()

    def _read(job: int) -> int:
        t0 = time.perf_counter()
        slot = free_q.get()                        # blocks until the consumer returns one
        if slot == _STOP:
            return _STOP
        t1 = time.perf_counter()
        ev = free_events[slot]
        if ev is not None:
            ev.synchronize()                       # the slot's last host->device copy is done
        t2 = time.perf_counter()
        a, b = ranges[job]
        nbytes = (b - a) * frame_bytes
        if mem_src is not None:
            dst = slots[slot][: b - a].numpy()
            if isinstance(mem_src, FrameChunks):
                mem_src.copy_range_into(a, b, dst)
            else:
                np.copyto(dst, mem_src[a:b])
        else:
            with open(path, "rb", buffering=0) as fh:
                fh.seek(int(offset) + a * frame_bytes)
                mv = views[slot][:nbytes]
                got = 0
                while got < nbytes:
                    k = fh.readinto(mv[got:])
                    if not k:
                        raise IOError(f"short read at frame {a} of {path}")
                    got += k
        if _dbg:
            t3 = time.perf_counter()
            _st["slot_wait"] += t1 - t0; _st["event_wait"] += t2 - t1; _st["read"] += t3 - t2
            _st["read_max"] = max(_st["read_max"], t3 - t2); _st["bytes"] += nbytes; _st["jobs"] += 1
        return slot

    tdtype = getattr(torch, np.dtype(dtype).name)

    def _alloc_on(stream, n):
        with torch.cuda.stream(stream):
            t = torch.empty((n, ly, lx), dtype=tdtype, device=device)
        ev = torch.cuda.Event()
        ev.record(stream)
        return t, ev

    spare = None
    held = DeviceHold() if recycle else None
    prev = None
    pool = ThreadPoolExecutor(max_workers=n_readers, thread_name_prefix="jupiterscope_frames")
    futures: dict[int, object] = {}
    try:
        ahead = min(depth - 1, len(ranges))
        for job in range(ahead):
            futures[job] = pool.submit(_read, job)
        for job, (a, b) in enumerate(ranges):
            if prev is not None:               # recycle: the consumer is back, done with the previous range
                rel = torch.cuda.Event()
                rel.record(torch.cuda.current_stream(torch.device(device)))
                held.add(rel, prev)
                prev = None
            slot = futures.pop(job).result()
            if slot == _STOP:  # only after teardown began; cannot happen mid-iteration
                return
            nxt = job + ahead
            if nxt < len(ranges):
                futures[nxt] = pool.submit(_read, nxt)
            host = slots[slot][: b - a]
            if device is None:
                yield a, b, host
                free_q.put(slot)
                continue
            if copy_stream is not None:
                compute = torch.cuda.current_stream(torch.device(device))
                # recycle: from the copy stream's pool, handed back through `held` (see the docstring).
                # Otherwise the device tensor comes from the consumer's stream, the one it is freed on,
                # and the copy waits until that stream has passed the allocation (every earlier use of a
                # recycled block); it is allocated one range early, so the wait does not queue the copy
                # behind the previous range's kernels. No record_stream: see DeviceHold.
                if held is not None:
                    with torch.cuda.stream(copy_stream):
                        dev_t = torch.empty((b - a, ly, lx), dtype=tdtype, device=device)
                    alloc_ev = None
                else:
                    dev_t, alloc_ev = spare if spare is not None else _alloc_on(compute, b - a)
                    spare = (_alloc_on(compute, ranges[job + 1][1] - ranges[job + 1][0])
                             if job + 1 < len(ranges) else None)
                if alloc_ev is not None:
                    copy_stream.wait_event(alloc_ev)
                with torch.cuda.stream(copy_stream):
                    dev_t.copy_(host, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(copy_stream)
                compute.wait_event(ev)
                free_events[slot] = ev
            else:
                dev_t = host.to(device, non_blocking=pin)
                if pin:
                    ev = torch.cuda.Event()
                    ev.record()
                    free_events[slot] = ev
            free_q.put(slot)
            yield a, b, dev_t
            if held is not None:
                prev = dev_t
            del dev_t
    finally:
        for _ in range(n_readers + len(futures)):
            free_q.put(_STOP)                      # unblock readers parked on the queue
        pool.shutdown(wait=True, cancel_futures=True)
        if held is not None:
            if prev is not None:
                rel = torch.cuda.Event()
                rel.record(torch.cuda.current_stream(torch.device(device)))
                held.add(rel, prev)
                prev = None
            held.drain()                           # nothing goes back to the pool before its consumer is done
        if _dbg and _st["jobs"]:
            wall = time.perf_counter() - _t_start
            print(f"  [frame_stream] {_st['jobs']} reads, {_st['bytes'] / 1e9:.1f} GB in {wall:.1f}s wall "
                  f"({_st['bytes'] / 1e9 / max(wall, 1e-9):.2f} GB/s); per-reader sums: slot wait {_st['slot_wait']:.1f}s, "
                  f"event wait {_st['event_wait']:.1f}s, read {_st['read']:.1f}s "
                  f"(mean {_st['bytes'] / 1e9 / max(_st['read'], 1e-9):.2f} GB/s per read, max {_st['read_max'] * 1e3:.0f} ms); "
                  f"readers={n_readers} depth={depth}", flush=True)


_M2D = {"set": False}


def _h2d_frames(dst, src: np.ndarray, stream) -> None:
    """Host frames ``src`` (C-contiguous, or a pinned frames view) -> the contiguous device tensor ``dst``,
    queued on ``stream``: the plain copy, or ONE pitched DMA (cudaMemcpy2DAsync; 28.6 GB/s like a flat copy)
    for frames at a stride -- torch would first gather such a view into a pageable temporary."""
    if src.flags["C_CONTIGUOUS"]:
        dst.copy_(torch.from_numpy(src), non_blocking=True)
        return
    lib = _cudart()
    if lib is not None and frames_view_ok(src) and is_pinned_array(src) and dst.is_contiguous():
        import ctypes
        if not _M2D["set"]:
            lib.cudaMemcpy2DAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                              ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
            _M2D["set"] = True
        fb = int(src.shape[1]) * int(src.shape[2]) * int(src.itemsize)
        rc = lib.cudaMemcpy2DAsync(ctypes.c_void_p(dst.data_ptr()), fb, ctypes.c_void_p(int(src.ctypes.data)),
                                   int(src.strides[0]), fb, int(src.shape[0]), 1, ctypes.c_void_p(stream.cuda_stream))
        if rc == 0:
            return
        try:
            lib.cudaGetLastError()
        except Exception:
            pass
    with torch.cuda.stream(stream):
        dst.copy_(torch.from_numpy(np.ascontiguousarray(src)))


def _stream_frames_pinned(src: "FrameChunks", ranges, device, dtype,
                          recycle: bool = False) -> Iterator[tuple[int, int, "torch.Tensor"]]:
    """``stream_frames`` for an in-RAM movie whose blocks are all registered: each range is copied
    straight from the blocks to a device tensor on a copy stream (a DMA, no reader threads, no
    staging slots), issued ahead of the consumer. A range is host-synchronised before it is handed
    out, so when the consumer then releases the blocks below it (the re-warp recycles them for its
    output) nothing is still reading them -- with ``recycle`` the copies are not gated on the
    consumer's kernels and the wait returns at once (gated copies made it cap the host's lead: the
    gate pass lost 0.5 s; without the wait the re-warp lost 0.4 s). Same tensors, same order."""
    dev = torch.device(device)
    compute = torch.cuda.current_stream(dev)
    copy_stream = torch.cuda.Stream(device=dev)
    tdtype = getattr(torch, np.dtype(dtype).name)
    fshape = tuple(int(v) for v in src.frame_shape)
    _pin_on_read = bool(getattr(src, "pin_on_read", False))
    _dbg = bool(int(os.environ.get("JSCOPE_FRAME_STREAM_DEBUG", "0") or 0))
    _t_start = time.perf_counter()
    _wait = 0.0

    # Two ranges in flight keep the copy engine queued while the re-warp's output copies share it
    # (JSCOPE_STREAM_AHEAD overrides).
    _env_ahead = os.environ.get("JSCOPE_STREAM_AHEAD", "").strip()
    ahead = int(_env_ahead) if _env_ahead.isdigit() and int(_env_ahead) > 0 else 2
    held = DeviceHold() if recycle else None
    tail = getattr(src, "device_tail", None)
    if tail is not None:
        tdev = tail[1].device
        if tdev.type != dev.type or (dev.index is not None and tdev.index != dev.index) or (
                dev.index is None and tdev.index != torch.cuda.current_device()):
            tail = None

    def issue(j):
        # recycle: from the copy stream's pool, handed back through `held` once the consumer's stream is past
        # the range (see stream_frames). Otherwise a tensor allocated on the consumer's stream, the one it is
        # freed on, copied once that stream has passed the allocation (every earlier use of a recycled block).
        # No record_stream: see DeviceHold.
        a, b = ranges[j]
        if tail is not None and a >= tail[0]:
            return tail[1][a - tail[0]: b - tail[0]], None     # frames kept on the device: a view, no copy
        if held is not None:
            with torch.cuda.stream(copy_stream):
                dev_t = torch.empty((b - a,) + fshape, dtype=tdtype, device=dev)
            alloc_ev = None
        else:
            with torch.cuda.stream(compute):
                dev_t = torch.empty((b - a,) + fshape, dtype=tdtype, device=dev)
            alloc_ev = torch.cuda.Event()
            alloc_ev.record(compute)
        with torch.cuda.stream(copy_stream):
            if alloc_ev is not None:
                copy_stream.wait_event(alloc_ev)
            pos = a
            i = src._block_of(a)
            while pos < b:
                blk = src._block(i)
                if _pin_on_read and not is_pinned_array(blk):
                    # registered here, on the launching thread, not in bursts by the producer (a
                    # registration holds the driver; a burst of them stalled the launches for 10-40 ms);
                    # a block that cannot be registered is still copied, synchronously
                    pin_host_array(blk)
                b0 = int(src.starts[i])
                s0, e0 = pos - b0, min(b - b0, int(blk.shape[0]))
                _h2d_frames(dev_t[pos - a: pos - a + (e0 - s0)], blk[s0:e0], copy_stream)
                pos = b0 + e0
                i += 1
            ev = torch.cuda.Event()
            ev.record(copy_stream)
        return dev_t, ev

    pending = [issue(j) for j in range(min(ahead + 1, len(ranges)))]
    prev = None
    try:
        for j, (a, b) in enumerate(ranges):
            if prev is not None:                   # recycle: the consumer is back, done with the previous range
                rel = torch.cuda.Event()
                rel.record(compute)
                held.add(rel, prev)
                prev = None
            dev_t, ev = pending.pop(0)
            nxt = j + ahead + 1
            if nxt < len(ranges):
                pending.append(issue(nxt))
            if ev is not None:
                t0 = time.perf_counter()
                ev.synchronize()               # the source blocks may be released after this
                _wait += time.perf_counter() - t0
                compute.wait_event(ev)
            yield a, b, dev_t
            if held is not None:
                prev = dev_t
            del dev_t
    finally:
        copy_stream.synchronize()              # every copy issued (handed out or not) has read its blocks
        if held is not None:
            if prev is not None:
                rel = torch.cuda.Event()
                rel.record(compute)
                held.add(rel, prev)
                prev = None
            pending = None
            held.drain()                       # nothing goes back to the pool before its consumer is done
        if _dbg:
            n_bytes = sum((b - a) for a, b in ranges) * int(np.prod(fshape)) * np.dtype(dtype).itemsize
            wall = time.perf_counter() - _t_start
            print(f"  [frame_stream] {len(ranges)} pinned DMA ranges, {n_bytes / 1e9:.1f} GB in {wall:.1f}s wall "
                  f"({n_bytes / 1e9 / max(wall, 1e-9):.2f} GB/s); consumer waited {_wait:.1f}s for copies", flush=True)


def read_frames_into(path, offset: int, start: int, count: int, frame_bytes: int, dst) -> int:
    """Read ``count`` frames beginning at frame ``start`` straight into ``dst`` -- a tensor or
    array of at least ``count * frame_bytes`` bytes (typically a pinned buffer slice). One
    direct read, no memmap page faults, no intermediate copy. Returns ``count``."""
    nbytes = int(count) * int(frame_bytes)
    arr = dst.numpy() if hasattr(dst, "numpy") else np.asarray(dst)
    mv = memoryview(arr).cast("B")[:nbytes]
    with open(path, "rb", buffering=0) as fh:
        fh.seek(int(offset) + int(start) * int(frame_bytes))
        got = 0
        while got < nbytes:
            k = fh.readinto(mv[got:])
            if not k:
                raise IOError(f"short read at frame {start} of {path}")
            got += k
    return int(count)


def write_block(fh, block) -> int:
    """Write a contiguous array (or buffer) fully to an unbuffered binary file; returns bytes."""
    mv = memoryview(np.ascontiguousarray(block)).cast("B")
    total = len(mv)
    while len(mv):
        k = fh.write(mv)
        if not k:
            raise IOError("write stalled")
        mv = mv[k:]
    return total


def chunk_ranges(n_frames: int, chunk: int, start: int = 0, stop: "int | None" = None) -> list[tuple[int, int]]:
    """Consecutive (start, stop) ranges of ``chunk`` frames over [start, stop)."""
    stop = int(n_frames if stop is None else stop)
    chunk = max(1, int(chunk))
    return [(s, min(stop, s + chunk)) for s in range(int(start), stop, chunk)]


# Windows: file output through FILE_FLAG_NO_BUFFERING (see _DirectWriter) on local fixed volumes. Buffered, the
# registered binary reached the NVMe at ~1.9-2.3 GB/s (the kernel copies each byte into the page cache, the lazy
# writer flushes it); unbuffered the same volume sustains 6.3 GB/s. JSCOPE_UNBUFFERED_WRITE=0 keeps the buffered
# handle; network shares, removable drives and volumes whose sector does not divide 4096 keep it anyway.
_UNBUFFERED_WRITE = os.name == "nt" and os.environ.get("JSCOPE_UNBUFFERED_WRITE", "1").strip().lower() not in (
    "0", "false", "no", "off")
_SECTOR = 4096


def _direct_write_ok(path) -> bool:
    """True for a path on a local fixed volume whose logical sector size divides _SECTOR."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        drive = os.path.splitdrive(os.path.abspath(str(path)))[0]
        if len(drive) != 2 or drive[1] != ":":
            return False                                   # UNC path
        root = drive + "\\"
        k32 = ctypes.windll.kernel32
        if k32.GetDriveTypeW(ctypes.c_wchar_p(root)) != 3:  # DRIVE_FIXED
            return False
        spc, bps, nfree, ntotal = (ctypes.c_ulong() for _ in range(4))
        if not k32.GetDiskFreeSpaceW(ctypes.c_wchar_p(root), ctypes.byref(spc), ctypes.byref(bps),
                                     ctypes.byref(nfree), ctypes.byref(ntotal)):
            return False
        return 0 < bps.value <= _SECTOR and _SECTOR % bps.value == 0
    except Exception:
        return False


def direct_writer(path) -> "_DirectWriter | None":
    """An unbuffered sequential writer for ``path`` (see _DirectWriter) where unbuffered writes apply (Windows,
    a local fixed volume, JSCOPE_UNBUFFERED_WRITE not 0), else None: the caller writes as before. Buffered
    write() calls of GB-sized outputs are paced by the cache manager at ~2 GB/s on the NVMe volumes here
    (C: and D: alike, measured 2026-10-09) against 5.6-9.5 GB/s unbuffered; a batch session writes ~25 GB
    of them beside its 56 GB binary, and spike inference waited on that queue for 7-16 s per session."""
    if not (_UNBUFFERED_WRITE and _direct_write_ok(path)):
        return None
    try:
        return _DirectWriter(path)
    except OSError:
        return None


def _aligned_bytes(n: int) -> np.ndarray:
    """A uint8 array of ``n`` bytes whose data starts on a 4096-byte boundary."""
    raw = np.empty(int(n) + _SECTOR, dtype=np.uint8)
    off = (-raw.ctypes.data) % _SECTOR
    return raw[off:off + int(n)]


class _DirectWriter:
    """Sequential writer through FILE_FLAG_NO_BUFFERING (Windows). Bytes are staged into 4096-aligned buffers of
    ``stage`` bytes; a disk thread writes each full buffer while the caller fills the next (``n_stage`` in
    rotation). ``write(data, stable=True)`` promises that ``data`` stays unchanged until it is on disk: when its
    address shares the file position's offset within a sector, the aligned interior is written straight from
    it (a reference is held until then) and only the edge bytes are staged -- no memcpy of the bulk, which
    cost the copy-bound registration re-warp 0.5 s in memory traffic. ``close()`` writes the last partial buffer
    zero-padded to a sector and trims the file to the exact byte count. ``write_at(off, data)`` overwrites bytes
    already passed to write(): those still in the current staging buffer in place, the rest by an aligned
    read-modify-write on disk. The file ends up with exactly the bytes of the equivalent plain write() /
    seek + write() calls."""

    _MIN_DIRECT = 1 << 20                 # smaller interiors are staged

    def __init__(self, path, stage: int = 64 << 20, n_stage: int = 3):
        import ctypes
        import ctypes.wintypes as wt
        self._ct = ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wt.HANDLE
        k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
        k32.WriteFile.restype = wt.BOOL
        k32.WriteFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
        k32.ReadFile.restype = wt.BOOL
        k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
        k32.SetFilePointerEx.restype = wt.BOOL
        k32.SetFilePointerEx.argtypes = [wt.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), wt.DWORD]
        k32.CloseHandle.restype = wt.BOOL
        k32.CloseHandle.argtypes = [wt.HANDLE]
        self._k = k32
        self._dw = wt.DWORD
        # GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, CREATE_ALWAYS,
        # FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING
        h = k32.CreateFileW(str(path), 0xC0000000, 0x3, None, 2, 0x20000080, None)
        if h is None or h == wt.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), f"CreateFileW(NO_BUFFERING) failed for {path}")
        self._h = h
        self.path = str(path)
        self._stage = max(_SECTOR, (int(stage) // _SECTOR) * _SECTOR)
        self._bufs = [_aligned_bytes(self._stage) for _ in range(max(2, int(n_stage)))]
        self._free: "queue.Queue[int]" = queue.Queue()
        for i in range(len(self._bufs)):
            self._free.put(i)
        self._q: "queue.Queue" = queue.Queue()
        self._cur = self._free.get()
        self._fill = 0
        self._disk = 0                    # bytes handed to the disk thread; always a multiple of _SECTOR
        self.n_direct = 0                 # bytes written straight from callers' memory
        self._exc: list = []
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="jupiterscope_direct_write", daemon=True)
        self._thread.start()

    # -- low level (disk thread, or the caller once the disk thread is idle) --
    def _seek(self, off: int) -> None:
        if not self._k.SetFilePointerEx(self._h, int(off), None, 0):
            raise OSError(self._ct.get_last_error(), f"SetFilePointerEx({off}) failed on {self.path}")

    def _write_raw(self, buf, n: int) -> None:
        done = self._dw()
        ptr = int(buf) if isinstance(buf, int) else int(buf.ctypes.data)
        off = 0
        while off < n:
            chunk = min(n - off, 1 << 30)
            if not self._k.WriteFile(self._h, ptr + off, chunk, self._ct.byref(done), None) or done.value == 0:
                raise OSError(self._ct.get_last_error(), f"WriteFile failed at +{off} on {self.path}")
            off += int(done.value)

    def _read_raw(self, buf: np.ndarray, n: int) -> None:
        done = self._dw()
        ptr = int(buf.ctypes.data)
        off = 0
        while off < n:
            if not self._k.ReadFile(self._h, ptr + off, n - off, self._ct.byref(done), None) or done.value == 0:
                raise OSError(self._ct.get_last_error(), f"ReadFile failed at +{off} on {self.path}")
            off += int(done.value)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                if item[0] == "direct":           # ("direct", address, bytes, keep-alive reference)
                    try:
                        if not self._exc:
                            self._write_raw(item[1], item[2])
                    except BaseException as exc:
                        self._exc.append(exc)
                    continue
                i, n = item
                try:
                    if not self._exc:
                        self._write_raw(self._bufs[i], n)
                except BaseException as exc:
                    self._exc.append(exc)
                finally:
                    self._free.put(i)
            finally:
                self._q.task_done()

    def _check(self) -> None:
        if self._exc:
            raise self._exc[0]

    # -- API --
    def _stage_in(self, src_ptr: int, n: int) -> None:
        pos = 0
        while pos < n:
            self._check()
            take = min(n - pos, self._stage - self._fill)
            self._ct.memmove(int(self._bufs[self._cur].ctypes.data) + self._fill, src_ptr + pos, take)   # GIL released
            self._fill += take
            pos += take
            if self._fill == self._stage:
                self._q.put((self._cur, self._stage))
                self._disk += self._stage
                self._cur = self._free.get()
                self._fill = 0

    def write(self, data, stable: bool = False) -> int:
        mv = memoryview(data).cast("B")
        n = len(mv)
        src = np.frombuffer(mv, dtype=np.uint8)
        src_ptr = int(src.ctypes.data)
        if stable and n and (src_ptr - (self._disk + self._fill)) % _SECTOR == 0:
            head = min(n, (-(self._disk + self._fill)) % _SECTOR)
            body = (n - head) // _SECTOR * _SECTOR
            if body >= self._MIN_DIRECT:
                self._check()
                self._stage_in(src_ptr, head)             # completes the sector in the staging buffer
                if self._fill:                            # a whole number of sectors now
                    self._q.put((self._cur, self._fill))
                    self._disk += self._fill
                    self._cur = self._free.get()
                    self._fill = 0
                self._q.put(("direct", src_ptr + head, body, (data, mv)))
                self._disk += body
                self.n_direct += body
                self._stage_in(src_ptr + head + body, n - head - body)
                return n
        self._stage_in(src_ptr, n)
        return n

    def write_at(self, off: int, data) -> None:
        mv = memoryview(data).cast("B")
        src = np.frombuffer(mv, dtype=np.uint8)
        off = int(off)
        end = off + len(src)
        if end > self._disk + self._fill or off < 0:
            raise IOError(f"write_at: bytes [{off}, {end}) of {self.path} are not written yet")
        self._q.join()                    # the disk thread idle: the file position is ours
        self._check()
        if end > self._disk:              # the part still in the current staging buffer
            s = max(off, self._disk)
            self._bufs[self._cur][s - self._disk:end - self._disk] = src[s - off:]
            end = s
        if end > off:                     # the part on disk: aligned read-modify-write
            a0 = off - off % _SECTOR
            a1 = end + (-end) % _SECTOR   # <= self._disk (a whole number of sectors)
            tmp = _aligned_bytes(a1 - a0)
            self._seek(a0)
            self._read_raw(tmp, a1 - a0)
            tmp[off - a0:end - a0] = src[:end - off]
            self._seek(a0)
            self._write_raw(tmp, a1 - a0)
            self._seek(self._disk)        # back to the sequential position

    def close(self) -> int:
        """Write what is staged, trim the file to the bytes written, close; returns that count."""
        if self._closed:
            return self._disk + self._fill
        self._closed = True
        total = self._disk + self._fill
        pad = 0
        try:
            if self._fill:
                pad = (-self._fill) % _SECTOR
                if pad:
                    self._bufs[self._cur][self._fill:self._fill + pad] = 0
                self._q.put((self._cur, self._fill + pad))
            self._q.join()
            self._q.put(None)
            self._thread.join()
            self._check()
        finally:
            self._k.CloseHandle(self._h)
        if pad:                           # drop the zero padding of the last sector
            os.truncate(self.path, total)
        return total

    def abort(self) -> None:
        """Error path: stop the disk thread and close the handle (the file content is undefined)."""
        if self._closed:
            return
        self._closed = True
        self._exc.append(IOError("aborted"))
        self._q.put(None)
        self._thread.join()
        self._k.CloseHandle(self._h)


class FrameWriter:
    """Sequential writer for an ``(n_frames, Ly, Lx)`` int16 binary.

    Frame blocks must arrive in order and contiguous; ``write(block)`` takes any object
    exposing the buffer protocol (a numpy array, the ``.numpy()`` view of a pinned tensor).
    ``close()`` verifies the byte count. Measured 2-3x memmap assignment for 54 MB blocks.
    """

    def __init__(self, path, shape, dtype=np.int16):
        # Never pre-size the file: measured on Windows/NTFS, sequential writes into a file
        # extended up front ran at less than half the speed of plain appends (the re-warp
        # went 5.1 -> 12.7 s on the 60-file subset). Frames already written are patched
        # through ``write_at`` on this same handle instead of a memmap.
        self.path = str(path)
        self.n_frames, self.ly, self.lx = (int(v) for v in shape)
        self.frame_bytes = self.ly * self.lx * np.dtype(dtype).itemsize
        self.expected = self.n_frames * self.frame_bytes
        self._dwr = None
        if _UNBUFFERED_WRITE and _direct_write_ok(self.path):
            try:
                self._dwr = _DirectWriter(self.path)
            except Exception:
                self._dwr = None
        self._fh = None if self._dwr is not None else open(self.path, "wb", buffering=0)
        self.written = 0

    def write_at(self, frame: int, block) -> None:
        """Overwrite frames starting at ``frame`` (already written) with ``block``; the
        sequential position is restored afterwards. Same bytes as a memmap patch."""
        mv = memoryview(np.ascontiguousarray(block)).cast("B")
        off = int(frame) * self.frame_bytes
        if off + len(mv) > self.written:
            raise IOError(f"write_at: frames from {frame} are not written yet")
        if self._dwr is not None:
            self._dwr.write_at(off, mv)
            return
        pos = self._fh.tell()
        self._fh.seek(off)
        n = len(mv)
        while n:
            k = self._fh.write(mv)
            if not k:
                raise IOError(f"write_at stalled at byte {off} of {self.path}")
            if k < n:
                mv = mv[k:]
            n -= k
        self._fh.seek(pos)

    def write(self, block, stable: bool = False) -> None:
        """``stable``: the block stays unchanged until the writer is closed (it may be written from in
        place, asynchronously -- the RAM copy's blocks); a reused buffer must not pass it."""
        mv = memoryview(np.ascontiguousarray(block)).cast("B")
        n = len(mv)
        if self._dwr is not None:
            self._dwr.write(mv, stable=stable)
            self.written += n
            return
        while n:
            k = self._fh.write(mv)
            if not k:
                raise IOError(f"write stalled at byte {self.written} of {self.path}")
            if k < n:
                mv = mv[k:]
            n -= k
            self.written += k

    def close(self) -> None:
        if self._dwr is not None:
            dwr, self._dwr = self._dwr, None
            dwr.close()
        elif self._fh is None:
            return
        else:
            self._fh.close()
            self._fh = None
        if self.written != self.expected:
            raise IOError(f"{self.path}: wrote {self.written} bytes, expected {self.expected}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PinnedFrameWriter:
    """Write GPU-resident frame batches to a binary in order without blocking the producer.

    The producer calls ``acquire()`` for a free pinned host slot, copies its batch into
    ``slots[slot][:n]`` with a non-blocking device->host copy, records a CUDA event and
    calls ``submit(slot, n, event)``. A writer thread waits on each event in submission
    order, writes the frames with :class:`FrameWriter` and returns the slot to the free
    queue -- so the producer only ever blocks when all ``n_slots`` batches are queued
    behind the disk. Bytes and order are exactly those of the synchronous
    ``write()``-per-batch loop it replaces. With ``path=None`` nothing is written; the
    slots and events still pace the producer (mean-only passes).
    """

    def __init__(self, path, shape, batch: int, *, n_slots: int = 6, dtype=np.int16, on_batch=None):
        """``on_batch(host_view)`` (optional) is called by a second consumer thread with the
        pinned ``(n, Ly, Lx)`` view of every batch, in order, once its copy has landed -- the
        in-process pipeline uses it to keep a RAM copy of the movie while the file is written.
        When the callback RETURNS its copy (a host ndarray of the same frames) the disk writer
        writes those bytes from that copy and the pinned slot is recycled by the copier alone,
        so the producer is paced by the memcpy, not by the disk (the re-warp of a 158k-frame
        session ran at the disk's 2.2 GB/s and the file kept draining anyway in the background).
        A callback that returns None keeps the old contract: the slot is recycled only after
        both consumers are done with it. Bytes and order are identical either way."""
        if torch is None:
            raise RuntimeError("PinnedFrameWriter needs torch")
        n_slots = max(2, int(n_slots))
        self._fw = FrameWriter(path, shape, dtype) if path is not None else None
        # pinned slots allocated on first acquire(): a writer fed by submit_block (the in-process re-warp,
        # which writes from the RAM copy's blocks) never pins them
        self._slot_shape = (int(batch),) + tuple(int(v) for v in shape[1:])
        self._slot_dtype = getattr(torch, np.dtype(dtype).name)
        self.slots = [None] * n_slots
        self._free: "queue.Queue[int]" = queue.Queue()
        for i in range(n_slots):
            self._free.put(i)
        self._q: "queue.Queue" = queue.Queue()
        self._q2: "queue.Queue | None" = queue.Queue() if on_batch is not None else None
        self._q3: "queue.Queue | None" = queue.Queue() if on_batch is not None else None
        self._on_batch = on_batch
        self._pending: dict = {}
        self._lock = threading.Lock()
        self._exc: list = []
        self._finished = False
        self._dbg = bool(int(os.environ.get("JSCOPE_FRAME_STREAM_DEBUG", "0") or 0))
        self._st = {"acquire": 0.0, "copy": 0.0, "copy_ev": 0.0, "write": 0.0, "write_ev": 0.0, "n": 0, "t0": time.perf_counter()}
        # pause()/resume(): the disk writer waits here before each item (join()/patch() reopen it)
        self._gate = threading.Event()
        self._gate.set()
        self._thread = threading.Thread(target=self._run, name="jupiterscope_frame_writer", daemon=True)
        self._thread.start()
        self._thread2 = None
        if self._q2 is not None:
            self._thread2 = threading.Thread(target=self._run_copier, name="jupiterscope_frame_copier", daemon=True)
            self._thread2.start()
        self._feeders: list = []             # threads still submitting blocks after finish() (join waits)

    def _release(self, slot: int) -> None:
        with self._lock:
            self._pending[slot] -= 1
            done = self._pending[slot] <= 0
        if done:
            self._free.put(slot)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            self._gate.wait()
            if isinstance(item, tuple) and item[0] == "patch":
                _, frame, block, done = item
                try:
                    if not self._exc and self._fw is not None:
                        self._fw.write_at(frame, block)
                except BaseException as exc:
                    self._exc.append(exc)
                done.set()
                continue
            if isinstance(item, tuple) and item[0] == "block":
                # the producer copied this batch straight into a RAM block (pinned); write it once
                # its copy event has fired
                _, event, block = item
                try:
                    if not self._exc:
                        _te = time.perf_counter()
                        event.synchronize()
                        _tw = time.perf_counter()
                        self._st["write_ev"] += _tw - _te
                        if self._fw is not None:
                            self._fw.write(block, stable=True)
                            self._st["write"] += time.perf_counter() - _tw
                except BaseException as exc:
                    self._exc.append(exc)
                self._st["n"] += 1
                continue
            event, slot, n = item
            block = None
            if self._q3 is not None:
                # the copier hands over its RAM copy (or None -> write from the slot)
                block = self._q3.get()
            try:
                if not self._exc:
                    if block is not None:
                        if self._fw is not None:
                            _tw = time.perf_counter()
                            self._fw.write(block)
                            self._st["write"] += time.perf_counter() - _tw
                    else:
                        _te = time.perf_counter()
                        event.synchronize()
                        _tw = time.perf_counter()
                        self._st["write_ev"] += _tw - _te
                        if self._fw is not None:
                            self._fw.write(self.slots[slot][:n].numpy())
                            self._st["write"] += time.perf_counter() - _tw
            except BaseException as exc:  # keep freeing slots so the producer never hangs
                self._exc.append(exc)
            if block is None:
                self._release(slot)       # the slot share of a block-fed batch was released by the copier

    def _run_copier(self) -> None:
        while True:
            item = self._q2.get()
            if item is None:
                return
            event, slot, n = item
            block = None
            try:
                if not self._exc:
                    _te = time.perf_counter()
                    event.synchronize()
                    _tc = time.perf_counter()
                    self._st["copy_ev"] += _tc - _te
                    block = self._on_batch(self.slots[slot][:n])
                    self._st["copy"] += time.perf_counter() - _tc
                    if block is not None and not (isinstance(block, np.ndarray) and block.shape[0] == n):
                        block = None
            except BaseException as exc:
                self._exc.append(exc)
            self._release(slot)           # the copier's share
            if block is not None:
                self._release(slot)       # ... and the writer's: it will write from the block
            self._q3.put(block)

    def _check(self) -> None:
        if self._exc:
            raise self._exc[0]

    def acquire(self) -> int:
        """Block until a pinned slot is free and return its index."""
        _t = time.perf_counter()
        slot = self._free.get()
        self._st["acquire"] += time.perf_counter() - _t
        self._st["n"] += 1
        self._check()
        if self.slots[slot] is None:
            self.slots[slot] = torch.empty(self._slot_shape, dtype=self._slot_dtype, pin_memory=True)
        return slot

    def _report(self) -> None:
        if self._dbg and self._st["n"]:
            st = self._st; wall = time.perf_counter() - st["t0"]
            print(f"  [frame_writer] {st['n']} batches in {wall:.1f}s wall: producer waited for a slot {st['acquire']:.1f}s; "
                  f"copier: event wait {st['copy_ev']:.1f}s + memcpy {st['copy']:.1f}s; "
                  f"disk: event wait {st['write_ev']:.1f}s + write {st['write']:.1f}s", flush=True)

    def submit_block(self, block, event) -> None:
        """Queue a host block the producer filled by a device->host copy itself (a pinned block
        of the RAM copy); it is written once ``event`` (recorded after that copy) fires."""
        self._q.put(("block", event, block))

    def add_feeder(self, thread: threading.Thread, go: "threading.Event | None" = None) -> None:
        """A thread that keeps submitting blocks after the producer's loop (``join`` sets its ``go``
        and waits for it before ending the queue)."""
        self._feeders.append((thread, go))

    def _join_feeders(self) -> None:
        for th, go in self._feeders:
            if go is not None:
                go.set()
            th.join()
        self._feeders = []

    def submit(self, slot: int, n: int, event) -> None:
        """Queue ``slots[slot][:n]`` for writing once ``event`` (recorded after the copy) fires."""
        with self._lock:
            self._pending[int(slot)] = 2 if self._q2 is not None else 1
        self._q.put((event, int(slot), int(n)))
        if self._q2 is not None:
            self._q2.put((event, int(slot), int(n)))

    def patch(self, frame: int, block, wait: bool = True) -> None:
        """Queue an overwrite of frames from ``frame`` with ``block`` (host array). The writer
        thread applies it after every batch queued before it -- so a patch of the leading
        frames issued once they have been submitted always lands after them. Usable after
        ``finish()`` too, as long as ``join()`` has not run. ``wait`` blocks until applied."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("PinnedFrameWriter.patch: writer thread is not running")
        self._gate.set()                     # a paused writer would never apply it
        done = threading.Event()
        self._q.put(("patch", int(frame), np.ascontiguousarray(block), done))
        if wait:
            done.wait()
            self._check()

    def pause(self) -> None:
        """Hold the disk writes (queued items wait; the producer side is unaffected) until resume(),
        patch() or join()."""
        self._gate.clear()

    def resume(self) -> None:
        self._gate.set()

    def finish(self) -> None:
        """Signal that no more batches will come. The copier thread ends once every batch is
        copied; the writer thread keeps draining in the background and still accepts
        ``patch`` requests until ``join()`` (or ``close()``), which waits for it and verifies
        the byte count."""
        if not self._finished:
            self._finished = True
            if self._q2 is not None:
                self._q2.put(None)

    def join(self) -> None:
        self._gate.set()
        self.finish()
        self._join_feeders()
        if self._thread.is_alive():
            self._q.put(None)
        self._thread.join()
        if self._thread2 is not None:
            self._thread2.join()
        self._check()
        if self._fw is not None:
            self._fw.close()
        self._report()

    def close(self) -> None:
        self.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:  # error path: stop the threads, do not verify the byte count
            self._gate.set()
            self.finish()
            self._join_feeders()
            self._q.put(None)
            self._thread.join()
            if self._thread2 is not None:
                self._thread2.join()
            if self._fw is not None:
                try:
                    self._fw.close()
                except Exception:
                    pass

