"""SETSUNA's GPU kernels under gpu_kernel_check's per-session rule: the checks, and prewarm().

Each kernel family is checked once per process and GPU before its first use, against the path it replaces on
representative inputs of the shapes SETSUNA runs (d = 24 features, 2 heads of 12, FFN 48, 15-tap depthwise
convolutions, windows of up to 384 frames with the band half-width the mask mode gives, M = 512 fine bins, three
dual-head models), and a float64 computation of the same function for rule (b):

    setsuna.attn            Triton band attention + its fused LayerNorm/projection kernels (_band_attn_fwd,
                            _ln_proj_fwd, _proj_res_fwd, _proj_res_ffn_fwd) vs the vendored LocalMHABlock (SDPA)
    setsuna.ffn             _ffn_fwd vs the vendored FFNBlock
    setsuna.convblock       _convblock_fwd vs the vendored ConvBlock
    setsuna.head_prologue   _proj_gelu_fwd vs Linear + exact GELU
    setsuna.pooled_head     _head_pool_fwd (Triton) vs the dual heads, gate mix and pooling in torch ops
    setsuna.encoder.L<L>    CudaEncoders (NVRTC, mma.sync: sm_80+) vs the Triton encoder path, per window length
    setsuna.prologue        CudaEncoders' prologue kernel vs its torch ops (bitwise)
    setsuna.head_pool.<v>   CudaHeadPool (NVRTC) vs the Triton head-pool kernel, per pool factor / outputs
    setsuna.head_stream.<v> CudaHeadPool's stream mode vs the plain CUDA head + overlap-add (bitwise)
    setsuna.slab            CudaSlabAdd vs torch.add + div_ (bitwise)
    setsuna.fine_asm.<v>    CudaFineAssembly vs the torch assembly, on the session's first ROI (bitwise)
    setsuna.tail.P<n>       CudaPooledTail vs the torch tail, on the session's first ROI (bitwise)
    decode.pursuit          the timing decoder's pursuit + segment-count kernels vs the torch lockstep loop
    decode.verified         the float32 matched filter + verified pursuit vs the float64 FFT path (same picks)
    evcount.gather          evc_gather vs torch indexing (bitwise)

The torch.compile fusions (_fused: LayerNorm, FFN, prologue, post-head) are inductor's and are probed once per
process by setsuna._compile_available; they are not checked here.
"""
from __future__ import annotations

import copy
import math
import threading
import types

import numpy as np
import torch
import torch.nn.functional as F

from setsuna._support import gpu_kernel_check as gkc

D, FF, KS, M_BINS, W_MAX, N_MODELS = 24, 48, 15, 512, 384, 3
#: (batch windows, window length, band window) of the block checks: the mask mode's full window (half = W // 2) at
#: the longest windows, a shorter window (a lower frame rate), and the corner's fixed 52-frame band
BLOCK_SHAPES = ((4, 384, 384), (3, 200, 200), (2, 96, 52))
_LOCK = threading.Lock()
_SYNTH: dict = {}


def _S():
    from . import engine as setsuna
    return setsuna


def _blocks_module():
    _S()                                   # (puts the vendored modules on sys.path)
    import models_small_conformer_corner as mod
    return mod


def _gen(seed):
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _rand(gen, *shape, scale=1.0, shift=0.0, device=None, dtype=torch.float32):
    return (torch.randn(shape, generator=gen, dtype=torch.float64) * scale + shift).to(device=device, dtype=dtype)


def _jitter_ln(ln, gen):
    """A LayerNorm with non-trivial affine parameters (the trained ones are)."""
    with torch.no_grad():
        ln.weight.copy_(_rand(gen, *ln.weight.shape, scale=0.3, shift=1.0))
        ln.bias.copy_(_rand(gen, *ln.bias.shape, scale=0.2))


# ----------------------------------------------------------------------------------------------------------------------
# float64 references
# ----------------------------------------------------------------------------------------------------------------------
def _attn_block64(blk, x, window):
    """The vendored LocalMHABlock's function in float64 (LayerNorm, packed projection, band softmax attention,
    out-projection, residual) -- its own forward cannot take float64 (the float32 band mask)."""
    mha = blk.attn
    E, H = int(mha.embed_dim), int(mha.num_heads)
    dh = E // H
    h = F.layer_norm(x, (E,), blk.ln.weight.double(), blk.ln.bias.double(), float(blk.ln.eps))
    qkv = F.linear(h, mha.in_proj_weight.double(), mha.in_proj_bias.double())
    B, L, _ = x.shape
    q, k, v = (t.reshape(B, L, H, dh).transpose(1, 2) for t in qkv.split(E, dim=-1))
    s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    if window is not None:
        idx = torch.arange(L, device=x.device)
        band = (idx[None, :] - idx[:, None]).abs() <= int(window) // 2
        s = s.masked_fill(~band, float("-inf"))
    o = (torch.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, L, E)
    ob = mha.out_proj.bias.double() if mha.out_proj.bias is not None else None
    return x + F.linear(o, mha.out_proj.weight.double(), ob)


def vendored64(module):
    """A float64 copy of ``module`` with SETSUNA's session-local surgery removed (the vendored forwards), its
    attention blocks computed by _attn_block64."""
    m = copy.deepcopy(module)
    for sub in m.modules():
        for key in ("forward", "features", "_band_attn", "_jscope_ffn", "_fused_params", "_mask_cache"):
            sub.__dict__.pop(key, None)
        if type(sub).__name__ == "LocalMHABlock":
            sub._mask_cache = {}
            sub.forward = types.MethodType(
                lambda self, x, eff_window=None: _attn_block64(
                    self, x, eff_window if eff_window is not None else self.local_window), sub)
    return m.double()


def _vendored32(module):
    """A float32 copy with the surgery removed (the replaced path)."""
    m = copy.deepcopy(module)
    for sub in m.modules():
        for key in ("forward", "features", "_band_attn", "_jscope_ffn", "_fused_params"):
            sub.__dict__.pop(key, None)
    return m


def _head_ref(H1, WT, BIAS, G, n_models, M, L, dtype):
    """The pooled heads' per-bin values with torch ops in ``dtype``: per model the bulk / corner count and binary
    logits (24 features x head weights + bias), the gate mix (lerp), softplus / sigmoid, summed over the models.
    Returns (count (rows, M), binary (rows, M))."""
    rows = int(H1.shape[1])
    win = torch.arange(rows, device=H1.device) // int(L)
    acc_c = torch.zeros((rows, M), dtype=dtype, device=H1.device)
    acc_b = torch.zeros_like(acc_c)
    h, w, b, g = H1.to(dtype), WT.to(dtype), BIAS.to(dtype), G.to(dtype)
    for i in range(int(n_models)):
        tb, tc = 2 * i, 2 * i + 1
        lbc, lbb = h[tb] @ w[tb, 0] + b[tb, 0], h[tb] @ w[tb, 1] + b[tb, 1]
        lcc, lcb = h[tc] @ w[tc, 0] + b[tc, 0], h[tc] @ w[tc, 1] + b[tc, 1]
        gw = g[i, win][:, None]
        acc_c = acc_c + F.softplus(torch.lerp(lbc, lcc, gw))
        acc_b = acc_b + torch.sigmoid(torch.lerp(lbb, lcb, gw))
    return acc_c, acc_b


def _pooled(acc, kk, sum32=False):
    """Pooled sums of K = kk bins in float64; ``sum32``: the K bins summed in the values' float32 first, then
    widened (the kernels' contract under JSCOPE_SETSUNA_POOL_SUM32, the default)."""
    rows, M = acc.shape
    if sum32 and acc.dtype == torch.float32:
        return acc.view(rows, M // kk, kk).sum(dim=2).to(torch.float64)
    return acc.to(torch.float64).view(rows, M // kk, kk).sum(dim=2)


def _parts(acc, row_off, chunk=None):
    """Per row the sum of its first row_off bins in float64; ``chunk``: float32 partial sums over chunks of that
    many bins first (CudaHeadPool's contract: a warp's 32 bins in float32, the warps' partials in float64)."""
    cols = torch.arange(acc.shape[1], device=acc.device)
    v = torch.where(cols[None, :] < row_off[:, None].long(), acc, torch.zeros((), dtype=acc.dtype, device=acc.device))
    if chunk and acc.dtype == torch.float32:
        return v.view(v.shape[0], -1, int(chunk)).sum(dim=2).to(torch.float64).sum(dim=1)
    return v.to(torch.float64).sum(dim=1)


#: Rule (b)'s slack in these checks (gpu_kernel_check.ERR_SLACK is 1.25): the kernels here differ from the paths they
#: replace by summation order -- dot products of 24-48 terms, pooled sums of 32-64 bins -- so both errors are random
#: walks of the same size, and the larger of a few thousand of them varies by up to ~40 % between two equally
#: accurate orders (measured on the RTX 5090: FFN 1.47x in max error at 1.22x in RMS, pooled sums 1.39x at 1.08x).
#: 1.5 keeps a reorder-level difference from deciding the path; an arithmetic regression is far outside it.
REORDER_SLACK = 1.5


def _cmp(c, got, ref, ref64=None, what=""):
    return c.compare(got, ref, ref64, what=what, slack=REORDER_SLACK)


def _compare2(c, got, bits_ref, err_ref, ref64, what):
    """Bitwise against the replaced kernel (``bits_ref``); otherwise rule (b) with ``err_ref`` -- the replaced
    computation done by torch ops under the summation contract both kernels document (float32 partial sums over a
    fixed number of bins, float64 thereafter) -- as the replaced path's error."""
    from setsuna._support.gpu_kernel_check import _bitwise
    if _bitwise(got, bits_ref):
        return
    c.bitwise = False
    if _cmp(c, got, err_ref, ref64, what=what) == gkc.BITWISE:
        c.bitwise = False


def _synthetic_heads(device, gen, n_models=N_MODELS, M=M_BINS):
    """(WT (2N, 2, 32, M), BIAS (2N, 2, M)) of the shape _head_stack builds, rows 24..31 zero."""
    WT = torch.zeros((2 * n_models, 2, 32, M), dtype=torch.float32)
    WT[:, :, :D, :] = _rand(gen, 2 * n_models, 2, D, M, scale=0.3)
    BIAS = _rand(gen, 2 * n_models, 2, M, scale=0.5, shift=-1.0)
    return WT.to(device).contiguous(), BIAS.to(device).contiguous()


def _synthetic_h1(device, gen, n_models, rows):
    H1 = torch.zeros((2 * n_models, rows, 32), dtype=torch.float32)
    H1[:, :, :D] = _rand(gen, 2 * n_models, rows, D, scale=0.7)
    return H1.to(device).contiguous()


# ----------------------------------------------------------------------------------------------------------------------
# model-free checks (CHECKS)
# ----------------------------------------------------------------------------------------------------------------------
def check_attn(c, device):
    """setsuna.attn: the plain path (x + _BandAttention(ln(x))) and the fused block (and fused block + FFN) against
    the vendored LocalMHABlock (+ FFNBlock) on BLOCK_SHAPES, float64 reference by _attn_block64."""
    S, mod = _S(), _blocks_module()
    gen = _gen(101)
    torch.manual_seed(101)
    blk = mod.LocalMHABlock(D, 2, W=W_MAX, local_window=None).eval()
    ffn = mod.FFNBlock(D, FF).eval()
    _jitter_ln(blk.ln, gen)
    _jitter_ln(ffn.ln, gen)
    blk, ffn = blk.to(device), ffn.to(device)
    b64, f64 = vendored64(blk), vendored64(ffn)
    ba = S._BandAttention(blk.attn, S.attn_precision(device), ln=blk.ln)
    fparams = S._ffn_params(ffn)
    with torch.no_grad():
        for B, L, window in BLOCK_SHAPES:
            half = window // 2
            x = _rand(gen, B, L, D, device=device)
            ref = type(blk).forward(blk, x, eff_window=window)
            r64 = b64(x.double(), eff_window=window)
            _cmp(c, x + ba(blk.ln(x), half), ref, r64, what=f"attention {B}x{L} band {half}")
            if S._FUSED_BLOCKS_ENABLED:
                _cmp(c, ba.fused_block(x, half, blk.ln), ref, r64, what=f"fused block {B}x{L}")
                if fparams is not None:
                    _cmp(c, ba.fused_block(x, half, blk.ln, ffn=fparams), type(ffn).forward(ffn, ref), f64(r64),
                              what=f"fused block + FFN {B}x{L}")


def check_ffn(c, device):
    """setsuna.ffn: _ffn_apply against the vendored FFNBlock."""
    S, mod = _S(), _blocks_module()
    gen = _gen(202)
    torch.manual_seed(202)
    ffn = mod.FFNBlock(D, FF).eval()
    _jitter_ln(ffn.ln, gen)
    ffn = ffn.to(device)
    f64 = vendored64(ffn)
    p = S._ffn_params(ffn)
    with torch.no_grad():
        for B, L, _w in BLOCK_SHAPES:
            x = _rand(gen, B, L, D, device=device)
            _cmp(c, S._ffn_apply(x, p), type(ffn).forward(ffn, x), f64(x.double()), what=f"FFN {B}x{L}")


def check_convblock(c, device):
    """setsuna.convblock: _convblock_apply against the vendored ConvBlock."""
    S, mod = _S(), _blocks_module()
    gen = _gen(303)
    torch.manual_seed(303)
    blk = mod.ConvBlock(D, KS).eval()
    _jitter_ln(blk.ln, gen)
    blk = blk.to(device)
    b64 = vendored64(blk)
    p = S._convblock_params(blk)
    with torch.no_grad():
        for B, L, _w in BLOCK_SHAPES:
            x = _rand(gen, B, L, D, device=device)
            _cmp(c, S._convblock_apply(blk, x, p), type(blk).forward(blk, x), b64(x.double()),
                      what=f"conv block {B}x{L}")


def check_head_prologue(c, device):
    """setsuna.head_prologue: _proj_gelu_fwd (sr_head[0] + exact GELU into a zero-padded 32-wide slab)."""
    S = _S()
    gen = _gen(404)
    torch.manual_seed(404)
    lin = torch.nn.Linear(D, D).to(device)
    w0t, b0 = S._pad_mat_t(lin.weight, 32, 32), S._pad_vec(lin.bias, 32)
    with torch.no_grad():
        for rows in (64 * 6 + 17, 4 * 384):
            h = _rand(gen, rows, D, scale=1.5, device=device)
            out = torch.full((rows, 32), float("nan"), device=device)
            S._proj_gelu_fwd[(S._triton.cdiv(rows, 64),)](h, out, w0t, b0, rows, D=D, DP=32, O=D, OP=32, BL=64,
                                                          num_warps=4)
            ref = F.gelu(F.linear(h, lin.weight, lin.bias))
            r64 = F.gelu(F.linear(h.double(), lin.weight.double(), lin.bias.double()))
            _cmp(c, out[:, :D].contiguous(), ref, r64, what=f"head prologue {rows} rows")
            c.equal(out[:, D:], torch.zeros_like(out[:, D:]), "head prologue padding")


def check_pooled_head(c, device):
    """setsuna.pooled_head: _head_pool_fwd's pooled sums, fine streams and boundary parts against _head_ref's torch
    ops (float32 values, float64 pooled sums) and float64."""
    S = _S()
    gen = _gen(505)
    N, M, L, B = N_MODELS, M_BINS, 128, 4
    rows = B * L
    WT, BIAS = _synthetic_heads(device, gen, N, M)
    H1 = _synthetic_h1(device, gen, N, rows)
    G = torch.rand((N, B), generator=gen).to(device)
    row_off = torch.randint(0, M, (rows,), generator=gen).to(device=device, dtype=torch.int32)
    with torch.no_grad():
        rc, rb = _head_ref(H1, WT, BIAS, G, N, M, L, torch.float32)
        tc, tb = _head_ref(H1, WT, BIAS, G, N, M, L, torch.float64)
        for k, pf, lf, part in ((32, True, False, True), (64, False, False, False), (32, True, True, False)):
            kk = min(k, 64)
            oc = torch.empty((rows, M // kk), dtype=torch.float64, device=device)
            ob = torch.empty_like(oc)
            opf = torch.empty((rows, M), dtype=torch.float32, device=device) if pf else ob
            olf = torch.empty((rows, M), dtype=torch.float32, device=device) if lf else ob
            opart = torch.empty(rows, dtype=torch.float64, device=device) if part else oc
            S._head_pool_triton(H1, WT, BIAS, G, oc, ob, opf, olf, row_off if part else None, opart, rows, L, N, 32,
                                M, k, kk, pf, lf, part)
            tag = f"k{k}"
            _cmp(c, oc, _pooled(rc, kk, S._POOL_SUM32), _pooled(tc, kk), what=f"pooled count {tag}")
            _cmp(c, ob, _pooled(rb, kk, S._POOL_SUM32), _pooled(tb, kk), what=f"pooled binary {tag}")
            if pf:
                _cmp(c, opf, rb, tb, what=f"binary stream {tag}")
            if lf:
                _cmp(c, olf, rc, tc, what=f"count stream {tag}")
            if part:
                _cmp(c, opart, _parts(rc, row_off), _parts(tc, row_off), what=f"boundary parts {tag}")


def check_slab(c, device):
    """setsuna.slab: CudaSlabAdd against torch.add then div_ (and the sum alone for d == 0)."""
    from . import _setsuna_cuda
    gen = _gen(606)
    k = _setsuna_cuda.CudaSlabAdd(device)
    base = _rand(gen, 3 * 8 * 1040, scale=0.3, shift=0.4, device=device)
    a = base[: 8 * 1040].view(8, 1040)[:, :1024]
    b = base[8 * 1040: 16 * 1040].view(8, 1040)[:, 8:1032]
    for d in (6.0, 0.0):
        out = torch.empty((8, 1024), dtype=torch.float32, device=device)
        k.run(a, b, out, d)
        ref = torch.add(a, b)
        r64 = a.double() + b.double()
        if d:
            ref = ref.div_(torch.full((), d, dtype=torch.float32, device=device))   # (a true division, as the
            r64 = r64 / d                                                          # torch path's)
        _cmp(c, out, ref, r64, what=f"slab d={d}")


def _pursuit_case(device, seed):
    """Three synthetic probability segments (float32, narrow bumps on a floor, the last one shorter), their float64
    matched-filter responses by direct correlation on the host (decode_readout's definition), the template, its
    half-width and the picks asked for."""
    gen = _gen(seed)
    L, hw, sig = 6000, 12, 4.0
    lens, counts = [6000, 6000, 4321], [10, 14, 6]          # fewer picks than peaks: no pick on the floor's ties
    t = np.arange(L, dtype=np.float64)
    prob = 2e-4 + 1e-3 * torch.rand((3, L), generator=gen, dtype=torch.float64).numpy()
    for r in range(3):
        for ci in torch.randint(0, lens[r], (25,), generator=gen).tolist():
            amp = 0.05 + 0.9 * float(torch.rand((), generator=gen))
            prob[r] += amp * np.exp(-0.5 * ((t - ci) / 2.5) ** 2)
        prob[r, lens[r]:] = 0.0
    prob = np.minimum(prob, 0.999).astype(np.float32)
    tmpl = np.exp(-0.5 * (np.arange(-hw, hw + 1, dtype=np.float64) / sig) ** 2)
    work = np.zeros((3, L), dtype=np.float64)
    for r in range(3):
        work[r, :lens[r]] = np.correlate(prob[r, :lens[r]].astype(np.float64), tmpl, mode="same")
    return (torch.from_numpy(prob).to(device), torch.from_numpy(work).to(device), lens, counts, hw,
            torch.from_numpy(tmpl).to(device))


def check_decode_pursuit(c, device):
    """decode.pursuit: the pursuit kernel's picks against the torch lockstep loop on the same float64 responses,
    and setsuna_segment_counts against the torch sums (float64 reference: exact sums)."""
    from . import _decode_cuda
    S = _S()
    prob, work, lens, counts, hw, tmpl = _pursuit_case(device, 707)
    got = _decode_cuda.pursuit(work.clone(), lens, counts, hw, tmpl)
    ref = S._greedy_pursuit_batched_cuda(work.clone(), lens, counts, hw, tmpl)
    for j, (g, r) in enumerate(zip(got, ref)):
        c.equal(np.asarray(g, np.int64), np.asarray(r, np.int64), f"pursuit picks, segment {j}")
    gen = _gen(708)
    M = 32
    n_frames, seg_bins = 1000, 3 * M + 7
    lf = (torch.rand(n_frames, generator=gen, dtype=torch.float64) * 3).to(device)
    pt = (lf.cpu() * torch.rand(n_frames, generator=gen, dtype=torch.float64)).to(device)
    got_s = _decode_cuda.segment_counts(lf, pt, M, seg_bins, n_frames * M)
    with gkc.forced("decode.pursuit", False):
        ref_s = S._segment_count_sums(lf, (seg_bins, pt), M, seg_bins, n_frames * M)
    lfh, pth = lf.cpu().numpy(), pt.cpu().numpy()
    exact = []
    for a in range(0, n_frames * M, seg_bins):
        b = min(a + seg_bins, n_frames * M)
        terms = list(lfh[-(-a // M): b // M])
        if a % M:
            terms += [lfh[a // M], -pth[a // M]]
        if b % M:
            terms.append(pth[b // M])
        exact.append(math.fsum(terms))
    _cmp(c, got_s, ref_s, torch.tensor(exact, dtype=torch.float64), what="segment counts")


def check_decode_verified(c, device):
    """decode.verified: the float32 matched filter + verified pursuit (from the tile bounds, and from the computed
    response) against the float64 pursuit on the direct float64 responses -- the same picks."""
    from . import _decode_cuda
    prob, work, lens, counts, hw, tmpl = _pursuit_case(device, 709)
    L = int(prob.shape[1])
    flat = prob.reshape(-1).contiguous()
    seg0 = [r * L for r in range(len(lens))]
    slack = torch.full((len(lens),), 1e-9, dtype=torch.float64, device=device)
    rho = _decode_cuda.bound_rho(hw)
    ref = _S()._greedy_pursuit_batched_cuda(work.clone(), lens, counts, hw, tmpl)
    lazy = _decode_cuda.pursuit_from_bounds(flat, seg0, lens, counts, slack, hw, tmpl, rho=rho)
    resp, mf = _decode_cuda.matched_filter32(flat, seg0, lens, hw, tmpl)
    full = _decode_cuda.pursuit_verified(resp, flat, seg0, lens, counts, slack, hw, tmpl, rho=rho, mf=mf)
    for name, got in (("from bounds", lazy), ("matched filter", full)):
        if got is None or any(g is None for g in got):
            c.fail(f"verified pursuit ({name}) gave up")
            continue
        for j, (g, r) in enumerate(zip(got, ref)):
            c.equal(np.asarray(g, np.int64), np.asarray(r, np.int64), f"verified picks ({name}), segment {j}")


def check_evc_gather(c, device):
    """evcount.gather: evc_gather's windows against torch indexing (zero outside the stream)."""
    from . import _cuda_rt, evcount_fast
    gen = _gen(808)
    M, Ld, off0, Wc = 64, 5000, 37, 300
    pf = torch.rand(Ld, generator=gen).to(device)
    ks = torch.tensor([0, 1, 20, 77, 78], dtype=torch.int64, device=device)
    k = evcount_fast._GATHER.get((torch.device(device).type, torch.device(device).index))
    if k is None:
        k = _cuda_rt.CudaKernel(evcount_fast.GATHER_SRC, "evc_gather", device=torch.device(device))
    out = torch.full((int(ks.numel()), Wc), float("nan"), device=device)
    k.launch(((Wc + 255) // 256, int(ks.numel())), (256,), 0,
             (pf, ks, (M, "q"), (off0, "q"), (Ld, "q"), (Wc, "i"), out))
    idx = ks[:, None] * M - off0 + torch.arange(Wc, device=device)[None, :]
    inside = (idx >= 0) & (idx < Ld)
    ref = torch.where(inside, pf[idx.clamp(0, Ld - 1)], torch.zeros((), device=device))
    c.equal(out, ref, "gathered windows")


#: the checks that need no model, by kernel name (setsuna._gpu_ok)
CHECKS = {
    "setsuna.attn": check_attn,
    "setsuna.ffn": check_ffn,
    "setsuna.convblock": check_convblock,
    "setsuna.head_prologue": check_head_prologue,
    "setsuna.pooled_head": check_pooled_head,
    "setsuna.slab": check_slab,
    "decode.pursuit": check_decode_pursuit,
    "decode.verified": check_decode_verified,
    "evcount.gather": check_evc_gather,
}


# ----------------------------------------------------------------------------------------------------------------------
# checks on an ensemble (the session's own, or prewarm's synthetic one)
# ----------------------------------------------------------------------------------------------------------------------
def check_encoder(c, device, enc, models, xb, ind_t, nu_t, fps_t, ct_t, eff_window):
    """setsuna.encoder.L<L>: CudaEncoders.h1 on four windows of a fixed pseudo-random trace against the Triton
    encoder path (features through the installed kernels + _proj_gelu_fwd) and the float64 vendored branches."""
    S = _S()
    n = min(4, int(xb.shape[0]))
    L = int(xb.shape[1])
    g = torch.Generator(device=xb.device).manual_seed(1234)
    x = torch.randn((n, L), generator=g, device=xb.device, dtype=torch.float32)
    args = (ind_t[:n], nu_t[:n], fps_t[:n], ct_t[:n] if ct_t is not None else None)
    WT, BIAS, Dm, DP = S._head_stack(models)
    P0 = S._head_prologue_params(models, Dm, DP)
    branches = [br for m in models for br in (m.bulk, m.corner)]
    ref = torch.empty((len(branches), n * L, DP), dtype=torch.float32, device=xb.device)
    # the reference forward without a torch.compile probe (eager when none has run): the kernel path never compiles
    S._NO_PROBE.on = True
    try:
        with torch.no_grad():
            for t, br in enumerate(branches):
                h = br.features(x, *args, eff_window).reshape(-1, Dm).contiguous()
                w0t, b0 = P0[t]
                S._proj_gelu_fwd[(S._triton.cdiv(int(h.shape[0]), 64),)](
                    h, ref[t], w0t, b0, int(h.shape[0]), D=Dm, DP=DP, O=Dm, OP=DP, BL=64, num_warps=4)
    finally:
        S._NO_PROBE.on = False
    out = torch.full_like(ref, float("nan"))
    with torch.no_grad():
        enc.h1(x, *args, eff_window, out)
        r64 = torch.zeros((len(branches), n * L, DP), dtype=torch.float64, device=xb.device)
        a64 = tuple(None if a is None else (a.double() if a.is_floating_point() else a) for a in args)
        for t, br in enumerate(branches):
            b64 = vendored64(br)
            h = type(b64).features(b64, x.double(), *a64, eff_window)
            r64[t, :, :Dm] = b64.sr_head[1](b64.sr_head[0](h)).reshape(-1, Dm)
    _cmp(c, out, ref, r64, what=f"H1 windows of {L}")


def check_prologue(c, device, enc):
    """setsuna.prologue: CudaEncoders.h1 with the session constants and the prologue kernel against the same call
    with the prologue's torch ops (bit for bit)."""
    gen = _gen(909)
    B, L = 3, 128
    dev = torch.device(device)
    xb = _rand(gen, B, L, device=dev)
    ind_t = torch.zeros(B, dtype=torch.long, device=dev)
    ct_t = torch.zeros(B, dtype=torch.long, device=dev)
    nu_t = torch.full((B,), 0.37, device=dev)
    fps_t = torch.full((B,), 29.5, device=dev)
    ids = (0, 0, 29.5, "gpu-check")
    outs = []
    with torch.no_grad():
        for on in (True, False):
            H1 = torch.full((enc.nbr, B * L, 32), float("nan"), device=dev)
            with gkc.forced("setsuna.prologue", on):
                enc.h1(xb, ind_t, nu_t, fps_t, ct_t, L, H1, ids=ids)
            outs.append(H1)
    _cmp(c, outs[0], outs[1], what="H1 with the prologue kernel")


def head_pool_name(kk, pf, part) -> str:
    return f"setsuna.head_pool.k{int(kk)}{'.pf' if pf else ''}{'.part' if part else ''}"


def check_head_pool(c, device, hp, WT, BIAS, n_models, M, kk, pf, part, k=None):
    """setsuna.head_pool.<variant>: CudaHeadPool.run (pool factor kk, with the fine binary stream / boundary parts)
    on a synthetic batch against the Triton kernel and _head_ref in float64."""
    S = _S()
    gen = _gen(1010)
    L, B = 128, 4
    rows = B * L
    k = int(k or kk)
    H1 = _synthetic_h1(device, gen, n_models, rows)
    G = torch.rand((n_models, B), generator=gen).to(device)
    row_off = torch.randint(0, M, (rows,), generator=gen).to(device=device, dtype=torch.int32) if part else None
    with torch.no_grad():
        oc = torch.empty((rows, M // kk), dtype=torch.float64, device=device)
        ob = torch.empty_like(oc)
        opf = torch.empty((rows, M), dtype=torch.float32, device=device) if pf else None
        p2 = torch.empty((rows, M // (32 * hp.NW)), dtype=torch.float64, device=device) if part else None
        hp.run(H1, G, oc, ob, rows, L, kk, out_pf=opf, row_off=row_off, out_part=p2)
        tc, tb = torch.empty_like(oc), torch.empty_like(ob)
        tpf = torch.empty((rows, M), dtype=torch.float32, device=device) if pf else tb
        tpart = torch.empty(rows, dtype=torch.float64, device=device) if part else tc
        S._head_pool_triton(H1, WT, BIAS, G, tc, tb, tpf, tb, row_off, tpart, rows, L, n_models, 32, M, k, kk,
                            bool(pf), False, bool(part))
        ec, eb = _head_ref(H1, WT, BIAS, G, n_models, M, L, torch.float64)
        rc, rb = _head_ref(H1, WT, BIAS, G, n_models, M, L, torch.float32)
        _compare2(c, oc, tc, _pooled(rc, kk, True), _pooled(ec, kk), "pooled count")
        _compare2(c, ob, tb, _pooled(rb, kk, True), _pooled(eb, kk), "pooled binary")
        if pf:
            _cmp(c, opf, tpf, eb, what="binary stream")
        if part:
            _compare2(c, p2.sum(dim=1), tpart, _parts(rc, row_off, 32), _parts(ec, row_off), "boundary parts")


def synthetic_ensemble(device):
    """Three dual-head models of the deployed architecture (d = 24, FFN 48, 4 layers, 2 heads, M = 512, windows of
    up to 384 frames) with random weights, SETSUNA's attention / FFN / conv-block surgery installed as a session
    installs it; cached per device."""
    key = str(torch.device(device))
    with _LOCK:
        got = _SYNTH.get(key)
        if got is not None:
            return got
        S = _S()
        torch.manual_seed(4321)
        models = []
        for _ in range(N_MODELS):
            m = S.BinarySRHybrid(S.NUM_INDICATORS, S.NUM_CELL_TYPES, m=M_BINS, head_type="count", W=W_MAX,
                                 bulk_d=D, bulk_n_heads=2, bulk_ffn=FF, bulk_layers=4, corner_d=D, corner_n_heads=2,
                                 corner_ffn=FF, corner_layers=4, n_corner=1, gate_rich=False, dual=True,
                                 bulk_backbone="attn", corner_backbone="attn").eval()
            models.append(m.to(device))
        S._install_triton_attention(models)
        S._install_triton_ffn(models)
        S._install_fused_convblock(models)
        _SYNTH[key] = models
        return models


#: the kernel-name prefixes of the spike-inference kernels (report)
PREFIXES = ("setsuna.", "decode.", "evcount.")


def report() -> dict:
    """gpu_kernel_check.report() of the spike-inference kernels checked in this process (the run metadata's
    'gpu_kernels', with the JSCOPE_GPU_KERNEL_CHECK mode)."""
    out = {}
    for prefix in PREFIXES:
        for name, rec in gkc.report(prefix).items():
            rec.pop("check_s", None)            # (a timing: the metadata stays the same from run to run)
            out[name] = rec
    if out:
        out["mode"] = gkc.mode()
    return out


def prewarm(device="cuda") -> None:
    """Run every SETSUNA check that needs no session data on ``device`` (model-free ones, then the CUDA encoder for
    windows of 384 frames, its prologue and the CUDA head's pool factors 32 / 64 on a synthetic ensemble), so that a
    session finds them cached. The fused assembly and the pooled tail check on the session's first ROI."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available() or not gkc.enabled():
        return
    S = _S()
    if S._triton is None:
        return
    names = ["decode.pursuit", "decode.verified", "evcount.gather"]
    if S._TRITON_ATTN_ENABLED:
        names.append("setsuna.attn")
    if S._FUSED_BLOCKS_ENABLED:
        names.append("setsuna.ffn")
    if S._FUSED_CONV_ENABLED:
        names.append("setsuna.convblock")
    if S._HEAD_PROLOGUE_FUSED:
        names.append("setsuna.head_prologue")
    if S._POOLED_TAIL_ENABLED:
        names.append("setsuna.pooled_head")
    if S._PF_SLAB_ENABLED:
        names.append("setsuna.slab")
    for name in names:
        gkc.verdict(name, dev, CHECKS[name])
    if not (S._CUDA_ENCODER_ENABLED or S._CUDA_HEAD_POOL_ENABLED):
        return
    models = synthetic_ensemble(dev)
    WT, BIAS, Dm, DP = S._head_stack(models)
    if S._CUDA_ENCODER_ENABLED and gkc.cached("setsuna.attn", dev) in (gkc.BITWISE, gkc.CHECKED):
        from . import _setsuna_cuda
        try:
            enc = _setsuna_cuda.CudaEncoders(models, dev)
        except Exception:
            enc = None
        if enc is not None:
            B, L = 4, W_MAX
            ind_t = torch.zeros(B, dtype=torch.long, device=dev)
            ct_t = torch.zeros(B, dtype=torch.long, device=dev)
            nu_t = torch.full((B,), 0.2, device=dev)
            fps_t = torch.full((B,), 30.0, device=dev)
            xb = torch.zeros((B, L), device=dev)
            gkc.verdict(f"setsuna.encoder.L{L}", dev,
                        lambda c, d: check_encoder(c, d, enc, models, xb, ind_t, nu_t, fps_t, ct_t, L))
            if enc.prologue_ok and gkc.cached(f"setsuna.encoder.L{L}", dev) in (gkc.BITWISE, gkc.CHECKED):
                gkc.verdict("setsuna.prologue", dev, lambda c, d: check_prologue(c, d, enc))
    if S._CUDA_HEAD_POOL_ENABLED:
        from . import _setsuna_cuda
        try:
            hp = _setsuna_cuda.CudaHeadPool(WT, BIAS, len(models), M_BINS, dev)
        except Exception:
            hp = None
        if hp is not None:
            for kk, pf, part in ((32, True, False), (64, False, False), (32, True, True)):
                gkc.verdict(head_pool_name(kk, pf, part), dev,
                            lambda c, d, kk=kk, pf=pf, part=part: check_head_pool(c, d, hp, WT, BIAS, len(models),
                                                                                   M_BINS, kk, pf, part))
