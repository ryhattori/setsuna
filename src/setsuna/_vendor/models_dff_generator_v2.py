"""Enhanced conditional dF/F generator (v2): predicts dF/F traces from spike
rate / sub-frame occupancy + indicator kinetics, with several upgrades over
v1 (models_dff_generator.DffGenerator):

  (1) ADAPTIVE W  -- the attention span is fps-scaled per forward. Each
      LocalMHABlock's local window is set to
          clamp(round(t_sec * fps), w_floor, w_ceil)
      (when ``w_mode="adaptive"``) so a SINGLE model covers ~1-120 Hz: at high
      fps a fixed frame-window would shrink the real-time receptive field, so
      we widen the band proportionally to keep ~t_sec of context. The window is
      threaded through the blocks via the ``eff_window`` override that the
      existing LocalMHABlock already supports (same mechanism as our main
      model's adaptive mask).

  (2) SUB-FRAME I/O  -- if ``sub_frame_m > 1`` the input is (B, W*sub_frame_m)
      sub-frame occupancy, reshaped to W positions of m-channel features before
      in_proj. If ``out_m > 1`` the output head emits (B, W*out_m) so the
      generated dF/F carries sub-frame (super-resolution-label) structure.

  (3) CAPACITY  -- d / layers / ffn / n_heads fully configurable.

  (4) backbone="ssm"  -- a simple diagonal state-space (S4-/Mamba-lite) block
      replaces the conformer attention. A real-valued diagonal linear
      recurrence is exactly the calcium-decay inductive bias (h_t = a*h_{t-1}+
      b*x_t), and is computed time-aware (decay per real second via fps).

  (5) backbone="moe"  -- n_experts parallel conformer branches combined by a
      soft mixture gated on (log nu, log fps) -- same flavor as our bulk/corner
      gate.

  (6) latent_dim>0  -- accept an optional z code, projected and added into the
      conditioning vector for stochastic generation.

Keeps the v1 (nu, fps, cell) FiLM + biophysical kin embedding (reusing the same
log_norm normalization, so the biophys_table kin + conditioning are reusable)
and time-aware positional encoding.

Forward signature:
    model(spike_input, kin, nu, fps, cell_type, z) -> dff
    spike_input: (B, W)              if sub_frame_m == 1   (per-frame rate)
                 (B, W*sub_frame_m)  if sub_frame_m  > 1   (sub-frame occupancy)
    kin:         (B, 5)   raw [tau_rise_s, tau_decay_s, n_Hill, amp, baseline]
    nu:          (B,)     target noise level                         (optional)
    fps:         (B,)     frame rate (time-aware PE + adaptive span)  (optional)
    cell_type:   (B,)     categorical cell type                      (optional)
    z:           (B, latent_dim) latent code (if latent_dim > 0)      (optional)
    -> dff:      (B, W*out_m)   predicted dF/F (sub-frame structure if out_m>1)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_small_conformer_corner import (
    ConvBlock, LocalMHABlock, FFNBlock, time_aware_sinusoid,
)


DEFAULT_LOG_NORM = dict(
    log_tau_rise_mean=-3.91, log_tau_rise_std=1.33,
    log_tau_decay_mean=-1.15, log_tau_decay_std=1.07,
    hill_mean=2.75,           hill_std=0.72,
    log_amp_mean=-1.50,       log_amp_std=0.86,
    baseline_mean=0.0,        baseline_std=0.17,
)


# --------------------------------------------------------------------------- #
#  Diagonal state-space block (S4-lite / Mamba-lite)                          #
# --------------------------------------------------------------------------- #
class DiagSSMBlock(nn.Module):
    """Real diagonal linear-recurrence block: h_t = a*h_{t-1} + b*x_t, y = mix(h).

    Each channel decays with its own (learned) per-second time-constant; when an
    fps is supplied the per-step decay is a = exp(-dt / tau) with dt = 1/fps, so
    the same model produces consistent dynamics across frame rates (the
    calcium-decay inductive bias). Bidirectional: a forward and a backward scan
    are concatenated and mixed. Residual + LayerNorm wrapper, like the conformer
    blocks, so it is a drop-in replacement.
    """

    def __init__(self, d: int, dropout: float = 0.0, bidirectional: bool = True):
        super().__init__()
        self.d = d
        self.bidirectional = bidirectional
        self.ln = nn.LayerNorm(d)
        # log of per-second decay rate; init to a spread covering fast->slow.
        log_rate = torch.linspace(math.log(0.5), math.log(20.0), d)
        self.log_rate = nn.Parameter(log_rate)        # (d,) decay rate (1/s)
        self.in_lin = nn.Linear(d, d)
        n_dir = 2 if bidirectional else 1
        self.out_lin = nn.Linear(d * n_dir, d)
        self.drop = nn.Dropout(dropout)

    def _scan(self, x, a):
        """Causal scan h_t = a*h_{t-1} + x_t. EXACT FFT reformulation: since `a` is fixed within a
        sequence (LTI), the recurrence IS a causal convolution with the geometric kernel a^t, so
        h = irfft(rfft(x)*rfft(a^t)) -- O(L log L) parallel instead of the O(L) sequential loop
        (~10-50x faster on GPU, numerically identical to `_scan_seq`; same weights/output)."""
        B, L, d = x.shape
        if a.dim() == 3 and a.size(0) == 1:
            a = a.expand(B, 1, d)
        a = a.squeeze(1)                               # (B, d), decay in (0,1)
        t = torch.arange(L, device=x.device, dtype=torch.float32)
        K = torch.exp(t.view(1, 1, L) * torch.log(a.clamp(min=1e-8).float()).unsqueeze(-1))  # (B,d,L)=a^t
        nfft = 2 * L
        U = torch.fft.rfft(x.float(), n=nfft, dim=1)                  # (B,F,d)
        Kf = torch.fft.rfft(K, n=nfft, dim=2).transpose(1, 2)         # (B,F,d)
        h = torch.fft.irfft(U * Kf, n=nfft, dim=1)[:, :L, :]          # (B,L,d)
        return h.to(x.dtype)

    def _scan_seq(self, x, a):
        """Reference sequential scan (slow; kept for the equivalence check / fallback)."""
        B, L, d = x.shape
        if a.dim() == 3 and a.size(0) == 1:
            a = a.expand(B, 1, d)
        a = a.squeeze(1)
        outs = []
        h = x.new_zeros(B, d)
        for t in range(L):
            h = a * h + x[:, t, :]
            outs.append(h)
        return torch.stack(outs, dim=1)                # (B, L, d)

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        # eff_window is accepted (and ignored) so the SSM block has the same call
        # signature as LocalMHABlock and can be threaded identically.
        h = self.ln(x)
        u = self.in_lin(h)                             # (B, L, d)
        rate = torch.exp(self.log_rate)                # (d,) 1/s
        if fps is not None:
            dt = (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1)     # (B,1,1) seconds/frame
            a = torch.exp(-rate.view(1, 1, -1) * dt)            # (B,1,d)
        else:
            a = torch.exp(-rate).view(1, 1, -1)                 # (1,1,d) dt=1 frame
        fwd = self._scan(u, a)
        if self.bidirectional:
            bwd = self._scan(torch.flip(u, dims=[1]), a)
            bwd = torch.flip(bwd, dims=[1])
            y = torch.cat([fwd, bwd], dim=-1)          # (B, L, 2d)
        else:
            y = fwd
        y = self.out_lin(y)
        return x + self.drop(y)


# --------------------------------------------------------------------------- #
#  S4D: multi-state COMPLEX diagonal SSM (fps-aware), drop-in for DiagSSMBlock #
# --------------------------------------------------------------------------- #
class S4DBlock(nn.Module):
    """Diagonal complex state-space (S4D): N states/channel with complex poles
    A = -exp(log_rate) + i*omega, discretized continuously as a = exp(A*dt),
    dt = 1/fps (frame-rate-consistent dynamics -- the calcium inductive bias, now
    with the multi-state + oscillatory capacity a single REAL pole lacks, e.g. a
    native rise+decay kernel). y = Re(C h) + D u. Bidirectional. Real arithmetic
    (no torch.complex) for clean autograd. Drop-in for DiagSSMBlock."""

    def __init__(self, d: int, dropout: float = 0.0, n_state: int = 8, bidirectional: bool = True):
        super().__init__()
        self.d = d; self.N = int(n_state); self.bidirectional = bidirectional
        self.ln = nn.LayerNorm(d)
        self.in_lin = nn.Linear(d, d)
        log_rate = torch.log(torch.linspace(0.5, 20.0, self.N)).view(1, self.N).repeat(d, 1)
        self.log_rate = nn.Parameter(log_rate.clone())                 # (d,N) decay rate (1/s) = exp(.)
        self.omega = nn.Parameter(torch.linspace(0.0, 2.0, self.N).view(1, self.N).repeat(d, 1).clone())  # (d,N) rad/s
        self.C_re = nn.Parameter(torch.randn(d, self.N) / math.sqrt(self.N))
        self.C_im = nn.Parameter(torch.randn(d, self.N) / math.sqrt(self.N))
        self.D = nn.Parameter(torch.ones(d))
        self.out_lin = nn.Linear(d * (2 if bidirectional else 1), d)
        self.drop = nn.Dropout(dropout)

    def _kernel(self, m, th, L):
        # SSM impulse response K[b,d,t] = Re(sum_n C_n (m_n e^{i th_n})^t), t=0..L-1 -> (B,d,L).
        # Accumulate over the N states in a small loop so the (B,d,N,L) intermediate is NEVER
        # materialized -- memory is O(B d L), not O(B d N L). N is small (=8); L is the window.
        t = torch.arange(L, device=m.device, dtype=m.dtype)            # (L,)
        K = m.new_zeros(m.size(0), self.d, L)
        for n in range(self.N):
            mt = torch.exp(torch.log(m[:, :, n:n + 1].clamp(min=1e-8)) * t)   # (B,d,L) m_n^t
            ang = th[:, :, n:n + 1] * t                                       # (B,d,L)
            K = K + mt * (self.C_re[:, n].view(1, -1, 1) * torch.cos(ang)
                          - self.C_im[:, n].view(1, -1, 1) * torch.sin(ang))
        return K

    @staticmethod
    def _causal_conv(u, K):
        # u,K:(B,d,L). y[t]=sum_{k=0..t} K[k] u[t-k] (causal). FFT, O(L log L), no scan.
        L = u.size(-1); n = 2 * L
        y = torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(K, n=n), n=n)
        return y[..., :L]

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        h = self.ln(x); u = self.in_lin(h)                            # (B,L,d)
        L = u.size(1)
        rate = torch.exp(self.log_rate).unsqueeze(0)                  # (1,d,N)
        dt = (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1) if fps is not None else u.new_ones(1, 1, 1)
        m = torch.exp(-rate * dt)                                     # (B,d,N) magnitude (<1)
        th = self.omega.unsqueeze(0) * dt                            # (B,d,N) phase
        ut = u.transpose(1, 2)                                        # (B,d,L)
        K = self._kernel(m, th, L)                                    # (B,d,L)
        fwd = self._causal_conv(ut, K)
        if self.bidirectional:
            bwd = torch.flip(self._causal_conv(torch.flip(ut, [-1]), K), [-1])
            y = torch.cat([fwd, bwd], 1).transpose(1, 2)             # (B,L,2d)
        else:
            y = fwd.transpose(1, 2)
        y = self.out_lin(y) + self.D.view(1, 1, -1) * u
        return x + self.drop(y)


# --------------------------------------------------------------------------- #
#  Mamba: SELECTIVE (input-dependent) SSM, fps-aware, drop-in for DiagSSMBlock #
# --------------------------------------------------------------------------- #
class MambaBlock(nn.Module):
    """Selective state-space (Mamba/S6 core): input-dependent dt, B, C with a real
    structured A = -exp(A_log). Discretize a = exp(dt*A), B_bar = dt*B; the input-
    predicted dt is scaled by 1/fps so continuous-time dynamics stay frame-rate
    consistent. Depthwise causal conv + SiLU front end (Mamba's short conv). Run
    bidirectionally since generation is offline. Drop-in for DiagSSMBlock."""

    def __init__(self, d: int, dropout: float = 0.0, n_state: int = 8, d_conv: int = 4, bidirectional: bool = True):
        super().__init__()
        self.d = d; self.N = int(n_state); self.bidirectional = bidirectional
        self.ln = nn.LayerNorm(d)
        self.in_lin = nn.Linear(d, d)
        self.conv = nn.Conv1d(d, d, d_conv, groups=d, padding=d_conv - 1)   # causal depthwise
        self.dt_lin = nn.Linear(d, d)
        self.dt_bias = nn.Parameter(torch.zeros(d))
        self.B_lin = nn.Linear(d, self.N)
        self.C_lin = nn.Linear(d, self.N)
        A = torch.arange(1, self.N + 1, dtype=torch.float32).view(1, self.N).repeat(d, 1)
        self.A_log = nn.Parameter(torch.log(A))                        # (d,N) -> A=-exp(.) <0
        self.D = nn.Parameter(torch.ones(d))
        self.out_lin = nn.Linear(d * (2 if bidirectional else 1), d)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def _sscan(self, u, dt, Bm, Cm, A):
        # Selective scan of the diagonal recurrence h_t = a_t ⊙ h_{t-1} + b_t, where
        # a_t = exp(dt_t ⊙ A) ∈ (0,1] and b_t = dt_t ⊙ B_t ⊙ u_t, via a PARALLEL associative
        # (Hillis–Steele) scan over composed affine maps (a,b): combine ∘ = (a2·a1, a2·b1+b2).
        # O(log L) VECTORIZED steps instead of the old O(L) python loop -> ~L/log2(L) fewer
        # sequential kernel launches (that loop, + per-chunk gradient checkpointing, was the
        # cause of Mamba being >8× slower than S4D and timing out at the 12h B200 wall).
        # Stable across the full 1–120 fps range: a ∈ (0,1], no division (the cumsum trick
        # overflows at low fps / large dt; this affine-compose form does not). The whole scan
        # is checkpointed so backward recomputes the (now fast) scan -> memory stays O(L) inputs.
        from torch.utils.checkpoint import checkpoint
        F = nn.functional

        def _impl(u, dt, Bm, Cm, A):
            L = u.size(1)
            a = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, self.d, self.N))   # (B,L,d,N) ∈ (0,1]
            b = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)         # (B,L,d,N) = dt⊙B⊙u
            step = 1
            while step < L:
                a_sh = F.pad(a, (0, 0, 0, 0, step, 0), value=1.0)[:, :L]     # a[t-step] (identity a=1 for t<step)
                b_sh = F.pad(b, (0, 0, 0, 0, step, 0), value=0.0)[:, :L]     # b[t-step] (identity b=0 for t<step)
                b = a * b_sh + b
                a = a * a_sh
                step *= 2
            return (b * Cm.unsqueeze(2)).sum(-1)                            # y_t = (h_t ⊙ C_t).sum(N) -> (B,L,d)

        return checkpoint(_impl, u, dt, Bm, Cm, A, use_reentrant=False)

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        h = self.ln(x); u = self.in_lin(h)                            # (B,L,d)
        L = u.size(1)
        uc = self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2)    # causal depthwise conv
        u = self.act(uc)
        dt = nn.functional.softplus(self.dt_lin(u) + self.dt_bias)    # (B,L,d) >0
        if fps is not None:
            dt = dt * (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1)      # fps-aware
        Bm = self.B_lin(u); Cm = self.C_lin(u)                        # (B,L,N)
        A = -torch.exp(self.A_log)                                    # (d,N) <0
        fwd = self._sscan(u, dt, Bm, Cm, A)
        if self.bidirectional:
            bwd = torch.flip(self._sscan(torch.flip(u, [1]), torch.flip(dt, [1]),
                                         torch.flip(Bm, [1]), torch.flip(Cm, [1]), A), [1])
            y = torch.cat([fwd, bwd], -1)
        else:
            y = fwd
        y = self.out_lin(y) + self.D.view(1, 1, -1) * u
        return x + self.drop(y)


# --------------------------------------------------------------------------- #
#  HGRN-2: hierarchically-gated linear RNN -- multi-timescale forgetting       #
# --------------------------------------------------------------------------- #
def _diag_affine_scan(a, b):
    """Exact parallel (Hillis-Steele) inclusive scan of the diagonal affine recurrence
    h_t = a_t * h_{t-1} + b_t  for a,b:(B,L,d). O(log L) vectorized steps. Out-of-range
    (t<step) entries pad to the IDENTITY map (a=1,b=0) so early elements stay correct.
    Same composition the MambaBlock scan uses, minus the N-state axis."""
    L = a.size(1); step = 1
    while step < L:
        a_sh = F.pad(a, (0, 0, step, 0), value=1.0)[:, :L]   # a[t-step]
        b_sh = F.pad(b, (0, 0, step, 0), value=0.0)[:, :L]   # b[t-step]
        b = a * b_sh + b
        a = a * a_sh
        step *= 2
    return b


class HGRN2Block(nn.Module):
    """Hierarchically-gated linear RNN (HGRN-2 spirit): a data-dependent forget gate with a SPREAD of
    intrinsic timescales across channels (the multi-exponential calcium-kinetics bias), made frame-rate
    consistent (retain a_t = exp(-rate_t/fps)) and run bidirectionally. The recurrence
    h_t = a_t*h_{t-1} + (1-a_t)*v_t is a content-aware EMA whose per-channel, data-dependent rate gives
    multi-timescale memory (fast channels track transients, slow channels hold baseline). Output is
    GLA-style gated. Drop-in for DiagSSMBlock -- screen-grade reference; productionize via
    flash-linear-attention's HGRN2 (true per-LAYER forget-gate hierarchy) if it wins the bake-off."""

    def __init__(self, d: int, dropout: float = 0.0, bidirectional: bool = True):
        super().__init__()
        self.d = d; self.bidirectional = bidirectional
        self.ln = nn.LayerNorm(d)
        self.in_lin = nn.Linear(d, d)
        self.v_lin = nn.Linear(d, d)
        self.rate_lin = nn.Linear(d, d)
        # per-channel intrinsic-rate baseline spread 0.5..20 /s (slow..fast); inverse-softplus init so
        # softplus(rate_lin(h)+bias) ~ that spread at start. data-dependent term shifts it per token.
        self.rate_bias = nn.Parameter(torch.log(torch.expm1(torch.linspace(0.5, 20.0, d))))
        self.g_lin = nn.Linear(d, d)                                   # output gate (GLA-style)
        self.out_lin = nn.Linear(d * (2 if bidirectional else 1), d)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def _dir(self, v, a):
        return _diag_affine_scan(a, (1.0 - a) * v)

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        h = self.ln(x)
        v = self.v_lin(self.act(self.in_lin(h)))                       # (B,L,d) value
        rate = F.softplus(self.rate_lin(h) + self.rate_bias)           # (B,L,d) >=0 data-dependent + spread
        dt = (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1) if fps is not None else h.new_ones(1, 1, 1)
        a = torch.exp(-rate * dt)                                       # (B,L,d) retain in (0,1], fps-aware
        fwd = self._dir(v, a)
        if self.bidirectional:
            bwd = torch.flip(self._dir(torch.flip(v, [1]), torch.flip(a, [1])), [1])
            y = torch.cat([fwd, bwd], -1)
        else:
            y = fwd
        y = self.out_lin(y) * torch.sigmoid(self.g_lin(h))             # output gating
        return x + self.drop(y)


# --------------------------------------------------------------------------- #
#  GatedDeltaNet: delta-rule fast-weight memory + decay gate (timing bet)      #
# --------------------------------------------------------------------------- #
class GatedDeltaNetBlock(nn.Module):
    """Gated DeltaNet (Yang/Kautz/Hatamizadeh 2024) reference: a fast-weight associative memory written
    with the DELTA (error-correcting) rule plus a Mamba-2-style global decay gate, fps-aware & bidirectional.
    Per-head matrix state S (d_head x d_head):
        S_t = a_t * S_{t-1} + b_t (v_t - a_t (S_{t-1} k_t)) k_t^T ,   read  o_t = S_t q_t
    with L2-normalized keys, write rate b_t=beta in (0,1), decay a_t=exp(-rate_t/fps). The delta term
    corrects the STORED value toward v_t -- precise, content-addressable updates (a spike-timing /
    overlapping-event-separation bias, complementary to HGRN-2's smooth multi-timescale gating).
    Chunkwise-parallel scan (_dir, O(L/C) chunk-steps -- exact vs the _dir_seq reference, ~4x faster); the
    _dir_seq sequential form is kept for verification. Drop-in for DiagSSMBlock."""

    def __init__(self, d: int, dropout: float = 0.0, n_heads: int = 4, bidirectional: bool = True,
                 chunk_size: int = 16):
        super().__init__()
        assert d % n_heads == 0, "d must divide n_heads"
        self.d = d; self.h = n_heads; self.hd = d // n_heads; self.bidir = bidirectional; self.chunk = chunk_size
        self.ln = nn.LayerNorm(d)
        self.q_lin = nn.Linear(d, d); self.k_lin = nn.Linear(d, d); self.v_lin = nn.Linear(d, d)
        self.beta_lin = nn.Linear(d, n_heads)                          # write rate beta in (0,1) per head
        self.rate_lin = nn.Linear(d, n_heads)                          # decay rate per head (fps-aware)
        self.rate_bias = nn.Parameter(torch.zeros(n_heads))
        self.g_lin = nn.Linear(d, d)                                   # output gate
        self.out_lin = nn.Linear(d * (2 if bidirectional else 1), d)
        self.drop = nn.Dropout(dropout)

    def _dir_seq(self, q, k, v, alpha, beta):
        # reference O(L) sequential scan -- ground truth for verifying the chunked path
        B, L, h, hd = q.shape
        S = q.new_zeros(B, h, hd, hd)                                  # (B,h, d_v, d_k)
        outs = []
        for t in range(L):
            kt = k[:, t]; vt = v[:, t]; qt = q[:, t]                   # (B,h,hd)
            at = alpha[:, t][..., None]; bt = beta[:, t][..., None]    # (B,h,1)
            Sk = (S * kt[:, :, None, :]).sum(-1)                       # (B,h,hd) = S k_t
            upd = vt - at * Sk                                         # (B,h,hd) delta error
            S = at[..., None] * S + bt[..., None] * torch.einsum('bhv,bhk->bhvk', upd, kt)
            outs.append((S * qt[:, :, None, :]).sum(-1))              # (B,h,hd) = S q_t
        return torch.stack(outs, 1).reshape(B, L, h * hd)             # (B,L,d)

    def _dir(self, q, k, v, alpha, beta):
        # Chunkwise parallel gated delta rule (Yang et al. 2024), SPEEDUP form. Within a chunk, normalize the state
        # by cumulative decay gamma=cumprod(alpha) -> pure delta rule, pseudo-values U=(I+A)^-1 beta(v~ - K S0^T).
        # KEY: U is AFFINE in the carried state S0 -> split U = U_loc - S0 @ Wt where U_loc=(I+A)^-1 beta v~ and
        # Wt=(I+A)^-1 diag(beta) K are BOTH state-free. So the expensive bits (the CxC inverse M=(I+A)^-1, U_loc, Wt,
        # the QK attention matrices, gamma) are computed ONCE, BATCHED over all chunks; the sequential loop then only
        # carries the state with cheap matmuls (no per-chunk triangular solve). Exact vs _dir_seq; C small -> 1/gamma safe.
        B, L, h, hd = q.shape
        C = min(self.chunk, L); pad = (C - L % C) % C
        if pad:
            q = F.pad(q, (0, 0, 0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, 0, 0, pad))
            alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0); beta = F.pad(beta, (0, 0, 0, pad), value=0.0)
        Lp = L + pad; nC = Lp // C
        rs = lambda t: t.permute(0, 2, 1, 3).reshape(B, h, nC, C, hd)
        qc, kc, vc = rs(q), rs(k), rs(v)                                  # (B,h,nC,C,hd)
        ac = alpha.permute(0, 2, 1).reshape(B, h, nC, C); bc = beta.permute(0, 2, 1).reshape(B, h, nC, C)
        gamma = torch.cumprod(ac, dim=-1)                                # (B,h,nC,C)
        vt = vc / gamma.clamp(min=1e-8).unsqueeze(-1)                    # v~
        A = torch.tril(bc.unsqueeze(-1) * torch.einsum('bhnik,bhnjk->bhnij', kc, kc), -1)  # beta_i k_i.k_j strictly lower
        eye = torch.eye(C, device=q.device, dtype=q.dtype).expand(B, h, nC, C, C)
        M = torch.linalg.solve_triangular(A, eye, upper=False, unitriangular=True)         # (I+A)^-1, BATCHED over chunks
        U_loc = torch.einsum('bhnij,bhnjv->bhniv', M, bc.unsqueeze(-1) * vt)               # state-free part of U
        Wt = torch.einsum('bhnij,bhnjk->bhnik', M, bc.unsqueeze(-1) * kc)                  # state-correction: U -= S0 @ Wt
        QK = torch.tril(torch.einsum('bhnik,bhnjk->bhnij', qc, kc), 0)                     # inclusive lower q_i.k_j
        S = q.new_zeros(B, h, hd, hd); outs = []
        for c in range(nC):
            U = U_loc[:, :, c] - torch.einsum('bhvk,bhik->bhiv', S, Wt[:, :, c])           # U_c (state-corrected)
            Sq = torch.einsum('bhvk,bhck->bhcv', S, qc[:, :, c])                           # S0 q_i
            outs.append(gamma[:, :, c].unsqueeze(-1) * (Sq + torch.einsum('bhij,bhjv->bhiv', QK[:, :, c], U)))
            S = gamma[:, :, c, -1][..., None, None] * (S + torch.einsum('bhcv,bhck->bhvk', U, kc[:, :, c]))
        out = torch.stack(outs, 2).reshape(B, h, Lp, hd).permute(0, 2, 1, 3).reshape(B, Lp, h * hd)
        return out[:, :L]

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        B, L, _ = x.shape
        h = self.ln(x)
        q = self.q_lin(h).view(B, L, self.h, self.hd)
        k = F.normalize(self.k_lin(h).view(B, L, self.h, self.hd), dim=-1)
        v = self.v_lin(h).view(B, L, self.h, self.hd)
        beta = torch.sigmoid(self.beta_lin(h))                         # (B,L,h)
        rate = F.softplus(self.rate_lin(h) + self.rate_bias)          # (B,L,h) >=0
        dt = (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1) if fps is not None else x.new_ones(1, 1, 1)
        alpha = torch.exp(-rate * dt)                                  # (B,L,h) decay in (0,1], fps-aware
        if self.bidir:                                                # batch fwd + reversed-bwd -> ONE scan (loop runs once)
            fl = lambda t: torch.flip(t, [1])
            yb = self._dir(torch.cat([q, fl(q)], 0), torch.cat([k, fl(k)], 0), torch.cat([v, fl(v)], 0),
                           torch.cat([alpha, fl(alpha)], 0), torch.cat([beta, fl(beta)], 0))
            y = torch.cat([yb[:B], torch.flip(yb[B:], [1])], -1)
        else:
            y = self._dir(q, k, v, alpha, beta)
        y = self.out_lin(y) * torch.sigmoid(self.g_lin(h))            # output gating
        return x + self.drop(y)


def _ssd_chunked(xh, a, B, C, chunk: int = 64):
    """Chunked state-space duality scan (shared by Mamba-2 and Mamba-3). h_t=a_t h_{t-1}+x_t B_t^T, y=H C, with
    scalar-per-head decay a. Per chunk: INTRA=tril(decay-weighted C_i.B_j)@x (quadratic; no (P,N) state across L),
    INTER=gamma_i(H0 C_i), carry H=gamma_C H0 + sum_j (gamma_C/gamma_j) x_j B_j^T. Decay ratios <=1 (clamp log at 0)
    -> stable with large chunks (no 1/gamma like GDN). xh:(b,L,h,P) a:(b,L,h) B,C:(b,L,h,N)."""
    b_, L, h, P = xh.shape
    Cs = min(chunk, L); pad = (Cs - L % Cs) % Cs
    if pad:
        xh = F.pad(xh, (0, 0, 0, 0, 0, pad)); B = F.pad(B, (0, 0, 0, 0, 0, pad)); C = F.pad(C, (0, 0, 0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
    Lp = L + pad; nC = Lp // Cs
    rs = lambda t: t.permute(0, 2, 1, 3).reshape(b_, h, nC, Cs, t.shape[-1])
    xc, Bc, Cc = rs(xh), rs(B), rs(C)
    ac = a.permute(0, 2, 1).reshape(b_, h, nC, Cs)
    causal = torch.tril(torch.ones(Cs, Cs, device=xh.device, dtype=torch.bool))
    H = xh.new_zeros(b_, h, P, B.shape[-1]); outs = []
    for c in range(nC):
        xk, Bk, Ck, a_c = xc[:, :, c], Bc[:, :, c], Cc[:, :, c], ac[:, :, c]
        logg = torch.cumsum(torch.log(a_c.clamp(min=1e-8)), dim=-1)
        decay = torch.exp((logg[..., :, None] - logg[..., None, :]).clamp(max=0.0))
        M = torch.where(causal, torch.einsum('bhin,bhjn->bhij', Ck, Bk) * decay, xh.new_zeros(1))
        gamma = torch.exp(logg)
        outs.append(torch.einsum('bhij,bhjp->bhip', M, xk) + gamma.unsqueeze(-1) * torch.einsum('bhin,bhpn->bhip', Ck, H))
        w = torch.exp((logg[..., -1:] - logg).clamp(max=0.0))
        H = gamma[..., -1][..., None, None] * H + torch.einsum('bhjp,bhjn->bhpn', xk * w.unsqueeze(-1), Bk)
    return torch.stack(outs, 2).reshape(b_, h, Lp, P).permute(0, 2, 1, 3).reshape(b_, Lp, h * P)[:, :L]


# --------------------------------------------------------------------------- #
#  Mamba-2 / SSD: scalar-per-head decay + larger state (state-space duality)   #
# --------------------------------------------------------------------------- #
class Mamba2Block(nn.Module):
    """Mamba-2 (state-space duality) reference: a selective SSM whose decay a_t is SCALAR per head
    (vs Mamba-1's per-channel diagonal) with a larger shared state N -- less expressive per-channel
    decay, more state capacity, matmul-friendly. Outer-product linear-attention state
    H_t = a_t H_{t-1} + x_t B_t^T ; y_t = H_t C_t, scalar a_t lets us reuse the exact parallel affine
    scan. Depthwise causal conv + SiLU front end; fps-aware (a_t=exp(-dt_t*exp(A_log)), dt~1/fps);
    bidirectional. Drop-in for DiagSSMBlock (screen-grade; chunked SSD for long generator windows later)."""

    def __init__(self, d: int, dropout: float = 0.0, n_heads: int = 4, n_state: int = 16,
                 d_conv: int = 4, bidirectional: bool = True):
        super().__init__()
        assert d % n_heads == 0, "d must divide n_heads"
        self.d = d; self.h = n_heads; self.hd = d // n_heads; self.N = int(n_state); self.bidir = bidirectional
        self.ln = nn.LayerNorm(d)
        self.in_lin = nn.Linear(d, d)
        self.conv = nn.Conv1d(d, d, d_conv, groups=d, padding=d_conv - 1)   # causal depthwise
        self.B_lin = nn.Linear(d, n_heads * self.N)
        self.C_lin = nn.Linear(d, n_heads * self.N)
        self.dt_lin = nn.Linear(d, n_heads); self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.A_log = nn.Parameter(torch.zeros(n_heads))                # scalar A per head, A=-exp(A_log)<0
        self.D = nn.Parameter(torch.ones(n_heads))
        self.out_lin = nn.Linear(d * (2 if bidirectional else 1), d)
        self.act = nn.SiLU(); self.drop = nn.Dropout(dropout)

    def _dir_ref(self, xh, a, B, C):
        # reference via the flat affine scan -- CORRECT but materializes (B,L,P,N) (slow/mem-heavy at long L);
        # used only to verify _dir at short length. H_t=a_t H_{t-1}+x_t B_t^T (outer), y=H C.
        b_, L, h, P = xh.shape; N = B.shape[-1]
        bx = torch.einsum('blhp,blhn->blhpn', xh, B)
        a_full = a[..., None, None].expand(b_, L, h, P, N)
        H = _diag_affine_scan(a_full.reshape(b_, L, h * P * N),
                              bx.reshape(b_, L, h * P * N)).reshape(b_, L, h, P, N)
        return torch.einsum('blhpn,blhn->blhp', H, C).reshape(b_, L, h * P)

    def _dir(self, xh, a, B, C):
        return _ssd_chunked(xh, a, B, C)                              # shared chunked SSD scan (exact vs _dir_ref)

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        b_, L, _ = x.shape
        h = self.ln(x); u = self.in_lin(h)
        u = self.act(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))   # causal conv + SiLU
        xh = u.view(b_, L, self.h, self.hd)
        B = self.B_lin(u).view(b_, L, self.h, self.N)
        C = self.C_lin(u).view(b_, L, self.h, self.N)
        dt = F.softplus(self.dt_lin(u) + self.dt_bias)                 # (b,L,h) >0
        if fps is not None:
            dt = dt * (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1)       # fps-aware
        a = torch.exp(dt * (-torch.exp(self.A_log)))                   # (b,L,h) scalar decay in (0,1)
        fwd = self._dir(xh, a, B, C)
        if self.bidir:
            bwd = torch.flip(self._dir(torch.flip(xh, [1]), torch.flip(a, [1]),
                                       torch.flip(B, [1]), torch.flip(C, [1])), [1])
            y = torch.cat([fwd, bwd], -1)
        else:
            y = fwd
        y = self.out_lin(y) + self.D.repeat_interleave(self.hd).view(1, 1, -1) * u
        return x + self.drop(y)


# --------------------------------------------------------------------------- #
#  Mamba-3: exponential-TRAPEZOIDAL discretization (+ complex rotation = full) #
# --------------------------------------------------------------------------- #
def _rope_pairs(x, cos, sin):
    # rotate adjacent pairs of the last dim by per-pair angle. x:(...,N even); cos,sin:(...,N/2).
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(x.shape)


def _trap_ssd(xh, alpha, beta, gamma, B, C):
    # exponential-trapezoidal SSD: h_t = alpha h_{t-1} + beta x_{t-1}B_{t-1} + gamma x_t B_t. Two _ssd_chunked
    # passes (current gamma-tap + shifted-previous beta-tap) summed by linearity. alpha,beta,gamma:(b,L,h).
    y_cur = _ssd_chunked(xh * gamma.unsqueeze(-1), alpha, B, C)
    xs = F.pad(xh, (0, 0, 0, 0, 1, 0))[:, :-1]                         # x_{t-1}
    Bs = F.pad(B, (0, 0, 0, 0, 1, 0))[:, :-1]                          # B_{t-1}
    return y_cur + _ssd_chunked(xs * beta.unsqueeze(-1), alpha, Bs, C)


class Mamba3LiteBlock(nn.Module):
    """Mamba-3 LITE -- Mamba-2 SSD with ONLY the exponential-TRAPEZOIDAL discretization (Mamba-3 Eq.6); isolates
    the lever that maps to our physics. The 2nd-order rule adds a decayed PREVIOUS-input tap, most accurate at
    large dt = LOW fps (where 1st-order Euler is worst = our deploy weak spot). alpha=exp(dt A), beta=(1-lam)dt alpha,
    gamma=lam dt, lam in (0,1) data-dependent (lam=1 -> Mamba-2). Real state, no rotation/MIMO. Drop-in."""

    def __init__(self, d: int, dropout: float = 0.0, n_heads: int = 4, n_state: int = 16, d_conv: int = 4,
                 bidirectional: bool = True):
        super().__init__()
        assert d % n_heads == 0, "d must divide n_heads"
        self.d = d; self.h = n_heads; self.hd = d // n_heads; self.N = int(n_state); self.bidir = bidirectional
        self.ln = nn.LayerNorm(d); self.in_lin = nn.Linear(d, d)
        self.conv = nn.Conv1d(d, d, d_conv, groups=d, padding=d_conv - 1)
        self.B_lin = nn.Linear(d, n_heads * self.N); self.C_lin = nn.Linear(d, n_heads * self.N)
        self.dt_lin = nn.Linear(d, n_heads); self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.lam_lin = nn.Linear(d, n_heads)
        self.A_log = nn.Parameter(torch.zeros(n_heads)); self.D = nn.Parameter(torch.ones(n_heads))
        self.out_lin = nn.Linear(d * (2 if bidirectional else 1), d); self.act = nn.SiLU(); self.drop = nn.Dropout(dropout)

    def _abc(self, u, fps):
        b_, L = u.shape[0], u.shape[1]
        xh = u.view(b_, L, self.h, self.hd)
        B = self.B_lin(u).view(b_, L, self.h, self.N); C = self.C_lin(u).view(b_, L, self.h, self.N)
        dt = F.softplus(self.dt_lin(u) + self.dt_bias)
        if fps is not None:
            dt = dt * (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1)
        A = -torch.exp(self.A_log); lam = torch.sigmoid(self.lam_lin(u))
        alpha = torch.exp(dt * A); gamma = lam * dt; beta = (1.0 - lam) * dt * alpha
        return xh, B, C, alpha, beta, gamma

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        b_, L, _ = x.shape
        u = self.act(self.conv(self.in_lin(self.ln(x)).transpose(1, 2))[..., :L].transpose(1, 2))
        xh, B, C, alpha, beta, gamma = self._abc(u, fps)
        fwd = _trap_ssd(xh, alpha, beta, gamma, B, C)
        if self.bidir:
            fl = lambda t: torch.flip(t, [1])
            bwd = fl(_trap_ssd(fl(xh), fl(alpha), fl(beta), fl(gamma), fl(B), fl(C)))
            y = torch.cat([fwd, bwd], -1)
        else:
            y = fwd
        return x + self.drop(self.out_lin(y) + self.D.repeat_interleave(self.hd).view(1, 1, -1) * u)


class Mamba3Block(nn.Module):
    """Mamba-3 (full): exponential-TRAPEZOIDAL discretization + COMPLEX rotation (Eq.11 -- cumulative block-diagonal
    rotations R_t on B,C = data-dependent RoPE on the SSM state -> oscillatory/rotational state). The angle
    phi_t=cumsum(dt*theta_t) is applied to B,C before the trapezoidal SSD, so the RELATIVE rotation phi_t-phi_j
    appears in the scores. MIMO (rank R>1, the capacity knob) left at R=1 -- our width sweep was flat; trivial to
    raise. fps-aware, bidirectional. Drop-in for DiagSSMBlock."""

    def __init__(self, d: int, dropout: float = 0.0, n_heads: int = 4, n_state: int = 16, d_conv: int = 4,
                 bidirectional: bool = True):
        super().__init__()
        assert d % n_heads == 0 and n_state % 2 == 0, "need d%heads==0 and even n_state"
        self.d = d; self.h = n_heads; self.hd = d // n_heads; self.N = int(n_state); self.bidir = bidirectional
        self.ln = nn.LayerNorm(d); self.in_lin = nn.Linear(d, d)
        self.conv = nn.Conv1d(d, d, d_conv, groups=d, padding=d_conv - 1)
        self.B_lin = nn.Linear(d, n_heads * self.N); self.C_lin = nn.Linear(d, n_heads * self.N)
        self.dt_lin = nn.Linear(d, n_heads); self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.lam_lin = nn.Linear(d, n_heads); self.theta_lin = nn.Linear(d, n_heads * (self.N // 2))
        self.A_log = nn.Parameter(torch.zeros(n_heads)); self.D = nn.Parameter(torch.ones(n_heads))
        self.out_lin = nn.Linear(d * (2 if bidirectional else 1), d); self.act = nn.SiLU(); self.drop = nn.Dropout(dropout)

    def _dir(self, xh, B, C, dt, lam, theta):
        A = -torch.exp(self.A_log)
        alpha = torch.exp(dt * A); gamma = lam * dt; beta = (1.0 - lam) * dt * alpha
        phi = torch.cumsum(dt.unsqueeze(-1) * theta, dim=1)            # (b,L,h,N/2) cumulative rotation angle
        cos, sin = torch.cos(phi), torch.sin(phi)
        return _trap_ssd(xh, alpha, beta, gamma, _rope_pairs(B, cos, sin), _rope_pairs(C, cos, sin))

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        b_, L, _ = x.shape
        u = self.act(self.conv(self.in_lin(self.ln(x)).transpose(1, 2))[..., :L].transpose(1, 2))
        xh = u.view(b_, L, self.h, self.hd)
        B = self.B_lin(u).view(b_, L, self.h, self.N); C = self.C_lin(u).view(b_, L, self.h, self.N)
        dt = F.softplus(self.dt_lin(u) + self.dt_bias)
        if fps is not None:
            dt = dt * (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1)
        lam = torch.sigmoid(self.lam_lin(u)); theta = self.theta_lin(u).view(b_, L, self.h, self.N // 2)
        fwd = self._dir(xh, B, C, dt, lam, theta)
        if self.bidir:
            fl = lambda t: torch.flip(t, [1])
            bwd = fl(self._dir(fl(xh), fl(B), fl(C), fl(dt), fl(lam), fl(theta)))
            y = torch.cat([fwd, bwd], -1)
        else:
            y = fwd
        return x + self.drop(self.out_lin(y) + self.D.repeat_interleave(self.hd).view(1, 1, -1) * u)


# --------------------------------------------------------------------------- #
#  GatedDeltaNet-3: delta-rule memory (connectivity) x Mamba-3 rotation (timing)#
# --------------------------------------------------------------------------- #
class GatedDeltaNet3Block(GatedDeltaNetBlock):
    """FUSION of the two bake-off lever-winners. GatedDeltaNet's delta-rule associative memory was the
    CONNECTIVITY winner (error-correcting, content-addressable writes separate overlapping synaptic events in
    clutter); Mamba-3's COMPLEX ROTATION was the timing/jitter lever. This block keeps the delta-rule
    recurrence verbatim -- it INHERITS the verified chunkwise scan (_dir) and its sequential ground truth
    (_dir_seq) unchanged -- and adds ONLY a cumulative, data-dependent, fps-aware rotation of the read query
    q and the write key k by angle phi_t = cumsum(dt * theta_t) (Mamba-3 Eq.11 RoPE, applied to the delta
    rule's q/k rather than an SSM's B/C). The value v (the stored CONTENT) is left unrotated, mirroring
    Mamba-3 rotating B,C but not x.

    Why the fusion is clean (and exact): a rotation is an ORTHOGONAL map on the head dimension, so it
    (a) preserves the L2-normalized key's unit norm and (b) commutes with every inner product the chunked
    scan relies on -- q_i.k_j, k_i.k_j -- turning each into a RELATIVE-rotation score q_i^T R_{i-j} k_j.
    Hence rotating q,k BEFORE _dir leaves the chunked path exact vs _dir_seq by construction (the scan never
    assumes anything about q,k beyond being vectors). The relative phase phi_i - phi_j then injects an
    oscillatory / relative-timing code into the associative memory -- a phase representation of spike latency
    layered on top of the error-correcting writes. theta == 0 recovers the base GatedDeltaNetBlock exactly.
    Partial rotary when the head dim is odd (rotate the even sub-block, pass the last channel through), so it
    stays a (d, dropout) drop-in for any width. fps-aware, bidirectional (each direction gets its own causal
    cumulative rotation, batched into the single GDN scan)."""

    def __init__(self, d: int, dropout: float = 0.0, n_heads: int = 4, bidirectional: bool = True,
                 chunk_size: int = 16):
        super().__init__(d, dropout, n_heads, bidirectional, chunk_size)
        self.nr = self.hd // 2                                          # # rotation pairs (partial rotary if hd odd)
        self.theta_lin = nn.Linear(d, n_heads * self.nr) if self.nr > 0 else None   # data-dependent rotation rate/pair
        if self.theta_lin is not None:
            nn.init.zeros_(self.theta_lin.bias)                         # start near no-rotation -> base-GDN init

    def _rot(self, t, cos, sin):
        # rotate the first 2*nr channels of the head dim in adjacent pairs; pass an odd trailing channel
        # through unrotated. t:(B,L,h,hd); cos,sin:(B,L,h,nr).
        if self.nr == 0:
            return t
        r = t[..., :2 * self.nr]; rest = t[..., 2 * self.nr:]
        r1 = r[..., 0::2]; r2 = r[..., 1::2]
        ro = torch.stack([r1 * cos - r2 * sin, r1 * sin + r2 * cos], dim=-1).reshape(r.shape)
        return torch.cat([ro, rest], dim=-1) if rest.size(-1) else ro

    def _rotqk(self, q, k, theta, dt):
        # cumulative fps-aware rotation angle phi_t = cumsum(dt * theta_t) over time; rotate q,k by it.
        phi = torch.cumsum(dt.unsqueeze(-1) * theta, dim=1)            # (B,L,h,nr)
        cos, sin = torch.cos(phi), torch.sin(phi)
        return self._rot(q, cos, sin), self._rot(k, cos, sin)

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        B, L, _ = x.shape
        h = self.ln(x)
        q = self.q_lin(h).view(B, L, self.h, self.hd)
        k = F.normalize(self.k_lin(h).view(B, L, self.h, self.hd), dim=-1)
        v = self.v_lin(h).view(B, L, self.h, self.hd)
        beta = torch.sigmoid(self.beta_lin(h))                         # (B,L,h)
        rate = F.softplus(self.rate_lin(h) + self.rate_bias)          # (B,L,h) >=0
        dt = (1.0 / fps.clamp(min=1e-3)).view(-1, 1, 1) if fps is not None else x.new_ones(1, 1, 1)
        alpha = torch.exp(-rate * dt)                                  # fps-aware decay in (0,1]
        if self.nr > 0:
            theta = self.theta_lin(h).view(B, L, self.h, self.nr)
            qf, kf = self._rotqk(q, k, theta, dt)
        else:
            qf, kf = q, k
        if self.bidir:                                                # fwd + reversed-bwd batched -> ONE scan
            fl = lambda t: torch.flip(t, [1])
            if self.nr > 0:
                qb, kb = self._rotqk(fl(q), fl(k), fl(theta), dt)     # bwd stream: own causal cumulative rotation
            else:
                qb, kb = fl(q), fl(k)
            yb = self._dir(torch.cat([qf, qb], 0), torch.cat([kf, kb], 0), torch.cat([v, fl(v)], 0),
                           torch.cat([alpha, fl(alpha)], 0), torch.cat([beta, fl(beta)], 0))
            y = torch.cat([yb[:B], torch.flip(yb[B:], [1])], -1)
        else:
            y = self._dir(qf, kf, v, alpha, beta)
        y = self.out_lin(y) * torch.sigmoid(self.g_lin(h))            # output gating
        return x + self.drop(y)


# --------------------------------------------------------------------------- #
#  Hybrid (Mamba + GLOBAL attention) and MoE-Mamba                             #
# --------------------------------------------------------------------------- #
class HybridBlock(nn.Module):
    """Bamba/Jamba-style hybrid: a Mamba (selective SSM) sublayer for linear-time calcium kinetics, PLUS a
    GLOBAL multi-head self-attention sublayer for content-based routing that the conformer's LOCAL (fps-banded)
    attention lacks -- each residual. fps flows to the Mamba sublayer. Drop-in for DiagSSMBlock."""

    def __init__(self, d: int, dropout: float = 0.0, n_heads: int = 4):
        super().__init__()
        self.mamba = MambaBlock(d, dropout)                            # selective SSM sublayer (own residual)
        self.ln = nn.LayerNorm(d); self.nh = n_heads
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d); self.drop = nn.Dropout(dropout)

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        x = self.mamba(x, eff_window=eff_window, fps=fps)             # kinetics sublayer
        B, L, d = x.shape
        qkv = self.ln(x).reshape(B, L, d)
        qkv = self.qkv(qkv).reshape(B, L, 3, self.nh, d // self.nh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])    # GLOBAL (no mask) bidirectional attn
        o = o.transpose(1, 2).reshape(B, L, d)
        return x + self.drop(self.proj(o))                           # global-routing sublayer (residual)


class MoEMambaBlock(nn.Module):
    """Mixture-of-Experts Mamba: K MambaBlock experts with a per-token softmax router. Tests whether expert
    SPECIALIZATION / added capacity helps -- the capacity lever the flat conformer width-sweep deprioritized.
    Soft-weighted (computes all experts, residual-preserving since sum(w)=1) -- screen-grade; sparse top-1
    routing for efficiency if it wins. Drop-in for DiagSSMBlock."""

    def __init__(self, d: int, dropout: float = 0.0, n_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([MambaBlock(d, dropout) for _ in range(n_experts)])
        self.router = nn.Linear(d, n_experts); self.ln = nn.LayerNorm(d)

    def forward(self, x, eff_window: int = None, fps: torch.Tensor = None):
        w = torch.softmax(self.router(self.ln(x)), dim=-1)            # (B,L,K) per-token expert weights
        ys = torch.stack([e(x, eff_window=eff_window, fps=fps) for e in self.experts], -1)  # (B,L,d,K)
        return (ys * w.unsqueeze(-2)).sum(-1)                         # (B,L,d) convex mix (keeps residual)


# --------------------------------------------------------------------------- #
#  A single backbone "stage": conformer (conv + local-MHA + FFN) OR ssm       #
# --------------------------------------------------------------------------- #
class _BackboneStack(nn.Module):
    """One stack of `layers` blocks, either conformer or ssm flavored.

    Exposes a uniform forward(h, eff_window, fps). Used directly for the
    conformer/ssm backbones and as the per-expert branch for moe.
    """

    def __init__(self, kind: str, d: int, n_heads: int, ffn: int, layers: int,
                 conv_kernel: int, W: int, dropout: float):
        super().__init__()
        self.kind = kind
        self.blocks = nn.ModuleList()
        # Track which blocks are attention/ssm (need eff_window / fps) so forward
        # can pass the right kwargs without isinstance churn.
        self._block_kinds = []
        # The generator owns a NATIVE "moe" backbone (MoE over conformer branches) -> expose MoEMambaBlock under
        # the distinct key "moemamba" so it never shadows that. "hybrid" has no native collision. These keys are
        # also gated by the DffGeneratorV2 backbone assert (widened to enable the generator-side bake-off).
        _ssm_cls = {"ssm": DiagSSMBlock, "s4d": S4DBlock, "mamba": MambaBlock, "hgrn2": HGRN2Block,
                    "gdn": GatedDeltaNetBlock, "mamba2": Mamba2Block, "hybrid": HybridBlock, "moemamba": MoEMambaBlock,
                    "mamba3lite": Mamba3LiteBlock, "mamba3": Mamba3Block, "gdn3": GatedDeltaNet3Block}
        for _ in range(layers):
            if kind in _ssm_cls:
                self.blocks.append(ConvBlock(d, conv_kernel, dropout))
                self._block_kinds.append("conv")
                self.blocks.append(_ssm_cls[kind](d, dropout))   # tagged "ssm" -> gets fps for continuous-time discretization
                self._block_kinds.append("ssm")
                self.blocks.append(FFNBlock(d, ffn, dropout))
                self._block_kinds.append("ffn")
            else:                                       # conformer (also moe branch)
                self.blocks.append(ConvBlock(d, conv_kernel, dropout))
                self._block_kinds.append("conv")
                self.blocks.append(LocalMHABlock(d, n_heads, W=W, dropout=dropout))
                self._block_kinds.append("attn")
                self.blocks.append(FFNBlock(d, ffn, dropout))
                self._block_kinds.append("ffn")

    def forward(self, h, eff_window: int = None, fps: torch.Tensor = None):
        for blk, k in zip(self.blocks, self._block_kinds):
            if k == "attn":
                h = blk(h, eff_window=eff_window)
            elif k == "ssm":
                h = blk(h, eff_window=eff_window, fps=fps)
            else:
                h = blk(h)
        return h


class DffGeneratorV2(nn.Module):
    """Enhanced Conformer/SSM/MoE dF/F generator.

    See module docstring for the full feature list. Backwards-compatible
    conditioning (biophys kin + (nu,fps,cell) FiLM + time-aware PE) with v1.
    """

    def __init__(self, num_cell_types: int = 4,
                 d: int = 64, n_heads: int = 4, ffn: int = 128,
                 layers: int = 6, conv_kernel: int = 15,
                 W: int = 128,
                 w_mode: str = "adaptive", t_sec: float = 3.0,
                 w_floor: int = 32, w_ceil: int = 512,
                 sub_frame_m: int = 1, out_m: int = 1,
                 backbone: str = "conformer",
                 n_experts: int = 1, dropout: float = 0.0,
                 time_aware_pos: bool = True,
                 log_norm: dict = None,
                 latent_dim: int = 0,
                 cond_mode: str = "add",
                 phys_template: bool = False):
        super().__init__()
        assert backbone in ("conformer", "ssm", "s4d", "mamba", "moe",
                            "hgrn2", "gdn", "mamba2", "hybrid", "moemamba",
                            "mamba3lite", "mamba3", "gdn3"), backbone
        assert w_mode in ("adaptive", "fixed"), w_mode
        assert cond_mode in ("add", "film"), cond_mode
        self.W = W
        self.d = d
        self.backbone = backbone
        self.w_mode = w_mode
        self.t_sec = float(t_sec)
        self.w_floor = int(w_floor)
        self.w_ceil = int(w_ceil)
        self.sub_frame_m = int(sub_frame_m)
        self.out_m = int(out_m)
        self.n_experts = int(n_experts) if backbone == "moe" else 1
        self.time_aware_pos = time_aware_pos
        self.latent_dim = int(latent_dim)

        # --- synth-realism injection (per-spike onset jitter + amplitude CV) ---
        # Buffers (not cfg) so they ride in state_dict past the loader's cfg-whitelist.
        # Default 0 == OFF == identical to before. Set via the re-save script.
        self.register_buffer("inj_jit_ms", torch.tensor(0.0))   # onset-jitter std, ms
        self.register_buffer("inj_amp_cv", torch.tensor(0.0))   # per-spike amplitude CV

        # --- input projection: m sub-frame channels per frame -> d -----------
        # sub_frame_m==1 reduces to the v1 (1 -> d) projection.
        self.in_proj = nn.Linear(self.sub_frame_m, d)

        # --- biophysical kin -> d (FiLM), identical normalization to v1 ------
        self.kin_mlp = nn.Sequential(
            nn.Linear(5, 16), nn.GELU(),
            nn.Linear(16, d),
        )
        with torch.no_grad():
            nn.init.trunc_normal_(self.kin_mlp[-1].weight, std=0.05)
            nn.init.zeros_(self.kin_mlp[-1].bias)

        self.cell_type_emb = nn.Embedding(num_cell_types, d)
        self.nu_proj = nn.Linear(1, d)
        self.fps_proj = nn.Linear(1, d)
        nn.init.trunc_normal_(self.cell_type_emb.weight, std=0.02)

        # --- optional latent code -> d (stochastic generation) ---------------
        if self.latent_dim > 0:
            self.z_proj = nn.Sequential(
                nn.Linear(self.latent_dim, d), nn.GELU(), nn.Linear(d, d),
            )

        # --- conditioning-fidelity upgrades (default OFF == v1 additive FiLM) -
        # Address the documented decay-SQUASH: the additive-once cond vector cannot
        # make the backbone render the conditioned tau, so the learned kernel collapses
        # to ~impulse (gen tail ~0 vs real ~0.18 @ +0.5s). Both knobs are flag-gated so
        # legacy ckpts/arms are bit-identical when off (no new params created).
        self.cond_mode = cond_mode
        self.phys_template = bool(phys_template)
        if self.cond_mode == "film":
            # true FiLM: (gamma, beta) from the summed conditioning vector; ZERO-init so
            # training STARTS at h*(1+0)+0 = h (identity) and learns to modulate.
            self.film = nn.Linear(d, 2 * d)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        if self.phys_template:
            # project the biexp kinetics-template (1 channel) into width d; small init so
            # it starts as a gentle decay-skeleton prior that the backbone refines.
            self.tmpl_proj = nn.Linear(1, d)
            nn.init.trunc_normal_(self.tmpl_proj.weight, std=0.05)
            nn.init.zeros_(self.tmpl_proj.bias)

        if not time_aware_pos:
            self.pos = nn.Parameter(torch.zeros(1, W, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
        self.emb_drop = nn.Dropout(dropout)

        # --- backbone --------------------------------------------------------
        if backbone == "moe":
            self.experts = nn.ModuleList([
                _BackboneStack("conformer", d, n_heads, ffn, layers,
                               conv_kernel, W, dropout)
                for _ in range(self.n_experts)
            ])
            # gate on (log nu, log fps) -> n_experts soft weights (bulk/corner style)
            self.gate = nn.Sequential(
                nn.Linear(2, max(8, self.n_experts * 2)), nn.GELU(),
                nn.Linear(max(8, self.n_experts * 2), self.n_experts),
            )
        else:
            self.stack = _BackboneStack(backbone, d, n_heads, ffn, layers,
                                        conv_kernel, W, dropout)

        self.ln = nn.LayerNorm(d)
        # Output head emits out_m sub-frame channels per frame.
        self.out = nn.Linear(d, self.out_m)

        ln = log_norm or DEFAULT_LOG_NORM
        for k in ("log_tau_rise_mean", "log_tau_rise_std",
                  "log_tau_decay_mean", "log_tau_decay_std",
                  "hill_mean", "hill_std",
                  "log_amp_mean", "log_amp_std",
                  "baseline_mean", "baseline_std"):
            self.register_buffer("_n_" + k, torch.tensor(float(ln[k])))

    # ----------------------------------------------------------------------- #
    def _kin_embed(self, kin: torch.Tensor) -> torch.Tensor:
        """kin: (B, 5) raw -> (B, d). Same normalization as v1."""
        log_tr = torch.log(kin[:, 0].clamp(min=1e-5))
        log_td = torch.log(kin[:, 1].clamp(min=1e-5))
        hill   = kin[:, 2]
        log_a  = torch.log(kin[:, 3].clamp(min=1e-5))
        base   = kin[:, 4]
        tr_n = (log_tr - self._n_log_tau_rise_mean)  / self._n_log_tau_rise_std.clamp(min=1e-6)
        td_n = (log_td - self._n_log_tau_decay_mean) / self._n_log_tau_decay_std.clamp(min=1e-6)
        h_n  = (hill   - self._n_hill_mean)          / self._n_hill_std.clamp(min=1e-6)
        a_n  = (log_a  - self._n_log_amp_mean)       / self._n_log_amp_std.clamp(min=1e-6)
        b_n  = (base   - self._n_baseline_mean)      / self._n_baseline_std.clamp(min=1e-6)
        kin_in = torch.stack([tr_n, td_n, h_n, a_n, b_n], dim=-1)
        return self.kin_mlp(kin_in)

    def _eff_window(self, fps: torch.Tensor = None) -> int:
        """Adaptive attention span (frames) = clamp(round(t_sec*fps), floor, ceil).

        Returns None in fixed mode (the blocks then use their built-in window,
        which here is None -> full attention). fps may be a (B,) tensor; we use
        a representative (median) fps since the band mask is shared across the
        batch (chunks in a batch share target_fps in the trainer)."""
        if self.w_mode != "adaptive" or fps is None:
            return None
        f = float(fps.float().median().item())
        win = int(round(self.t_sec * f))
        win = max(self.w_floor, min(self.w_ceil, win))
        return win

    # ----------------------------------------------------------------------- #
    def _phys_template(self, rate: torch.Tensor, kin: torch.Tensor,
                       fps: torch.Tensor) -> torch.Tensor:
        """Differentiable biexp dF/F template from the per-frame spike rate + the
        CONDITIONED kinetics -- the decay skeleton the additive cond vector fails to
        impose. PER-SAMPLE kernel (each row's own tau_rise/tau_decay/amp/fps) via an
        rFFT causal convolution, so a mixed-indicator batch is honored exactly.

        rate: (B, W) per-frame rate (Hz).  kin: (B, 5) raw [tr_s, td_s, nHill, amp, b0].
        fps:  (B,) frame rate.  Returns (B, W) template dF/F (linear biexp; the Hill
        supralinearity + indicator-specific realism are left to the backbone)."""
        B, W = rate.shape
        K = W                                                          # full-window kernel
        dev, dt_t = rate.device, rate.dtype
        dt = (1.0 / fps.clamp(min=1e-3)).to(dev).view(B, 1)            # (B,1) s/frame
        tr = kin[:, 0].clamp(min=1e-4).view(B, 1)
        td = kin[:, 1].clamp(min=1e-4).view(B, 1)
        amp = kin[:, 3].view(B, 1)
        j = torch.arange(K, device=dev, dtype=dt_t).view(1, K)        # (1,K) frame lag
        jt = j * dt                                                   # (B,K) seconds
        kbar = (1.0 - torch.exp(-jt / tr)) * torch.exp(-jt / td)      # (B,K) biexp shape
        kbar = kbar / kbar.amax(dim=1, keepdim=True).clamp(min=1e-6)  # peak-normalized
        k = amp * kbar                                               # (B,K) kernel scaled to amp/spike
        counts = rate * dt                                          # (B,W) ~ spikes/frame
        # Causal linear conv template = conv(counts, k)[:W] via rFFT: O(B W log W), GPU-native.
        # (An F.conv1d with groups=B=batch hit a pathologically slow grouped-conv kernel at
        #  B=1024/K=384 -> GPU pegged but 0 train steps in 24 min; rFFT is the SSM blocks' idiom.)
        n = W + K
        ft = torch.fft.irfft(torch.fft.rfft(counts, n=n, dim=1)
                             * torch.fft.rfft(k, n=n, dim=1), n=n, dim=1)
        return ft[:, :W]                                            # (B, W) causal biexp template

    # ----------------------------------------------------------------------- #
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Self-configure the conditioning-fidelity modules from a checkpoint even when a
        loader constructed us with DEFAULT flags. The production deploy loader
        (train_binary_sr_fpsW._load_gen) rebuilds DffGeneratorV2 from a config filtered to a
        FIXED key set that omits cond_mode/phys_template, then calls load_state_dict(strict=
        False) -- which would silently DROP the film / tmpl_proj weights and run legacy mode.
        To make the upgraded generator deploy correctly THROUGH THAT UNMODIFIED loader, detect
        film.* / tmpl_proj.* in the incoming state and instantiate the matching modules + flip
        the flags BEFORE delegating, so the trained conditioning is honoured. No-op when the
        modules already exist (our own train/diagnose paths) or absent (legacy ckpts)."""
        ref = next((p for p in self.parameters()), None)
        dev = ref.device if ref is not None else None
        if any(k.startswith("film.") for k in state_dict) and not hasattr(self, "film"):
            self.cond_mode = "film"
            self.film = nn.Linear(self.d, 2 * self.d)
            if dev is not None:
                self.film.to(dev)
        if any(k.startswith("tmpl_proj.") for k in state_dict) and not hasattr(self, "tmpl_proj"):
            self.phys_template = True
            self.tmpl_proj = nn.Linear(1, self.d)
            if dev is not None:
                self.tmpl_proj.to(dev)
        return super().load_state_dict(state_dict, *args, **kwargs)

    # ----------------------------------------------------------------------- #
    def forward(self, spike_input: torch.Tensor, kin: torch.Tensor,
                nu: torch.Tensor = None, fps: torch.Tensor = None,
                cell_type: torch.Tensor = None,
                z: torch.Tensor = None) -> torch.Tensor:
        B = spike_input.size(0)

        # --- reshape sub-frame occupancy to (B, W, m) ------------------------
        if self.sub_frame_m > 1:
            assert spike_input.size(-1) == self.W * self.sub_frame_m, (
                f"expected sub-frame input width {self.W * self.sub_frame_m}, "
                f"got {spike_input.size(-1)}")
            x = spike_input.view(B, self.W, self.sub_frame_m)        # (B, W, m)
        else:
            x = spike_input.unsqueeze(-1)                            # (B, W, 1)

        # --- stochastic per-spike onset-jitter + amplitude injection (synth realism) ---
        # Perturb the fine spike occupancy BEFORE in_proj: scale each bin (amplitude CV) and
        # sub-bin time-shift it (Gaussian, std=inj_jit_ms) via fractional scatter. Stochastic per
        # forward -> the decoder sees the jitter/amp DISTRIBUTION instead of the MSE mean. OFF when 0.
        if (float(self.inj_jit_ms) > 0.0 or float(self.inj_amp_cv) > 0.0) and fps is not None:
            Wd = x.size(1); xf = x.reshape(B, -1); Tf = xf.size(1)    # (B, W*m) fine occupancy
            if float(self.inj_amp_cv) > 0.0:
                xf = xf * (1.0 + self.inj_amp_cv * torch.randn_like(xf)).clamp(min=0.0)
            if float(self.inj_jit_ms) > 0.0:
                sigma = (self.inj_jit_ms / 1000.0) * (fps.reshape(B) * self.sub_frame_m)   # (B,) fine bins
                pos = torch.arange(Tf, device=x.device).float().unsqueeze(0) + sigma.view(B, 1) * torch.randn_like(xf)
                lo = pos.floor().long().clamp(0, Tf - 1); hi = (lo + 1).clamp(0, Tf - 1); frac = pos - pos.floor()
                out = torch.zeros_like(xf)
                out.scatter_add_(1, lo, xf * (1.0 - frac)); out.scatter_add_(1, hi, xf * frac)
                xf = out
            x = xf.reshape(B, Wd, self.sub_frame_m)
        h = self.in_proj(x)                                          # (B, W, d)

        # --- physics-template conditioning channel --------------------------
        # Inject a biexp dF/F skeleton rendered from the conditioned kinetics + the
        # per-frame rate so the backbone REFINES the correct decay instead of inventing
        # kinetics from one additive vector (fixes the kernel-collapse / decay squash).
        # Needs fps; skips cleanly when absent so the legacy call path is unaffected.
        if self.phys_template and fps is not None:
            perframe = x.sum(-1) if self.sub_frame_m > 1 else x.squeeze(-1)   # (B, W)
            tmpl = self._phys_template(perframe, kin, fps)          # (B, W)
            h = h + self.tmpl_proj(tmpl.unsqueeze(-1))             # (B, W, d)

        # --- conditioning: additive (v1) or true-FiLM (cond_mode='film') ----
        cond = self._kin_embed(kin).unsqueeze(1)                    # (B, 1, d)
        if cell_type is not None:
            cond = cond + self.cell_type_emb(cell_type).unsqueeze(1)
        if nu is not None:
            nu_emb = self.nu_proj(torch.log(nu.clamp(min=1e-3)).unsqueeze(-1)).unsqueeze(1)
            cond = cond + nu_emb
        if fps is not None:
            fps_emb = self.fps_proj(torch.log(fps.clamp(min=1e-3)).unsqueeze(-1)).unsqueeze(1)
            cond = cond + fps_emb
        if self.latent_dim > 0 and z is not None:
            cond = cond + self.z_proj(z).unsqueeze(1)
        if self.cond_mode == "film":
            gamma, beta = self.film(cond).chunk(2, dim=-1)         # (B,1,d) each
            h = h * (1.0 + gamma) + beta                           # true FiLM (init -> identity)
        else:
            h = h + cond                                           # legacy additive cond

        # --- positional encoding --------------------------------------------
        if self.time_aware_pos and fps is not None:
            h = h + time_aware_sinusoid(self.W, self.d, fps)
        elif not self.time_aware_pos:
            h = h + self.pos
        h = self.emb_drop(h)

        # --- adaptive attention span ----------------------------------------
        eff_window = self._eff_window(fps)

        # --- backbone --------------------------------------------------------
        if self.backbone == "moe":
            # soft mixture over experts gated on (log nu, log fps)
            lognu = (torch.log(nu.clamp(min=1e-3)) if nu is not None
                     else h.new_zeros(B))
            logfps = (torch.log(fps.clamp(min=1e-3)) if fps is not None
                      else h.new_zeros(B))
            gate_in = torch.stack([lognu, logfps], dim=-1)          # (B, 2)
            w = torch.softmax(self.gate(gate_in), dim=-1)           # (B, n_experts)
            out = h.new_zeros(B, self.W, self.d)
            for e, expert in enumerate(self.experts):
                he = expert(h, eff_window=eff_window, fps=fps)
                out = out + w[:, e].view(B, 1, 1) * he
            h = out
        else:
            h = self.stack(h, eff_window=eff_window, fps=fps)

        h = self.ln(h)
        y = self.out(h)                                             # (B, W, out_m)

        # --- output head: flatten sub-frame channels into the time axis ------
        if self.out_m > 1:
            return y.reshape(B, self.W * self.out_m)               # (B, W*out_m)
        return y.squeeze(-1)                                        # (B, W)
