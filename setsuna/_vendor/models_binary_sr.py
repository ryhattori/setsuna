"""Learned BINARY / COUNT SUPER-RESOLUTION head on the two-conformer backbone.

Brings the binary-prior super-resolution idea (Sarangi, Hattori, Komiyama & Pal 2023, IEEE T-SP --
binarity >> sparsity for sub-grid identifiability) into the deep model. The production Hybrid mean-pools
the conformer features to a CENTER-BIN scalar (frame-grid rate). Here we tap the PRE-POOL (B, W, d)
feature sequence and attach a PER-POSITION, m-upsampled head: each W-frame input window (dF/F at
target_fps) predicts spikes on an (m*W)-bin grid at m*target_fps. Two branches (bulk + corner) gated on
(log nu, log fps) exactly like TwoConformerHybrid.

HEAD TYPES:
  - "binary"  : sigmoid(logit) = P(spike in fine bin); train with BCEWithLogits + a binarization penalty.
                The faithful binary-SR head. PRIMARY for excitatory/pyramidal (moderate rate, <=1 spike/bin).
  - "count"   : softplus(logit) = lambda = expected count in fine bin; train with Poisson NLL. Handles
                MULTIPLE spikes per bin under calcium saturation -> the graceful-degradation head for
                high-rate cells (PV/FSI). At high m for low-rate cells lambda->{0,1} so it ~recovers binary.

Either way the module emits RAW LOGITS (B, W*m); .decode() maps them to probability (binary) or lambda
(count). ISOLATED from production: imports the conformer building blocks but does NOT modify
SmallConformerCornerSpecialist, so the PRODH pipeline is untouched.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_small_conformer_corner import ConvBlock, LocalMHABlock, FFNBlock, time_aware_sinusoid
from models_dff_generator_v2 import (DiagSSMBlock, MambaBlock, S4DBlock, HGRN2Block,  # state-space backbones
                                      GatedDeltaNetBlock, Mamba2Block, HybridBlock, MoEMambaBlock,
                                      Mamba3LiteBlock, Mamba3Block, GatedDeltaNet3Block)  # (all fps-aware, bidirectional, (d,dropout) drop-ins) for the SR decoder bulk branch

# Sequence-mixer backbones selectable for a branch's attention slot. Each shares the (d,dropout)
# ctor + forward(x, eff_window, fps) and consumes fps for frame-rate-consistent (1/fps) discretization.
# Add a backbone (HGRN-2, GatedDeltaNet, Mamba-2, ...) by extending THIS dict -- the per-layer dispatch
# and the fps-passing isinstance check below both key off it, so nothing else needs touching.
_SEQ_BLOCKS = {"ssm": DiagSSMBlock, "s4d": S4DBlock, "mamba": MambaBlock, "hgrn2": HGRN2Block,
               "gdn": GatedDeltaNetBlock, "mamba2": Mamba2Block, "hybrid": HybridBlock, "moe": MoEMambaBlock,
               "mamba3lite": Mamba3LiteBlock, "mamba3": Mamba3Block, "gdn3": GatedDeltaNet3Block}
_FPS_BLOCK_TYPES = tuple(dict.fromkeys(_SEQ_BLOCKS.values()))


class SRBranch(nn.Module):
    """Conformer feature extractor (same embedding+blocks as SmallConformerCornerSpecialist) + a
    per-position m-upsampling head. forward -> (B, W*m) logits at m*target_fps."""

    def __init__(self, num_indicators: int, num_cell_types: int = 4, m: int = 4,
                 d: int = 24, n_heads: int = 2, ffn: int = 48, layers: int = 4,
                 conv_kernel: int = 15, W: int = 128, dropout: float = 0.0,
                 time_aware_pos: bool = True, local_attn_window: int = None,
                 head_type: str = "binary", dual: bool = False, ssm_backbone: bool = False,
                 mamba_backbone: bool = False, seq_kind: str = None):
        super().__init__()
        # sequence-mixer backbone for the per-layer attention slot. explicit seq_kind wins; else
        # derive from the legacy bools (keeps old --bulk_ssm/--bulk_mamba checkpoints reconstructable).
        self.seq_kind = seq_kind or ("mamba" if mamba_backbone else "ssm" if ssm_backbone else "attn")
        self.W = W; self.d = d; self.m = m; self.time_aware_pos = time_aware_pos
        self.head_type = head_type   # "binary" (sigmoid/BCE) or "count" (softplus/Poisson)
        self.dual = dual             # dual head: emit (count_logits, binary_logits) for detect-then-count fusion
        self.in_proj = nn.Linear(1, d)
        self.indicator_emb = nn.Embedding(num_indicators, d)
        self.cell_type_emb = nn.Embedding(num_cell_types, d)
        self.nu_proj = nn.Linear(1, d)
        self.fps_proj = nn.Linear(1, d)
        nn.init.trunc_normal_(self.indicator_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.cell_type_emb.weight, std=0.02)
        if not time_aware_pos:
            self.pos = nn.Parameter(torch.zeros(1, W, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(ConvBlock(d, conv_kernel, dropout))
            if self.seq_kind in _SEQ_BLOCKS:   # state-space mixer (calcium-kinetics bias) in place of local attention
                self.blocks.append(_SEQ_BLOCKS[self.seq_kind](d, dropout))
            else:                              # "attn" = conformer local multi-head attention (baseline)
                self.blocks.append(LocalMHABlock(d, n_heads, W=W, local_window=local_attn_window, dropout=dropout))
            self.blocks.append(FFNBlock(d, ffn, dropout))
        self.ln = nn.LayerNorm(d)
        # per-position upsampling head: each frame's feature -> m (or 2m if dual: count|binary) sub-bin logits
        self.sr_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, (2 * m) if dual else m))

    def features(self, x, indicator_id, nu, fps, cell_type, eff_window=None):
        """Same conditioning/encoder as the production branch, but returns the (B, L, d) sequence
        BEFORE the mean-pool (which the production head discards). eff_window (path b) overrides the
        attention band half-width for this call (fps-adaptive span on a fixed-length input)."""
        B = x.size(0)
        h = self.in_proj(x.unsqueeze(-1))                                  # (B, W, d)
        ind = self.indicator_emb(indicator_id).unsqueeze(1)
        cell = self.cell_type_emb(cell_type if cell_type is not None
                                  else torch.zeros(B, dtype=torch.long, device=x.device)).unsqueeze(1)
        if nu is not None:
            ind = ind + self.nu_proj(torch.log(nu.clamp(min=1e-3)).unsqueeze(-1)).unsqueeze(1)
        if fps is not None:
            ind = ind + self.fps_proj(torch.log(fps.clamp(min=1e-3)).unsqueeze(-1)).unsqueeze(1)
        h = h + ind + cell
        if self.time_aware_pos and fps is not None:
            h = h + time_aware_sinusoid(h.size(1), self.d, fps)   # h.size(1) not self.W -> length-agnostic (fps-scaled W)
        elif not self.time_aware_pos:
            h = h + self.pos
        h = self.emb_drop(h)
        for blk in self.blocks:
            if isinstance(blk, LocalMHABlock):
                h = blk(h, eff_window=eff_window)
            elif isinstance(blk, _FPS_BLOCK_TYPES):
                h = blk(h, eff_window=eff_window, fps=fps)   # state-space blocks use fps for per-step calcium decay
            else:
                h = blk(h)
        return self.ln(h)                                                  # (B, L, d)

    def forward(self, x, indicator_id, nu=None, fps=None, cell_type=None, eff_window=None):
        h = self.features(x, indicator_id, nu, fps, cell_type, eff_window)  # (B, L, d)
        logits = self.sr_head(h)                                           # (B, L, m) or (B, L, 2m)
        if self.dual:
            c = logits[..., :self.m].reshape(h.size(0), h.size(1) * self.m)
            b = logits[..., self.m:].reshape(h.size(0), h.size(1) * self.m)
            return c, b                                                    # (count_logits, binary_logits)
        return logits.reshape(h.size(0), h.size(1) * self.m)               # (B, L*m), L=h.size(1) (fps-scaled W)


class CountAdapter(nn.Module):
    """Count-specific residual refinement CONDITIONED on indicator (its OWN FiLM embedding). Lets the count/rate
    path re-calibrate per-indicator WITHOUT touching the shared indicator_emb -> the binary/timing stream stays
    frozen (f1 preserved). Zero-init -> identity at start (== the init checkpoint exactly)."""
    def __init__(self, num_indicators, h=16, k=7):
        super().__init__()
        self.c1 = nn.Conv1d(1, h, k, padding=k // 2)
        self.c2 = nn.Conv1d(h, h, k, padding=k // 2)
        self.c3 = nn.Conv1d(h, 1, k, padding=k // 2)
        self.film = nn.Embedding(num_indicators, 2 * h)               # per-indicator (gamma, beta) FiLM on hidden features
        nn.init.zeros_(self.c3.weight); nn.init.zeros_(self.c3.bias)  # residual = 0 at start
        nn.init.zeros_(self.film.weight)                             # gamma=beta=0 -> identity FiLM at start

    def forward(self, count_logits, indicator_id):
        h = F.gelu(self.c1(count_logits.unsqueeze(1)))               # (B, h, L*m)
        g, b = self.film(indicator_id).chunk(2, dim=-1)              # (B, h), (B, h)
        h = h * (1.0 + g.unsqueeze(-1)) + b.unsqueeze(-1)           # FiLM (identity at init)
        h = F.gelu(self.c2(h))
        return self.c3(h).squeeze(1)                                 # (B, L*m) residual


class BinarySRHybrid(nn.Module):
    """Two SR branches (bulk + corner) gated on (log nu, log fps), mirroring TwoConformerHybrid.
    Output = per-position LOGITS (B, W*m) at m*target_fps. .decode() -> probability (binary head) or
    expected count lambda (count head). Gate applied in logit space (convex combo), matching the
    production output-gating style."""

    def __init__(self, num_indicators: int, num_cell_types: int = 4, m: int = 4,
                 head_type: str = "binary", W: int = 128, dropout: float = 0.0,
                 bulk_d: int = 24, bulk_n_heads: int = 2, bulk_ffn: int = 48,
                 bulk_layers: int = 4, bulk_conv_kernel: int = 15,
                 bulk_local_attn_window: int = None, bulk_time_aware_pos: bool = True,
                 corner_d: int = 20, corner_n_heads: int = 2, corner_ffn: int = 32,
                 corner_layers: int = 3, corner_conv_kernel: int = 15,
                 corner_local_attn_window: int = 52, corner_time_aware_pos: bool = True,
                 n_corner: int = 1, gate_rich: bool = False, dual: bool = False, bulk_ssm: bool = False,
                 bulk_mamba: bool = False, bulk_backbone: str = None, corner_backbone: str = None,
                 gate_fps_bias: float = 0.0, gate_fps_ref: float = 20.0, count_adapter: bool = False):
        super().__init__()
        self.W = W; self.m = m; self.head_type = head_type
        self.n_corner = n_corner; self.gate_rich = gate_rich; self.dual = dual
        self.bulk_backbone = bulk_backbone or ("mamba" if bulk_mamba else "ssm" if bulk_ssm else "attn")
        self.corner_backbone = corner_backbone or "attn"   # default conformer corner; set = bulk for a FULL-backbone model
        # GATE WIDENING: structural prior that engages the corner MORE as fps drops below gate_fps_ref. The
        # learned gate optimizes detection/rate loss, which under-weights the corner at low fps/low nu; this
        # prior lets the (connectivity-carrying) corner contribute at low fps regardless. 0 = off (default).
        self.gate_fps_bias = float(gate_fps_bias)
        self.gate_log_fps_ref = float(torch.log(torch.tensor(float(gate_fps_ref))))
        self.bulk = SRBranch(num_indicators, num_cell_types, m, bulk_d, bulk_n_heads, bulk_ffn,
                             bulk_layers, bulk_conv_kernel, W, dropout, bulk_time_aware_pos,
                             bulk_local_attn_window, head_type=head_type, dual=dual,
                             seq_kind=self.bulk_backbone)
        def _mk_corner():
            return SRBranch(num_indicators, num_cell_types, m, corner_d, corner_n_heads, corner_ffn,
                            corner_layers, corner_conv_kernel, W, dropout, corner_time_aware_pos,
                            corner_local_attn_window, head_type=head_type, dual=dual,
                            seq_kind=self.corner_backbone)
        if n_corner == 1:
            self.corner = _mk_corner()                                       # backward-compatible single corner
        else:
            self.corners = nn.ModuleList([_mk_corner() for _ in range(n_corner)])   # MoE: K independent corners
        _gin = 2 + (8 if gate_rich else 0)
        _gout = 1 if n_corner == 1 else (n_corner + 1)                        # sigmoid(2-way) vs softmax(bulk+K)
        if gate_rich:
            self.gate_ind = nn.Embedding(num_indicators, 4); self.gate_cell = nn.Embedding(num_cell_types, 4)
            nn.init.trunc_normal_(self.gate_ind.weight, std=0.02); nn.init.trunc_normal_(self.gate_cell.weight, std=0.02)
        self.gate_mlp = nn.Sequential(nn.Linear(_gin, 16), nn.GELU(), nn.Linear(16, _gout))
        if n_corner == 1:
            with torch.no_grad():
                self.gate_mlp[-1].bias.fill_(-1.4)   # init w_corner ~ 0.2
        self.count_adapter = None
        if count_adapter and dual:                   # count-specific refinement adapter, CONDITIONED on indicator -> count path gets its OWN 7f conditioning (shared indicator_emb + binary stream untouched -> f1 frozen)
            self.count_adapter = CountAdapter(num_indicators)

    def compute_gate(self, nu, fps, indicator_id=None, cell_type=None):
        gin = torch.stack([torch.log(nu.clamp(min=1e-3)), torch.log(fps.clamp(min=1e-3))], dim=-1)
        if self.gate_rich:
            ci = cell_type if cell_type is not None else torch.zeros_like(indicator_id)
            gin = torch.cat([gin, self.gate_ind(indicator_id), self.gate_cell(ci)], dim=-1)
        out = self.gate_mlp(gin)
        if self.gate_fps_bias > 0 and fps is not None and self.n_corner == 1:   # low-fps corner-engagement prior
            out = out + self.gate_fps_bias * F.relu(self.gate_log_fps_ref - torch.log(fps.clamp(min=1e-3))).unsqueeze(-1)
        return torch.sigmoid(out).squeeze(-1) if self.n_corner == 1 else torch.softmax(out, dim=-1)  # (B,) or (B,K+1)

    def forward(self, x, indicator_id, nu=None, fps=None, cell_type=None, route_mode="soft", eff_window=None, return_gate=False, corner_eff_window=None):
        _cew = corner_eff_window if corner_eff_window is not None else eff_window   # SEPARATE corner attn window (local-bulk / global-corner split); default = bulk's eff_window (unchanged behaviour)
        if self.n_corner == 0:                    # pure-bulk specialist (n_corner==0): no corner branch to mix (dual -> bulk returns (count,bin) tuple directly)
            _o = self.bulk(x, indicator_id, nu, fps, cell_type, eff_window)
            return (_o, torch.zeros(x.shape[0], device=x.device)) if return_gate else _o
        if self.n_corner == 1:
            if self.dual:
                bc, bb = self.bulk(x, indicator_id, nu, fps, cell_type, eff_window)
                cc, cb = self.corner(x, indicator_id, nu, fps, cell_type, _cew)
                w = self.compute_gate(nu, fps, indicator_id, cell_type).unsqueeze(-1)
                _count = (1.0 - w) * bc + w * cc; _bin = (1.0 - w) * bb + w * cb
                if self.count_adapter is not None:                            # CONDITIONED residual count refinement; binary stream (_bin) untouched -> f1 stays frozen
                    _count = _count + self.count_adapter(_count, indicator_id)
                return _count, _bin                                          # (count_logits, binary_logits)
            if route_mode == "hard_corner":
                return self.corner(x, indicator_id, nu, fps, cell_type, _cew)
            if route_mode == "hard_bulk":
                return self.bulk(x, indicator_id, nu, fps, cell_type, eff_window)
            bulk_logits = self.bulk(x, indicator_id, nu, fps, cell_type, eff_window)       # (B, L*m)
            corner_logits = self.corner(x, indicator_id, nu, fps, cell_type, _cew)   # (B, L*m)
            w = self.compute_gate(nu, fps, indicator_id, cell_type).unsqueeze(-1)          # (B, 1)
            out = (1.0 - w) * bulk_logits + w * corner_logits                              # (B, W*m)
            return (out, w.squeeze(-1)) if return_gate else out
        # MoE: bulk + K independent corners, softmax gate over (log nu, log fps[, ind, cell])
        experts = [self.bulk(x, indicator_id, nu, fps, cell_type, eff_window)]
        for c in self.corners:
            experts.append(c(x, indicator_id, nu, fps, cell_type, _cew))
        gw = self.compute_gate(nu, fps, indicator_id, cell_type)                           # (B, K+1)
        out = (torch.stack(experts, dim=-1) * gw.unsqueeze(1)).sum(-1)                      # (B, W*m)
        return (out, gw) if return_gate else out

    def decode(self, logits):
        """Raw logits -> per-bin spike PROBABILITY (binary head) or expected COUNT lambda (count head)."""
        return torch.sigmoid(logits) if self.head_type == "binary" else F.softplus(logits)
