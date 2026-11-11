"""Small conformer corner specialist (single-branch, no MoE).

Two variants controlled by `local_attn_window`:
  - None        : full self-attention over W frames (original conformer)
  - int (e.g. 52): each position attends only to ±(local_attn_window//2) neighbors

Includes FiLM conditioning on (indicator, nu, fps, cell_type) and time-aware
positional encoding — same conditioning as IndicatorConformerNuFpsCell so it can
be a drop-in corner specialist.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float = 0.0):
        super().__init__()
        pad = (kernel - 1) // 2
        self.ln = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, kernel, padding=pad, groups=d)
        self.pw = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = h.transpose(1, 2)
        h = self.dw(h)
        h = F.gelu(h)
        h = self.pw(h)
        h = h.transpose(1, 2)
        return x + self.drop(h)


class LocalMHABlock(nn.Module):
    """MHA with optional local-window mask. If local_window is None, behaves as full attn."""

    def __init__(self, d: int, n_heads: int, W: int, local_window: int = None,
                 dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.W = W
        self.local_window = local_window
        self._mask_cache = {}                       # (L, window) -> additive band mask, built per ACTUAL length

    def _band_mask(self, L, window, device):
        """Additive band mask (-inf outside +-(window//2)) at the actual sequence length L.
        Length-agnostic (supports fps-scaled variable W); cached per (L, window)."""
        if window is None:
            return None
        key = (L, window)
        m = self._mask_cache.get(key)
        if m is None or m.device != device:
            half = window // 2
            idx = torch.arange(L, device=device)
            inw = (idx[None, :] >= idx[:, None] - half) & (idx[None, :] <= idx[:, None] + half)
            m = torch.where(inw, 0.0, float('-inf'))
            self._mask_cache[key] = m
        return m

    def forward(self, x, eff_window: int = None):
        # eff_window overrides the band half-width for THIS call (fps-adaptive span = path b);
        # default falls back to the fixed self.local_window (unchanged behaviour).
        window = eff_window if eff_window is not None else self.local_window
        h = self.ln(x)
        a, _ = self.attn(h, h, h, attn_mask=self._band_mask(x.size(1), window, x.device),
                         need_weights=False)
        return x + self.drop(a)


class FFNBlock(nn.Module):
    def __init__(self, d: int, ffn: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.lin1 = nn.Linear(d, ffn)
        self.lin2 = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.lin1(h))
        h = self.lin2(h)
        return x + self.drop(h)


def time_aware_sinusoid(W: int, d: int, fps: torch.Tensor) -> torch.Tensor:
    """Sinusoidal PE scaled by 1/fps so distances reflect real time.

    Returns (B, W, d) where each row's frequency scaling depends on the per-sample fps.
    """
    B = fps.shape[0]
    device = fps.device
    pos = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)  # (B, W)
    pos_t = pos / fps.unsqueeze(1).clamp(min=1e-3)   # time in seconds
    half = d // 2
    div = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -(math.log(10000.0) / half))
    arg = pos_t.unsqueeze(-1) * div.view(1, 1, -1)
    pe = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (B, W, d)
    if pe.shape[-1] < d:
        pe = F.pad(pe, (0, d - pe.shape[-1]))
    return pe


class SmallConformerCornerSpecialist(nn.Module):
    """Single-branch small conformer with FiLM conditioning.

    No gate, no hardroute branching — pure single-model corner specialist.
    Designed for direct comparison vs CASCADE-CNN at the corner (and against
    the full-attention conformer specialist).
    """

    def __init__(self, num_indicators: int, num_cell_types: int = 4,
                 d: int = 20, n_heads: int = 2, ffn: int = 32,
                 layers: int = 3, conv_kernel: int = 15,
                 W: int = 128, dropout: float = 0.0,
                 time_aware_pos: bool = True,
                 local_attn_window: int = None):
        super().__init__()
        self.W = W
        self.d = d
        self.time_aware_pos = time_aware_pos
        self.in_proj = nn.Linear(1, d)
        self.indicator_emb = nn.Embedding(num_indicators, d)
        self.cell_type_emb = nn.Embedding(num_cell_types, d)
        self.nu_proj = nn.Linear(1, d)
        self.fps_proj = nn.Linear(1, d)
        nn.init.trunc_normal_(self.indicator_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.cell_type_emb.weight, std=0.02)
        if not time_aware_pos:
            self.pos = nn.Parameter(torch.zeros(1, W, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(ConvBlock(d, conv_kernel, dropout))
            self.blocks.append(LocalMHABlock(d, n_heads, W=W,
                                              local_window=local_attn_window,
                                              dropout=dropout))
            self.blocks.append(FFNBlock(d, ffn, dropout))
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, indicator_id: torch.Tensor,
                nu: torch.Tensor = None, fps: torch.Tensor = None,
                cell_type: torch.Tensor = None) -> torch.Tensor:
        B = x.size(0)
        h = self.in_proj(x.unsqueeze(-1))  # (B, W, d)
        # FiLM-like additive conditioning
        ind = self.indicator_emb(indicator_id).unsqueeze(1)  # (B, 1, d)
        cell = self.cell_type_emb(cell_type if cell_type is not None
                                   else torch.zeros(B, dtype=torch.long, device=x.device)
                                   ).unsqueeze(1)
        if nu is not None:
            nu_emb = self.nu_proj(torch.log(nu.clamp(min=1e-3)).unsqueeze(-1)).unsqueeze(1)
            ind = ind + nu_emb
        if fps is not None:
            fps_emb = self.fps_proj(torch.log(fps.clamp(min=1e-3)).unsqueeze(-1)).unsqueeze(1)
            ind = ind + fps_emb
        h = h + ind + cell
        # Positional encoding
        if self.time_aware_pos and fps is not None:
            h = h + time_aware_sinusoid(self.W, self.d, fps)
        elif not self.time_aware_pos:
            h = h + self.pos
        h = self.emb_drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h).mean(dim=1)
        return self.out(h).squeeze(-1)
