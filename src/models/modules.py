"""Reusable neural building blocks for the enzyme-substrate models.

Contains the components behind the ranked innovations:
  * AttentionPooling  -> I1 (attention / binding-site-weighted pooling over PLM residues)
  * CrossAttentionBlock -> I2 (enzyme <-> substrate interaction)
  * build_mlp         -> shared trunks / heads
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(dims: list[int], dropout: float = 0.1, act: str = "gelu",
              norm: bool = True) -> nn.Sequential:
    """MLP with LayerNorm + activation + dropout between hidden layers (last layer linear)."""
    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(acts[act]())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class AttentionPooling(nn.Module):
    """I1 — learned attention pooling over a token set (KinForm-style weighted pooling).

    A small scorer network assigns a weight to every residue/token; the pooled vector is the
    weighted sum. Multi-head queries capture several complementary "views" (e.g. active site vs
    global fold). Falls back gracefully with masking for padded tokens.
    """

    def __init__(self, d_model: int, n_heads: int = 4, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_model
        self.n_heads = n_heads
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden), nn.Tanh(), nn.Linear(hidden, n_heads)
        )
        self.proj = nn.Linear(d_model * n_heads, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (B, L, D); mask: (B, L) with 1 for valid tokens
        scores = self.scorer(x)  # (B, L, H)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
        attn = torch.softmax(scores, dim=1)  # over L
        # weighted sum per head -> (B, H, D) -> (B, H*D)
        pooled = torch.einsum("blh,bld->bhd", attn, x).reshape(x.size(0), -1)
        return self.proj(pooled), attn


class CrossAttentionBlock(nn.Module):
    """I2 — one bidirectional enzyme<->substrate cross-attention layer.

    Enzyme tokens attend to substrate tokens and vice versa, each followed by a residual FFN.
    Stacking several of these lets binding specificity emerge from pairwise interaction rather
    than from concatenation of independently-pooled vectors.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.e2s = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.s2e = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_e1, self.ln_e2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ln_s1, self.ln_s2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff_e = build_mlp([d_model, d_model * 2, d_model], dropout, norm=False)
        self.ff_s = build_mlp([d_model, d_model * 2, d_model], dropout, norm=False)

    @staticmethod
    def _kpm(mask: torch.Tensor | None):
        # MultiheadAttention wants key_padding_mask with True = ignore
        return None if mask is None else ~mask.bool()

    def forward(self, enz, enz_mask, sub, sub_mask):
        e = self.ln_e1(enz)
        s = self.ln_s1(sub)
        enz2, _ = self.s2e(e, s, s, key_padding_mask=self._kpm(sub_mask))
        sub2, _ = self.e2s(s, e, e, key_padding_mask=self._kpm(enz_mask))
        enz = enz + enz2
        sub = sub + sub2
        enz = enz + self.ff_e(self.ln_e2(enz))
        sub = sub + self.ff_s(self.ln_s2(sub))
        return enz, sub
