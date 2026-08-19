"""Configurable enzyme-substrate interaction model.

Composes the ranked innovations behind config flags so each can be ablated additively:
  * pooling='mean'|'attention'      -> I1
  * n_cross_layers>0                -> I2 (enzyme<->substrate cross-attention)
  * tasks=['kcat','km','kcat_km']   -> I3 (multi-task heads; consistency loss in the trainer)

Inputs per batch:
  prot_tokens (B, Lp, Dp)  per-residue PLM embeddings (e.g. ESM2 1280-d)
  prot_mask   (B, Lp)      1 for valid residues
  sub_tokens  (B, Ls, Ds)  substrate token embeddings (fingerprint-derived or ChemBERTa/MolT5)
  sub_mask    (B, Ls)      1 for valid substrate tokens
  fp          (B, Dfp)     optional flat substrate fingerprint appended at fusion (MACCS/Morgan)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .modules import AttentionPooling, CrossAttentionBlock, build_mlp

TASKS = ["kcat", "km", "kcat_km"]


@dataclass
class ModelConfig:
    protein_dim: int = 1280           # ESM2-650M
    substrate_dim: int = 256          # substrate token dim (after its own encoder)
    fp_dim: int = 0                   # flat fingerprint dim appended at fusion (0 to disable)
    d_model: int = 256
    pooling: str = "attention"        # 'mean' | 'attention'   (I1)
    pool_heads: int = 4
    n_cross_layers: int = 2           # 0 disables cross-attention (I2)
    cross_heads: int = 8
    trunk_dims: list[int] = field(default_factory=lambda: [256, 256])
    head_hidden: int = 128
    dropout: float = 0.1
    tasks: list[str] = field(default_factory=lambda: ["kcat"])  # (I3 when >1)
    mve: bool = False                 # predict (mean, var) per task for noise-robustness


def _masked_mean(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class EnzSubModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.prot_proj = nn.Sequential(nn.Linear(cfg.protein_dim, d), nn.LayerNorm(d))
        self.sub_proj = nn.Sequential(nn.Linear(cfg.substrate_dim, d), nn.LayerNorm(d))

        self.cross = nn.ModuleList(
            [CrossAttentionBlock(d, cfg.cross_heads, cfg.dropout) for _ in range(cfg.n_cross_layers)]
        )

        if cfg.pooling == "attention":
            self.pool_e = AttentionPooling(d, cfg.pool_heads)
            self.pool_s = AttentionPooling(d, cfg.pool_heads)
        else:
            self.pool_e = self.pool_s = None

        fusion_dim = 2 * d + (cfg.fp_dim if cfg.fp_dim else 0)
        self.trunk = build_mlp([fusion_dim, *cfg.trunk_dims], cfg.dropout)
        trunk_out = cfg.trunk_dims[-1]

        out_per_task = 2 if cfg.mve else 1
        self.heads = nn.ModuleDict(
            {t: build_mlp([trunk_out, cfg.head_hidden, out_per_task], cfg.dropout)
             for t in cfg.tasks}
        )

    def _pool(self, tokens, mask, pooler):
        if pooler is None:
            return _masked_mean(tokens, mask)
        pooled, _ = pooler(tokens, mask)
        return pooled

    def forward(self, prot_tokens, prot_mask, sub_tokens, sub_mask, fp=None):
        e = self.prot_proj(prot_tokens)
        s = self.sub_proj(sub_tokens)
        for layer in self.cross:
            e, s = layer(e, prot_mask, s, sub_mask)
        e_vec = self._pool(e, prot_mask, self.pool_e)
        s_vec = self._pool(s, sub_mask, self.pool_s)
        feats = [e_vec, s_vec]
        if self.cfg.fp_dim and fp is not None:
            feats.append(fp)
        z = self.trunk(torch.cat(feats, dim=-1))
        return {t: head(z) for t, head in self.heads.items()}
