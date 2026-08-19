"""End-to-end fine-tuning of an ESM2 backbone with the interaction head.

Unlike the frozen-feature path, the PLM is trained (per fold, on train folds only) so it can learn
task-specific enzyme representations. Kept deliberately regularized (dropout, differential LR,
optional layer freezing, OOD-fold early stopping) to fight overfitting on the noisy 0.4-simi splits.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .modules import AttentionPooling, build_mlp


class SeqTokenDataset(torch.utils.data.Dataset):
    """Yields tokenized enzyme sequences + substrate fingerprint + target."""

    def __init__(self, sequences, fps, targets, tokenizer, max_len=1024):
        self.seqs = sequences
        self.fps = fps.astype(np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        enc = self.tok(self.seqs[i], truncation=True, max_length=self.max_len)
        return {"input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "fp": torch.from_numpy(self.fps[i]),
                "target": torch.tensor(self.targets[i])}


def collate_ft(batch, pad_id=1):
    L = max(x["input_ids"].shape[0] for x in batch)
    B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    am = torch.zeros(B, L, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        ids[i, :n] = x["input_ids"]
        am[i, :n] = x["attention_mask"]
    return {"input_ids": ids, "attention_mask": am,
            "fp": torch.stack([x["fp"] for x in batch]),
            "target": torch.stack([x["target"] for x in batch])}


class FinetuneModel(nn.Module):
    def __init__(self, esm_dir, fp_dim, d_model=256, dropout=0.3, freeze_layers=0):
        super().__init__()
        from transformers import EsmModel

        self.esm = EsmModel.from_pretrained(str(esm_dir))
        h = self.esm.config.hidden_size
        if freeze_layers > 0:
            for p in self.esm.embeddings.parameters():
                p.requires_grad = False
            for layer in self.esm.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
        self.pool = AttentionPooling(h, n_heads=4)
        self.head = build_mlp([h + fp_dim, d_model, d_model, 1], dropout)

    def forward(self, input_ids, attention_mask, fp):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # drop cls/eos via attention mask on interior tokens
        mask = attention_mask.clone().float()
        mask[:, 0] = 0.0
        idx = attention_mask.sum(1) - 1
        mask[torch.arange(mask.shape[0], device=mask.device), idx.long()] = 0.0
        pooled, _ = self.pool(out, mask)
        return self.head(torch.cat([pooled, fp], dim=-1)).squeeze(-1)
