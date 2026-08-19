"""Torch Dataset + padding collate feeding the interaction model.

A sample yields per-residue enzyme tokens, substrate tokens, an optional flat fingerprint, and a
target vector aligned to the model's task list (NaN where a label is absent — enables multi-task).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..features.providers import ProteinResidueProvider, SubstrateProvider


class FeatureDataset(Dataset):
    def __init__(self, sequences, smiles, targets, prot: ProteinResidueProvider,
                 sub: SubstrateProvider, use_fp: bool):
        # targets: (N, T) float array, NaN for missing
        self.seqs = sequences
        self.smiles = smiles
        self.targets = np.asarray(targets, dtype=np.float32)
        self.prot = prot
        self.sub = sub
        self.use_fp = use_fp

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        pt = self.prot.get(self.seqs[i]).astype(np.float32)   # (Lp, Dp)
        st = self.sub.get_tokens(self.smiles[i])              # (Ls, Ds)
        item = {
            "prot_tokens": torch.from_numpy(pt),
            "sub_tokens": torch.from_numpy(st),
            "targets": torch.from_numpy(self.targets[i]),
        }
        if self.use_fp:
            item["fp"] = torch.from_numpy(self.sub.get_flat(self.smiles[i]))
        return item


def collate(batch):
    B = len(batch)
    Lp = max(x["prot_tokens"].shape[0] for x in batch)
    Ls = max(x["sub_tokens"].shape[0] for x in batch)
    Dp = batch[0]["prot_tokens"].shape[1]
    Ds = batch[0]["sub_tokens"].shape[1]

    prot = torch.zeros(B, Lp, Dp)
    prot_mask = torch.zeros(B, Lp)
    sub = torch.zeros(B, Ls, Ds)
    sub_mask = torch.zeros(B, Ls)
    targets = torch.stack([x["targets"] for x in batch])

    for i, x in enumerate(batch):
        lp = x["prot_tokens"].shape[0]
        ls = x["sub_tokens"].shape[0]
        prot[i, :lp] = x["prot_tokens"]
        prot_mask[i, :lp] = 1.0
        sub[i, :ls] = x["sub_tokens"]
        sub_mask[i, :ls] = 1.0

    out = {"prot_tokens": prot, "prot_mask": prot_mask,
           "sub_tokens": sub, "sub_mask": sub_mask, "targets": targets}
    if "fp" in batch[0]:
        out["fp"] = torch.stack([x["fp"] for x in batch])
    return out
