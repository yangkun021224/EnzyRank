"""GPU-resident feature store + index batcher.

Frozen ESM2 features never change during training, so we upload each unique sequence's
per-residue tensor to the GPU once and batch by index — eliminating all per-batch host->device
transfer and DataLoader overhead (the dominant cost when features are precomputed). Batches are
assembled and padded directly on the GPU.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def _key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


class GpuFeatureStore:
    def __init__(self, prot_provider, sub_provider, sequences, smiles, use_fp: bool,
                 device: str = "cuda"):
        self.device = device
        self.use_fp = use_fp
        # unique enzyme per-residue tensors on GPU (fp16)
        self.prot: dict[str, torch.Tensor] = {}
        for s in dict.fromkeys(sequences):
            arr = prot_provider.get(s)  # (L,1280) fp16 np
            self.prot[_key(s)] = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        # unique substrate token + flat fingerprint tensors on GPU (fp32)
        self.sub: dict[str, torch.Tensor] = {}
        self.fp: dict[str, torch.Tensor] = {}
        for s in dict.fromkeys(smiles):
            self.sub[s] = torch.from_numpy(sub_provider.get_tokens(s).astype(np.float32)).to(device)
            if use_fp:
                self.fp[s] = torch.from_numpy(sub_provider.get_flat(s).astype(np.float32)).to(device)
        self.prot_dim = next(iter(self.prot.values())).shape[-1]
        self.sub_dim = next(iter(self.sub.values())).shape[-1]

    def get_prot(self, seq):
        return self.prot[_key(seq)]


class IndexBatcher:
    """Iterable of GPU batches for a set of row indices."""

    def __init__(self, store: GpuFeatureStore, sequences, smiles, targets, indices,
                 batch_size: int, shuffle: bool, seed: int = 0):
        self.store = store
        self.sequences = sequences
        self.smiles = smiles
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32)).to(store.device)
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def _lengths(self):
        if getattr(self, "_len_cache", None) is None:
            self._len_cache = np.array(
                [self.store.get_prot(self.sequences[r]).shape[0] for r in self.indices])
        return self._len_cache

    def _batch_order(self):
        """Batch construction.

        Training (shuffle=True): length-bucketed with jitter to minimize padding waste, then the
        batch order is shuffled each epoch to preserve SGD stochasticity.

        Eval (shuffle=False): keep the ORIGINAL index order so predictions returned by the trainer
        align exactly with the caller's indices (critical — predictions are assigned back by
        position). Padding waste on a single eval pass is negligible.
        """
        if not self.shuffle:
            return [self.indices[i:i + self.batch_size]
                    for i in range(0, len(self.indices), self.batch_size)]
        lens = self._lengths()
        noise = self._rng.random(len(lens)) * 16
        order = np.argsort(lens + noise - 8, kind="stable")  # positions into self.indices
        rows_sorted = self.indices[order]
        batches = [rows_sorted[i:i + self.batch_size]
                   for i in range(0, len(rows_sorted), self.batch_size)]
        self._rng.shuffle(batches)
        return batches

    def __iter__(self):
        from torch.nn.utils.rnn import pad_sequence

        dev = self.store.device
        for rows in self._batch_order():
            seqs = [self.sequences[r] for r in rows]
            smis = [self.smiles[r] for r in rows]
            prot_list = [self.store.get_prot(s) for s in seqs]           # fp16 GPU tensors
            lens = torch.tensor([t.shape[0] for t in prot_list], device=dev)
            prot = pad_sequence(prot_list, batch_first=True)             # (B, Lp, D) fp16
            Lp = prot.shape[1]
            prot_mask = (torch.arange(Lp, device=dev)[None, :] < lens[:, None]).float()
            # substrate tokens: single fp token (Ls=1) -> stack; variable (ChemBERTa) -> pad
            sub_list = [self.store.sub[s] for s in smis]
            if all(t.shape[0] == 1 for t in sub_list):
                sub = torch.stack(sub_list)                              # (B, 1, Ds)
                sub_mask = torch.ones(sub.shape[0], 1, device=dev)
            else:
                slens = torch.tensor([t.shape[0] for t in sub_list], device=dev)
                sub = pad_sequence(sub_list, batch_first=True)           # (B, Ls, Ds)
                sub_mask = (torch.arange(sub.shape[1], device=dev)[None, :] < slens[:, None]).float()
            batch = {"prot_tokens": prot, "prot_mask": prot_mask,
                     "sub_tokens": sub, "sub_mask": sub_mask,
                     "targets": self.targets[torch.as_tensor(rows, device=dev)]}
            if self.store.use_fp:
                batch["fp"] = torch.stack([self.store.fp[s] for s in smis])
            yield batch
