"""ProtT5-XL-UniRef50 enzyme encoder (CataPro's protein feature basis).

Produces per-residue (1024-d) and pooled embeddings, matching CataPro's preprocessing:
space-separate residues, map U/Z/O/B -> X, mean-pool over residues (drop the final </s> token).
Long sequences (>1000) are head+tail truncated (500+500), as in CataPro.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from ..paths import PROTT5_DIR

PROTT5_LOCAL = PROTT5_DIR
PROTT5_DIM = 1024


def _key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _prep(seq: str, max_res: int = 1000) -> str:
    if len(seq) > max_res:
        seq = seq[:500] + seq[-500:]
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


class ProtT5Encoder:
    def __init__(self, model_dir: Path | str = PROTT5_LOCAL, device: str | None = None,
                 batch_tokens: int = 4096):
        import torch

        self.model_dir = Path(model_dir)
        self.batch_tokens = batch_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tok = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from transformers import T5EncoderModel, T5Tokenizer

        self._tok = T5Tokenizer.from_pretrained(str(self.model_dir), do_lower_case=False,
                                                legacy=True)
        self._model = T5EncoderModel.from_pretrained(str(self.model_dir)).to(self.device).eval()
        if self.device == "cuda":
            self._model.half()
        torch.set_grad_enabled(False)

    def encode_residue(self, seq: str) -> np.ndarray:
        import torch

        self._lazy()
        enc = self._tok([_prep(seq)], return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hs = self._model(**enc).last_hidden_state[0]  # (L+1, 1024) incl </s>
        L = int(enc["attention_mask"][0].sum()) - 1  # drop trailing </s>
        return hs[:L].float().cpu().numpy().astype(np.float16)

    def encode_residue_to_h5(self, sequences: list[str], h5_path: Path | str,
                             log_every: int = 200) -> None:
        import h5py
        import torch

        self._lazy()
        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "a") as h:
            todo = [(s, _key(s)) for s in dict.fromkeys(sequences)]
            todo = [(s, k) for s, k in todo if k not in h]
            todo.sort(key=lambda x: len(x[0]))
            i, done = 0, 0
            while i < len(todo):
                batch, maxlen = [], 0
                while i < len(todo):
                    p = _prep(todo[i][0])
                    L = len(p.split()) + 1
                    if batch and (len(batch) + 1) * max(maxlen, L) > self.batch_tokens:
                        break
                    batch.append((p, todo[i][1])); maxlen = max(maxlen, L); i += 1
                enc = self._tok([p for p, _ in batch], return_tensors="pt", padding=True)
                am = enc["attention_mask"]
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    hs = self._model(**enc).last_hidden_state.half().cpu().numpy()
                for j, (_, key) in enumerate(batch):
                    L = int(am[j].sum()) - 1
                    h.create_dataset(key, data=hs[j, :L])
                done += len(batch)
                if done // log_every > (done - len(batch)) // log_every:
                    print(f"  [prott5] {done}/{len(todo)} sequences", flush=True)

    def encode_pooled_to_h5(self, sequences: list[str], h5_path: Path | str) -> None:
        """mean+max pooled (2048-d) per sequence -> small h5 for GBDT."""
        import h5py
        import torch

        self._lazy()
        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "a") as h:
            todo = [(s, _key(s)) for s in dict.fromkeys(sequences)]
            todo = [(s, k) for s, k in todo if k not in h]
            todo.sort(key=lambda x: len(x[0]))
            i, done = 0, 0
            while i < len(todo):
                batch, maxlen = [], 0
                while i < len(todo):
                    p = _prep(todo[i][0])
                    L = len(p.split()) + 1
                    if batch and (len(batch) + 1) * max(maxlen, L) > self.batch_tokens:
                        break
                    batch.append((p, todo[i][1])); maxlen = max(maxlen, L); i += 1
                enc = self._tok([p for p, _ in batch], return_tensors="pt", padding=True)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                am = enc["attention_mask"].clone().float()
                # zero out the trailing </s> token of each row so pooling uses residues only
                lengths = enc["attention_mask"].sum(1)
                am[torch.arange(am.shape[0], device=self.device), (lengths - 1).long()] = 0.0
                am = am.unsqueeze(-1)
                with torch.no_grad():
                    hs = self._model(**enc).last_hidden_state.float()
                mean = (hs * am).sum(1) / am.sum(1).clamp(min=1)
                mx = hs.masked_fill(am == 0, float("-inf")).max(1).values
                vecs = torch.cat([mean, mx], -1).cpu().numpy().astype(np.float16)
                for j, (_, key) in enumerate(batch):
                    h.create_dataset(key, data=vecs[j])
                done += len(batch)
