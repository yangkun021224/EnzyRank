"""Protein (enzyme) featurizers from pretrained language models.

Default encoder: ESM2-650M (facebook/esm2_t33_650M_UR50D), loaded from a local dir. Produces
per-residue embeddings (1280-d) and pooled vectors. Per-unique-sequence caching keeps repeated
extraction cheap. Cross-attention innovations consume the per-residue tensors; simple baselines
consume mean/attention-pooled vectors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..paths import CACHE_DIR, ESM2_650M_DIR

ESM2_LOCAL = ESM2_650M_DIR
ESM2_DIM = 1280
ESM2_MAX_RESIDUES = 1022  # ESM2 context 1024 incl. <cls>/<eos>


def _seq_key(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def _truncate(seq: str, max_res: int = ESM2_MAX_RESIDUES) -> str:
    """Head+tail truncation for over-long sequences (keeps termini, matches common practice)."""
    if len(seq) <= max_res:
        return seq
    half = max_res // 2
    return seq[:half] + seq[-half:]


class ESM2Encoder:
    """Thin wrapper over HuggingFace ESM2 with pooled/per-residue extraction + disk cache."""

    def __init__(
        self,
        model_dir: Path | str = ESM2_LOCAL,
        device: str | None = None,
        cache_dir: Path | str = CACHE_DIR / "esm2",
        batch_tokens: int = 8192,
    ):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_tokens = batch_tokens
        self._model = None
        self._tok = None
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, EsmModel

        self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = EsmModel.from_pretrained(str(self.model_dir)).to(self.device).eval()
        self._model.half() if self.device == "cuda" else None
        torch.set_grad_enabled(False)

    def _pooled_cache_path(self) -> Path:
        return self.cache_dir / "pooled_meanmax.h5"

    def encode_pooled(self, sequences: list[str], pooling: str = "mean") -> np.ndarray:
        """Return (N, ESM2_DIM) pooled embeddings, cached per unique sequence.

        pooling: 'mean' | 'max' | 'meanmax' (concat -> 2*DIM).
        """
        import h5py

        uniq = list(dict.fromkeys(sequences))
        keys = [_seq_key(s) for s in uniq]
        store: dict[str, np.ndarray] = {}
        path = self._pooled_cache_path()
        if path.exists():
            with h5py.File(path, "r") as h:
                for k in keys:
                    if k in h:
                        store[k] = h[k][:]
        todo = [(s, k) for s, k in zip(uniq, keys) if k not in store]
        if todo:
            self._lazy_load()
            self._encode_missing_pooled(todo, store, path)
        # assemble mean/max/meanmax from stored [mean(0:D), max(D:2D)]
        out = []
        for s in sequences:
            v = store[_seq_key(s)]
            if pooling == "mean":
                out.append(v[:ESM2_DIM])
            elif pooling == "max":
                out.append(v[ESM2_DIM:])
            else:
                out.append(v)
        return np.stack(out).astype(np.float32)

    def _encode_missing_pooled(self, todo, store, path):
        import h5py
        import torch

        # simple token-budget batching by length
        todo_sorted = sorted(todo, key=lambda x: len(x[0]))
        i = 0
        with h5py.File(path, "a") as h:
            while i < len(todo_sorted):
                batch = []
                maxlen = len(_truncate(todo_sorted[i][0])) + 2
                while i < len(todo_sorted):
                    seq = _truncate(todo_sorted[i][0])
                    L = len(seq) + 2
                    if batch and (len(batch) + 1) * max(maxlen, L) > self.batch_tokens:
                        break
                    batch.append(todo_sorted[i]); maxlen = max(maxlen, L); i += 1
                seqs = [_truncate(s) for s, _ in batch]
                enc = self._tok(seqs, return_tensors="pt", padding=True, truncation=True,
                                max_length=ESM2_MAX_RESIDUES + 2)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    hs = self._model(**enc).last_hidden_state  # (B, L, D)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                # drop special tokens by using attention mask minus cls/eos handled crudely:
                summed = (hs * mask).sum(1)
                cnt = mask.sum(1).clamp(min=1)
                mean = (summed / cnt).float().cpu().numpy()
                masked = hs.masked_fill(mask == 0, float("-inf"))
                mx = masked.max(1).values.float().cpu().numpy()
                for (s, k), mv, xv in zip(batch, mean, mx):
                    vec = np.concatenate([mv, xv]).astype(np.float16)
                    store[k] = vec
                    if k not in h:
                        h.create_dataset(k, data=vec)

    def encode_residue(self, sequence: str) -> np.ndarray:
        """Return (L, ESM2_DIM) per-residue embeddings for one sequence (no special tokens)."""
        import torch

        self._lazy_load()
        seq = _truncate(sequence)
        enc = self._tok([seq], return_tensors="pt", truncation=True,
                        max_length=ESM2_MAX_RESIDUES + 2)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hs = self._model(**enc).last_hidden_state[0]  # (L+2, D)
        return hs[1:-1].float().cpu().numpy().astype(np.float16)

    def encode_residue_to_h5(self, sequences: list[str], h5_path: Path | str,
                             log_every: int = 200) -> None:
        """Batch-extract per-residue embeddings for unique sequences into an h5 store.

        Keyed by md5(sequence); each dataset is (L, ESM2_DIM) fp16 with special tokens removed.
        Resumable: sequences already present are skipped.
        """
        import h5py
        import torch

        self._lazy_load()
        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "a") as h:
            todo = [(s, _seq_key(s)) for s in dict.fromkeys(sequences)]
            todo = [(s, k) for s, k in todo if k not in h]
            todo.sort(key=lambda x: len(x[0]))
            i, done = 0, 0
            while i < len(todo):
                batch, maxlen = [], 0
                while i < len(todo):
                    seq = _truncate(todo[i][0])
                    L = len(seq) + 2
                    if batch and (len(batch) + 1) * max(maxlen, L) > self.batch_tokens:
                        break
                    batch.append((seq, todo[i][1])); maxlen = max(maxlen, L); i += 1
                seqs = [s for s, _ in batch]
                enc = self._tok(seqs, return_tensors="pt", padding=True, truncation=True,
                                max_length=ESM2_MAX_RESIDUES + 2)
                lens = enc["attention_mask"].sum(1).tolist()
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    hs = self._model(**enc).last_hidden_state.half().cpu().numpy()
                for j, (_, key) in enumerate(batch):
                    L = int(lens[j])
                    h.create_dataset(key, data=hs[j, 1:L - 1], compression=None)
                done += len(batch)
                if done // log_every > (done - len(batch)) // log_every:
                    print(f"  [esm2] {done}/{len(todo)} sequences", flush=True)
