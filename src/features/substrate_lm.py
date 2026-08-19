"""ChemBERTa per-token substrate encoder (I6).

Produces per-token hidden states for each SMILES so the cross-attention block (I2) can attend to
individual substructures, rather than a single pooled fingerprint token. Cached per unique SMILES.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..paths import CHEMBERTA_DIR

CHEMBERTA_LOCAL = CHEMBERTA_DIR


def _key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


class ChemBERTaEncoder:
    def __init__(self, model_dir: Path | str = CHEMBERTA_LOCAL, device: str | None = None,
                 max_len: int = 128):
        import torch

        self.model_dir = Path(model_dir)
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tok = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModel.from_pretrained(str(self.model_dir)).to(self.device).eval()
        if self.device == "cuda":
            self._model.half()
        torch.set_grad_enabled(False)
        self.dim = self._model.config.hidden_size

    def encode_tokens(self, smiles: str) -> np.ndarray:
        """(T, dim) per-token embeddings (excluding pad); fp16."""
        import torch

        self._lazy()
        enc = self._tok([smiles], return_tensors="pt", truncation=True, max_length=self.max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hs = self._model(**enc).last_hidden_state[0]  # (T, dim)
        return hs.float().cpu().numpy().astype(np.float16)

    def encode_batch_to_dict(self, smiles_list: list[str]) -> dict[str, np.ndarray]:
        return {s: self.encode_tokens(s) for s in dict.fromkeys(smiles_list)}
