"""Feature providers that map raw sequences / SMILES to model-ready arrays, cached per unique
entity in RAM (we have >600 GB). Enzyme features are per-residue (for pooling + cross-attention);
substrate features are token sets (for cross-attention) plus an optional flat fingerprint.
"""
from __future__ import annotations

import numpy as np

from .protein import ESM2Encoder
from .substrate import featurize_smiles


class ProteinResidueProvider:
    """Per-residue ESM2 embeddings.

    Preferred path: back onto a precomputed h5 cache (see scripts/extract_features.py); `prime`
    loads the needed sequences into a RAM dict once (we have >600 GB RAM) so every experiment
    reuses them without recompute. Falls back to on-the-fly encoding for missing keys.
    """

    def __init__(self, encoder: ESM2Encoder | None = None, h5_path=None):
        import hashlib

        self.encoder = encoder
        self.h5_path = h5_path
        self.cache: dict[str, np.ndarray] = {}
        self.dim = 1280
        self._key = lambda s: hashlib.md5(s.encode()).hexdigest()

    def _ensure_encoder(self):
        if self.encoder is None:
            self.encoder = ESM2Encoder()

    def prime(self, sequences: list[str]):
        need = [s for s in dict.fromkeys(sequences) if s not in self.cache]
        if not need:
            return self
        if self.h5_path is not None:
            import h5py

            with h5py.File(self.h5_path, "r") as h:
                for s in need:
                    k = self._key(s)
                    if k in h:
                        self.cache[s] = h[k][:]
        missing = [s for s in need if s not in self.cache]
        if missing:
            self._ensure_encoder()
            for s in missing:
                self.cache[s] = self.encoder.encode_residue(s)
        return self

    def get(self, seq: str) -> np.ndarray:
        if seq not in self.cache:
            self.prime([seq])
        return self.cache[seq]


class SubstrateProvider:
    """Substrate token features + optional flat fingerprint.

    mode='fp'       : one token = [MACCS ⊕ Morgan]; token_dim = 167 + morgan_bits.
    mode='chemberta': per-token ChemBERTa hidden states (added once weights are available).
    A flat MACCS(+Morgan) fingerprint is always available for fusion-time concatenation.
    """

    def __init__(self, mode: str = "fp", morgan_bits: int = 2048, flat_kinds=("maccs",),
                 chemberta_encoder=None):
        self.mode = mode
        self.morgan_bits = morgan_bits
        self.flat_kinds = flat_kinds
        self.tok_cache: dict[str, np.ndarray] = {}
        self.flat_cache: dict[str, np.ndarray] = {}
        if mode == "fp":
            self.token_dim = 167 + morgan_bits
        elif mode == "chemberta":
            if chemberta_encoder is None:
                from .substrate_lm import ChemBERTaEncoder
                chemberta_encoder = ChemBERTaEncoder()
            self.cb = chemberta_encoder
            self.cb._lazy()
            self.token_dim = self.cb.dim
        else:
            raise NotImplementedError(f"substrate mode {mode!r} not supported")
        self.flat_dim = (167 if "maccs" in flat_kinds else 0) + \
                        (morgan_bits if "morgan" in flat_kinds else 0)

    def prime(self, smiles: list[str]):
        uniq = [s for s in dict.fromkeys(smiles) if s not in self.tok_cache]
        if not uniq:
            return self
        flat = featurize_smiles(uniq, kinds=self.flat_kinds, morgan_bits=self.morgan_bits)
        for s, f in zip(uniq, flat):
            self.flat_cache[s] = f.astype(np.float32)
        if self.mode == "fp":
            tok = featurize_smiles(uniq, kinds=("maccs", "morgan"), morgan_bits=self.morgan_bits)
            for s, t in zip(uniq, tok):
                self.tok_cache[s] = t[None, :].astype(np.float32)  # (1, token_dim)
        elif self.mode == "chemberta":
            for s in uniq:
                self.tok_cache[s] = self.cb.encode_tokens(s).astype(np.float32)  # (T, dim)
        return self

    def get_tokens(self, smiles: str) -> np.ndarray:
        if smiles not in self.tok_cache:
            self.prime([smiles])
        return self.tok_cache[smiles]

    def get_flat(self, smiles: str) -> np.ndarray:
        if smiles not in self.flat_cache:
            self.prime([smiles])
        return self.flat_cache[smiles]
