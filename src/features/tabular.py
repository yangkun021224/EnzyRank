"""Tabular feature assembly for GBDT / linear baselines.

Concatenates pooled enzyme features (ESM2 mean+max, 2560-d) with substrate fingerprints
(MACCS 167, Morgan 2048). Built once per dataset and sliced per fold.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..paths import CACHE_DIR as CACHE


def _key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _load_h5_dict(h5_path: Path | str) -> dict:
    import h5py

    out = {}
    with h5py.File(h5_path, "r") as h:
        for k in h:
            out[k] = h[k][:].astype(np.float32)
    return out


def load_pooled_esm2(h5_path: Path | str = CACHE / "esm2" / "pooled_meanmax.h5") -> dict:
    return _load_h5_dict(h5_path)


def load_pooled_prott5(h5_path: Path | str = CACHE / "prott5_pooled.h5") -> dict:
    return _load_h5_dict(h5_path)


def load_pooled_esmc(h5_path: Path | str = CACHE / "esmc_pooled.h5") -> dict:
    return _load_h5_dict(h5_path)


def load_chemberta_mean(npz_path: Path | str = CACHE / "chemberta_mean.npz") -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {s: v.astype(np.float32) for s, v in zip(d["smiles"], d["mean"])}


def load_substrate(npz_path: Path | str = CACHE / "substrate.npz") -> dict:
    d = np.load(npz_path, allow_pickle=True)
    smi = list(d["smiles"])
    maccs = d["maccs"].astype(np.float32)
    morgan = d["morgan"].astype(np.float32)
    return {s: (maccs[i], morgan[i]) for i, s in enumerate(smi)}


def build_matrix(sequences, smiles, blocks=("esm2", "maccs", "morgan")) -> np.ndarray:
    """Assemble the (N, D) feature matrix for the requested blocks.

    Protein blocks: 'esm2' (mean+max 2560), 'prott5' (mean+max 2048).
    Substrate blocks: 'maccs' (167), 'morgan' (2048), 'cbmean' (ChemBERTa mean, 384).
    """
    esm = load_pooled_esm2() if "esm2" in blocks else None
    t5 = load_pooled_prott5() if "prott5" in blocks else None
    esmc = load_pooled_esmc() if "esmc" in blocks else None
    sub = load_substrate() if ("maccs" in blocks or "morgan" in blocks) else None
    cb = load_chemberta_mean() if "cbmean" in blocks else None
    rows = []
    for seq, smi in zip(sequences, smiles):
        parts = []
        if esm is not None:
            parts.append(esm[_key(seq)])
        if t5 is not None:
            parts.append(t5[_key(seq)])
        if esmc is not None:
            parts.append(esmc[_key(seq)])
        if sub is not None:
            maccs, morgan = sub[smi]
            if "maccs" in blocks:
                parts.append(maccs)
            if "morgan" in blocks:
                parts.append(morgan)
        if cb is not None:
            parts.append(cb[smi])
        rows.append(np.concatenate(parts))
    return np.stack(rows).astype(np.float32)
