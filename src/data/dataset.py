"""CataPro benchmark data loading with fold-exact splits.

The three CataPro datasets share the schema:
    Unnamed: 0, EC, EnzymeType, Organism, Sequence, Substrate, Smiles, UniProtID, <target(s)>, fold

Unit conventions (must match CataPro to compare against its published numbers):
    kcat     : log10(kcat[s^-1])
    km       : log10(Km[mM])          = log10(Km[M]) + 3
    kcat_km  : log10(kcat/Km[s^-1 mM^-1]) = log10(kcat[s^-1]) - log10(Km[M]) - 3

`fold` is a float column 0..9 defining the leakage-controlled (0.4 sequence-similarity
clustered) 10-fold cross-validation split. We never re-split; we use these folds verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA_DIR as DATA_ROOT

DATASET_FILES = {
    "kcat": "kcat-data_0.4simi-10fold.csv",
    "km": "Km-data_0.4simi-10fold.csv",
    "kcat_km": "kcat-over-Km-data_0.4simi-10fold.csv",
}

N_FOLDS = 10


@dataclass
class KineticDataset:
    """Holds a single-parameter dataset with log10 target and fold ids."""

    param: str                 # 'kcat' | 'km' | 'kcat_km'
    df: pd.DataFrame           # original rows (post filtering)
    target: np.ndarray         # log10 target in CataPro units
    fold: np.ndarray           # int fold id 0..9
    sequence: np.ndarray       # enzyme sequences (str)
    smiles: np.ndarray         # substrate SMILES (str)

    def __len__(self) -> int:
        return len(self.df)

    def split(self, test_fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_idx, test_idx) for a held-out fold."""
        test = self.fold == test_fold
        return np.where(~test)[0], np.where(test)[0]


def _compute_target(param: str, df: pd.DataFrame) -> np.ndarray:
    if param == "kcat":
        return np.log10(df["kcat(s^-1)"].to_numpy(dtype=np.float64))
    if param == "km":
        return np.log10(df["Km(M)"].to_numpy(dtype=np.float64)) + 3.0
    if param == "kcat_km":
        log_kcat = np.log10(df["kcat(s^-1)"].to_numpy(dtype=np.float64))
        log_km_mM = np.log10(df["Km(M)"].to_numpy(dtype=np.float64)) + 3.0
        return log_kcat - log_km_mM
    raise ValueError(f"unknown param {param!r}")


def load_dataset(param: str, data_root: Path | str = DATA_ROOT) -> KineticDataset:
    """Load one CataPro dataset with the correct log10 target and folds."""
    path = Path(data_root) / DATASET_FILES[param]
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)

    target = _compute_target(param, df)
    fold = df["fold"].to_numpy(dtype=np.float64).round().astype(int)
    sequence = df["Sequence"].astype(str).to_numpy()
    smiles = df["Smiles"].astype(str).to_numpy()

    finite = np.isfinite(target)
    if not finite.all():
        keep = np.where(finite)[0]
        df = df.iloc[keep].reset_index(drop=True)
        target, fold = target[keep], fold[keep]
        sequence, smiles = sequence[keep], smiles[keep]

    return KineticDataset(param, df, target, fold, sequence, smiles)


def unique_sequences(*datasets: KineticDataset) -> list[str]:
    """Union of unique enzyme sequences across datasets (for feature caching)."""
    seen: set[str] = set()
    for d in datasets:
        seen.update(d.sequence.tolist())
    return sorted(seen)


def unique_smiles(*datasets: KineticDataset) -> list[str]:
    """Union of unique substrate SMILES across datasets (for feature caching)."""
    seen: set[str] = set()
    for d in datasets:
        seen.update(d.smiles.tolist())
    return sorted(seen)
