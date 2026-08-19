"""Substrate (small-molecule) featurizers based on RDKit — no downloads required.

Provides fixed-length fingerprints used both as standalone GBDT inputs and as an auxiliary
view alongside learned molecular encoders (MolT5 / ChemBERTa).
"""
from __future__ import annotations

import numpy as np

_WARNED: set[str] = set()


def _mol_from_smiles(smiles: str):
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


def maccs_fingerprint(smiles: str) -> np.ndarray:
    """167-bit MACCS keys (matches CataPro's fingerprint block)."""
    from rdkit.Chem import MACCSkeys

    mol = _mol_from_smiles(smiles)
    fp = np.zeros(167, dtype=np.float32)
    if mol is None:
        return fp
    bitvect = MACCSkeys.GenMACCSKeys(mol)
    for i in range(167):
        if bitvect.GetBit(i):
            fp[i] = 1.0
    return fp


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan / ECFP count-less bit fingerprint."""
    from rdkit.Chem import AllChem

    mol = _mol_from_smiles(smiles)
    fp = np.zeros(n_bits, dtype=np.float32)
    if mol is None:
        return fp
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    for i in bitvect.GetOnBits():
        fp[i] = 1.0
    return fp


def featurize_smiles(
    smiles_list: list[str],
    kinds: tuple[str, ...] = ("maccs",),
    morgan_bits: int = 2048,
) -> np.ndarray:
    """Featurize a list of SMILES into a concatenated fingerprint matrix."""
    blocks = []
    for kind in kinds:
        if kind == "maccs":
            blocks.append(np.stack([maccs_fingerprint(s) for s in smiles_list]))
        elif kind == "morgan":
            blocks.append(
                np.stack([morgan_fingerprint(s, n_bits=morgan_bits) for s in smiles_list])
            )
        else:
            raise ValueError(f"unknown fingerprint kind {kind!r}")
    return np.concatenate(blocks, axis=1)
