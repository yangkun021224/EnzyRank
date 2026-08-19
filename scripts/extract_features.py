"""Precompute and cache all features once, so every experiment reuses them.

Outputs (under cache/):
  esm2_residue.h5   per-residue ESM2 embeddings, keyed by md5(sequence), fp16
  esm2_pooled.h5    mean+max pooled ESM2 (2560-d), keyed by md5(sequence), fp16
  substrate.npz     MACCS(167), Morgan(2048) for every unique SMILES

Protein extraction runs on GPU; substrate fingerprints run on CPU in parallel.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_dataset, unique_sequences, unique_smiles  # noqa: E402
from src.features.protein import ESM2Encoder  # noqa: E402
from src.features.substrate import featurize_smiles  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache"


def extract_substrate(smiles: list[str]):
    out = CACHE / "substrate.npz"
    if out.exists():
        print(f"[substrate] cache exists: {out}")
        return
    t0 = time.time()
    maccs = featurize_smiles(smiles, kinds=("maccs",))
    morgan = featurize_smiles(smiles, kinds=("morgan",), morgan_bits=2048)
    np.savez_compressed(out, smiles=np.array(smiles), maccs=maccs.astype(np.float32),
                        morgan=morgan.astype(np.float32))
    print(f"[substrate] {len(smiles)} SMILES -> {out} ({time.time()-t0:.1f}s)")


def extract_protein(sequences: list[str], residue: bool):
    enc = ESM2Encoder()
    t0 = time.time()
    enc.encode_pooled(sequences, pooling="meanmax")  # writes cache/esm2/pooled_meanmax.h5
    print(f"[esm2-pooled] done ({time.time()-t0:.1f}s)")
    if residue:  # only needed for the ESM2-based NN ablation, not the final pipeline
        t0 = time.time()
        enc.encode_residue_to_h5(sequences, CACHE / "esm2_residue.h5")
        print(f"[esm2-residue] done ({time.time()-t0:.1f}s)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm2_residue", action="store_true",
                    help="also extract ESM2 per-residue (only for the ESM2 NN ablation)")
    args = ap.parse_args()

    dss = [load_dataset(p) for p in ["kcat", "km", "kcat_km"]]
    useq = unique_sequences(*dss)
    usmi = unique_smiles(*dss)
    print(f"unique sequences={len(useq)} unique smiles={len(usmi)}")
    extract_substrate(usmi)
    extract_protein(useq, residue=args.esm2_residue)
    print("feature extraction complete.")


if __name__ == "__main__":
    main()
