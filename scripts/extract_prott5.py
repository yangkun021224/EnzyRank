"""Extract ProtT5 features (pooled always; per-residue optional) and ChemBERTa substrate means.

Usage:
    python scripts/extract_prott5.py --pooled            # ProtT5 mean+max -> cache/prott5_pooled.h5
    python scripts/extract_prott5.py --residue           # ProtT5 per-residue -> cache/prott5_residue.h5 (~17 GB)
    python scripts/extract_prott5.py --chemberta_mean    # ChemBERTa mean -> cache/chemberta_mean.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_dataset, unique_sequences, unique_smiles  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache"


def extract_chemberta_mean(smiles):
    out = CACHE / "chemberta_mean.npz"
    if out.exists():
        print(f"[cbmean] exists {out}"); return
    from src.features.substrate_lm import ChemBERTaEncoder
    cb = ChemBERTaEncoder()
    cb._lazy()
    means = []
    for s in smiles:
        toks = cb.encode_tokens(s).astype(np.float32)
        means.append(toks.mean(0))
    np.savez_compressed(out, smiles=np.array(smiles), mean=np.stack(means))
    print(f"[cbmean] {len(smiles)} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--residue", action="store_true")
    ap.add_argument("--chemberta_mean", action="store_true")
    args = ap.parse_args()

    dss = [load_dataset(p) for p in ["kcat", "km", "kcat_km"]]
    useq = unique_sequences(*dss)
    usmi = unique_smiles(*dss)

    if args.chemberta_mean:
        extract_chemberta_mean(usmi)

    if args.pooled or args.residue:
        from src.features.protein_t5 import ProtT5Encoder
        enc = ProtT5Encoder()
        if args.pooled:
            t0 = time.time()
            enc.encode_pooled_to_h5(useq, CACHE / "prott5_pooled.h5")
            print(f"[prott5-pooled] {len(useq)} seqs ({time.time()-t0:.0f}s)")
        if args.residue:
            t0 = time.time()
            enc.encode_residue_to_h5(useq, CACHE / "prott5_residue.h5")
            print(f"[prott5-residue] done ({time.time()-t0:.0f}s)")
    print("done.")


if __name__ == "__main__":
    main()
