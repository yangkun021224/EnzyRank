"""Extract ESM-C 600M features (N1). Pooled (for GBDT) always; per-residue (for the NN) optional.

    python scripts/extract_esmc.py --pooled
    python scripts/extract_esmc.py --residue    # ~19 GB; free disk first
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_dataset, unique_sequences  # noqa: E402
from src.features.protein_esmc import ESMCEncoder  # noqa: E402
from src.paths import CACHE_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--residue", action="store_true")
    args = ap.parse_args()

    useq = unique_sequences(*[load_dataset(p) for p in ["kcat", "km", "kcat_km"]])
    print(f"unique sequences: {len(useq)}", flush=True)
    enc = ESMCEncoder()
    if args.pooled:
        t0 = time.time()
        enc.encode_pooled_to_h5(useq, CACHE_DIR / "esmc_pooled.h5")
        print(f"[esmc-pooled] done ({time.time()-t0:.0f}s)")
    if args.residue:
        t0 = time.time()
        enc.encode_residue_to_h5(useq, CACHE_DIR / "esmc_residue.h5")
        print(f"[esmc-residue] done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
