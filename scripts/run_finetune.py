"""Run end-to-end ESM2 fine-tuning through fold-exact CV.

Usage:
    python scripts/run_finetune.py --param kcat --esm esm2_t12_35M_UR50D [--folds 0 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_dataset  # noqa: E402
from src.eval.cv import report, run_cv  # noqa: E402
from src.features.substrate import featurize_smiles  # noqa: E402
from src.models.finetune_fold import FinetuneFoldModel  # noqa: E402
from src.utils.common import save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True, choices=["kcat", "km", "kcat_km"])
    ap.add_argument("--esm", default="esm2_t12_35M_UR50D")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--freeze_layers", type=int, default=0)
    ap.add_argument("--lr_esm", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--tag", default="ft35")
    args = ap.parse_args()

    set_seed(42)
    from transformers import AutoTokenizer

    from src.paths import PRETRAINED_DIR
    esm_dir = str(PRETRAINED_DIR / args.esm)
    tok = AutoTokenizer.from_pretrained(esm_dir)

    ds = load_dataset(args.param)
    uniq = sorted(set(ds.smiles.tolist()))
    fpmap = {s: v for s, v in zip(uniq, featurize_smiles(uniq, kinds=("maccs", "morgan")))}
    fp = np.stack([fpmap[s] for s in ds.smiles]).astype(np.float32)

    def factory():
        return FinetuneFoldModel(esm_dir, fp, tok, epochs=args.epochs,
                                 batch_size=args.batch_size, freeze_layers=args.freeze_layers,
                                 lr_esm=args.lr_esm, dropout=args.dropout)

    print(f"=== FINETUNE {args.esm} param={args.param} folds={args.folds or 'all'} ===", flush=True)
    res = run_cv(ds, factory, folds=args.folds, verbose=True)
    print(report(res))
    name = f"I5_{args.tag}"
    save_json({"config": name, "param": args.param,
               "pooled": res["pooled"], "per_fold_agg": res["per_fold_agg"]},
              f"results/{name}_{args.param}.json")
    Path("results/oof").mkdir(parents=True, exist_ok=True)
    np.save(f"results/oof/{name}_{args.param}.npy", res["oof_pred"])


if __name__ == "__main__":
    main()
