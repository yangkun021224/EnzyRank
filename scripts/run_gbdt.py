"""I4 — run GBDT (LightGBM/XGBoost/CatBoost) on pooled features via fold-exact CV.

Usage:
    python scripts/run_gbdt.py --param kcat --kind lightgbm
    python scripts/run_gbdt.py --param km --kind lightgbm --blocks esm2 maccs morgan
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_dataset  # noqa: E402
from src.eval.cv import report, run_cv  # noqa: E402
from src.models.gbdt_model import GBDTFoldModel, TabularCV  # noqa: E402
from src.utils.common import save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True, choices=["kcat", "km", "kcat_km"])
    ap.add_argument("--kind", default="lightgbm",
                    choices=["lightgbm", "xgboost", "catboost", "ridge", "extratrees"])
    ap.add_argument("--blocks", nargs="+", default=["esm2", "maccs"])
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--n_estimators", type=int, default=1200)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--tag", default=None, help="name suffix to distinguish feature sets")
    ap.add_argument("--target_encode", nargs="*", default=None,
                    help="categorical cols to fold-safe target-encode, e.g. EC Organism")
    ap.add_argument("--retrieval", action="store_true",
                    help="N4: add fold-safe kNN retrieval features (enzyme+substrate embedding)")
    args = ap.parse_args()
    name = f"I4_{args.kind}" + (f"_{args.tag}" if args.tag else "")

    set_seed(42)
    ds = load_dataset(args.param)
    tab = TabularCV(ds, blocks=tuple(args.blocks))
    print(f"=== GBDT {args.kind} param={args.param} blocks={args.blocks} X={tab.X.shape} ===")

    retrieval_emb = None
    if args.retrieval:
        from src.features.tabular import build_matrix
        retrieval_emb = build_matrix(list(ds.sequence), list(ds.smiles),
                                     blocks=("prott5", "cbmean"))  # enzyme + substrate embedding

    def factory():
        return GBDTFoldModel(tab, kind=args.kind, threads=args.threads,
                             n_estimators=args.n_estimators,
                             params={"num_leaves": args.num_leaves},
                             te_cols=tuple(args.target_encode) if args.target_encode else None,
                             retrieval_emb=retrieval_emb)

    res = run_cv(ds, factory, verbose=True)
    print(report(res))
    save_json({"config": name, "param": args.param, "blocks": args.blocks,
               "pooled": res["pooled"], "per_fold_agg": res["per_fold_agg"]},
              f"results/{name}_{args.param}.json")
    import numpy as np
    Path("results/oof").mkdir(parents=True, exist_ok=True)
    np.save(f"results/oof/{name}_{args.param}.npy", res["oof_pred"])


if __name__ == "__main__":
    main()
