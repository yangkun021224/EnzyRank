"""Blend out-of-fold predictions from several models for one parameter.

Reports each model's pooled metrics, an equal-weight blend (leakage-free), and a
non-negative-weight stacked blend fit on OOF (reported separately; mild optimism).
Usage:
    python scripts/ensemble.py --param kcat --models I2_crossattn I4_lightgbm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_dataset  # noqa: E402
from src.eval.metrics import CATAPRO_TARGET, compute_all  # noqa: E402

OOF = Path(__file__).resolve().parents[1] / "results" / "oof"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True, choices=["kcat", "km", "kcat_km"])
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    ds = load_dataset(args.param)
    y = ds.target
    preds, names = [], []
    for m in args.models:
        f = OOF / f"{m}_{args.param}.npy"
        if not f.exists():
            print(f"skip missing {f}")
            continue
        preds.append(np.load(f))
        names.append(m)
    P = np.vstack(preds)  # (M, N)
    mask = np.isfinite(P).all(0) & np.isfinite(y)
    Pm, ym = P[:, mask], y[mask]

    t = CATAPRO_TARGET[args.param]
    print(f"=== {args.param} (CataPro PCC {t['PCC']} RMSE {t['RMSE']}) ===")
    for name, p in zip(names, Pm):
        m = compute_all(ym, p)
        print(f"  {name:<22} PCC={m['PCC']:.4f} SCC={m['SCC']:.4f} RMSE={m['RMSE']:.4f}")

    eq = Pm.mean(0)
    me = compute_all(ym, eq)
    print(f"  {'BLEND(equal)':<22} PCC={me['PCC']:.4f} SCC={me['SCC']:.4f} RMSE={me['RMSE']:.4f}")

    from scipy.optimize import nnls

    # (a) global nnls weights fit on all OOF -> optimistic (leakage), reported with *
    w, _ = nnls(Pm.T, ym)
    if w.sum() > 0:
        w = w / w.sum()
        ms = compute_all(ym, Pm.T @ w)
        print(f"  {'BLEND(nnls*)':<22} PCC={ms['PCC']:.4f} SCC={ms['SCC']:.4f} RMSE={ms['RMSE']:.4f}"
              f"  w={dict(zip(names, np.round(w,3)))}")

    # (b) leakage-free STACK: for each held-out fold, fit nnls weights on the other folds' OOF
    fold = ds.fold[mask]
    stacked = np.zeros_like(ym)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        wf, _ = nnls(Pm[:, tr].T, ym[tr])
        wf = wf / wf.sum() if wf.sum() > 0 else np.ones(len(names)) / len(names)
        stacked[te] = Pm[:, te].T @ wf
    mst = compute_all(ym, stacked)
    print(f"  {'BLEND(stack)':<22} PCC={mst['PCC']:.4f} SCC={mst['SCC']:.4f} "
          f"RMSE={mst['RMSE']:.4f}  <- leakage-free")


if __name__ == "__main__":
    main()
