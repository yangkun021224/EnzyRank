"""Export the headline EnzyRank vs CataPro results as spreadsheet tables (CSV + XLSX),
under both aggregation conventions (pooled OOF and per-fold mean). Mirrors final_report.py but
writes results/final_table.csv and results/final_table.xlsx instead of markdown/text.

Usage:  python scripts/export_final_table.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_dataset  # noqa: E402
from src.eval.metrics import CATAPRO_TARGET, compute_all  # noqa: E402

OOF_DIRS = [ROOT / "results" / "oof", ROOT / "baseline_v1_oof"]
PARAMS = ["kcat", "km", "kcat_km"]
V1 = {"kcat": 0.508, "km": 0.642, "kcat_km": 0.407}  # v1 EnzyStack PCC (reference)


def load_oof(name, param):
    for d in OOF_DIRS:
        f = d / f"{name}_{param}.npy"
        if f.exists():
            return np.load(f)
    raise FileNotFoundError(f"{name}_{param}")


def stacked(param, members):
    ds = load_dataset(param)
    y, fold = ds.target, ds.fold
    P = np.vstack([load_oof(m, param) for m in members])
    mask = np.isfinite(P).all(0) & np.isfinite(y)
    out = np.full_like(y, np.nan)
    perfold = {"PCC": [], "SCC": [], "RMSE": []}
    for f in np.unique(fold):
        tr, te = mask & (fold != f), mask & (fold == f)
        w, _ = nnls(P[:, tr].T, y[tr])
        w = w / w.sum() if w.sum() > 0 else np.ones(len(members)) / len(members)
        out[te] = P[:, te].T @ w
        mf = compute_all(y[te], out[te])
        for k in perfold:
            perfold[k].append(mf[k])
    pooled = compute_all(y[mask], out[mask])
    pfmean = {k: float(np.mean(v)) for k, v in perfold.items()}
    return pooled, pfmean


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "ensemble.yaml"))["base_models"]
    rows = []
    for p in PARAMS:
        pooled, pf = stacked(p, cfg[p])
        t = CATAPRO_TARGET[p]
        for k in ["PCC", "SCC", "RMSE"]:
            sign = 1 if k != "RMSE" else -1
            margin = (pooled[k] - t[k]) / t[k] * 100 * sign
            rows.append({
                "parameter": p, "metric": k,
                "EnzyRank_pooled": round(pooled[k], 4),
                "EnzyRank_perfold_mean": round(pf[k], 4),
                "v1_EnzyStack_PCC": (V1[p] if k == "PCC" else ""),
                "CataPro": t[k],
                "rel_margin_pooled_%": round(margin, 1),
                "beats_CataPro": bool((pooled[k] > t[k]) if k != "RMSE" else (pooled[k] < t[k])),
            })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "final_table.csv", index=False)
    with pd.ExcelWriter(ROOT / "results" / "final_table.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="EnzyRank_vs_CataPro", index=False)
    print("wrote results/final_table.csv and results/final_table.xlsx")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
