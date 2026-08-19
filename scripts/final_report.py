"""EnzyRank final table: leakage-free stacked ensemble combining inherited v1 (EnzyStack)
out-of-fold predictions with EnzyRank's new members (correlation-aligned NN, ESM-C, etc.).
Reports EnzyRank vs CataPro (the SOTA baseline) and vs v1 EnzyStack.

OOF files are searched first in results/oof/ (EnzyRank), then in baseline_v1_oof/ (inherited).
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_dataset  # noqa: E402
from src.eval.metrics import CATAPRO_TARGET, compute_all  # noqa: E402

OOF_DIRS = [ROOT / "results" / "oof", ROOT / "baseline_v1_oof"]
V1 = {"kcat": 0.508, "km": 0.642, "kcat_km": 0.407}  # v1 EnzyStack PCC, for reference


def load_oof(name: str, param: str) -> np.ndarray:
    for d in OOF_DIRS:
        f = d / f"{name}_{param}.npy"
        if f.exists():
            return np.load(f)
    raise FileNotFoundError(f"{name}_{param}.npy in {[str(x) for x in OOF_DIRS]}")


def stacked(param: str, models: list[str]):
    ds = load_dataset(param)
    y, fold = ds.target, ds.fold
    P = np.vstack([load_oof(m, param) for m in models])
    mask = np.isfinite(P).all(0) & np.isfinite(y)
    from scipy.optimize import nnls

    out = np.full_like(y, np.nan)
    for f in np.unique(fold):
        tr, te = mask & (fold != f), mask & (fold == f)
        w, _ = nnls(P[:, tr].T, y[tr])
        w = w / w.sum() if w.sum() > 0 else np.ones(len(models)) / len(models)
        out[te] = P[:, te].T @ w
    return compute_all(y[mask], out[mask])


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "ensemble.yaml"))
    print(f"{'param':<9}{'metric':<6}{'EnzyRank':>9}{'v1':>8}{'CataPro':>9}{'vsCataPro':>10}")
    for p in ["kcat", "km", "kcat_km"]:
        m = stacked(p, cfg["base_models"][p])
        t = CATAPRO_TARGET[p]
        for k in ["PCC", "SCC", "RMSE"]:
            better = (m[k] > t[k]) if k != "RMSE" else (m[k] < t[k])
            margin = (m[k] - t[k]) / t[k] * 100 * (1 if k != "RMSE" else -1)
            v1 = V1[p] if k == "PCC" else ""
            print(f"{p:<9}{k:<6}{m[k]:>9.4f}{str(v1):>8}{t[k]:>9.3f}{margin:>+9.1f}% "
                  f"{'win' if better else '-'}")
        print()


if __name__ == "__main__":
    main()
