"""Fold-exact 10-fold cross-validation runner and reporting.

Two evaluation views are always reported:
  * pooled out-of-fold (OOF): concatenate all held-out fold predictions, compute one metric.
    This is the single-number view directly comparable to CataPro's published rows.
  * per-fold mean +/- std: robustness view.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from ..data.dataset import KineticDataset, N_FOLDS
from .metrics import CATAPRO_TARGET, aggregate_folds, compute_all


class FoldModel(Protocol):
    """Anything that can fit on a train split and predict a test split."""

    def fit_predict(
        self,
        ds: KineticDataset,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fold: int,
    ) -> np.ndarray:
        """Return predictions for `test_idx` (log10 units)."""
        ...


def run_cv(
    ds: KineticDataset,
    model_factory: Callable[[], FoldModel],
    folds: list[int] | None = None,
    verbose: bool = True,
) -> dict:
    """Run fold-exact CV and return pooled + per-fold metrics."""
    folds = list(range(N_FOLDS)) if folds is None else folds
    oof_pred = np.full(len(ds), np.nan, dtype=np.float64)
    per_fold: list[dict[str, float]] = []

    for f in folds:
        train_idx, test_idx = ds.split(f)
        model = model_factory()
        pred = np.asarray(model.fit_predict(ds, train_idx, test_idx, f), dtype=np.float64)
        oof_pred[test_idx] = pred
        m = compute_all(ds.target[test_idx], pred)
        per_fold.append(m)
        if verbose:
            print(f"  fold {f}: PCC={m['PCC']:.4f} SCC={m['SCC']:.4f} "
                  f"RMSE={m['RMSE']:.4f} (n={m['n']})")

    mask = ~np.isnan(oof_pred)
    pooled = compute_all(ds.target[mask], oof_pred[mask])
    agg = aggregate_folds(per_fold)
    return {
        "param": ds.param,
        "pooled": pooled,
        "per_fold_agg": agg,
        "per_fold": per_fold,
        "oof_pred": oof_pred,
    }


def report(result: dict) -> str:
    """Pretty summary comparing pooled metrics to the CataPro target row."""
    param = result["param"]
    p = result["pooled"]
    agg = result["per_fold_agg"]
    tgt = CATAPRO_TARGET.get(param, {})
    lines = [f"=== {param} ===",
             f"{'metric':<6}{'ours(pooled)':>14}{'ours(fold mean±std)':>24}"
             f"{'CataPro':>10}{'beat?':>7}"]
    for k in ["PCC", "SCC", "RMSE"]:
        ours = p[k]
        fm, fs = agg[k]["mean"], agg[k]["std"]
        t = tgt.get(k, float("nan"))
        if k == "RMSE":
            beat = ours < t
        else:
            beat = ours > t
        flag = "YES" if beat else "no"
        lines.append(f"{k:<6}{ours:>14.4f}{fm:>18.4f}±{fs:<5.4f}{t:>10.3f}{flag:>7}")
    return "\n".join(lines)
