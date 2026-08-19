"""Evaluation metrics for enzyme kinetic parameter prediction.

All metrics are computed in log10 space, matching the CataPro benchmark protocol
(PCC, SCC, RMSE). Targets follow CataPro units: kcat in s^-1, Km in mM, kcat/Km in
s^-1 mM^-1 (see src/data/dataset.py for the unit conventions).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return float(stats.pearsonr(y_true, y_pred)[0])


def scc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    return float(stats.spearmanr(y_true, y_pred)[0])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (auxiliary; also used by the CatPred benchmark)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (auxiliary)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the full metric dict for one prediction set."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "PCC": pcc(y_true, y_pred),
        "SCC": scc(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "n": int(y_true.size),
    }


def aggregate_folds(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and std of each metric across folds (per-fold aggregation)."""
    keys = [k for k in per_fold[0] if k != "n"]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([f[k] for f in per_fold], dtype=np.float64)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out


# Published CataPro numbers to beat (unbiased 0.4-simi 10-fold CV, log10 space).
CATAPRO_TARGET = {
    "kcat":    {"PCC": 0.497, "SCC": 0.495, "RMSE": 1.329},
    "km":      {"PCC": 0.633, "SCC": 0.629, "RMSE": 0.998},
    "kcat_km": {"PCC": 0.413, "SCC": 0.416, "RMSE": 1.619},
}
