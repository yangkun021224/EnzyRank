"""Fold-safe smoothed target encoding of categorical enzyme metadata (EC, Organism).

The 0.4-similarity split separates enzymes by SEQUENCE similarity, not by EC — so a test enzyme
typically shares its EC class with training enzymes. The mean (log) kinetic value of a class is a
strong, legitimate prior that CataPro does not use. Encodings are computed on the TRAIN folds only
and applied to the test fold (leakage-free). Within train we use out-of-fold encoding so the
downstream model does not overfit the encoded column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ec_levels(ec: pd.Series) -> dict[str, pd.Series]:
    s = ec.astype(str)
    parts = s.str.split(".")
    return {
        "ec1": parts.str[0],
        "ec12": parts.str[:2].str.join("."),
        "ec123": parts.str[:3].str.join("."),
        "ec": s,
    }


def _smooth_map(cat: pd.Series, y: np.ndarray, idx: np.ndarray, smoothing: float):
    """Return {category: encoded_value} from rows `idx`, smoothed toward the global mean."""
    g = pd.DataFrame({"c": cat.iloc[idx].values, "y": y[idx]})
    agg = g.groupby("c")["y"].agg(["sum", "count"])
    global_mean = float(y[idx].mean())
    enc = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)
    return enc.to_dict(), global_mean


def target_encode_columns(df: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray,
                          test_idx: np.ndarray, cols=("EC", "Organism"),
                          smoothing: float = 10.0, n_inner: int = 5, seed: int = 0):
    """Return (te_train, te_test) feature matrices.

    - test rows: encoded with maps fit on ALL train rows (leakage-free wrt test fold).
    - train rows: encoded out-of-fold within train (n_inner splits) to avoid target leakage
      into the downstream model.
    """
    cat_series: dict[str, pd.Series] = {}
    for col in cols:
        if col == "EC":
            cat_series.update(_ec_levels(df["EC"]))
        else:
            cat_series[col.lower()] = df[col].astype(str)

    names = list(cat_series)
    te_train = np.zeros((len(train_idx), len(names)), dtype=np.float32)
    te_test = np.zeros((len(test_idx), len(names)), dtype=np.float32)

    for j, name in enumerate(names):
        cat = cat_series[name]
        # test: fit on all train
        enc_all, gmean = _smooth_map(cat, y, train_idx, smoothing)
        te_test[:, j] = [enc_all.get(c, gmean) for c in cat.iloc[test_idx].values]
        # train: out-of-fold within train
        rng = np.random.default_rng(seed + j)
        perm = rng.permutation(len(train_idx))
        folds = np.array_split(perm, n_inner)
        for f in folds:
            inner_val = train_idx[f]
            inner_tr = train_idx[np.setdiff1d(np.arange(len(train_idx)), f, assume_unique=False)]
            enc, gm = _smooth_map(cat, y, inner_tr, smoothing)
            te_train[f, j] = [enc.get(c, gm) for c in cat.iloc[inner_val].values]

    return te_train, te_test, names
