"""Fold-safe retrieval-augmented features (N4).

For each enzyme-substrate query, retrieve its k nearest neighbours in the TRAIN folds (by cosine
similarity in a joint enzyme+substrate embedding space) and summarise their known kinetic values.
This turns "known measurements of similar pairs" into a strong prior. Leakage-free: test-fold rows
retrieve only from training rows; within train, retrieval is done out-of-fold.
"""
from __future__ import annotations

import numpy as np


def _normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-8, None)


def _neighbour_features(Xq: np.ndarray, Xr: np.ndarray, yr: np.ndarray, k: int) -> np.ndarray:
    """For each query row, similarity-weighted stats of its k nearest reference rows.

    Returns columns: [weighted_mean_y, mean_top_sim, top1_sim, std_y]. Computed in similarity blocks
    to bound memory.
    """
    feats = np.zeros((Xq.shape[0], 4), dtype=np.float32)
    block = 2048
    for s in range(0, Xq.shape[0], block):
        sim = Xq[s:s + block] @ Xr.T                       # (b, R) cosine (inputs pre-normalised)
        kk = min(k, Xr.shape[0])
        idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        row = np.arange(sim.shape[0])[:, None]
        topsim = sim[row, idx]                             # (b, k)
        topy = yr[idx]                                     # (b, k)
        w = np.clip(topsim, 0, None) + 1e-6
        feats[s:s + block, 0] = (w * topy).sum(1) / w.sum(1)
        feats[s:s + block, 1] = topsim.mean(1)
        feats[s:s + block, 2] = topsim.max(1)
        feats[s:s + block, 3] = topy.std(1)
    return feats


def retrieval_features(embeddings: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                       test_idx: np.ndarray, k: int = 16, n_inner: int = 5,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (feat_train, feat_test); each row has 4 retrieval features (see _neighbour_features)."""
    E = _normalise(embeddings.astype(np.float32))
    # test rows retrieve from all train rows
    feat_test = _neighbour_features(E[test_idx], E[train_idx], y[train_idx], k)
    # train rows retrieve out-of-fold within train
    feat_train = np.zeros((len(train_idx), 4), dtype=np.float32)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(train_idx)), n_inner)
    for f in folds:
        val = train_idx[f]
        ref_mask = np.ones(len(train_idx), dtype=bool)
        ref_mask[f] = False
        ref = train_idx[ref_mask]
        feat_train[f] = _neighbour_features(E[val], E[ref], y[ref], k)
    return feat_train, feat_test
