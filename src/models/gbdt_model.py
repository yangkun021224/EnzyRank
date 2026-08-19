"""I4 — Gradient-boosted / tree ensemble on pooled features.

Runs entirely on CPU (208 threads), in parallel with GPU NN training. Serves two roles:
  * a strong, cheap standalone contender (UniKP/CataPro-ExtraTree evidence);
  * an ensemble partner for the NN (blend OOF predictions).

`GBDTFoldModel` implements the CV `fit_predict` API. The full feature matrix is built once and
shared across folds via `TabularCV`.
"""
from __future__ import annotations

import numpy as np

from ..features.tabular import build_matrix


class TabularCV:
    """Holds the prebuilt feature matrix for a dataset so folds only slice it."""

    def __init__(self, ds, blocks=("esm2", "maccs", "morgan")):
        self.X = build_matrix(list(ds.sequence), list(ds.smiles), blocks=blocks)
        self.y = ds.target


class GBDTFoldModel:
    def __init__(self, tab: TabularCV, kind: str = "lightgbm", params: dict | None = None,
                 n_estimators: int = 1500, threads: int = 64, te_cols=None,
                 retrieval_emb=None, retrieval_k: int = 16):
        self.tab = tab
        self.kind = kind
        self.params = params or {}
        self.n_estimators = n_estimators
        self.threads = threads
        self.te_cols = te_cols  # e.g. ("EC","Organism") -> fold-safe target-encoded features
        self.retrieval_emb = retrieval_emb  # (N, D) embeddings for fold-safe kNN retrieval (N4)
        self.retrieval_k = retrieval_k

    def fit_predict(self, ds, train_idx, test_idx, fold):
        X, y = self.tab.X, self.tab.y
        Xtr, ytr, Xte = X[train_idx], y[train_idx], X[test_idx]
        if self.te_cols:
            from ..features.target_encode import target_encode_columns
            te_tr, te_te, _ = target_encode_columns(ds.df, y, train_idx, test_idx,
                                                    cols=self.te_cols)
            Xtr = np.hstack([Xtr, te_tr])
            Xte = np.hstack([Xte, te_te])
        if self.retrieval_emb is not None:
            from ..features.retrieval import retrieval_features
            r_tr, r_te = retrieval_features(self.retrieval_emb, y, train_idx, test_idx,
                                            k=self.retrieval_k)
            Xtr = np.hstack([Xtr, r_tr])
            Xte = np.hstack([Xte, r_te])
        # inner val for early stopping
        rng = np.random.default_rng(42 + fold)
        perm = rng.permutation(len(train_idx))
        n_val = max(200, int(0.1 * len(perm)))
        vi, ti = perm[:n_val], perm[n_val:]

        if self.kind == "lightgbm":
            import lightgbm as lgb

            p = dict(objective="regression", metric="rmse", num_leaves=127,
                     learning_rate=0.03, feature_fraction=0.5, bagging_fraction=0.8,
                     bagging_freq=1, min_child_samples=20, verbose=-1,
                     num_threads=self.threads)
            p.update(self.params)
            dtr = lgb.Dataset(Xtr[ti], label=ytr[ti])
            dva = lgb.Dataset(Xtr[vi], label=ytr[vi])
            model = lgb.train(p, dtr, num_boost_round=self.n_estimators, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False),
                                         lgb.log_evaluation(0)])
            return model.predict(Xte)

        if self.kind == "xgboost":
            import xgboost as xgb

            p = dict(objective="reg:squarederror", eval_metric="rmse", max_depth=8,
                     eta=0.03, subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                     nthread=self.threads)
            p.update(self.params)
            dtr = xgb.DMatrix(Xtr[ti], label=ytr[ti])
            dva = xgb.DMatrix(Xtr[vi], label=ytr[vi])
            model = xgb.train(p, dtr, num_boost_round=self.n_estimators,
                              evals=[(dva, "val")], early_stopping_rounds=100,
                              verbose_eval=False)
            return model.predict(xgb.DMatrix(Xte))

        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler

            sc = StandardScaler().fit(Xtr)
            model = Ridge(alpha=self.params.get("alpha", 100.0))
            model.fit(sc.transform(Xtr), ytr)
            return model.predict(sc.transform(Xte))

        if self.kind == "extratrees":
            from sklearn.ensemble import ExtraTreesRegressor

            model = ExtraTreesRegressor(n_estimators=self.params.get("n_estimators", 600),
                                        max_features=0.3, min_samples_leaf=3,
                                        n_jobs=self.threads, random_state=42)
            model.fit(Xtr, ytr)
            return model.predict(Xte)

        if self.kind == "catboost":
            from catboost import CatBoostRegressor, Pool

            model = CatBoostRegressor(iterations=self.n_estimators, depth=8,
                                      learning_rate=0.03, loss_function="RMSE",
                                      thread_count=self.threads, verbose=False,
                                      early_stopping_rounds=100, **self.params)
            model.fit(Pool(Xtr[ti], ytr[ti]), eval_set=Pool(Xtr[vi], ytr[vi]), verbose=False)
            return model.predict(Xte)

        raise ValueError(f"unknown gbdt kind {self.kind!r}")
