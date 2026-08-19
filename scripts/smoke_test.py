"""End-to-end harness smoke test (no external model weights).

Validates: dataset loading + unit conventions + fold-exact CV + metrics + reporting,
using a substrate-fingerprint-only LightGBM. This model sees NO enzyme information, so it is
only a lower-bound sanity check that the whole pipeline runs and reports correctly.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_dataset  # noqa: E402
from src.eval.cv import report, run_cv  # noqa: E402
from src.features.substrate import featurize_smiles  # noqa: E402
from src.utils.common import set_seed  # noqa: E402


class MaccsLGBM:
    """LightGBM on MACCS fingerprint of the substrate only (baseline floor)."""

    def __init__(self, feat_cache: dict[str, np.ndarray]):
        self.feat_cache = feat_cache
        self.model = None

    def fit_predict(self, ds, train_idx, test_idx, fold):
        import lightgbm as lgb

        X = np.stack([self.feat_cache[s] for s in ds.smiles])
        y = ds.target
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        params = dict(objective="regression", metric="rmse", num_leaves=63,
                      learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, verbose=-1, num_threads=32)
        self.model = lgb.train(params, dtrain, num_boost_round=300)
        return self.model.predict(X[test_idx])


def main():
    set_seed(42)
    for param in ["kcat", "km", "kcat_km"]:
        ds = load_dataset(param)
        print(f"\n[{param}] n={len(ds)} target: mean={ds.target.mean():.3f} "
              f"std={ds.target.std():.3f} min={ds.target.min():.3f} max={ds.target.max():.3f}")
        uniq = sorted(set(ds.smiles.tolist()))
        fp = featurize_smiles(uniq, kinds=("maccs",))
        cache = {s: v for s, v in zip(uniq, fp)}
        res = run_cv(ds, lambda: MaccsLGBM(cache), verbose=False)
        print(report(res))


if __name__ == "__main__":
    main()
