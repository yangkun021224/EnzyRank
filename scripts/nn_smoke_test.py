"""Validate NN plumbing only (model + trainer + collate + fold API), without loading large PLMs.

This smoke test uses deterministic in-memory probe tensors on a tiny subset to check shapes, masking,
loss, early stopping, and prediction.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_dataset  # noqa: E402
from src.eval.cv import run_cv  # noqa: E402
from src.features.providers import SubstrateProvider  # noqa: E402
from src.models.fold_model import NNFoldModel  # noqa: E402
from src.models.interaction_model import ModelConfig  # noqa: E402
from src.utils.common import get_device, set_seed  # noqa: E402


class ProbeProtProvider:
    """Returns deterministic sequence-keyed probe tensors with the expected residue-feature shape."""

    def __init__(self):
        self.cache = {}
        self.dim = 1280

    def prime(self, seqs):
        return self

    def get(self, seq):
        if seq not in self.cache:
            L = min(max(len(seq), 8), 200)
            rng = np.random.default_rng(abs(hash(seq)) % (2**32))
            self.cache[seq] = rng.standard_normal((L, 1280)).astype(np.float16)
        return self.cache[seq]


def subset(ds, n_per_fold=40, folds=(0, 1)):
    keep = []
    for f in folds:
        idx = np.where(ds.fold == f)[0][:n_per_fold]
        keep.extend(idx.tolist())
    keep = np.array(sorted(keep))
    ds.df = ds.df.iloc[keep].reset_index(drop=True)
    ds.target = ds.target[keep]
    ds.fold = ds.fold[keep]
    ds.sequence = ds.sequence[keep]
    ds.smiles = ds.smiles[keep]
    return ds


def run(param, config_name, tasks, pooling, cross, consistency):
    print(f"\n### {config_name} on {param} (device={get_device()}) ###")
    ds = subset(load_dataset(param))
    cfg = ModelConfig(protein_dim=1280, substrate_dim=167 + 2048, fp_dim=167,
                      d_model=64, pool_heads=2, cross_heads=4, trunk_dims=[64],
                      head_hidden=32, pooling=pooling, n_cross_layers=cross, tasks=tasks)
    tk = dict(lr=1e-3, epochs=3, patience=5, consistency_w=consistency,
              device=get_device(), verbose=False)
    prot, sub = ProbeProtProvider(), SubstrateProvider(mode="fp")

    def factory():
        return NNFoldModel(cfg, primary_task=param, prot=prot, sub=sub,
                           trainer_kwargs=tk, batch_size=8)

    res = run_cv(ds, factory, folds=[0, 1], verbose=True)
    print(f"  pooled PCC={res['pooled']['PCC']:.3f} RMSE={res['pooled']['RMSE']:.3f} "
          f"(plumbing smoke check only)")


def main():
    set_seed(0)
    run("kcat", "E1_meanpool", ["kcat"], "mean", 0, 0.0)
    run("kcat", "I1_attnpool", ["kcat"], "attention", 0, 0.0)
    run("kcat", "I2_crossattn", ["kcat"], "attention", 2, 0.0)
    run("kcat_km", "I3_multitask", ["kcat", "km", "kcat_km"], "attention", 2, 0.1)
    print("\nAll NN plumbing checks ran.")


if __name__ == "__main__":
    main()
