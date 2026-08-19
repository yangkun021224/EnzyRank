"""NNFoldModel — adapts the interaction model + trainer to the CV harness `fit_predict` API.

Uses a GPU-resident feature store (frozen ESM2 features uploaded once) and index batching, so
folds share the store and no per-batch host->device transfer occurs.

Supports:
  * single-task: cfg.tasks == [param]; targets = ds.target.
  * multi-task (I3): cfg.tasks == ['kcat','km','kcat_km'] on the kcat_km dataset, whose rows carry
    kcat and Km columns -> a (N,3) target matrix on the SAME folds (leakage-safe).
"""
from __future__ import annotations

import numpy as np
import torch

from ..features.providers import ProteinResidueProvider, SubstrateProvider
from ..models.interaction_model import ModelConfig
from ..training.gpu_batcher import GpuFeatureStore, IndexBatcher
from ..training.trainer import TorchTrainer


def multitask_targets(ds, tasks: list[str]) -> np.ndarray:
    """Build an (N, len(tasks)) target matrix from a kcat_km KineticDataset's raw columns."""
    df = ds.df
    cols = {}
    if "kcat" in tasks:
        cols["kcat"] = np.log10(df["kcat(s^-1)"].to_numpy(dtype=np.float64))
    if "km" in tasks:
        cols["km"] = np.log10(df["Km(M)"].to_numpy(dtype=np.float64)) + 3.0
    if "kcat_km" in tasks:
        cols["kcat_km"] = ds.target
    return np.stack([cols[t] for t in tasks], axis=1).astype(np.float32)


class NNFoldModel:
    def __init__(self, cfg: ModelConfig, primary_task: str,
                 prot: ProteinResidueProvider, sub: SubstrateProvider,
                 store: GpuFeatureStore | None = None,
                 trainer_kwargs: dict | None = None, batch_size: int = 64,
                 val_frac: float = 0.1, seed: int = 42, val_mode: str = "fold"):
        self.cfg = cfg
        self.primary_task = primary_task
        self.prot = prot
        self.sub = sub
        self.store = store
        self.trainer_kwargs = trainer_kwargs or {}
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed
        # 'fold': hold out a neighbouring fold as validation -> OOD-aware early stopping (the test
        # folds are dissimilar sequence clusters, so a random in-distribution val stops too late).
        # 'random': legacy random val_frac split.
        self.val_mode = val_mode

    def _targets(self, ds):
        if len(self.cfg.tasks) == 1:
            return ds.target[:, None].astype(np.float32)
        return multitask_targets(ds, self.cfg.tasks)

    def _ensure_store(self, ds):
        if self.store is None:
            self.prot.prime(list(ds.sequence))
            self.sub.prime(list(ds.smiles))
            self.store = GpuFeatureStore(self.prot, self.sub, list(ds.sequence),
                                         list(ds.smiles), use_fp=self.cfg.fp_dim > 0,
                                         device=self.trainer_kwargs.get("device", "cuda"))
        return self.store

    def fit_predict(self, ds, train_idx, test_idx, fold):
        store = self._ensure_store(ds)
        targets = self._targets(ds)

        rng = np.random.default_rng(self.seed + fold)
        if self.val_mode == "fold":
            from ..data.dataset import N_FOLDS
            val_fold = (fold + 1) % N_FOLDS
            val_idx = train_idx[ds.fold[train_idx] == val_fold]
            tr_idx = train_idx[ds.fold[train_idx] != val_fold]
            if len(val_idx) == 0:  # safety fallback
                perm = rng.permutation(train_idx)
                n_val = max(1, int(len(perm) * self.val_frac))
                val_idx, tr_idx = perm[:n_val], perm[n_val:]
        else:
            perm = rng.permutation(train_idx)
            n_val = max(1, int(len(perm) * self.val_frac))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]

        def batcher(idx, shuffle):
            return IndexBatcher(store, ds.sequence, ds.smiles, targets, idx,
                                self.batch_size, shuffle, seed=self.seed + fold)

        trainer = TorchTrainer(self.cfg, **self.trainer_kwargs)
        trainer.fit(batcher(tr_idx, True), batcher(val_idx, False), primary_task=self.primary_task)
        preds = trainer.predict(batcher(test_idx, False), task=self.primary_task)
        del trainer
        torch.cuda.empty_cache()
        return preds
