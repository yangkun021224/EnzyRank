"""Fold runner for end-to-end ESM2 fine-tuning (CV `fit_predict` API)."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import N_FOLDS
from .finetune import FinetuneModel, SeqTokenDataset, collate_ft


class FinetuneFoldModel:
    def __init__(self, esm_dir, fp_matrix, tokenizer, d_model=256, dropout=0.3,
                 freeze_layers=0, lr_head=1e-3, lr_esm=1e-5, weight_decay=0.01,
                 epochs=8, patience=3, batch_size=32, device="cuda", seed=42):
        self.esm_dir = esm_dir
        self.fp = fp_matrix
        self.tok = tokenizer
        self.d_model, self.dropout, self.freeze_layers = d_model, dropout, freeze_layers
        self.lr_head, self.lr_esm, self.wd = lr_head, lr_esm, weight_decay
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.device, self.seed = device, seed

    def _loader(self, ds, idx, shuffle):
        d = SeqTokenDataset(ds.sequence[idx], self.fp[idx], ds.target[idx], self.tok)
        return DataLoader(d, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=collate_ft, num_workers=6, pin_memory=True,
                          persistent_workers=True)

    def fit_predict(self, ds, train_idx, test_idx, fold):
        val_fold = (fold + 1) % N_FOLDS
        val_idx = train_idx[ds.fold[train_idx] == val_fold]
        tr_idx = train_idx[ds.fold[train_idx] != val_fold]

        model = FinetuneModel(self.esm_dir, self.fp.shape[1], self.d_model,
                              self.dropout, self.freeze_layers).to(self.device)
        esm_p = [p for n, p in model.named_parameters() if n.startswith("esm.") and p.requires_grad]
        head_p = [p for n, p in model.named_parameters() if not n.startswith("esm.")]
        opt = torch.optim.AdamW([{"params": esm_p, "lr": self.lr_esm},
                                 {"params": head_p, "lr": self.lr_head}], weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)

        tr_loader = self._loader(ds, tr_idx, True)
        va_loader = self._loader(ds, val_idx, False)
        best, best_state, bad = 1e9, None, 0
        for ep in range(self.epochs):
            model.train()
            for b in tr_loader:
                b = {k: v.to(self.device) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(b["input_ids"], b["attention_mask"], b["fp"])
                    loss = F.smooth_l1_loss(pred, b["target"])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            v = self._rmse(model, va_loader)
            if v < best - 1e-4:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            print(f"    fold{fold} ep{ep} val_rmse={v:.4f} best={best:.4f}", flush=True)
            if bad >= self.patience:
                break
        model.load_state_dict(best_state)
        preds = self._predict(model, self._loader(ds, test_idx, False))
        del model; torch.cuda.empty_cache()
        return preds

    @torch.no_grad()
    def _rmse(self, model, loader):
        model.eval(); se = n = 0
        for b in loader:
            b = {k: v.to(self.device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(b["input_ids"], b["attention_mask"], b["fp"]).float()
            se += ((p - b["target"]) ** 2).sum().item(); n += len(p)
        return math.sqrt(se / max(n, 1))

    @torch.no_grad()
    def _predict(self, model, loader):
        model.eval(); out = []
        for b in loader:
            b = {k: v.to(self.device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(b["input_ids"], b["attention_mask"], b["fp"]).float().cpu().numpy())
        return np.concatenate(out)
