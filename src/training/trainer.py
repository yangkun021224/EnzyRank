"""Training loop: masked multi-task regression with physical-consistency regularization.

Loss pieces:
  * per-task MSE (or MVE Gaussian NLL when cfg.mve) over non-missing labels.
  * physical-consistency penalty (I3): when kcat, km, kcat_km heads are all present, push
      pred_kcat - pred_km  ==  pred_kcat_km        (log identity kcat/Km = kcat / Km)
    as an unsupervised structural constraint on the outputs.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..models.interaction_model import EnzSubModel, ModelConfig


def _task_loss(pred: torch.Tensor, y: torch.Tensor, mve: bool):
    valid = ~torch.isnan(y)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    if mve:
        mean = pred[:, 0][valid]
        var = F.softplus(pred[:, 1][valid]) + 1e-6
        yy = y[valid]
        return (0.5 * torch.log(2 * math.pi * var) + (mean - yy) ** 2 / (2 * var)).mean()
    return F.mse_loss(pred[:, 0][valid], y[valid])


def masked_multitask_loss(preds, targets, tasks, mve, consistency_w: float):
    """preds: dict task->(B,out); targets: (B,T) aligned to `tasks` order."""
    total = 0.0
    for i, t in enumerate(tasks):
        total = total + _task_loss(preds[t], targets[:, i], mve)
    if consistency_w > 0 and {"kcat", "km", "kcat_km"} <= set(tasks):
        mk = preds["kcat"][:, 0]
        mm = preds["km"][:, 0]
        mkm = preds["kcat_km"][:, 0]
        total = total + consistency_w * F.mse_loss(mk - mm, mkm)
    return total


class TorchTrainer:
    def __init__(self, cfg: ModelConfig, lr=3e-4, weight_decay=1e-2, epochs=60,
                 consistency_w=0.1, patience=10, device="cuda", grad_clip=1.0,
                 amp=True, verbose=False, corr_weights=None):
        self.cfg = cfg
        self.lr, self.wd, self.epochs = lr, weight_decay, epochs
        self.consistency_w = consistency_w
        self.patience, self.device, self.grad_clip, self.verbose = patience, device, grad_clip, verbose
        self.amp = amp and device == "cuda"
        # N2: correlation-aligned loss weights {mse_w, pearson_w, spearman_w}; None -> plain MSE/MVE
        self.corr_weights = corr_weights

    def _loss(self, preds, targets):
        if self.corr_weights and len(self.cfg.tasks) == 1 and not self.cfg.mve:
            from .losses import correlation_loss
            t = self.cfg.tasks[0]
            y = targets[:, 0]
            valid = ~torch.isnan(y)
            if valid.sum() < 4:
                return masked_multitask_loss(preds, targets, self.cfg.tasks, self.cfg.mve,
                                             self.consistency_w)
            return correlation_loss(preds[t][:, 0][valid], y[valid], **self.corr_weights)
        return masked_multitask_loss(preds, targets, self.cfg.tasks, self.cfg.mve,
                                     self.consistency_w)

    def _run_epoch(self, model, loader, opt=None):
        train = opt is not None
        model.train(train)
        losses = []
        for batch in loader:
            b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.set_grad_enabled(train), \
                 torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                preds = model(b["prot_tokens"], b["prot_mask"], b["sub_tokens"],
                              b["sub_mask"], b.get("fp"))
                loss = self._loss(preds, b["targets"])
            if train:
                opt.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def fit(self, train_loader, val_loader, primary_task: str):
        model = EnzSubModel(self.cfg).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        best_val, best_state, bad = float("inf"), None, 0
        pidx = self.cfg.tasks.index(primary_task)
        # correlation mode: select the checkpoint by validation (1 - PCC), aligning early stopping
        # with the benchmark metric (and stopping sooner than RMSE, which the corr loss ignores).
        use_pcc = self.corr_weights is not None and len(self.cfg.tasks) == 1
        for ep in range(self.epochs):
            tr = self._run_epoch(model, train_loader, opt)
            sched.step()
            val = self._val_score(model, val_loader, pidx, use_pcc)
            if val < best_val - 1e-4:
                best_val, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if self.verbose:
                tag = "1-val_pcc" if use_pcc else "val_rmse"
                print(f"    ep{ep:03d} train_loss={tr:.4f} {tag}={val:.4f} best={best_val:.4f}")
            if bad >= self.patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.best_val = best_val
        return self

    @torch.no_grad()
    def _val_score(self, model, loader, pidx, use_pcc: bool):
        """Validation score to MINIMISE: RMSE, or (1 - Pearson) when use_pcc."""
        model.eval()
        task = self.cfg.tasks[pidx]
        preds_all, y_all = [], []
        for batch in loader:
            b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                preds = model(b["prot_tokens"], b["prot_mask"], b["sub_tokens"], b["sub_mask"], b.get("fp"))
            y = b["targets"][:, pidx]
            valid = ~torch.isnan(y)
            if valid.sum() == 0:
                continue
            preds_all.append(preds[task][:, 0][valid].float().cpu().numpy())
            y_all.append(y[valid].float().cpu().numpy())
        p = np.concatenate(preds_all)
        yv = np.concatenate(y_all)
        if use_pcc:
            pc = np.corrcoef(p, yv)[0, 1]
            return 1.0 - (pc if np.isfinite(pc) else -1.0)
        return math.sqrt(float(np.mean((p - yv) ** 2)))

    @torch.no_grad()
    def predict(self, loader, task: str) -> np.ndarray:
        self.model.eval()
        out = []
        for batch in loader:
            b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                preds = self.model(b["prot_tokens"], b["prot_mask"], b["sub_tokens"], b["sub_mask"], b.get("fp"))
            out.append(preds[task][:, 0].float().cpu().numpy())
        return np.concatenate(out)
