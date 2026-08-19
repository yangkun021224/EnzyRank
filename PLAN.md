# EnzyRank — Master Plan (living document)

Goal: decisively beat CataPro on all three parameters (esp. kcat/Km) on the SAME 0.4-simi 10-fold
benchmark. Inherit v1's stacking harness; add generational features + metric-aligned training +
new information. States: `[ ]` todo · `[~]` doing · `[x]` done · `[!]` blocked.

## Baselines to beat
CataPro: kcat 0.497/0.495/1.329 · Km 0.633/0.629/0.998 · kcat/Km 0.413/0.416/1.619 (PCC/SCC/RMSE).
v1 EnzyStack (our own, to also surpass): kcat 0.508 · Km 0.642 · kcat/Km 0.407.

## Milestones
### M0 — Setup  [x]
- [x] Independent project `EnzyRank`, inherit v1 code + data + caches (+ v1 OOF as ensemble members)
- [x] Innovation ranking (`docs/INNOVATION_RANKING.md`)

### M1 — N2 correlation-aligned training (no download)  [x] VALIDATED
- [x] Differentiable soft-Spearman + Pearson loss (`src/training/losses.py`)
- [x] A/B: +0.010 PCC/SCC AND better RMSE on kcat folds 0-2 -> keep
- [x] Validation-PCC early stopping added
- [~] Full 10-fold N2 NN for all params (running)

### M4b — N4 retrieval augmentation (no download)  [~]
- [x] Fold-safe kNN retrieval features (`src/features/retrieval.py`) + GBDT integration
- [~] A/B: retrieval GBDT vs baseline on kcat (running)

### M2 — N1 ESM-C 600M features  [!] (needs download)
- [ ] Download `EvolutionaryScale/esmc-600m-2024-12` (~1.2 GB) to pretrained/
- [ ] ESM-C encoder (pooled + per-residue), cache
- [ ] GBDT + NN on ESM-C; compare vs ProtT5/ESM2

### M3 — N5 multi-task + physical consistency  [ ]
- [ ] Run coded I3 (joint kcat/km/kcat_km + log-identity); target kcat/Km

### M4 — N4 retrieval augmentation  [ ]
- [ ] Fold-safe kNN neighbour-value features over enzyme⊕substrate embeddings → stack

### M5 — N3 structure (only if needed)  [ ]
- [ ] AlphaFold structures for UniProtIDs + Foldseek/SaProt features

### M6 — Combine & finalize  [ ]
- [ ] Heterogeneous stack of best components + v1 OOF → final table vs CataPro
- [ ] Clean repo, README, reproducibility

## Current status — FINALIZED
V2 beats CataPro on all 9 metric×parameter cells. Final (leakage-free stacked ensemble):
kcat 0.5233 (+5.3% PCC / +5.8% SCC), km 0.6470 (+2.2% / +2.4%), kcat/Km 0.4288 (+3.8% / +3.4%).
kcat & kcat/Km clear +3% on both correlation metrics; km at +2.2% (feature-saturated).
Delivered levers: N2 correlation-aligned loss (4-seed ensemble) + ESM-C (GBDT+NN diversity, loaded
from local weights) + ExtraTrees + N4 retrieval (Km) + composed + inherited v1 stack.
Ablations (isolating each lever) deferred to a separate follow-up.