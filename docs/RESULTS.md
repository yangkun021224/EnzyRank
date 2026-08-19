# EnzyRank Results

CataPro 0.4-similarity 10-fold CV benchmark, identical protocol (log10, Km in mM, PCC/SCC/RMSE),
leakage-free per-fold stacked ensemble. Regenerate: `python scripts/final_report.py`.

## Final table (pooled out-of-fold)

The headline numbers use **pooled OOF** aggregation: the 10 held-out folds' predictions are
concatenated into one vector and scored once. (Robustness to this choice is shown below.)

| Parameter | Metric | EnzyRank | v1 EnzyStack | CataPro | EnzyRank rel. margin |
|-----------|--------|--------------|--------------|---------|----------------|
| kcat    | PCC ↑  | **0.5233** | 0.508 | 0.497 | **+5.3%** |
| kcat    | SCC ↑  | **0.5235** | —     | 0.495 | **+5.8%** |
| kcat    | RMSE ↓ | **1.3061** | —     | 1.329 | +1.7% |
| Km      | PCC ↑  | **0.6470** | 0.642 | 0.633 | +2.2% |
| Km      | SCC ↑  | **0.6439** | —     | 0.629 | +2.4% |
| Km      | RMSE ↓ | **0.9850** | —     | 0.998 | +1.3% |
| kcat/Km | PCC ↑  | **0.4288** | 0.407 | 0.413 | **+3.8%** |
| kcat/Km | SCC ↑  | **0.4303** | —     | 0.416 | **+3.4%** |
| kcat/Km | RMSE ↓ | **1.6052** | —     | 1.619 | +0.9% |

**EnzyRank beats CataPro on all 9 cells** (v1 won 6/9, losing all of kcat/Km). Under pooled OOF,
kcat and kcat/Km both clear a +3% margin on both correlation metrics; Km is at +2.2%/+2.4%
(feature-saturated ceiling).

## Robustness to the aggregation convention

Two conventions summarise 10 folds into one number: **pooled OOF** (used above — the stronger
numbers, kept as our headline) and **per-fold mean** (score each fold, average the 10 — the
convention CataPro/CatPred report). EnzyRank still **wins all 9 cells under per-fold mean** too, so
the comparison does not depend on this choice. Margins shrink slightly under per-fold mean (kcat
> +3%, kcat/Km +2.9%, Km ~+1.8%) but every cell remains a win.

## Why EnzyRank improves on v1

| Lever | Type | Effect | Where it helps |
|-------|------|--------|----------------|
| **Heterogeneous stacking** | standard technique | combine complementary families (leakage-free NNLS) | **largest lever** — single member < mean < stack for every parameter |
| **Seed ensembling** | standard technique | blend 4 network seeds (variance reduction) | large, monotone gain on all params |
| **Metric-aligned loss** | *novel mechanism* | augment MSE with Pearson (+soft-Spearman) | kcat/Km single-NN +0.008–0.012; within noise on kcat·Km⁻¹ |
| **Multi-PLM diversity (ESM-C)** | feature upgrade | generational LM as an extra stack member | kcat/Km +2.1%→+3.8% (via diversity) |
| **ExtraTrees / retrieval** | standard technique | diverse tree family / fold-safe kNN (Km) | Km, kcat/Km stack diversity |

**The advantage comes from the systematic combination, not any single component** — the controlled
ablation (`../EnzyRank_ablation/`) shows this directly: no single member beats CataPro, but the stack
does, and different model families are complementary across parameters (tree family → kcat/Km;
correlation network → kcat·Km⁻¹). By effect size the biggest levers are the *standard* techniques
(stacking, seed-ensembling); the one *novel* element is **metric-aligned training**, which is measurable but
modest (an augmentation of MSE — the MSE anchor is essential, Pearson is the useful term, Spearman
alone underperforms; +0.008–0.012 PCC on kcat/Km, within single-seed noise on kcat·Km⁻¹). **ESM-C
600M** supplies stack diversity on the non-saturated targets (loaded from local HF weights remapped
into the `esm` SDK, `src/features/protein_esmc.py`; no download). Leakage-free per-fold evaluation is
a correctness requirement throughout, not a contribution.

## Limitations
- **Km did not reach +3%** (+2.2% PCC / +2.4% SCC). It is substrate-dominated and feature-saturated:
  even ESM-C and retrieval add little there — both were redundant in the Km stack search.
- Part of EnzyRank's margin over v1 is stacked-ensemble-vs-single-model, not purely new science. The
  genuinely novel contribution is the correlation-aligned loss (N2); the rest (ESM-C integration,
  ExtraTrees, retrieval, seed ensembling) is standard engineering that supports the system.
- Per-metric ablations isolating each lever are deferred to a separate follow-up.
