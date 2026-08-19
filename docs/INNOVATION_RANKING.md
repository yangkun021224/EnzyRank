# EnzyRank — Innovation Potential Ranking

> **Post-ablation note (final positioning).** This document records the *a-priori* planning ranking
> (impact × feasibility × novelty) used to choose what to build. The controlled ablation
> (`../EnzyRank_ablation/`) since refined the conclusions, and the paper's positioning follows the
> ablation, not these prior scores:
> - EnzyRank is sold as a **system** (multi-PLM + complementary predictor families + metric-aligned
>   training, integrated by stacking), not as any single trick. Its main contributions are the
>   **fair-comparison 9/9 SOTA**, the **metric-aligned training** mechanism, and the **ablation
>   finding that model families are complementary across parameters**.
> - **Metric-aligned training (N2)** is the one genuinely novel mechanism, but it is an *augmentation
>   of* MSE (not a replacement), Pearson is the useful term (Spearman-alone underperforms), and its
>   single-model effect is **modest** (+0.008–0.012 on kcat/Km; within single-seed noise on kcat·Km⁻¹).
> - The largest effect sizes come from **standard** techniques (stacking, seed-ensembling); these are
>   described as *how the system is built*, not as novelty. Multi-PLM features and leakage-free
>   evaluation are likewise not sold as contributions.

Goal: **break past v1's ceiling** on the CataPro 0.4-simi 10-fold benchmark (same data/splits/metrics),
especially (a) the kcat/Km deficit and (b) the small (~1-2%) overall margins. v1 already beats CataPro
on kcat & Km; EnzyRank aims for a decisive margin on all three parameters.

## What v1 (EnzyStack) achieved and where it fell short
Final v1: kcat 0.508 / Km 0.642 (beat CataPro on all 3 metrics, +1-2%); kcat/Km 0.407 (−1.5%).

Diagnosed remaining deficiencies (targets for EnzyRank):
- **D1 — Feature ceiling.** ProtT5 ⊕ ESM2 are same-generation seq-only PLMs and proved *redundant*
  (concatenation added +0.003). No generational feature upgrade tried.
- **D2 — Loss/metric misalignment.** Training minimises MSE, but the benchmark scores **PCC/SCC**
  (rank/linear correlation). MSE does not directly optimise these.
- **D3 — No structural / 3D information.** Catalysis is a 3D active-site phenomenon; v1 (and CataPro)
  use only sequence + 2D substrate.
- **D4 — kcat/Km modelled as a noisy scalar.** No effective use of the physical identity
  log(kcat/Km)=log kcat − log km (multi-task was coded but never run).
- **D5 — No use of neighbours.** Each prediction is made in isolation; known measurements of similar
  enzyme–substrate pairs are not retrieved.
- **D6 — Label noise ignored.** Kinetic measurements are noisy; all samples weighted equally.

## Grounded innovation ranking (impact × feasibility × novelty × deficiency-fit)

| # | Innovation | Fixes | Impact | Feas. | Novelty | Notes / evidence |
|---|-----------|-------|:---:|:---:|:---:|------|
| **N1** | **ESM-C 600M features** (EvolutionaryScale, 2024) | D1 | 5 | 4 | 4 | ESM-C 600M rivals ESM2-**3B**, approaches 15B — a substantial generational jump, not redundant. Drop-in feature upgrade for GBDT + NN. ~1.2 GB. |
| **N2** | **Correlation-aligned training** (differentiable soft-Spearman + Pearson loss) | D2 | 4 | 5 | 5 | Metrics ARE PCC/SCC; MSE is misaligned. SoDeep-style soft-rank approximates Spearman; a Pearson loss directly targets the linear metric. Pure code, no download, applies to the NN today. |
| **N3** | **Structure-aware features** (SaProt 3Di, or AlphaFold active-site pocket) | D3 | 4 | 2 | 4 | SaESM2 beats ESM2 on 6/9 tasks. NEW 3D info CataPro ignores. Cost: fetch ~13 k AlphaFold structures + Foldseek tokens. Highest new-signal potential, highest effort. |
| **N4** | **Retrieval augmentation** (kNN over enzyme⊕substrate embeddings → neighbour-value prior) | D5 | 3 | 4 | 4 | Use nearest known measurements as features/prior. Fold-safe (retrieve from train only). Cheap given cached embeddings. |
| **N5** | **Multi-task + physical consistency** (joint kcat, km, kcat/Km with log-identity constraint) | D4 | 3 | 5 | 3 | Coded in v1 (`I3`), never run. Cheap; targets the kcat/Km deficit directly. |
| **N6** | **Noise-robust training** (Huber/βNLL, label-uncertainty or replicate weighting) | D6 | 2 | 5 | 3 | Kinetic labels are noisy; robust losses can raise the ceiling a little. Cheap add-on. |
| **N7** | **3D molecular encoder** (Uni-Mol / MolFormer substrate) | D1(sub) | 2 | 3 | 3 | Substrate side is under-modelled; lower priority (v1 showed substrate isn't the bottleneck). |

## Execution plan (cheap-and-metric-aligned first, then new information)
The final winner is expected to be a **composition**: ESM-C features + correlation-aligned NN +
retrieval + multi-task, all fed into the (inherited) heterogeneous leakage-free stack.

1. **N2 correlation-aligned loss** — no download; validate on the inherited ProtT5 cross-attention NN
   immediately. Directly targets PCC/SCC. (Phase 1)
2. **N1 ESM-C features** — request download; extract; add to GBDT + NN. (Phase 2, feature upgrade)
3. **N5 multi-task** — run the coded I3 on ESM-C/ProtT5; target kcat/Km. (Phase 3)
4. **N4 retrieval** — kNN neighbour-value features into the stack. (Phase 4)
5. **N3 structure** — only if 1–4 leave a gap worth the AlphaFold+Foldseek pipeline. (Phase 5)
6. **Combine** best components → final leakage-free stack vs CataPro on all three parameters.

Each phase logs to `EXPERIMENTS.md`; we keep only what improves the stacked metric,
and carry v1's out-of-fold predictions (`baseline_v1_oof/`) as ready-made ensemble members.
