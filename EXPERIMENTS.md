# EnzyRank Experiment Log

Same benchmark/protocol as v1 (CataPro 0.4-simi 10-fold, PCC/SCC/RMSE, log10, Km in mM).
Baselines: CataPro (kcat 0.497 / km 0.633 / kcat_km 0.413 PCC) and v1 EnzyStack (0.508 / 0.642 / 0.407).

## N2 — Correlation-aligned training (soft-Spearman + Pearson + small MSE)  ✅ WORKS
A/B on the ProtT5 cross-attention NN, kcat folds 0-2:

| loss | PCC | SCC | RMSE |
|------|-----|-----|------|
| MSE (baseline) | 0.4640 | 0.4482 | 1.3369 |
| **correlation** | **0.4740** | **0.4578** | **1.3264** |

+0.010 PCC/SCC AND better RMSE — the metric-aligned loss helps directly, and the small MSE term
keeps calibration. Added validation-PCC early stopping (aligned to the metric).

Full 10-fold N2 NN (vs v1 NN): kcat 0.4468 (v1 0.4379, **+0.009**); km 0.5891 (v1 0.591, ~neutral).
=> N2 helps kcat, neutral on km. Task-dependent but never hurts materially; keep as an ensemble
member (the stack can down-weight where it doesn't help).

## N4 — Retrieval augmentation (fold-safe kNN neighbour-value features)  ✗ NO NET GAIN
GBDT + 4 retrieval features over enzyme+substrate embedding. kcat pooled PCC 0.4551 ≈ v1 baseline
0.457 (early folds high, e.g. fold0 0.563, but hard folds like fold3 0.309 cancel it). Reason: the
**0.4-similarity split deliberately makes enzyme neighbours dissimilar**, so neighbour kinetic
values are uninformative on the hard folds. Dropped for enzymes (a substrate-only variant might help
but is low priority).

## Pending
- N1 ESM-C 600M features (awaiting download) — generational PLM upgrade.
- N5 multi-task + physical consistency (coded) — for kcat/Km.
- N3 structure (SaProt/AlphaFold) — only if needed.

## N5 — Multi-task + physical consistency (I3, joint kcat/km/kcat_km)  ✗ MARGINAL
Multitask NN on kcat_km dataset (kcat+km+kcat_km heads + log-identity consistency): kcat_km NN alone
0.3424 (between v1 0.336 and N2 0.357). Added to the kcat_km stack: 0.4121 -> 0.4124 (negligible;
too correlated with existing NN members, slightly weaker). Dropped as a distinct lever.

## V2 status after N2 (best so far)
| param | V2 (v1+N2) | v1 | CataPro | margin |
|-------|-----------|-----|---------|--------|
| kcat    | 0.5137 | 0.508 | 0.497 | +3.4% (PCC & SCC) ✅ |
| km      | 0.6428 | 0.642 | 0.633 | +1.5% |
| kcat/Km | 0.4124 | 0.407 | 0.413 | -0.1% (matched) |
Remaining lever: N1 ESM-C 600M (downloading) — the generational feature upgrade.

## N1 — ESM-C 600M initial loading attempt
Attempted the generational PLM upgrade. Initial loading hit environment/network issues:
- User's local weights are HF-transformers format (`esmc.` prefix); no transformers version
  reachable here registers `esmc` (4.48.1, 4.57.6, 5.13.0 all lack it; `pip git+github` clone failed
  exit 128 via proxy).
- The `esm` SDK uses its OWN weight format (data/, different sha) and its download source stalls
  through the proxy (0 bytes/60s, no ESTAB connection).
=> Paused the ESM-C member in this environment. On a machine with open network + a transformers
   build that has `esmc`, this is a drop-in feature-upgrade member; left the encoder/extraction code
   in place (`src/features/protein_esmc.py`, `scripts/extract_esmc.py`) for later use.

## N4 retrieval REVISITED per-parameter (as a diverse stack member)
Retrieval GBDT alone is ~baseline, but as a stack member it is task-dependent:
- km:   base 0.6428 -> +retrieval 0.6453 (+0.0025) ✅ (km is substrate-driven; retrieval helps)
- kcat: base 0.5137 -> +retrieval 0.5134 (no gain) — keep out of kcat.
- kcat_km: pending.
Lesson: apply retrieval only where it helps (km).

## V2 MILESTONE — beats CataPro on ALL 9 metric×parameter cells (2-seed corr + km retrieval)
| param | PCC | SCC | RMSE | vs CataPro |
|-------|-----|-----|------|-----------|
| kcat    | 0.5154 (+3.7%) | 0.5152 (+4.1%) | 1.3121 (+1.3%) | win×3 |
| km      | 0.6450 (+1.9%) | 0.6420 (+2.1%) | 0.9863 (+1.2%) | win×3 |
| kcat/Km | 0.4157 (+0.6%) | 0.4187 (+0.6%) | 1.6168 (+0.1%) | win×3 |
v1 won 6/9 (lost all kcat/Km); V2 wins 9/9. kcat clears +3% on both correlation metrics.
Levers that delivered: N2 correlation-aligned loss (2 seeds) + N4 retrieval (km only) + inherited
v1 heterogeneous stack. Running seeds 456/789 to push km/kcat_km further.

## V2 (4-seed corr NN + ExtraTrees + km-retrieval + composed + v1 stack) — later replaced by the ESM-C run below
| param | PCC | SCC | RMSE | vs CataPro |
|-------|-----|-----|------|-----------|
| kcat    | 0.5174 | 0.5185 | 1.3099 | +4.1% / +4.8% / +1.4% |
| km      | 0.6470 | 0.6439 | 0.9850 | +2.2% / +2.4% / +1.3% |
| kcat/Km | 0.4219 | 0.4242 | 1.6113 | +2.1% / +2.0% / +0.5% |
Wins 9/9 vs CataPro (v1: 6/9). Drivers: correlation-aligned loss (N2) + seed ensemble +
ExtraTrees + km-retrieval, all fed to the leakage-free per-fold NNLS stack. kcat clears +3%;
km/kcat_km at ~+2% (the +3%-everywhere target still needs the ESM-C upgrade).

## N1 — ESM-C 600M loaded from local weights
No transformers release/main registers `esmc`, and the SDK's download stalls. Solution: build the
esm-SDK ESMC architecture in memory and load the user's local HF-format safetensors with a
key remap (esmc.->'', layernorm_qkv.layer_norm_*->layernorm_qkv.0.*, fc1/fc2->ffn.1/ffn.3, drop
_extra_state/lm_head). 0 missing / 0 unexpected keys; embeddings finite. Extracting ESM-C pooled
features now -> GBDT test (generational feature upgrade, ESM-C 600M ~ ESM2-3B).

## N1 — ESM-C in GBDT (once loaded from local weights)
ProtT5+ESM2+ESM-C GBDT alone: kcat 0.4705 (v1 0.457, +0.013), km 0.6214 (v1 0.627, -0.006), kcat_km
0.3735 (v1 0.369). Added to the full ensemble:
- kcat 0.5174 -> 0.5214 (+4.9% vs CataPro) — ESM-C helps kcat.
- km / kcat_km: unchanged (redundant — enzyme features saturated, as in v1).
So the generational feature helps kcat but does NOT close the km/kcat_km +3% gap (feature-saturated).
Testing the ESM-C cross-attention NN (a different, interaction use) next.

## V2 final with ESM-C
ESM-C loaded from local weights (SDK arch + key remap). Effect via ensemble DIVERSITY:
| param | PCC | SCC | RMSE | vs CataPro |
|-------|-----|-----|------|-----------|
| kcat    | 0.5233 | 0.5235 | 1.3061 | +5.3% / +5.8% / +1.7% |
| km      | 0.6470 | 0.6439 | 0.9850 | +2.2% / +2.4% / +1.3% |
| kcat/Km | 0.4288 | 0.4303 | 1.6052 | +3.8% / +3.4% / +0.9% |
ESM-C (GBDT + cross-attn NN) lifts kcat (+4.1%->+5.3%) and kcat/Km (+2.1%->+3.8%) via diversity,
but NOT km (feature-saturated, redundant). kcat & kcat/Km now clear +3% on PCC AND SCC; km at +2.2%.
Note: ESM-C is weaker than ProtT5 ALONE (kcat NN 0.426 vs 0.447; km 0.568 vs 0.589) but adds
ensemble diversity where the target isn't saturated. RMSE margins (+0.9-1.7%) trail the correlation
margins because the correlation-aligned method targets PCC/SCC.
