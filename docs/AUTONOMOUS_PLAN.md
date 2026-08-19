# 10-Hour Autonomous Run Plan (EnzyRank)

Operating rules: proceed step by step; after every step, self-check (numbers sane? leakage-free?
did it improve the stacked metric?); keep only what helps; log every result to `EXPERIMENTS.md`;
record outcomes exactly, including negative results.

Targets: beat CataPro (kcat 0.497 / km 0.633 / kcat_km 0.413) and v1 (0.508 / 0.642 / 0.407),
especially closing kcat/Km and widening margins.

## Agenda (priority order)

### A. Finish N2 + first V2 ensemble  [in progress]
- [~] N2 correlation-NN full 10-fold for km, kcat_km (running)
- [ ] `final_report.py`: stack v1 OOF + N2 corr-NN → does it beat v1? (self-check)

### B. N1 — ESM-C 600M features  [MAIN WORK]
- [ ] Get ESM-C loading (esm SDK or transformers-from-source); write `src/features/protein_esmc.py`
- [ ] Extract ESM-C POOLED (mean+max, tiny cache) → GBDT all params; compare vs ESM2/ProtT5 GBDT
- [ ] Disk mgmt: delete regenerable ProtT5 residue (16 GB, shared w/ delivered v1) → free space
- [ ] Extract ESM-C PER-RESIDUE (~19 GB) → cache
- [ ] Train ESM-C cross-attention NN + N2 corr loss (all params, seeds 42/123)
- [ ] Ensemble ESM-C members + v1 + N2 → measure (self-check: stack gain?)

### C. N5 — multi-task + physical consistency  [kcat/Km focus]
- [ ] Run I3 multitask (kcat+km+kcat_km, log-identity constraint) + correlation loss
- [ ] Add to kcat/Km ensemble; check if it closes the deficit

### D. Combine & finalize
- [ ] Best-of-all leakage-free stack → final table vs CataPro + v1
- [ ] Update README/RESULTS/EXPERIMENTS; clean repo

### E. N3 — structure (only if a gap remains and time allows)
- [ ] AlphaFold structures for UniProtIDs + SaProt/Foldseek features (heavy)

## Disk budget (50 GB total, ~21 GB free now)
- ESM-C pooled: ~0.1 GB. ESM-C residue: ~19 GB → requires freeing ProtT5 residue first.
- Keep the shared ProtT5 residue until the N2 sweep (which uses it) finishes.

## Self-check checklist per experiment
1. Does the pooled/stacked metric actually exceed the prior best? (not just some folds)
2. Fold-safe? (features from train only; OOF aligned to index order)
3. Sane ranges? (PCC in [0,0.7]; RMSE ~1-1.7)
4. Logged to EXPERIMENTS.md with the measured number, including negative results.

## Progress journal (append per step)
- (init) N2 validated (+0.010 PCC on NN). N4 retrieval dropped (no net gain). ESM-C downloaded.
- (step A done) V2 = v1 stack + N2 corr-NN: kcat 0.5137 (+3.4% PCC/SCC over CataPro), km 0.6428
  (+1.5%), kcat_km 0.4121 (matched, up from v1 0.407). N2 lifted the stack.
- (N5) multitask: marginal (kcat_km stack 0.4121->0.4124). Dropped.
- (ESM-C loading saga) SDK `from_pretrained` downloads its OWN weight format (data/, sha 8ef856)
  != user's HF-format model.safetensors (e4232c); SDK network download stalled at 804MB. Pivoted:
  installing transformers-from-git (has `esmc`) to load the user's LOCAL HF weights directly.
- (parallel GPU use) running N2 corr-NN seed 123 (diverse members) while transformers installs.
- (ESM-C paused) network/proxy prevented both SDK download and transformers-git. Pivot: maximize N2
  (validated) via a robust correlation-NN SEED ENSEMBLE
  (seeds 42/123/456/789) — like CataPro's 10 replicates — to stabilise + strengthen the NN member.
- Remaining realistic levers (no downloads): N2 seed ensemble; N2 loss tuning; then finalize V2.
- (N2 2-seed + km retrieval) kcat 0.5154 (+3.7% PCC, +4.1% SCC), km 0.6450 (+1.9% PCC, +2.1% SCC).
  kcat clears +3% on correlations. Queuing more seeds (456/789) for km stability. kcat_km s123 pending.
- (ESM-C loaded) loaded user's local HF weights into esm-SDK ESMC via key-remap (0 missing keys),
  batched extraction (pooled 9min, residue 9min). Testing: ESM-C cross-attention corr-NN (GPU) +
  all-PLM GBDT (CPU). Freed ProtT5 residue (regenerable) for disk. If ESM-C NN beats ProtT5 NN
  (0.4468 kcat) or adds to the stack, it's the generational lever toward +3% on km/kcat_km.
