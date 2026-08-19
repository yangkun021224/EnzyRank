#!/usr/bin/env bash
# Reproduce EnzyRank on the CataPro 0.4-similarity 10-fold benchmark.
#
# Prerequisites:
#   conda env create -f environment.yml && conda activate enzyrank
#   Feature caches under cache/ (bundled): ProtT5/ESM2 pooled, substrate, ChemBERTa-mean,
#   ProtT5 per-residue (regenerate via the v1 extraction scripts if absent), ESM-C pooled+residue.
#   Inherited v1 out-of-fold predictions under baseline_v1_oof/ (bundled).
#   ESM-C weights at pretrained/ESMC-600M/ or under ENZYRANK_PRETRAINED/ESMC-600M/
#   — loaded locally, no download.
set -euo pipefail
cd "$(dirname "$0")"
PY=python

echo "== (0) ESM-C features (if not cached) =="
[ -f cache/esmc_pooled.h5 ]  || $PY scripts/extract_esmc.py --pooled
[ -f cache/esmc_residue.h5 ] || $PY scripts/extract_esmc.py --residue

echo "== (1) Train V2-new base members per parameter =="
BLK="prott5 esm2 maccs morgan cbmean"
for p in kcat km kcat_km; do
  # correlation-aligned cross-attention NN (N2), 4 seeds, on ProtT5 and on ESM-C
  for s in 42 123 456 789; do
    A=(--config I2_crossattn --param "$p" --protein prott5 --corr); [ "$s" != 42 ] && A+=(--seed "$s")
    $PY scripts/run_experiment.py "${A[@]}"
  done
  $PY scripts/run_experiment.py --config I2_crossattn --param "$p" --protein esmc --corr   # ESM-C NN (diversity)
  # tree members
  $PY scripts/run_gbdt.py --param "$p" --kind extratrees --blocks $BLK
done
# param-specific members
$PY scripts/run_gbdt.py --param km      --kind lightgbm --blocks $BLK --retrieval --tag retr        # N4 (Km only)
$PY scripts/run_gbdt.py --param kcat    --kind lightgbm --blocks prott5 esm2 esmc maccs morgan cbmean --tag allplm  # ESM-C GBDT (kcat)
$PY scripts/compositional_kcatkm.py                                                                 # composed (kcat/Km)

echo "== (2) Final leakage-free stacked ensemble =="
$PY scripts/final_report.py | tee results/final_table.txt
echo "Done. See results/final_table.txt and docs/RESULTS.md"
