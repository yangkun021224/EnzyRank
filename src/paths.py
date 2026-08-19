"""Centralised, portable paths for the project.

All resources (datasets, feature caches, pretrained weights) are resolved relative to the project
root so the repository is self-contained. Large pretrained models (>1 GB) live under `pretrained/`
by default; override the location with the `ENZYRANK_PRETRAINED` environment variable if desired.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "catapro_benchmark"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
PRETRAINED_DIR = Path(os.getenv("ENZYRANK_PRETRAINED", PROJECT_ROOT / "pretrained"))

# Pretrained model directories (see pretrained/README.md for download links).
ESM2_650M_DIR = PRETRAINED_DIR / "esm2_t33_650M_UR50D"
PROTT5_DIR = PRETRAINED_DIR / "prot_t5_xl_half_uniref50-enc"
CHEMBERTA_DIR = PRETRAINED_DIR / "ChemBERTa-77M-MTR"
ESM2_35M_DIR = PRETRAINED_DIR / "esm2_t12_35M_UR50D"
