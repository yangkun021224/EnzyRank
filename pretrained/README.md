# Pretrained models

EnzyRank uses four frozen pretrained encoders to featurise enzymes and substrates. All are resolved
from this directory by `src/paths.py` (override the location with the `ENZYRANK_PRETRAINED`
environment variable).

Per the packaging policy, the small model (< 1 GB) is **bundled in this repository**; the large
models (> 1 GB) are **downloaded from Hugging Face** and placed here. After download, the layout
must be exactly:

```
pretrained/
├── ChemBERTa-77M-MTR/              # bundled (14 MB) — no download needed
├── prot_t5_xl_half_uniref50-enc/   # download
├── esm2_t33_650M_UR50D/            # download
└── ESMC-600M/                      # download (must contain model.safetensors)
```

| Model | Role | Size | Bundled? | Source |
|-------|------|------|----------|--------|
| **ChemBERTa-77M-MTR** | substrate SMILES encoder | 14 MB | ✅ in-repo | https://huggingface.co/DeepChem/ChemBERTa-77M-MTR |
| **ProtT5-XL-U50 (enc, half)** | enzyme encoder (1024-d) | ~2.3 GB | ⬇ download | https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc |
| **ESM2-650M** | enzyme encoder (1280-d) | ~2.5 GB | ⬇ download | https://huggingface.co/facebook/esm2_t33_650M_UR50D |
| **ESM-C 600M** | enzyme encoder (1152-d), generational upgrade | ~2.2 GB | ⬇ download | https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12 |

## Download (Hugging Face CLI)

```bash
pip install -U huggingface_hub
cd pretrained
huggingface-cli download Rostlab/prot_t5_xl_half_uniref50-enc --local-dir prot_t5_xl_half_uniref50-enc
huggingface-cli download facebook/esm2_t33_650M_UR50D        --local-dir esm2_t33_650M_UR50D
huggingface-cli download EvolutionaryScale/esmc-600m-2024-12 --local-dir ESMC-600M
```

Notes
- **ESM-C is loaded without the `esm` SDK download.** EnzyRank builds the SDK's `ESMC` architecture
  in memory and loads the Hugging Face `model.safetensors` directly via a key remap
  (`src/features/protein_esmc.py`), so only the HF weights above are needed.
- The download models are **not required to reproduce the headline numbers** — the out-of-fold
  predictions in `results/oof/` and `baseline_v1_oof/` already reproduce the final table via
  `python scripts/final_report.py`. The encoders are only needed to regenerate feature caches and
  retrain base members from scratch.
- These large directories are excluded from git (`.gitignore`); the table above is the source of
  truth for obtaining them.
