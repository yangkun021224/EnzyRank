# Benchmark data

EnzyRank is evaluated on the **CataPro** enzyme-kinetics benchmark (Nat. Commun. 2025), reused
verbatim so the comparison is minimum-fair (identical data, identical folds, identical metrics).

```
data/catapro_benchmark/
├── kcat-data_0.4simi-10fold.csv          # turnover number  kcat        (27,658 rows)
├── Km-data_0.4simi-10fold.csv            # Michaelis const. Km          (42,018 rows)
└── kcat-over-Km-data_0.4simi-10fold.csv  # catalytic eff.   kcat/Km     (25,831 rows)
```

Each row is an (enzyme sequence, substrate SMILES, measured value) triple with a `fold` column
(integers 0–9) giving CataPro's pre-computed **0.4-sequence-similarity clustered 10-fold split**:
enzymes in different folds share < 0.4 sequence similarity, so the cross-validation measures
generalisation to dissimilar enzymes (no train/test leakage). We use these folds exactly as
provided — we do not re-split.

## Target / unit conventions (identical to CataPro)

Metrics (PCC, SCC, RMSE) are computed in the following spaces (`src/eval/metrics.py`):

| Parameter | Regression target | Unit |
|-----------|-------------------|------|
| kcat    | `log10(kcat)`                    | kcat in s⁻¹ |
| Km      | `log10(Km_M) + 3`                | Km in mM |
| kcat/Km | `log10(kcat) − log10(Km_M) − 3`  | kcat/Km in mM⁻¹ s⁻¹ |

## Provenance
Files are copied unmodified from the CataPro release
(`catapro_method/datasets/`). See the CataPro paper for how the measurements were curated from
BRENDA/SABIO-RK and how the similarity split was constructed.
