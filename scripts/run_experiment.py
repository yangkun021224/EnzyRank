"""Run one innovation configuration through fold-exact 10-fold CV and report vs CataPro.

Usage:
    python scripts/run_experiment.py --config I1_attnpool --param kcat
    python scripts/run_experiment.py --config I3_multitask --param kcat_km

Configs are additive-ablation presets over ModelConfig (see docs/INNOVATION_RANKING.md).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_dataset  # noqa: E402
from src.eval.cv import report, run_cv  # noqa: E402
from src.features.providers import ProteinResidueProvider, SubstrateProvider  # noqa: E402
from src.models.fold_model import NNFoldModel  # noqa: E402
from src.models.interaction_model import ModelConfig  # noqa: E402
from src.utils.common import save_json, set_seed  # noqa: E402

MORGAN_BITS = 2048
SUB_TOKEN_DIM = 167 + MORGAN_BITS
FP_DIM = 167  # MACCS at fusion


def make_config(name: str, param: str) -> tuple[ModelConfig, str, dict]:
    """Return (ModelConfig, primary_task, trainer_kwargs) for a preset name."""
    base = dict(protein_dim=1280, substrate_dim=SUB_TOKEN_DIM, fp_dim=FP_DIM,
                d_model=256, pool_heads=4, cross_heads=8, dropout=0.25)
    tk = dict(lr=3e-4, weight_decay=2e-2, epochs=35, patience=7, consistency_w=0.0)

    if name == "E1_meanpool":       # enzyme+substrate baseline, no I1/I2/I3
        cfg = ModelConfig(**base, pooling="mean", n_cross_layers=0, tasks=[param])
        return cfg, param, tk
    if name == "I1_attnpool":       # + attention pooling (I1)
        cfg = ModelConfig(**base, pooling="attention", n_cross_layers=0, tasks=[param])
        return cfg, param, tk
    if name == "I2_crossattn":      # + cross-attention (I1+I2)
        cfg = ModelConfig(**base, pooling="attention", n_cross_layers=2, tasks=[param])
        return cfg, param, tk
    if name == "I3_multitask":      # + multi-task & consistency (I1+I2+I3); kcat_km dataset
        cfg = ModelConfig(**base, pooling="attention", n_cross_layers=2,
                          tasks=["kcat", "km", "kcat_km"])
        tk["consistency_w"] = 0.1
        return cfg, param, tk
    raise ValueError(f"unknown config {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--param", required=True, choices=["kcat", "km", "kcat_km"])
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--val_mode", default="fold", choices=["fold", "random"])
    ap.add_argument("--substrate", default="fp", choices=["fp", "chemberta"])
    ap.add_argument("--protein", default="prott5", choices=["esm2", "prott5", "esmc"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corr", action="store_true",
                    help="N2: correlation-aligned loss (soft-Spearman + Pearson + small MSE)")
    args = ap.parse_args()

    set_seed(args.seed)
    ds = load_dataset(args.param)
    cfg, primary, tk = make_config(args.config, args.param)
    if args.epochs:
        tk["epochs"] = args.epochs
    if args.corr:
        tk["corr_weights"] = {"mse_w": 0.2, "pearson_w": 1.0, "spearman_w": 0.5, "temp": 0.1}
        args.config = f"{args.config}_corr"

    from src.paths import CACHE_DIR
    prot_cfg = {"esm2": (str(CACHE_DIR/"esm2_residue.h5"), 1280),
                "prott5": (str(CACHE_DIR/"prott5_residue.h5"), 1024),
                "esmc": (str(CACHE_DIR/"esmc_residue.h5"), 1152)}[args.protein]
    prot = ProteinResidueProvider(h5_path=prot_cfg[0])
    cfg.protein_dim = prot_cfg[1]
    sub = SubstrateProvider(mode=args.substrate, morgan_bits=MORGAN_BITS, flat_kinds=("maccs",))
    cfg.substrate_dim = sub.token_dim
    tag = []
    if args.protein != "esm2":
        tag.append(args.protein)
    if args.substrate != "fp":
        tag.append(args.substrate)
    if tag:
        args.config = f"{args.config}_{'_'.join(tag)}"
    prot.prime(list(ds.sequence))
    sub.prime(list(ds.smiles))
    from src.training.gpu_batcher import GpuFeatureStore
    store = GpuFeatureStore(prot, sub, list(ds.sequence), list(ds.smiles),
                            use_fp=cfg.fp_dim > 0, device="cuda")

    if args.seed != 42:
        args.config = f"{args.config}_s{args.seed}"

    def factory():
        return NNFoldModel(cfg, primary_task=primary, prot=prot, sub=sub, store=store,
                           trainer_kwargs=tk, batch_size=args.batch_size, val_mode=args.val_mode,
                           seed=args.seed)

    print(f"=== config={args.config} param={args.param} tasks={cfg.tasks} "
          f"pooling={cfg.pooling} cross={cfg.n_cross_layers} ===")
    res = run_cv(ds, factory, folds=args.folds, verbose=True)
    print(report(res))

    out = {"config": args.config, "param": args.param,
           "pooled": res["pooled"], "per_fold_agg": res["per_fold_agg"]}
    save_json(out, f"results/{args.config}_{args.param}.json")
    # persist out-of-fold predictions for later ensembling
    import numpy as np
    Path("results/oof").mkdir(parents=True, exist_ok=True)
    np.save(f"results/oof/{args.config}_{args.param}.npy", res["oof_pred"])


if __name__ == "__main__":
    main()
