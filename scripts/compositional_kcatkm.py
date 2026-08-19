"""Compositional kcat/Km: predict log(kcat) and log(Km) separately on the kcat_km dataset, then
derive log(kcat/Km) = log(kcat) - log(Km[mM]). The ratio is noisier than its parts, so composing
two well-predicted components can beat direct ratio regression (physical-consistency idea).

Outputs OOF for both components and the composed prediction for ensembling.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_dataset  # noqa: E402
from src.eval.cv import run_cv  # noqa: E402
from src.eval.metrics import CATAPRO_TARGET, compute_all  # noqa: E402
from src.models.gbdt_model import GBDTFoldModel, TabularCV  # noqa: E402
from src.utils.common import set_seed  # noqa: E402

BLOCKS = ("prott5", "esm2", "maccs", "morgan", "cbmean")


def main():
    set_seed(42)
    ds = load_dataset("kcat_km")
    df = ds.df
    log_kcat = np.log10(df["kcat(s^-1)"].to_numpy(dtype=np.float64))
    log_km = np.log10(df["Km(M)"].to_numpy(dtype=np.float64)) + 3.0  # mM
    X = TabularCV(ds, blocks=BLOCKS).X

    oof = {}
    for name, y in [("kcat", log_kcat), ("km", log_km)]:
        tab = type("T", (), {"X": X, "y": y})()
        # target-encode EC/Organism helps the component regressions
        def factory():
            return GBDTFoldModel(tab, kind="lightgbm", threads=48, n_estimators=1200,
                                 params={"num_leaves": 63}, te_cols=("EC", "Organism"))
        res = run_cv(ds, factory, verbose=False)  # ds only used for fold ids + te cols
        oof[name] = res["oof_pred"]
        cm = compute_all(y, res["oof_pred"])  # component metric vs its OWN target
        print(f"  component {name}: PCC={cm['PCC']:.4f} RMSE={cm['RMSE']:.4f}", flush=True)

    composed = oof["kcat"] - oof["km"]
    m = compute_all(ds.target, composed)
    t = CATAPRO_TARGET["kcat_km"]
    print("\n=== kcat_km COMPOSED (kcat - km) ===")
    print(f"  PCC={m['PCC']:.4f} SCC={m['SCC']:.4f} RMSE={m['RMSE']:.4f}  "
          f"(CataPro {t['PCC']}/{t['RMSE']}; direct GBDT was 0.369)")
    Path("results/oof").mkdir(parents=True, exist_ok=True)
    np.save("results/oof/composed_kcat_km.npy", composed)


if __name__ == "__main__":
    main()
