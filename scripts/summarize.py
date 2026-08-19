"""Aggregate all results/*.json into a leaderboard vs the CataPro target rows."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import CATAPRO_TARGET  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    rows = []
    for f in sorted(RES.glob("*.json")):
        d = json.load(open(f))
        p = d["pooled"]
        rows.append((d["config"], d["param"], p["PCC"], p["SCC"], p["RMSE"]))

    for param in ["kcat", "km", "kcat_km"]:
        t = CATAPRO_TARGET[param]
        print(f"\n=== {param}  (CataPro target: PCC {t['PCC']} / SCC {t['SCC']} / RMSE {t['RMSE']}) ===")
        print(f"{'config':<22}{'PCC':>8}{'SCC':>8}{'RMSE':>8}{'beats':>8}")
        sub = sorted([r for r in rows if r[1] == param], key=lambda r: -r[2])
        for cfg, _, pcc, scc, rmse in sub:
            beats = "YES" if (pcc > t["PCC"] and rmse < t["RMSE"]) else \
                    ("PCC" if pcc > t["PCC"] else "no")
            print(f"{cfg:<22}{pcc:>8.4f}{scc:>8.4f}{rmse:>8.4f}{beats:>8}")


if __name__ == "__main__":
    main()
