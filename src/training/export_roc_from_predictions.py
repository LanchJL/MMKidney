import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.training.prognosis_metrics import horizon_binary_labels, safe_auc


def _read_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _roc_points(y_true: List[int], y_score: List[float]) -> Tuple[List[float], List[float], List[float]]:
    try:
        from sklearn.metrics import roc_curve
    except Exception as e:
        raise RuntimeError("scikit-learn is required for ROC curve export.") from e
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr.tolist(), tpr.tolist(), thr.tolist()


def main():
    p = argparse.ArgumentParser("Export horizon ROC points from prediction csv")
    p.add_argument("--pred-csv", required=True, help="e.g. outputs/.../val_predictions.csv")
    p.add_argument("--out-dir", required=True, help="where roc csv/json are written")
    p.add_argument("--horizons-days", default="365,1095,1825")
    args = p.parse_args()

    rows = _read_csv(Path(args.pred_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = [float(r.get("time_days", 0.0) or 0.0) for r in rows]
    events = [int(float(r.get("event", 0.0) or 0.0)) for r in rows]
    horizons = [int(float(x.strip())) for x in args.horizons_days.split(",") if x.strip()]

    summary = {"source_pred_csv": args.pred_csv, "horizons": {}}

    for h in horizons:
        key = f"risk_{h}d"
        if key not in (rows[0].keys() if rows else []):
            summary["horizons"][str(h)] = {"error": f"column {key} not found"}
            continue
        scores = [float(r.get(key, 0.0) or 0.0) for r in rows]
        idx, y = horizon_binary_labels(times, events, h)
        p_used = [scores[i] for i in idx]
        auc = safe_auc(y, p_used)

        if len(y) == 0 or len(set(y)) < 2:
            summary["horizons"][str(h)] = {
                "n_used": len(y),
                "auc": auc,
                "note": "ROC undefined: only one class in used samples",
            }
            continue

        fpr, tpr, thr = _roc_points(y, p_used)
        roc_rows = [{"fpr": float(a), "tpr": float(b), "threshold": float(c)} for a, b, c in zip(fpr, tpr, thr)]
        _write_csv(out_dir / f"roc_{h}d.csv", roc_rows, ["fpr", "tpr", "threshold"])
        summary["horizons"][str(h)] = {
            "n_used": len(y),
            "auc": auc,
            "roc_csv": str(out_dir / f"roc_{h}d.csv"),
        }

    with (out_dir / "roc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[roc-export] summary:", out_dir / "roc_summary.json")


if __name__ == "__main__":
    main()
