import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _nanmean(xs: List[float]) -> float:
    ys = [x for x in xs if not math.isnan(x)]
    if not ys:
        return float("nan")
    return float(sum(ys) / len(ys))


def main():
    p = argparse.ArgumentParser("Repeated stratified group K-fold for prognosis training")
    p.add_argument("--prognosis-cohort", default="data/processed/prognosis_cohort.csv")
    p.add_argument("--feature-manifest", default="data/processed/prognosis_feature_manifest.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--out-dir", default="outputs/prognosis_cv")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-patches", type=int, default=512)
    p.add_argument("--survival-head", choices=["cox", "discrete"], default="discrete")
    p.add_argument("--time-bins-days", default="365,1095,1825")
    p.add_argument("--horizons-days", default="365,1095,1825")
    args = p.parse_args()

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as e:
        raise RuntimeError("crossval_prognosis requires scikit-learn with StratifiedGroupKFold.") from e

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(Path(args.prognosis_cohort))
    if not rows:
        raise RuntimeError("Empty prognosis cohort.")

    y = [int(float(r.get("event", "0") or 0.0)) for r in rows]
    groups = [str(r.get("patient_id", "") or r.get("patient_index", "") or r.get("sample_id", "")) for r in rows]
    fieldnames = list(rows[0].keys())

    run_summaries = []
    fold_scores = []
    fold_id = 0
    for rep in range(args.n_repeats):
        cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + rep)
        for k, (tr_idx, va_idx) in enumerate(cv.split(rows, y, groups)):
            fold_id += 1
            fold_dir = out_root / f"rep_{rep+1:02d}_fold_{k+1:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            cohort_fold = []
            tr_set = set(int(i) for i in tr_idx)
            va_set = set(int(i) for i in va_idx)
            for i, r in enumerate(rows):
                rr = dict(r)
                if i in tr_set:
                    rr["split"] = "train"
                elif i in va_set:
                    rr["split"] = "val"
                else:
                    rr["split"] = "test"
                cohort_fold.append(rr)
            cohort_path = fold_dir / "prognosis_cohort_fold.csv"
            _write_csv(cohort_path, cohort_fold, fieldnames=fieldnames)

            cmd = [
                sys.executable,
                "-m",
                "src.training.train_prognosis",
                "--prognosis-cohort",
                str(cohort_path),
                "--feature-manifest",
                args.feature_manifest,
                "--tabular-features",
                args.tabular_features,
                "--lab-features",
                args.lab_features,
                "--stain-vocab",
                args.stain_vocab,
                "--train-manifest",
                args.train_manifest,
                "--val-manifest",
                args.val_manifest,
                "--test-manifest",
                args.test_manifest,
                "--out-dir",
                str(fold_dir / "model"),
                "--seed",
                str(args.seed + fold_id),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--max-patches",
                str(args.max_patches),
                "--survival-head",
                args.survival_head,
                "--time-bins-days",
                args.time_bins_days,
                "--horizons-days",
                args.horizons_days,
            ]
            print("[cv] running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            best_path = fold_dir / "model" / "val_best.json"
            best = json.loads(best_path.read_text(encoding="utf-8"))
            cidx = float(best.get("c_index", float("nan")))
            fold_scores.append(cidx)
            run_summaries.append(
                {
                    "rep": rep + 1,
                    "fold": k + 1,
                    "val_c_index": cidx,
                    "dir": str(fold_dir),
                }
            )
            print(f"[cv] rep={rep+1} fold={k+1} val_c_index={cidx:.4f}")

    summary = {
        "n_runs": len(run_summaries),
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "mean_val_c_index": _nanmean(fold_scores),
        "runs": run_summaries,
    }
    (out_root / "cv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[cv] done. mean_val_c_index:", summary["mean_val_c_index"])
    print("[cv] summary:", out_root / "cv_summary.json")


if __name__ == "__main__":
    main()
