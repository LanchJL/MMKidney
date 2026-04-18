import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser("Check batch L3 refinement results")
    p.add_argument(
        "--summary-csv",
        default="",
        help="Path to batch_l3_summary.csv; if empty, auto-pick latest under output-dir/batch_l3_runs",
    )
    p.add_argument("--output-dir", default="prototype_pipeline_output")
    p.add_argument("--top1-ratio-thr", type=float, default=0.30)
    p.add_argument("--min-l3-kept", type=int, default=2)
    args = p.parse_args()

    summary_csv = args.summary_csv
    if not summary_csv:
        root = Path(args.output_dir) / "batch_l3_runs"
        if not root.exists():
            raise FileNotFoundError(f"batch run dir not found: {root}")
        cands = sorted(root.glob("*/batch_l3_summary.csv"))
        if not cands:
            raise FileNotFoundError(f"no batch_l3_summary.csv found under: {root}")
        summary_csv = str(cands[-1])

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(summary_csv)

    df = pd.read_csv(summary_csv)
    for c in ["status", "target_l2", "top1_cluster"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["n_l3_kept", "top1_ratio", "n_clusters"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    decisions = []
    for _, r in df.iterrows():
        status = str(r.get("status", ""))
        n_l3 = int(r.get("n_l3_kept", 0))
        top1_ratio = float(r.get("top1_ratio", 1.0))
        target = str(r.get("target_l2", ""))

        if status != "refined":
            rec = "DROP"
            reason = "not_refined"
        elif n_l3 < args.min_l3_kept:
            rec = "DROP"
            reason = f"n_l3_kept<{args.min_l3_kept}"
        elif top1_ratio > args.top1_ratio_thr:
            rec = "REVIEW"
            reason = f"top1_ratio>{args.top1_ratio_thr}"
        else:
            rec = "KEEP"
            reason = "balanced_and_refined"

        decisions.append(
            {
                "target_l2": target,
                "status": status,
                "n_l3_kept": n_l3,
                "top1_ratio": round(top1_ratio, 6),
                "n_clusters": int(r.get("n_clusters", 0)),
                "recommendation": rec,
                "reason": reason,
            }
        )

    out_df = pd.DataFrame(decisions).sort_values(
        ["recommendation", "top1_ratio", "n_l3_kept"], ascending=[True, True, False]
    )

    out_json = str(Path(summary_csv).with_name("batch_l3_check_report.json"))
    out_csv = str(Path(summary_csv).with_name("batch_l3_check_report.csv"))
    out_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"[check] summary_csv: {summary_csv}")
    print("[check] decisions:")
    print(out_df.to_string(index=False))
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()

