import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser("Plan candidate L2 targets for local L3 refinement")
    p.add_argument("--output-dir", default="prototype_pipeline_output")
    p.add_argument("--target-stain", default="HE")
    p.add_argument("--n-prototypes", type=int, default=2048)
    p.add_argument("--min-patches", type=int, default=100000)
    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--only-l2", action="store_true", help="Only consider clusters like '<L1>_<L2>'")
    p.add_argument("--out-json", default="")
    args = p.parse_args()

    size_csv = os.path.join(
        args.output_dir, f"final_cluster_sizes_{args.target_stain}_{args.n_prototypes}.csv"
    )
    if not os.path.exists(size_csv):
        raise FileNotFoundError(size_csv)

    df = pd.read_csv(size_csv).copy()
    df["final_cluster"] = df["final_cluster"].astype(str)
    df["n_patches"] = pd.to_numeric(df["n_patches"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("n_patches", ascending=False).reset_index(drop=True)

    cand = df[df["n_patches"] >= args.min_patches].copy()
    if args.only_l2:
        cand = cand[cand["final_cluster"].str.contains("_", regex=False)].copy()
    if args.top_n > 0:
        cand = cand.head(args.top_n).copy()

    targets = cand["final_cluster"].astype(str).tolist()
    targets_csv = ",".join(targets)

    out = {
        "source": size_csv,
        "min_patches": int(args.min_patches),
        "top_n": int(args.top_n),
        "only_l2": bool(args.only_l2),
        "n_candidates": int(len(targets)),
        "targets": targets,
        "targets_csv": targets_csv,
    }

    out_json = args.out_json or os.path.join(args.output_dir, "l3_target_plan.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[plan] L3 target plan:")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()

