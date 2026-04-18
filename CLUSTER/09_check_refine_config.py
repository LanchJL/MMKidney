import argparse
import json
import os
from typing import Dict, Any

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file format: {path}")


def build_refine_config(
    size_df: pd.DataFrame,
    min_refine: int = 30,
    min_l2: int = 15,
) -> Dict[str, Dict[str, Any]]:
    cfg: Dict[str, Dict[str, Any]] = {}
    for _, r in size_df.iterrows():
        c = str(r["proto_cluster"])
        n = int(r["n_prototypes"])

        # Conservative defaults close to original CLUSTER strategy
        enabled = n >= min_refine
        resolution = 0.60 if n >= 120 else 0.45
        n_neighbors = 12 if n >= 60 else 10
        n_pcs = 30

        cfg[c] = {
            "enabled": bool(enabled),
            "n_pcs": int(n_pcs),
            "n_neighbors": int(n_neighbors),
            "resolution": float(resolution),
            "umap_min_dist": 0.25,
            "umap_spread": 1.0,
            "min_prototypes_to_refine": int(min_refine),
            "min_l2_prototypes": int(min_l2),
        }
    return cfg


def main():
    p = argparse.ArgumentParser("Check L1 clusters and suggest REFINE_CONFIG")
    p.add_argument("--output-dir", default="prototype_pipeline_output")
    p.add_argument("--target-stain", default="HE")
    p.add_argument("--n-prototypes", type=int, default=2048)
    p.add_argument("--min-refine", type=int, default=30)
    p.add_argument("--min-l2", type=int, default=15)
    args = p.parse_args()

    size_csv = os.path.join(
        args.output_dir,
        f"prototype_cluster_sizes_{args.target_stain}_{args.n_prototypes}.csv",
    )
    cluster_csv = os.path.join(
        args.output_dir,
        f"prototype_clusters_{args.target_stain}_{args.n_prototypes}.csv",
    )
    usage_csv = os.path.join(
        args.output_dir,
        f"prototype_usage_{args.target_stain}_{args.n_prototypes}.csv",
    )

    size_df = load_table(size_csv).copy()
    size_df["proto_cluster"] = size_df["proto_cluster"].astype(str)
    size_df = size_df.sort_values("n_prototypes", ascending=False).reset_index(drop=True)

    cluster_df = load_table(cluster_csv).copy()
    cluster_df["proto_cluster"] = cluster_df["proto_cluster"].astype(str)

    usage_df = None
    if os.path.exists(usage_csv):
        usage_df = load_table(usage_csv).copy()

    report = {
        "n_l1_clusters": int(size_df["proto_cluster"].nunique()),
        "n_prototypes_total": int(size_df["n_prototypes"].sum()),
        "largest_cluster": {
            "proto_cluster": str(size_df.iloc[0]["proto_cluster"]),
            "n_prototypes": int(size_df.iloc[0]["n_prototypes"]),
            "ratio": float(size_df.iloc[0]["n_prototypes"] / max(1, size_df["n_prototypes"].sum())),
        },
    }

    if usage_df is not None and "n_patches" in usage_df.columns:
        merged = cluster_df.merge(usage_df, on="proto_id", how="left")
        agg = (
            merged.groupby("proto_cluster", as_index=False)["n_patches"]
            .sum()
            .sort_values("n_patches", ascending=False)
        )
        agg["patch_ratio"] = agg["n_patches"] / max(1, agg["n_patches"].sum())
        report["patch_mass_by_l1"] = agg.to_dict(orient="records")
    else:
        report["patch_mass_by_l1"] = []

    suggest_cfg = build_refine_config(
        size_df=size_df,
        min_refine=args.min_refine,
        min_l2=args.min_l2,
    )

    out_json = os.path.join(args.output_dir, "refine_config_suggestion.json")
    out_txt = os.path.join(args.output_dir, "refine_config_suggestion.txt")
    report_json = os.path.join(args.output_dir, "l1_cluster_check_report.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(suggest_cfg, f, ensure_ascii=False, indent=2)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("REFINE_CONFIG = ")
        f.write(json.dumps(suggest_cfg, ensure_ascii=False, indent=2))
        f.write("\n")

    print("[check] L1 summary:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] {report_json}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_txt}")
    print(
        "[next] Copy refine_config_suggestion.txt into "
        "CLUSTER/05b_refine_prototype_clusters.py (REFINE_CONFIG)."
    )


if __name__ == "__main__":
    main()

