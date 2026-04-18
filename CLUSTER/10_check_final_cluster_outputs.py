import argparse
import json
import os
from typing import Dict, Any

import pandas as pd


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def main():
    p = argparse.ArgumentParser("Check final cluster size and overlay color outputs")
    p.add_argument(
        "--final-sizes",
        default="prototype_pipeline_output/final_cluster_sizes_HE_2048.csv",
        help="Path to final_cluster_sizes_HE_2048.csv",
    )
    p.add_argument(
        "--color-sizes",
        default="prototype_pipeline_output/overlay_final_clusters_HE_2048/final_cluster_colors.csv",
        help="Path to overlay final_cluster_colors.csv",
    )
    p.add_argument(
        "--out-json",
        default="prototype_pipeline_output/final_cluster_check_report.json",
        help="Where to save report json",
    )
    args = p.parse_args()

    final_df = _load_csv(args.final_sizes).copy()
    color_df = _load_csv(args.color_sizes).copy()

    need_final_cols = {"final_cluster", "n_patches"}
    need_color_cols = {"final_cluster", "n_patches"}
    miss_final = sorted(list(need_final_cols - set(final_df.columns)))
    miss_color = sorted(list(need_color_cols - set(color_df.columns)))
    if miss_final:
        raise ValueError(f"Missing columns in final sizes: {miss_final}")
    if miss_color:
        raise ValueError(f"Missing columns in color sizes: {miss_color}")

    final_df["final_cluster"] = final_df["final_cluster"].astype(str)
    color_df["final_cluster"] = color_df["final_cluster"].astype(str)
    final_df["n_patches"] = _to_num(final_df["n_patches"])
    color_df["n_patches"] = _to_num(color_df["n_patches"])

    final_df = final_df.sort_values("n_patches", ascending=False).reset_index(drop=True)
    color_df = color_df.sort_values("n_patches", ascending=False).reset_index(drop=True)

    total_final = int(final_df["n_patches"].sum())
    total_color = int(color_df["n_patches"].sum())
    n_clusters_final = int(final_df["final_cluster"].nunique())
    n_clusters_color = int(color_df["final_cluster"].nunique())

    top1_cluster = str(final_df.iloc[0]["final_cluster"]) if len(final_df) else ""
    top1_count = int(final_df.iloc[0]["n_patches"]) if len(final_df) else 0
    top1_ratio = float(top1_count / max(1, total_final))

    set_final = set(final_df["final_cluster"].tolist())
    set_color = set(color_df["final_cluster"].tolist())
    only_final = sorted(list(set_final - set_color))
    only_color = sorted(list(set_color - set_final))

    merged = final_df.merge(
        color_df[["final_cluster", "n_patches"]],
        on="final_cluster",
        how="outer",
        suffixes=("_final", "_color"),
    ).fillna(0)
    merged["n_patches_final"] = _to_num(merged["n_patches_final"])
    merged["n_patches_color"] = _to_num(merged["n_patches_color"])
    merged["abs_diff"] = (merged["n_patches_final"] - merged["n_patches_color"]).abs()
    total_abs_diff = int(merged["abs_diff"].sum())
    max_abs_diff = int(merged["abs_diff"].max()) if len(merged) else 0

    small_clusters = (
        final_df[final_df["n_patches"] < max(100, 0.001 * max(1, total_final))]
        ["final_cluster"]
        .astype(str)
        .tolist()
    )

    flags = []
    if top1_ratio > 0.90:
        flags.append("top1_ratio_gt_0.90")
    elif top1_ratio > 0.80:
        flags.append("top1_ratio_gt_0.80")
    if total_final != total_color:
        flags.append("total_patch_count_mismatch")
    if only_final or only_color:
        flags.append("cluster_set_mismatch")
    if max_abs_diff > 0:
        flags.append("per_cluster_count_mismatch")

    report: Dict[str, Any] = {
        "final_sizes_path": args.final_sizes,
        "color_sizes_path": args.color_sizes,
        "n_clusters_final": n_clusters_final,
        "n_clusters_color": n_clusters_color,
        "n_patches_final": total_final,
        "n_patches_color": total_color,
        "top1_cluster": top1_cluster,
        "top1_count": top1_count,
        "top1_ratio": top1_ratio,
        "only_in_final_sizes": only_final,
        "only_in_color_sizes": only_color,
        "total_abs_diff_between_tables": total_abs_diff,
        "max_abs_diff_between_tables": max_abs_diff,
        "n_small_clusters": int(len(small_clusters)),
        "small_clusters": small_clusters[:30],
        "flags": flags,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[check] final cluster outputs summary:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] {args.out_json}")
    if flags:
        print("[warn] flags:", ", ".join(flags))
    else:
        print("[ok] no consistency flags")


if __name__ == "__main__":
    main()

