import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_table_auto(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def save_table_auto(df: pd.DataFrame, out_base: str) -> str:
    pq = f"{out_base}.parquet"
    try:
        df.to_parquet(pq, index=False)
        return pq
    except Exception:
        csv = f"{out_base}.csv"
        df.to_csv(csv, index=False)
        return csv


def parse_targets(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def build_proto_map_for_target(proto_csv: str, target_l2: str) -> Dict[int, str]:
    df = pd.read_csv(proto_csv).copy()
    for c in ["proto_id", "proto_cluster_L2", "final_cluster_l3"]:
        if c not in df.columns:
            raise ValueError(f"{proto_csv} missing column: {c}")
    df["proto_id"] = pd.to_numeric(df["proto_id"], errors="coerce").fillna(-1).astype(int)
    df["proto_cluster_L2"] = df["proto_cluster_L2"].astype(str)
    df["final_cluster_l3"] = df["final_cluster_l3"].astype(str)
    sub = df[df["proto_cluster_L2"] == str(target_l2)].copy()
    if len(sub) == 0:
        return {}
    return dict(zip(sub["proto_id"].tolist(), sub["final_cluster_l3"].tolist()))


def main():
    p = argparse.ArgumentParser("Build final patch version from selected local-L3 targets")
    p.add_argument("--output-dir", default="prototype_pipeline_output")
    p.add_argument("--target-stain", default="HE")
    p.add_argument("--n-prototypes", type=int, default=2048)
    p.add_argument("--keep-targets", default="2_0,2_1,2_2,2_3,4_2")
    p.add_argument(
        "--base-patch-final",
        default="",
        help="Default: output-dir/patch_final_clusters_<stain>_<k>.parquet",
    )
    p.add_argument("--out-tag", default="final_l3_selected_v1")
    args = p.parse_args()

    od = args.output_dir
    stain = args.target_stain
    k = args.n_prototypes
    targets = parse_targets(args.keep_targets)

    if not targets:
        raise ValueError("No keep targets provided.")

    if args.base_patch_final:
        base_path = args.base_patch_final
    else:
        base_path = os.path.join(od, f"patch_final_clusters_{stain}_{k}.parquet")
        if not os.path.exists(base_path):
            alt = os.path.join(od, f"patch_final_clusters_{stain}_{k}.pkl")
            if os.path.exists(alt):
                base_path = alt
            else:
                alt = os.path.join(od, f"patch_final_clusters_{stain}_{k}.csv")
                base_path = alt

    patch_df = load_table_auto(base_path).copy()
    if "proto_id" not in patch_df.columns or "final_cluster" not in patch_df.columns:
        raise ValueError("Base patch_final must contain columns: proto_id, final_cluster")

    patch_df["proto_id"] = pd.to_numeric(patch_df["proto_id"], errors="coerce").fillna(-1).astype(int)
    patch_df["final_cluster"] = patch_df["final_cluster"].astype(str)
    patch_df["final_cluster_base"] = patch_df["final_cluster"].astype(str)

    applied_stats: List[Dict] = []
    missing_targets: List[str] = []

    for t in targets:
        proto_csv = os.path.join(od, f"prototype_clusters_L1L2L3_target_{t}_{stain}_{k}.csv")
        if not os.path.exists(proto_csv):
            missing_targets.append(t)
            continue

        mp = build_proto_map_for_target(proto_csv, t)
        if not mp:
            applied_stats.append(
                {"target_l2": t, "status": "no_rows_in_target", "n_proto_in_map": 0, "n_patches_changed": 0}
            )
            continue

        before = patch_df["final_cluster"].copy()
        idx = patch_df["proto_id"].isin(mp.keys())
        patch_df.loc[idx, "final_cluster"] = patch_df.loc[idx, "proto_id"].map(mp).astype(str)
        n_changed = int((before != patch_df["final_cluster"]).sum())

        applied_stats.append(
            {
                "target_l2": t,
                "status": "applied",
                "n_proto_in_map": int(len(mp)),
                "n_patches_in_target": int(idx.sum()),
                "n_patches_changed": int(n_changed),
                "proto_csv": proto_csv,
            }
        )

    id_col = "slide_id" if "slide_id" in patch_df.columns else ("sample_id" if "sample_id" in patch_df.columns else "")
    if not id_col:
        raise ValueError("Base patch_final must contain slide_id or sample_id")

    size_df = (
        patch_df["final_cluster"]
        .value_counts()
        .rename_axis("final_cluster")
        .reset_index(name="n_patches")
        .sort_values("n_patches", ascending=False)
        .reset_index(drop=True)
    )
    total = int(size_df["n_patches"].sum()) if len(size_df) else 0
    top1 = str(size_df.iloc[0]["final_cluster"]) if len(size_df) else ""
    top1_ratio = float(size_df.iloc[0]["n_patches"] / max(1, total)) if len(size_df) else 0.0

    hist_df = patch_df.groupby([id_col, "final_cluster"]).size().reset_index(name="count")
    total_df = patch_df.groupby(id_col).size().reset_index(name="slide_total")
    hist_df = hist_df.merge(total_df, on=id_col, how="left")
    hist_df["fraction"] = hist_df["count"] / hist_df["slide_total"]

    out_root = os.path.join(od, "final_versions", args.out_tag)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    out_patch = save_table_auto(patch_df, os.path.join(out_root, f"patch_final_clusters_{stain}_{k}_{args.out_tag}"))
    out_sizes = save_table_auto(size_df, os.path.join(out_root, f"final_cluster_sizes_{stain}_{k}_{args.out_tag}"))
    out_hist = save_table_auto(hist_df, os.path.join(out_root, f"slide_final_cluster_hist_{stain}_{k}_{args.out_tag}"))

    meta = {
        "out_tag": args.out_tag,
        "output_dir": od,
        "base_patch_final": base_path,
        "kept_targets": targets,
        "missing_targets": missing_targets,
        "applied_stats": applied_stats,
        "n_clusters": int(size_df["final_cluster"].nunique()),
        "n_patches": int(total),
        "top1_cluster": top1,
        "top1_ratio": float(top1_ratio),
        "outputs": {
            "patch_final": out_patch,
            "final_cluster_sizes": out_sizes,
            "slide_hist": out_hist,
        },
    }

    meta_json = os.path.join(out_root, "final_version_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    notes_md = os.path.join(out_root, "final_version_notes.md")
    with open(notes_md, "w", encoding="utf-8") as f:
        f.write("# Final Version Summary\n\n")
        f.write(f"- out_tag: `{args.out_tag}`\n")
        f.write(f"- base_patch_final: `{base_path}`\n")
        f.write(f"- kept_targets: `{','.join(targets)}`\n")
        f.write(f"- missing_targets: `{','.join(missing_targets) if missing_targets else 'None'}`\n")
        f.write(f"- n_clusters: `{meta['n_clusters']}`\n")
        f.write(f"- n_patches: `{meta['n_patches']}`\n")
        f.write(f"- top1_cluster: `{top1}`\n")
        f.write(f"- top1_ratio: `{top1_ratio:.6f}`\n")
        f.write("\n## Outputs\n")
        f.write(f"- patch_final: `{out_patch}`\n")
        f.write(f"- final_cluster_sizes: `{out_sizes}`\n")
        f.write(f"- slide_hist: `{out_hist}`\n")
        f.write(f"- meta: `{meta_json}`\n")
        f.write("\n## Apply Stats\n")
        for x in applied_stats:
            f.write(
                f"- {x['target_l2']}: status={x['status']}, "
                f"n_proto={x.get('n_proto_in_map', 0)}, n_changed={x.get('n_patches_changed', 0)}\n"
            )

    print("[final] version built")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[saved] {meta_json}")
    print(f"[saved] {notes_md}")


if __name__ == "__main__":
    main()

