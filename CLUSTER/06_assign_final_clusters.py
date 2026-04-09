import os
import json
import pandas as pd


# ================= 配置 =================
OUTPUT_DIR = "prototype_pipeline_output"

TARGET_STAIN = "HE"
N_PROTOTYPES = 2048

PATCH_ASSIGNMENTS = os.path.join(
    OUTPUT_DIR, f"patch_assignments_{TARGET_STAIN}_{N_PROTOTYPES}.parquet"
)
PROTO_CLUSTERS_L1L2 = os.path.join(
    OUTPUT_DIR, f"prototype_clusters_L1L2_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
)


def load_table_auto(path: str):
    if os.path.exists(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".pkl"):
            return pd.read_pickle(path)

    alt_pkl = path.replace(".parquet", ".pkl")
    if os.path.exists(alt_pkl):
        return pd.read_pickle(alt_pkl)

    raise FileNotFoundError(f"Cannot find input table: {path} or fallback pickle.")


def save_table_auto(df: pd.DataFrame, path_parquet: str):
    try:
        df.to_parquet(path_parquet, index=False)
        print(f"[Saved] {path_parquet}")
        return path_parquet
    except Exception as e:
        path_pkl = path_parquet.replace(".parquet", ".pkl")
        print(f"[Warning] Failed to save parquet: {e}")
        print(f"[Fallback] Saving pickle: {path_pkl}")
        df.to_pickle(path_pkl)
        print(f"[Saved] {path_pkl}")
        return path_pkl


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patch_df = load_table_auto(PATCH_ASSIGNMENTS)
    proto_df = pd.read_csv(PROTO_CLUSTERS_L1L2)

    print(f"[Load] patch rows = {len(patch_df)}")
    print(f"[Load] proto rows = {len(proto_df)}")

    # 统一类型
    patch_df["proto_id"] = patch_df["proto_id"].astype(int)
    proto_df["proto_id"] = proto_df["proto_id"].astype(int)

    # 只保留需要的列，并统一成 pandas string dtype
    keep_cols = ["proto_id", "proto_cluster_L1", "proto_cluster_L2"]
    proto_df = proto_df[keep_cols].copy()

    proto_df["proto_cluster_L1"] = proto_df["proto_cluster_L1"].astype("string")
    proto_df["proto_cluster_L2"] = proto_df["proto_cluster_L2"].astype("string")

    # merge
    patch_final = patch_df.merge(
        proto_df,
        on="proto_id",
        how="left",
        validate="many_to_one"
    )

    # merge 后检查
    n_missing_l1 = patch_final["proto_cluster_L1"].isna().sum()
    n_missing_l2 = patch_final["proto_cluster_L2"].isna().sum()
    print(f"[Check] Missing proto_cluster_L1 after merge: {n_missing_l1}")
    print(f"[Check] Missing proto_cluster_L2 after merge: {n_missing_l2}")

    # 更稳的 final_cluster 赋值逻辑
    # 先默认用 L2
    patch_final["final_cluster"] = patch_final["proto_cluster_L2"].astype("string")

    # 如果 L2 是缺失或字符串 'NA'，则回退到 L1
    mask_use_l1 = patch_final["final_cluster"].isna() | (patch_final["final_cluster"] == "NA")
    patch_final.loc[mask_use_l1, "final_cluster"] = patch_final.loc[mask_use_l1, "proto_cluster_L1"]

    # 再做一次安全检查：如果 final_cluster 仍缺失，记录出来
    missing_final = patch_final["final_cluster"].isna().sum()
    print(f"[Check] Missing final_cluster after fallback: {missing_final}")

    if missing_final > 0:
        bad_proto = (
            patch_final.loc[patch_final["final_cluster"].isna(), "proto_id"]
            .value_counts()
            .rename_axis("proto_id")
            .reset_index(name="n_patches")
            .sort_values("n_patches", ascending=False)
        )
        bad_csv = os.path.join(
            OUTPUT_DIR, f"debug_missing_final_cluster_protos_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
        )
        bad_proto.to_csv(bad_csv, index=False)
        print(f"[Debug] Saved missing-final-cluster proto list: {bad_csv}")

    # 强制成字符串，避免保存时混入 float nan
    patch_final["proto_cluster_L1"] = patch_final["proto_cluster_L1"].astype("string")
    patch_final["proto_cluster_L2"] = patch_final["proto_cluster_L2"].astype("string")
    patch_final["final_cluster"] = patch_final["final_cluster"].astype("string")

    # 统计
    final_sizes = (
        patch_final["final_cluster"]
        .value_counts(dropna=False)
        .rename_axis("final_cluster")
        .reset_index(name="n_patches")
        .sort_values("final_cluster", na_position="last")
    )

    l1_sizes = (
        patch_final["proto_cluster_L1"]
        .value_counts(dropna=False)
        .rename_axis("proto_cluster_L1")
        .reset_index(name="n_patches")
        .sort_values("proto_cluster_L1", na_position="last")
    )

    l2_non_na = patch_final.loc[
        patch_final["proto_cluster_L2"].notna() & (patch_final["proto_cluster_L2"] != "NA"),
        "proto_cluster_L2"
    ]
    l2_sizes = (
        l2_non_na
        .value_counts(dropna=False)
        .rename_axis("proto_cluster_L2")
        .reset_index(name="n_patches")
        .sort_values("proto_cluster_L2", na_position="last")
    )

    # 每张 slide 的 final cluster composition
    slide_final = (
        patch_final.groupby(["slide_id", "final_cluster"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    slide_total = (
        patch_final.groupby("slide_id")
        .size()
        .reset_index(name="slide_total")
    )
    slide_final = slide_final.merge(slide_total, on="slide_id", how="left")
    slide_final["fraction"] = slide_final["count"] / slide_final["slide_total"]

    # 输出文件
    out_patch = os.path.join(
        OUTPUT_DIR, f"patch_final_clusters_{TARGET_STAIN}_{N_PROTOTYPES}.parquet"
    )
    out_slide = os.path.join(
        OUTPUT_DIR, f"slide_final_cluster_hist_{TARGET_STAIN}_{N_PROTOTYPES}.parquet"
    )
    out_final_sizes = os.path.join(
        OUTPUT_DIR, f"final_cluster_sizes_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
    )
    out_l1_sizes = os.path.join(
        OUTPUT_DIR, f"patch_cluster_sizes_L1_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
    )
    out_l2_sizes = os.path.join(
        OUTPUT_DIR, f"patch_cluster_sizes_L2_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
    )
    out_meta = os.path.join(
        OUTPUT_DIR, f"final_cluster_meta_{TARGET_STAIN}_{N_PROTOTYPES}.json"
    )

    saved_patch = save_table_auto(patch_final, out_patch)
    saved_slide = save_table_auto(slide_final, out_slide)
    final_sizes.to_csv(out_final_sizes, index=False)
    l1_sizes.to_csv(out_l1_sizes, index=False)
    l2_sizes.to_csv(out_l2_sizes, index=False)

    # 再打印一次前几项，方便你快速核对
    print("\n[Top final clusters]")
    print(final_sizes.head(20).to_string(index=False))

    meta = {
        "target_stain": TARGET_STAIN,
        "n_prototypes": int(N_PROTOTYPES),
        "n_patches": int(len(patch_final)),
        "n_slides": int(patch_final["slide_id"].nunique()),
        "n_final_clusters_non_na": int(patch_final["final_cluster"].dropna().nunique()),
        "n_missing_final_cluster": int(patch_final["final_cluster"].isna().sum()),
        "patch_final_path": saved_patch,
        "slide_final_hist_path": saved_slide,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {out_final_sizes}")
    print(f"[Saved] {out_l1_sizes}")
    print(f"[Saved] {out_l2_sizes}")
    print(f"[Saved] {out_meta}")
    print("[Done] final patch-level cluster assignment finished.")


if __name__ == "__main__":
    main()
