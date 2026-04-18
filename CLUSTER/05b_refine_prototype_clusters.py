import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt


# ================= 配置 =================
OUTPUT_DIR = "prototype_pipeline_output"

TARGET_STAIN = "HE"
N_PROTOTYPES = 2048

PROTOTYPE_CENTERS = os.path.join(
    OUTPUT_DIR, f"prototype_centers_{TARGET_STAIN}_{N_PROTOTYPES}.npy"
)
PROTOTYPE_CLUSTERS_L1 = os.path.join(
    OUTPUT_DIR, f"prototype_clusters_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
)

# --------------------------------------------------
# 每个 L1 cluster 单独控制是否细分，以及细分参数
# enabled=False 表示跳过细分，直接保留 L1
# --------------------------------------------------
REFINE_CONFIG = {
    # Dominant patch-mass cluster (~86.9%): prioritize refinement.
    "2": {
        "enabled": True,
        "n_pcs": 40,
        "n_neighbors": 15,
        "resolution": 0.75,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
        "min_prototypes_to_refine": 30,
        "min_l2_prototypes": 20,
    },
    # Secondary cluster (~9.0%): moderate refinement.
    "4": {
        "enabled": True,
        "n_pcs": 30,
        "n_neighbors": 12,
        "resolution": 0.55,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
        "min_prototypes_to_refine": 30,
        "min_l2_prototypes": 15,
    },
    # Smaller but non-trivial clusters: light refinement.
    "5": {
        "enabled": True,
        "n_pcs": 25,
        "n_neighbors": 10,
        "resolution": 0.45,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
        "min_prototypes_to_refine": 30,
        "min_l2_prototypes": 12,
    },
    "6": {
        "enabled": True,
        "n_pcs": 25,
        "n_neighbors": 10,
        "resolution": 0.45,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
        "min_prototypes_to_refine": 30,
        "min_l2_prototypes": 12,
    },
    # Tiny/low-mass clusters: keep as L1 in first pass.
    "0": {
        "enabled": False,
        "n_pcs": 30,
        "n_neighbors": 12,
        "resolution": 0.45,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
        "min_prototypes_to_refine": 30,
        "min_l2_prototypes": 15,
    },
    "1": {
        "enabled": False,
        "n_pcs": 30,
        "n_neighbors": 12,
        "resolution": 0.45,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
        "min_prototypes_to_refine": 30,
        "min_l2_prototypes": 15,
    },
    "3": {
        "enabled": False,
        "n_pcs": 30,
        "n_neighbors": 12,
        "resolution": 0.45,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
        "min_prototypes_to_refine": 30,
        "min_l2_prototypes": 15,
    },
    "7": {
        "enabled": False,
        "n_pcs": 30,
        "n_neighbors": 12,
        "resolution": 0.45,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
        "min_prototypes_to_refine": 30,
        "min_l2_prototypes": 15,
    },
}


def run_l2_refine_for_cluster(centers, full_df, l1_cluster, cfg):
    sub_df = full_df[full_df["proto_cluster_L1"] == str(l1_cluster)].copy()
    n_sub = len(sub_df)

    print(f"\n[Refine] L1 cluster {l1_cluster}: {n_sub} prototypes")

    if not cfg["enabled"]:
        print("  -> Skip by config: keep L1 only.")
        return full_df, {
            "l1_cluster": str(l1_cluster),
            "enabled": False,
            "n_input_prototypes": int(n_sub),
            "n_l2_clusters": 0,
            "note": "skip_refine_keep_L1"
        }

    if n_sub < cfg["min_prototypes_to_refine"]:
        print(f"  -> Skip: n_prototypes < {cfg['min_prototypes_to_refine']}")
        return full_df, {
            "l1_cluster": str(l1_cluster),
            "enabled": False,
            "n_input_prototypes": int(n_sub),
            "n_l2_clusters": 0,
            "note": "too_small_to_refine"
        }

    proto_ids = sub_df["proto_id"].to_numpy(dtype=int)
    sub_centers = centers[proto_ids]

    adata_sub = ad.AnnData(X=sub_centers)
    adata_sub.obs_names = [f"proto_{pid}" for pid in proto_ids]
    adata_sub.obs["proto_id"] = proto_ids.astype(np.int32)
    adata_sub.obs["proto_cluster_L1"] = str(l1_cluster)

    if "n_patches" in sub_df.columns:
        adata_sub.obs["n_patches"] = sub_df["n_patches"].to_numpy()
        adata_sub.obs["log1p_n_patches"] = np.log1p(adata_sub.obs["n_patches"].to_numpy())
    else:
        adata_sub.obs["n_patches"] = 0
        adata_sub.obs["log1p_n_patches"] = 0.0

    n_pcs = min(cfg["n_pcs"], max(2, adata_sub.n_vars - 1), max(2, adata_sub.n_obs - 1))
    n_neighbors = min(cfg["n_neighbors"], max(2, adata_sub.n_obs - 1))

    sc.tl.pca(adata_sub, svd_solver="randomized", n_comps=n_pcs)
    sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.leiden(adata_sub, resolution=cfg["resolution"], key_added="proto_cluster_L2_local")
    sc.tl.umap(
        adata_sub,
        min_dist=cfg["umap_min_dist"],
        spread=cfg["umap_spread"]
    )

    # 局部标签 -> 全局标签，例如 0_0, 0_1
    local_labels = adata_sub.obs["proto_cluster_L2_local"].astype(str).to_numpy()
    global_l2 = np.array([f"{l1_cluster}_{x}" for x in local_labels], dtype=object)
    adata_sub.obs["proto_cluster_L2"] = global_l2

    # --------------------------------------
    # 过滤过小 L2 子类：太小则回退到 L1
    # --------------------------------------
    l2_sizes = adata_sub.obs["proto_cluster_L2"].value_counts()
    small_l2 = set(l2_sizes[l2_sizes < cfg["min_l2_prototypes"]].index.tolist())

    if len(small_l2) > 0:
        print(f"  -> Small L2 clusters to fallback to L1: {sorted(small_l2)}")
        mask_small = adata_sub.obs["proto_cluster_L2"].isin(small_l2)
        adata_sub.obs.loc[mask_small, "proto_cluster_L2"] = "NA"

    # 写回主表
    map_df = adata_sub.obs[["proto_id", "proto_cluster_L2"]].copy()
    map_df["proto_id"] = map_df["proto_id"].astype(int)

    idx = full_df["proto_id"].isin(map_df["proto_id"])
    full_df.loc[idx, "proto_cluster_L2"] = (
        full_df.loc[idx, "proto_id"]
        .map(map_df.set_index("proto_id")["proto_cluster_L2"])
        .astype(str)
        .values
    )

    # 保存 size 表
    size_df = (
        adata_sub.obs["proto_cluster_L2"]
        .value_counts()
        .rename_axis("proto_cluster_L2")
        .reset_index(name="n_prototypes")
        .sort_values("proto_cluster_L2")
    )

    size_csv = os.path.join(
        OUTPUT_DIR, f"prototype_refined_sizes_L1_{l1_cluster}_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
    )
    size_df.to_csv(size_csv, index=False)

    # 保存 UMAP：按 L2 上色
    sc.pl.umap(
        adata_sub,
        color="proto_cluster_L2",
        size=120,
        legend_loc="on data",
        show=False,
        title=f"L2 refinement of L1 cluster {l1_cluster}"
    )
    umap_png = os.path.join(
        OUTPUT_DIR, f"prototype_refined_umap_L1_{l1_cluster}_{TARGET_STAIN}_{N_PROTOTYPES}.png"
    )
    plt.savefig(umap_png, dpi=300, bbox_inches="tight")
    plt.close()

    # 保存 usage 图
    sc.pl.umap(
        adata_sub,
        color="log1p_n_patches",
        size=120,
        show=False,
        title=f"L2 refinement usage of L1 cluster {l1_cluster}"
    )
    usage_png = os.path.join(
        OUTPUT_DIR, f"prototype_refined_umap_usage_L1_{l1_cluster}_{TARGET_STAIN}_{N_PROTOTYPES}.png"
    )
    plt.savefig(usage_png, dpi=300, bbox_inches="tight")
    plt.close()

    # 统计真正保留下来的 L2（排除 NA）
    kept_l2 = adata_sub.obs.loc[adata_sub.obs["proto_cluster_L2"] != "NA", "proto_cluster_L2"].nunique()

    print(f"  -> Kept L2 clusters: {kept_l2}")
    print(f"  -> Saved: {size_csv}")
    print(f"  -> Saved: {umap_png}")
    print(f"  -> Saved: {usage_png}")

    return full_df, {
        "l1_cluster": str(l1_cluster),
        "enabled": True,
        "n_input_prototypes": int(n_sub),
        "n_l2_clusters_kept": int(kept_l2),
        "n_pcs": int(n_pcs),
        "n_neighbors": int(n_neighbors),
        "resolution": float(cfg["resolution"]),
        "min_l2_prototypes": int(cfg["min_l2_prototypes"]),
        "size_csv": size_csv,
        "umap_png": umap_png,
        "usage_png": usage_png,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    centers = np.load(PROTOTYPE_CENTERS).astype(np.float32, copy=False)
    l1_df = pd.read_csv(PROTOTYPE_CLUSTERS_L1)

    l1_df["proto_cluster"] = l1_df["proto_cluster"].astype(str)
    l1_df["proto_id"] = l1_df["proto_id"].astype(int)

    full_df = l1_df.copy()
    full_df["proto_cluster_L1"] = full_df["proto_cluster"].astype(str)
    full_df["proto_cluster_L2"] = "NA"

    refine_summary = []

    # 按配置逐个处理
    for l1_cluster, cfg in REFINE_CONFIG.items():
        full_df, summary = run_l2_refine_for_cluster(centers, full_df, l1_cluster, cfg)
        refine_summary.append(summary)

    # 保存总表
    out_csv = os.path.join(
        OUTPUT_DIR, f"prototype_clusters_L1L2_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
    )
    full_df.to_csv(out_csv, index=False)

    meta = {
        "target_stain": TARGET_STAIN,
        "n_prototypes": int(N_PROTOTYPES),
        "refine_config": REFINE_CONFIG,
        "refine_summary": refine_summary,
    }

    meta_json = os.path.join(
        OUTPUT_DIR, f"prototype_refine_meta_{TARGET_STAIN}_{N_PROTOTYPES}.json"
    )
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {out_csv}")
    print(f"[Saved] {meta_json}")
    print("[Done] L2 prototype refinement finished.")


if __name__ == "__main__":
    main()
