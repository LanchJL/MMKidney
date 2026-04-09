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
PROTOTYPE_USAGE = os.path.join(
    OUTPUT_DIR, f"prototype_usage_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
)

# 真正影响最终 cluster 数的参数
N_PCS = 50
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.45

UMAP_MIN_DIST = 0.3
UMAP_SPREAD = 1.0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    centers = np.load(PROTOTYPE_CENTERS).astype(np.float32, copy=False)

    adata = ad.AnnData(X=centers)
    adata.obs_names = [f"proto_{i}" for i in range(centers.shape[0])]
    adata.obs["proto_id"] = np.arange(centers.shape[0], dtype=np.int32)

    if os.path.exists(PROTOTYPE_USAGE):
        usage_df = pd.read_csv(PROTOTYPE_USAGE)
        usage_df = usage_df.sort_values("proto_id")
        adata.obs["n_patches"] = usage_df["n_patches"].to_numpy(dtype=np.int64)
        adata.obs["log1p_n_patches"] = np.log1p(adata.obs["n_patches"].to_numpy())
    else:
        adata.obs["n_patches"] = 0
        adata.obs["log1p_n_patches"] = 0.0

    print(f"[Cluster prototypes] n_prototypes={adata.n_obs}, dim={adata.n_vars}")

    # centers 本身已是 2048 x 1536，不大，可以直接做 graph clustering
    sc.tl.pca(adata, svd_solver="randomized", n_comps=min(N_PCS, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="proto_cluster")
    sc.tl.umap(adata, min_dist=UMAP_MIN_DIST, spread=UMAP_SPREAD)

    n_clusters = adata.obs["proto_cluster"].nunique()
    print(f"[Done] Found {n_clusters} prototype clusters.")

    # 保存 cluster 表
    cluster_df = adata.obs[["proto_id", "proto_cluster", "n_patches", "log1p_n_patches"]].copy()
    cluster_df = cluster_df.sort_values(["proto_cluster", "proto_id"])

    cluster_csv = os.path.join(
        OUTPUT_DIR, f"prototype_clusters_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
    )
    graph_h5ad = os.path.join(
        OUTPUT_DIR, f"prototype_graph_{TARGET_STAIN}_{N_PROTOTYPES}.h5ad"
    )
    cluster_sizes_csv = os.path.join(
        OUTPUT_DIR, f"prototype_cluster_sizes_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
    )
    meta_json = os.path.join(
        OUTPUT_DIR, f"prototype_cluster_meta_{TARGET_STAIN}_{N_PROTOTYPES}.json"
    )

    cluster_df.to_csv(cluster_csv, index=False)
    adata.write(graph_h5ad)

    cluster_sizes = (
        adata.obs["proto_cluster"]
        .value_counts()
        .rename_axis("proto_cluster")
        .reset_index(name="n_prototypes")
        .sort_values("proto_cluster")
    )
    cluster_sizes.to_csv(cluster_sizes_csv, index=False)

    # UMAP 图
    sc.pl.umap(
        adata,
        color="proto_cluster",
        size=80,
        legend_loc="on data",
        show=False,
        title=f"Prototype clusters ({TARGET_STAIN}, K={N_PROTOTYPES})"
    )
    umap_png = os.path.join(
        OUTPUT_DIR, f"prototype_umap_{TARGET_STAIN}_{N_PROTOTYPES}.png"
    )
    plt.savefig(umap_png, dpi=300, bbox_inches="tight")
    plt.close()

    sc.pl.umap(
        adata,
        color="log1p_n_patches",
        size=80,
        show=False,
        title=f"Prototype usage ({TARGET_STAIN}, K={N_PROTOTYPES})"
    )
    umap_usage_png = os.path.join(
        OUTPUT_DIR, f"prototype_umap_usage_{TARGET_STAIN}_{N_PROTOTYPES}.png"
    )
    plt.savefig(umap_usage_png, dpi=300, bbox_inches="tight")
    plt.close()

    meta = {
        "target_stain": TARGET_STAIN,
        "n_prototypes": int(N_PROTOTYPES),
        "n_final_clusters": int(n_clusters),
        "n_pcs": int(N_PCS),
        "n_neighbors": int(N_NEIGHBORS),
        "leiden_resolution": float(LEIDEN_RESOLUTION),
        "umap_min_dist": float(UMAP_MIN_DIST),
        "umap_spread": float(UMAP_SPREAD),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {cluster_csv}")
    print(f"[Saved] {graph_h5ad}")
    print(f"[Saved] {cluster_sizes_csv}")
    print(f"[Saved] {umap_png}")
    print(f"[Saved] {umap_usage_png}")
    print(f"[Saved] {meta_json}")


if __name__ == "__main__":
    main()
