import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def load_table_auto(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        if path.endswith(".pkl"):
            return pd.read_pickle(path)
        if path.endswith(".csv"):
            return pd.read_csv(path)
    raise FileNotFoundError(path)


def save_table_auto(df: pd.DataFrame, path_parquet: str) -> str:
    try:
        df.to_parquet(path_parquet, index=False)
        return path_parquet
    except Exception:
        pkl = path_parquet.replace(".parquet", ".pkl")
        df.to_pickle(pkl)
        return pkl


def cluster_with_scanpy(
    x: np.ndarray,
    n_pcs: int,
    n_neighbors: int,
    resolution: float,
    random_state: int,
) -> Optional[np.ndarray]:
    try:
        import anndata as ad
        import scanpy as sc
    except Exception:
        return None

    adata = ad.AnnData(X=x)
    n_pcs = min(n_pcs, max(2, adata.n_vars - 1), max(2, adata.n_obs - 1))
    n_neighbors = min(n_neighbors, max(2, adata.n_obs - 1))
    sc.tl.pca(adata, svd_solver="randomized", n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=resolution, key_added="cluster", random_state=random_state)
    return adata.obs["cluster"].astype(str).to_numpy()


def main():
    p = argparse.ArgumentParser("Refine a single L2 cluster to L3 (CLUSTER-style experiment)")
    p.add_argument("--output-dir", default="prototype_pipeline_output")
    p.add_argument("--target-stain", default="HE")
    p.add_argument("--n-prototypes", type=int, default=2048)
    p.add_argument("--target-l2", default="2_0")
    p.add_argument("--method", choices=["leiden", "kmeans"], default="leiden")
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=12)
    p.add_argument("--resolution", type=float, default=0.60)
    p.add_argument("--kmeans-k", type=int, default=3)
    p.add_argument("--min-prototypes-to-refine", type=int, default=30)
    p.add_argument("--min-l3-prototypes", type=int, default=12)
    p.add_argument("--random-state", type=int, default=0)
    args = p.parse_args()

    od = args.output_dir
    stain = args.target_stain
    k = args.n_prototypes

    centers_npy = os.path.join(od, f"prototype_centers_{stain}_{k}.npy")
    proto_l1l2_csv = os.path.join(od, f"prototype_clusters_L1L2_{stain}_{k}.csv")
    patch_final_path = os.path.join(od, f"patch_final_clusters_{stain}_{k}.parquet")

    centers = np.load(centers_npy).astype(np.float32, copy=False)
    proto_df = pd.read_csv(proto_l1l2_csv).copy()
    proto_df["proto_id"] = pd.to_numeric(proto_df["proto_id"], errors="coerce").fillna(-1).astype(int)
    proto_df["proto_cluster_L1"] = proto_df["proto_cluster_L1"].astype(str)
    proto_df["proto_cluster_L2"] = proto_df["proto_cluster_L2"].astype(str)

    # baseline final cluster (L2 preferred, fallback L1)
    proto_df["final_cluster_baseline"] = proto_df["proto_cluster_L2"]
    m = proto_df["final_cluster_baseline"].isin(["NA", "nan", "None"])
    proto_df.loc[m, "final_cluster_baseline"] = proto_df.loc[m, "proto_cluster_L1"]

    target = str(args.target_l2)
    idx = np.where(proto_df["proto_cluster_L2"].to_numpy(dtype=str) == target)[0]
    n_target = len(idx)
    print(f"[L3] target_l2={target}, n_target_prototypes={n_target}")

    proto_df["proto_cluster_L3"] = "NA"
    proto_df["final_cluster_l3"] = proto_df["final_cluster_baseline"].astype(str)

    meta = {
        "target_l2": target,
        "n_target_prototypes": int(n_target),
        "method": args.method,
        "resolution": float(args.resolution),
        "n_pcs": int(args.n_pcs),
        "n_neighbors": int(args.n_neighbors),
        "min_prototypes_to_refine": int(args.min_prototypes_to_refine),
        "min_l3_prototypes": int(args.min_l3_prototypes),
        "status": "not_refined",
    }

    if n_target < args.min_prototypes_to_refine:
        print(f"[L3] skip: n_target < {args.min_prototypes_to_refine}")
    else:
        sub_proto_ids = proto_df.iloc[idx]["proto_id"].to_numpy(dtype=int)
        sub_centers = centers[sub_proto_ids]

        labels_local = None
        if args.method == "leiden":
            labels_local = cluster_with_scanpy(
                sub_centers,
                n_pcs=args.n_pcs,
                n_neighbors=args.n_neighbors,
                resolution=args.resolution,
                random_state=args.random_state,
            )
        if labels_local is None:
            kk = min(max(2, args.kmeans_k), max(2, n_target - 1))
            km = KMeans(n_clusters=kk, random_state=args.random_state, n_init="auto")
            labels_local = km.fit_predict(sub_centers).astype(str)

        labels_global = np.array([f"{target}_{x}" for x in labels_local], dtype=object)
        vc = pd.Series(labels_global).value_counts()
        small = set(vc[vc < args.min_l3_prototypes].index.tolist())
        for i, g in enumerate(labels_global):
            if g in small:
                labels_global[i] = "NA"

        # write back L3 / final
        write_df = pd.DataFrame(
            {
                "proto_id": sub_proto_ids.astype(int),
                "proto_cluster_L3": labels_global.astype(str),
            }
        )
        proto_df = proto_df.merge(write_df, on="proto_id", how="left", suffixes=("", "_new"))
        use_new = proto_df["proto_cluster_L3_new"].notna()
        proto_df.loc[use_new, "proto_cluster_L3"] = proto_df.loc[use_new, "proto_cluster_L3_new"].astype(str)
        proto_df = proto_df.drop(columns=["proto_cluster_L3_new"])

        use_l3 = ~proto_df["proto_cluster_L3"].isin(["NA", "nan", "None"])
        proto_df.loc[use_l3, "final_cluster_l3"] = proto_df.loc[use_l3, "proto_cluster_L3"]

        kept = sorted([x for x in set(labels_global.tolist()) if x != "NA"])
        meta["status"] = "refined"
        meta["n_l3_kept"] = int(len(kept))
        meta["l3_kept"] = kept
        meta["l3_counts"] = pd.Series(labels_global).value_counts().to_dict()
        print(f"[L3] kept_l3={len(kept)} -> {kept}")

    # save prototype map
    proto_out_csv = os.path.join(od, f"prototype_clusters_L1L2L3_target_{target}_{stain}_{k}.csv")
    proto_df.to_csv(proto_out_csv, index=False)
    print(f"[saved] {proto_out_csv}")

    meta_json = os.path.join(od, f"l3_refine_meta_target_{target}_{stain}_{k}.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[saved] {meta_json}")

    # patch-level output (for overlay / statistics)
    if os.path.exists(patch_final_path):
        patch_df = load_table_auto(patch_final_path).copy()
        patch_df["proto_id"] = pd.to_numeric(patch_df["proto_id"], errors="coerce").fillna(-1).astype(int)
        map_df = proto_df[["proto_id", "proto_cluster_L3", "final_cluster_l3"]].copy()
        patch_df = patch_df.merge(map_df, on="proto_id", how="left")

        patch_df["final_cluster_prev"] = patch_df["final_cluster"].astype(str)
        patch_df["final_cluster"] = patch_df["final_cluster_l3"].astype(str)

        patch_out = os.path.join(od, f"patch_final_clusters_{stain}_{k}_l3_target_{target}.parquet")
        saved_patch = save_table_auto(patch_df, patch_out)
        print(f"[saved] {saved_patch}")

        size_df = (
            patch_df["final_cluster"]
            .value_counts()
            .rename_axis("final_cluster")
            .reset_index(name="n_patches")
            .sort_values("n_patches", ascending=False)
        )
        size_out = os.path.join(od, f"final_cluster_sizes_{stain}_{k}_l3_target_{target}.csv")
        size_df.to_csv(size_out, index=False)
        print(f"[saved] {size_out}")

        slide_hist = (
            patch_df.groupby(["slide_id", "final_cluster"])
            .size()
            .reset_index(name="count")
        )
        slide_total = patch_df.groupby("slide_id").size().reset_index(name="slide_total")
        slide_hist = slide_hist.merge(slide_total, on="slide_id", how="left")
        slide_hist["fraction"] = slide_hist["count"] / slide_hist["slide_total"]
        hist_out = os.path.join(od, f"slide_final_cluster_hist_{stain}_{k}_l3_target_{target}.csv")
        slide_hist.to_csv(hist_out, index=False)
        print(f"[saved] {hist_out}")
    else:
        print(f"[warn] patch_final not found, skip patch-level outputs: {patch_final_path}")


if __name__ == "__main__":
    main()

