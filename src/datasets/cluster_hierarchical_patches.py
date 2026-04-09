import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / n).astype(np.float32, copy=False)


def _save_table_auto(df: pd.DataFrame, out_base: Path) -> str:
    pq = str(out_base.with_suffix(".parquet"))
    try:
        df.to_parquet(pq, index=False)
        return pq
    except Exception:
        csv = str(out_base.with_suffix(".csv"))
        df.to_csv(csv, index=False)
        return csv


def _fit_prototypes(
    manifest_rows: List[Dict],
    n_prototypes: int,
    feature_key: str,
    max_patches_per_sample: int,
    batch_size: int,
    random_state: int,
) -> np.ndarray:
    mbkm = MiniBatchKMeans(
        n_clusters=n_prototypes,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=200,
        reassignment_ratio=0.01,
        n_init="auto",
    )

    init_buf = []
    init_n = 0
    init_target = max(n_prototypes, 3 * n_prototypes)
    fitted = False

    for i, rec in enumerate(manifest_rows, 1):
        h5_path = rec["h5"]
        with h5py.File(h5_path, "r") as f:
            if feature_key not in f:
                raise KeyError(f"feature key '{feature_key}' not found in {h5_path}; keys={list(f.keys())}")
            x = f[feature_key][:]
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        if max_patches_per_sample > 0 and x.shape[0] > max_patches_per_sample:
            idx = np.random.permutation(x.shape[0])[:max_patches_per_sample]
            x = x[idx]
        x = _l2_normalize(x)

        if not fitted:
            init_buf.append(x)
            init_n += x.shape[0]
            if init_n >= init_target:
                x0 = np.concatenate(init_buf, axis=0)
                mbkm.partial_fit(x0)
                fitted = True
                init_buf = []
            continue

        mbkm.partial_fit(x)

    if not fitted:
        if init_n < n_prototypes:
            raise ValueError(f"Total samples {init_n} < n_prototypes {n_prototypes}; reduce n_prototypes.")
        x0 = np.concatenate(init_buf, axis=0)
        mbkm.partial_fit(x0)

    centers = _l2_normalize(mbkm.cluster_centers_.astype(np.float32, copy=False))
    return centers


def _assign_prototypes(
    manifest_rows: List[Dict],
    centers: np.ndarray,
    feature_key: str,
    max_patches_per_sample: int,
) -> pd.DataFrame:
    rows = []
    proto_count = centers.shape[0]

    for rec in manifest_rows:
        sid = rec.get("sample_id", "")
        split = rec.get("split", "")
        h5_path = rec["h5"]
        with h5py.File(h5_path, "r") as f:
            x = f[feature_key][:]
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        coords = None
        with h5py.File(h5_path, "r") as f:
            if "coords" in f:
                coords = f["coords"][:]

        if max_patches_per_sample > 0 and x.shape[0] > max_patches_per_sample:
            idx = np.random.permutation(x.shape[0])[:max_patches_per_sample]
            x = x[idx]
            if coords is not None:
                coords = coords[idx]
        x = _l2_normalize(x)

        # brute-force nearest center
        # dist^2 = ||x-c||^2, with normalized vectors this is fine
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        proto_id = d.argmin(axis=1).astype(np.int32)
        proto_dist = np.sqrt(d[np.arange(d.shape[0]), proto_id]).astype(np.float32)

        if coords is None:
            xs = np.zeros(x.shape[0], dtype=np.int32)
            ys = np.zeros(x.shape[0], dtype=np.int32)
        else:
            xs = coords[:, 0].astype(np.int32)
            ys = coords[:, 1].astype(np.int32)

        rows.append(
            pd.DataFrame(
                {
                    "sample_id": sid,
                    "split": split,
                    "patch_id": np.arange(x.shape[0], dtype=np.int32),
                    "x": xs,
                    "y": ys,
                    "proto_id": proto_id,
                    "proto_dist": proto_dist,
                }
            )
        )

    if not rows:
        raise RuntimeError("No patch assignments generated")
    return pd.concat(rows, axis=0, ignore_index=True)


def _cluster_with_scanpy(centers: np.ndarray, n_pcs: int, n_neighbors: int, resolution: float):
    try:
        import anndata as ad
        import scanpy as sc
    except Exception:
        return None

    adata = ad.AnnData(X=centers)
    n_pcs = min(n_pcs, max(2, adata.n_vars - 1), max(2, adata.n_obs - 1))
    n_neighbors = min(n_neighbors, max(2, adata.n_obs - 1))
    sc.tl.pca(adata, svd_solver="randomized", n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=resolution, key_added="cluster")
    return adata.obs["cluster"].astype(str).to_numpy()


def _cluster_level1(centers: np.ndarray, method: str, n_l1_clusters: int, n_pcs: int, n_neighbors: int, resolution: float, random_state: int):
    if method == "leiden":
        labels = _cluster_with_scanpy(centers, n_pcs=n_pcs, n_neighbors=n_neighbors, resolution=resolution)
        if labels is not None:
            return labels
    km = KMeans(n_clusters=n_l1_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(centers).astype(str)


def _refine_level2(
    centers: np.ndarray,
    l1_labels: np.ndarray,
    method: str,
    min_prototypes_to_refine: int,
    min_l2_prototypes: int,
    refine_resolution: float,
    n_pcs: int,
    n_neighbors: int,
    random_state: int,
) -> np.ndarray:
    out = np.array(["NA"] * len(l1_labels), dtype=object)

    for l1 in sorted(set(l1_labels.tolist())):
        idx = np.where(l1_labels == l1)[0]
        if len(idx) < min_prototypes_to_refine:
            continue
        sub = centers[idx]

        labels_local = None
        if method == "leiden":
            labels_local = _cluster_with_scanpy(sub, n_pcs=n_pcs, n_neighbors=n_neighbors, resolution=refine_resolution)

        if labels_local is None:
            # kmeans fallback; heuristic cluster count
            k = max(2, min(8, int(round(len(idx) / 24.0))))
            if k >= len(idx):
                continue
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels_local = km.fit_predict(sub).astype(str)

        # global labels l1_local
        gl = np.array([f"{l1}_{x}" for x in labels_local], dtype=object)

        # drop small L2 clusters -> NA fallback
        vc = pd.Series(gl).value_counts()
        small = set(vc[vc < min_l2_prototypes].index.tolist())
        for j, g in enumerate(gl):
            if g in small:
                gl[j] = "NA"

        out[idx] = gl

    return out


def main():
    p = argparse.ArgumentParser("Hierarchical clustering (L1->L2) for HE/fused patch features")
    p.add_argument("--manifest", required=True, help="jsonl with fields: sample_id, split, h5")
    p.add_argument("--output-dir", default="data/processed/hier_cluster")
    p.add_argument("--feature-key", default="features")
    p.add_argument("--n-prototypes", type=int, default=2048)
    p.add_argument("--max-patches-per-sample", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--l1-method", choices=["leiden", "kmeans"], default="leiden")
    p.add_argument("--n-l1-clusters", type=int, default=12)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--l1-resolution", type=float, default=0.45)
    p.add_argument("--refine-resolution", type=float, default=0.60)
    p.add_argument("--min-prototypes-to-refine", type=int, default=30)
    p.add_argument("--min-l2-prototypes", type=int, default=15)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_jsonl(Path(args.manifest))
    if not manifest_rows:
        raise RuntimeError("Empty manifest")

    centers = _fit_prototypes(
        manifest_rows=manifest_rows,
        n_prototypes=args.n_prototypes,
        feature_key=args.feature_key,
        max_patches_per_sample=args.max_patches_per_sample,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    np.save(str(out_dir / "prototype_centers.npy"), centers)

    patch_df = _assign_prototypes(
        manifest_rows=manifest_rows,
        centers=centers,
        feature_key=args.feature_key,
        max_patches_per_sample=args.max_patches_per_sample,
    )

    l1_labels = _cluster_level1(
        centers=centers,
        method=args.l1_method,
        n_l1_clusters=args.n_l1_clusters,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        resolution=args.l1_resolution,
        random_state=args.random_state,
    )

    l2_labels = _refine_level2(
        centers=centers,
        l1_labels=l1_labels,
        method=args.l1_method,
        min_prototypes_to_refine=args.min_prototypes_to_refine,
        min_l2_prototypes=args.min_l2_prototypes,
        refine_resolution=args.refine_resolution,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )

    proto_map = pd.DataFrame(
        {
            "proto_id": np.arange(centers.shape[0], dtype=np.int32),
            "proto_cluster_L1": l1_labels.astype(str),
            "proto_cluster_L2": l2_labels.astype(str),
        }
    )

    proto_map["final_cluster"] = proto_map["proto_cluster_L2"].copy()
    m = proto_map["final_cluster"].isna() | (proto_map["final_cluster"] == "NA")
    proto_map.loc[m, "final_cluster"] = proto_map.loc[m, "proto_cluster_L1"]

    patch_final = patch_df.merge(proto_map, on="proto_id", how="left", validate="many_to_one")

    slide_hist = (
        patch_final.groupby(["sample_id", "final_cluster"]).size().reset_index(name="count")
    )
    slide_total = patch_final.groupby("sample_id").size().reset_index(name="sample_total")
    slide_hist = slide_hist.merge(slide_total, on="sample_id", how="left")
    slide_hist["fraction"] = slide_hist["count"] / slide_hist["sample_total"]

    p1 = _save_table_auto(patch_df, out_dir / "patch_assignments")
    p2 = _save_table_auto(proto_map, out_dir / "prototype_clusters_L1L2")
    p3 = _save_table_auto(patch_final, out_dir / "patch_final_clusters")
    p4 = _save_table_auto(slide_hist, out_dir / "slide_final_cluster_hist")

    meta = {
        "manifest": args.manifest,
        "n_samples": len(manifest_rows),
        "n_prototypes": int(args.n_prototypes),
        "l1_method": args.l1_method,
        "n_l1_clusters": int(len(set(l1_labels.tolist()))),
        "n_l2_clusters_non_na": int(len(set([x for x in l2_labels.tolist() if x != "NA"]))),
        "patch_assignments": p1,
        "prototype_clusters_L1L2": p2,
        "patch_final_clusters": p3,
        "slide_final_cluster_hist": p4,
    }
    with (out_dir / "cluster_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[done] hierarchical clustering finished")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
