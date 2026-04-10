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
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_manifest_rows(manifest_rows: List[Dict], stain_name: str = "HE") -> List[Dict]:
    """
    Accept both formats:
    1) {"sample_id","split","h5": "..."}
    2) {"sample_id","split","h5s": {"HE":"...", ...}}
    Return normalized rows with key "h5".
    """
    out = []
    missing = []
    for rec in manifest_rows:
        rr = dict(rec)
        h5 = rr.get("h5", None)
        if (not h5) and isinstance(rr.get("h5s", None), dict):
            h5 = rr["h5s"].get(stain_name, None)
        if not h5:
            missing.append(str(rr.get("sample_id", "")))
            continue
        rr["h5"] = h5
        out.append(rr)

    if not out:
        raise RuntimeError(
            f"No valid rows after manifest normalization for stain='{stain_name}'. "
            "Expected each record to have `h5` or `h5s[stain]`."
        )
    if missing:
        print(f"[warn] {len(missing)} samples missing stain '{stain_name}', skipped.")
    return out


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
    prototype_patches_per_sample: int,
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
        cap = prototype_patches_per_sample if prototype_patches_per_sample > 0 else max_patches_per_sample
        if cap > 0 and x.shape[0] > cap:
            idx = np.random.permutation(x.shape[0])[:cap]
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


def _cluster_with_scanpy(centers: np.ndarray, n_pcs: int, n_neighbors: int, resolution: float, random_state: int):
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
    sc.tl.leiden(adata, resolution=resolution, key_added="cluster", random_state=random_state)
    return adata.obs["cluster"].astype(str).to_numpy()


def _cluster_level1(centers: np.ndarray, method: str, n_l1_clusters: int, n_pcs: int, n_neighbors: int, resolution: float, random_state: int):
    if method == "leiden":
        labels = _cluster_with_scanpy(centers, n_pcs=n_pcs, n_neighbors=n_neighbors, resolution=resolution, random_state=random_state)
        if labels is not None:
            return labels
    km = KMeans(n_clusters=n_l1_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(centers).astype(str)


def _safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    labs = np.asarray(labels)
    n_clusters = len(set(labs.tolist()))
    if x.shape[0] < 4 or n_clusters < 2:
        return -1.0
    try:
        return float(silhouette_score(x, labs, metric="euclidean"))
    except Exception:
        return -1.0


def _mean_pairwise_ari(label_runs: List[np.ndarray]) -> float:
    if len(label_runs) < 2:
        return -1.0
    vals = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            vals.append(float(adjusted_rand_score(label_runs[i], label_runs[j])))
    if not vals:
        return -1.0
    return float(np.mean(vals))


def _cluster_local_with_seed(
    sub: np.ndarray,
    method: str,
    n_pcs: int,
    n_neighbors: int,
    refine_resolution: float,
    random_state: int,
) -> Optional[np.ndarray]:
    labels_local = None
    if method == "leiden":
        labels_local = _cluster_with_scanpy(
            sub,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            resolution=refine_resolution,
            random_state=random_state,
        )
    if labels_local is None:
        k = max(2, min(8, int(round(sub.shape[0] / 24.0))))
        if k >= sub.shape[0]:
            return None
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels_local = km.fit_predict(sub).astype(str)
    return labels_local


def _build_proto_to_sample_set(patch_df: pd.DataFrame) -> Dict[int, set]:
    out: Dict[int, set] = {}
    if patch_df.empty:
        return out
    g = patch_df.groupby("proto_id")["sample_id"]
    for pid, vals in g:
        out[int(pid)] = set(vals.astype(str).tolist())
    return out


def _refine_next_level(
    centers: np.ndarray,
    parent_labels: np.ndarray,
    method: str,
    level_name: str,
    min_prototypes_to_refine: int,
    min_child_prototypes: int,
    min_child_slides: int,
    min_child_patch_frac: float,
    min_silhouette: float,
    refine_resolution: float,
    n_pcs: int,
    n_neighbors: int,
    random_state: int,
    stability_n_seeds: int,
    min_stability_ari: float,
    patch_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    out = np.array(["NA"] * len(parent_labels), dtype=object)
    logs: List[Dict] = []
    proto_to_samples = _build_proto_to_sample_set(patch_df) if patch_df is not None else {}
    total_patches = int(len(patch_df)) if patch_df is not None else 0

    for parent in sorted(set(parent_labels.tolist())):
        idx = np.where(parent_labels == parent)[0]
        if len(idx) < min_prototypes_to_refine:
            logs.append(
                {
                    "level": level_name,
                    "parent": str(parent),
                    "status": "skip_parent_small",
                    "n_parent_prototypes": int(len(idx)),
                }
            )
            continue
        sub = centers[idx]

        labels_local = _cluster_local_with_seed(
            sub=sub,
            method=method,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            refine_resolution=refine_resolution,
            random_state=random_state,
        )
        if labels_local is None:
            logs.append(
                {
                    "level": level_name,
                    "parent": str(parent),
                    "status": "skip_parent_no_valid_local_cluster",
                    "n_parent_prototypes": int(len(idx)),
                }
            )
            continue

        sil = _safe_silhouette(sub, labels_local)
        if sil < min_silhouette:
            logs.append(
                {
                    "level": level_name,
                    "parent": str(parent),
                    "status": "skip_parent_low_silhouette",
                    "n_parent_prototypes": int(len(idx)),
                    "silhouette": float(sil),
                }
            )
            continue

        stability_ari = -1.0
        if stability_n_seeds > 1:
            label_runs: List[np.ndarray] = []
            for ofs in range(stability_n_seeds):
                li = _cluster_local_with_seed(
                    sub=sub,
                    method=method,
                    n_pcs=n_pcs,
                    n_neighbors=n_neighbors,
                    refine_resolution=refine_resolution,
                    random_state=random_state + ofs,
                )
                if li is not None:
                    label_runs.append(li)
            stability_ari = _mean_pairwise_ari(label_runs)
            if stability_ari < min_stability_ari:
                logs.append(
                    {
                        "level": level_name,
                        "parent": str(parent),
                        "status": "skip_parent_low_stability",
                        "n_parent_prototypes": int(len(idx)),
                        "silhouette": float(sil),
                        "stability_ari": float(stability_ari),
                        "stability_n_seeds": int(stability_n_seeds),
                    }
                )
                continue

        # global labels parent_local
        gl = np.array([f"{parent}_{x}" for x in labels_local], dtype=object)

        # drop small child clusters -> NA fallback
        vc = pd.Series(gl).value_counts()
        small = set(vc[vc < min_child_prototypes].index.tolist())
        for j, g in enumerate(gl):
            if g in small:
                gl[j] = "NA"

        # drop child clusters with low slide/patch coverage -> NA fallback
        valid_children = [c for c in sorted(set(gl.tolist())) if c != "NA"]
        for c in valid_children:
            child_idx_local = np.where(gl == c)[0]
            child_proto_ids = idx[child_idx_local]

            n_slides = 0
            patch_frac = 0.0
            if patch_df is not None and total_patches > 0:
                sample_set = set()
                for pid in child_proto_ids.tolist():
                    sample_set |= proto_to_samples.get(int(pid), set())
                n_slides = len(sample_set)

                sub_patch = patch_df[patch_df["proto_id"].isin(child_proto_ids.tolist())]
                patch_frac = float(len(sub_patch)) / float(total_patches)

            if n_slides < min_child_slides or patch_frac < min_child_patch_frac:
                gl[np.where(gl == c)[0]] = "NA"

        out[idx] = gl
        logs.append(
            {
                "level": level_name,
                "parent": str(parent),
                "status": "refined",
                "n_parent_prototypes": int(len(idx)),
                "n_child_clusters_before_filter": int(len(set(labels_local.tolist()))),
                "n_child_clusters_after_filter": int(len(set([x for x in gl.tolist() if x != "NA"]))),
                "silhouette": float(sil),
                "stability_ari": float(stability_ari),
            }
        )

    return out, logs


def _spatial_smooth_majority(
    patch_final: pd.DataFrame,
    label_col: str = "final_cluster",
    n_neighbors: int = 7,
    n_iter: int = 1,
) -> pd.DataFrame:
    if n_neighbors <= 1 or n_iter <= 0:
        return patch_final
    if patch_final.empty or "sample_id" not in patch_final.columns or "x" not in patch_final.columns or "y" not in patch_final.columns:
        return patch_final

    out = patch_final.copy()
    work_col = label_col
    smooth_col = f"{label_col}_smoothed"
    out[smooth_col] = out[work_col].astype(str)

    for _ in range(n_iter):
        chunks = []
        for sid, d in out.groupby("sample_id", sort=False):
            d = d.copy()
            if len(d) < 3:
                chunks.append(d)
                continue
            xy = d[["x", "y"]].to_numpy(dtype=np.float32)
            k = min(n_neighbors, len(d))
            if k <= 1:
                chunks.append(d)
                continue

            nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
            nn.fit(xy)
            idxs = nn.kneighbors(xy, return_distance=False)
            labs = d[smooth_col].astype(str).to_numpy()
            new_labs = labs.copy()

            for i in range(len(d)):
                neigh = labs[idxs[i]]
                vc = pd.Series(neigh).value_counts()
                if not vc.empty:
                    new_labs[i] = str(vc.index[0])
            d[smooth_col] = new_labs
            chunks.append(d)
        out = pd.concat(chunks, axis=0, ignore_index=True)

    out[work_col] = out[smooth_col]
    return out


def main():
    p = argparse.ArgumentParser("Hierarchical clustering (L1->L2) for HE/fused patch features")
    p.add_argument("--manifest", required=True, help="jsonl with fields: sample_id, split, h5")
    p.add_argument("--stain-name", default="HE", help="Used when manifest has `h5s` dict instead of `h5`.")
    p.add_argument("--output-dir", default="data/processed/hier_cluster")
    p.add_argument("--feature-key", default="features")
    p.add_argument("--n-prototypes", type=int, default=2048)
    p.add_argument("--max-patches-per-sample", type=int, default=0)
    p.add_argument("--prototype-patches-per-sample", type=int, default=0, help="Cap per-slide patches only during prototype fitting for balanced sampling.")
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
    p.add_argument("--min-l2-slides", type=int, default=3)
    p.add_argument("--min-l2-patch-frac", type=float, default=0.005)
    p.add_argument("--min-l2-silhouette", type=float, default=0.05)
    p.add_argument("--stability-n-seeds", type=int, default=1)
    p.add_argument("--min-stability-ari", type=float, default=0.0)
    p.add_argument("--enable-l3", action="store_true")
    p.add_argument("--l3-resolution", type=float, default=0.75)
    p.add_argument("--min-l2-to-refine-l3", type=int, default=35)
    p.add_argument("--min-l3-prototypes", type=int, default=10)
    p.add_argument("--min-l3-slides", type=int, default=3)
    p.add_argument("--min-l3-patch-frac", type=float, default=0.003)
    p.add_argument("--min-l3-silhouette", type=float, default=0.06)
    p.add_argument("--spatial-smooth-k", type=int, default=0, help="If >1, apply per-slide KNN majority smoothing on final labels.")
    p.add_argument("--spatial-smooth-iter", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_jsonl(Path(args.manifest))
    manifest_rows = _normalize_manifest_rows(manifest_rows, stain_name=args.stain_name)
    if not manifest_rows:
        raise RuntimeError("Empty manifest")

    centers = _fit_prototypes(
        manifest_rows=manifest_rows,
        n_prototypes=args.n_prototypes,
        feature_key=args.feature_key,
        max_patches_per_sample=args.max_patches_per_sample,
        prototype_patches_per_sample=args.prototype_patches_per_sample,
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

    l2_labels, l2_logs = _refine_next_level(
        centers=centers,
        parent_labels=l1_labels,
        method=args.l1_method,
        level_name="L2",
        min_prototypes_to_refine=args.min_prototypes_to_refine,
        min_child_prototypes=args.min_l2_prototypes,
        min_child_slides=args.min_l2_slides,
        min_child_patch_frac=args.min_l2_patch_frac,
        min_silhouette=args.min_l2_silhouette,
        refine_resolution=args.refine_resolution,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
        stability_n_seeds=args.stability_n_seeds,
        min_stability_ari=args.min_stability_ari,
        patch_df=patch_df,
    )

    l3_labels = np.array(["NA"] * centers.shape[0], dtype=object)
    l3_logs: List[Dict] = []
    if args.enable_l3:
        l3_labels, l3_logs = _refine_next_level(
            centers=centers,
            parent_labels=l2_labels.astype(str),
            method=args.l1_method,
            level_name="L3",
            min_prototypes_to_refine=args.min_l2_to_refine_l3,
            min_child_prototypes=args.min_l3_prototypes,
            min_child_slides=args.min_l3_slides,
            min_child_patch_frac=args.min_l3_patch_frac,
            min_silhouette=args.min_l3_silhouette,
            refine_resolution=args.l3_resolution,
            n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state,
            stability_n_seeds=args.stability_n_seeds,
            min_stability_ari=args.min_stability_ari,
            patch_df=patch_df,
        )

    proto_map = pd.DataFrame(
        {
            "proto_id": np.arange(centers.shape[0], dtype=np.int32),
            "proto_cluster_L1": l1_labels.astype(str),
            "proto_cluster_L2": l2_labels.astype(str),
            "proto_cluster_L3": l3_labels.astype(str),
        }
    )

    if args.enable_l3:
        proto_map["final_cluster"] = proto_map["proto_cluster_L3"].copy()
        m = proto_map["final_cluster"].isna() | (proto_map["final_cluster"] == "NA")
        proto_map.loc[m, "final_cluster"] = proto_map.loc[m, "proto_cluster_L2"]
        m2 = proto_map["final_cluster"].isna() | (proto_map["final_cluster"] == "NA")
        proto_map.loc[m2, "final_cluster"] = proto_map.loc[m2, "proto_cluster_L1"]
    else:
        proto_map["final_cluster"] = proto_map["proto_cluster_L2"].copy()
        m = proto_map["final_cluster"].isna() | (proto_map["final_cluster"] == "NA")
        proto_map.loc[m, "final_cluster"] = proto_map.loc[m, "proto_cluster_L1"]

    patch_final = patch_df.merge(proto_map, on="proto_id", how="left", validate="many_to_one")
    if args.spatial_smooth_k > 1:
        patch_final = _spatial_smooth_majority(
            patch_final=patch_final,
            label_col="final_cluster",
            n_neighbors=args.spatial_smooth_k,
            n_iter=max(1, int(args.spatial_smooth_iter)),
        )

    slide_hist = (
        patch_final.groupby(["sample_id", "final_cluster"]).size().reset_index(name="count")
    )
    slide_total = patch_final.groupby("sample_id").size().reset_index(name="sample_total")
    slide_hist = slide_hist.merge(slide_total, on="sample_id", how="left")
    slide_hist["fraction"] = slide_hist["count"] / slide_hist["sample_total"]

    p1 = _save_table_auto(patch_df, out_dir / "patch_assignments")
    p2 = _save_table_auto(proto_map, out_dir / "prototype_clusters_L1L2L3")
    p3 = _save_table_auto(patch_final, out_dir / "patch_final_clusters")
    p4 = _save_table_auto(slide_hist, out_dir / "slide_final_cluster_hist")

    meta = {
        "manifest": args.manifest,
        "n_samples": len(manifest_rows),
        "n_prototypes": int(args.n_prototypes),
        "l1_method": args.l1_method,
        "n_l1_clusters": int(len(set(l1_labels.tolist()))),
        "n_l2_clusters_non_na": int(len(set([x for x in l2_labels.tolist() if x != "NA"]))),
        "enable_l3": bool(args.enable_l3),
        "n_l3_clusters_non_na": int(len(set([x for x in l3_labels.tolist() if x != "NA"]))),
        "stability_n_seeds": int(args.stability_n_seeds),
        "min_stability_ari": float(args.min_stability_ari),
        "prototype_patches_per_sample": int(args.prototype_patches_per_sample),
        "spatial_smooth_k": int(args.spatial_smooth_k),
        "spatial_smooth_iter": int(args.spatial_smooth_iter),
        "patch_assignments": p1,
        "prototype_clusters_L1L2L3": p2,
        "prototype_clusters_L1L2": p2,
        "patch_final_clusters": p3,
        "slide_final_cluster_hist": p4,
    }
    with (out_dir / "cluster_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with (out_dir / "refine_decisions.json").open("w", encoding="utf-8") as f:
        json.dump({"L2": l2_logs, "L3": l3_logs}, f, ensure_ascii=False, indent=2)

    print("[done] hierarchical clustering finished")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
