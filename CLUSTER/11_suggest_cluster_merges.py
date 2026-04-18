import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _parse_parent_l1(cluster_name: str) -> str:
    s = str(cluster_name)
    return s.split("_")[0] if "_" in s else s


def _safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    sim = float(np.dot(a, b) / (na * nb))
    return float(1.0 - sim)


def _cluster_centroids(
    proto_map: pd.DataFrame,
    centers: np.ndarray,
    proto_usage: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    usage = proto_usage.set_index("proto_id")["n_patches"].to_dict() if len(proto_usage) else {}

    out = {}
    for c, d in proto_map.groupby("final_cluster"):
        pids = d["proto_id"].to_numpy(dtype=int)
        w = np.array([float(usage.get(int(pid), 1.0)) for pid in pids], dtype=np.float64)
        if np.sum(w) <= 0:
            w = np.ones_like(w)
        x = centers[pids].astype(np.float64, copy=False)
        cen = np.average(x, axis=0, weights=w)
        out[str(c)] = cen.astype(np.float32)
    return out


def main():
    p = argparse.ArgumentParser("Suggest merge targets for small final clusters")
    p.add_argument("--output-dir", default="prototype_pipeline_output")
    p.add_argument("--target-stain", default="HE")
    p.add_argument("--n-prototypes", type=int, default=2048)
    p.add_argument("--small-max-ratio", type=float, default=0.005)
    p.add_argument("--small-max-patches", type=int, default=5000)
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    od = args.output_dir
    s = args.target_stain
    k = args.n_prototypes

    final_sizes_csv = os.path.join(od, f"final_cluster_sizes_{s}_{k}.csv")
    proto_l1l2_csv = os.path.join(od, f"prototype_clusters_L1L2_{s}_{k}.csv")
    centers_npy = os.path.join(od, f"prototype_centers_{s}_{k}.npy")
    usage_csv = os.path.join(od, f"prototype_usage_{s}_{k}.csv")

    final_df = _load_csv(final_sizes_csv)
    proto_df = _load_csv(proto_l1l2_csv)
    centers = np.load(centers_npy).astype(np.float32, copy=False)
    usage_df = _load_csv(usage_csv) if os.path.exists(usage_csv) else pd.DataFrame(columns=["proto_id", "n_patches"])

    final_df["final_cluster"] = final_df["final_cluster"].astype(str)
    final_df["n_patches"] = pd.to_numeric(final_df["n_patches"], errors="coerce").fillna(0).astype(int)
    total = int(final_df["n_patches"].sum())

    proto_df["proto_id"] = pd.to_numeric(proto_df["proto_id"], errors="coerce").fillna(-1).astype(int)
    proto_df["proto_cluster_L1"] = proto_df["proto_cluster_L1"].astype(str)
    proto_df["proto_cluster_L2"] = proto_df["proto_cluster_L2"].astype(str)
    proto_df["final_cluster"] = np.where(
        proto_df["proto_cluster_L2"].isin(["NA", "nan", "None"]),
        proto_df["proto_cluster_L1"],
        proto_df["proto_cluster_L2"],
    )

    cents = _cluster_centroids(proto_df, centers, usage_df)

    threshold = max(int(args.small_max_patches), int(total * float(args.small_max_ratio)))
    final_df["ratio"] = final_df["n_patches"] / max(1, total)
    final_df["is_small"] = final_df["n_patches"] <= threshold

    small_clusters = final_df[final_df["is_small"]]["final_cluster"].astype(str).tolist()
    large_clusters = final_df[~final_df["is_small"]]["final_cluster"].astype(str).tolist()

    rows: List[Dict] = []
    for sc in small_clusters:
        if sc not in cents:
            continue
        svec = cents[sc]
        sparent = _parse_parent_l1(sc)

        d_all: List[Tuple[str, float]] = []
        d_same: List[Tuple[str, float]] = []

        for tc in large_clusters:
            if tc == sc or tc not in cents:
                continue
            dist = _safe_cosine_distance(svec, cents[tc])
            d_all.append((tc, dist))
            if _parse_parent_l1(tc) == sparent:
                d_same.append((tc, dist))

        d_all = sorted(d_all, key=lambda x: x[1])
        d_same = sorted(d_same, key=lambda x: x[1])

        cand_all = [x[0] for x in d_all[: args.topk]]
        cand_same = [x[0] for x in d_same[: args.topk]]

        best_target = cand_same[0] if len(cand_same) else (cand_all[0] if len(cand_all) else "")
        best_dist = None
        for t, dd in d_all:
            if t == best_target:
                best_dist = float(dd)
                break

        n_small = int(final_df.loc[final_df["final_cluster"] == sc, "n_patches"].iloc[0])
        n_target = int(final_df.loc[final_df["final_cluster"] == best_target, "n_patches"].iloc[0]) if best_target else 0

        rows.append(
            {
                "small_cluster": sc,
                "small_parent_l1": sparent,
                "small_n_patches": n_small,
                "small_ratio": float(n_small / max(1, total)),
                "suggested_target": best_target,
                "target_n_patches": n_target,
                "cosine_dist": best_dist,
                "topk_any_parent": "|".join(cand_all),
                "topk_same_parent": "|".join(cand_same),
                "recommendation": "merge_to_same_parent" if len(cand_same) else "merge_to_nearest",
+                "enabled": 1,
            }
        )

    sugg_df = pd.DataFrame(rows)
    out_csv = os.path.join(od, f"cluster_merge_suggestions_{s}_{k}.csv")
    out_json = os.path.join(od, f"cluster_merge_suggestions_{s}_{k}.json")

    if len(sugg_df):
        sugg_df.to_csv(out_csv, index=False)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(sugg_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    summary = {
        "n_clusters": int(final_df["final_cluster"].nunique()),
        "n_patches_total": int(total),
        "small_threshold_patches": int(threshold),
        "n_small_clusters": int(len(small_clusters)),
        "small_clusters": small_clusters,
        "suggestion_csv": out_csv,
        "suggestion_json": out_json,
    }
    print("[suggest] merge summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(sugg_df):
        print("\n[suggest] preview:")
        print(sugg_df.head(20).to_string(index=False))
    else:
        print("[suggest] no small clusters found; nothing to merge")


if __name__ == "__main__":
    main()
