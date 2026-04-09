import os
import gc
import json
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_argmin_min


# ================= 配置 =================
PATCH_INDEX_QC = "prototype_pipeline_output/patch_index_qc.parquet"
OUTPUT_DIR = "prototype_pipeline_output"

TARGET_STAIN = "HE"
N_PROTOTYPES = 2048

PROTOTYPE_CENTERS = os.path.join(
    OUTPUT_DIR, f"prototype_centers_{TARGET_STAIN}_{N_PROTOTYPES}.npy"
)

MIN_PATCHES_PER_SLIDE = 1
ASSIGN_ONLY_QC_PASS = True

# 每次从单张 slide 读取多少 patch 进行 assignment
CHUNK_SIZE = 50000


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


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x = x / norm
    return x.astype(np.float32, copy=False)


def load_h5_rows(feature_file: str, row_indices: np.ndarray) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    order = np.argsort(row_indices)
    rev = np.argsort(order)
    sorted_idx = row_indices[order]

    with h5py.File(feature_file, "r") as f:
        feat = f["features"]
        x_sorted = feat[sorted_idx]

    x = x_sorted[rev]
    return x.astype(np.float32, copy=False)


def iter_feature_chunks(df_slide: pd.DataFrame, chunk_size: int = 50000):
    row_idx = df_slide["feature_row"].to_numpy(dtype=np.int64)
    feature_file = df_slide["feature_file"].iloc[0]

    n = len(df_slide)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sub_df = df_slide.iloc[start:end].copy()
        rows = row_idx[start:end]
        x = load_h5_rows(feature_file, rows)
        x = l2_normalize(x)
        yield sub_df, x


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patch_index = load_table_auto(PATCH_INDEX_QC)

    df = patch_index.copy()
    if TARGET_STAIN is not None:
        df = df[df["stain"] == TARGET_STAIN]

    if ASSIGN_ONLY_QC_PASS:
        df = df[df["qc_pass"] == True]

    counts = df["slide_id"].value_counts()
    keep_slides = counts[counts >= MIN_PATCHES_PER_SLIDE].index
    df = df[df["slide_id"].isin(keep_slides)].copy()

    if len(df) == 0:
        raise RuntimeError("No patches available for prototype assignment.")

    centers = np.load(PROTOTYPE_CENTERS).astype(np.float32, copy=False)
    centers = l2_normalize(centers)

    if centers.shape[0] != N_PROTOTYPES:
        raise ValueError(f"Prototype count mismatch: expected {N_PROTOTYPES}, got {centers.shape[0]}")

    slide_ids = df["slide_id"].unique()
    print(f"[Assign] slides={len(slide_ids)}, patches={len(df)}, prototypes={centers.shape[0]}")

    assign_rows = []
    proto_usage = np.zeros(N_PROTOTYPES, dtype=np.int64)

    for slide_id in tqdm(slide_ids, desc="Assigning prototypes"):
        df_slide = df[df["slide_id"] == slide_id].sort_values("feature_row")

        for sub_df, x in iter_feature_chunks(df_slide, chunk_size=CHUNK_SIZE):
            proto_id, proto_dist = pairwise_distances_argmin_min(
                x, centers, axis=1, metric="euclidean"
            )

            out = sub_df[[
                "slide_id", "patch_id", "x", "y", "feature_file", "feature_row",
                "stain", "feature_dim", "tissue_frac", "qc_pass"
            ]].copy()
            out["proto_id"] = proto_id.astype(np.int32)
            out["proto_dist"] = proto_dist.astype(np.float32)

            assign_rows.append(out)

            binc = np.bincount(proto_id, minlength=N_PROTOTYPES)
            proto_usage += binc.astype(np.int64)

        gc.collect()

    patch_assignments = pd.concat(assign_rows, axis=0, ignore_index=True)

    # 每张 slide 的 prototype histogram
    hist = (
        patch_assignments.groupby(["slide_id", "proto_id"])
        .size()
        .reset_index(name="count")
    )

    slide_total = (
        patch_assignments.groupby("slide_id")
        .size()
        .reset_index(name="slide_total")
    )

    hist = hist.merge(slide_total, on="slide_id", how="left")
    hist["fraction"] = hist["count"] / hist["slide_total"]

    # prototype usage
    usage_df = pd.DataFrame({
        "proto_id": np.arange(N_PROTOTYPES, dtype=np.int32),
        "n_patches": proto_usage
    }).sort_values("n_patches", ascending=False)

    patch_assign_path = os.path.join(
        OUTPUT_DIR, f"patch_assignments_{TARGET_STAIN}_{N_PROTOTYPES}.parquet"
    )
    hist_path = os.path.join(
        OUTPUT_DIR, f"slide_prototype_hist_{TARGET_STAIN}_{N_PROTOTYPES}.parquet"
    )
    usage_path = os.path.join(
        OUTPUT_DIR, f"prototype_usage_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
    )
    meta_path = os.path.join(
        OUTPUT_DIR, f"prototype_assignment_stats_{TARGET_STAIN}_{N_PROTOTYPES}.json"
    )

    saved_assign = save_table_auto(patch_assignments, patch_assign_path)
    saved_hist = save_table_auto(hist, hist_path)
    usage_df.to_csv(usage_path, index=False)

    meta = {
        "target_stain": TARGET_STAIN,
        "n_prototypes": N_PROTOTYPES,
        "n_slides": int(patch_assignments["slide_id"].nunique()),
        "n_patches_assigned": int(len(patch_assignments)),
        "assign_only_qc_pass": bool(ASSIGN_ONLY_QC_PASS),
        "chunk_size": CHUNK_SIZE,
        "patch_assignments_path": saved_assign,
        "slide_hist_path": saved_hist,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {usage_path}")
    print(f"[Saved] {meta_path}")
    print("[Done] prototype assignment finished.")


if __name__ == "__main__":
    main()
