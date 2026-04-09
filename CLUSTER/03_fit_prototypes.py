import os
import gc
import json
import pickle
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


# ================= 配置 =================
PATCH_INDEX_QC = "prototype_pipeline_output/patch_index_qc.parquet"
OUTPUT_DIR = "prototype_output_0.6"

# 先只做 HE；以后可按 stain 分开训练
TARGET_STAIN = "HE"

# prototype 数量：第一版建议 2048
N_PROTOTYPES = 2048

# MiniBatchKMeans 参数
BATCH_SIZE = 10000
MAX_ITER = 200
RANDOM_STATE = 0
REASSIGNMENT_RATIO = 0.01
INIT_SIZE = None

# 每次从一个 slide 读多少 patch 进入 partial_fit
CHUNK_SIZE = 50000

# slide 最少通过 QC 的 patch 数
MIN_PATCHES_PER_SLIDE = 10

# 初始化时至少缓存多少样本再开始第一次 partial_fit
INIT_BUFFER_MULTIPLIER = 3   # 建议 >= 2，3 更稳


def load_table_auto(path: str):
    if os.path.exists(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".pkl"):
            return pd.read_pickle(path)

    alt_pkl = path.replace(".parquet", ".pkl")
    if os.path.exists(alt_pkl):
        return pd.read_pickle(alt_pkl)

    raise FileNotFoundError(f"Cannot find input table: {path} or fallback pickle.")


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
        rows = row_idx[start:end]
        x = load_h5_rows(feature_file, rows)
        x = l2_normalize(x)
        yield x


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patch_index = load_table_auto(PATCH_INDEX_QC)

    df = patch_index.copy()
    df = df[df["qc_pass"] == True]

    if TARGET_STAIN is not None:
        df = df[df["stain"] == TARGET_STAIN]

    counts = df["slide_id"].value_counts()
    keep_slides = counts[counts >= MIN_PATCHES_PER_SLIDE].index
    df = df[df["slide_id"].isin(keep_slides)].copy()

    if len(df) == 0:
        raise RuntimeError("No QC-passed patches available for prototype training.")

    feature_dims = sorted(df["feature_dim"].unique().tolist())
    if len(feature_dims) != 1:
        raise ValueError(f"Mixed feature dims detected: {feature_dims}")
    feature_dim = int(feature_dims[0])

    slide_ids = df["slide_id"].unique()
    print(f"[Train] slides={len(slide_ids)}, patches={len(df)}, dim={feature_dim}, stain={TARGET_STAIN}")

    model = MiniBatchKMeans(
        n_clusters=N_PROTOTYPES,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
        reassignment_ratio=REASSIGNMENT_RATIO,
        init_size=INIT_SIZE,
        n_init="auto",
        verbose=0,
    )

    n_seen = 0
    chunk_counter = 0
    fitted = False

    # ---- warm start buffer for first partial_fit ----
    init_target = max(N_PROTOTYPES, INIT_BUFFER_MULTIPLIER * N_PROTOTYPES)
    init_chunks = []
    init_n = 0

    for slide_id in tqdm(slide_ids, desc="Fitting prototypes"):
        df_slide = df[df["slide_id"] == slide_id].sort_values("feature_row")

        for x in iter_feature_chunks(df_slide, chunk_size=CHUNK_SIZE):
            if x.shape[0] == 0:
                continue

            # first partial_fit needs at least n_clusters samples
            if not fitted:
                init_chunks.append(x)
                init_n += x.shape[0]
                n_seen += int(x.shape[0])
                chunk_counter += 1

                if init_n >= init_target:
                    x0 = np.concatenate(init_chunks, axis=0)
                    print(f"[Init] first partial_fit with {x0.shape[0]} samples")
                    model.partial_fit(x0)
                    fitted = True

                    del x0
                    del init_chunks
                    init_chunks = []
                    gc.collect()
                continue

            model.partial_fit(x)
            n_seen += int(x.shape[0])
            chunk_counter += 1

        gc.collect()

    # 如果循环结束还没 fitted，说明总样本本身都不够
    if not fitted:
        if init_n < N_PROTOTYPES:
            raise ValueError(
                f"Total training samples ({init_n}) < n_clusters ({N_PROTOTYPES}). "
                f"Please reduce N_PROTOTYPES."
            )
        x0 = np.concatenate(init_chunks, axis=0)
        print(f"[Init-final] first partial_fit with {x0.shape[0]} samples")
        model.partial_fit(x0)
        fitted = True
        del x0
        gc.collect()

    centers = model.cluster_centers_.astype(np.float32, copy=False)
    centers = l2_normalize(centers)

    centers_npy = os.path.join(OUTPUT_DIR, f"prototype_centers_{TARGET_STAIN}_{N_PROTOTYPES}.npy")
    model_pkl = os.path.join(OUTPUT_DIR, f"prototype_model_{TARGET_STAIN}_{N_PROTOTYPES}.pkl")
    stats_json = os.path.join(OUTPUT_DIR, f"prototype_training_stats_{TARGET_STAIN}_{N_PROTOTYPES}.json")

    np.save(centers_npy, centers)

    with open(model_pkl, "wb") as f:
        pickle.dump(model, f)

    stats = {
        "target_stain": TARGET_STAIN,
        "n_prototypes": N_PROTOTYPES,
        "feature_dim": feature_dim,
        "n_training_patches": int(len(df)),
        "n_training_slides": int(len(slide_ids)),
        "n_seen_by_partial_fit": int(n_seen),
        "n_chunks": int(chunk_counter),
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "max_iter": MAX_ITER,
        "random_state": RANDOM_STATE,
        "reassignment_ratio": REASSIGNMENT_RATIO,
        "min_patches_per_slide": MIN_PATCHES_PER_SLIDE,
        "init_buffer_multiplier": INIT_BUFFER_MULTIPLIER,
        "init_target_samples": int(init_target),
    }

    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {centers_npy}")
    print(f"[Saved] {model_pkl}")
    print(f"[Saved] {stats_json}")
    print(f"[Done] trained {N_PROTOTYPES} prototypes for stain={TARGET_STAIN}")


if __name__ == "__main__":
    main()
