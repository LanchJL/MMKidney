import os
import re
import glob
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


# ================= 配置 =================
H5_DIR = "/media/a6000/3E5C99E35C9995ED/jcy/dataset/medical/WSIs_patch_HE_40/40x_512px_256px_overlap/features_uni_v2"
OUTPUT_DIR = "prototype_pipeline_output"

# 可选：按文件名自动推断 stain
DEFAULT_STAIN = "HE"

SKIP_SLIDES = {
    "17S01765-HE", "17S03033-HE", "21S05313-HE", "21S05313-HE_v2",
    "21S05313-HE_v3", "21S05313-HE_v4", "21S05313-HE_v5", "21S05313-HE_v6",
    "21S05313-HE_v7", "21S41417-HE", "21S41417-HE_v2", "21S41417-HE_v3",
    "21S41417-HE_v4", "21S41417-HE_v5", "21S41417-HE_v6", "21S41417-HE_v7",
    "21S41417-HE_v8", "21S41417-HE_v9", "21S41417-HE_v10", "24S051181-HE",
    "24S051181-HE_v2", "24S051181-HE_v3", "24S062454-HE", "24S062454-HE_v2",
    "24S062454-HE_v3", "24S062454-HE_v4", "24S088829-HE", "24S088829-HE_v2",
    "24S088829-HE_v3", "24S088829-HE_v4", "24S088829-HE", "24S094750-HE",
    "24S096211-HE", "24S099867-HE",
}


def infer_stain(slide_id: str, default_stain: str = "HE") -> str:
    s = slide_id.upper()
    for stain in ["HE", "H&E", "C4D", "PAS", "PASM", "MT", "TRI", "IF"]:
        if stain in s:
            return stain.replace("&", "")
    return default_stain


def normalize_slide_id(file_path: str) -> str:
    return Path(file_path).stem.replace("_patches", "")


def scan_h5_files(h5_dir: str):
    files = glob.glob(os.path.join(h5_dir, "*.h5"))
    if not files:
        files = glob.glob(os.path.join(h5_dir, "**", "*.h5"), recursive=True)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under: {h5_dir}")
    return files


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = scan_h5_files(H5_DIR)
    rows = []
    slide_rows = []

    print(f"[Index] Found {len(files)} h5 files.")

    for i, fp in enumerate(files, 1):
        slide_id = normalize_slide_id(fp)
        if slide_id in SKIP_SLIDES:
            continue

        try:
            with h5py.File(fp, "r") as f:
                if "features" not in f or "coords" not in f:
                    print(f"[Skip] Missing 'features' or 'coords': {fp}")
                    continue

                n_patches = int(f["features"].shape[0])
                feat_dim = int(f["features"].shape[1])
                coords = f["coords"][:]

            if n_patches == 0:
                continue

            stain = infer_stain(slide_id, DEFAULT_STAIN)

            patch_df = pd.DataFrame({
                "slide_id": slide_id,
                "patch_id": np.arange(n_patches, dtype=np.int32),
                "x": coords[:, 0].astype(np.int32),
                "y": coords[:, 1].astype(np.int32),
                "feature_file": fp,
                "feature_row": np.arange(n_patches, dtype=np.int32),
                "stain": stain,
                "feature_dim": feat_dim,
            })
            rows.append(patch_df)

            slide_rows.append({
                "slide_id": slide_id,
                "feature_file": fp,
                "n_patches": n_patches,
                "feature_dim": feat_dim,
                "stain": stain,
            })

            if i % 50 == 0 or i == len(files):
                print(f"  -> processed {i}/{len(files)} files")

        except Exception as e:
            print(f"[Warning] Failed on {fp}: {e}")

    if not rows:
        raise RuntimeError("No valid patch rows collected.")

    patch_index = pd.concat(rows, axis=0, ignore_index=True)
    slide_index = pd.DataFrame(slide_rows)

    raw_parquet = os.path.join(OUTPUT_DIR, "patch_index_raw.parquet")
    slide_csv = os.path.join(OUTPUT_DIR, "slide_index.csv")
    meta_json = os.path.join(OUTPUT_DIR, "index_metadata.json")

    patch_index.to_parquet(raw_parquet, index=False)
    slide_index.to_csv(slide_csv, index=False)

    meta = {
        "n_slides": int(slide_index["slide_id"].nunique()),
        "n_patches": int(len(patch_index)),
        "stains": sorted(patch_index["stain"].unique().tolist()),
        "feature_dims": sorted(patch_index["feature_dim"].unique().tolist()),
        "h5_dir": H5_DIR,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {raw_parquet}")
    print(f"[Saved] {slide_csv}")
    print(f"[Saved] {meta_json}")
    print(f"[Done] slides={meta['n_slides']}, patches={meta['n_patches']}")


if __name__ == "__main__":
    main()
