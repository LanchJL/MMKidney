import os
import json
import numpy as np
import pandas as pd
import openslide
from tqdm import tqdm


# ================= 配置 =================
PATCH_INDEX_IN = "prototype_pipeline_output/patch_index_raw.parquet"
OUTPUT_DIR = "prototype_output_0.6"

WSI_DIR = "/media/a6000/3E5C99E35C9995ED/jcy/dataset/medical/WSIs/"

PATCH_SIZE = 512
MASK_DOWNSAMPLE = 64
TISSUE_FRAC_MIN = 0.40

# HSV 背景过滤参数
BG_V_THR = 0.95
BG_S_THR = 0.08


def find_wsi_path(wsi_dir: str, slide_id: str):
    for ext in [".mrxs", ".svs", ".ndpi", ".tiff", ".tif"]:
        p1 = os.path.join(wsi_dir, slide_id + ext)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(wsi_dir, slide_id.split("_")[0] + ext)
        if os.path.exists(p2):
            return p2
    return None


def build_tissue_mask(slide, downsample_for_mask=64, bg_v_thr=0.92, bg_s_thr=0.10):
    w, h = slide.dimensions
    tw, th = max(1, w // downsample_for_mask), max(1, h // downsample_for_mask)

    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    thumb_np = np.asarray(thumb).astype(np.float32) / 255.0

    cmax = np.max(thumb_np, axis=-1)
    cmin = np.min(thumb_np, axis=-1)
    delta = cmax - cmin

    v = cmax
    s = np.where(cmax == 0, 0, delta / (cmax + 1e-8))

    bg = (v > bg_v_thr) & (s < bg_s_thr)
    tissue_mask = ~bg
    return tissue_mask, (w, h)


def compute_tissue_frac_for_slide(df_slide, wsi_path, patch_size=512, downsample_for_mask=64,
                                  bg_v_thr=0.92, bg_s_thr=0.10):
    slide = openslide.OpenSlide(wsi_path)
    tissue_mask, (w, h) = build_tissue_mask(
        slide,
        downsample_for_mask=downsample_for_mask,
        bg_v_thr=bg_v_thr,
        bg_s_thr=bg_s_thr
    )

    sx = tissue_mask.shape[1] / w
    sy = tissue_mask.shape[0] / h

    xs = df_slide["x"].to_numpy(dtype=np.int32)
    ys = df_slide["y"].to_numpy(dtype=np.int32)
    tissue_fracs = np.zeros(len(df_slide), dtype=np.float32)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x0 = int(x * sx)
        y0 = int(y * sy)
        x1 = int((x + patch_size) * sx)
        y1 = int((y + patch_size) * sy)

        x0 = max(0, min(x0, tissue_mask.shape[1] - 1))
        x1 = max(0, min(x1, tissue_mask.shape[1]))
        y0 = max(0, min(y0, tissue_mask.shape[0] - 1))
        y1 = max(0, min(y1, tissue_mask.shape[0]))

        region = tissue_mask[y0:y1, x0:x1]
        tissue_fracs[i] = float(region.mean()) if region.size > 0 else 0.0

    slide.close()
    return tissue_fracs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patch_index = pd.read_parquet(PATCH_INDEX_IN)
    out_rows = []
    slide_stats = []

    slide_ids = patch_index["slide_id"].unique()
    print(f"[QC] slides to process: {len(slide_ids)}")

    for slide_id in tqdm(slide_ids, desc="Tissue QC"):
        df_slide = patch_index[patch_index["slide_id"] == slide_id].copy()
        wsi_path = find_wsi_path(WSI_DIR, slide_id)

        if wsi_path is None:
            df_slide["tissue_frac"] = 1.0
            df_slide["qc_pass"] = True
            slide_stats.append({
                "slide_id": slide_id,
                "wsi_found": False,
                "n_patches": int(len(df_slide)),
                "tissue_frac_mean": 1.0,
                "tissue_frac_p10": 1.0,
                "qc_pass_rate": 1.0,
            })
            out_rows.append(df_slide)
            continue

        try:
            tissue_fracs = compute_tissue_frac_for_slide(
                df_slide,
                wsi_path=wsi_path,
                patch_size=PATCH_SIZE,
                downsample_for_mask=MASK_DOWNSAMPLE,
                bg_v_thr=BG_V_THR,
                bg_s_thr=BG_S_THR,
            )
            df_slide["tissue_frac"] = tissue_fracs
            df_slide["qc_pass"] = df_slide["tissue_frac"] >= TISSUE_FRAC_MIN

            slide_stats.append({
                "slide_id": slide_id,
                "wsi_found": True,
                "n_patches": int(len(df_slide)),
                "tissue_frac_mean": float(df_slide["tissue_frac"].mean()),
                "tissue_frac_p10": float(np.percentile(df_slide["tissue_frac"], 10)),
                "qc_pass_rate": float(df_slide["qc_pass"].mean()),
            })
            out_rows.append(df_slide)

        except Exception as e:
            print(f"[Warning] Tissue QC failed for {slide_id}: {e}")
            df_slide["tissue_frac"] = 1.0
            df_slide["qc_pass"] = True
            slide_stats.append({
                "slide_id": slide_id,
                "wsi_found": False,
                "n_patches": int(len(df_slide)),
                "tissue_frac_mean": 1.0,
                "tissue_frac_p10": 1.0,
                "qc_pass_rate": 1.0,
            })
            out_rows.append(df_slide)

    patch_index_qc = pd.concat(out_rows, axis=0, ignore_index=True)
    slide_stats_df = pd.DataFrame(slide_stats)

    out_parquet = os.path.join(OUTPUT_DIR, "patch_index_qc.parquet")
    out_slide_csv = os.path.join(OUTPUT_DIR, "slide_qc_stats.csv")
    out_meta_json = os.path.join(OUTPUT_DIR, "qc_metadata.json")

    patch_index_qc.to_parquet(out_parquet, index=False)
    slide_stats_df.to_csv(out_slide_csv, index=False)

    meta = {
        "patch_size": PATCH_SIZE,
        "mask_downsample": MASK_DOWNSAMPLE,
        "tissue_frac_min": TISSUE_FRAC_MIN,
        "bg_v_thr": BG_V_THR,
        "bg_s_thr": BG_S_THR,
        "n_patches_total": int(len(patch_index_qc)),
        "n_patches_pass": int(patch_index_qc["qc_pass"].sum()),
        "pass_rate": float(patch_index_qc["qc_pass"].mean()),
    }
    with open(out_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {out_parquet}")
    print(f"[Saved] {out_slide_csv}")
    print(f"[Saved] {out_meta_json}")
    print(f"[Done] qc pass patches = {meta['n_patches_pass']} / {meta['n_patches_total']}")


if __name__ == "__main__":
    main()
