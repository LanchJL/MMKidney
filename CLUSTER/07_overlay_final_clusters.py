import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw
import openslide


# ================= 配置 =================
OUTPUT_DIR = "prototype_pipeline_output"

TARGET_STAIN = "HE"
N_PROTOTYPES = 2048

PATCH_FINAL = os.path.join(
    OUTPUT_DIR, f"patch_final_clusters_{TARGET_STAIN}_{N_PROTOTYPES}.parquet"
)

WSI_DIR = "/media/a6000/3E5C99E35C9995ED/jcy/dataset/medical/WSIs/"

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR, f"overlay_final_clusters_{TARGET_STAIN}_{N_PROTOTYPES}"
)

DOWNSAMPLE = 32
ALPHA = 0.40

# 可选：只画 patch 数最多的前 N 个 slide；None 表示全画
TOP_N_SLIDES = None

# 可选：只画 patch 数最多的前 N 个 cluster；None 表示全部
TOP_N_CLUSTERS = None


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


def find_wsi_path(wsi_dir: str, slide_id: str):
    for ext in [".mrxs", ".svs", ".ndpi", ".tiff", ".tif"]:
        p1 = os.path.join(wsi_dir, slide_id + ext)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(wsi_dir, slide_id.split("_")[0] + ext)
        if os.path.exists(p2):
            return p2
    return None


def build_color_map(cluster_labels):
    cluster_labels = [str(x) for x in cluster_labels]
    n = len(cluster_labels)

    # cluster 数较多时用 turbo，更容易区分
    cmap = plt.cm.turbo if n > 20 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(n, 1)))

    color_map = {}
    for i, lab in enumerate(cluster_labels):
        rgb = mcolors.to_rgb(colors[i])
        color_map[lab] = tuple(int(v * 255) for v in rgb)
    return color_map


def estimate_patch_step(xs):
    xs = np.sort(np.unique(xs))
    if len(xs) > 1:
        diffs = np.diff(xs)
        step = int(np.min(diffs))
        if step < 10:
            step = 256
    else:
        step = 256
    return step


def overlay_one_slide(df_slide, wsi_path, save_path, color_map, downsample=32, alpha=0.4):
    slide = openslide.OpenSlide(wsi_path)
    w_orig, h_orig = slide.dimensions

    w_target = max(1, w_orig // downsample)
    h_target = max(1, h_orig // downsample)

    bg_img = slide.get_thumbnail((w_target, h_target)).convert("RGBA")
    w_real, h_real = bg_img.size

    scale_x = w_real / w_orig
    scale_y = h_real / h_orig

    xs = df_slide["x"].to_numpy(dtype=np.int32)
    ys = df_slide["y"].to_numpy(dtype=np.int32)
    cls = df_slide["final_cluster"].astype(str).to_numpy()

    step = estimate_patch_step(xs)
    bw = int(step * scale_x + 1)
    bh = int(step * scale_y + 1)

    mask_img = Image.new("RGBA", (w_real, h_real), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_img)

    for x, y, lab in zip(xs, ys, cls):
        if lab not in color_map:
            continue
        x0 = int(x * scale_x)
        y0 = int(y * scale_y)
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=color_map[lab] + (255,))

    _, _, _, a = mask_img.split()
    mask_img.putalpha(a.point(lambda p: int(255 * alpha) if p > 0 else 0))
    final_img = Image.alpha_composite(bg_img, mask_img)
    final_img.save(save_path)

    slide.close()
    del bg_img, mask_img, final_img
    gc.collect()


def main():
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    patch_df = load_table_auto(PATCH_FINAL)
    patch_df["final_cluster"] = patch_df["final_cluster"].astype(str)

    # 选择 cluster
    cluster_counts = patch_df["final_cluster"].value_counts()
    if TOP_N_CLUSTERS is not None:
        keep_clusters = cluster_counts.head(TOP_N_CLUSTERS).index.tolist()
        patch_df = patch_df[patch_df["final_cluster"].isin(keep_clusters)].copy()
    else:
        keep_clusters = cluster_counts.index.tolist()

    color_map = build_color_map(keep_clusters)

    # 保存颜色表
    color_df = pd.DataFrame({
        "final_cluster": list(color_map.keys()),
        "r": [color_map[k][0] for k in color_map],
        "g": [color_map[k][1] for k in color_map],
        "b": [color_map[k][2] for k in color_map],
        "n_patches": [int(cluster_counts.get(k, 0)) for k in color_map],
    }).sort_values("final_cluster")
    color_df.to_csv(os.path.join(OVERLAY_DIR, "final_cluster_colors.csv"), index=False)

    # 选择 slide
    slide_counts = patch_df["slide_id"].value_counts()
    if TOP_N_SLIDES is not None:
        slide_ids = slide_counts.head(TOP_N_SLIDES).index.tolist()
    else:
        slide_ids = slide_counts.index.tolist()

    print(f"[Overlay] slides={len(slide_ids)}, clusters={len(keep_clusters)}")

    skipped = []
    for i, slide_id in enumerate(slide_ids, 1):
        wsi_path = find_wsi_path(WSI_DIR, slide_id)
        if wsi_path is None:
            skipped.append(slide_id)
            continue

        df_slide = patch_df[patch_df["slide_id"] == slide_id].copy()
        save_path = os.path.join(OVERLAY_DIR, f"{slide_id}_final_cluster_overlay.png")

        try:
            overlay_one_slide(
                df_slide=df_slide,
                wsi_path=wsi_path,
                save_path=save_path,
                color_map=color_map,
                downsample=DOWNSAMPLE,
                alpha=ALPHA
            )
            if i % 20 == 0 or i == len(slide_ids):
                print(f"  -> [{i}/{len(slide_ids)}] saved: {save_path}")
        except Exception as e:
            print(f"[Warning] Failed on {slide_id}: {e}")
            skipped.append(slide_id)

    if skipped:
        pd.DataFrame({"slide_id": skipped}).to_csv(
            os.path.join(OVERLAY_DIR, "overlay_skipped_slides.csv"),
            index=False
        )

    print(f"[Done] overlays saved to: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
