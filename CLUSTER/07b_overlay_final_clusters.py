import os
import gc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import openslide


# ================= 配置 =================
OUTPUT_DIR = "prototype_pipeline_output"

TARGET_STAIN = "HE"
N_PROTOTYPES = 2048

PATCH_FINAL = os.path.join(
    OUTPUT_DIR, f"patch_final_clusters_{TARGET_STAIN}_{N_PROTOTYPES}.parquet"
)

WSI_DIR = "/media/a6000/3E5C99E35C9995ED/jcy/dataset/medical/WSIs/"

OVERLAY_ROOT = os.path.join(
    OUTPUT_DIR, f"overlay_final_clusters_{TARGET_STAIN}_{N_PROTOTYPES}"
)
OVERLAY_ALL_DIR = os.path.join(OVERLAY_ROOT, "all_clusters")
OVERLAY_FILTERED_DIR = os.path.join(OVERLAY_ROOT, "filtered_clusters")

# 高清输出：16 通常比较合适；8 更清晰但更慢更占内存
DOWNSAMPLE = 16
ALPHA = 0.40

# 可选：只画 patch 数最多的前 N 个 slide；None 表示全部
TOP_N_SLIDES = None

# 可选：只保留 patch 数最多的前 N 个 cluster 来上色；None 表示全部
TOP_N_CLUSTERS = None

# 这些 cluster 会从 filtered 版本中移除，但 all 版本仍保留
FILTER_CLUSTERS = {
    "10",
    "7",
    "5",
    "8",
    "2",
    "4",
}

# 是否在图右上角加 legend
ADD_LEGEND = True

# legend 中最多显示多少个 cluster
MAX_LEGEND_ITEMS = 20

# 是否按 patch 数对 legend 排序
LEGEND_SORT_BY_COUNT = True


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


def get_font(size=18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def add_legend(image_rgba, color_map, cluster_counts, filtered=False):
    if not ADD_LEGEND:
        return image_rgba

    draw = ImageDraw.Draw(image_rgba)
    font_title = get_font(24)
    font_text = get_font(18)

    items = list(cluster_counts.items())
    if LEGEND_SORT_BY_COUNT:
        items = sorted(items, key=lambda x: x[1], reverse=True)
    else:
        items = sorted(items, key=lambda x: str(x[0]))

    if filtered:
        items = [(k, v) for k, v in items if str(k) not in FILTER_CLUSTERS]

    items = items[:MAX_LEGEND_ITEMS]

    if len(items) == 0:
        return image_rgba

    pad = 14
    swatch = 18
    line_h = 24
    title = "Filtered clusters" if filtered else "All clusters"

    legend_w = 280
    legend_h = pad * 2 + 28 + len(items) * line_h

    x0 = image_rgba.width - legend_w - 20
    y0 = 20
    x1 = image_rgba.width - 20
    y1 = y0 + legend_h

    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=12,
        fill=(255, 255, 255, 210),
        outline=(50, 50, 50, 255),
        width=2
    )

    draw.text((x0 + pad, y0 + 8), title, fill=(0, 0, 0, 255), font=font_title)

    yy = y0 + 40
    for lab, cnt in items:
        lab = str(lab)
        color = color_map.get(lab, (128, 128, 128))
        draw.rectangle(
            [x0 + pad, yy + 3, x0 + pad + swatch, yy + 3 + swatch],
            fill=color + (255,),
            outline=(0, 0, 0, 255)
        )
        text = f"{lab} ({cnt})"
        draw.text((x0 + pad + swatch + 10, yy), text, fill=(0, 0, 0, 255), font=font_text)
        yy += line_h

    return image_rgba


def overlay_one_slide(df_slide, wsi_path, save_all, save_filtered, color_map, downsample=16, alpha=0.4):
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

    mask_all = Image.new("RGBA", (w_real, h_real), (0, 0, 0, 0))
    mask_filtered = Image.new("RGBA", (w_real, h_real), (0, 0, 0, 0))

    draw_all = ImageDraw.Draw(mask_all)
    draw_filtered = ImageDraw.Draw(mask_filtered)

    for x, y, lab in zip(xs, ys, cls):
        if lab not in color_map:
            continue

        x0 = int(x * scale_x)
        y0 = int(y * scale_y)
        rect = [x0, y0, x0 + bw, y0 + bh]

        draw_all.rectangle(rect, fill=color_map[lab] + (255,))

        if lab not in FILTER_CLUSTERS:
            draw_filtered.rectangle(rect, fill=color_map[lab] + (255,))

    _, _, _, a_all = mask_all.split()
    mask_all.putalpha(a_all.point(lambda p: int(255 * alpha) if p > 0 else 0))

    _, _, _, a_filtered = mask_filtered.split()
    mask_filtered.putalpha(a_filtered.point(lambda p: int(255 * alpha) if p > 0 else 0))

    final_all = Image.alpha_composite(bg_img, mask_all)
    final_filtered = Image.alpha_composite(bg_img, mask_filtered)

    slide_cluster_counts = df_slide["final_cluster"].astype(str).value_counts().to_dict()
    final_all = add_legend(final_all, color_map, slide_cluster_counts, filtered=False)
    final_filtered = add_legend(final_filtered, color_map, slide_cluster_counts, filtered=True)

    final_all.save(save_all)
    final_filtered.save(save_filtered)

    slide.close()
    del bg_img, mask_all, mask_filtered, final_all, final_filtered
    gc.collect()


def main():
    os.makedirs(OVERLAY_ROOT, exist_ok=True)
    os.makedirs(OVERLAY_ALL_DIR, exist_ok=True)
    os.makedirs(OVERLAY_FILTERED_DIR, exist_ok=True)

    patch_df = load_table_auto(PATCH_FINAL)
    patch_df["final_cluster"] = patch_df["final_cluster"].astype(str)

    cluster_counts = patch_df["final_cluster"].value_counts()

    if TOP_N_CLUSTERS is not None:
        keep_clusters = cluster_counts.head(TOP_N_CLUSTERS).index.tolist()
        patch_df = patch_df[patch_df["final_cluster"].isin(keep_clusters)].copy()
        cluster_counts = patch_df["final_cluster"].value_counts()
    else:
        keep_clusters = cluster_counts.index.tolist()

    color_map = build_color_map(keep_clusters)

    color_df = pd.DataFrame({
        "final_cluster": list(color_map.keys()),
        "r": [color_map[k][0] for k in color_map],
        "g": [color_map[k][1] for k in color_map],
        "b": [color_map[k][2] for k in color_map],
        "n_patches": [int(cluster_counts.get(k, 0)) for k in color_map],
        "filtered_out": [str(k) in FILTER_CLUSTERS for k in color_map],
    }).sort_values("final_cluster")
    color_df.to_csv(os.path.join(OVERLAY_ROOT, "final_cluster_colors.csv"), index=False)

    slide_counts = patch_df["slide_id"].value_counts()
    if TOP_N_SLIDES is not None:
        slide_ids = slide_counts.head(TOP_N_SLIDES).index.tolist()
    else:
        slide_ids = slide_counts.index.tolist()

    print(f"[Overlay] slides={len(slide_ids)}, clusters={len(keep_clusters)}, downsample={DOWNSAMPLE}")

    skipped = []
    for i, slide_id in enumerate(slide_ids, 1):
        wsi_path = find_wsi_path(WSI_DIR, slide_id)
        if wsi_path is None:
            skipped.append(slide_id)
            continue

        df_slide = patch_df[patch_df["slide_id"] == slide_id].copy()

        save_all = os.path.join(
            OVERLAY_ALL_DIR,
            f"{slide_id}_overlay_all.png"
        )
        save_filtered = os.path.join(
            OVERLAY_FILTERED_DIR,
            f"{slide_id}_overlay_filtered.png"
        )

        try:
            overlay_one_slide(
                df_slide=df_slide,
                wsi_path=wsi_path,
                save_all=save_all,
                save_filtered=save_filtered,
                color_map=color_map,
                downsample=DOWNSAMPLE,
                alpha=ALPHA
            )
            if i % 20 == 0 or i == len(slide_ids):
                print(f"  -> [{i}/{len(slide_ids)}] saved: {slide_id}")
        except Exception as e:
            print(f"[Warning] Failed on {slide_id}: {e}")
            skipped.append(slide_id)

    if skipped:
        pd.DataFrame({"slide_id": skipped}).to_csv(
            os.path.join(OVERLAY_ROOT, "overlay_skipped_slides.csv"),
            index=False
        )

    print(f"[Done] overlays saved to: {OVERLAY_ROOT}")


if __name__ == "__main__":
    main()
