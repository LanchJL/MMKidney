import os
import gc
import math
import numpy as np
import pandas as pd
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

CLUSTER_PROTO_DIR = os.path.join(
    OUTPUT_DIR, f"cluster_representative_prototypes_{TARGET_STAIN}_{N_PROTOTYPES}"
)

# patch 实际裁剪大小（与你特征提取时保持一致）
PATCH_SIZE = 512

# 每个 cluster 选多少个代表性 prototype
TOP_PROTOS_PER_CLUSTER = 4

# 每个 prototype 取多少个 exemplar patch
TOP_PATCHES_PER_PROTO = 9

# montage 网格
N_COLS = 3
N_ROWS = 3

# 每个 tile 的显示尺寸
TILE_SIZE = 180

# 可选：只可视化部分 cluster；None 表示全部
SELECTED_CLUSTERS = None
# 例如：
# SELECTED_CLUSTERS = ["0_3", "1_3", "9"]

# 是否跳过特别小的 cluster
MIN_PATCHES_PER_CLUSTER = 300

# 是否在 tile 上写 slide_id / 坐标
DRAW_TILE_TEXT = True

TITLE_FONT_SIZE = 24
TEXT_FONT_SIZE = 16


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


def read_patch_from_wsi(wsi_path, x, y, patch_size=512):
    slide = openslide.OpenSlide(wsi_path)
    region = slide.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
    slide.close()
    return region


def make_placeholder(tile_size, text="Missing"):
    img = Image.new("RGB", (tile_size, tile_size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    draw.rectangle([0, 0, tile_size - 1, tile_size - 1], outline=(120, 120, 120), width=2)
    draw.text((10, tile_size // 2 - 10), text, fill=(80, 80, 80), font=font)
    return img


def draw_tile_annotation(img, text):
    draw = ImageDraw.Draw(img)
    font = get_font(TEXT_FONT_SIZE)
    pad = 4
    text_box_h = 20

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, img.height - text_box_h, img.width, img.height], fill=(255, 255, 255, 180))

    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)

    draw = ImageDraw.Draw(img_rgba)
    draw.text((pad, img.height - text_box_h + 2), text, fill=(0, 0, 0, 255), font=font)
    return img_rgba.convert("RGB")


def build_montage(images, title, n_cols=3, tile_size=180):
    n = len(images)
    n_rows = math.ceil(n / n_cols)

    title_font = get_font(TITLE_FONT_SIZE)
    pad = 10
    title_h = 42

    width = n_cols * tile_size + (n_cols + 1) * pad
    height = n_rows * tile_size + (n_rows + 1) * pad + title_h

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), title, fill=(0, 0, 0), font=title_font)

    for i, img in enumerate(images):
        r = i // n_cols
        c = i % n_cols
        x0 = pad + c * (tile_size + pad)
        y0 = title_h + pad + r * (tile_size + pad)

        img_resized = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(img_resized, (x0, y0))

    return canvas


def choose_clusters(cluster_sizes, selected=None, min_patches=300):
    if selected is not None:
        return [str(x) for x in selected]

    cluster_sizes = cluster_sizes[cluster_sizes["n_patches"] >= min_patches].copy()
    return cluster_sizes["final_cluster"].astype(str).tolist()


def main():
    os.makedirs(CLUSTER_PROTO_DIR, exist_ok=True)

    patch_df = load_table_auto(PATCH_FINAL)
    patch_df["final_cluster"] = patch_df["final_cluster"].astype(str)
    patch_df["proto_id"] = patch_df["proto_id"].astype(int)

    # cluster 总量统计
    cluster_sizes = (
        patch_df["final_cluster"]
        .value_counts()
        .rename_axis("final_cluster")
        .reset_index(name="n_patches")
        .sort_values("n_patches", ascending=False)
    )

    cluster_list = choose_clusters(
        cluster_sizes=cluster_sizes,
        selected=SELECTED_CLUSTERS,
        min_patches=MIN_PATCHES_PER_CLUSTER
    )

    print(f"[Cluster representative prototypes] n_clusters = {len(cluster_list)}")

    global_summary_rows = []

    for i, cluster_name in enumerate(cluster_list, 1):
        df_cluster = patch_df[patch_df["final_cluster"] == str(cluster_name)].copy()

        if len(df_cluster) == 0:
            continue

        cluster_dir = os.path.join(CLUSTER_PROTO_DIR, f"cluster_{cluster_name}")
        os.makedirs(cluster_dir, exist_ok=True)

        # 统计该 cluster 内各 prototype 的 patch 数
        proto_summary = (
            df_cluster.groupby("proto_id")
            .size()
            .reset_index(name="n_patches")
            .sort_values("n_patches", ascending=False)
        )

        proto_summary["rank_within_cluster"] = np.arange(1, len(proto_summary) + 1)

        # 选代表性 prototype
        rep_proto_df = proto_summary.head(TOP_PROTOS_PER_CLUSTER).copy()

        # 保存 cluster summary
        cluster_summary_path = os.path.join(cluster_dir, f"cluster_{cluster_name}_summary.csv")
        proto_summary.to_csv(cluster_summary_path, index=False)

        # 对每个 prototype 输出 montage
        rep_rows = []
        for _, row in rep_proto_df.iterrows():
            proto_id = int(row["proto_id"])
            n_patches = int(row["n_patches"])
            rank_in_cluster = int(row["rank_within_cluster"])

            df_proto = df_cluster[df_cluster["proto_id"] == proto_id].copy()
            df_proto = df_proto.sort_values("proto_dist", ascending=True).head(TOP_PATCHES_PER_PROTO).copy()

            images = []
            exemplar_rows = []

            for _, prow in df_proto.iterrows():
                slide_id = prow["slide_id"]
                x = int(prow["x"])
                y = int(prow["y"])
                proto_dist = float(prow["proto_dist"])

                wsi_path = find_wsi_path(WSI_DIR, slide_id)
                if wsi_path is None:
                    img = make_placeholder(TILE_SIZE, text="WSI missing")
                else:
                    try:
                        img = read_patch_from_wsi(wsi_path, x, y, patch_size=PATCH_SIZE)
                    except Exception:
                        img = make_placeholder(TILE_SIZE, text="Read failed")

                if DRAW_TILE_TEXT:
                    text = f"{slide_id[:12]} ({x},{y})"
                    img = draw_tile_annotation(img, text)

                images.append(img)
                exemplar_rows.append({
                    "final_cluster": cluster_name,
                    "proto_id": proto_id,
                    "slide_id": slide_id,
                    "x": x,
                    "y": y,
                    "proto_dist": proto_dist,
                    "rank_within_cluster": rank_in_cluster,
                    "proto_n_patches_in_cluster": n_patches,
                })

            while len(images) < N_ROWS * N_COLS:
                images.append(make_placeholder(TILE_SIZE, text="Empty"))

            title = (
                f"Cluster {cluster_name} | Proto {proto_id} | "
                f"rank={rank_in_cluster} | cluster_patches={len(df_cluster)} | proto_patches={n_patches}"
            )

            montage = build_montage(
                images=images[:N_ROWS * N_COLS],
                title=title,
                n_cols=N_COLS,
                tile_size=TILE_SIZE
            )

            save_png = os.path.join(
                cluster_dir,
                f"cluster_{cluster_name}_proto_{proto_id:04d}_montage.png"
            )
            montage.save(save_png)

            exemplar_csv = os.path.join(
                cluster_dir,
                f"cluster_{cluster_name}_proto_{proto_id:04d}_exemplars.csv"
            )
            pd.DataFrame(exemplar_rows).to_csv(exemplar_csv, index=False)

            rep_rows.append({
                "final_cluster": cluster_name,
                "proto_id": proto_id,
                "rank_within_cluster": rank_in_cluster,
                "n_patches_in_cluster": n_patches,
                "montage_path": save_png,
                "exemplar_csv": exemplar_csv,
            })

            global_summary_rows.append({
                "final_cluster": cluster_name,
                "cluster_total_patches": int(len(df_cluster)),
                "proto_id": proto_id,
                "rank_within_cluster": rank_in_cluster,
                "n_patches_in_cluster": n_patches,
                "cluster_summary_csv": cluster_summary_path,
                "montage_path": save_png,
                "exemplar_csv": exemplar_csv,
            })

        rep_summary_df = pd.DataFrame(rep_rows)
        rep_summary_path = os.path.join(
            cluster_dir,
            f"cluster_{cluster_name}_representative_prototypes.csv"
        )
        rep_summary_df.to_csv(rep_summary_path, index=False)

        if i % 5 == 0 or i == len(cluster_list):
            print(f"  -> [{i}/{len(cluster_list)}] finished cluster {cluster_name}")

        gc.collect()

    global_summary_df = pd.DataFrame(global_summary_rows)
    global_summary_path = os.path.join(CLUSTER_PROTO_DIR, "all_clusters_representative_prototypes.csv")
    global_summary_df.to_csv(global_summary_path, index=False)

    print(f"[Saved] {global_summary_path}")
    print(f"[Done] representative prototypes saved to: {CLUSTER_PROTO_DIR}")


if __name__ == "__main__":
    main()
