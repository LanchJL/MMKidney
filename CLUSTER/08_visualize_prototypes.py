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

PATCH_ASSIGNMENTS = os.path.join(
    OUTPUT_DIR, f"patch_assignments_{TARGET_STAIN}_{N_PROTOTYPES}.parquet"
)

PROTOTYPE_USAGE = os.path.join(
    OUTPUT_DIR, f"prototype_usage_{TARGET_STAIN}_{N_PROTOTYPES}.csv"
)

WSI_DIR = "/media/a6000/3E5C99E35C9995ED/jcy/dataset/medical/WSIs/"

MONTAGE_DIR = os.path.join(
    OUTPUT_DIR, f"prototype_montages_{TARGET_STAIN}_{N_PROTOTYPES}"
)

# patch 实际裁剪大小（应与你特征提取时 patch 大小一致）
PATCH_SIZE = 512

# 每个 prototype 取多少个 exemplar
TOP_K = 16

# montage 网格
N_COLS = 4
N_ROWS = 4

# 每个 tile 最终显示大小（为了拼图更紧凑）
TILE_SIZE = 160

# 默认可视化使用频率最高的前 N 个 prototype
TOP_N_PROTOTYPES = 50

# 也可以手动指定 prototype id；若不为 None，则优先使用这个列表
SELECTED_PROTOTYPES = None
# 例如：
# SELECTED_PROTOTYPES = [0, 1, 2, 10, 25]

# 是否在每个 tile 上标注 slide_id 和坐标
DRAW_TILE_TEXT = True

# montage 标题字号
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

    # 半透明底条
    text_box_h = 20
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, img.height - text_box_h, img.width, img.height], fill=(255, 255, 255, 180))
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)

    draw = ImageDraw.Draw(img_rgba)
    draw.text((pad, img.height - text_box_h + 2), text, fill=(0, 0, 0, 255), font=font)
    return img_rgba.convert("RGB")


def build_montage(images, title, n_cols=4, tile_size=160):
    n = len(images)
    n_rows = math.ceil(n / n_cols)

    title_font = get_font(TITLE_FONT_SIZE)
    pad = 10
    title_h = 40

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


def choose_prototypes(usage_df, selected=None, top_n=50):
    if selected is not None:
        return [int(x) for x in selected]

    usage_df = usage_df.sort_values("n_patches", ascending=False)
    return usage_df.head(top_n)["proto_id"].astype(int).tolist()


def main():
    os.makedirs(MONTAGE_DIR, exist_ok=True)

    patch_df = load_table_auto(PATCH_ASSIGNMENTS)
    usage_df = pd.read_csv(PROTOTYPE_USAGE)

    patch_df["proto_id"] = patch_df["proto_id"].astype(int)
    usage_df["proto_id"] = usage_df["proto_id"].astype(int)

    proto_ids = choose_prototypes(
        usage_df=usage_df,
        selected=SELECTED_PROTOTYPES,
        top_n=TOP_N_PROTOTYPES
    )

    print(f"[Visualize] n_selected_prototypes = {len(proto_ids)}")

    usage_map = dict(zip(usage_df["proto_id"], usage_df["n_patches"]))

    for i, proto_id in enumerate(proto_ids, 1):
        df_proto = patch_df[patch_df["proto_id"] == proto_id].copy()

        if len(df_proto) == 0:
            print(f"[Skip] proto {proto_id}: no assigned patches")
            continue

        # 最接近 prototype 的 exemplar
        df_proto = df_proto.sort_values("proto_dist", ascending=True).head(TOP_K).copy()

        images = []
        exemplar_rows = []

        for _, row in df_proto.iterrows():
            slide_id = row["slide_id"]
            x = int(row["x"])
            y = int(row["y"])
            proto_dist = float(row["proto_dist"])

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
                "proto_id": proto_id,
                "slide_id": slide_id,
                "x": x,
                "y": y,
                "proto_dist": proto_dist,
            })

        # 如果 exemplar 数不足，补空白
        while len(images) < N_ROWS * N_COLS:
            images.append(make_placeholder(TILE_SIZE, text="Empty"))

        title = f"Prototype {proto_id} | n_patches={usage_map.get(proto_id, 0)} | top-{TOP_K} exemplars"
        montage = build_montage(
            images=images[:N_ROWS * N_COLS],
            title=title,
            n_cols=N_COLS,
            tile_size=TILE_SIZE
        )

        save_png = os.path.join(MONTAGE_DIR, f"proto_{proto_id:04d}_montage.png")
        montage.save(save_png)

        exemplar_csv = os.path.join(MONTAGE_DIR, f"proto_{proto_id:04d}_exemplars.csv")
        pd.DataFrame(exemplar_rows).to_csv(exemplar_csv, index=False)

        if i % 10 == 0 or i == len(proto_ids):
            print(f"  -> [{i}/{len(proto_ids)}] saved proto {proto_id}")

        del images
        gc.collect()

    print(f"[Done] prototype montages saved to: {MONTAGE_DIR}")


if __name__ == "__main__":
    main()
