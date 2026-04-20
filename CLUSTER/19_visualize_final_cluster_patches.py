import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import openslide


def load_table_auto(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def find_wsi_path(wsi_dir: str, slide_id: str):
    for ext in [".mrxs", ".svs", ".ndpi", ".tiff", ".tif"]:
        p1 = os.path.join(wsi_dir, str(slide_id) + ext)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(wsi_dir, str(slide_id).split("_")[0] + ext)
        if os.path.exists(p2):
            return p2
    return None


def safe_font(size=14):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for fp in cands:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def select_representatives(
    df_cluster: pd.DataFrame,
    id_col: str,
    n_select: int = 9,
    max_per_slide: int = 2,
) -> pd.DataFrame:
    # Lower proto_dist is treated as more representative.
    if "proto_dist" in df_cluster.columns:
        d = df_cluster.sort_values("proto_dist", ascending=True).copy()
    else:
        d = df_cluster.copy()

    selected_rows = []
    per_slide = {}
    used_patch = set()

    # 1st pass: prioritize slide diversity (one per slide)
    for _, r in d.iterrows():
        sid = str(r[id_col])
        pid = int(r["patch_id"]) if "patch_id" in d.columns else int(_)
        key = (sid, pid)
        if key in used_patch:
            continue
        if sid in per_slide:
            continue
        selected_rows.append(r)
        used_patch.add(key)
        per_slide[sid] = 1
        if len(selected_rows) >= n_select:
            break

    # 2nd pass: fill remaining slots with cap per slide
    if len(selected_rows) < n_select:
        for _, r in d.iterrows():
            sid = str(r[id_col])
            pid = int(r["patch_id"]) if "patch_id" in d.columns else int(_)
            key = (sid, pid)
            if key in used_patch:
                continue
            if per_slide.get(sid, 0) >= max_per_slide:
                continue
            selected_rows.append(r)
            used_patch.add(key)
            per_slide[sid] = per_slide.get(sid, 0) + 1
            if len(selected_rows) >= n_select:
                break

    if not selected_rows:
        return d.head(0)
    return pd.DataFrame(selected_rows).head(n_select)


def read_patch(wsi_path: str, x: int, y: int, patch_size: int) -> Image.Image:
    slide = openslide.OpenSlide(wsi_path)
    img = slide.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
    slide.close()
    return img


def make_montage(
    tiles: List[Image.Image],
    labels: List[str],
    n_cols: int = 3,
    tile_size: int = 256,
    pad: int = 8,
) -> Image.Image:
    n = len(tiles)
    n_rows = max(1, int(np.ceil(n / n_cols)))
    title_h = 22
    W = n_cols * tile_size + (n_cols + 1) * pad
    H = n_rows * (tile_size + title_h) + (n_rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = safe_font(13)

    for i, (img, txt) in enumerate(zip(tiles, labels)):
        r = i // n_cols
        c = i % n_cols
        x0 = pad + c * (tile_size + pad)
        y0 = pad + r * (tile_size + title_h + pad)
        tile = img.resize((tile_size, tile_size), Image.BILINEAR)
        canvas.paste(tile, (x0, y0))
        draw.rectangle([x0, y0 + tile_size, x0 + tile_size, y0 + tile_size + title_h], fill=(255, 255, 255))
        draw.text((x0 + 4, y0 + tile_size + 3), txt, fill=(30, 30, 30), font=font)
    return canvas


def main():
    p = argparse.ArgumentParser("Visualize 9 representative patches per final cluster")
    p.add_argument("--patch-final", required=True, help="Final patch table (parquet/csv/pkl)")
    p.add_argument("--wsi-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--n-per-cluster", type=int, default=9)
    p.add_argument("--max-per-slide", type=int, default=2)
    p.add_argument("--top-n-clusters", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table_auto(args.patch_final).copy()
    id_col = "slide_id" if "slide_id" in df.columns else ("sample_id" if "sample_id" in df.columns else "")
    if not id_col:
        raise ValueError("patch-final missing slide_id/sample_id")
    for c in ["final_cluster", "x", "y"]:
        if c not in df.columns:
            raise ValueError(f"patch-final missing {c}")

    df["final_cluster"] = df["final_cluster"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0).astype(int)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)

    cluster_order = df["final_cluster"].value_counts()
    if args.top_n_clusters > 0:
        cluster_order = cluster_order.head(args.top_n_clusters)

    selected_all = []
    for i, cluster_name in enumerate(cluster_order.index.tolist(), 1):
        dc = df[df["final_cluster"] == cluster_name].copy()
        sel = select_representatives(
            dc, id_col=id_col, n_select=args.n_per_cluster, max_per_slide=args.max_per_slide
        )
        if len(sel) == 0:
            continue

        tiles: List[Image.Image] = []
        labels: List[str] = []
        out_rows = []

        for _, r in sel.iterrows():
            sid = str(r[id_col])
            x, y = int(r["x"]), int(r["y"])
            wsi_path = find_wsi_path(args.wsi_dir, sid)
            if not wsi_path:
                continue
            try:
                img = read_patch(wsi_path, x, y, args.patch_size)
            except Exception:
                continue
            txt = f"{sid[:12]} ({x},{y})"
            tiles.append(img)
            labels.append(txt)
            out_rows.append(
                {
                    "final_cluster": cluster_name,
                    id_col: sid,
                    "x": x,
                    "y": y,
                    "wsi_path": wsi_path,
                }
            )

        if len(tiles) == 0:
            continue

        montage = make_montage(tiles, labels, n_cols=3, tile_size=256, pad=8)
        out_png = out_dir / f"cluster_{cluster_name}_top{len(tiles)}.png"
        montage.save(out_png)
        print(f"[saved] {out_png}")

        out_csv = out_dir / f"cluster_{cluster_name}_selected.csv"
        pd.DataFrame(out_rows).to_csv(out_csv, index=False)
        selected_all.extend(out_rows)

        if i % 10 == 0 or i == len(cluster_order):
            print(f"[progress] {i}/{len(cluster_order)} clusters")

    if selected_all:
        all_csv = out_dir / "all_selected_representative_patches.csv"
        pd.DataFrame(selected_all).to_csv(all_csv, index=False)
        print(f"[saved] {all_csv}")
    print("[done] representative patch visualization completed")


if __name__ == "__main__":
    main()

