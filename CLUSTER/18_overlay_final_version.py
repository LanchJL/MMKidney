import argparse
import gc
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

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


def build_color_map(cluster_labels):
    cluster_labels = [str(x) for x in cluster_labels]
    n = len(cluster_labels)
    cmap = plt.cm.turbo if n > 20 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(1, n)))
    out = {}
    for i, c in enumerate(cluster_labels):
        rgb = mcolors.to_rgb(colors[i])
        out[c] = tuple(int(v * 255) for v in rgb)
    return out


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
    for x, y, c in zip(xs, ys, cls):
        if c not in color_map:
            continue
        x0 = int(x * scale_x)
        y0 = int(y * scale_y)
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=color_map[c] + (255,))

    _, _, _, a = mask_img.split()
    mask_img.putalpha(a.point(lambda p: int(255 * alpha) if p > 0 else 0))
    out = Image.alpha_composite(bg_img, mask_img)
    out.save(save_path)
    slide.close()
    del bg_img, mask_img, out
    gc.collect()


def main():
    p = argparse.ArgumentParser("Overlay final version patch clusters to WSI")
    p.add_argument("--patch-final", required=True)
    p.add_argument("--wsi-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--downsample", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.40)
    p.add_argument("--top-n-slides", type=int, default=0)
    p.add_argument("--top-n-clusters", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table_auto(args.patch_final).copy()
    if "final_cluster" not in df.columns:
        raise ValueError("patch-final missing final_cluster")
    id_col = "slide_id" if "slide_id" in df.columns else ("sample_id" if "sample_id" in df.columns else "")
    if not id_col:
        raise ValueError("patch-final missing slide_id/sample_id")
    for c in ["x", "y"]:
        if c not in df.columns:
            raise ValueError(f"patch-final missing {c}")

    df["final_cluster"] = df["final_cluster"].astype(str)

    cluster_counts = df["final_cluster"].value_counts()
    if args.top_n_clusters > 0:
        keep = cluster_counts.head(args.top_n_clusters).index.tolist()
        df = df[df["final_cluster"].isin(keep)].copy()
        cluster_counts = df["final_cluster"].value_counts()

    color_map = build_color_map(cluster_counts.index.tolist())
    color_df = pd.DataFrame(
        {
            "final_cluster": list(color_map.keys()),
            "r": [color_map[k][0] for k in color_map],
            "g": [color_map[k][1] for k in color_map],
            "b": [color_map[k][2] for k in color_map],
            "n_patches": [int(cluster_counts.get(k, 0)) for k in color_map],
        }
    ).sort_values("final_cluster")
    color_df.to_csv(out_dir / "final_cluster_colors.csv", index=False)

    slide_counts = df[id_col].value_counts()
    slide_ids = slide_counts.head(args.top_n_slides).index.tolist() if args.top_n_slides > 0 else slide_counts.index.tolist()

    skipped = []
    print(f"[overlay] slides={len(slide_ids)} clusters={len(cluster_counts)}")
    for i, sid in enumerate(slide_ids, 1):
        wsi_path = find_wsi_path(args.wsi_dir, sid)
        if wsi_path is None:
            skipped.append(str(sid))
            continue
        d = df[df[id_col] == sid].copy()
        out_png = out_dir / f"{sid}_final_overlay.png"
        try:
            overlay_one_slide(
                d,
                str(wsi_path),
                str(out_png),
                color_map=color_map,
                downsample=args.downsample,
                alpha=args.alpha,
            )
            if i % 20 == 0 or i == len(slide_ids):
                print(f"  -> [{i}/{len(slide_ids)}] saved {out_png}")
        except Exception as e:
            print(f"[warn] {sid} failed: {e}")
            skipped.append(str(sid))

    if skipped:
        pd.DataFrame({id_col: skipped}).to_csv(out_dir / "overlay_skipped_slides.csv", index=False)
    print(f"[done] overlay dir: {out_dir}")


if __name__ == "__main__":
    main()

