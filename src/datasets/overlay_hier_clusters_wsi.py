import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import openslide
except Exception as e:
    openslide = None


def _load_table_auto(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".pkl":
        return pd.read_pickle(p)
    raise ValueError(f"Unsupported table format: {path}")


def _norm_id(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\.(mrxs|svs|ndpi|tiff|tif)$", "", s)
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def _build_wsi_index(wsi_dir: str) -> Tuple[Dict[str, str], List[str]]:
    exts = [".mrxs", ".svs", ".ndpi", ".tiff", ".tif"]
    norm2path: Dict[str, str] = {}
    paths: List[str] = []
    for ext in exts:
        for fp in Path(wsi_dir).glob(f"*{ext}"):
            p = str(fp)
            paths.append(p)
            norm2path[_norm_id(fp.stem)] = p
    return norm2path, paths


def _find_wsi_path(sample_id: str, norm2path: Dict[str, str], all_paths: List[str]) -> Optional[str]:
    sid = str(sample_id)
    cand_ids = []
    sid_norm = _norm_id(sid)
    cand_ids.append(sid_norm)
    if "_" in sid_norm:
        cand_ids.append(sid_norm.split("_")[0])
    if "-" in sid_norm:
        cand_ids.append(sid_norm.split("-")[0])

    for c in cand_ids:
        if c in norm2path:
            return norm2path[c]

    # Fuzzy: filename startswith/contains sample id token.
    for c in cand_ids:
        if not c:
            continue
        for p in all_paths:
            stem = _norm_id(Path(p).stem)
            if stem.startswith(c) or c in stem:
                return p
    return None


def _resolve_default_wsi_dir() -> Optional[str]:
    """
    Try to reuse the same WSI directory used by CLUSTER scripts.
    Priority:
    1) env: MMKIDNEY_WSI_DIR
    2) parse CLUSTER/*.py for WSI_DIR = "..."
    """
    env_path = os.environ.get("MMKIDNEY_WSI_DIR", "").strip()
    if env_path and os.path.isdir(env_path):
        return env_path

    cluster_dir = Path("CLUSTER")
    if not cluster_dir.exists():
        return None

    candidates = sorted(cluster_dir.glob("*.py"))
    pat = re.compile(r'^\s*WSI_DIR\s*=\s*["\']([^"\']+)["\']')
    for fp in candidates:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            m = pat.search(line)
            if not m:
                continue
            wsi_dir = m.group(1).strip()
            if os.path.isdir(wsi_dir):
                return wsi_dir
    return None


def _build_color_map(cluster_labels: List[str]) -> Dict[str, tuple]:
    labs = [str(x) for x in cluster_labels]
    n = len(labs)
    cmap = plt.cm.turbo if n > 20 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(1, n)))
    out = {}
    for i, lab in enumerate(labs):
        rgb = mcolors.to_rgb(colors[i])
        out[lab] = tuple(int(v * 255) for v in rgb)
    return out


def _estimate_patch_step(xs: np.ndarray) -> int:
    xs = np.sort(np.unique(xs))
    if len(xs) > 1:
        d = np.diff(xs)
        step = int(np.min(d))
        if step < 10:
            step = 256
    else:
        step = 256
    return step


def _get_font(size=18):
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


def _add_legend(image_rgba, color_map, cluster_counts, title="Clusters", max_items=20):
    draw = ImageDraw.Draw(image_rgba)
    font_title = _get_font(24)
    font_text = _get_font(18)

    items = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:max_items]
    if not items:
        return image_rgba

    pad = 12
    sw = 18
    line_h = 24
    legend_w = 320
    legend_h = pad * 2 + 30 + len(items) * line_h

    x0 = image_rgba.width - legend_w - 20
    y0 = 20
    x1 = image_rgba.width - 20
    y1 = y0 + legend_h

    draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=(255, 255, 255, 210), outline=(40, 40, 40, 255), width=2)
    draw.text((x0 + pad, y0 + 8), title, fill=(0, 0, 0, 255), font=font_title)

    yy = y0 + 40
    for lab, cnt in items:
        col = color_map.get(str(lab), (128, 128, 128))
        draw.rectangle([x0 + pad, yy + 2, x0 + pad + sw, yy + 2 + sw], fill=col + (255,), outline=(0, 0, 0, 255))
        draw.text((x0 + pad + sw + 10, yy), f"{lab} ({cnt})", fill=(0, 0, 0, 255), font=font_text)
        yy += line_h

    return image_rgba


def _overlay_one_slide(df_slide, wsi_path, save_path, color_map, downsample=16, alpha=0.4, add_legend=True, filter_clusters=None):
    if openslide is None:
        raise RuntimeError("openslide not available")

    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.dimensions
    tw, th = max(1, w // downsample), max(1, h // downsample)
    bg = slide.get_thumbnail((tw, th)).convert("RGBA")

    wr, hr = bg.size
    sx, sy = wr / w, hr / h

    xs = df_slide["x"].to_numpy(dtype=np.int32)
    ys = df_slide["y"].to_numpy(dtype=np.int32)
    cl = df_slide["final_cluster"].astype(str).to_numpy()

    step = _estimate_patch_step(xs)
    bw = int(step * sx + 1)
    bh = int(step * sy + 1)

    mask = Image.new("RGBA", (wr, hr), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)

    filt = set(str(x) for x in (filter_clusters or []))

    for x, y, lab in zip(xs, ys, cl):
        if lab in filt:
            continue
        if lab not in color_map:
            continue
        x0 = int(x * sx)
        y0 = int(y * sy)
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=color_map[lab] + (255,))

    _, _, _, a = mask.split()
    mask.putalpha(a.point(lambda p: int(255 * alpha) if p > 0 else 0))
    out = Image.alpha_composite(bg, mask)

    if add_legend:
        counts = pd.Series(cl).value_counts().to_dict()
        if filt:
            counts = {k: v for k, v in counts.items() if str(k) not in filt}
        out = _add_legend(out, color_map, counts, title="Final clusters")

    out.save(save_path)
    slide.close()
    del bg, mask, out
    gc.collect()


def main():
    p = argparse.ArgumentParser("Overlay hierarchical final clusters back to WSI")
    p.add_argument("--patch-final", required=True, help="patch_final_clusters.(parquet|csv|pkl)")
    p.add_argument("--wsi-dir", default="", help="Optional. Auto-resolve from CLUSTER if omitted.")
    p.add_argument("--output-dir", default="data/processed/hier_cluster_overlay")
    p.add_argument("--downsample", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--top-n-slides", type=int, default=0)
    p.add_argument("--top-n-clusters", type=int, default=0)
    p.add_argument("--filter-clusters", default="", help="comma-separated labels to hide in filtered output")
    p.add_argument("--legend", action="store_true")
    p.add_argument("--strict-match", action="store_true", help="If set, disable fuzzy id matching.")
    args = p.parse_args()

    wsi_dir = args.wsi_dir.strip() if isinstance(args.wsi_dir, str) else ""
    if not wsi_dir:
        wsi_dir = _resolve_default_wsi_dir() or ""
    if not wsi_dir:
        raise ValueError(
            "WSI dir not provided and auto-resolve failed. "
            "Please pass --wsi-dir or set MMKIDNEY_WSI_DIR."
        )
    if not os.path.isdir(wsi_dir):
        raise ValueError(f"WSI dir does not exist: {wsi_dir}")

    norm2path, all_wsi_paths = _build_wsi_index(wsi_dir)
    if not all_wsi_paths:
        raise ValueError(
            f"No WSI files found in: {wsi_dir}. "
            "Expected one of [.mrxs, .svs, .ndpi, .tiff, .tif]."
        )

    out_root = Path(args.output_dir)
    out_all = out_root / "all_clusters"
    out_filtered = out_root / "filtered_clusters"
    out_all.mkdir(parents=True, exist_ok=True)
    out_filtered.mkdir(parents=True, exist_ok=True)

    df = _load_table_auto(args.patch_final).copy()

    # support both sample_id and slide_id
    if "sample_id" in df.columns:
        id_col = "sample_id"
    elif "slide_id" in df.columns:
        id_col = "slide_id"
    else:
        raise ValueError("patch table must contain sample_id or slide_id")

    for c in ["x", "y", "final_cluster"]:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    df["final_cluster"] = df["final_cluster"].astype(str)

    cluster_counts = df["final_cluster"].value_counts()
    if args.top_n_clusters > 0:
        keep = cluster_counts.head(args.top_n_clusters).index.tolist()
        df = df[df["final_cluster"].isin(keep)].copy()
        cluster_counts = df["final_cluster"].value_counts()

    color_map = _build_color_map(cluster_counts.index.tolist())

    slide_counts = df[id_col].value_counts()
    if args.top_n_slides > 0:
        sample_ids = slide_counts.head(args.top_n_slides).index.tolist()
    else:
        sample_ids = slide_counts.index.tolist()

    filter_clusters = [x.strip() for x in args.filter_clusters.split(",") if x.strip()]

    # preflight match check
    pre_missing = []
    for sid in sample_ids:
        if args.strict_match:
            k = _norm_id(sid)
            p = norm2path.get(k, None)
        else:
            p = _find_wsi_path(sid, norm2path, all_wsi_paths)
        if p is None:
            pre_missing.append(str(sid))
    matched_n = len(sample_ids) - len(pre_missing)
    print(
        f"[precheck] slides={len(sample_ids)}, matched_wsi={matched_n}, "
        f"missing={len(pre_missing)}, wsi_dir={wsi_dir}"
    )
    if pre_missing:
        pd.DataFrame({"sample_id": pre_missing}).to_csv(out_root / "precheck_missing_wsi.csv", index=False)
    if matched_n == 0:
        raise RuntimeError(
            "Precheck found 0 matched WSI files. "
            "Please verify --wsi-dir or ID naming rules. "
            "See output precheck_missing_wsi.csv."
        )

    skipped = []
    for i, sid in enumerate(sample_ids, 1):
        if args.strict_match:
            wsi_path = norm2path.get(_norm_id(sid), None)
        else:
            wsi_path = _find_wsi_path(sid, norm2path, all_wsi_paths)
        if wsi_path is None:
            skipped.append(str(sid))
            continue

        d = df[df[id_col] == sid].copy()
        save_all = out_all / f"{sid}_overlay_all.png"
        save_filtered = out_filtered / f"{sid}_overlay_filtered.png"

        try:
            _overlay_one_slide(
                d,
                wsi_path,
                str(save_all),
                color_map,
                downsample=args.downsample,
                alpha=args.alpha,
                add_legend=args.legend,
                filter_clusters=[],
            )
            _overlay_one_slide(
                d,
                wsi_path,
                str(save_filtered),
                color_map,
                downsample=args.downsample,
                alpha=args.alpha,
                add_legend=args.legend,
                filter_clusters=filter_clusters,
            )
        except Exception as e:
            print(f"[warn] {sid} failed: {e}")
            skipped.append(str(sid))

        if i % 20 == 0 or i == len(sample_ids):
            print(f"[overlay] {i}/{len(sample_ids)}")

    color_df = pd.DataFrame(
        {
            "final_cluster": list(color_map.keys()),
            "r": [color_map[k][0] for k in color_map],
            "g": [color_map[k][1] for k in color_map],
            "b": [color_map[k][2] for k in color_map],
            "n_patches": [int(cluster_counts.get(k, 0)) for k in color_map],
            "filtered_out": [str(k) in set(filter_clusters) for k in color_map],
        }
    )
    color_df.to_csv(out_root / "final_cluster_colors.csv", index=False)

    if skipped:
        pd.DataFrame({"sample_id": skipped}).to_csv(out_root / "overlay_skipped_slides.csv", index=False)

    meta = {
        "patch_final": args.patch_final,
        "wsi_dir": wsi_dir,
        "n_slides": len(sample_ids),
        "n_clusters": len(color_map),
        "top_n_slides": args.top_n_slides,
        "top_n_clusters": args.top_n_clusters,
        "filter_clusters": filter_clusters,
        "downsample": args.downsample,
        "alpha": args.alpha,
    }
    with (out_root / "overlay_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] overlay saved to: {out_root}")


if __name__ == "__main__":
    main()
