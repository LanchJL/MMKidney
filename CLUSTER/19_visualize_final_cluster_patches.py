import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

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


def _norm_token(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def build_wsi_index(wsi_dir: str) -> Dict[str, str]:
    exts = {".mrxs", ".svs", ".ndpi", ".tiff", ".tif"}
    index: Dict[str, str] = {}
    for root, _, files in os.walk(wsi_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            stem = os.path.splitext(fn)[0]
            full = os.path.join(root, fn)
            keys = {
                stem,
                stem.split("_")[0],
                _norm_token(stem),
                _norm_token(stem.split("_")[0]),
            }
            for k in keys:
                if k and k not in index:
                    index[k] = full
    return index


def find_wsi_path(wsi_index: Dict[str, str], slide_id: str):
    sid = str(slide_id).strip()
    sid_stem = os.path.splitext(sid)[0]
    cands = [
        sid,
        sid_stem,
        sid.split("_")[0],
        sid_stem.split("_")[0],
        _norm_token(sid),
        _norm_token(sid_stem),
        _norm_token(sid.split("_")[0]),
        _norm_token(sid_stem.split("_")[0]),
    ]
    for k in cands:
        if k in wsi_index:
            return wsi_index[k]
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


def select_consistency_core(
    df_cluster: pd.DataFrame,
    id_col: str,
    n_select: int = 9,
    core_quantile: float = 0.30,
    max_per_slide: int = 0,
    use_prototype_anchor: bool = True,
) -> pd.DataFrame:
    # Consistency-first selection for expert review:
    # 1) keep low-distance core region
    # 2) optionally anchor selection by dominant proto_id
    d = df_cluster.copy()
    if "proto_dist" in d.columns:
        d = d.sort_values("proto_dist", ascending=True).reset_index(drop=True)
    if "proto_dist" in d.columns and 0.0 < float(core_quantile) < 1.0 and len(d) > 0:
        thr = float(d["proto_dist"].quantile(core_quantile))
        core = d[d["proto_dist"] <= thr].copy()
        if len(core) < n_select:
            core = d.head(max(n_select, min(len(d), n_select * 4))).copy()
    else:
        core = d.copy()

    selected_rows = []
    used_patch = set()
    per_slide = {}

    def _try_add(r) -> bool:
        sid = str(r[id_col])
        pid = int(r["patch_id"]) if "patch_id" in core.columns else int(len(selected_rows))
        key = (sid, pid)
        if key in used_patch:
            return False
        if max_per_slide > 0 and per_slide.get(sid, 0) >= max_per_slide:
            return False
        selected_rows.append(r)
        used_patch.add(key)
        per_slide[sid] = per_slide.get(sid, 0) + 1
        return True

    if use_prototype_anchor and "proto_id" in core.columns:
        proto_rank = core["proto_id"].value_counts().index.tolist()
        for proto_id in proto_rank:
            one = core[core["proto_id"] == proto_id].head(1)
            if len(one) == 0:
                continue
            _try_add(one.iloc[0])
            if len(selected_rows) >= n_select:
                break

    if len(selected_rows) < n_select:
        for _, r in core.iterrows():
            _try_add(r)
            if len(selected_rows) >= n_select:
                break

    if not selected_rows:
        return core.head(0)
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
    p.add_argument("--n-cols", type=int, default=3, help="Montage columns")
    p.add_argument("--cluster-col", type=str, default="final_cluster", help="Cluster column to group by")
    p.add_argument("--max-per-slide", type=int, default=2)
    p.add_argument(
        "--review-mode",
        choices=["both", "consistency", "diversity"],
        default="both",
        help="both=core+diversity, consistency=core only, diversity=diversity only",
    )
    p.add_argument("--core-quantile", type=float, default=0.30, help="Low-distance core quantile for consistency mode")
    p.add_argument("--consistency-max-per-slide", type=int, default=0, help="0 means no cap")
    p.add_argument(
        "--consistency-no-proto-anchor",
        action="store_true",
        help="Disable prototype-anchored core selection",
    )
    p.add_argument("--top-n-clusters", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi_index = build_wsi_index(args.wsi_dir)
    if len(wsi_index) == 0:
        raise ValueError(f"No WSI files found under: {args.wsi_dir}")
    print(f"[index] wsi keys indexed = {len(wsi_index)}")

    df = load_table_auto(args.patch_final).copy()
    id_col = "slide_id" if "slide_id" in df.columns else ("sample_id" if "sample_id" in df.columns else "")
    if not id_col:
        raise ValueError("patch-final missing slide_id/sample_id")
    cluster_col = str(args.cluster_col)
    for c in [cluster_col, "x", "y"]:
        if c not in df.columns:
            raise ValueError(f"patch-final missing {c}")

    df[cluster_col] = df[cluster_col].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0).astype(int)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)

    cluster_order = df[cluster_col].value_counts()
    if args.top_n_clusters > 0:
        cluster_order = cluster_order.head(args.top_n_clusters)

    selected_all = []
    missing_wsi_rows = []
    per_cluster_stats = []
    for i, cluster_name in enumerate(cluster_order.index.tolist(), 1):
        dc = df[df[cluster_col] == cluster_name].copy()
        mode_to_sel = {}
        if args.review_mode in ("both", "consistency"):
            mode_to_sel["core"] = select_consistency_core(
                dc,
                id_col=id_col,
                n_select=args.n_per_cluster,
                core_quantile=args.core_quantile,
                max_per_slide=args.consistency_max_per_slide,
                use_prototype_anchor=(not args.consistency_no_proto_anchor),
            )
        if args.review_mode in ("both", "diversity"):
            mode_to_sel["diversity"] = select_representatives(
                dc,
                id_col=id_col,
                n_select=args.n_per_cluster,
                max_per_slide=args.max_per_slide,
            )

        cluster_stat = {"cluster_col": cluster_col, "cluster": cluster_name}
        has_any = False
        for mode_name, sel in mode_to_sel.items():
            if len(sel) == 0:
                cluster_stat[f"n_{mode_name}_selected"] = 0
                cluster_stat[f"n_{mode_name}_rendered"] = 0
                continue

            tiles: List[Image.Image] = []
            labels: List[str] = []
            out_rows = []

            for _, r in sel.iterrows():
                sid = str(r[id_col])
                x, y = int(r["x"]), int(r["y"])
                wsi_path = find_wsi_path(wsi_index, sid)
                if not wsi_path:
                    missing_wsi_rows.append(
                        {"mode": mode_name, "cluster_col": cluster_col, "cluster": cluster_name, id_col: sid, "x": x, "y": y}
                    )
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
                        "mode": mode_name,
                        "cluster_col": cluster_col,
                        "cluster": cluster_name,
                        id_col: sid,
                        "x": x,
                        "y": y,
                        "wsi_path": wsi_path,
                    }
                )

            cluster_stat[f"n_{mode_name}_selected"] = int(len(sel))
            cluster_stat[f"n_{mode_name}_rendered"] = int(len(out_rows))
            if len(tiles) == 0:
                continue

            has_any = True
            montage = make_montage(tiles, labels, n_cols=max(1, int(args.n_cols)), tile_size=256, pad=8)
            out_png = out_dir / f"cluster_{cluster_name}_{mode_name}_top{len(tiles)}.png"
            montage.save(out_png)
            print(f"[saved] {out_png}")

            out_csv = out_dir / f"cluster_{cluster_name}_{mode_name}_selected.csv"
            pd.DataFrame(out_rows).to_csv(out_csv, index=False)
            selected_all.extend(out_rows)

        if not has_any:
            # keep row for visibility even when nothing rendered
            if "n_core_selected" not in cluster_stat:
                cluster_stat["n_core_selected"] = 0
                cluster_stat["n_core_rendered"] = 0
            if "n_diversity_selected" not in cluster_stat:
                cluster_stat["n_diversity_selected"] = 0
                cluster_stat["n_diversity_rendered"] = 0
        per_cluster_stats.append(cluster_stat)

        if i % 10 == 0 or i == len(cluster_order):
            print(f"[progress] {i}/{len(cluster_order)} clusters")

    if selected_all:
        all_csv = out_dir / "all_selected_representative_patches.csv"
        pd.DataFrame(selected_all).to_csv(all_csv, index=False)
        print(f"[saved] {all_csv}")
    if missing_wsi_rows:
        miss_csv = out_dir / "missing_wsi_for_selected.csv"
        pd.DataFrame(missing_wsi_rows).to_csv(miss_csv, index=False)
        print(f"[saved] {miss_csv}")
    if per_cluster_stats:
        st_csv = out_dir / "cluster_render_stats.csv"
        pd.DataFrame(per_cluster_stats).to_csv(st_csv, index=False)
        print(f"[saved] {st_csv}")
    print("[done] representative patch visualization completed")


if __name__ == "__main__":
    main()
