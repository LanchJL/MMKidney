#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import openslide
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape


CLASS_COLORS = {
    "Cortex": (0, 190, 0),
    "Medulla": (220, 0, 0),
    "Other": (190, 190, 190),
    "Capsule": (0, 190, 190),
    "IFTACortex": (120, 120, 120),
    "ProximalTubule": (255, 90, 160),
    "DistalTubule": (0, 120, 220),
    "Glomerulus": (30, 80, 255),
    "Artery": (255, 0, 255),
    "Arteriole": (255, 230, 0),
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def geometry_polygons(geom):
    g = shape(geom)
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if g.geom_type == "MultiPolygon":
        return list(g.geoms)
    if hasattr(g, "geoms"):
        return [x for x in g.geoms if x.geom_type == "Polygon"]
    return []


def class_name(feature):
    props = feature.get("properties") or {}
    value = props.get("classification") or props.get("pathClass") or props.get("class") or props.get("name") or "Unknown"
    if isinstance(value, dict):
        value = value.get("name", "Unknown")
    return str(value)


def fit_region(meta, slide_w, slide_h, margin=512):
    if meta.get("tissue_bbox_l0"):
        x0, y0, x1, y1 = [int(v) for v in meta["tissue_bbox_l0"]]
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(slide_w, x1 + margin)
        y1 = min(slide_h, y1 + margin)
        return x0, y0, x1, y1
    return 0, 0, slide_w, slide_h


def draw_legend(img, counts):
    draw = ImageDraw.Draw(img, "RGBA")
    x, y = 18, 18
    line_h = 24
    width = 330
    height = 38 + line_h * len(counts)
    draw.rectangle([x - 10, y - 10, x + width, y + height], fill=(255, 255, 255, 210), outline=(0, 0, 0, 160))
    draw.text((x, y), "QuPath GeoJSON overlay", fill=(0, 0, 0, 255))
    y += 30
    for name, count in sorted(counts.items(), key=lambda kv: kv[0]):
        color = CLASS_COLORS.get(name, (255, 128, 0))
        draw.rectangle([x, y + 4, x + 16, y + 20], fill=color + (180,), outline=(0, 0, 0, 160))
        draw.text((x + 24, y), f"{name}: {count}", fill=(0, 0, 0, 255))
        y += line_h


def main():
    parser = argparse.ArgumentParser("Render QuPath/OpenSlide-bounds GeoJSON annotations on an MRXS thumbnail.")
    parser.add_argument("--slide", required=True)
    parser.add_argument("--geojson", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-size", type=int, default=3000)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--fill-alpha", type=int, default=42)
    parser.add_argument("--outline-alpha", type=int, default=210)
    args = parser.parse_args()

    slide_path = Path(args.slide)
    geojson_path = Path(args.geojson)
    meta_path = Path(args.meta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = load_json(meta_path)
    geo = load_json(geojson_path)
    slide = openslide.OpenSlide(str(slide_path))
    slide_w, slide_h = slide.dimensions
    bx, by = [int(v) for v in meta.get("openslide_bounds_l0", [0, 0, slide_w, slide_h])[:2]]

    x0, y0, x1, y1 = fit_region(meta, slide_w, slide_h)
    region_w = max(1, x1 - x0)
    region_h = max(1, y1 - y0)
    scale = min(float(args.max_size) / region_w, float(args.max_size) / region_h, 1.0)
    out_w = max(1, int(round(region_w * scale)))
    out_h = max(1, int(round(region_h * scale)))

    thumb = slide.read_region((x0, y0), 0, (region_w, region_h)).convert("RGB")
    thumb = thumb.resize((out_w, out_h), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    counts = {}
    for feature in geo.get("features", []):
        name = class_name(feature)
        counts[name] = counts.get(name, 0) + 1
        color = CLASS_COLORS.get(name, (255, 128, 0))
        for poly in geometry_polygons(feature.get("geometry")):
            exterior = [((x + bx - x0) * scale, (y + by - y0) * scale) for x, y in poly.exterior.coords]
            if len(exterior) >= 3:
                draw.polygon(exterior, fill=color + (args.fill_alpha,))
                draw.line(exterior, fill=color + (args.outline_alpha,), width=args.line_width, joint="curve")
            for interior in poly.interiors:
                hole = [((x + bx - x0) * scale, (y + by - y0) * scale) for x, y in interior.coords]
                if len(hole) >= 3:
                    draw.polygon(hole, fill=(0, 0, 0, 0))

    rendered = Image.alpha_composite(thumb.convert("RGBA"), overlay).convert("RGB")
    draw_legend(rendered, counts)
    rendered.save(out_path, quality=92)

    print(json.dumps({
        "out": str(out_path),
        "slide_dimensions": [slide_w, slide_h],
        "render_region_l0": [x0, y0, x1, y1],
        "openslide_bounds_l0": meta.get("openslide_bounds_l0"),
        "scale": scale,
        "output_size": [out_w, out_h],
        "feature_count": len(geo.get("features", [])),
        "class_counts": counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
