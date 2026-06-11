#!/usr/bin/env python3
"""Run a small VALIS registration pilot using ROI-cropped WSI thumbnails."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import openslide
from PIL import Image


def read_roi(xml_path: Path) -> dict[str, int] | None:
    if not xml_path.exists():
        return None
    roi = ET.parse(xml_path).find(".//region_of_interest")
    if roi is None:
        return None
    return {
        "offset_left": int(float(roi.attrib["offset_left"])),
        "offset_top": int(float(roi.attrib["offset_top"])),
        "width": int(float(roi.attrib["width"])),
        "height": int(float(roi.attrib["height"])),
    }


def export_roi_thumbnail(slide_path: Path, out_path: Path, max_dim: int) -> dict[str, object]:
    xml_path = slide_path.with_suffix(".xml")
    roi = read_roi(xml_path)
    slide = openslide.OpenSlide(str(slide_path))
    slide_w, slide_h = slide.dimensions

    if roi and (roi["width"], roi["height"]) != (slide_w, slide_h):
        x, y, w, h = roi["offset_left"], roi["offset_top"], roi["width"], roi["height"]
    else:
        x, y, w, h = 0, 0, slide_w, slide_h

    scale = min(1.0, max_dim / max(w, h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    downsample = 1.0 / scale
    level = slide.get_best_level_for_downsample(downsample)
    level_downsample = float(slide.level_downsamples[level])
    level_size = (
        max(1, int(math.ceil(w / level_downsample))),
        max(1, int(math.ceil(h / level_downsample))),
    )
    img = slide.read_region((x, y), level, level_size).convert("RGB")
    if img.size != (out_w, out_h):
        img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return {
        "slide_path": str(slide_path),
        "xml_path": str(xml_path) if xml_path.exists() else None,
        "slide_dimensions": [slide_w, slide_h],
        "roi": roi,
        "crop_l0": [x, y, w, h],
        "thumbnail_size": [out_w, out_h],
        "thumbnail_scale_from_l0": scale,
        "openslide_level": level,
        "openslide_level_downsample": level_downsample,
    }


def run_valis(src_dir: Path, dst_dir: Path, reference_name: str) -> None:
    from valis import registration

    registrar = registration.Valis(
        str(src_dir),
        str(dst_dir),
        reference_img_f=reference_name,
        align_to_reference=True,
        max_image_dim_px=2048,
        max_processed_image_dim_px=1024,
        max_non_rigid_registration_dim_px=2048,
        thumbnail_size=1024,
    )
    registrar.register()
    registration.kill_jvm()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument("--case-id", default="16S02292")
    parser.add_argument("--reference", default="HE-A1")
    parser.add_argument("--moving", default="CD3-A1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-dim", type=int, default=4096)
    args = parser.parse_args()

    ref_slide = args.wsi_dir / f"{args.case_id}-{args.reference}.mrxs"
    moving_slide = args.wsi_dir / f"{args.case_id}-{args.moving}.mrxs"
    if not ref_slide.exists():
        raise FileNotFoundError(ref_slide)
    if not moving_slide.exists():
        raise FileNotFoundError(moving_slide)

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    img_dir = args.out_dir / "roi_images"
    valis_dir = args.out_dir / "valis"
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / "mplconfig"))

    ref_img = img_dir / f"{ref_slide.stem}.png"
    moving_img = img_dir / f"{moving_slide.stem}.png"
    metadata = {
        ref_slide.stem: export_roi_thumbnail(ref_slide, ref_img, args.max_dim),
        moving_slide.stem: export_roi_thumbnail(moving_slide, moving_img, args.max_dim),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "roi_thumbnail_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_valis(img_dir, valis_dir, ref_img.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
