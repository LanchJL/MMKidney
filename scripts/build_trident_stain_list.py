#!/usr/bin/env python3
"""Build a TRIDENT custom WSI list by stain, optionally restricted to ROI slides."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def stem_has_stain(stem: str, stain: str) -> bool:
    stain = stain.upper()
    parts = stem.upper().split("-")
    return stain in parts


def build_list(wsi_dir: Path, stain: str, roi_only: bool) -> list[str]:
    xml_stems = {p.stem for p in wsi_dir.glob("*.xml")}
    slides = []
    for slide in sorted(wsi_dir.glob("*.mrxs")):
        if not stem_has_stain(slide.stem, stain):
            continue
        if roi_only and slide.stem not in xml_stems:
            continue
        slides.append(slide.name)
    return slides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument("--stain", required=True, help="Example: HE, MASSON, PAS, C4d, SV40")
    parser.add_argument("--roi-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    slides = build_list(args.wsi_dir, args.stain, args.roi_only)
    if args.limit is not None:
        slides = slides[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi"])
        writer.writerows([[name] for name in slides])

    print(f"wrote={args.out}")
    print(f"stain={args.stain} roi_only={args.roi_only} slides={len(slides)}")
    for name in slides[:10]:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
