#!/usr/bin/env python3
"""Create visual QC panels for VALIS registered image pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def _fit_width(img: Image.Image, width: int) -> Image.Image:
    if img.width == width:
        return img.copy()
    height = max(1, round(img.height * width / img.width))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def _checkerboard(a: Image.Image, b: Image.Image, tile: int = 64) -> Image.Image:
    out = Image.new("RGB", a.size)
    for y in range(0, a.height, tile):
        for x in range(0, a.width, tile):
            src = a if ((x // tile) + (y // tile)) % 2 == 0 else b
            box = (x, y, min(x + tile, a.width), min(y + tile, a.height))
            out.paste(src.crop(box), box)
    return out


def _labelled(img: Image.Image, label: str) -> Image.Image:
    bar_h = 34
    canvas = Image.new("RGB", (img.width, img.height + bar_h), "white")
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), label, fill=(0, 0, 0))
    return canvas


def make_qc_panel(
    moving_path: Path,
    fixed_path: Path,
    out_path: Path,
    max_panel_width: int = 900,
    labels: tuple[str, str] = ("moving", "fixed"),
) -> Path:
    moving = Image.open(moving_path).convert("RGB")
    fixed = Image.open(fixed_path).convert("RGB")
    if moving.size != fixed.size:
        fixed = fixed.resize(moving.size, Image.Resampling.LANCZOS)

    panel_w = min(max_panel_width, moving.width)
    moving = _fit_width(moving, panel_w)
    fixed = _fit_width(fixed, panel_w)
    if moving.size != fixed.size:
        fixed = fixed.resize(moving.size, Image.Resampling.LANCZOS)

    overlay = Image.blend(fixed, moving, 0.5)
    checker = _checkerboard(fixed, moving, tile=max(32, panel_w // 12))

    panels = [
        _labelled(moving, labels[0]),
        _labelled(fixed, labels[1]),
        _labelled(overlay, "alpha overlay"),
        _labelled(checker, "checkerboard"),
    ]
    width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-panel-width", type=int, default=900)
    parser.add_argument("--moving-label", default="moving")
    parser.add_argument("--fixed-label", default="fixed")
    args = parser.parse_args()

    out = make_qc_panel(
        args.moving,
        args.fixed,
        args.out,
        max_panel_width=args.max_panel_width,
        labels=(args.moving_label, args.fixed_label),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
