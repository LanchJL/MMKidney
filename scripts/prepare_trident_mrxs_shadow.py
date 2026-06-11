#!/usr/bin/env python3
"""Create non-destructive MRXS shadow copies that OpenSlide/TRIDENT can read."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SlideIniInfo:
    slide_path: Path
    ini_path: Path
    has_objective: bool
    mpp_x: float | None


@dataclass(frozen=True)
class ShadowResult:
    slide_path: Path
    shadow_slide_path: Path
    status: str
    mpp_x: float | None
    objective: int | None
    message: str = ""


def _read_ini_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _extract_first_float(text: str, key: str) -> float | None:
    prefix = f"{key} ="
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            try:
                return float(stripped.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def inspect_slide_ini(slide_path: Path) -> SlideIniInfo:
    sidecar = slide_path.with_suffix("")
    ini_path = sidecar / "Slidedat.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"Missing MRXS sidecar ini: {ini_path}")

    text = _read_ini_text(ini_path)
    return SlideIniInfo(
        slide_path=slide_path,
        ini_path=ini_path,
        has_objective="OBJECTIVE_MAGNIFICATION" in text,
        mpp_x=_extract_first_float(text, "MICROMETER_PER_PIXEL_X"),
    )


def _patch_objective(text: str, objective: int) -> str:
    if "OBJECTIVE_MAGNIFICATION" in text:
        return text

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("SLIDE_ID ="):
            lines.insert(idx + 1, f"OBJECTIVE_MAGNIFICATION = {objective}")
            return "\n".join(lines) + "\n"

    for idx, line in enumerate(lines):
        if line.strip() == "[GENERAL]":
            lines.insert(idx + 1, f"OBJECTIVE_MAGNIFICATION = {objective}")
            return "\n".join(lines) + "\n"

    return f"[GENERAL]\nOBJECTIVE_MAGNIFICATION = {objective}\n" + text


def build_shadow_slide(
    slide_path: Path,
    out_dir: Path,
    default_objective: int = 20,
    overwrite: bool = True,
) -> ShadowResult:
    slide_path = slide_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    info = inspect_slide_ini(slide_path)

    src_sidecar = slide_path.with_suffix("")
    dst_slide = out_dir / slide_path.name
    dst_sidecar = out_dir / slide_path.stem
    dst_sidecar.mkdir(parents=True, exist_ok=True)

    if dst_slide.exists() or dst_slide.is_symlink():
        if overwrite:
            dst_slide.unlink()
    if not dst_slide.exists():
        shutil.copy2(slide_path, dst_slide)

    src_xml = slide_path.with_suffix(".xml")
    dst_xml = out_dir / src_xml.name
    if src_xml.exists():
        if dst_xml.exists() or dst_xml.is_symlink():
            if overwrite:
                dst_xml.unlink()
        if not dst_xml.exists():
            dst_xml.symlink_to(src_xml)

    for src_child in src_sidecar.iterdir():
        dst_child = dst_sidecar / src_child.name
        if src_child.name == "Slidedat.ini":
            text = _read_ini_text(src_child)
            patched = _patch_objective(text, default_objective)
            dst_child.write_text(patched, encoding="utf-8")
            continue

        if dst_child.exists() or dst_child.is_symlink():
            if overwrite:
                if dst_child.is_dir() and not dst_child.is_symlink():
                    shutil.rmtree(dst_child)
                else:
                    dst_child.unlink()
            else:
                continue

        dst_child.symlink_to(src_child)

    status = "already_ok" if info.has_objective else "patched"
    return ShadowResult(
        slide_path=slide_path,
        shadow_slide_path=dst_slide,
        status=status,
        mpp_x=info.mpp_x,
        objective=default_objective,
    )


def iter_mrxs(src_dir: Path) -> list[Path]:
    return sorted(p for p in src_dir.glob("*.mrxs") if p.is_file())


def write_manifest(results: list[ShadowResult], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slide_path",
                "shadow_slide_path",
                "status",
                "mpp_x",
                "objective",
                "message",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "slide_path": str(r.slide_path),
                    "shadow_slide_path": str(r.shadow_slide_path),
                    "status": r.status,
                    "mpp_x": "" if r.mpp_x is None else r.mpp_x,
                    "objective": "" if r.objective is None else r.objective,
                    "message": r.message,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--default-objective", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    slides = iter_mrxs(args.src_dir)
    if args.limit is not None:
        slides = slides[: args.limit]

    results: list[ShadowResult] = []
    for slide in slides:
        try:
            results.append(
                build_shadow_slide(
                    slide,
                    args.out_dir,
                    default_objective=args.default_objective,
                )
            )
        except Exception as exc:
            results.append(
                ShadowResult(
                    slide_path=slide,
                    shadow_slide_path=args.out_dir / slide.name,
                    status="error",
                    mpp_x=None,
                    objective=args.default_objective,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

    manifest = args.manifest or args.out_dir / "trident_mrxs_shadow_manifest.csv"
    write_manifest(results, manifest)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"slides={len(results)} manifest={manifest}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
