#!/usr/bin/env python
"""Verify that the valis-wsi environment can load WSI dependencies."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jvm",
        action="store_true",
        help="also initialize the Bio-Formats JVM; first run may download Java artifacts",
    )
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]}")
    for package in (
        "valis-wsi",
        "pyvips",
        "openslide-python",
        "torch",
        "torchvision",
        "scyjava",
    ):
        print(f"{package} {pkg_version(package)}")

    import openslide
    import pyvips
    import torch

    libvips = ".".join(str(pyvips.version(i)) for i in range(3))
    print(f"libvips {libvips}")
    print(f"openslideload {bool(pyvips.type_find('VipsOperation', 'openslideload'))}")
    print(f"openslide-library {getattr(openslide, '__library_version__', 'unknown')}")
    print(f"torch-cuda-available {torch.cuda.is_available()}")

    from valis import slide_io

    print(f"valis-openslide-svs {'.svs' in slide_io.ALL_OPENSLIDE_READABLE_FORMATS}")
    print(f"valis-openslide-mrxs {'.mrxs' in slide_io.ALL_OPENSLIDE_READABLE_FORMATS}")

    if args.jvm:
        slide_io.init_jvm(mem_gb=2)
        try:
            print(f"bioformats {slide_io.get_bioformats_version()}")
        finally:
            slide_io.kill_jvm()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
