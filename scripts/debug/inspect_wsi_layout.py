#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

EXTS = {".mrxs", ".svs", ".ndpi", ".svslide", ".tif", ".tiff"}


def parse_case_stain(name: str):
    stem = Path(name).stem
    m = re.match(r"^([A-Za-z0-9]+)[-_ ]+(.+)$", stem)
    if m:
        return m.group(1), m.group(2)
    return stem, "UNK"


def main():
    ap = argparse.ArgumentParser(description="Inspect WSI directory layout and naming")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--max-cases", type=int, default=50)
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"[ERR] path not exists: {root}")
        return
    if not root.is_dir():
        print(f"[ERR] not a directory: {root}")
        return

    print(f"[INFO] data_root={root}")
    print(f"[INFO] recursive={args.recursive}")

    files = list(root.rglob("*")) if args.recursive else list(root.glob("*"))
    files = [p for p in files if p.is_file()]
    dirs = list(root.rglob("*")) if args.recursive else list(root.glob("*"))
    dirs = [p for p in dirs if p.is_dir()]

    print(f"[INFO] dirs_found={len(dirs)}")
    print(f"[INFO] files_found={len(files)}")

    ext_counter = Counter(p.suffix.lower() for p in files)
    print("\n[INFO] top file extensions:")
    for ext, n in ext_counter.most_common(20):
        print(f"  {ext or '<no_ext>'}: {n}")

    wsi_files = [p for p in files if p.suffix.lower() in EXTS]
    print(f"\n[INFO] wsi_files_found={len(wsi_files)}")
    for p in wsi_files[: args.max_files]:
        print(f"  WSI: {p}")
    if len(wsi_files) > args.max_files:
        print(f"  ... only first {args.max_files} shown")

    case_to_files = defaultdict(list)
    case_to_stains = defaultdict(set)
    for p in wsi_files:
        case_id, stain = parse_case_stain(p.name)
        case_to_files[case_id].append(p)
        case_to_stains[case_id].add(stain)

    print(f"\n[INFO] parsed_cases={len(case_to_files)}")
    sorted_cases = sorted(case_to_files.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for i, (case_id, items) in enumerate(sorted_cases):
        if i >= args.max_cases:
            print(f"  ... only first {args.max_cases} cases shown")
            break
        stains = sorted(case_to_stains[case_id])
        print(f"  CASE {case_id}: slides={len(items)}, stains={stains}")
        for fp in sorted(items)[:10]:
            print(f"    - {fp.name}")
        if len(items) > 10:
            print("    ...")


if __name__ == "__main__":
    main()
