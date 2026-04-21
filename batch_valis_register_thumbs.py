import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SUPPORTED_EXTS = {".mrxs", ".svs", ".ndpi", ".svslide", ".tif", ".tiff"}
DEFAULT_REF_STAINS = ["HE", "H&E", "HEMATOXYLIN_EOSIN"]

DEF_MAX_PROC = 1500
DEF_MAX_NONR = 2200
DEF_CROP = "reference"
BIG_MAX_PROC = 1200
BIG_MAX_NONR = 1400
BIG_CROP = "overlap"


@dataclass
class CaseGroup:
    case_id: str
    slides: List[Path]
    stain_map: Dict[str, List[Path]]
    reference_slide: Path


def normalize_stain(stain: str) -> str:
    s = stain.strip().upper()
    if s in {"H&E", "HE", "H E"}:
        return "HE"
    return s.replace("&", "").replace(" ", "_")


def parse_case_and_stain_from_filename(
    slide_path: Path, case_stain_splitter: str
) -> Tuple[str, str]:
    stem = slide_path.stem
    if case_stain_splitter in stem:
        parts = [x for x in stem.split(case_stain_splitter) if x]
        if len(parts) >= 2:
            case_id = parts[0]
            stain = parts[-1]
            return case_id, normalize_stain(stain)
    match = re.match(r"^([A-Za-z0-9]+)[-_ ]+(.+)$", stem)
    if match:
        return match.group(1), normalize_stain(match.group(2))
    return stem, "UNK"


def discover_slides(data_root: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in data_root.rglob("*") if p.is_file()]
    else:
        files = [p for p in data_root.glob("*") if p.is_file()]
    slides = [p for p in files if p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(slides)


def build_case_groups(
    slides: Sequence[Path],
    case_stain_splitter: str,
    ref_stains: Sequence[str],
    strict_ref: bool,
) -> List[CaseGroup]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    stains_by_case: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))

    for slide in slides:
        case_id, stain = parse_case_and_stain_from_filename(slide, case_stain_splitter)
        grouped[case_id].append(slide)
        stains_by_case[case_id][stain].append(slide)

    norm_ref_stains = [normalize_stain(s) for s in ref_stains]
    case_groups: List[CaseGroup] = []
    for case_id in sorted(grouped):
        stain_map = stains_by_case[case_id]
        ref_slide: Optional[Path] = None
        for ref_stain in norm_ref_stains:
            if ref_stain in stain_map:
                ref_slide = sorted(stain_map[ref_stain])[0]
                break
        if ref_slide is None:
            if strict_ref:
                continue
            ref_slide = sorted(grouped[case_id])[0]
        case_groups.append(
            CaseGroup(
                case_id=case_id,
                slides=sorted(grouped[case_id]),
                stain_map={k: sorted(v) for k, v in stain_map.items()},
                reference_slide=ref_slide,
            )
        )
    return case_groups


def level0_max_side(slide_f: Path) -> int:
    try:
        from valis import slide_io

        reader_cls = slide_io.get_slide_reader(str(slide_f), series=0)
        reader = reader_cls(str(slide_f), series=0)
        sizes = reader.metadata.slide_dimensions
        if not sizes:
            return 0
        w0, h0 = sizes[0]
        return int(max(w0, h0))
    except Exception:
        return 0


def calc_big_case(slides: Iterable[Path], big_side_px: int) -> Tuple[bool, int]:
    max_side = 0
    for slide in slides:
        ms = level0_max_side(slide)
        max_side = max(max_side, ms)
        if ms >= big_side_px:
            return True, max_side
    return max_side >= big_side_px, max_side


def run_case_registration(
    case_group: CaseGroup,
    results_base: Path,
    big_side_px: int,
    run_micro: bool,
) -> None:
    from valis import registration

    case_id = case_group.case_id
    results_dst_dir = results_base / case_id
    registered_slide_dst_dir = results_dst_dir / "registered_slides"
    big_case, max_side = calc_big_case(case_group.slides, big_side_px)
    max_proc = BIG_MAX_PROC if big_case else DEF_MAX_PROC
    max_nonrigid = BIG_MAX_NONR if big_case else DEF_MAX_NONR
    crop_mode = BIG_CROP if big_case else DEF_CROP
    do_micro = run_micro and (not big_case)

    print(f"\n=== {case_id} ===")
    print(f"[INFO] reference: {case_group.reference_slide.name}")
    print(f"[INFO] slides: {len(case_group.slides)}, max level-0 side={max_side}, big_case={big_case}")
    if big_case:
        print(
            f"[INFO] BIG-GUARD -> max_proc={max_proc}, "
            f"max_nonrigid={max_nonrigid}, crop={crop_mode}, register_micro={do_micro}"
        )

    src_dir = Path(os.path.commonpath([str(p.parent) for p in case_group.slides]))
    try:
        registrar = registration.Valis(
            str(src_dir),
            str(results_dst_dir),
            max_processed_image_dim_px=max_proc,
            max_non_rigid_registration_dim_px=max_nonrigid,
            create_masks=True,
            reference_img_f=str(case_group.reference_slide),
            img_list=[str(p) for p in case_group.slides],
            micro_rigid_registrar_cls=None,
            crop_for_rigid_reg=False,
            check_for_reflections=False,
        )
        registrar.register()
        if do_micro:
            registrar.register_micro(
                max_non_rigid_registration_dim_px=max_nonrigid,
                align_to_reference=True,
            )
        registrar.warp_and_save_slides(
            str(registered_slide_dst_dir),
            crop=crop_mode,
            non_rigid=True,
        )
    finally:
        try:
            registration.kill_jvm()
        except Exception:
            pass


def print_case_preview(case_groups: Sequence[CaseGroup], limit_cases: int, limit_slides: int) -> None:
    print(f"[INFO] case groups found: {len(case_groups)}")
    for idx, cg in enumerate(case_groups):
        if idx >= limit_cases:
            print(f"[INFO] ... only showing first {limit_cases} cases")
            break
        stains = sorted(cg.stain_map.keys())
        print(f"\n[{cg.case_id}] slides={len(cg.slides)} stains={stains} ref={cg.reference_slide.name}")
        for s_idx, slide in enumerate(cg.slides):
            if s_idx >= limit_slides:
                print(f"  ... only showing first {limit_slides} slides")
                break
            print(f"  - {slide.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch register multi-stain WSI by case id with VALIS")
    p.add_argument("--data-root", required=True, help="Path containing WSI files")
    p.add_argument("--results-base", required=True, help="Output root directory")
    p.add_argument("--recursive", action="store_true", help="Recursively discover slides under data-root")
    p.add_argument(
        "--case-stain-splitter",
        default="-",
        help="Tokenizer between case id and stain in filename stem, default='-'",
    )
    p.add_argument(
        "--ref-stains",
        nargs="+",
        default=DEFAULT_REF_STAINS,
        help="Reference stain priority list",
    )
    p.add_argument("--strict-ref", action="store_true", help="Skip cases without requested reference stain")
    p.add_argument("--big-side-px", type=int, default=12000, help="Any slide edge >= this is treated as BIG case")
    p.add_argument("--run-micro", action="store_true", help="Enable register_micro on non-BIG cases")
    p.add_argument("--dry-run", action="store_true", help="Only print parsed case groups and filenames")
    p.add_argument("--limit-cases", type=int, default=50, help="Preview max case groups to print")
    p.add_argument("--limit-slides", type=int, default=30, help="Preview max slides per case to print")
    p.add_argument("--skip-done", action="store_true", help="Skip case_id if results-base/case_id already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    results_base = Path(args.results_base).expanduser().resolve()

    if not data_root.is_dir():
        print(f"[ERR] data root not found: {data_root}")
        sys.exit(1)

    slides = discover_slides(data_root, recursive=args.recursive)
    if not slides:
        print(f"[WARN] no supported WSI files under {data_root}")
        return

    case_groups = build_case_groups(
        slides=slides,
        case_stain_splitter=args.case_stain_splitter,
        ref_stains=args.ref_stains,
        strict_ref=args.strict_ref,
    )
    if not case_groups:
        print("[WARN] no valid case groups after parsing filenames")
        return

    if args.skip_done and results_base.is_dir():
        done_cases = {d.name for d in results_base.iterdir() if d.is_dir()}
        case_groups = [cg for cg in case_groups if cg.case_id not in done_cases]
        print(f"[INFO] remaining cases after skip-done: {len(case_groups)}")

    print_case_preview(case_groups, limit_cases=args.limit_cases, limit_slides=args.limit_slides)
    if args.dry_run:
        print("\n[INFO] dry-run enabled, no registration executed.")
        return

    results_base.mkdir(parents=True, exist_ok=True)
    try:
        from valis import registration
    except ModuleNotFoundError:
        print("[ERR] valis is not installed in current environment.")
        print("[INFO] please run this script in VALIS docker image: cdgatenbee/valis-wsi:1.2.0")
        sys.exit(1)

    for cg in case_groups:
        try:
            run_case_registration(
                case_group=cg,
                results_base=results_base,
                big_side_px=args.big_side_px,
                run_micro=args.run_micro,
            )
        except Exception as e:
            print(f"[ERROR] case={cg.case_id} failed: {e}")
            try:
                registration.kill_jvm()
            except Exception:
                pass


if __name__ == "__main__":
    main()




# 例：Docker（VALIS 官方镜像）
# docker run --rm -it \
#   --memory=64g --shm-size=16g \
#   -e PYTHONUNBUFFERED=1 \
#   -v /your/path:/your/path \
#   cdgatenbee/valis-wsi:1.2.0 \
#   python3 /your/path/MMKidney/batch_valis_register_thumbs.py \
#   --data-root /your/path/WSI_DIR \
#   --results-base /your/path/WSI_REG_RESULTS \
#   --recursive --dry-run

# docker run --rm -it --memory=64g --shm-size=16g \
#    -v /your/path:/your/path \
#    cdgatenbee/valis-wsi:1.2.0 \
#    python3 /your/path/MMKidney/batch_valis_register_thumbs.py \
#    --data-root /your/path/WSI_DIR \
#    --results-base /your/path/WSI_REG_RESULTS \
#    --recursive --skip-done
