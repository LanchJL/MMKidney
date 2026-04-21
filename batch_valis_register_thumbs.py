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
DEFAULT_STAIN_ALIASES = [
    (r"CK\W*PAN|PAN\W*CK|CK\{PAN\}|CK_PAN", "CK_PAN"),
    (r"H\s*&\s*E|HEMATOXYLIN[_\s-]*EOSIN|\bHE\b", "HE"),
    (r"GRANZYME\W*B|\bGRANB\b", "GRANB"),
    (r"\bPERFORIN\b", "PERFORIN"),
    (r"\bMASSON\b", "MASSON"),
    (r"\bPASM\b", "PASM"),
    (r"\bPAS\b", "PAS"),
    (r"\bC4D\b", "C4D"),
    (r"\bCMV\b", "CMV"),
    (r"\bEBER\b", "EBER"),
    (r"\bSV40\b", "SV40"),
    (r"\bFOX[P]?[3]\b", "FOXP3"),
    (r"\bKI\W*67\b", "KI67"),
    (r"\bTLXW\b", "TLXW"),
    (r"\bHB[C]AG\b", "HBCAG"),
    (r"\bHB[S]AG\b", "HBSAG"),
    (r"\bBM\b", "BM"),
    (r"\bRS\b", "RS"),
    (r"\bTIA\b", "TIA"),
    (r"\bCD20\b", "CD20"),
    (r"\bCD79\b", "CD79"),
    (r"\bCD68\b", "CD68"),
    (r"\bCD8\b", "CD8"),
    (r"\bCD4\b", "CD4"),
    (r"\bCD3\b", "CD3"),
    (r"\bCK\b", "CK"),
]

DEF_MAX_PROC = 1500
DEF_MAX_NONR = 2200
DEF_CROP = "reference"
BIG_MAX_PROC = 1200
BIG_MAX_NONR = 1400
BIG_CROP = "overlap"
HUGE_SIDE_PX = 30000
HUGE_MAX_PROC = 1000
HUGE_MAX_NONR = 1024
FALLBACK1_MAX_PROC = 1000
FALLBACK1_MAX_NONR = 1024
FALLBACK2_MAX_PROC = 800
FALLBACK2_MAX_NONR = 1024


@dataclass
class CaseGroup:
    case_id: str
    slides: List[Path]
    stain_map: Dict[str, List[Path]]
    reference_slide: Path


@dataclass
class SkippedCase:
    case_id: str
    reason: str
    details: str


@dataclass
class RegAttempt:
    case_id: str
    attempt: int
    strategy: str
    non_rigid: bool
    max_proc: int
    max_nonrigid: int
    n_slides: int
    status: str
    error: str


def normalize_stain(stain: str) -> str:
    raw = stain.strip().upper().replace("Α", "A")
    version_match = re.search(r"[_\s-]V(\d+)$", raw)
    version = f"_V{version_match.group(1)}" if version_match else ""
    cleaned = re.sub(r"[^A-Z0-9]+", " ", raw).strip()
    for pat, canon in DEFAULT_STAIN_ALIASES:
        if re.search(pat, cleaned):
            return f"{canon}{version}"
    return re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")


def parse_case_and_stain_from_filename(
    slide_path: Path, case_stain_splitter: str
) -> Tuple[str, str]:
    stem = slide_path.stem
    if case_stain_splitter in stem:
        parts = stem.split(case_stain_splitter, 1)
        if len(parts) == 2:
            case_id = parts[0].strip()
            stain = parts[1].strip()
            return case_id, normalize_stain(stain)
    match = re.match(r"^([A-Za-z0-9]+)[-_ ]+(.+)$", stem)
    if match:
        return match.group(1), normalize_stain(match.group(2))
    return stem, "UNK"


def stain_base(stain: str) -> str:
    return re.sub(r"_V\d+$", "", stain)


def choose_best_slide_for_stain(slides: Sequence[Path], base_stain: str) -> Path:
    def score(p: Path) -> Tuple[int, int, str]:
        stem = p.stem.upper()
        has_exact = int(stem.endswith(f"-{base_stain}") or stem.endswith(f"_{base_stain}") or stem == base_stain)
        return (-has_exact, len(p.name), p.name)

    return sorted(slides, key=score)[0]


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
    dedup_stain: bool,
) -> Tuple[List[CaseGroup], List[SkippedCase]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    stains_by_case: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))

    for slide in slides:
        case_id, stain = parse_case_and_stain_from_filename(slide, case_stain_splitter)
        grouped[case_id].append(slide)
        stains_by_case[case_id][stain].append(slide)

    case_groups: List[CaseGroup] = []
    skipped_cases: List[SkippedCase] = []
    for case_id in sorted(grouped):
        raw_stain_map = stains_by_case[case_id]
        selected_slides = sorted(grouped[case_id])
        selected_stain_map: Dict[str, List[Path]] = {k: sorted(v) for k, v in raw_stain_map.items()}

        he_candidates: List[Path] = []
        for st, items in selected_stain_map.items():
            if stain_base(st) == "HE":
                he_candidates.extend(items)
        he_candidates = sorted(he_candidates)

        if len(he_candidates) == 0:
            skipped_cases.append(
                SkippedCase(
                    case_id=case_id,
                    reason="MISSING_HE",
                    details="No HE slide found in this case",
                )
            )
            continue
        if len(he_candidates) > 1:
            skipped_cases.append(
                SkippedCase(
                    case_id=case_id,
                    reason="MULTIPLE_HE",
                    details=";".join(x.name for x in he_candidates),
                )
            )
            continue
        ref_slide = he_candidates[0]

        if dedup_stain:
            base_to_candidates: Dict[str, List[Path]] = defaultdict(list)
            for st, items in selected_stain_map.items():
                base_to_candidates[stain_base(st)].extend(items)
            dedup_picks: List[Path] = []
            dedup_map: Dict[str, List[Path]] = {}
            for base, candidates in sorted(base_to_candidates.items()):
                if base == "HE":
                    picked = ref_slide
                else:
                    picked = choose_best_slide_for_stain(candidates, base)
                dedup_picks.append(picked)
                dedup_map[base] = [picked]
            selected_slides = sorted(dedup_picks)
            selected_stain_map = dedup_map
        case_groups.append(
            CaseGroup(
                case_id=case_id,
                slides=selected_slides,
                stain_map=selected_stain_map,
                reference_slide=ref_slide,
            )
        )
    return case_groups, skipped_cases


def write_skip_report(skipped_cases: Sequence[SkippedCase], skip_log: Path) -> None:
    skip_log.parent.mkdir(parents=True, exist_ok=True)
    with skip_log.open("w", encoding="utf-8") as f:
        f.write("case_id\treason\tdetails\n")
        for s in skipped_cases:
            f.write(f"{s.case_id}\t{s.reason}\t{s.details}\n")


def write_attempt_report(attempts: Sequence[RegAttempt], report_f: Path) -> None:
    report_f.parent.mkdir(parents=True, exist_ok=True)
    with report_f.open("w", encoding="utf-8") as f:
        f.write("case_id\tattempt\tstrategy\tnon_rigid\tmax_proc\tmax_nonrigid\tn_slides\tstatus\terror\n")
        for a in attempts:
            err = a.error.replace("\n", " ").replace("\t", " ")
            f.write(
                f"{a.case_id}\t{a.attempt}\t{a.strategy}\t{int(a.non_rigid)}\t{a.max_proc}\t{a.max_nonrigid}\t"
                f"{a.n_slides}\t{a.status}\t{err}\n"
            )


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


def estimate_tissue_fraction(slide_f: Path) -> Optional[float]:
    try:
        import numpy as np
        from valis import slide_io

        reader_cls = slide_io.get_slide_reader(str(slide_f), series=0)
        reader = reader_cls(str(slide_f), series=0)
        sizes = reader.metadata.slide_dimensions
        if not sizes:
            return None
        level = max(0, len(sizes) - 1)
        img = reader.slide2image(level=level, series=0)
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.ndim == 3:
            gray = arr[..., :3].mean(axis=2)
        else:
            gray = arr.astype(np.float32)
        if gray.size == 0:
            return None
        # Quick tissue proxy on thumbnail: darker-than-background pixels.
        p95 = float(np.percentile(gray, 95))
        thresh = max(20.0, p95 * 0.88)
        tissue = (gray < thresh).astype(np.uint8)
        return float(tissue.mean())
    except Exception:
        return None


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
    huge_side_px: int,
    run_micro: bool,
    tissue_min_frac: float,
    use_tissue_filter: bool,
) -> List[RegAttempt]:
    from valis import registration

    case_id = case_group.case_id
    results_dst_dir = results_base / case_id
    registered_slide_dst_dir = results_dst_dir / "registered_slides"
    big_case, max_side = calc_big_case(case_group.slides, big_side_px)
    max_proc = BIG_MAX_PROC if big_case else DEF_MAX_PROC
    max_nonrigid = BIG_MAX_NONR if big_case else DEF_MAX_NONR
    crop_mode = BIG_CROP if big_case else DEF_CROP
    do_micro = run_micro and (not big_case)
    attempts: List[RegAttempt] = []

    print(f"\n=== {case_id} ===")
    print(f"[INFO] reference: {case_group.reference_slide.name}")
    print(f"[INFO] slides: {len(case_group.slides)}, max level-0 side={max_side}, big_case={big_case}")
    if big_case:
        print(
            f"[INFO] BIG-GUARD -> max_proc={max_proc}, "
            f"max_nonrigid={max_nonrigid}, crop={crop_mode}, register_micro={do_micro}"
        )

    slide_stats: Dict[Path, Tuple[int, Optional[float]]] = {}
    huge_case = False
    for s in case_group.slides:
        side = level0_max_side(s)
        tf = estimate_tissue_fraction(s) if use_tissue_filter else None
        slide_stats[s] = (side, tf)
        if side >= huge_side_px:
            huge_case = True

    selected_slides = list(case_group.slides)
    if use_tissue_filter:
        filtered: List[Path] = [case_group.reference_slide]
        for s in case_group.slides:
            if s == case_group.reference_slide:
                continue
            _, tf = slide_stats[s]
            if tf is None or tf >= tissue_min_frac:
                filtered.append(s)
        selected_slides = sorted(set(filtered))
        print(f"[INFO] tissue filter kept {len(selected_slides)}/{len(case_group.slides)} slides")

    if len(selected_slides) < 2:
        raise RuntimeError("Too few slides after filtering (<2), skip")

    if huge_case:
        max_proc = min(max_proc, HUGE_MAX_PROC)
        max_nonrigid = min(max_nonrigid, HUGE_MAX_NONR)
        crop_mode = "overlap"
        do_micro = False
        print(
            f"[INFO] HUGE-GUARD -> side>={huge_side_px}, force max_proc={max_proc}, "
            f"max_nonrigid={max_nonrigid}, non_rigid baseline may downgrade"
        )

    candidates_without_ref = [s for s in selected_slides if s != case_group.reference_slide]
    candidates_without_ref = sorted(
        candidates_without_ref,
        key=lambda x: (slide_stats[x][1] if slide_stats[x][1] is not None else 1.0),
        reverse=True,
    )
    small_set = [case_group.reference_slide] + candidates_without_ref[: min(6, len(candidates_without_ref))]

    strategies = [
        {
            "name": "default_or_huge_guard",
            "slides": selected_slides,
            "max_proc": max_proc,
            "max_nonrigid": max_nonrigid,
            "non_rigid": (not huge_case),
            "imgs_ordered": True,
            "align_to_reference": True,
            "do_micro": do_micro and (not huge_case),
            "crop": crop_mode,
        },
        {
            "name": "fallback_rigid_1000",
            "slides": selected_slides,
            "max_proc": FALLBACK1_MAX_PROC,
            "max_nonrigid": FALLBACK1_MAX_NONR,
            "non_rigid": False,
            "imgs_ordered": True,
            "align_to_reference": True,
            "do_micro": False,
            "crop": "overlap",
        },
        {
            "name": "fallback_rigid_800_smallset",
            "slides": sorted(set(small_set)),
            "max_proc": FALLBACK2_MAX_PROC,
            "max_nonrigid": FALLBACK2_MAX_NONR,
            "non_rigid": False,
            "imgs_ordered": True,
            "align_to_reference": True,
            "do_micro": False,
            "crop": "overlap",
        },
    ]

    src_dir = Path(os.path.commonpath([str(p.parent) for p in case_group.slides]))
    last_error = ""
    for i, st in enumerate(strategies, start=1):
        if len(st["slides"]) < 2:
            attempts.append(
                RegAttempt(
                    case_id=case_id,
                    attempt=i,
                    strategy=st["name"],
                    non_rigid=bool(st["non_rigid"]),
                    max_proc=int(st["max_proc"]),
                    max_nonrigid=int(st["max_nonrigid"]),
                    n_slides=len(st["slides"]),
                    status="SKIP",
                    error="n_slides<2",
                )
            )
            continue
        print(
            f"[INFO] attempt={i} strategy={st['name']} n_slides={len(st['slides'])} "
            f"max_proc={st['max_proc']} max_nonrigid={st['max_nonrigid']} non_rigid={st['non_rigid']}"
        )
        try:
            registrar = registration.Valis(
                str(src_dir),
                str(results_dst_dir),
                max_processed_image_dim_px=int(st["max_proc"]),
                max_non_rigid_registration_dim_px=int(st["max_nonrigid"]),
                create_masks=True,
                reference_img_f=str(case_group.reference_slide),
                img_list=[str(p) for p in st["slides"]],
                imgs_ordered=bool(st["imgs_ordered"]),
                align_to_reference=bool(st["align_to_reference"]),
                micro_rigid_registrar_cls=None,
                crop_for_rigid_reg=False,
                check_for_reflections=False,
            )
            registrar.register()
            if bool(st["do_micro"]):
                registrar.register_micro(
                    max_non_rigid_registration_dim_px=int(st["max_nonrigid"]),
                    align_to_reference=True,
                )
            registrar.warp_and_save_slides(
                str(registered_slide_dst_dir),
                crop=str(st["crop"]),
                non_rigid=bool(st["non_rigid"]),
            )
            attempts.append(
                RegAttempt(
                    case_id=case_id,
                    attempt=i,
                    strategy=st["name"],
                    non_rigid=bool(st["non_rigid"]),
                    max_proc=int(st["max_proc"]),
                    max_nonrigid=int(st["max_nonrigid"]),
                    n_slides=len(st["slides"]),
                    status="SUCCESS",
                    error="",
                )
            )
            return attempts
        except Exception as e:
            last_error = str(e)
            attempts.append(
                RegAttempt(
                    case_id=case_id,
                    attempt=i,
                    strategy=st["name"],
                    non_rigid=bool(st["non_rigid"]),
                    max_proc=int(st["max_proc"]),
                    max_nonrigid=int(st["max_nonrigid"]),
                    n_slides=len(st["slides"]),
                    status="FAIL",
                    error=str(e),
                )
            )
        finally:
            try:
                registration.kill_jvm()
            except Exception:
                pass

    raise RuntimeError(f"all registration strategies failed: {last_error}")


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
        "--skip-log",
        default=None,
        help="Path to save skipped case report. Default: <results-base>/skipped_cases_he_rule.tsv",
    )
    p.add_argument("--big-side-px", type=int, default=12000, help="Any slide edge >= this is treated as BIG case")
    p.add_argument("--huge-side-px", type=int, default=HUGE_SIDE_PX, help="Any slide edge >= this triggers hard downgrade")
    p.add_argument("--run-micro", action="store_true", help="Enable register_micro on non-BIG cases")
    p.add_argument("--dry-run", action="store_true", help="Only print parsed case groups and filenames")
    p.add_argument("--limit-cases", type=int, default=50, help="Preview max case groups to print")
    p.add_argument("--limit-slides", type=int, default=30, help="Preview max slides per case to print")
    p.add_argument("--skip-done", action="store_true", help="Skip case_id if results-base/case_id already exists")
    p.add_argument(
        "--dedup-stain",
        action="store_true",
        help="Keep only one slide per canonical stain (e.g. HE/HE_v2 -> pick one)",
    )
    p.add_argument("--tissue-min-frac", type=float, default=0.01, help="Min tissue fraction for non-reference slides")
    p.add_argument("--disable-tissue-filter", action="store_true", help="Disable thumbnail tissue fraction filter")
    p.add_argument(
        "--attempt-log",
        default=None,
        help="Path to save registration attempts report. Default: <results-base>/registration_attempts.tsv",
    )
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

    case_groups, skipped_cases = build_case_groups(
        slides=slides,
        case_stain_splitter=args.case_stain_splitter,
        dedup_stain=args.dedup_stain,
    )
    if not case_groups:
        print("[WARN] no valid case groups after parsing filenames")
    print(f"[INFO] skipped cases by HE rule: {len(skipped_cases)}")
    for s in skipped_cases[:20]:
        print(f"[WARN] skip {s.case_id}: {s.reason} ({s.details})")
    if len(skipped_cases) > 20:
        print("[INFO] ... skipped list truncated in console (see skip log file)")

    if args.skip_done and results_base.is_dir():
        done_cases = {d.name for d in results_base.iterdir() if d.is_dir()}
        case_groups = [cg for cg in case_groups if cg.case_id not in done_cases]
        print(f"[INFO] remaining cases after skip-done: {len(case_groups)}")

    skip_log = Path(args.skip_log).expanduser().resolve() if args.skip_log else (results_base / "skipped_cases_he_rule.tsv")
    write_skip_report(skipped_cases, skip_log)
    print(f"[INFO] skip report saved: {skip_log}")

    if not case_groups:
        print("[WARN] no cases left for registration after HE rule/skip-done")
        return

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

    all_attempts: List[RegAttempt] = []
    for cg in case_groups:
        try:
            case_attempts = run_case_registration(
                case_group=cg,
                results_base=results_base,
                big_side_px=args.big_side_px,
                huge_side_px=args.huge_side_px,
                run_micro=args.run_micro,
                tissue_min_frac=args.tissue_min_frac,
                use_tissue_filter=(not args.disable_tissue_filter),
            )
            all_attempts.extend(case_attempts)
        except Exception as e:
            print(f"[ERROR] case={cg.case_id} failed: {e}")
            all_attempts.append(
                RegAttempt(
                    case_id=cg.case_id,
                    attempt=0,
                    strategy="terminal",
                    non_rigid=False,
                    max_proc=0,
                    max_nonrigid=0,
                    n_slides=len(cg.slides),
                    status="FAIL",
                    error=str(e),
                )
            )
            try:
                registration.kill_jvm()
            except Exception:
                pass

    attempt_log = (
        Path(args.attempt_log).expanduser().resolve()
        if args.attempt_log
        else (results_base / "registration_attempts.tsv")
    )
    write_attempt_report(all_attempts, attempt_log)
    print(f"[INFO] registration attempt report saved: {attempt_log}")


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
