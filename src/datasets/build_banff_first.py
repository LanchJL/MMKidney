import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.banff.rules import derive_ifta_grade, derive_pvn_class, parse_banff_score
from src.banff.schema import BANFF_TASKS, CI_CT_STAIN_MODES, EXCLUDED_BANFF_TARGETS, assess_feasibility
from src.datasets.stain_utils import normalize_stain_name, standardize_pathology_id


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_distance(pathology_id: str, sample_id: str) -> Optional[int]:
    pid = standardize_pathology_id(pathology_id)
    sid = standardize_pathology_id(sample_id)
    if not pid or not sid:
        return None
    if pid == sid:
        return 0
    if pid in sid:
        return len(sid) - len(pid)
    if sid in pid:
        return len(pid) - len(sid)
    return None


def match_wsi_record(pathology_id: str, records: List[Dict]) -> Optional[Dict]:
    candidates = []
    for rec in records:
        dist = _candidate_distance(pathology_id, rec.get("sample_id", ""))
        if dist is not None:
            candidates.append((dist, len(str(rec.get("sample_id", ""))), rec))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def _normalize_h5s(h5s: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for stain, path in (h5s or {}).items():
        name = normalize_stain_name(stain)
        out.setdefault(name, path)
    return out


def _has_task_stain(h5s: Dict[str, str], stains: List[str]) -> bool:
    present = set(h5s)
    return any(normalize_stain_name(stain) in present for stain in stains)


def _row_label(row, source_col: str) -> Optional[int]:
    if source_col == "__derived_ifta_grade__":
        return derive_ifta_grade(row.get("ci"), row.get("ct"))
    if source_col not in row:
        return None
    score = parse_banff_score(row[source_col])
    if score is None:
        return None
    return max(0, min(int(score), 3))


def build_banff_records(df: pd.DataFrame, wsi_records: List[Dict]) -> Tuple[List[Dict], Dict]:
    out = []
    stats = {
        name: {"label_count": 0, "aligned_count": 0, "class_counts": Counter(), "stain_mode_aligned_counts": Counter()}
        for name in BANFF_TASKS
    }
    unmatched = []

    for _, row in df.iterrows():
        pathology_id = str(row.get("病理号", "")).strip()
        rec = match_wsi_record(pathology_id, wsi_records)
        if rec is None:
            unmatched.append(pathology_id)
            continue

        h5s = _normalize_h5s(rec.get("h5s", {}))
        labels = {}
        masks = {}
        raw_values = {}
        for name, spec in BANFF_TASKS.items():
            label = _row_label(row, spec.source_col)
            raw_values[name] = None if spec.source_col not in row else str(row.get(spec.source_col))
            labels[name] = -1 if label is None else label
            has_label = label is not None
            has_stain = _has_task_stain(h5s, spec.stains)
            masks[name] = 1.0 if has_label and has_stain else 0.0
            if has_label:
                stats[name]["label_count"] += 1
                stats[name]["class_counts"][str(label)] += 1
            if has_label and has_stain:
                stats[name]["aligned_count"] += 1
            if name in {"ci", "ct", "ifta"} and has_label:
                for mode, stains in CI_CT_STAIN_MODES.items():
                    if _has_task_stain(h5s, stains):
                        stats[name]["stain_mode_aligned_counts"][mode] += 1

        derived = {
            "ifta_grade": derive_ifta_grade(row.get("ci"), row.get("ct")),
            "pvn_class": derive_pvn_class(row.get("pvl"), row.get("ci")),
        }
        out.append(
            {
                "sample_id": rec.get("sample_id", pathology_id),
                "pathology_id": pathology_id,
                "split": rec.get("split", ""),
                "h5s": h5s,
                "available_stains": sorted(h5s),
                "banff_labels": labels,
                "banff_masks": masks,
                "banff_raw": raw_values,
                "derived": derived,
                "source": {
                    "GT2": None if pd.isna(row.get("GT2")) else str(row.get("GT2")),
                    "GT3": None if pd.isna(row.get("GT3")) else str(row.get("GT3")),
                },
            }
        )

    task_report = {}
    for name, spec in BANFF_TASKS.items():
        feasibility = assess_feasibility(
            label_count=stats[name]["label_count"],
            aligned_count=stats[name]["aligned_count"],
            class_counts=stats[name]["class_counts"],
            requires_extra_annotation=spec.requires_extra_annotation,
            excluded=spec.excluded,
        )
        task_report[name] = {
            "source_col": spec.source_col,
            "stains": spec.stains,
            "label_count": stats[name]["label_count"],
            "aligned_count": stats[name]["aligned_count"],
            "class_counts": dict(stats[name]["class_counts"]),
            "tier": feasibility.tier,
            "recommendation": feasibility.recommendation,
            "reason": feasibility.reason,
        }
        if name in {"ci", "ct", "ifta"}:
            task_report[name]["stain_mode_aligned_counts"] = {
                mode: int(stats[name]["stain_mode_aligned_counts"].get(mode, 0))
                for mode in CI_CT_STAIN_MODES
            }

    for name, spec in EXCLUDED_BANFF_TARGETS.items():
        feasibility = assess_feasibility(0, 0, {}, spec.requires_extra_annotation, spec.excluded)
        task_report[name] = {
            "source_col": spec.source_col,
            "stains": spec.stains,
            "label_count": 0,
            "aligned_count": 0,
            "class_counts": {},
            "tier": feasibility.tier,
            "recommendation": feasibility.recommendation,
            "reason": spec.reason or feasibility.reason,
        }

    report = {
        "n_excel_rows": int(len(df)),
        "n_matched_wsi": int(len(out)),
        "n_unmatched_wsi": int(len(unmatched)),
        "unmatched_pathology_ids": unmatched,
        "tasks": task_report,
    }
    return out, report


def split_records(records: List[Dict]) -> Dict[str, List[Dict]]:
    splits = defaultdict(list)
    for rec in records:
        splits[rec.get("split", "") or "unknown"].append(rec)
    return dict(splits)


def main() -> None:
    parser = argparse.ArgumentParser("Build Banff-first manifests from pathology Excel and WSI manifests.")
    parser.add_argument("--excel", default="data/pathology_reports_v7.3_with_GT2__20260414  Censored.xlsx")
    parser.add_argument("--sheet", default="移植肾汇总-筛选后")
    parser.add_argument("--wsi-manifest", default="data/processed/manifests/all_manifest.jsonl")
    parser.add_argument("--out-dir", default="data/processed_banff/manifests")
    args = parser.parse_args()

    df = pd.read_excel(args.excel, sheet_name=args.sheet)
    wsi_records = read_jsonl(Path(args.wsi_manifest))
    records, report = build_banff_records(df, wsi_records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "all_banff_manifest.jsonl", records)
    for split, rows in split_records(records).items():
        write_jsonl(out_dir / f"{split}_banff_manifest.jsonl", rows)
    with (out_dir / "banff_feasibility_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"records": len(records), "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
