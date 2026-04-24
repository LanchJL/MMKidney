import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.io import try_parse_datetime, write_json


META_COLS = {
    "sample_id",
    "split",
    "patient_index",
    "anchor_timestamp",
    "anchor_date_policy",
    "anchor_date_raw",
}

TEXT_FIELDS = [
    "pathology_summary",
    "gross_findings",
    "structured_report",
    "clinical_notes",
    "gt15_nephrology_dx",
    "microscopic_findings",
    "special_stain_description",
    "puncture_note",
    "general_note",
    "gt2",
]

TREATMENT_FIELDS = [
    "rejection_treatment",
    "maintenance_immunosuppression_regimen",
    "cni_dose",
    "mpa_dose",
    "steroid_dose",
    "mpa_note",
    "treatment_after_graft_loss",
]

LEAKAGE_KEYWORDS = [
    "patient_survival",
    "graft_survival",
    "death",
    "graft_loss",
    "latest_scr",
    "latest_date_scr",
    "survival",
    "failure",
    "loss",
    "endpoint",
    "outcome",
    "followup",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_date(x: str):
    return try_parse_datetime(x)


def _days_diff(a, b) -> Optional[int]:
    if a is None or b is None:
        return None
    return int((a - b).total_seconds() // 86400)


def _choose_t0(row: Dict[str, str], source: str):
    if source == "anchor":
        return _parse_date(row.get("anchor_date_raw", ""))
    if source == "application":
        return _parse_date(row.get("application_date", ""))
    if source == "report":
        return _parse_date(row.get("report_time", ""))
    if source == "transplant":
        return _parse_date(row.get("date_of_transplant", ""))
    # biopsy/report proxy
    return (
        _parse_date(row.get("application_date", ""))
        or _parse_date(row.get("report_time", ""))
        or _parse_date(row.get("anchor_date_raw", ""))
    )


def _safe_is_nonempty(row: Dict[str, str], keys: List[str]) -> int:
    return 1 if any((row.get(k, "") or "").strip() for k in keys) else 0


def _norm_yes_no(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"yes", "y", "1", "true", "是", "存活", "成功"}:
        return "yes"
    if s in {"no", "n", "0", "false", "否", "死亡", "失败"}:
        return "no"
    return "missing"


def _modality_name(c: str) -> str:
    if c.startswith("qwen_global_") or c.startswith("medbert_local_") or c.startswith("text_"):
        return "report"
    if c.startswith("banff_"):
        return "banff"
    if c.startswith("timeline_"):
        return "timeline"
    if c.startswith("treatment_"):
        return "treatment"
    if c.startswith("structured_"):
        return "tabular"
    return "other"


def _feature_policy(c: str, allow_post_tx: bool) -> Tuple[str, str, str, str]:
    low = c.lower()
    if c in META_COLS:
        return "exclude", "unknown", "low", "metadata"

    if any(k in low for k in LEAKAGE_KEYWORDS):
        return "exclude", "post_t0", "high", "label/censoring proxy"

    if "post_tx_" in low:
        if allow_post_tx:
            return "input", "unknown", "medium", "post_tx allowed by flag"
        return "exclude", "unknown", "high", "post_tx timing ambiguous"

    if low.startswith("timeline_tx_to_report") or low.startswith("timeline_apply_to_report"):
        return "input", "pre_t0", "low", "timeline before/around t0"

    if low.startswith("timeline_"):
        return "exclude", "unknown", "medium", "timeline provenance uncertain"

    if low.startswith("text_") or low.startswith("qwen_global_") or low.startswith("medbert_local_"):
        return "input", "pre_t0", "low", "pathology report at t0"

    return "input", "baseline", "low", "default safe baseline feature"


def build_feature_manifest(tabular_features_path: Path, allow_post_tx: bool) -> Dict:
    rows = _read_csv(tabular_features_path)
    cols = list(rows[0].keys()) if rows else []
    entries = []
    selected = []
    excluded = []
    for c in cols:
        role, available_time, leakage_risk, reason = _feature_policy(c, allow_post_tx=allow_post_tx)
        item = {
            "feature_name": c,
            "source_column": c,
            "modality": _modality_name(c),
            "available_time": available_time,
            "role": role,
            "leakage_risk": leakage_risk,
            "reason": reason,
        }
        entries.append(item)
        if role == "input":
            selected.append(c)
        else:
            excluded.append(c)
    return {
        "n_total_features": len(cols),
        "n_selected_input_features": len(selected),
        "n_excluded_features": len(excluded),
        "selected_input_features": selected,
        "excluded_features": excluded,
        "features": entries,
    }


def main():
    p = argparse.ArgumentParser("Build leakage-controlled prognosis cohort and feature manifest")
    p.add_argument("--cohort-tabular", default="data/processed/cohort_tabular.csv")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--t0-source", choices=["biopsy_proxy", "anchor", "application", "report", "transplant"], default="biopsy_proxy")
    p.add_argument("--endpoint", choices=["graft_loss", "composite", "patient_death"], default="graft_loss")
    p.add_argument("--allow-post-tx-features", action="store_true")
    p.add_argument("--min-followup-days", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = _read_csv(Path(args.cohort_tabular))
    tab_feat = _read_csv(Path(args.tabular_features))
    lab_feat = _read_csv(Path(args.lab_features))
    tab_ids = {r.get("sample_id", "") for r in tab_feat}
    lab_ids = {r.get("sample_id", "") for r in lab_feat}

    rows = []
    dropped = Counter()
    split_counter = Counter()
    event_counter = Counter()

    for r in cohort:
        sid = r.get("sample_id", "")
        split = r.get("split", "")
        t0 = _choose_t0(r, source=args.t0_source)
        if t0 is None:
            dropped["missing_t0"] += 1
            continue

        d_graft = _parse_date(r.get("date_of_graft_loss", ""))
        d_death = _parse_date(r.get("date_of_death", ""))
        d_latest_scr = _parse_date(r.get("latest_date_scr", ""))

        # ignore events that occurred before or at t0
        if d_graft is not None and d_graft <= t0:
            d_graft = None
        if d_death is not None and d_death <= t0:
            d_death = None

        # censoring proxy: latest known post-t0 date
        censor_candidates = []
        if d_latest_scr is not None and d_latest_scr > t0:
            censor_candidates.append(d_latest_scr)
        if d_graft is not None:
            censor_candidates.append(d_graft)
        if d_death is not None:
            censor_candidates.append(d_death)
        if not censor_candidates:
            dropped["missing_censor_and_event"] += 1
            continue
        censor_date = max(censor_candidates)

        event_graft = 1 if d_graft is not None else 0
        time_graft = _days_diff(d_graft if event_graft else censor_date, t0)

        first_comp = None
        if d_graft is not None and d_death is not None:
            first_comp = min(d_graft, d_death)
        elif d_graft is not None:
            first_comp = d_graft
        elif d_death is not None:
            first_comp = d_death
        event_comp = 1 if first_comp is not None else 0
        time_comp = _days_diff(first_comp if event_comp else censor_date, t0)

        event_death = 1 if d_death is not None else 0
        time_death = _days_diff(d_death if event_death else censor_date, t0)

        followup_days = _days_diff(censor_date, t0)
        if followup_days is None or followup_days < args.min_followup_days:
            dropped["short_followup"] += 1
            continue

        if args.endpoint == "graft_loss":
            event = event_graft
            time_days = time_graft
        elif args.endpoint == "composite":
            event = event_comp
            time_days = time_comp
        else:
            event = event_death
            time_days = time_death

        if time_days is None or time_days < args.min_followup_days:
            dropped["invalid_endpoint_time"] += 1
            continue

        out = {
            "case_id": sid,
            "sample_id": sid,
            "patient_id": r.get("patient_index", ""),
            "patient_index": r.get("patient_index", ""),
            "split": split,
            "t0_date": t0.isoformat(sep=" "),
            "date_of_transplant": r.get("date_of_transplant", ""),
            "date_of_graft_loss": r.get("date_of_graft_loss", ""),
            "date_of_death": r.get("date_of_death", ""),
            "latest_date_scr": r.get("latest_date_scr", ""),
            "event": int(event),
            "time_days": int(time_days),
            "event_graft_loss": int(event_graft),
            "time_graft_loss_days": int(time_graft) if time_graft is not None else "",
            "event_composite": int(event_comp),
            "time_composite_days": int(time_comp) if time_comp is not None else "",
            "event_patient_death": int(event_death),
            "time_patient_death_days": int(time_death) if time_death is not None else "",
            "censor_date": censor_date.isoformat(sep=" "),
            "followup_days": int(followup_days),
            "wsi_available": 1,
            "tabular_available": 1 if sid in tab_ids else 0,
            "lab_available": 1 if sid in lab_ids else 0,
            "report_available": _safe_is_nonempty(r, TEXT_FIELDS),
            "treatment_available": _safe_is_nonempty(r, TREATMENT_FIELDS),
            "followup_success": _norm_yes_no(r.get("followup_success", "")),
            "patient_survival_raw": r.get("patient_survival", ""),
            "graft_survival_raw": r.get("graft_survival", ""),
        }
        rows.append(out)
        split_counter[split] += 1
        event_counter[split] += int(event)

    # stable order
    rows = sorted(rows, key=lambda x: (x.get("split", ""), x.get("case_id", "")))

    fieldnames = list(rows[0].keys()) if rows else [
        "case_id",
        "sample_id",
        "patient_id",
        "split",
        "t0_date",
        "event",
        "time_days",
    ]
    _write_csv(out_dir / "prognosis_cohort.csv", rows, fieldnames)
    _write_jsonl(out_dir / "prognosis_cohort.jsonl", rows)

    manifest = build_feature_manifest(Path(args.tabular_features), allow_post_tx=args.allow_post_tx_features)
    write_json(out_dir / "prognosis_feature_manifest.json", manifest)

    audit = {
        "config": {
            "t0_source": args.t0_source,
            "endpoint": args.endpoint,
            "allow_post_tx_features": bool(args.allow_post_tx_features),
            "min_followup_days": int(args.min_followup_days),
        },
        "n_input_rows": len(cohort),
        "n_output_rows": len(rows),
        "dropped": dict(dropped),
        "split_n": dict(split_counter),
        "split_events": dict(event_counter),
        "event_rate": (sum(x["event"] for x in rows) / len(rows)) if rows else 0.0,
        "median_followup_days": (
            sorted([x["followup_days"] for x in rows])[len(rows) // 2] if rows else 0
        ),
        "feature_manifest_summary": {
            "n_total_features": manifest["n_total_features"],
            "n_selected_input_features": manifest["n_selected_input_features"],
            "n_excluded_features": manifest["n_excluded_features"],
        },
    }
    write_json(out_dir / "prognosis_audit.json", audit)

    print("[prognosis] input:", len(cohort))
    print("[prognosis] output:", len(rows))
    print("[prognosis] split_n:", dict(split_counter))
    print("[prognosis] split_events:", dict(event_counter))
    print("[prognosis] event_rate:", f"{audit['event_rate']:.4f}")
    print("[prognosis] feature selected/excluded:", manifest["n_selected_input_features"], manifest["n_excluded_features"])
    print("[prognosis] files:")
    print(" -", out_dir / "prognosis_cohort.csv")
    print(" -", out_dir / "prognosis_cohort.jsonl")
    print(" -", out_dir / "prognosis_feature_manifest.json")
    print(" -", out_dir / "prognosis_audit.json")


if __name__ == "__main__":
    main()
