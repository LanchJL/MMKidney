import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.io import try_parse_datetime, write_json

CREATININE_ALIASES = {"Blood Creatinine", "血肌酐", "肌酐"}
UMOL_UNITS = {"umol/l", "μmol/l", "µmol/l", "micromol/l"}
MGDL_UNITS = {"mg/dl"}


def _safe_float(x: str) -> Optional[float]:
    s = (x or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None


def normalize_test_name(name: str) -> str:
    n = (name or "").strip()
    if n in CREATININE_ALIASES:
        return "Blood Creatinine"
    low = n.lower()
    if "blood creatinine" in low:
        return "Blood Creatinine"
    if "肌酐" in n:
        return "Blood Creatinine"
    return n


def normalize_unit(unit: str) -> str:
    u = (unit or "").strip().lower().replace(" ", "")
    if u in UMOL_UNITS:
        return "umol/L"
    if u in MGDL_UNITS:
        return "mg/dL"
    return ""


def to_umol(value_raw: str, unit_raw: str, qc: Counter) -> Optional[float]:
    v = _safe_float(value_raw)
    if v is None:
        qc["invalid_numeric"] += 1
        return None
    if v <= 0:
        qc["non_positive"] += 1
        return None

    u = normalize_unit(unit_raw)
    if u == "umol/L":
        if v < 10:
            qc["suspect_umol_too_small"] += 1
            return None
        return float(v)
    if u == "mg/dL":
        if v > 20:
            qc["suspect_mgdl_too_large"] += 1
            return None
        return float(v) * 88.4

    qc["unknown_unit"] += 1
    return None


def choose_t0(row: Dict[str, str]):
    return (
        try_parse_datetime(row.get("anchor_date_raw", ""))
        or try_parse_datetime(row.get("application_date", ""))
        or try_parse_datetime(row.get("report_time", ""))
    )


def main():
    p = argparse.ArgumentParser("Build pre-anchor Blood Creatinine sequence features per sample")
    p.add_argument("--cohort-csv", default="data/processed/cohort_trimodal.csv")
    p.add_argument("--lab-csv", default="data/Merged_Lab_Results_Final.csv")
    p.add_argument("--out-csv", default="data/processed/scr_sequence_features.csv")
    p.add_argument("--out-qc", default="data/processed/scr_sequence_qc.json")
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--exclude-t0-day", action="store_true")
    args = p.parse_args()

    with Path(args.cohort_csv).open("r", encoding="utf-8", newline="") as f:
        cohort = list(csv.DictReader(f))

    samples = []
    by_pid = defaultdict(list)
    for r in cohort:
        sid = (r.get("sample_id", "") or "").strip()
        pid = (r.get("patient_index", "") or "").strip()
        if not sid or not pid:
            continue
        t0 = choose_t0(r)
        if t0 is None:
            continue
        samples.append((sid, pid, t0, (r.get("split", "") or "").strip()))
        by_pid[pid].append((sid, t0))

    obs = defaultdict(list)
    qc = Counter()

    with Path(args.lab_csv).open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Lab CSV has empty header")
        pid_col = reader.fieldnames[0]

        for row in reader:
            pid = (row.get(pid_col) or "").strip()
            if pid not in by_pid:
                continue
            test = normalize_test_name(row.get("检验", ""))
            if test != "Blood Creatinine":
                continue
            t = try_parse_datetime(row.get("检验时间", ""))
            if t is None:
                qc["invalid_time"] += 1
                continue
            v_umol = to_umol(row.get("定量结果", ""), row.get("定量结果单位", ""), qc)
            if v_umol is None:
                continue
            if v_umol > 2500:
                qc["extreme_too_large"] += 1
                continue

            for sid, t0 in by_pid[pid]:
                d = (t - t0).total_seconds() / 86400.0
                if args.exclude_t0_day:
                    is_pre = d < 0
                else:
                    is_pre = d <= 0
                if not is_pre:
                    continue
                if d < -float(args.lookback_days):
                    continue
                # keep days before t0 as positive lag for stability
                lag = float(-d)
                obs[sid].append((lag, v_umol))

    rows = []
    split_cov = defaultdict(lambda: {"n": 0, "has_seq": 0})
    for sid, pid, t0, split in samples:
        seq = sorted(obs.get(sid, []), key=lambda x: x[0], reverse=True)
        # sort reverse -> most recent first; keep max_len recent then restore ascending
        seq = seq[: max(1, int(args.max_len))]
        seq = sorted(seq, key=lambda x: x[0])

        days = [round(x[0], 4) for x in seq]
        vals = [round(x[1], 4) for x in seq]
        has_seq = 1 if seq else 0

        rows.append(
            {
                "sample_id": sid,
                "patient_index": pid,
                "split": split,
                "seq_len": str(len(seq)),
                "has_seq": str(has_seq),
                "scr_days_json": json.dumps(days, ensure_ascii=False),
                "scr_values_umol_json": json.dumps(vals, ensure_ascii=False),
            }
        )

        split_cov[split]["n"] += 1
        split_cov[split]["has_seq"] += has_seq

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["sample_id", "seq_len", "has_seq", "scr_days_json", "scr_values_umol_json"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    qc_out = {
        "config": {
            "cohort_csv": args.cohort_csv,
            "lab_csv": args.lab_csv,
            "lookback_days": int(args.lookback_days),
            "max_len": int(args.max_len),
            "exclude_t0_day": bool(args.exclude_t0_day),
        },
        "n_samples": len(samples),
        "n_rows_out": len(rows),
        "split_seq_coverage": {
            sp: {
                "has_seq": int(v["has_seq"]),
                "n": int(v["n"]),
                "rate": (float(v["has_seq"]) / float(v["n"])) if v["n"] > 0 else 0.0,
            }
            for sp, v in split_cov.items()
        },
        "lab_qc_drop": dict(qc),
    }
    write_json(Path(args.out_qc), qc_out)

    print("[scr_seq] samples:", len(samples))
    print("[scr_seq] out:", args.out_csv)
    print("[scr_seq] qc:", args.out_qc)


if __name__ == "__main__":
    main()
