import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.utils.io import try_parse_datetime, write_json


TEST_ALIASES = {
    "Blood Creatinine": "Blood Creatinine",
    "UACR": "UACR",
    "Urine Cr": "Urine Cr",
    "Urine 24h Cr": "Urine 24h Cr",
    "Blood Urea to Creatinine Ratio": "Blood Urea to Creatinine Ratio",
    "血肌酐": "Blood Creatinine",
    "肌酐": "Blood Creatinine",
}

DEFAULT_TARGET_TESTS = [
    "Blood Creatinine",
    "UACR",
    "Urine Cr",
    "Urine 24h Cr",
    "Blood Urea to Creatinine Ratio",
]
DEFAULT_WINDOWS_DAYS = [7, 30, 90, 180, 365]


def normalize_test_name(name: str) -> str:
    n = (name or "").strip()
    if n in TEST_ALIASES:
        return TEST_ALIASES[n]
    low = n.lower()
    if "blood creatinine" in low:
        return "Blood Creatinine"
    if "uacr" == low:
        return "UACR"
    if low in {"urine cr", "urinecr"}:
        return "Urine Cr"
    if "urine 24h cr" in low:
        return "Urine 24h Cr"
    if "urea to creatinine ratio" in low:
        return "Blood Urea to Creatinine Ratio"
    if "肌酐" in n:
        return "Blood Creatinine"
    return n


def _safe_float(x: str):
    s = (x or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _stats(vals: List[float], day_offsets: List[float], anchor: datetime, times: List[datetime]) -> Dict[str, float]:
    if not vals:
        return {
            "count": 0.0,
            "last": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "slope": 0.0,
            "delta_last_first": 0.0,
            "days_since_last_measurement": 9999.0,
            "has_measurement": 0.0,
        }

    order = sorted(range(len(times)), key=lambda i: times[i])
    vals_ord = [vals[i] for i in order]
    d_ord = [day_offsets[i] for i in order]
    t_ord = [times[i] for i in order]

    slope = 0.0
    if len(vals_ord) >= 2 and len(set(d_ord)) >= 2:
        n = float(len(d_ord))
        sx = sum(d_ord)
        sy = sum(vals_ord)
        sxx = sum(x * x for x in d_ord)
        sxy = sum(x * y for x, y in zip(d_ord, vals_ord))
        denom = n * sxx - sx * sx
        if denom != 0.0:
            slope = float((n * sxy - sx * sy) / denom)

    days_since_last = float((anchor - t_ord[-1]).total_seconds() / 86400.0)

    return {
        "count": float(len(vals_ord)),
        "last": float(vals_ord[-1]),
        "mean": float(statistics.mean(vals_ord)),
        "median": float(statistics.median(vals_ord)),
        "std": float(statistics.pstdev(vals_ord)) if len(vals_ord) > 1 else 0.0,
        "min": float(min(vals_ord)),
        "max": float(max(vals_ord)),
        "slope": slope,
        "delta_last_first": float(vals_ord[-1] - vals_ord[0]),
        "days_since_last_measurement": days_since_last,
        "has_measurement": 1.0,
    }


def _try_write_parquet(rows: List[Dict], path: Path) -> bool:
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _parse_days_list(text: str, default_vals: List[int]) -> List[int]:
    vals = [int(x.strip()) for x in (text or "").split(",") if x.strip()]
    return vals if vals else list(default_vals)


def _parse_tests_list(text: str, default_vals: List[str]) -> List[str]:
    vals = [x.strip() for x in (text or "").split(",") if x.strip()]
    return vals if vals else list(default_vals)


def _select_tests_high_coverage(
    lab_csv: Path,
    by_pid: Dict[str, List[Tuple[str, datetime]]],
    min_patient_coverage: float,
    min_patient_count: int,
    max_tests: int,
    include_targets: List[str],
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    test_rows = Counter()
    test_patients: Dict[str, Set[str]] = defaultdict(set)

    with lab_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return include_targets, {}
        pid_col = reader.fieldnames[0]

        for row in reader:
            pid = (row.get(pid_col) or "").strip()
            if pid not in by_pid:
                continue
            val = _safe_float(row.get("定量结果", ""))
            if val is None:
                continue
            t = try_parse_datetime(row.get("检验时间", ""))
            if t is None:
                continue
            test = normalize_test_name(row.get("检验", ""))
            if not test:
                continue
            test_rows[test] += 1
            test_patients[test].add(pid)

    n_pat = max(1, len(by_pid))
    threshold = max(int(math.ceil(min_patient_coverage * n_pat)), int(min_patient_count))

    selected = []
    for test, pset in test_patients.items():
        if len(pset) >= threshold:
            selected.append(test)

    selected = sorted(selected, key=lambda t: (-len(test_patients[t]), -test_rows[t], t))

    for t in include_targets:
        if t not in selected:
            selected.append(t)

    if max_tests > 0 and len(selected) > max_tests:
        must_keep = [t for t in include_targets if t in selected]
        others = [t for t in selected if t not in must_keep]
        keep_n = max(0, max_tests - len(must_keep))
        selected = must_keep + others[:keep_n]

    coverage = {
        t: {
            "n_rows": int(test_rows.get(t, 0)),
            "n_patients": int(len(test_patients.get(t, set()))),
        }
        for t in selected
    }
    return selected, coverage


def main():
    p = argparse.ArgumentParser("Build longitudinal lab summary features (pre-anchor only)")
    p.add_argument("--cohort-trimodal", default="data/processed/cohort_trimodal.csv")
    p.add_argument("--lab-csv", default="data/Merged_Lab_Results_Final.csv")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--windows-days", default="7,30,90,180,365")
    p.add_argument("--lab-mode", choices=["target5", "aux_high_coverage"], default="target5")
    p.add_argument("--target-tests", default=",".join(DEFAULT_TARGET_TESTS))
    p.add_argument("--min-patient-coverage", type=float, default=0.10)
    p.add_argument("--min-patient-count", type=int, default=20)
    p.add_argument("--max-tests", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    windows_days = _parse_days_list(args.windows_days, DEFAULT_WINDOWS_DAYS)
    target_tests = _parse_tests_list(args.target_tests, DEFAULT_TARGET_TESTS)

    with Path(args.cohort_trimodal).open("r", encoding="utf-8", newline="") as f:
        cohort = list(csv.DictReader(f))

    samples = []
    by_pid = defaultdict(list)
    for r in cohort:
        sid = r["sample_id"]
        pid = r["patient_index"]
        anchor = try_parse_datetime(r.get("anchor_date_raw", ""))
        if anchor is None:
            anchor = try_parse_datetime(r.get("report_time", ""))
        if anchor is None:
            anchor = try_parse_datetime(r.get("application_date", ""))
        if anchor is None:
            continue
        samples.append((sid, pid, anchor, r.get("split", "")))
        by_pid[pid].append((sid, anchor))

    sample_split = {sid: sp for sid, _, _, sp in samples}

    if args.lab_mode == "target5":
        selected_tests = list(target_tests)
        selected_coverage = {}
    else:
        selected_tests, selected_coverage = _select_tests_high_coverage(
            lab_csv=Path(args.lab_csv),
            by_pid=by_pid,
            min_patient_coverage=float(args.min_patient_coverage),
            min_patient_count=int(args.min_patient_count),
            max_tests=int(args.max_tests),
            include_targets=target_tests,
        )

    selected_set = set(selected_tests)

    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    global_count = defaultdict(lambda: defaultdict(int))
    unit_counter = defaultdict(Counter)

    with Path(args.lab_csv).open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        pat_col = reader.fieldnames[0]
        for row in reader:
            pid = (row.get(pat_col) or "").strip()
            if pid not in by_pid:
                continue

            test = normalize_test_name(row.get("检验", ""))
            if test not in selected_set:
                continue

            val = _safe_float(row.get("定量结果", ""))
            if val is None:
                continue

            t = try_parse_datetime(row.get("检验时间", ""))
            if t is None:
                continue

            unit = (row.get("定量结果单位", "") or "").strip()
            if unit:
                unit_counter[test][unit] += 1

            for sid, anchor in by_pid[pid]:
                if t >= anchor:
                    continue
                delta_days = (anchor - t).total_seconds() / 86400.0
                if delta_days < 0:
                    continue
                global_count[sid][test] += 1
                for w in windows_days:
                    if delta_days < w:
                        store[sid][test][w].append((val, -delta_days, t))

    unit_main = {}
    unit_warnings = {}
    for test, c in unit_counter.items():
        if not c:
            continue
        main_u, _ = c.most_common(1)[0]
        unit_main[test] = main_u
        non_main = {u: n for u, n in c.items() if u != main_u}
        if non_main:
            unit_warnings[test] = non_main

    feature_rows = []
    for sid, pid, anchor, _ in samples:
        rec = {
            "sample_id": sid,
            "patient_index": pid,
            "anchor_timestamp": anchor.isoformat(sep=" "),
        }

        for test in selected_tests:
            rec[f"global_count_{test}"] = float(global_count[sid].get(test, 0))
            for w in windows_days:
                obs = store[sid][test][w]
                vals = [x[0] for x in obs]
                dts = [x[1] for x in obs]
                ts = [x[2] for x in obs]
                st = _stats(vals, dts, anchor, ts)
                prefix = f"{test}__w{w}d"
                for k, v in st.items():
                    rec[f"{prefix}__{k}"] = v

        feature_rows.append(rec)

    out_csv = out_dir / "lab_features.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(feature_rows[0].keys()) if feature_rows else ["sample_id", "patient_index", "anchor_timestamp"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(feature_rows)

    parquet_ok = _try_write_parquet(feature_rows, out_dir / "lab_features.parquet")

    split_cov = defaultdict(lambda: {"n": 0, "has_any_lab": 0})
    for sid, _, _, sp in samples:
        split_cov[sp]["n"] += 1
        has_any = 1 if any(global_count[sid].get(t, 0) > 0 for t in selected_tests) else 0
        split_cov[sp]["has_any_lab"] += has_any

    write_json(
        out_dir / "lab_unit_report.json",
        {
            "lab_mode": args.lab_mode,
            "selected_tests": selected_tests,
            "selected_tests_n": len(selected_tests),
            "target_tests": target_tests,
            "window_days": windows_days,
            "main_unit": unit_main,
            "non_main_unit_warnings": unit_warnings,
            "selected_coverage": selected_coverage,
            "split_any_lab_coverage": {
                sp: {
                    "has_any_lab": int(v["has_any_lab"]),
                    "n": int(v["n"]),
                }
                for sp, v in split_cov.items()
            },
        },
    )

    print("[lab] mode:", args.lab_mode)
    print("[lab] selected tests:", len(selected_tests))
    print("[lab] windows:", windows_days)
    print("[lab] samples:", len(feature_rows))
    print("[lab] csv:", out_csv)
    print("[lab] parquet_written:", parquet_ok)
    print("[lab] unit warnings tests:", len(unit_warnings))


if __name__ == "__main__":
    main()
