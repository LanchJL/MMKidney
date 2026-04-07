import argparse
import csv
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

TARGET_TESTS = [
    "Blood Creatinine",
    "UACR",
    "Urine Cr",
    "Urine 24h Cr",
    "Blood Urea to Creatinine Ratio",
]
WINDOWS_DAYS = [7, 30, 90, 180, 365]


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
        # simple linear regression slope y = a*x + b
        n = float(len(d_ord))
        sx = sum(d_ord)
        sy = sum(vals_ord)
        sxx = sum(x * x for x in d_ord)
        sxy = sum(x * y for x, y in zip(d_ord, vals_ord))
        denom = (n * sxx - sx * sx)
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


def main():
    p = argparse.ArgumentParser("Build longitudinal lab summary features (pre-anchor only)")
    p.add_argument("--cohort-trimodal", default="data/processed/cohort_trimodal.csv")
    p.add_argument("--lab-csv", default="data/Merged_Lab_Results_Final.csv")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    with Path(args.cohort_trimodal).open("r", encoding="utf-8", newline="") as f:
        cohort = list(csv.DictReader(f))

    # sample-level anchor map
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

    # storage: sample -> test -> window -> list of observations
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
            test_raw = row.get("检验", "")
            test = normalize_test_name(test_raw)
            if test not in TARGET_TESTS:
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
                    continue  # pre-anchor only
                delta_days = (anchor - t).total_seconds() / 86400.0
                if delta_days < 0:
                    continue
                global_count[sid][test] += 1
                for w in WINDOWS_DAYS:
                    if delta_days < w:
                        store[sid][test][w].append((val, -delta_days, t))

    # major unit and warnings
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

        for test in TARGET_TESTS:
            rec[f"global_count_{test}"] = float(global_count[sid].get(test, 0))
            for w in WINDOWS_DAYS:
                obs = store[sid][test][w]
                vals = [x[0] for x in obs]
                dts = [x[1] for x in obs]
                ts = [x[2] for x in obs]
                st = _stats(vals, dts, anchor, ts)
                prefix = f"{test}__w{w}d"
                for k, v in st.items():
                    rec[f"{prefix}__{k}"] = v

        feature_rows.append(rec)

    # write csv
    out_csv = out_dir / "lab_features.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(feature_rows[0].keys()) if feature_rows else ["sample_id", "patient_index", "anchor_timestamp"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(feature_rows)

    parquet_ok = _try_write_parquet(feature_rows, out_dir / "lab_features.parquet")

    write_json(
        out_dir / "lab_unit_report.json",
        {
            "target_tests": TARGET_TESTS,
            "window_days": WINDOWS_DAYS,
            "main_unit": unit_main,
            "non_main_unit_warnings": unit_warnings,
        },
    )

    print("[lab] samples:", len(feature_rows))
    print("[lab] csv:", out_csv)
    print("[lab] parquet_written:", parquet_ok)
    print("[lab] unit warnings tests:", len(unit_warnings))


if __name__ == "__main__":
    main()
