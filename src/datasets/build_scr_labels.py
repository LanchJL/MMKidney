import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.io import try_parse_datetime, write_json


UMOL_UNITS = {"umol/l", "μmol/l", "µmol/l", "micromol/l"}
MGDL_UNITS = {"mg/dl"}
CREATININE_ALIASES = {
    "Blood Creatinine",
    "血肌酐",
    "肌酐",
}


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


def parse_window_spec(spec: str) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for part in (spec or "").split(","):
        p = part.strip()
        if not p:
            continue
        k, a, b = p.split(":")
        h = int(k)
        out[h] = (float(a), float(b))
    return out


def parse_days_csv(x: str) -> List[int]:
    return [int(v.strip()) for v in (x or "").split(",") if v.strip()]


def to_umol(value_raw: str, unit_raw: str, qc: Counter) -> Optional[float]:
    s = (value_raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        qc["invalid_numeric"] += 1
        return None

    if v <= 0:
        qc["non_positive"] += 1
        return None

    u = normalize_unit(unit_raw)

    # Conservative conversion for quality: only known units are accepted.
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


def fmt_float(x: Optional[float], nd: int = 6) -> str:
    if x is None:
        return ""
    return f"{float(x):.{nd}f}"


def main():
    p = argparse.ArgumentParser("Build short-horizon Blood Creatinine prediction labels")
    p.add_argument("--cohort-csv", default="data/processed/cohort_trimodal.csv")
    p.add_argument("--lab-csv", default="data/Merged_Lab_Results_Final.csv")
    p.add_argument("--out-csv", default="data/processed/scr_labels.csv")
    p.add_argument("--out-qc", default="data/processed/scr_label_qc.json")
    p.add_argument("--horizons-days", default="14,30,90")
    p.add_argument("--window-spec", default="14:10:22,30:21:46,90:75:121,180:150:241")
    p.add_argument("--baseline-windows-days", default="30,90,180,365")
    p.add_argument("--exclude-t0-day", action="store_true")
    args = p.parse_args()

    horizons = parse_days_csv(args.horizons_days)
    window_map = parse_window_spec(args.window_spec)
    base_windows = parse_days_csv(args.baseline_windows_days)

    for h in horizons:
        if h not in window_map:
            raise ValueError(f"Missing window spec for horizon {h} days")

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

        dg = try_parse_datetime(r.get("date_of_graft_loss", ""))
        dd = try_parse_datetime(r.get("date_of_death", ""))
        censor = None
        if dg is not None and dd is not None:
            censor = min(dg, dd)
        elif dg is not None:
            censor = dg
        elif dd is not None:
            censor = dd

        rec = {
            "sample_id": sid,
            "patient_index": pid,
            "split": (r.get("split", "") or "").strip(),
            "t0": t0,
            "date_of_transplant": (r.get("date_of_transplant", "") or "").strip(),
            "date_of_graft_loss": (r.get("date_of_graft_loss", "") or "").strip(),
            "date_of_death": (r.get("date_of_death", "") or "").strip(),
            "censor": censor,
        }
        idx = len(samples)
        samples.append(rec)
        by_pid[pid].append(idx)

    obs_by_idx = defaultdict(list)
    qc_counts = Counter()

    with Path(args.lab_csv).open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Lab CSV has empty header")
        pid_col = reader.fieldnames[0]

        for row in reader:
            pid = (row.get(pid_col, "") or "").strip()
            if pid not in by_pid:
                continue
            test = normalize_test_name(row.get("检验", ""))
            if test != "Blood Creatinine":
                continue

            t = try_parse_datetime(row.get("检验时间", ""))
            if t is None:
                qc_counts["invalid_time"] += 1
                continue

            v_umol = to_umol(row.get("定量结果", ""), row.get("定量结果单位", ""), qc_counts)
            if v_umol is None:
                continue

            if v_umol > 2500:
                qc_counts["extreme_too_large"] += 1
                continue

            for idx in by_pid[pid]:
                t0 = samples[idx]["t0"]
                delta = (t - t0).total_seconds() / 86400.0
                obs_by_idx[idx].append((delta, v_umol, t))

    out_rows = []
    qc_h = {h: Counter() for h in horizons}
    baseline_available = 0

    for idx, smp in enumerate(samples):
        sid = smp["sample_id"]
        pid = smp["patient_index"]
        t0 = smp["t0"]
        censor = smp["censor"]
        obs = sorted(obs_by_idx.get(idx, []), key=lambda x: x[0])

        if censor is not None:
            censor_delta = (censor - t0).total_seconds() / 86400.0
        else:
            censor_delta = None

        pre = []
        for d, v, _ in obs:
            if args.exclude_t0_day:
                cond = d < 0
            else:
                cond = d <= 0
            if cond:
                pre.append((d, v))

        baseline = None
        baseline_days_from_t0 = None
        baseline_window_used = None
        baseline_n = 0

        for bw in base_windows:
            cand = [(d, v) for d, v in pre if d >= -float(bw)]
            if cand:
                vals = [x[1] for x in cand]
                baseline = float(statistics.median(vals))
                near = max(cand, key=lambda x: x[0])
                baseline_days_from_t0 = float(abs(near[0]))
                baseline_window_used = bw
                baseline_n = len(cand)
                break

        if baseline is not None:
            baseline_available += 1

        rec = {
            "sample_id": sid,
            "patient_index": pid,
            "split": smp["split"],
            "t0": t0.isoformat(sep=" "),
            "date_of_transplant": smp["date_of_transplant"],
            "date_of_graft_loss": smp["date_of_graft_loss"],
            "date_of_death": smp["date_of_death"],
            "baseline_scr_umol": fmt_float(baseline, 4),
            "baseline_log_scr": fmt_float(math.log(max(baseline, 1e-6)), 6) if baseline is not None else "",
            "baseline_days_from_t0": fmt_float(baseline_days_from_t0, 2),
            "baseline_window_days": str(baseline_window_used or ""),
            "baseline_source_count": str(baseline_n),
        }

        for h in horizons:
            start, end = window_map[h]
            key = f"d{h}"

            if censor_delta is not None and censor_delta <= start:
                rec[f"mask_{key}"] = "0"
                rec[f"label_source_count_{key}"] = "0"
                rec[f"scr_{key}_umol"] = ""
                rec[f"log_scr_{key}"] = ""
                rec[f"delta_log_scr_{key}"] = ""
                rec[f"label_days_from_horizon_{key}"] = ""
                qc_h[h]["censored_before_window"] += 1
                continue

            cand = []
            for d, v, _ in obs:
                if d < start or d >= end:
                    continue
                if censor_delta is not None and d >= censor_delta:
                    continue
                cand.append((d, v))

            if not cand:
                rec[f"mask_{key}"] = "0"
                rec[f"label_source_count_{key}"] = "0"
                rec[f"scr_{key}_umol"] = ""
                rec[f"log_scr_{key}"] = ""
                rec[f"delta_log_scr_{key}"] = ""
                rec[f"label_days_from_horizon_{key}"] = ""
                qc_h[h]["missing_in_window"] += 1
                continue

            nearest = min(cand, key=lambda x: abs(x[0] - float(h)))
            v_h = float(nearest[1])
            log_h = math.log(max(v_h, 1e-6))
            delta_log = (log_h - math.log(max(baseline, 1e-6))) if baseline is not None else None

            rec[f"mask_{key}"] = "1"
            rec[f"label_source_count_{key}"] = str(len(cand))
            rec[f"scr_{key}_umol"] = fmt_float(v_h, 4)
            rec[f"log_scr_{key}"] = fmt_float(log_h, 6)
            rec[f"delta_log_scr_{key}"] = fmt_float(delta_log, 6)
            rec[f"label_days_from_horizon_{key}"] = fmt_float(abs(nearest[0] - float(h)), 2)
            qc_h[h]["labeled"] += 1

        out_rows.append(rec)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else ["sample_id", "patient_index", "split", "t0"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    qc = {
        "config": {
            "cohort_csv": args.cohort_csv,
            "lab_csv": args.lab_csv,
            "horizons_days": horizons,
            "window_spec": {str(k): list(v) for k, v in window_map.items()},
            "baseline_windows_days": base_windows,
            "exclude_t0_day": bool(args.exclude_t0_day),
        },
        "n_samples": len(samples),
        "n_rows_out": len(out_rows),
        "baseline_available": baseline_available,
        "baseline_rate": (baseline_available / len(out_rows)) if out_rows else 0.0,
        "horizon": {
            f"d{h}": {
                "labeled": int(qc_h[h]["labeled"]),
                "missing_in_window": int(qc_h[h]["missing_in_window"]),
                "censored_before_window": int(qc_h[h]["censored_before_window"]),
                "label_rate": (qc_h[h]["labeled"] / len(out_rows)) if out_rows else 0.0,
            }
            for h in horizons
        },
        "lab_qc_drop": dict(qc_counts),
    }
    write_json(Path(args.out_qc), qc)

    print("[scr_labels] samples:", len(samples))
    print("[scr_labels] out:", out_path)
    print("[scr_labels] qc:", args.out_qc)
    for h in horizons:
        print(f"[scr_labels] d{h} labeled:", int(qc_h[h]["labeled"]))


if __name__ == "__main__":
    main()
