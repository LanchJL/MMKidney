import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.io import try_parse_datetime, write_json


NUMERIC_FIELDS = ["age", "latest_scr"]
CATEGORICAL_FIELDS = ["sex", "followup_success"]
DATE_FIELDS = ["date_of_transplant", "latest_date_scr", "application_date", "report_time", "anchor_date_raw"]


def _safe_float(x: str):
    s = (x or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _build_vocab(rows: List[Dict[str, str]], key: str) -> Dict[str, int]:
    vals = sorted({(r.get(key, "") or "").strip() for r in rows if (r.get(key, "") or "").strip()})
    return {v: i for i, v in enumerate(vals)}


def _days_diff(a, b):
    if a is None or b is None:
        return None
    return float((a - b).total_seconds() / 86400.0)


def _to_float_or_zero(v):
    return 0.0 if v is None else float(v)


def _to_mask(v):
    return 0.0 if v is None else 1.0


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _try_write_parquet(rows: List[Dict], path: Path) -> bool:
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser("Build tabular features")
    p.add_argument("--cohort-tabular", default="data/processed/cohort_tabular.csv")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()

    in_path = Path(args.cohort_tabular)
    out_dir = Path(args.out_dir)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    sex_vocab = _build_vocab(rows, "sex")
    follow_vocab = _build_vocab(rows, "followup_success")
    vocab = {"sex": sex_vocab, "followup_success": follow_vocab}
    write_json(out_dir / "tabular_vocab.json", vocab)

    out_rows = []
    for r in rows:
        anchor = try_parse_datetime(r.get("anchor_date_raw", ""))
        dt_tx = try_parse_datetime(r.get("date_of_transplant", ""))
        dt_apply = try_parse_datetime(r.get("application_date", ""))
        dt_report = try_parse_datetime(r.get("report_time", ""))
        dt_scr = try_parse_datetime(r.get("latest_date_scr", ""))

        age = _safe_float(r.get("age", ""))
        latest_scr = _safe_float(r.get("latest_scr", ""))

        sex = (r.get("sex", "") or "").strip()
        follow = (r.get("followup_success", "") or "").strip()
        sex_code = float(sex_vocab[sex]) if sex in sex_vocab else None
        follow_code = float(follow_vocab[follow]) if follow in follow_vocab else None

        days_tx_to_report = _days_diff(dt_report, dt_tx)
        days_apply_to_report = _days_diff(dt_report, dt_apply)
        days_latest_scr_to_report = _days_diff(dt_report, dt_scr)
        days_latest_scr_to_anchor = _days_diff(anchor, dt_scr)

        rec = {
            "sample_id": r.get("sample_id", ""),
            "split": r.get("split", ""),
            "patient_index": r.get("patient_index", ""),
            "anchor_date_policy": r.get("anchor_date_policy", "application_date_first_else_report_time"),
            "anchor_date_raw": r.get("anchor_date_raw", ""),
            "anchor_timestamp": anchor.isoformat(sep=" ") if anchor else "",
            "age": _to_float_or_zero(age),
            "age_mask": _to_mask(age),
            "sex_code": _to_float_or_zero(sex_code),
            "sex_mask": _to_mask(sex_code),
            "followup_success_code": _to_float_or_zero(follow_code),
            "followup_success_mask": _to_mask(follow_code),
            "latest_scr": _to_float_or_zero(latest_scr),
            "latest_scr_mask": _to_mask(latest_scr),
            "days_transplant_to_report": _to_float_or_zero(days_tx_to_report),
            "days_transplant_to_report_mask": _to_mask(days_tx_to_report),
            "days_application_to_report": _to_float_or_zero(days_apply_to_report),
            "days_application_to_report_mask": _to_mask(days_apply_to_report),
            "days_latest_scr_to_report": _to_float_or_zero(days_latest_scr_to_report),
            "days_latest_scr_to_report_mask": _to_mask(days_latest_scr_to_report),
            "days_latest_scr_to_anchor": _to_float_or_zero(days_latest_scr_to_anchor),
            "days_latest_scr_to_anchor_mask": _to_mask(days_latest_scr_to_anchor),
            # keep raw fields for debugging
            "raw_age": r.get("age", ""),
            "raw_sex": sex,
            "raw_followup_success": follow,
            "raw_latest_scr": r.get("latest_scr", ""),
            "raw_date_of_transplant": r.get("date_of_transplant", ""),
            "raw_application_date": r.get("application_date", ""),
            "raw_report_time": r.get("report_time", ""),
            "raw_latest_date_scr": r.get("latest_date_scr", ""),
        }
        out_rows.append(rec)

    csv_out = out_dir / "tabular_features.csv"
    _write_csv(csv_out, out_rows, list(out_rows[0].keys()) if out_rows else [])
    parquet_ok = _try_write_parquet(out_rows, out_dir / "tabular_features.parquet")

    print("[tabular] rows:", len(out_rows))
    print("[tabular] csv:", csv_out)
    print("[tabular] parquet_written:", parquet_ok)


if __name__ == "__main__":
    main()
