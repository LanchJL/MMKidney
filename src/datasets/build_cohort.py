import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from .stain_utils import (
    build_stain_vocab,
    dump_stain_vocab,
    normalize_h5s,
    standardize_pathology_id,
)
from .xlsx_utils import read_sheet_rows


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_wsi_samples(data_dir: Path) -> List[Dict]:
    out = []
    for split, fn in [("train", "train_ms.json"), ("val", "val_ms.json"), ("test", "test_ms.json")]:
        arr = _read_json(data_dir / fn)
        for rec in arr:
            sid = standardize_pathology_id(rec.get("id", ""))
            h5s = rec.get("h5s", {}) or {}
            h5s_norm, stains_raw, stains_norm = normalize_h5s(h5s)
            out.append(
                {
                    "sample_id": sid,
                    "orig_id": rec.get("id", ""),
                    "split": split,
                    "labels_L1": rec["labels"]["L1"],
                    "labels_L2": rec["labels"]["L2"],
                    "labels_L3": rec["labels"]["L3"],
                    "stains_raw": stains_raw,
                    "stains_norm": stains_norm,
                    "h5s_norm": h5s_norm,
                }
            )
    return out


def _load_xlsx_index(data_dir: Path) -> Dict[str, Dict]:
    _, rows = read_sheet_rows(
        str(data_dir / "pathology_reports_v6.9_with_baseline_censored.xlsx"),
        "移植肾汇总-筛选后",
    )

    # column letters from inspected file
    # A:患者主索引号 C:年龄 D:性别 M:Date of Transplant AG:是否回访成功 AV/AW:sCr BO:病理号 BQ:申请日期 BS:报告时间 BU:检查结论
    idx = {}
    for r in rows:
        sid = standardize_pathology_id(r.get("BO", ""))
        if not sid:
            continue
        if sid in idx:
            continue
        idx[sid] = {
            "sample_id": sid,
            "patient_index": str(r.get("A", "")).strip(),
            "age": str(r.get("C", "")).strip(),
            "sex": str(r.get("D", "")).strip(),
            "date_of_transplant": str(r.get("M", "")).strip(),
            "followup_success": str(r.get("AG", "")).strip(),
            "latest_date_scr": str(r.get("AV", "")).strip(),
            "latest_scr": str(r.get("AW", "")).strip(),
            "application_date": str(r.get("BQ", "")).strip(),
            "report_time": str(r.get("BS", "")).strip(),
            "pathology_summary": str(r.get("BU", "")).strip(),
        }
    return idx


def _load_lab_patient_set(lab_csv: Path) -> set:
    with lab_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        p_idx = 0
        if "患者主索引" in header:
            p_idx = header.index("患者主索引")
        pats = set()
        for row in r:
            if not row:
                continue
            if p_idx >= len(row):
                continue
            pid = str(row[p_idx]).strip()
            if pid:
                pats.add(pid)
    return pats


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
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


def main():
    p = argparse.ArgumentParser("Build MMKidney cohorts and manifests")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    wsi = _load_wsi_samples(data_dir)
    xlsx_idx = _load_xlsx_index(data_dir)
    lab_patients = _load_lab_patient_set(data_dir / "Merged_Lab_Results_Final.csv")

    stain_vocab = build_stain_vocab([s for r in wsi for s in r["stains_norm"]])
    dump_stain_vocab(out_dir / "stain_vocab.json", stain_vocab)

    cohort_wsi = []
    for r in wsi:
        cohort_wsi.append(
            {
                "sample_id": r["sample_id"],
                "split": r["split"],
                "labels_L1": json.dumps(r["labels_L1"]),
                "labels_L2": json.dumps(r["labels_L2"]),
                "labels_L3": json.dumps(r["labels_L3"]),
                "available_stains": ";".join(r["stains_norm"]),
                "h5_paths_json": json.dumps(r["h5s_norm"], ensure_ascii=False),
            }
        )

    cohort_tabular = []
    matched_pathology = 0
    matched_patient = 0
    for r in wsi:
        sid = r["sample_id"]
        xr = xlsx_idx.get(sid)
        if xr is None:
            continue
        matched_pathology += 1
        pid = xr["patient_index"]
        if pid:
            matched_patient += 1
        row = {
            "sample_id": sid,
            "split": r["split"],
            "patient_index": pid,
            "labels_L1": json.dumps(r["labels_L1"]),
            "labels_L2": json.dumps(r["labels_L2"]),
            "labels_L3": json.dumps(r["labels_L3"]),
            "available_stains": ";".join(r["stains_norm"]),
            "h5_paths_json": json.dumps(r["h5s_norm"], ensure_ascii=False),
        }
        row.update(xr)
        row["anchor_date_policy"] = "application_date_first_else_report_time"
        row["anchor_date_raw"] = xr["application_date"] or xr["report_time"]
        cohort_tabular.append(row)

    cohort_trimodal = []
    for r in cohort_tabular:
        pid = r["patient_index"]
        if pid and pid in lab_patients:
            rr = dict(r)
            rr["has_lab"] = "1"
            cohort_trimodal.append(rr)

    _write_csv(
        out_dir / "cohort_wsi.csv",
        cohort_wsi,
        ["sample_id", "split", "labels_L1", "labels_L2", "labels_L3", "available_stains", "h5_paths_json"],
    )

    tab_fields = [
        "sample_id",
        "split",
        "patient_index",
        "labels_L1",
        "labels_L2",
        "labels_L3",
        "available_stains",
        "h5_paths_json",
        "age",
        "sex",
        "date_of_transplant",
        "followup_success",
        "latest_date_scr",
        "latest_scr",
        "application_date",
        "report_time",
        "pathology_summary",
        "anchor_date_policy",
        "anchor_date_raw",
    ]
    _write_csv(out_dir / "cohort_tabular.csv", cohort_tabular, tab_fields)
    _write_csv(out_dir / "cohort_trimodal.csv", cohort_trimodal, tab_fields + ["has_lab"])

    # manifests
    for split in ["train", "val", "test"]:
        rows = [r for r in cohort_wsi if r["split"] == split]
        manifest = []
        for r in rows:
            manifest.append(
                {
                    "sample_id": r["sample_id"],
                    "split": split,
                    "h5s": json.loads(r["h5_paths_json"]),
                    "labels": {
                        "L1": json.loads(r["labels_L1"]),
                        "L2": json.loads(r["labels_L2"]),
                        "L3": json.loads(r["labels_L3"]),
                    },
                }
            )
        _write_jsonl(out_dir / "manifests" / f"{split}_manifest.jsonl", manifest)

    trimodal_manifest = []
    for r in cohort_trimodal:
        trimodal_manifest.append(
            {
                "sample_id": r["sample_id"],
                "split": r["split"],
                "patient_index": r["patient_index"],
                "h5s": json.loads(r["h5_paths_json"]),
                "labels": {
                    "L1": json.loads(r["labels_L1"]),
                    "L2": json.loads(r["labels_L2"]),
                    "L3": json.loads(r["labels_L3"]),
                },
                "anchor_date_raw": r.get("anchor_date_raw", ""),
            }
        )
    _write_jsonl(out_dir / "manifests" / "trimodal_manifest.jsonl", trimodal_manifest)

    # coverage report
    print("[cohort] WSI total:", len(wsi))
    print("[cohort] matched pathology:", matched_pathology)
    print("[cohort] matched patient index:", matched_patient)
    print("[cohort] matched lab:", len(cohort_trimodal))
    print("[cohort] output dir:", out_dir)


if __name__ == "__main__":
    main()
