#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.xlsx_utils import read_sheet_rows
from src.utils.io import try_parse_datetime


def _count_json_array(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        arr = json.load(f)
    return len(arr)


def _read_jsonl(path: Path) -> List[Dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _norm_yes_no(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"yes", "y", "1", "true", "是", "存活", "成功"}:
        return "yes"
    if s in {"no", "n", "0", "false", "否", "死亡", "失败"}:
        return "no"
    return "missing"


def _split_counts(rows: List[Dict], key: str = "split") -> Dict[str, int]:
    c = Counter()
    for r in rows:
        c[(r.get(key, "") or "").strip()] += 1
    return dict(c)


def _feature_groups(fieldnames: List[str]) -> Dict[str, int]:
    meta = {"sample_id", "split", "patient_index", "anchor_timestamp", "anchor_date_policy", "anchor_date_raw"}
    feat = [c for c in fieldnames if c not in meta]

    def group(c: str) -> str:
        for pf in [
            "structured_",
            "banff_",
            "timeline_",
            "treatment_",
            "text_stat_",
            "text_kw_",
            "text_hash_",
            "qwen_global_",
            "medbert_local_",
        ]:
            if c.startswith(pf):
                return pf.rstrip("_")
        if c.startswith("text_section_present_"):
            return "text_section_present"
        if c.startswith("raw_"):
            return "raw"
        return "other"

    return dict(Counter(group(c) for c in feat))


def _xlsx_keyword_hits(xlsx_path: Path, sheet_name: str) -> Dict[str, List[str]]:
    header, _ = read_sheet_rows(str(xlsx_path), sheet_name)
    cols = [str(v or "").strip().replace("\n", " ") for _, v in sorted(header.items())]
    keywords = [
        "预后",
        "肌酐",
        "sCr",
        "creatinine",
        "survival",
        "death",
        "graft",
        "随访",
        "失功",
        "移植",
    ]
    hits = []
    for c in cols:
        low = c.lower()
        if any(k.lower() in low for k in keywords):
            hits.append(c)
    return {"columns": cols, "keyword_hits": hits}


def main():
    p = argparse.ArgumentParser("Audit multimodal prognosis data readiness")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--xlsx-name", default="pathology_reports_v7.3_with_GT2__20260414  Censored.xlsx")
    p.add_argument("--xlsx-sheet", default="移植肾汇总-筛选后")
    p.add_argument("--extra-xlsx", default="复旦数据库_reformatted.xlsx")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    proc = Path(args.processed_dir)

    report = {}

    # Raw WSI pools
    report["raw_wsi"] = {
        "train_ms": _count_json_array(data_dir / "train_ms.json"),
        "val_ms": _count_json_array(data_dir / "val_ms.json"),
        "test_ms": _count_json_array(data_dir / "test_ms.json"),
    }
    report["raw_wsi"]["total"] = sum(report["raw_wsi"].values())

    # Cohorts/manifests
    cohort_wsi = _read_csv(proc / "cohort_wsi.csv")
    cohort_tab = _read_csv(proc / "cohort_tabular.csv")
    cohort_tri = _read_csv(proc / "cohort_trimodal.csv")
    report["cohorts"] = {
        "cohort_wsi": len(cohort_wsi),
        "cohort_tabular": len(cohort_tab),
        "cohort_trimodal": len(cohort_tri),
        "cohort_trimodal_split": _split_counts(cohort_tri),
    }

    trimodal_manifest = _read_jsonl(proc / "manifests" / "trimodal_manifest.jsonl")
    report["manifests"] = {
        "train_manifest": len(_read_jsonl(proc / "manifests" / "train_manifest.jsonl")),
        "val_manifest": len(_read_jsonl(proc / "manifests" / "val_manifest.jsonl")),
        "test_manifest": len(_read_jsonl(proc / "manifests" / "test_manifest.jsonl")),
        "trimodal_manifest": len(trimodal_manifest),
        "trimodal_split": _split_counts(trimodal_manifest),
    }

    # Stain coverage
    stain_count = Counter()
    n_stains = []
    for r in trimodal_manifest:
        h5s = r.get("h5s", {}) or {}
        n_stains.append(len(h5s))
        for s in h5s.keys():
            stain_count[s] += 1
    if n_stains:
        n_stains_sorted = sorted(n_stains)
        mid = n_stains_sorted[len(n_stains_sorted) // 2]
        report["stains"] = {
            "unique_stains": len(stain_count),
            "top_20": stain_count.most_common(20),
            "per_sample_min_median_max": [min(n_stains), mid, max(n_stains)],
        }
    else:
        report["stains"] = {}

    # Tabular/lab rows and dimension
    tab_path = proc / "tabular_features.csv"
    lab_path = proc / "lab_features.csv"
    tab_rows = _read_csv(tab_path)
    lab_rows = _read_csv(lab_path)
    tab_fields = list(tab_rows[0].keys()) if tab_rows else []
    lab_fields = list(lab_rows[0].keys()) if lab_rows else []
    tab_meta = {"sample_id", "split", "patient_index", "anchor_timestamp", "anchor_date_policy", "anchor_date_raw"}
    lab_meta = {"sample_id", "patient_index", "anchor_timestamp"}
    report["features"] = {
        "tabular_rows": len(tab_rows),
        "lab_rows": len(lab_rows),
        "tabular_dim": len([c for c in tab_fields if c not in tab_meta]),
        "lab_dim": len([c for c in lab_fields if c not in lab_meta]),
        "tabular_groups": _feature_groups(tab_fields),
    }

    # Feature availability by split
    tab_ids = {r.get("sample_id", "") for r in tab_rows}
    lab_ids = {r.get("sample_id", "") for r in lab_rows}
    split_ids = defaultdict(list)
    for r in trimodal_manifest:
        split_ids[r.get("split", "")].append(r.get("sample_id", ""))
    split_cov = {}
    for sp, sids in split_ids.items():
        n = len(sids)
        split_cov[sp] = {
            "tabular_hit": f"{sum(1 for x in sids if x in tab_ids)}/{n}",
            "lab_hit": f"{sum(1 for x in sids if x in lab_ids)}/{n}",
        }
    report["feature_coverage_by_split"] = split_cov

    # Prognosis columns and date coverage in cohort_trimodal
    prog = {
        "patient_survival": Counter(),
        "graft_survival": Counter(),
        "followup_success": Counter(),
    }
    date_cov = {"date_of_transplant": 0, "date_of_death": 0, "date_of_graft_loss": 0, "latest_date_scr": 0}
    followup_days = []
    for r in cohort_tri:
        for k in prog.keys():
            prog[k][_norm_yes_no(r.get(k, ""))] += 1
        dt_tx = try_parse_datetime(r.get("date_of_transplant", ""))
        dt_death = try_parse_datetime(r.get("date_of_death", ""))
        dt_loss = try_parse_datetime(r.get("date_of_graft_loss", ""))
        dt_scr = try_parse_datetime(r.get("latest_date_scr", ""))
        if dt_tx is not None:
            date_cov["date_of_transplant"] += 1
        if dt_death is not None:
            date_cov["date_of_death"] += 1
        if dt_loss is not None:
            date_cov["date_of_graft_loss"] += 1
        if dt_scr is not None:
            date_cov["latest_date_scr"] += 1
        endpoint = dt_death or dt_loss or dt_scr
        if dt_tx is not None and endpoint is not None and endpoint >= dt_tx:
            followup_days.append((endpoint - dt_tx).days)

    followup = {}
    if followup_days:
        s = sorted(followup_days)
        followup = {
            "n": len(s),
            "min_days": s[0],
            "median_days": s[len(s) // 2],
            "max_days": s[-1],
        }
    report["prognosis_ready"] = {
        "n_trimodal": len(cohort_tri),
        "patient_survival": dict(prog["patient_survival"]),
        "graft_survival": dict(prog["graft_survival"]),
        "followup_success": dict(prog["followup_success"]),
        "date_coverage": date_cov,
        "proxy_followup_days": followup,
    }

    # Potential leakage columns for prognosis tasks
    leak_markers = [
        "patient_survival",
        "graft_survival",
        "death",
        "graft_loss",
        "post_tx_",
        "treatment_after_graft_loss",
    ]
    leak_cols = []
    for c in tab_fields:
        low = c.lower()
        if any(m in low for m in leak_markers):
            leak_cols.append(c)
    report["potential_label_leakage_columns"] = leak_cols

    # Xlsx keyword hits
    xlsx_main = data_dir / args.xlsx_name
    if xlsx_main.exists():
        report["xlsx_main"] = _xlsx_keyword_hits(xlsx_main, args.xlsx_sheet)
        report["xlsx_main"]["n_columns"] = len(report["xlsx_main"]["columns"])
    xlsx_extra = data_dir / args.extra_xlsx
    if xlsx_extra.exists():
        # use first sheet for quick scan
        header, _ = read_sheet_rows(str(xlsx_extra), "Sheet1")
        cols = [str(v or "").strip().replace("\n", " ") for _, v in sorted(header.items())]
        hits = []
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in ["survival", "death", "graft", "scr", "creatinine", "随访", "失功", "肌酐"]):
                hits.append(c)
        report["xlsx_extra"] = {"n_columns": len(cols), "keyword_hits": hits}

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
