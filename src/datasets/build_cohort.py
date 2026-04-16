import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .stain_utils import build_stain_vocab, dump_stain_vocab, normalize_h5s, standardize_pathology_id
from .xlsx_utils import read_sheet_rows


L1_RAW = [
    "Calcineurin Inhibitor Toxicity",
    "IFTA",
    "Normal Biopsy Or Nonspecific Changes",
    "Polyomavirus Nephropathy",
    "Rejection",
]
L2_RAW = [
    "AMR/MVI",
    "Calcineurin Inhibitor Toxicity",
    "IFTA",
    "Normal Biopsy Or Nonspecific Changes",
    "Polyomavirus Nephropathy",
    "TCMR",
]
L3_RAW = [
    "Active AMR",
    "Acute TCMR",
    "Calcineurin Inhibitor Toxicity",
    "Chronic and Chronic Active AMR",
    "IFTA",
    "Normal Biopsy Or Nonspecific Changes",
    "Other MVI and C4d",
    "Polyomavirus Nephropathy",
    "Suspicious (Borderline) For Acute TCMR",
]
NORMAL_NAME = "Normal Biopsy Or Nonspecific Changes"

L3_TO_L2 = {
    "Active AMR": "AMR/MVI",
    "Chronic and Chronic Active AMR": "AMR/MVI",
    "Other MVI and C4d": "AMR/MVI",
    "Acute TCMR": "TCMR",
    "Suspicious (Borderline) For Acute TCMR": "TCMR",
    "Calcineurin Inhibitor Toxicity": "Calcineurin Inhibitor Toxicity",
    "IFTA": "IFTA",
    "Polyomavirus Nephropathy": "Polyomavirus Nephropathy",
    "Normal Biopsy Or Nonspecific Changes": "Normal Biopsy Or Nonspecific Changes",
}
L2_TO_L1 = {
    "AMR/MVI": "Rejection",
    "TCMR": "Rejection",
    "Calcineurin Inhibitor Toxicity": "Calcineurin Inhibitor Toxicity",
    "IFTA": "IFTA",
    "Polyomavirus Nephropathy": "Polyomavirus Nephropathy",
    "Normal Biopsy Or Nonspecific Changes": "Normal Biopsy Or Nonspecific Changes",
}


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
                    "stains_raw": stains_raw,
                    "stains_norm": stains_norm,
                    "h5s_norm": h5s_norm,
                }
            )
    return out


def _normalize_id(raw: str) -> str:
    x = standardize_pathology_id(raw or "")
    x = re.sub(r"[^A-Z0-9]", "", x)
    if x.endswith("HE"):
        x = x[:-2]
    return x


def _id_candidates(raw: str) -> List[str]:
    b = _normalize_id(raw)
    cands = []
    for x in [b, b.lstrip("0"), b[:-1] if b.endswith("1") else b]:
        if x and x not in cands:
            cands.append(x)

    m = re.search(r"\d{2}S\d+", b)
    if m:
        x = m.group(0)
        if x and x not in cands:
            cands.append(x)
        if x.endswith("1"):
            y = x[:-1]
            if y and y not in cands:
                cands.append(y)
    return cands


def _resolve_id(raw: str, reference_ids: Set[str]) -> str:
    for c in _id_candidates(raw):
        if c in reference_ids:
            return c
    cs = _id_candidates(raw)
    return cs[0] if cs else ""


def _find_col_letter(header: Dict[str, str], candidates: List[str]) -> Optional[str]:
    pairs = [(k, (v or "").strip()) for k, v in header.items()]
    lower = {k: v.lower() for k, v in pairs}

    for c in candidates:
        t = c.strip()
        for k, v in pairs:
            if v == t:
                return k
    for c in candidates:
        t = c.strip().lower()
        for k, v in lower.items():
            if v == t:
                return k
    return None


def _get_cell(r: Dict[str, str], col: Optional[str]) -> str:
    if not col:
        return ""
    return str(r.get(col, "") or "").strip()


def _canonical_l3_token(token: str) -> Optional[str]:
    t = (token or "").strip()
    if not t:
        return None
    up = t.upper()

    if "SUSPICIOUS (BORDERLINE) FOR ACUTE TCMR" in up:
        return "Suspicious (Borderline) For Acute TCMR"
    # Historical choice in this project: Chronic Active TCMR merged into Acute TCMR.
    if "CHRONIC ACTIVE TCMR" in up or "CHRONIC TCMR" in up:
        return "Acute TCMR"
    if "ACUTE TCMR" in up:
        return "Acute TCMR"

    if "CHRONIC ACTIVE AMR" in up or "CHRONIC AMR" in up:
        return "Chronic and Chronic Active AMR"
    if "ACTIVE AMR" in up or "PROBABLE AMR" in up:
        return "Active AMR"

    if "MICROVASCULAR INFLAMMATION/INJURY (MVI)" in up:
        return "Other MVI and C4d"
    if "C4D STAINING WITHOUT EVIDENCE OF REJECTION" in up:
        return "Other MVI and C4d"

    if "CALCINEURIN INHIBITOR TOXICITY" in up:
        return "Calcineurin Inhibitor Toxicity"
    if "POLYOMAVIRUS NEPHROPATHY" in up:
        return "Polyomavirus Nephropathy"
    if "IFTA" in up:
        return "IFTA"
    if "NORMAL BIOPSY OR NONSPECIFIC CHANGES" in up:
        return "Normal Biopsy Or Nonspecific Changes"

    return None


def _drop_normal(vec: List[int], names: List[str]) -> List[int]:
    idx = names.index(NORMAL_NAME)
    return [v for i, v in enumerate(vec) if i != idx]


def _labels_from_gt2(gt2: str) -> Tuple[Optional[Dict[str, List[int]]], List[str]]:
    text = (gt2 or "").strip()
    if not text:
        return None, ["empty_gt2"]

    raw_tokens = [x.strip() for x in text.split("+") if x.strip()]
    mapped = set()
    unknown = []
    for t in raw_tokens:
        c = _canonical_l3_token(t)
        if c is None:
            unknown.append(t)
        else:
            mapped.add(c)

    if unknown:
        return None, unknown
    if not mapped:
        return None, ["empty_mapped_l3"]

    l3_raw = [1 if n in mapped else 0 for n in L3_RAW]
    l2_pos = {L3_TO_L2[n] for n in mapped}
    l2_raw = [1 if n in l2_pos else 0 for n in L2_RAW]
    l1_pos = {L2_TO_L1[n] for n in l2_pos}
    l1_raw = [1 if n in l1_pos else 0 for n in L1_RAW]

    return {
        "L1": _drop_normal(l1_raw, L1_RAW),
        "L2": _drop_normal(l2_raw, L2_RAW),
        "L3": _drop_normal(l3_raw, L3_RAW),
    }, []


def _load_xlsx_index(data_dir: Path, xlsx_name: str, sheet_name: str) -> Dict[str, Dict]:
    header, rows = read_sheet_rows(str(data_dir / xlsx_name), sheet_name)

    c_sample = _find_col_letter(header, ["病理号"])
    c_patient = _find_col_letter(header, ["患者主索引号"])
    c_age = _find_col_letter(header, ["年龄"])
    c_sex = _find_col_letter(header, ["性别"])
    c_tx = _find_col_letter(header, ["Date of Transplant"])
    c_follow = _find_col_letter(header, ["是否回访成功"])
    c_latest_date_scr = _find_col_letter(header, ["Latest Date of sCr"])
    c_latest_scr = _find_col_letter(header, ["Latest sCr"])
    c_apply_date = _find_col_letter(header, ["申请日期"])
    c_report_time = _find_col_letter(header, ["报告时间"])
    c_summary = _find_col_letter(header, ["检查结论"])
    c_gross = _find_col_letter(header, ["巨检所见"])
    c_structured_report = _find_col_letter(header, ["Structured Report"])
    c_clinical_notes = _find_col_letter(header, ["Clinical Notes and Gemini Response"])
    c_gt15 = _find_col_letter(header, ["GT1.5（肾内科诊断）"])
    c_microscopy = _find_col_letter(header, ["显微镜下所见"])
    c_special_stain_desc = _find_col_letter(header, ["特殊染色描述"])
    c_puncture_note = _find_col_letter(header, ["备注_穿刺GT2_1"])
    c_general_note = _find_col_letter(header, ["备注"])
    c_surv_p = _find_col_letter(header, ["Patient Survival", "患者是否存活"])
    c_surv_g = _find_col_letter(header, ["Graft Survival", "功能是否尚存"])
    c_death = _find_col_letter(header, ["Date of Death", "死亡时间"])
    c_graft_loss = _find_col_letter(header, ["Date of Graft Loss", "失功时间"])
    c_gt2 = _find_col_letter(header, ["GT2"])
    c_height = _find_col_letter(header, ["身高"])
    c_weight = _find_col_letter(header, ["体重"])
    c_bmi = _find_col_letter(header, ["BMI"])
    c_primary_clinical = _find_col_letter(header, ["Primary Kidney Disease Clinical Syndrome"])
    c_primary_path_dx = _find_col_letter(header, ["Primary Kidney Disease Pathological Dx"])
    c_dialysis_method = _find_col_letter(header, ["Dialysis Method"])
    c_dialysis_history = _find_col_letter(header, ["Dialysis History"])
    c_prev_tx_n = _find_col_letter(header, ["Number of Previous Transplants"])
    c_donor_type = _find_col_letter(header, ["Donor Type"])
    c_donor_blood = _find_col_letter(header, ["Donor Blood Type"])
    c_recip_blood = _find_col_letter(header, ["Recipient Blood Group"])
    c_cold_ischemia = _find_col_letter(header, ["Cold Ischemia Time"])
    c_warm_ischemia = _find_col_letter(header, ["Warm Ischemia Time"])
    c_induction = _find_col_letter(header, ["Induction Therapy"])
    c_maint_init = _find_col_letter(header, ["Initial Maintenance Regimen"])
    c_pra1 = _find_col_letter(header, ["PRA Type I"])
    c_pra2 = _find_col_letter(header, ["PRA Type II"])
    c_mic_ab = _find_col_letter(header, ["MIC Ab"])
    c_hlaa1 = _find_col_letter(header, ["hlaa1"])
    c_hlaa2 = _find_col_letter(header, ["hlaa2"])
    c_hlab1 = _find_col_letter(header, ["hlab1"])
    c_hlab2 = _find_col_letter(header, ["hlab2"])
    c_hladr1 = _find_col_letter(header, ["hladr1"])
    c_hladr2 = _find_col_letter(header, ["hladr2"])
    c_ctc_t = _find_col_letter(header, ["CTC T cell"])
    c_ctc_b = _find_col_letter(header, ["CTC B cell"])
    c_post_tx_rej = _find_col_letter(header, ["Post Tx Rejection"])
    c_post_tx_rej_cls = _find_col_letter(header, ["Post Tx Rejection Classification"])
    c_rej_treat = _find_col_letter(header, ["排斥反应后的治疗"])
    c_mpa = _find_col_letter(header, ["MPA"])
    c_maint_regimen = _find_col_letter(header, ["Maintanence Immunosuppression Regimen"])
    c_cni_dose = _find_col_letter(header, ["CNI（莫司）及剂量"])
    c_mpa_dose = _find_col_letter(header, ["MPA（麦考）及剂量"])
    c_steroid_dose = _find_col_letter(header, ["激素及剂量"])
    c_cause_death = _find_col_letter(header, ["Cause of Death"])
    c_cause_graft_loss = _find_col_letter(header, ["Cause of Graft Loss"])
    c_treat_after_graft_loss = _find_col_letter(header, ["失功后治疗"])
    c_tumor = _find_col_letter(header, ["Post Tx Tumor"])
    c_tumor_control = _find_col_letter(header, ["肿瘤控制情况"])
    c_proteinuria = _find_col_letter(header, ["Post Tx Proteinuria"])
    c_proteinuria_ctrl = _find_col_letter(header, ["蛋白尿控制情况"])
    c_htn = _find_col_letter(header, ["HTN"])
    c_hyperlip = _find_col_letter(header, ["Hyperlipidemia"])
    c_dm = _find_col_letter(header, ["DM"])
    c_hyperuric = _find_col_letter(header, ["Hyperuricemia"])
    c_other_comorb = _find_col_letter(header, ["其他"])
    c_recent_followup = _find_col_letter(header, ["近期是否来复诊"])
    c_serial_bx = _find_col_letter(header, ["Serial Bx"])
    c_day0_bx = _find_col_letter(header, ["Day Zero Bx"])
    c_day0_pid = _find_col_letter(header, ["Day Zero 病理号"])

    if c_sample is None or c_gt2 is None:
        raise ValueError(f"Cannot find required columns in {xlsx_name} sheet {sheet_name}: 病理号/GT2")

    idx = {}
    for r in rows:
        sid = _normalize_id(_get_cell(r, c_sample))
        if not sid:
            continue
        if sid in idx:
            continue
        idx[sid] = {
            "sample_id": sid,
            "patient_index": _get_cell(r, c_patient),
            "age": _get_cell(r, c_age),
            "sex": _get_cell(r, c_sex),
            "date_of_transplant": _get_cell(r, c_tx),
            "followup_success": _get_cell(r, c_follow),
            "latest_date_scr": _get_cell(r, c_latest_date_scr),
            "latest_scr": _get_cell(r, c_latest_scr),
            "application_date": _get_cell(r, c_apply_date),
            "report_time": _get_cell(r, c_report_time),
            "pathology_summary": _get_cell(r, c_summary),
            "gross_findings": _get_cell(r, c_gross),
            "structured_report": _get_cell(r, c_structured_report),
            "clinical_notes": _get_cell(r, c_clinical_notes),
            "gt15_nephrology_dx": _get_cell(r, c_gt15),
            "microscopic_findings": _get_cell(r, c_microscopy),
            "special_stain_description": _get_cell(r, c_special_stain_desc),
            "puncture_note": _get_cell(r, c_puncture_note),
            "general_note": _get_cell(r, c_general_note),
            "patient_survival": _get_cell(r, c_surv_p),
            "graft_survival": _get_cell(r, c_surv_g),
            "date_of_death": _get_cell(r, c_death),
            "date_of_graft_loss": _get_cell(r, c_graft_loss),
            "gt2": _get_cell(r, c_gt2),
            "height": _get_cell(r, c_height),
            "weight": _get_cell(r, c_weight),
            "bmi": _get_cell(r, c_bmi),
            "primary_kidney_clinical_syndrome": _get_cell(r, c_primary_clinical),
            "primary_kidney_pathological_dx": _get_cell(r, c_primary_path_dx),
            "dialysis_method": _get_cell(r, c_dialysis_method),
            "dialysis_history": _get_cell(r, c_dialysis_history),
            "num_previous_transplants": _get_cell(r, c_prev_tx_n),
            "donor_type": _get_cell(r, c_donor_type),
            "donor_blood_type": _get_cell(r, c_donor_blood),
            "recipient_blood_group": _get_cell(r, c_recip_blood),
            "cold_ischemia_time": _get_cell(r, c_cold_ischemia),
            "warm_ischemia_time": _get_cell(r, c_warm_ischemia),
            "induction_therapy": _get_cell(r, c_induction),
            "initial_maintenance_regimen": _get_cell(r, c_maint_init),
            "pra_type_i": _get_cell(r, c_pra1),
            "pra_type_ii": _get_cell(r, c_pra2),
            "mic_ab": _get_cell(r, c_mic_ab),
            "hlaa1": _get_cell(r, c_hlaa1),
            "hlaa2": _get_cell(r, c_hlaa2),
            "hlab1": _get_cell(r, c_hlab1),
            "hlab2": _get_cell(r, c_hlab2),
            "hladr1": _get_cell(r, c_hladr1),
            "hladr2": _get_cell(r, c_hladr2),
            "ctc_t_cell": _get_cell(r, c_ctc_t),
            "ctc_b_cell": _get_cell(r, c_ctc_b),
            "post_tx_rejection": _get_cell(r, c_post_tx_rej),
            "post_tx_rejection_classification": _get_cell(r, c_post_tx_rej_cls),
            "rejection_treatment": _get_cell(r, c_rej_treat),
            "mpa_note": _get_cell(r, c_mpa),
            "maintenance_immunosuppression_regimen": _get_cell(r, c_maint_regimen),
            "cni_dose": _get_cell(r, c_cni_dose),
            "mpa_dose": _get_cell(r, c_mpa_dose),
            "steroid_dose": _get_cell(r, c_steroid_dose),
            "cause_of_death": _get_cell(r, c_cause_death),
            "cause_of_graft_loss": _get_cell(r, c_cause_graft_loss),
            "treatment_after_graft_loss": _get_cell(r, c_treat_after_graft_loss),
            "post_tx_tumor": _get_cell(r, c_tumor),
            "tumor_control": _get_cell(r, c_tumor_control),
            "post_tx_proteinuria": _get_cell(r, c_proteinuria),
            "proteinuria_control": _get_cell(r, c_proteinuria_ctrl),
            "comorb_htn": _get_cell(r, c_htn),
            "comorb_hyperlipidemia": _get_cell(r, c_hyperlip),
            "comorb_dm": _get_cell(r, c_dm),
            "comorb_hyperuricemia": _get_cell(r, c_hyperuric),
            "comorb_other": _get_cell(r, c_other_comorb),
            "recent_followup": _get_cell(r, c_recent_followup),
            "serial_bx": _get_cell(r, c_serial_bx),
            "day_zero_bx": _get_cell(r, c_day0_bx),
            "day_zero_pathology_id": _get_cell(r, c_day0_pid),
        }
        # Banff细粒度评分（若存在）
        for name in ["i", "t", "t（10x视野（中倍），≤3处小管炎，打1*）", "v", "g", "ptc（仅考虑PMN）", "ptc（单个核）", "C4d.1", "C4d（备注为假阳性）", "ci", "ct", "cv", "cg", "mm", "ah", "aah", "ti", "i-IFTA", "t-IFTA", "pvl"]:
            col = _find_col_letter(header, [name])
            if col is None:
                continue
            key = "banff_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            idx[sid][key] = _get_cell(r, col)

        # 染色资产统计：所有 *_Count 列
        for col_letter, col_name in header.items():
            cname = (col_name or "").strip()
            if not cname.endswith("_Count"):
                continue
            key = "stain_count_" + re.sub(r"[^a-z0-9]+", "_", cname[:-6].lower()).strip("_")
            idx[sid][key] = _get_cell(r, col_letter)
    return idx


def _load_slice_removal_ids(data_dir: Path, xlsx_name: str, sheet_name: str) -> Set[str]:
    header, rows = read_sheet_rows(str(data_dir / xlsx_name), sheet_name)
    ids = set()
    # This sheet sometimes lacks true header row. Include both header A and row A.
    ids.add(_normalize_id(header.get("A", "")))
    for r in rows:
        ids.add(_normalize_id(r.get("A", "")))
    ids = {x for x in ids if x}
    return ids


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
    p.add_argument("--xlsx-name", default="pathology_reports_v7.3_with_GT2__20260414  Censored.xlsx")
    p.add_argument("--main-sheet", default="移植肾汇总-筛选后")
    p.add_argument("--slice-note-sheet", default="切片备注")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    wsi = _load_wsi_samples(data_dir)
    xlsx_idx = _load_xlsx_index(data_dir, args.xlsx_name, args.main_sheet)
    xlsx_ids = set(xlsx_idx.keys())
    slice_removed_ids = _load_slice_removal_ids(data_dir, args.xlsx_name, args.slice_note_sheet)
    lab_patients = _load_lab_patient_set(data_dir / "Merged_Lab_Results_Final.csv")

    stain_vocab = build_stain_vocab([s for r in wsi for s in r["stains_norm"]])
    dump_stain_vocab(out_dir / "stain_vocab.json", stain_vocab)

    cohort_wsi = []
    kept_wsi = []
    drop_stats = Counter()
    unknown_gt2_counter = Counter()
    for r in wsi:
        sid = _resolve_id(r["sample_id"], xlsx_ids)
        if not sid or sid not in xlsx_idx:
            drop_stats["missing_in_xlsx"] += 1
            continue
        if sid in slice_removed_ids:
            drop_stats["removed_by_slice_note"] += 1
            continue

        xr = xlsx_idx[sid]
        labels, unknown = _labels_from_gt2(xr.get("gt2", ""))
        if labels is None:
            drop_stats["removed_by_gt2_unmapped"] += 1
            for t in unknown:
                unknown_gt2_counter[t] += 1
            continue

        kept_wsi.append(
            {
                "sample_id": sid,
                "split": r["split"],
                "labels": labels,
                "available_stains": ";".join(r["stains_norm"]),
                "h5_paths_json": json.dumps(r["h5s_norm"], ensure_ascii=False),
            }
        )

    for r in kept_wsi:
        cohort_wsi.append(
            {
                "sample_id": r["sample_id"],
                "split": r["split"],
                "labels_L1": json.dumps(r["labels"]["L1"]),
                "labels_L2": json.dumps(r["labels"]["L2"]),
                "labels_L3": json.dumps(r["labels"]["L3"]),
                "available_stains": r["available_stains"],
                "h5_paths_json": r["h5_paths_json"],
            }
        )

    cohort_tabular = []
    matched_patient = 0
    for r in kept_wsi:
        sid = r["sample_id"]
        xr = xlsx_idx[sid]
        pid = xr["patient_index"]
        if pid:
            matched_patient += 1
        row = {
            "sample_id": sid,
            "split": r["split"],
            "patient_index": pid,
            "labels_L1": json.dumps(r["labels"]["L1"]),
            "labels_L2": json.dumps(r["labels"]["L2"]),
            "labels_L3": json.dumps(r["labels"]["L3"]),
            "available_stains": r["available_stains"],
            "h5_paths_json": r["h5_paths_json"],
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

    tab_fields = []
    if cohort_tabular:
        head = [
            "sample_id",
            "split",
            "patient_index",
            "labels_L1",
            "labels_L2",
            "labels_L3",
            "available_stains",
            "h5_paths_json",
        ]
        tail = ["anchor_date_policy", "anchor_date_raw"]
        seen = set()
        for k in head:
            if k in cohort_tabular[0]:
                tab_fields.append(k)
                seen.add(k)
        for r in cohort_tabular:
            for k in r.keys():
                if k in seen or k in tail:
                    continue
                tab_fields.append(k)
                seen.add(k)
        for k in tail:
            if k in cohort_tabular[0] and k not in seen:
                tab_fields.append(k)
                seen.add(k)

    _write_csv(out_dir / "cohort_tabular.csv", cohort_tabular, tab_fields)
    trimodal_fields = tab_fields + (["has_lab"] if "has_lab" not in tab_fields else [])
    _write_csv(out_dir / "cohort_trimodal.csv", cohort_trimodal, trimodal_fields)

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
    print("[cohort] WSI kept:", len(kept_wsi))
    print("[cohort] dropped missing_in_xlsx:", int(drop_stats["missing_in_xlsx"]))
    print("[cohort] dropped removed_by_slice_note:", int(drop_stats["removed_by_slice_note"]))
    print("[cohort] dropped removed_by_gt2_unmapped:", int(drop_stats["removed_by_gt2_unmapped"]))
    print("[cohort] matched patient index:", matched_patient)
    print("[cohort] matched lab:", len(cohort_trimodal))
    print("[cohort] slice-note unique ids:", len(slice_removed_ids))
    if unknown_gt2_counter:
        print("[cohort] top unmapped GT2 tokens:")
        for k, v in unknown_gt2_counter.most_common(20):
            print("  ", v, k)
    print("[cohort] output dir:", out_dir)


if __name__ == "__main__":
    main()
