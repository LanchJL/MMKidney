import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.io import try_parse_datetime, write_json

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

STRUCT_NUM_FIELDS = [
    "age",
    "height",
    "weight",
    "bmi",
    "latest_scr",
    "num_previous_transplants",
    "cold_ischemia_time",
    "warm_ischemia_time",
    "ctc_t_cell",
    "ctc_b_cell",
]

STRUCT_CAT_FIELDS = [
    "sex",
    "followup_success",
    "primary_kidney_clinical_syndrome",
    "primary_kidney_pathological_dx",
    "dialysis_method",
    "dialysis_history",
    "donor_type",
    "donor_blood_type",
    "recipient_blood_group",
    "induction_therapy",
    "initial_maintenance_regimen",
    "pra_type_i",
    "pra_type_ii",
    "mic_ab",
    "hlaa1",
    "hlaa2",
    "hlab1",
    "hlab2",
    "hladr1",
    "hladr2",
    "post_tx_rejection",
    "post_tx_rejection_classification",
    "post_tx_tumor",
    "post_tx_proteinuria",
    "comorb_htn",
    "comorb_hyperlipidemia",
    "comorb_dm",
    "comorb_hyperuricemia",
    "recent_followup",
    "serial_bx",
    "day_zero_bx",
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

BANFF_FIELDS = [
    "banff_i",
    "banff_t",
    "banff_t_10x_3_1",
    "banff_v",
    "banff_g",
    "banff_ptc_pmn",
    "banff_ptc",
    "banff_c4d_1",
    "banff_c4d",
    "banff_ci",
    "banff_ct",
    "banff_cv",
    "banff_cg",
    "banff_mm",
    "banff_ah",
    "banff_aah",
    "banff_ti",
    "banff_i_ifta",
    "banff_t_ifta",
    "banff_pvl",
]

TEXT_KEYWORDS = {
    "rejection": ["rejection", "排斥", "abmr", "amr", "tcmr", "borderline"],
    "ifta": ["ifta", "纤维化", "萎缩", "atrophy", "fibrosis"],
    "viral": ["bk", "polyoma", "sv40", "病毒"],
    "c4d": ["c4d"],
    "severity": ["mild", "moderate", "severe", "轻", "中", "重", "focal", "diffuse", "弥漫"],
    "negation": ["未见", "不支持", "否认", "无明显", "negative for", "no evidence"],
}


def _safe_float(x: str) -> Optional[float]:
    s = (x or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        pass

    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _parse_binary(x: str) -> Optional[float]:
    s = (x or "").strip().lower()
    if not s:
        return None
    pos = {"yes", "y", "1", "true", "positive", "是", "有", "阳性", "存活", "成功"}
    neg = {"no", "n", "0", "false", "negative", "否", "无", "阴性", "死亡", "失败"}
    if s in pos:
        return 1.0
    if s in neg:
        return 0.0
    if "yes" in s or "阳" in s or "存活" in s or "成功" in s:
        return 1.0
    if "no" in s or "阴" in s or "死亡" in s or "失败" in s:
        return 0.0
    return None


def _to_float_or_zero(v: Optional[float]) -> float:
    return 0.0 if v is None else float(v)


def _to_mask(v: Optional[float]) -> float:
    return 0.0 if v is None else 1.0


def _build_vocab(rows: List[Dict[str, str]], key: str, min_freq: int = 1) -> Dict[str, int]:
    cnt = {}
    for r in rows:
        x = (r.get(key, "") or "").strip()
        if not x:
            continue
        cnt[x] = cnt.get(x, 0) + 1
    vals = sorted([x for x, n in cnt.items() if n >= min_freq])
    return {v: i for i, v in enumerate(vals)}


def _days_diff(a, b):
    if a is None or b is None:
        return None
    return float((a - b).total_seconds() / 86400.0)


def _clean_text(s: str) -> str:
    x = (s or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def _count_cjk(s: str) -> int:
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")


def _stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _hash_text_ngrams(text: str, dim: int, n_min: int = 2, n_max: int = 4) -> List[float]:
    if dim <= 0:
        return []
    txt = _clean_text(text)
    if not txt:
        return [0.0] * dim
    dense = [0.0] * dim
    total = 0
    for n in range(n_min, n_max + 1):
        if len(txt) < n:
            continue
        for i in range(len(txt) - n + 1):
            gram = txt[i : i + n]
            h = _stable_hash_int(gram)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            dense[idx] += sign
            total += 1
    if total <= 0:
        return dense
    norm = sum(v * v for v in dense) ** 0.5
    if norm > 0:
        dense = [v / norm for v in dense]
    return dense


def _text_stats(text: str) -> Dict[str, float]:
    txt = _clean_text(text)
    if not txt:
        return {
            "present": 0.0,
            "len_char": 0.0,
            "len_token": 0.0,
            "digit_ratio": 0.0,
            "alpha_ratio": 0.0,
            "cjk_ratio": 0.0,
        }
    n_char = float(len(txt))
    n_tok = float(len([t for t in txt.split(" ") if t]))
    n_digit = float(sum(ch.isdigit() for ch in txt))
    n_alpha = float(sum(ch.isalpha() for ch in txt))
    n_cjk = float(_count_cjk(txt))
    return {
        "present": 1.0,
        "len_char": n_char,
        "len_token": n_tok,
        "digit_ratio": n_digit / n_char if n_char > 0 else 0.0,
        "alpha_ratio": n_alpha / n_char if n_char > 0 else 0.0,
        "cjk_ratio": n_cjk / n_char if n_char > 0 else 0.0,
    }


def _text_keyword_counts(text: str) -> Dict[str, float]:
    txt = _clean_text(text)
    out = {}
    for k, kws in TEXT_KEYWORDS.items():
        c = 0
        for w in kws:
            c += txt.count(w.lower())
        out[k] = float(c)
    return out


def _parse_banff_scalar(x: str) -> Optional[float]:
    s = (x or "").strip()
    if not s:
        return None
    # capture leading 0-3 like values with possible star suffix
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _read_dense_embedding_map(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        sid = (r.get("sample_id", "") or "").strip()
        if sid:
            out[sid] = r
    return out


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _try_write_parquet(rows: List[Dict], path: Path) -> bool:
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser("Build tabular features with clinicopath/text groups")
    p.add_argument("--cohort-tabular", default="data/processed/cohort_tabular.csv")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--text-hash-dim", type=int, default=64)
    p.add_argument("--text-ngram-min", type=int, default=2)
    p.add_argument("--text-ngram-max", type=int, default=4)
    p.add_argument("--qwen-embedding-csv", default="")
    p.add_argument("--medbert-embedding-csv", default="")
    args = p.parse_args()

    in_path = Path(args.cohort_tabular)
    out_dir = Path(args.out_dir)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    vocab = {}
    for k in STRUCT_CAT_FIELDS:
        if any((r.get(k, "") or "").strip() for r in rows):
            vocab[k] = _build_vocab(rows, k)

    active_text_fields = [k for k in TEXT_FIELDS if any((r.get(k, "") or "").strip() for r in rows)]
    qwen_map = _read_dense_embedding_map(args.qwen_embedding_csv)
    medbert_map = _read_dense_embedding_map(args.medbert_embedding_csv)

    meta = {
        "categorical_vocab": vocab,
        "active_text_fields": active_text_fields,
        "text_hash_dim": int(args.text_hash_dim),
        "text_ngram_min": int(args.text_ngram_min),
        "text_ngram_max": int(args.text_ngram_max),
        "qwen_embedding_csv": args.qwen_embedding_csv,
        "medbert_embedding_csv": args.medbert_embedding_csv,
    }
    write_json(out_dir / "tabular_vocab.json", meta)

    out_rows = []
    for r in rows:
        rec = {
            "sample_id": r.get("sample_id", ""),
            "split": r.get("split", ""),
            "patient_index": r.get("patient_index", ""),
            "anchor_date_policy": r.get("anchor_date_policy", "application_date_first_else_report_time"),
            "anchor_date_raw": r.get("anchor_date_raw", ""),
        }

        anchor = try_parse_datetime(r.get("anchor_date_raw", ""))
        dt_tx = try_parse_datetime(r.get("date_of_transplant", ""))
        dt_apply = try_parse_datetime(r.get("application_date", ""))
        dt_report = try_parse_datetime(r.get("report_time", ""))
        dt_scr = try_parse_datetime(r.get("latest_date_scr", ""))
        dt_death = try_parse_datetime(r.get("date_of_death", ""))
        dt_graft_loss = try_parse_datetime(r.get("date_of_graft_loss", ""))

        rec["anchor_timestamp"] = anchor.isoformat(sep=" ") if anchor else ""

        for k in STRUCT_NUM_FIELDS:
            v = _safe_float(r.get(k, ""))
            rec[f"structured_num_{k}"] = _to_float_or_zero(v)
            rec[f"structured_num_{k}_mask"] = _to_mask(v)

        for k, vm in vocab.items():
            x = (r.get(k, "") or "").strip()
            code = float(vm[x]) if x in vm else None
            rec[f"structured_cat_{k}_code"] = _to_float_or_zero(code)
            rec[f"structured_cat_{k}_mask"] = _to_mask(code)

        # binary comorb/survival style fields
        for k in ["patient_survival", "graft_survival", "post_tx_tumor", "post_tx_proteinuria", "comorb_htn", "comorb_hyperlipidemia", "comorb_dm", "comorb_hyperuricemia", "serial_bx", "day_zero_bx"]:
            b = _parse_binary(r.get(k, ""))
            rec[f"structured_bin_{k}"] = _to_float_or_zero(b)
            rec[f"structured_bin_{k}_mask"] = _to_mask(b)

        # timeline
        t_features = {
            "tx_to_report_days": _days_diff(dt_report, dt_tx),
            "apply_to_report_days": _days_diff(dt_report, dt_apply),
            "latest_scr_to_report_days": _days_diff(dt_report, dt_scr),
            "latest_scr_to_anchor_days": _days_diff(anchor, dt_scr),
            "death_to_anchor_days": _days_diff(anchor, dt_death),
            "graft_loss_to_anchor_days": _days_diff(anchor, dt_graft_loss),
        }
        for k, v in t_features.items():
            rec[f"timeline_{k}"] = _to_float_or_zero(v)
            rec[f"timeline_{k}_mask"] = _to_mask(v)

        # treatment
        treat_text = " ".join((r.get(k, "") or "") for k in TREATMENT_FIELDS)
        treat_clean = _clean_text(treat_text)
        treat_kws = {
            "cni": ["tac", "cyclospor", "莫司", "cni"],
            "mpa": ["mpa", "麦考", "mmf", "mycophen"],
            "steroid": ["steroid", "pred", "激素"],
            "antibody": ["atg", "ritux", "抗体", "免疫球蛋白"],
            "plasmapheresis": ["plasmapheresis", "血浆置换"],
        }
        rec["treatment_any_present"] = 1.0 if treat_clean else 0.0
        rec["treatment_any_present_mask"] = 1.0
        for k, kws in treat_kws.items():
            c = 0
            for w in kws:
                c += treat_clean.count(w)
            rec[f"treatment_kw_{k}"] = float(c)
            rec[f"treatment_kw_{k}_mask"] = 1.0 if treat_clean else 0.0

        # Banff
        for k in BANFF_FIELDS:
            v = _parse_banff_scalar(r.get(k, ""))
            rec[k] = _to_float_or_zero(v)
            rec[f"{k}_mask"] = _to_mask(v)

        # Banff axes
        def _g(keys):
            vals = [rec.get(x, 0.0) for x in keys]
            masks = [rec.get(f"{x}_mask", 0.0) for x in keys]
            if sum(masks) <= 0:
                return 0.0, 0.0
            return float(sum(vals)), 1.0

        axes = {
            "inflammation": ["banff_i", "banff_t", "banff_ti", "banff_i_ifta", "banff_t_ifta"],
            "microvascular": ["banff_g", "banff_ptc", "banff_ptc_pmn"],
            "chronic_damage": ["banff_ci", "banff_ct", "banff_cv", "banff_cg", "banff_mm"],
            "vascular": ["banff_v", "banff_ah", "banff_aah"],
            "viral": ["banff_pvl"],
        }
        for ax, keys in axes.items():
            vv, mm = _g(keys)
            rec[f"banff_axis_{ax}"] = vv
            rec[f"banff_axis_{ax}_mask"] = mm

        # text features
        text_parts = []
        for k in active_text_fields:
            x = (r.get(k, "") or "").strip()
            if x:
                text_parts.append(f"[{k}] {x}")
            rec[f"text_section_present_{k}"] = 1.0 if x else 0.0

        text_blob = " [SEP] ".join(text_parts)
        text_stat = _text_stats(text_blob)
        rec["text_stat_present"] = text_stat["present"]
        rec["text_stat_len_char"] = text_stat["len_char"]
        rec["text_stat_len_token"] = text_stat["len_token"]
        rec["text_stat_digit_ratio"] = text_stat["digit_ratio"]
        rec["text_stat_alpha_ratio"] = text_stat["alpha_ratio"]
        rec["text_stat_cjk_ratio"] = text_stat["cjk_ratio"]

        kw = _text_keyword_counts(text_blob)
        for k, v in kw.items():
            rec[f"text_kw_{k}"] = float(v)

        text_hash = _hash_text_ngrams(
            text_blob,
            dim=max(0, int(args.text_hash_dim)),
            n_min=max(1, int(args.text_ngram_min)),
            n_max=max(int(args.text_ngram_min), int(args.text_ngram_max)),
        )
        for i, v in enumerate(text_hash):
            rec[f"text_hash_{i:03d}"] = float(v)

        sid = rec["sample_id"]
        if sid in qwen_map:
            for k, v in qwen_map[sid].items():
                if k == "sample_id":
                    continue
                fv = _safe_float(v)
                if fv is None:
                    continue
                rec[f"qwen_global_{k}"] = float(fv)
        if sid in medbert_map:
            for k, v in medbert_map[sid].items():
                if k == "sample_id":
                    continue
                fv = _safe_float(v)
                if fv is None:
                    continue
                rec[f"medbert_local_{k}"] = float(fv)

        # debug raw text snapshot
        rec["raw_text_concat"] = text_blob[:5000]
        out_rows.append(rec)

    csv_out = out_dir / "tabular_features.csv"
    fields = list(out_rows[0].keys()) if out_rows else []
    _write_csv(csv_out, out_rows, fields)
    parquet_ok = _try_write_parquet(out_rows, out_dir / "tabular_features.parquet")

    print("[tabular] rows:", len(out_rows))
    print("[tabular] csv:", csv_out)
    print("[tabular] parquet_written:", parquet_ok)


if __name__ == "__main__":
    main()
