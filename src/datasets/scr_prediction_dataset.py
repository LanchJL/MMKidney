import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .h5_reader import H5BagReader
from .stain_utils import normalize_stain_name


META_TAB = {
    "sample_id",
    "split",
    "patient_index",
    "anchor_timestamp",
    "anchor_date_policy",
    "anchor_date_raw",
    "raw_text_concat",
}
META_LAB = {"sample_id", "patient_index", "anchor_timestamp"}
LEAK_KEYWORDS = [
    "latest_scr",
    "death",
    "graft_loss",
    "survival",
    "followup_success",
    "post_tx_",
    "endpoint",
    "outcome",
]


def _read_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _safe_float(v: str) -> Optional[float]:
    s = (v or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _feature_group_id(key: str) -> int:
    if key.startswith("structured_"):
        return 0
    if key.startswith("banff_"):
        return 1
    if key.startswith("timeline_"):
        return 2
    if key.startswith("treatment_"):
        return 3
    if key.startswith("text_stat_") or key.startswith("text_kw_") or key.startswith("text_section_present_"):
        return 4
    if key.startswith("text_hash_"):
        return 5
    if key.startswith("qwen_global_"):
        return 6
    if key.startswith("medbert_local_"):
        return 7
    return 8


def _infer_numeric_cols(rows: List[Dict], meta_cols: set, reject_keywords: Optional[List[str]] = None) -> List[str]:
    if not rows:
        return []
    reject_keywords = reject_keywords or []
    cols = []
    all_cols = list(rows[0].keys())
    for c in all_cols:
        if c in meta_cols:
            continue
        low = c.lower()
        if any(k in low for k in reject_keywords):
            continue
        vals = []
        for r in rows[:200]:
            fv = _safe_float(r.get(c, ""))
            if fv is not None:
                vals.append(fv)
        if vals:
            cols.append(c)
    return cols


class SCRPredictionDataset(Dataset):
    def __init__(
        self,
        cohort_csv: str,
        tabular_feature_csv: str,
        lab_feature_csv: str,
        stain_vocab_path: str,
        manifest_paths: List[str],
        horizons_days: List[int],
        target_type: str = "delta_log",
        feature_manifest_json: str = "",
        scr_seq_feature_csv: str = "",
        scr_seq_max_len: int = 32,
        split: Optional[str] = None,
        max_patches_per_stain: Optional[int] = None,
        h5_cache_size: int = 32,
        exclude_feature_prefixes: Optional[List[str]] = None,
    ):
        self.rows = _read_csv(Path(cohort_csv))
        if split is not None:
            self.rows = [r for r in self.rows if (r.get("split", "") or "").strip() == split]

        self.horizons = list(horizons_days)
        self.target_type = target_type
        self.scr_seq_max_len = int(max(1, scr_seq_max_len))

        tab_rows = _read_csv(Path(tabular_feature_csv))
        self.tab_map = {r.get("sample_id", ""): r for r in tab_rows}

        tab_cols = None
        if feature_manifest_json:
            p = Path(feature_manifest_json)
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    m = json.load(f)
                tab_cols = list(m.get("selected_input_features", []))

        if tab_cols is None:
            tab_cols = _infer_numeric_cols(tab_rows, meta_cols=META_TAB, reject_keywords=LEAK_KEYWORDS)

        ex_pf = exclude_feature_prefixes or []
        if ex_pf:
            tab_cols = [c for c in tab_cols if not any(c.startswith(pf) for pf in ex_pf)]
        self.tab_cols = tab_cols
        self.tab_group_ids = torch.tensor([_feature_group_id(k) for k in self.tab_cols], dtype=torch.long)

        lab_rows = _read_csv(Path(lab_feature_csv))
        self.lab_map = {r.get("sample_id", ""): r for r in lab_rows}
        self.lab_cols = _infer_numeric_cols(lab_rows, meta_cols=META_LAB)

        self.scr_seq_map = {}
        if scr_seq_feature_csv:
            p = Path(scr_seq_feature_csv)
            if p.exists():
                seq_rows = _read_csv(p)
                self.scr_seq_map = {r.get("sample_id", ""): r for r in seq_rows}

        with Path(stain_vocab_path).open("r", encoding="utf-8") as f:
            self.stain_vocab = json.load(f)

        self.wsi_map = {}
        for mp in manifest_paths:
            for rec in _read_jsonl(Path(mp)):
                sid = rec.get("sample_id", "")
                if sid and sid not in self.wsi_map:
                    self.wsi_map[sid] = rec.get("h5s", {})

        self.rows = [r for r in self.rows if (r.get("sample_id", "") in self.wsi_map)]

        self.max_patches_per_stain = max_patches_per_stain
        self.reader = H5BagReader(cache_size=h5_cache_size)

    def __len__(self):
        return len(self.rows)

    def _sample_patches(self, feats: torch.Tensor, coords: Optional[torch.Tensor]):
        if self.max_patches_per_stain is None:
            return feats, coords
        n = feats.shape[0]
        if n <= self.max_patches_per_stain:
            return feats, coords
        idx = torch.randperm(n)[: self.max_patches_per_stain]
        feats = feats[idx]
        if coords is not None:
            coords = coords[idx]
        return feats, coords

    def _vectorize(self, row: Optional[Dict], cols: List[str]):
        if row is None:
            vals = torch.zeros(len(cols), dtype=torch.float32)
            mask = torch.zeros(len(cols), dtype=torch.float32)
            return vals, mask, 0.0
        vals = []
        mask = []
        for c in cols:
            fv = _safe_float(row.get(c, ""))
            if fv is None:
                vals.append(0.0)
                mask.append(0.0)
            else:
                vals.append(float(fv))
                mask.append(1.0)
        m = 1.0 if any(x > 0.0 for x in mask) else 0.0
        return torch.tensor(vals, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), m

    def _targets(self, row: Dict):
        ys = []
        ms = []
        for h in self.horizons:
            key = f"d{h}"
            if self.target_type == "delta_log":
                yv = _safe_float(row.get(f"delta_log_scr_{key}", ""))
            elif self.target_type == "log":
                yv = _safe_float(row.get(f"log_scr_{key}", ""))
            else:
                yv = _safe_float(row.get(f"scr_{key}_umol", ""))
            mk = _safe_float(row.get(f"mask_{key}", ""))
            if mk is None:
                mk = 0.0
            if yv is None or mk < 0.5:
                ys.append(0.0)
                ms.append(0.0)
            else:
                ys.append(float(yv))
                ms.append(1.0)
        return torch.tensor(ys, dtype=torch.float32), torch.tensor(ms, dtype=torch.float32)

    def _scr_sequence(self, sample_id: str):
        row = self.scr_seq_map.get(sample_id)
        max_len = self.scr_seq_max_len
        vals = [0.0] * max_len
        days = [0.0] * max_len
        msk = [0.0] * max_len
        present = 0.0
        if row is None:
            return (
                torch.zeros((max_len, 2), dtype=torch.float32),
                torch.zeros(max_len, dtype=torch.float32),
                present,
            )

        try:
            arr_days = json.loads(row.get("scr_days_json", "[]") or "[]")
        except Exception:
            arr_days = []
        try:
            arr_vals = json.loads(row.get("scr_values_umol_json", "[]") or "[]")
        except Exception:
            arr_vals = []

        seq = []
        for d, v in zip(arr_days, arr_vals):
            fd = _safe_float(str(d))
            fv = _safe_float(str(v))
            if fd is None or fv is None:
                continue
            if fd < 0:
                continue
            seq.append((float(fd), float(fv)))
        seq = sorted(seq, key=lambda x: x[0])
        if len(seq) > max_len:
            seq = seq[-max_len:]
        n = len(seq)
        if n > 0:
            present = 1.0
            for i, (d, v) in enumerate(seq):
                vals[i] = v
                days[i] = d
                msk[i] = 1.0

        x = torch.stack(
            [
                torch.tensor(vals, dtype=torch.float32),
                torch.tensor(days, dtype=torch.float32),
            ],
            dim=1,
        )
        return x, torch.tensor(msk, dtype=torch.float32), present

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        sid = r["sample_id"]
        h5s = self.wsi_map.get(sid, {})

        stains = []
        for name_raw, path in h5s.items():
            name = normalize_stain_name(name_raw)
            try:
                item = self.reader.read(path)
            except Exception:
                continue
            feats = item["features"]
            coords = item["coords"]
            feats, coords = self._sample_patches(feats, coords)
            stain_id = int(self.stain_vocab.get(name, -1))
            stains.append({"name": name, "id": stain_id, "features": feats, "coords": coords})
        if not stains:
            raise RuntimeError(f"No readable stains for sample {sid}")

        tab_vec, tab_mask, tab_present = self._vectorize(self.tab_map.get(sid), self.tab_cols)
        lab_vec, lab_mask, lab_present = self._vectorize(self.lab_map.get(sid), self.lab_cols)
        scr_seq, scr_seq_mask, scr_seq_present = self._scr_sequence(sid)
        y, y_mask = self._targets(r)

        return {
            "sample_id": sid,
            "split": r.get("split", ""),
            "wsi_stains_list": stains,
            "tabular": tab_vec,
            "tabular_mask": tab_mask,
            "tabular_group_ids": self.tab_group_ids.clone(),
            "lab": lab_vec,
            "lab_mask": lab_mask,
            "modality_mask": torch.tensor([1.0, tab_present, lab_present], dtype=torch.float32),
            "scr_seq": scr_seq,
            "scr_seq_mask": scr_seq_mask,
            "scr_seq_present": torch.tensor(float(scr_seq_present), dtype=torch.float32),
            "target": y,
            "target_mask": y_mask,
        }


def collate_scr_prediction_batch(samples: List[Dict]) -> Dict:
    return {
        "sample_ids": [s["sample_id"] for s in samples],
        "splits": [s.get("split", "") for s in samples],
        "wsi": [{"stains": s["wsi_stains_list"]} for s in samples],
        "tabular": torch.stack([s["tabular"] for s in samples], dim=0),
        "tabular_mask": torch.stack([s["tabular_mask"] for s in samples], dim=0),
        "tabular_group_ids": torch.stack([s["tabular_group_ids"] for s in samples], dim=0),
        "lab": torch.stack([s["lab"] for s in samples], dim=0),
        "lab_mask": torch.stack([s["lab_mask"] for s in samples], dim=0),
        "modality_mask": torch.stack([s["modality_mask"] for s in samples], dim=0),
        "scr_seq": torch.stack([s["scr_seq"] for s in samples], dim=0),
        "scr_seq_mask": torch.stack([s["scr_seq_mask"] for s in samples], dim=0),
        "scr_seq_present": torch.stack([s["scr_seq_present"] for s in samples], dim=0),
        "target": torch.stack([s["target"] for s in samples], dim=0),
        "target_mask": torch.stack([s["target_mask"] for s in samples], dim=0),
    }
