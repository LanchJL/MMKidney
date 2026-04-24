import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .h5_reader import H5BagReader
from .stain_utils import normalize_stain_name


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
    # 0 structured, 1 banff, 2 timeline, 3 treatment, 4 text_stat/text_kw, 5 text_hash, 6 qwen_global, 7 medbert_local, 8 other
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


def _infer_feature_cols(rows: List[Dict], meta_cols: set) -> List[str]:
    if not rows:
        return []
    all_cols = list(rows[0].keys())
    return [c for c in all_cols if c not in meta_cols]


class PrognosisDataset(Dataset):
    def __init__(
        self,
        prognosis_cohort_csv: str,
        feature_manifest_json: str,
        tabular_feature_csv: str,
        lab_feature_csv: str,
        stain_vocab_path: str,
        manifest_paths: List[str],
        split: Optional[str] = None,
        max_patches_per_stain: Optional[int] = None,
        h5_cache_size: int = 32,
    ):
        self.rows = _read_csv(Path(prognosis_cohort_csv))
        if split is not None:
            self.rows = [r for r in self.rows if (r.get("split", "") or "").strip() == split]

        with Path(feature_manifest_json).open("r", encoding="utf-8") as f:
            feat_manifest = json.load(f)
        self.tab_cols = list(feat_manifest.get("selected_input_features", []))
        self.tab_group_ids = torch.tensor([_feature_group_id(k) for k in self.tab_cols], dtype=torch.long)

        self.tab_map = {r.get("sample_id", ""): r for r in _read_csv(Path(tabular_feature_csv))}
        lab_rows = _read_csv(Path(lab_feature_csv))
        self.lab_map = {r.get("sample_id", ""): r for r in lab_rows}
        self.lab_cols = _infer_feature_cols(lab_rows, meta_cols={"sample_id", "patient_index", "anchor_timestamp"})

        with Path(stain_vocab_path).open("r", encoding="utf-8") as f:
            self.stain_vocab = json.load(f)

        self.wsi_map = {}
        for mp in manifest_paths:
            for rec in _read_jsonl(Path(mp)):
                sid = rec.get("sample_id", "")
                if sid and sid not in self.wsi_map:
                    self.wsi_map[sid] = rec.get("h5s", {})

        # keep only rows with WSI records
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
            stains.append(
                {
                    "name": name,
                    "id": stain_id,
                    "features": feats,
                    "coords": coords,
                }
            )
        if not stains:
            raise RuntimeError(f"No readable stains for sample {sid}")

        tab_vec, tab_mask, tab_present = self._vectorize(self.tab_map.get(sid), self.tab_cols)
        lab_vec, lab_mask, lab_present = self._vectorize(self.lab_map.get(sid), self.lab_cols)

        out = {
            "sample_id": sid,
            "split": r.get("split", ""),
            "event": torch.tensor(float(_safe_float(r.get("event", "")) or 0.0), dtype=torch.float32),
            "time_days": torch.tensor(float(_safe_float(r.get("time_days", "")) or 0.0), dtype=torch.float32),
            "wsi_stains_list": stains,
            "tabular": tab_vec,
            "tabular_mask": tab_mask,
            "tabular_group_ids": self.tab_group_ids.clone(),
            "lab": lab_vec,
            "lab_mask": lab_mask,
            "modality_mask": torch.tensor([1.0, tab_present, lab_present], dtype=torch.float32),
        }
        return out


def collate_prognosis_batch(samples: List[Dict]) -> Dict:
    batch = {
        "sample_ids": [s["sample_id"] for s in samples],
        "splits": [s.get("split", "") for s in samples],
        "wsi": [{"stains": s["wsi_stains_list"]} for s in samples],
        "event": torch.stack([s["event"] for s in samples], dim=0),
        "time_days": torch.stack([s["time_days"] for s in samples], dim=0),
        "tabular": torch.stack([s["tabular"] for s in samples], dim=0),
        "tabular_mask": torch.stack([s["tabular_mask"] for s in samples], dim=0),
        "tabular_group_ids": torch.stack([s["tabular_group_ids"] for s in samples], dim=0),
        "lab": torch.stack([s["lab"] for s in samples], dim=0),
        "lab_mask": torch.stack([s["lab_mask"] for s in samples], dim=0),
        "modality_mask": torch.stack([s["modality_mask"] for s in samples], dim=0),
    }
    return batch
