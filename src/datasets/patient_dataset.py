import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .h5_reader import H5BagReader
from .stain_utils import normalize_stain_name


def apply_stain_dropout(stains: List[Dict], he_name: str = "HE", p: float = 0.2) -> List[Dict]:
    if p <= 0.0:
        return stains
    kept = []
    for s in stains:
        name = s["name"]
        if name == he_name:
            kept.append(s)
            continue
        if random.random() >= p:
            kept.append(s)
    if not any(s["name"] == he_name for s in kept):
        # never drop HE by design, but keep safe guard
        for s in stains:
            if s["name"] == he_name:
                kept.insert(0, s)
                break
    return kept if kept else stains


def _read_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_feature_map(path: Optional[Path]) -> Dict[str, Dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        sid = r.get("sample_id", "")
        out[sid] = r
    return out


def _vectorize_row(row: Dict, exclude_prefix: Optional[List[str]] = None, keys: Optional[List[str]] = None):
    if not row:
        return None, None
    ex = exclude_prefix or ["raw_"]
    vals, mask = [], []
    use_keys = keys or sorted(row.keys())
    for k in use_keys:
        if k in {"sample_id", "split", "patient_index", "anchor_timestamp", "anchor_date_policy", "anchor_date_raw"}:
            continue
        if any(k.startswith(pf) for pf in ex):
            continue
        v = row.get(k, "")
        try:
            fv = float(v)
            vals.append(fv)
            mask.append(1.0)
        except Exception:
            # non numeric is dropped from dense vector
            continue
    if not vals:
        return None, None
    return torch.tensor(vals, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


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


class MultiStainPatientDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        stain_vocab_path: str,
        tabular_feature_csv: Optional[str] = None,
        lab_feature_csv: Optional[str] = None,
        stain_dropout_p: float = 0.0,
        train_mode: bool = False,
        max_patches_per_stain: Optional[int] = None,
        h5_cache_size: int = 32,
    ):
        self.samples = _read_jsonl(Path(manifest_path))
        with Path(stain_vocab_path).open("r", encoding="utf-8") as f:
            self.stain_vocab = json.load(f)

        self.tab_map = _read_feature_map(Path(tabular_feature_csv) if tabular_feature_csv else None)
        self.lab_map = _read_feature_map(Path(lab_feature_csv) if lab_feature_csv else None)

        self.stain_dropout_p = float(stain_dropout_p)
        self.train_mode = bool(train_mode)
        self.max_patches_per_stain = max_patches_per_stain
        self.reader = H5BagReader(cache_size=h5_cache_size)

        # infer stable feature keys from first rows
        self.tab_keys = None
        self.lab_keys = None
        if self.tab_map:
            first = next(iter(self.tab_map.values()))
            self.tab_keys = [k for k in sorted(first.keys()) if k not in {"sample_id", "split", "patient_index", "anchor_timestamp", "anchor_date_policy", "anchor_date_raw"} and not k.startswith("raw_")]
            self.tab_group_ids = torch.tensor([_feature_group_id(k) for k in self.tab_keys], dtype=torch.long)
        else:
            self.tab_group_ids = None
        if self.lab_map:
            first = next(iter(self.lab_map.values()))
            self.lab_keys = [k for k in sorted(first.keys()) if k not in {"sample_id", "split", "patient_index", "anchor_timestamp"}]

    def __len__(self):
        return len(self.samples)

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

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        sid = rec["sample_id"]
        split = rec.get("split", "")
        h5s = rec.get("h5s", {})

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

        if self.train_mode:
            stains = apply_stain_dropout(stains, he_name="HE", p=self.stain_dropout_p)

        if not stains:
            raise RuntimeError(f"No readable stains for sample {sid}")

        labels = {
            "L1": torch.tensor(rec["labels"]["L1"], dtype=torch.float32),
            "L2": torch.tensor(rec["labels"]["L2"], dtype=torch.float32),
            "L3": torch.tensor(rec["labels"]["L3"], dtype=torch.float32),
        }

        tab_row = self.tab_map.get(sid)
        lab_row = self.lab_map.get(sid)
        tab_vec, tab_mask = _vectorize_row(tab_row, keys=self.tab_keys)
        lab_vec, lab_mask = _vectorize_row(lab_row, keys=self.lab_keys)
        tab_group_ids = self.tab_group_ids.clone() if tab_vec is not None and self.tab_group_ids is not None else None

        modality_mask = torch.tensor(
            [
                1.0,
                1.0 if tab_vec is not None else 0.0,
                1.0 if lab_vec is not None else 0.0,
            ],
            dtype=torch.float32,
        )

        return {
            "sample_id": sid,
            "split": split,
            "stains": {s["name"]: {"features": s["features"], "coords": s["coords"]} for s in stains},
            "stain_names": [s["name"] for s in stains],
            "stain_ids": torch.tensor([s["id"] for s in stains], dtype=torch.long),
            "wsi_stains_list": stains,
            "labels": labels,
            "tabular": tab_vec,
            "tabular_mask": tab_mask,
            "tabular_group_ids": tab_group_ids,
            "lab": lab_vec,
            "lab_mask": lab_mask,
            "modality_mask": modality_mask,
        }
