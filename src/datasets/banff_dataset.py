import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.banff.schema import BANFF_TASKS
from src.datasets.h5_reader import H5BagReader
from src.datasets.patient_dataset import apply_stain_dropout
from src.datasets.stain_utils import normalize_stain_name


def read_banff_manifest(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def labels_from_banff_record(
    rec: Dict,
    tasks: Optional[List[str]] = None,
    binary_tasks: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    task_names = tasks or list(BANFF_TASKS)
    binary_set = set(binary_tasks or [])
    raw_labels = rec.get("banff_labels", {})
    raw_masks = rec.get("banff_masks", {})
    labels = {}
    masks = {}
    for task in task_names:
        y = int(raw_labels.get(task, -1))
        m = float(raw_masks.get(task, 0.0))
        if y < 0:
            y = 0
            m = 0.0
        elif task in binary_set:
            y = 1 if y > 0 else 0
        labels[task] = torch.tensor(y, dtype=torch.long)
        masks[task] = torch.tensor(m, dtype=torch.float32)
    return labels, masks


class BanffPatientDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        stain_vocab_path: str,
        tasks: Optional[List[str]] = None,
        binary_tasks: Optional[List[str]] = None,
        stain_dropout_p: float = 0.0,
        train_mode: bool = False,
        max_patches_per_stain: Optional[int] = None,
        h5_cache_size: int = 32,
    ):
        self.samples = read_banff_manifest(Path(manifest_path))
        with Path(stain_vocab_path).open("r", encoding="utf-8") as f:
            self.stain_vocab = json.load(f)
        self.tasks = tasks or list(BANFF_TASKS)
        self.binary_tasks = binary_tasks or []
        self.stain_dropout_p = float(stain_dropout_p)
        self.train_mode = bool(train_mode)
        self.max_patches_per_stain = max_patches_per_stain
        self.reader = H5BagReader(cache_size=h5_cache_size)

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
        stains = []
        for raw_name, path in rec.get("h5s", {}).items():
            name = normalize_stain_name(raw_name)
            try:
                item = self.reader.read(path)
            except Exception:
                continue
            feats, coords = self._sample_patches(item["features"], item["coords"])
            stains.append(
                {
                    "name": name,
                    "id": int(self.stain_vocab.get(name, -1)),
                    "features": feats,
                    "coords": coords,
                }
            )
        if self.train_mode:
            stains = apply_stain_dropout(stains, he_name="HE", p=self.stain_dropout_p)
        if not stains:
            raise RuntimeError(f"No readable stains for sample {sid}")

        labels, masks = labels_from_banff_record(rec, tasks=self.tasks, binary_tasks=self.binary_tasks)
        return {
            "sample_id": sid,
            "split": rec.get("split", ""),
            "wsi_stains_list": stains,
            "banff_labels": labels,
            "banff_masks": masks,
            "derived": rec.get("derived", {}),
        }
