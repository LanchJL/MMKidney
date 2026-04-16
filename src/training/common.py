import json
from pathlib import Path
from typing import Dict

import torch


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = dict(batch)
    out["labels"] = {k: v.to(device) for k, v in batch["labels"].items()}
    out["modality_mask"] = batch["modality_mask"].to(device)
    if batch.get("tabular") is not None:
        out["tabular"] = batch["tabular"].to(device)
    if batch.get("tabular_mask") is not None:
        out["tabular_mask"] = batch["tabular_mask"].to(device)
    if batch.get("tabular_group_ids") is not None:
        out["tabular_group_ids"] = batch["tabular_group_ids"].to(device)
    if batch.get("lab") is not None:
        out["lab"] = batch["lab"].to(device)
    if batch.get("lab_mask") is not None:
        out["lab_mask"] = batch["lab_mask"].to(device)

    # nested stain tensors
    wsi = []
    for p in batch["wsi"]:
        stains = []
        for s in p["stains"]:
            ss = dict(s)
            ss["features"] = s["features"].to(device)
            if s.get("coords") is not None:
                ss["coords"] = s["coords"].to(device)
            stains.append(ss)
        wsi.append({"stains": stains})
    out["wsi"] = wsi
    return out


def save_json(path: str, payload: Dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
