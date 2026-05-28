from typing import Dict, List

import torch


def _stack_optional(vs: List[torch.Tensor]):
    if not vs:
        return None
    if any(v is None for v in vs):
        return None
    return torch.stack(vs, dim=0)


def collate_patient_batch(samples: List[Dict]) -> Dict:
    batch = {
        "sample_ids": [s["sample_id"] for s in samples],
        "splits": [s.get("split", "") for s in samples],
        "wsi": [{"stains": s["wsi_stains_list"]} for s in samples],
        "labels": {
            "L1": torch.stack([s["labels"]["L1"] for s in samples], dim=0),
            "L2": torch.stack([s["labels"]["L2"] for s in samples], dim=0),
            "L3": torch.stack([s["labels"]["L3"] for s in samples], dim=0),
        },
        "tabular": _stack_optional([s.get("tabular") for s in samples]),
        "tabular_mask": _stack_optional([s.get("tabular_mask") for s in samples]),
        "tabular_group_ids": _stack_optional([s.get("tabular_group_ids") for s in samples]),
        "lab": _stack_optional([s.get("lab") for s in samples]),
        "lab_mask": _stack_optional([s.get("lab_mask") for s in samples]),
        "modality_mask": torch.stack([s["modality_mask"] for s in samples], dim=0),
    }
    return batch


def collate_banff_patient_batch(samples: List[Dict]) -> Dict:
    task_names = list(samples[0]["banff_labels"].keys())
    return {
        "sample_ids": [s["sample_id"] for s in samples],
        "splits": [s.get("split", "") for s in samples],
        "wsi": [{"stains": s["wsi_stains_list"]} for s in samples],
        "banff_labels": {
            task: torch.stack([s["banff_labels"][task] for s in samples], dim=0)
            for task in task_names
        },
        "banff_masks": {
            task: torch.stack([s["banff_masks"][task] for s in samples], dim=0)
            for task in task_names
        },
    }
