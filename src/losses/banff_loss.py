from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_banff_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = []
    logs: Dict[str, float] = {}
    device = next(iter(logits.values())).device
    for task, task_logits in logits.items():
        if task not in labels or task not in masks:
            continue
        y = labels[task].to(device=device, dtype=torch.long)
        m = masks[task].to(device=device, dtype=torch.float32)
        valid = m > 0
        logs[f"{task}_n"] = int(valid.sum().item())
        if not bool(valid.any()):
            continue
        task_loss = F.cross_entropy(task_logits[valid], y[valid], reduction="mean")
        losses.append(task_loss)
        logs[f"{task}_loss"] = float(task_loss.detach().cpu().item())
    if not losses:
        zero = next(iter(logits.values())).sum() * 0.0
        return zero, logs
    loss = torch.stack(losses).mean()
    logs["loss"] = float(loss.detach().cpu().item())
    return loss, logs
