import os
import json
from dataclasses import dataclass
from typing import Dict, Optional
import random
import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


@dataclass
class TrainConfig:
    mode: str = "vector"
    out_dir: str = "./output"
    batch_size: int = 64
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lambda_hier: float = 0.2
    lambda_div: float = 0.0
    num_workers: int = 4
    extra_dim: int = 0
    max_patches: Optional[int] = None


def save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def move_labels_to_device(labels: Dict[str, torch.Tensor], device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in labels.items()}


def evaluate(model, loader, forward_fn_eval, device) -> Dict[str, Dict[str, float]]:
    model.eval()
    y_true, y_prob = {}, {}

    for batch in loader:
        if isinstance(batch[0], list):
            xs, labels, ids, extra = batch
            xs = [x.to(device) if isinstance(x, torch.Tensor) else {k: v.to(device) for k, v in x.items()} for x in xs]
            extra = extra.to(device) if extra is not None else None
            out = forward_fn_eval(xs, extra)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            feats, labels, ids, extra = batch
            feats = feats.to(device)
            extra = extra.to(device) if extra is not None else None
            out = forward_fn_eval(feats, extra)
            logits = out[0] if isinstance(out, tuple) else out

        for lv in logits.keys():
            p = torch.sigmoid(logits[lv]).detach().cpu()
            y = labels[lv].detach().cpu()
            y_prob.setdefault(lv, []).append(p)
            y_true.setdefault(lv, []).append(y)

    for lv in list(y_true.keys()):
        y_true[lv] = torch.cat(y_true[lv], 0).numpy().reshape(-1).astype("int32")
        y_prob[lv] = torch.cat(y_prob[lv], 0).numpy().reshape(-1).astype("float32")

    metrics = {}
    for lv in y_true.keys():
        y = y_true[lv]
        p = y_prob[lv]
        if HAVE_SK:
            try:
                auroc = roc_auc_score(y, p) if y.min() != y.max() else float("nan")
            except Exception:
                auroc = float("nan")
            try:
                auprc = average_precision_score(y, p)
            except Exception:
                auprc = float("nan")
            pred_05 = (p >= 0.5).astype("int32")
            try:
                f1_05 = f1_score(y, pred_05, average="binary", zero_division=0)
            except Exception:
                f1_05 = float("nan")
            metrics[lv] = {"AUROC": auroc, "AUPRC": auprc, "F1@0.5": f1_05}
        else:
            metrics[lv] = {"AUROC": float("nan"), "AUPRC": float("nan"), "F1@0.5": float("nan")}

    return metrics
