import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
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


def _collect_outputs(model, loader, forward_fn_eval, device) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
        y_true[lv] = torch.cat(y_true[lv], 0).numpy().astype("int32")
        y_prob[lv] = torch.cat(y_prob[lv], 0).numpy().astype("float32")

    return y_true, y_prob


def _macro_f1_with_thresholds(y: np.ndarray, p: np.ndarray, thresholds: List[float]) -> float:
    if not HAVE_SK:
        return float("nan")
    if y.ndim != 2 or p.ndim != 2:
        return float("nan")
    if y.shape[1] != len(thresholds):
        return float("nan")
    scores = []
    for c in range(y.shape[1]):
        yc = y[:, c]
        pc = p[:, c]
        pred = (pc >= float(thresholds[c])).astype("int32")
        scores.append(f1_score(yc, pred, average="binary", zero_division=0))
    return float(np.mean(scores)) if len(scores) > 0 else float("nan")


def fit_per_class_thresholds(
    model,
    loader,
    forward_fn_eval,
    device,
    search_grid: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    y_true, y_prob = _collect_outputs(model, loader, forward_fn_eval, device)
    if search_grid is None:
        search_grid = [i / 100.0 for i in range(5, 96, 5)]  # 0.05 ... 0.95

    thresholds = {}
    for lv in y_true.keys():
        y = y_true[lv]
        p = y_prob[lv]
        if y.ndim != 2:
            y = y.reshape(y.shape[0], -1)
        if p.ndim != 2:
            p = p.reshape(p.shape[0], -1)

        lv_thrs = []
        for c in range(y.shape[1]):
            yc = y[:, c]
            pc = p[:, c]
            if not HAVE_SK or yc.min() == yc.max():
                lv_thrs.append(0.5)
                continue

            best_t = 0.5
            best_f1 = -1.0
            for t in search_grid:
                pred = (pc >= t).astype("int32")
                f1 = f1_score(yc, pred, average="binary", zero_division=0)
                # tie-breaker: prefer threshold closer to 0.5 for stability
                if (f1 > best_f1) or (f1 == best_f1 and abs(t - 0.5) < abs(best_t - 0.5)):
                    best_f1 = f1
                    best_t = float(t)
            lv_thrs.append(float(best_t))
        thresholds[lv] = lv_thrs

    return thresholds


def evaluate(model, loader, forward_fn_eval, device, thresholds: Optional[Dict[str, List[float]]] = None) -> Dict[str, Dict[str, float]]:
    y_true, y_prob = _collect_outputs(model, loader, forward_fn_eval, device)

    metrics = {}
    for lv in y_true.keys():
        y = y_true[lv]
        p = y_prob[lv]
        y_flat = y.reshape(-1).astype("int32")
        p_flat = p.reshape(-1).astype("float32")
        if HAVE_SK:
            try:
                auroc = roc_auc_score(y_flat, p_flat) if y_flat.min() != y_flat.max() else float("nan")
            except Exception:
                auroc = float("nan")
            try:
                auprc = average_precision_score(y_flat, p_flat)
            except Exception:
                auprc = float("nan")
            pred_05 = (p_flat >= 0.5).astype("int32")
            try:
                f1_05 = f1_score(y_flat, pred_05, average="binary", zero_division=0)
            except Exception:
                f1_05 = float("nan")
            metrics[lv] = {"AUROC": auroc, "AUPRC": auprc, "F1@0.5": f1_05}
            if thresholds is not None and lv in thresholds:
                metrics[lv]["F1@tuned"] = _macro_f1_with_thresholds(y, p, thresholds[lv])
        else:
            metrics[lv] = {"AUROC": float("nan"), "AUPRC": float("nan"), "F1@0.5": float("nan")}
            if thresholds is not None and lv in thresholds:
                metrics[lv]["F1@tuned"] = float("nan")

    return metrics
