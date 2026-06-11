from typing import Dict

import torch


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.tensor(x)


def summarize_banff_metrics(
    probs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    accuracies = []
    macro_recalls = []
    for task, p in probs.items():
        if task not in labels or task not in masks:
            continue
        pp = _to_tensor(p)
        y = _to_tensor(labels[task]).long()
        m = _to_tensor(masks[task]).float() > 0
        n = int(m.sum().item())
        if n == 0:
            summary[task] = {"n": 0, "accuracy": None, "macro_recall": None}
            continue
        pred = pp.argmax(dim=1)
        pred = pred[m]
        yy = y[m]
        accuracy = float((pred == yy).float().mean().item())
        recalls = []
        class_counts = {}
        pred_class_counts = {}
        classes = sorted(set(yy.tolist()) | set(pred.tolist()))
        for cls in sorted(set(yy.tolist())):
            cls_mask = yy == int(cls)
            class_counts[str(int(cls))] = int(cls_mask.sum().item())
            recalls.append(float((pred[cls_mask] == yy[cls_mask]).float().mean().item()))
        for cls in sorted(set(pred.tolist())):
            pred_class_counts[str(int(cls))] = int((pred == int(cls)).sum().item())
        confusion = []
        for true_cls in classes:
            row = []
            true_mask = yy == int(true_cls)
            for pred_cls in classes:
                row.append(int((true_mask & (pred == int(pred_cls))).sum().item()))
            confusion.append(row)
        macro_recall = float(sum(recalls) / len(recalls)) if recalls else None
        summary[task] = {
            "n": n,
            "accuracy": accuracy,
            "macro_recall": macro_recall,
            "class_counts": class_counts,
            "pred_class_counts": pred_class_counts,
            "confusion_classes": [int(c) for c in classes],
            "confusion_matrix": confusion,
        }
        accuracies.append(accuracy)
        if macro_recall is not None:
            macro_recalls.append(macro_recall)
    summary["mean_accuracy"] = float(sum(accuracies) / len(accuracies)) if accuracies else None
    summary["mean_macro_recall"] = float(sum(macro_recalls) / len(macro_recalls)) if macro_recalls else None
    return summary
