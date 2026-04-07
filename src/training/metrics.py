from typing import Dict, List, Optional

import math

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    HAVE_SK = True
except Exception:
    HAVE_SK = False


def _safe_auroc(y_true, y_prob):
    if not HAVE_SK:
        return float("nan")
    try:
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _safe_auprc(y_true, y_prob):
    if not HAVE_SK:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _nanmean(xs: List[float]) -> float:
    ys = [x for x in xs if not math.isnan(x)]
    if not ys:
        return float("nan")
    return float(sum(ys) / len(ys))


def multilabel_metrics(y_true, y_prob, thr: float = 0.5) -> Dict[str, float]:
    # y_* shape [N, C]
    n_class = len(y_true[0]) if len(y_true) > 0 else 0
    aucs, aprs = [], []
    for c in range(n_class):
        yc = [row[c] for row in y_true]
        pc = [row[c] for row in y_prob]
        aucs.append(_safe_auroc(yc, pc))
        aprs.append(_safe_auprc(yc, pc))

    macro_auroc = _nanmean(aucs) if len(aucs) else float("nan")
    macro_auprc = _nanmean(aprs) if len(aprs) else float("nan")
    y_flat = [v for row in y_true for v in row]
    p_flat = [v for row in y_prob for v in row]
    micro_auroc = _safe_auroc(y_flat, p_flat)
    micro_auprc = _safe_auprc(y_flat, p_flat)

    pred = [1 if v >= thr else 0 for v in p_flat]
    if HAVE_SK:
        try:
            f1_micro = float(f1_score(y_flat, pred, average="binary", zero_division=0))
        except Exception:
            f1_micro = float("nan")
    else:
        f1_micro = float("nan")

    out = {
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "macro_auprc": macro_auprc,
        "micro_auprc": micro_auprc,
        "f1@0.5": f1_micro,
    }
    for c, v in enumerate(aucs):
        out[f"class_{c}_auroc"] = float(v)
    return out


def summarize_levels(preds: Dict[str, List[List[float]]], trues: Dict[str, List[List[float]]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for lv in ["L1", "L2", "L3"]:
        out[lv] = multilabel_metrics(trues[lv], preds[lv])
    out["mean_macro_auroc"] = _nanmean([out["L1"]["macro_auroc"], out["L2"]["macro_auroc"], out["L3"]["macro_auroc"]])
    return out
