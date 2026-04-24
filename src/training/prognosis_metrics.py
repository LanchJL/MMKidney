from typing import Dict, List, Tuple

import math

import numpy as np

try:
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

    HAVE_SK = True
except Exception:
    HAVE_SK = False


def harrell_c_index(time_days: List[float], event: List[int], risk: List[float]) -> float:
    n = len(time_days)
    if n <= 1:
        return float("nan")
    concordant = 0.0
    comparable = 0.0
    tied = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = time_days[i], time_days[j]
            ei, ej = int(event[i]), int(event[j])
            ri, rj = risk[i], risk[j]

            # comparable pair: shorter observed time must be an event
            if ti == tj:
                continue
            if ti < tj and ei == 1:
                comparable += 1.0
                if ri > rj:
                    concordant += 1.0
                elif ri == rj:
                    tied += 1.0
            elif tj < ti and ej == 1:
                comparable += 1.0
                if rj > ri:
                    concordant += 1.0
                elif ri == rj:
                    tied += 1.0
    if comparable <= 0:
        return float("nan")
    return float((concordant + 0.5 * tied) / comparable)


def horizon_binary_labels(time_days: List[float], event: List[int], horizon_days: int) -> Tuple[List[int], List[int]]:
    """
    Returns (indices_used, labels) for naive horizon evaluation:
      label=1 if event and time<=horizon
      label=0 if time>horizon
      censored before horizon are excluded
    """
    idx = []
    y = []
    for i, (t, e) in enumerate(zip(time_days, event)):
        if e == 1 and t <= horizon_days:
            idx.append(i)
            y.append(1)
        elif t > horizon_days:
            idx.append(i)
            y.append(0)
        else:
            # censored before/at horizon
            continue
    return idx, y


def safe_auc(y_true: List[int], y_score: List[float]) -> float:
    if not HAVE_SK:
        return float("nan")
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_brier(y_true: List[int], y_prob: List[float]) -> float:
    if not HAVE_SK:
        return float("nan")
    if len(y_true) == 0:
        return float("nan")
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_auprc(y_true: List[int], y_prob: List[float]) -> float:
    if not HAVE_SK:
        return float("nan")
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _cls_stats(y_true: List[int], y_prob: List[float], thr: float = 0.5) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
        }
    y_hat = [1 if p >= thr else 0 for p in y_prob]
    tp = sum(1 for a, b in zip(y_true, y_hat) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_hat) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_hat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_hat) if a == 1 and b == 0)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    if HAVE_SK:
        try:
            f1 = float(f1_score(y_true, y_hat, zero_division=0))
        except Exception:
            f1 = float("nan")
    else:
        f1 = float("nan")
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}


def horizon_metrics(time_days: List[float], event: List[int], risk_h: List[float], horizon_days: int) -> Dict[str, float]:
    idx, y = horizon_binary_labels(time_days, event, horizon_days)
    p = [risk_h[i] for i in idx]
    out = {
        "n_used": float(len(y)),
        "event_rate_used": float(sum(y) / len(y)) if y else float("nan"),
        "auc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": safe_brier(y, p),
    }
    out.update(_cls_stats(y, p, thr=0.5))
    return out


def _nanmean(xs: List[float]) -> float:
    ys = [x for x in xs if not math.isnan(x)]
    if not ys:
        return float("nan")
    return float(sum(ys) / len(ys))


def aggregate_horizon_metrics(ms: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    aucs = []
    auprcs = []
    briers = []
    f1s = []
    for k, v in ms.items():
        if k.endswith("_days"):
            aucs.append(v.get("auc", float("nan")))
            auprcs.append(v.get("auprc", float("nan")))
            briers.append(v.get("brier", float("nan")))
            f1s.append(v.get("f1", float("nan")))
    return {
        "mean_horizon_auc": _nanmean(aucs),
        "mean_horizon_auprc": _nanmean(auprcs),
        "mean_horizon_brier": _nanmean(briers),
        "mean_horizon_f1": _nanmean(f1s),
    }


def risk_group_summary(time_days: List[float], event: List[int], risk: List[float], n_groups: int = 3) -> Dict[str, Dict[str, float]]:
    rows = sorted(zip(risk, time_days, event), key=lambda x: x[0])
    n = len(rows)
    out = {}
    if n == 0:
        return out
    for g in range(n_groups):
        lo = int(g * n / n_groups)
        hi = int((g + 1) * n / n_groups)
        grp = rows[lo:hi]
        if not grp:
            continue
        t = [x[1] for x in grp]
        e = [int(x[2]) for x in grp]
        rr = [x[0] for x in grp]
        t_sorted = sorted(t)
        out[f"group_{g+1}"] = {
            "n": float(len(grp)),
            "event_rate": float(sum(e) / len(e)) if e else float("nan"),
            "median_time_days": float(t_sorted[len(t_sorted) // 2]) if t_sorted else float("nan"),
            "mean_risk": float(sum(rr) / len(rr)) if rr else float("nan"),
        }
    return out
