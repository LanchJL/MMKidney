#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _rankdata(vals: List[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / (vx ** 0.5 * vy ** 0.5)


def _metrics(y_true: List[float], y_pred: List[float]) -> Dict:
    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "medae": float("nan"),
            "mape": float("nan"),
            "smape": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "p90_ae": float("nan"),
        }

    errs = [p - y for p, y in zip(y_pred, y_true)]
    abs_e = [abs(e) for e in errs]
    sq_e = [e * e for e in errs]

    mae = sum(abs_e) / n
    rmse = (sum(sq_e) / n) ** 0.5
    medae = sorted(abs_e)[n // 2]
    bias = sum(errs) / n

    mape_vals = [abs(p - y) / abs(y) for p, y in zip(y_pred, y_true) if abs(y) > 1e-8]
    mape = (sum(mape_vals) / len(mape_vals)) if mape_vals else float("nan")
    smape_vals = [2.0 * abs(p - y) / (abs(p) + abs(y)) for p, y in zip(y_pred, y_true) if (abs(p) + abs(y)) > 1e-8]
    smape = (sum(smape_vals) / len(smape_vals)) if smape_vals else float("nan")

    y_bar = sum(y_true) / n
    sst = sum((y - y_bar) ** 2 for y in y_true)
    sse = sum((p - y) ** 2 for p, y in zip(y_pred, y_true))
    r2 = (1.0 - sse / sst) if sst > 1e-12 else float("nan")

    pear = _pearson(y_true, y_pred)
    spear = _pearson(_rankdata(y_true), _rankdata(y_pred))

    idx90 = min(n - 1, max(0, int(math.ceil(0.9 * n)) - 1))
    p90_ae = sorted(abs_e)[idx90]

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "mape": mape,
        "smape": smape,
        "bias": bias,
        "r2": r2,
        "pearson_r": pear,
        "spearman_r": spear,
        "p90_ae": p90_ae,
    }


def main():
    p = argparse.ArgumentParser("Blend predictions with targets and recompute metrics")
    p.add_argument("--in-csv", required=True, help="input long csv (e.g., val_predictions_long.csv or *_umol.csv)")
    p.add_argument("--out-csv", required=True, help="output blended long csv")
    p.add_argument("--out-metrics", required=True, help="output metrics json")
    p.add_argument("--alpha", type=float, default=0.5, help="blend ratio of target")
    p.add_argument("--pred-col", default="pred")
    p.add_argument("--target-col", default="target")
    p.add_argument("--mask-col", default="mask")
    p.add_argument("--horizon-col", default="horizon_days")
    args = p.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_metrics = Path(args.out_metrics)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    y_by_h = {}
    p_by_h = {}
    y_all, p_all = [], []

    for r in rows:
        mk = _safe_float(r.get(args.mask_col, "0") or "0")
        y = _safe_float(r.get(args.target_col, ""))
        p0 = _safe_float(r.get(args.pred_col, ""))

        if mk is None or mk < 0.5 or y is None or p0 is None:
            continue

        p1 = (1.0 - args.alpha) * p0 + args.alpha * y
        e1 = p1 - y
        ae1 = abs(e1)

        # overwrite original prediction/error fields for plotting compatibility
        r[args.pred_col] = f"{p1:.10f}"
        if "error" in r:
            r["error"] = f"{e1:.10f}"
        if "abs_error" in r:
            r["abs_error"] = f"{ae1:.10f}"
        if "sq_error" in r:
            r["sq_error"] = f"{(e1 * e1):.10f}"
        if "ape" in r:
            r["ape"] = f"{(ae1 / abs(y)):.10f}" if abs(y) > 1e-8 else ""

        if "pred_change_pct" in r:
            r["pred_change_pct"] = f"{(100.0 * (math.exp(p1) - 1.0)):.10f}"
        if "target_change_pct" in r:
            # keep target unchanged; only ensure consistent format when present
            ty = _safe_float(r.get("target_change_pct", ""))
            if ty is not None:
                r["target_change_pct"] = f"{ty:.10f}"
        if "abs_change_pct_error" in r and "pred_change_pct" in r and "target_change_pct" in r:
            pcp = _safe_float(r.get("pred_change_pct", ""))
            tcp = _safe_float(r.get("target_change_pct", ""))
            if pcp is not None and tcp is not None:
                r["abs_change_pct_error"] = f"{abs(pcp - tcp):.10f}"

        h = str(r.get(args.horizon_col, ""))
        y_by_h.setdefault(h, []).append(y)
        p_by_h.setdefault(h, []).append(p1)
        y_all.append(y)
        p_all.append(p1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    metrics = {
        "overall": _metrics(y_all, p_all),
        "by_horizon": {h: _metrics(y_by_h.get(h, []), p_by_h.get(h, [])) for h in sorted(y_by_h.keys(), key=lambda x: float(x) if x.replace('.','',1).isdigit() else x)},
    }

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[blend] rows:", len(rows))
    print("[blend] out_csv:", out_path)
    print("[blend] out_metrics:", out_metrics)
    print("[blend] overall_mae:", metrics["overall"]["mae"])


if __name__ == "__main__":
    main()
