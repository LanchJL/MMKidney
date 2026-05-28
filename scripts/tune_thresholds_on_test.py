#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def f1_binary(y_true, y_pred):
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else (2.0 * tp / den)


def flatten(mat):
    out = []
    for row in mat:
        out.extend(row)
    return out


def predict_global(prob, thr):
    return [[1 if v >= thr else 0 for v in row] for row in prob]


def predict_per_class(prob, thrs):
    out = []
    for row in prob:
        out.append([1 if row[j] >= thrs[j] else 0 for j in range(len(row))])
    return out


def score_f1(y_true, y_pred):
    ytf = flatten(y_true)
    ypf = flatten(y_pred)
    micro = f1_binary(ytf, ypf)

    n_class = len(y_true[0]) if y_true else 0
    per_class = []
    for j in range(n_class):
        yt = [r[j] for r in y_true]
        yp = [r[j] for r in y_pred]
        per_class.append(f1_binary(yt, yp))
    macro = sum(per_class) / len(per_class) if per_class else 0.0
    return {"micro_f1": micro, "macro_f1": macro, "class_f1": per_class}


def sweep_global(y_true, prob, lo, hi, step):
    best_micro = {"thr": 0.5, "score": -1.0}
    best_macro = {"thr": 0.5, "score": -1.0}
    x = lo
    while x <= hi + 1e-12:
        thr = round(x, 6)
        pred = predict_global(prob, thr)
        s = score_f1(y_true, pred)
        if s["micro_f1"] > best_micro["score"]:
            best_micro = {"thr": thr, "score": s["micro_f1"], "macro_f1": s["macro_f1"]}
        if s["macro_f1"] > best_macro["score"]:
            best_macro = {"thr": thr, "score": s["macro_f1"], "micro_f1": s["micro_f1"]}
        x += step
    return {"best_by_micro_f1": best_micro, "best_by_macro_f1": best_macro}


def sweep_per_class(y_true, prob, lo, hi, step):
    n_class = len(y_true[0]) if y_true else 0
    best_thrs = [0.5] * n_class
    best_f1 = [0.0] * n_class
    for j in range(n_class):
        yt = [r[j] for r in y_true]
        pj = [r[j] for r in prob]
        x = lo
        local_best_thr = 0.5
        local_best_f1 = -1.0
        while x <= hi + 1e-12:
            thr = round(x, 6)
            yp = [1 if p >= thr else 0 for p in pj]
            f1 = f1_binary(yt, yp)
            if f1 > local_best_f1:
                local_best_f1 = f1
                local_best_thr = thr
            x += step
        best_thrs[j] = local_best_thr
        best_f1[j] = local_best_f1

    pred = predict_per_class(prob, best_thrs)
    overall = score_f1(y_true, pred)
    return {
        "thresholds": best_thrs,
        "class_f1_at_best": best_f1,
        "overall_micro_f1": overall["micro_f1"],
        "overall_macro_f1": overall["macro_f1"],
    }


def parse_level(rows, level):
    pk = f"prob_{level}"
    tk = f"true_{level}"
    y_true = [r[tk] for r in rows]
    y_prob = [r[pk] for r in rows]
    return y_true, y_prob


def main():
    p = argparse.ArgumentParser("Tune thresholds on test predictions (oracle upper bound)")
    p.add_argument("--pred-json", required=True, help="Path to per_sample_predictions.json")
    p.add_argument("--out-json", required=True, help="Path to save threshold tuning results")
    p.add_argument("--lo", type=float, default=0.05)
    p.add_argument("--hi", type=float, default=0.95)
    p.add_argument("--step", type=float, default=0.01)
    args = p.parse_args()

    rows = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
    out = {
        "warning": "Thresholds tuned on test set (oracle upper bound). Do not report as unbiased performance.",
        "search_space": {"lo": args.lo, "hi": args.hi, "step": args.step},
        "n_samples": len(rows),
        "levels": {},
    }

    for lv in ["L1", "L2", "L3"]:
        y_true, y_prob = parse_level(rows, lv)
        default_pred = predict_global(y_prob, 0.5)
        default_score = score_f1(y_true, default_pred)
        global_best = sweep_global(y_true, y_prob, args.lo, args.hi, args.step)
        class_best = sweep_per_class(y_true, y_prob, args.lo, args.hi, args.step)
        out["levels"][lv] = {
            "default@0.5": {
                "micro_f1": default_score["micro_f1"],
                "macro_f1": default_score["macro_f1"],
                "class_f1": default_score["class_f1"],
            },
            "global_threshold_search": global_best,
            "per_class_threshold_search": class_best,
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] saved:", args.out_json)


if __name__ == "__main__":
    main()

