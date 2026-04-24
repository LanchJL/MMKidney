from typing import Dict, List, Tuple


def _km_curve(time_days: List[float], event: List[int]) -> Tuple[List[float], List[float]]:
    pairs = sorted(zip(time_days, event), key=lambda x: x[0])
    if not pairs:
        return [0.0], [1.0]
    n = len(pairs)
    at_risk = float(n)
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    i = 0
    while i < n:
        t = pairs[i][0]
        d = 0.0
        c = 0.0
        while i < n and pairs[i][0] == t:
            if int(pairs[i][1]) == 1:
                d += 1.0
            else:
                c += 1.0
            i += 1
        if at_risk > 0 and d > 0:
            surv *= (1.0 - d / at_risk)
            xs.append(float(t))
            ys.append(float(surv))
        at_risk -= (d + c)
        if at_risk <= 0:
            break
    return xs, ys


def _group_by_risk(pred_rows: List[Dict], n_groups: int = 3) -> List[List[Dict]]:
    rows = sorted(pred_rows, key=lambda r: float(r.get("risk_score", 0.0)))
    n = len(rows)
    groups = []
    for g in range(n_groups):
        lo = int(g * n / n_groups)
        hi = int((g + 1) * n / n_groups)
        groups.append(rows[lo:hi])
    return groups


def plot_km_by_risk_group(pred_rows: List[Dict], out_png: str, n_groups: int = 3) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    groups = _group_by_risk(pred_rows, n_groups=n_groups)
    plt.figure(figsize=(6, 5))
    labels = ["Low risk", "Mid risk", "High risk"] if n_groups == 3 else [f"Group {i+1}" for i in range(n_groups)]
    for i, grp in enumerate(groups):
        t = [float(r["time_days"]) for r in grp]
        e = [int(r["event"]) for r in grp]
        xs, ys = _km_curve(t, e)
        plt.step(xs, ys, where="post", label=f"{labels[i]} (n={len(grp)})")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.title("KM by Predicted Risk Group")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True


def _horizon_labels(pred_rows: List[Dict], horizon_days: int):
    y = []
    p = []
    for r in pred_rows:
        t = float(r["time_days"])
        e = int(r["event"])
        # exclude censored before horizon
        if e == 0 and t <= horizon_days:
            continue
        y.append(1 if (e == 1 and t <= horizon_days) else 0)
        p.append(float(r.get(f"risk_{horizon_days}d", r.get("risk_score", 0.0))))
    return y, p


def plot_calibration_curve(pred_rows: List[Dict], horizon_days: int, out_png: str, n_bins: int = 5) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    y, p = _horizon_labels(pred_rows, horizon_days)
    if len(y) < n_bins:
        return False
    order = np.argsort(np.array(p))
    y = np.array(y)[order]
    p = np.array(p)[order]

    bins = np.array_split(np.arange(len(y)), n_bins)
    obs = []
    pred = []
    for b in bins:
        if len(b) == 0:
            continue
        obs.append(float(np.mean(y[b])))
        pred.append(float(np.mean(p[b])))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.plot(pred, obs, marker="o")
    plt.xlabel("Predicted Risk")
    plt.ylabel("Observed Event Rate")
    plt.title(f"Calibration @ {horizon_days}d")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True
