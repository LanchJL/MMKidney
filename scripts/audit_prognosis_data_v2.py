#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser("Audit prognosis cohort + feature manifest")
    p.add_argument("--cohort", default="data/processed/prognosis_cohort.csv")
    p.add_argument("--manifest", default="data/processed/prognosis_feature_manifest.json")
    args = p.parse_args()

    rows = _read_csv(Path(args.cohort))
    manifest = _load_manifest(Path(args.manifest))

    split_n = Counter()
    split_events = Counter()
    followup = []
    time_days = []
    modality_cov = defaultdict(Counter)
    for r in rows:
        sp = r.get("split", "")
        split_n[sp] += 1
        split_events[sp] += int(r.get("event", 0) or 0)
        try:
            followup.append(float(r.get("followup_days", 0) or 0))
            time_days.append(float(r.get("time_days", 0) or 0))
        except Exception:
            pass
        for m in ["wsi_available", "tabular_available", "lab_available", "report_available", "treatment_available"]:
            modality_cov[m][sp] += int(r.get(m, 0) or 0)

    def _median(xs):
        if not xs:
            return 0.0
        ys = sorted(xs)
        return float(ys[len(ys) // 2])

    risk = Counter(x.get("leakage_risk", "unknown") for x in manifest.get("features", []))
    role = Counter(x.get("role", "unknown") for x in manifest.get("features", []))
    excluded_high = [x["feature_name"] for x in manifest.get("features", []) if x.get("role") == "exclude" and x.get("leakage_risk") == "high"]

    out = {
        "cohort_n": len(rows),
        "split_n": dict(split_n),
        "split_events": dict(split_events),
        "split_event_rate": {k: (split_events[k] / split_n[k]) if split_n[k] else 0.0 for k in split_n.keys()},
        "followup_days": {
            "median": _median(followup),
            "min": min(followup) if followup else 0.0,
            "max": max(followup) if followup else 0.0,
        },
        "time_days": {
            "median": _median(time_days),
            "min": min(time_days) if time_days else 0.0,
            "max": max(time_days) if time_days else 0.0,
        },
        "modality_coverage_by_split": {
            m: {sp: f"{modality_cov[m][sp]}/{split_n[sp]}" for sp in split_n.keys()} for m in modality_cov.keys()
        },
        "feature_manifest": {
            "n_total_features": manifest.get("n_total_features", 0),
            "n_selected_input_features": manifest.get("n_selected_input_features", 0),
            "n_excluded_features": manifest.get("n_excluded_features", 0),
            "role_distribution": dict(role),
            "leakage_risk_distribution": dict(risk),
            "excluded_high_risk_top30": excluded_high[:30],
        },
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
