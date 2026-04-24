import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "split"])
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _parse_horizons(text: str) -> List[int]:
    return [int(x.strip()) for x in (text or "").split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser("Build cohort manifest for short-horizon sCr prediction")
    p.add_argument("--scr-labels", default="data/processed/scr_labels.csv")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--horizons-days", default="14,30,90")
    p.add_argument("--out-csv", default="data/processed/scr_prediction_cohort.csv")
    p.add_argument("--min-labeled-horizons", type=int, default=1)
    args = p.parse_args()

    horizons = _parse_horizons(args.horizons_days)
    labels = _read_csv(Path(args.scr_labels))
    tab = _read_csv(Path(args.tabular_features))
    lab = _read_csv(Path(args.lab_features))

    tab_ids = {r.get("sample_id", "") for r in tab}
    lab_ids = {r.get("sample_id", "") for r in lab}

    wsi_ids = set()
    split_map = {}
    for mp in [args.train_manifest, args.val_manifest, args.test_manifest]:
        for rec in _read_jsonl(Path(mp)):
            sid = (rec.get("sample_id", "") or "").strip()
            if not sid:
                continue
            wsi_ids.add(sid)
            sp = (rec.get("split", "") or "").strip()
            if sid not in split_map and sp:
                split_map[sid] = sp

    out_rows = []
    dropped = {
        "no_wsi": 0,
        "no_label": 0,
    }

    for r in labels:
        sid = (r.get("sample_id", "") or "").strip()
        if not sid:
            continue
        if sid not in wsi_ids:
            dropped["no_wsi"] += 1
            continue

        n_label = 0
        for h in horizons:
            mk = (r.get(f"mask_d{h}", "0") or "0").strip()
            if mk == "1":
                n_label += 1

        if n_label < int(args.min_labeled_horizons):
            dropped["no_label"] += 1
            continue

        out = dict(r)
        out["split"] = out.get("split", "") or split_map.get(sid, "")
        out["wsi_available"] = "1"
        out["tabular_available"] = "1" if sid in tab_ids else "0"
        out["lab_available"] = "1" if sid in lab_ids else "0"
        out["n_labeled_horizons"] = str(n_label)
        out_rows.append(out)

    out_rows = sorted(out_rows, key=lambda x: (x.get("split", ""), x.get("sample_id", "")))
    _write_csv(Path(args.out_csv), out_rows)

    split_n = {}
    for r in out_rows:
        sp = r.get("split", "")
        split_n[sp] = split_n.get(sp, 0) + 1

    print("[scr_cohort] in_labels:", len(labels))
    print("[scr_cohort] out_rows:", len(out_rows))
    print("[scr_cohort] split_n:", split_n)
    print("[scr_cohort] dropped:", dropped)
    print("[scr_cohort] out:", args.out_csv)


if __name__ == "__main__":
    main()
