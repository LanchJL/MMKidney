import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.datasets.prognosis_dataset import PrognosisDataset, collate_prognosis_batch
from src.models.prognosis_heads import horizon_risk_from_logits, survival_probs_from_logits
from src.models.prognosis_teacher_model import MMKidneyPrognosisTeacher


def _parse_bins(s: str) -> List[int]:
    xs = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        xs.append(int(float(x)))
    if not xs:
        raise ValueError("time bins are empty")
    return sorted(set(xs))


def _move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = dict(batch)
    out["event"] = batch["event"].to(device)
    out["time_days"] = batch["time_days"].to(device)
    out["tabular"] = batch["tabular"].to(device)
    out["tabular_mask"] = batch["tabular_mask"].to(device)
    out["tabular_group_ids"] = batch["tabular_group_ids"].to(device)
    out["lab"] = batch["lab"].to(device)
    out["lab_mask"] = batch["lab_mask"].to(device)
    out["modality_mask"] = batch["modality_mask"].to(device)

    wsi = []
    for p in batch["wsi"]:
        stains = []
        for s in p["stains"]:
            ss = dict(s)
            ss["features"] = s["features"].to(device)
            if s.get("coords") is not None:
                ss["coords"] = s["coords"].to(device)
            stains.append(ss)
        wsi.append({"stains": stains})
    out["wsi"] = wsi
    return out


def _write_predictions(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser("Inference for prognosis model (supports unlabeled samples)")
    p.add_argument("--input-cohort", default="data/processed/cohort_tabular.csv")
    p.add_argument("--feature-manifest", default="data/processed/prognosis_feature_manifest.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-csv", default="outputs/prognosis_infer/all_predictions.csv")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-patches", type=int, default=512)
    p.add_argument("--feat-dim", type=int, default=768)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--stain-emb-dim", type=int, default=64)
    p.add_argument("--tab-dim", type=int, default=128)
    p.add_argument("--lab-dim", type=int, default=128)
    p.add_argument("--fused-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--profile", choices=["baseline_v1", "full_v1", "full_v2"], default="full_v1")
    p.add_argument("--survival-head", choices=["cox", "discrete"], default="discrete")
    p.add_argument("--time-bins-days", default="365,1095,1825")
    p.add_argument("--horizons-days", default="365,1095,1825")
    p.add_argument("--strict-treatment-history", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bins_days = _parse_bins(args.time_bins_days)
    horizons = _parse_bins(args.horizons_days)
    n_bins = len(bins_days) + 1
    use_clinicopath_encoder = args.profile in {"full_v1", "full_v2"}
    strict_treatment = bool(args.strict_treatment_history or args.profile == "full_v2")
    exclude_feature_prefixes = ["treatment_"] if strict_treatment else None

    manifest_paths = [args.train_manifest, args.val_manifest, args.test_manifest]
    ds = PrognosisDataset(
        prognosis_cohort_csv=args.input_cohort,
        feature_manifest_json=args.feature_manifest,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        split=None,
        max_patches_per_stain=args.max_patches,
        exclude_feature_prefixes=exclude_feature_prefixes,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_prognosis_batch)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = stain_vocab.get("HE", 0)
    one = ds[0]
    tab_in = int(one["tabular"].shape[0])
    lab_in = int(one["lab"].shape[0])

    model = MMKidneyPrognosisTeacher(
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        stain_vocab_size=len(stain_vocab),
        stain_emb_dim=args.stain_emb_dim,
        tab_in_dim=tab_in,
        lab_in_dim=lab_in,
        tab_dim=args.tab_dim,
        lab_dim=args.lab_dim,
        fused_dim=args.fused_dim,
        dropout=args.dropout,
        he_stain_id=he_stain_id,
        use_he_adapter=True,
        head_type=args.survival_head,
        n_bins=n_bins,
        use_clinicopath_encoder=use_clinicopath_encoder,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            b = _move_to_device(batch, device)
            out = model(b)
            if args.survival_head == "cox":
                r = out["risk"].detach().cpu()
                risk_h = {h: torch.sigmoid(r).tolist() for h in horizons}
            else:
                logits = out["hazard_logits"]
                s_bins = survival_probs_from_logits(logits)
                r = (1.0 - s_bins[:, -1]).detach().cpu()
                risk_h = {h: horizon_risk_from_logits(logits, bins_days=bins_days, horizon_days=h).detach().cpu().tolist() for h in horizons}

            for i, sid in enumerate(batch["sample_ids"]):
                rec = {
                    "sample_id": sid,
                    "split": batch["splits"][i],
                    "risk_score": float(r[i].item()),
                }
                for h in horizons:
                    rec[f"risk_{h}d"] = float(risk_h[h][i])
                    rec[f"survival_{h}d"] = float(1.0 - risk_h[h][i])
                rows.append(rec)

    _write_predictions(Path(args.out_csv), rows)
    print("[infer-prognosis] n_pred:", len(rows))
    print("[infer-prognosis] out:", args.out_csv)


if __name__ == "__main__":
    main()
