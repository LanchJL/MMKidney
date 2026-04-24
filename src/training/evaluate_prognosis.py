import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.prognosis_dataset import PrognosisDataset, collate_prognosis_batch
from src.models.prognosis_teacher_model import MMKidneyPrognosisTeacher
from src.training.train_prognosis import _parse_bins, _write_json, _write_predictions, evaluate


def main():
    p = argparse.ArgumentParser("Evaluate prognosis checkpoint")
    p.add_argument("--prognosis-cohort", default="data/processed/prognosis_cohort.csv")
    p.add_argument("--feature-manifest", default="data/processed/prognosis_feature_manifest.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--out-dir", default="outputs/prognosis_eval")
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
    p.add_argument("--survival-head", choices=["cox", "discrete"], default="discrete")
    p.add_argument("--time-bins-days", default="365,1095,1825")
    p.add_argument("--horizons-days", default="365,1095,1825")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bins_days = _parse_bins(args.time_bins_days)
    horizons = _parse_bins(args.horizons_days)

    manifest_paths = [args.train_manifest, args.val_manifest, args.test_manifest]
    ds = PrognosisDataset(
        prognosis_cohort_csv=args.prognosis_cohort,
        feature_manifest_json=args.feature_manifest,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        split=args.split,
        max_patches_per_stain=args.max_patches,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_prognosis_batch)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = stain_vocab.get("HE", 0)
    one = ds[0]
    tab_in = int(one["tabular"].shape[0])
    lab_in = int(one["lab"].shape[0])
    n_bins = len(bins_days) + 1

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
        use_clinicopath_encoder=True,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    model.eval()

    out = evaluate(model, loader, device, args.survival_head, bins_days=bins_days, horizons=horizons)
    _write_json(out_dir / f"{args.split}_metrics.json", out["metrics"])
    _write_predictions(out_dir / f"{args.split}_predictions.csv", out["predictions"])
    print("[eval] split:", args.split, "c_index:", float(out["metrics"]["c_index"]))
    print("[eval] outputs:", out_dir)


if __name__ == "__main__":
    main()
