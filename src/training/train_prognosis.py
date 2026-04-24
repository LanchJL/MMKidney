import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.prognosis_dataset import PrognosisDataset, collate_prognosis_batch
from src.models.prognosis_heads import cox_ph_loss, discrete_time_nll, horizon_risk_from_logits, survival_probs_from_logits
from src.models.prognosis_teacher_model import MMKidneyPrognosisTeacher
from src.training.callbacks import EarlyStopper
from src.training.prognosis_metrics import aggregate_horizon_metrics, harrell_c_index, horizon_metrics
from src.utils.seed import set_seed


def _parse_bins(s: str) -> List[int]:
    xs = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        xs.append(int(float(x)))
    if not xs:
        raise ValueError("time bins are empty")
    xs = sorted(set(xs))
    return xs


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


def _write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_predictions(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def evaluate(model, loader, device, head_type: str, bins_days: List[int], horizons: List[int]) -> Dict:
    model.eval()
    times = []
    events = []
    risk = []
    risk_h = {h: [] for h in horizons}
    pred_rows = []

    with torch.no_grad():
        for batch in loader:
            b = _move_to_device(batch, device)
            out = model(b)

            if head_type == "cox":
                r = out["risk"].detach().cpu()
                r_h = {h: torch.sigmoid(r).tolist() for h in horizons}
            else:
                logits = out["hazard_logits"]
                s_bins = survival_probs_from_logits(logits)
                r = (1.0 - s_bins[:, -1]).detach().cpu()
                r_h = {h: horizon_risk_from_logits(logits, bins_days=bins_days, horizon_days=h).detach().cpu().tolist() for h in horizons}

            t = b["time_days"].detach().cpu().tolist()
            e = b["event"].detach().cpu().tolist()
            rr = r.tolist()
            sids = batch["sample_ids"]
            splits = batch["splits"]

            times.extend(t)
            events.extend([int(x) for x in e])
            risk.extend(rr)
            for h in horizons:
                risk_h[h].extend(r_h[h])

            for i in range(len(sids)):
                row = {
                    "sample_id": sids[i],
                    "split": splits[i],
                    "time_days": float(t[i]),
                    "event": int(e[i]),
                    "risk_score": float(rr[i]),
                }
                for h in horizons:
                    row[f"risk_{h}d"] = float(r_h[h][i])
                    row[f"survival_{h}d"] = float(1.0 - r_h[h][i])
                pred_rows.append(row)

    out_m = {
        "c_index": harrell_c_index(times, events, risk),
        "n": float(len(times)),
        "n_event": float(sum(events)),
    }
    hm = {}
    for h in horizons:
        hm[f"{h}_days"] = horizon_metrics(times, events, risk_h[h], h)
    out_m["horizons"] = hm
    out_m.update(aggregate_horizon_metrics(hm))
    return {"metrics": out_m, "predictions": pred_rows}


def main():
    p = argparse.ArgumentParser("Train multimodal prognosis model (Cox/Discrete)")
    p.add_argument("--prognosis-cohort", default="data/processed/prognosis_cohort.csv")
    p.add_argument("--feature-manifest", default="data/processed/prognosis_feature_manifest.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--out-dir", default="outputs/prognosis")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
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
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bins_days = _parse_bins(args.time_bins_days)
    horizons = _parse_bins(args.horizons_days)

    manifest_paths = [args.train_manifest, args.val_manifest, args.test_manifest]
    ds_train = PrognosisDataset(
        prognosis_cohort_csv=args.prognosis_cohort,
        feature_manifest_json=args.feature_manifest,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        split="train",
        max_patches_per_stain=args.max_patches,
    )
    ds_val = PrognosisDataset(
        prognosis_cohort_csv=args.prognosis_cohort,
        feature_manifest_json=args.feature_manifest,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        split="val",
        max_patches_per_stain=args.max_patches,
    )
    ds_test = PrognosisDataset(
        prognosis_cohort_csv=args.prognosis_cohort,
        feature_manifest_json=args.feature_manifest,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        split="test",
        max_patches_per_stain=args.max_patches,
    )

    if len(ds_train) == 0 or len(ds_val) == 0:
        raise RuntimeError(f"Empty split after filtering: train={len(ds_train)} val={len(ds_val)}")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_prognosis_batch)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_prognosis_batch)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_prognosis_batch)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = stain_vocab.get("HE", 0)

    one = ds_train[0]
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

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early = EarlyStopper(patience=args.early_stop_patience, mode="max")

    best = -1e9
    best_path = out_dir / "best_prognosis.pt"
    _write_json(out_dir / "config.json", vars(args))

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"train-prognosis {ep}/{args.epochs}")
        for batch in pbar:
            b = _move_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(b)
                if args.survival_head == "cox":
                    loss = cox_ph_loss(out["risk"], b["event"], b["time_days"])
                else:
                    loss = discrete_time_nll(out["hazard_logits"], b["event"], b["time_days"], bins_days=bins_days)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{(sum(losses)/max(1,len(losses))):.4f}")

        val_out = evaluate(model, val_loader, device, args.survival_head, bins_days=bins_days, horizons=horizons)
        val_m = val_out["metrics"]
        score = float(val_m["c_index"])
        _write_json(out_dir / f"val_epoch_{ep:03d}.json", val_m)
        _write_predictions(out_dir / f"val_pred_epoch_{ep:03d}.csv", val_out["predictions"])
        print(f"[ep {ep}] train_loss={(sum(losses)/max(1,len(losses))):.4f} val_c_index={score:.4f}")

        if score > best:
            best = score
            torch.save(model.state_dict(), best_path)
            _write_json(out_dir / "val_best.json", val_m)
            _write_predictions(out_dir / "val_best_predictions.csv", val_out["predictions"])

        if early.step(score):
            print("[early-stop] triggered")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_out = evaluate(model, test_loader, device, args.survival_head, bins_days=bins_days, horizons=horizons)
    _write_json(out_dir / "test_metrics.json", test_out["metrics"])
    _write_predictions(out_dir / "test_predictions.csv", test_out["predictions"])
    print("[done] best_val_c_index:", float(best), "test_c_index:", float(test_out["metrics"]["c_index"]))


if __name__ == "__main__":
    main()
