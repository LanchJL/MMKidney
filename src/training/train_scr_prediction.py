import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.scr_prediction_dataset import SCRPredictionDataset, collate_scr_prediction_batch
from src.models.prognosis_teacher_model import MMKidneyPrognosisTeacher
from src.models.scr_seq_encoder import SCRSeqEncoder
from src.training.callbacks import EarlyStopper
from src.utils.seed import set_seed


def _parse_ints(s: str) -> List[int]:
    out = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(float(x)))
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


def _move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = dict(batch)
    out["tabular"] = batch["tabular"].to(device)
    out["tabular_mask"] = batch["tabular_mask"].to(device)
    out["tabular_group_ids"] = batch["tabular_group_ids"].to(device)
    out["lab"] = batch["lab"].to(device)
    out["lab_mask"] = batch["lab_mask"].to(device)
    out["modality_mask"] = batch["modality_mask"].to(device)
    out["scr_seq"] = batch["scr_seq"].to(device)
    out["scr_seq_mask"] = batch["scr_seq_mask"].to(device)
    out["scr_seq_present"] = batch["scr_seq_present"].to(device)
    out["target"] = batch["target"].to(device)
    out["target_mask"] = batch["target_mask"].to(device)

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


class SCRMultiHorizonModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        proj_dim: int,
        stain_vocab_size: int,
        stain_emb_dim: int,
        tab_in_dim: int,
        lab_in_dim: int,
        n_horizons: int,
        tab_dim: int = 128,
        lab_dim: int = 128,
        fused_dim: int = 256,
        dropout: float = 0.25,
        he_stain_id: int = 0,
        use_clinicopath_encoder: bool = True,
        use_scr_seq: bool = False,
        scr_seq_hidden_dim: int = 64,
        scr_seq_layers: int = 1,
        scr_seq_out_dim: int = 64,
    ):
        super().__init__()
        self.use_scr_seq = bool(use_scr_seq)
        self.scr_seq_out_dim = int(scr_seq_out_dim) if self.use_scr_seq else 0
        self.backbone = MMKidneyPrognosisTeacher(
            feat_dim=feat_dim,
            proj_dim=proj_dim,
            stain_vocab_size=stain_vocab_size,
            stain_emb_dim=stain_emb_dim,
            tab_in_dim=tab_in_dim,
            lab_in_dim=lab_in_dim,
            tab_dim=tab_dim,
            lab_dim=lab_dim,
            fused_dim=fused_dim,
            dropout=dropout,
            he_stain_id=he_stain_id,
            use_he_adapter=True,
            head_type="cox",
            n_bins=4,
            use_clinicopath_encoder=use_clinicopath_encoder,
        )
        if self.use_scr_seq:
            self.scr_seq_encoder = SCRSeqEncoder(
                in_dim=2,
                hidden_dim=int(scr_seq_hidden_dim),
                num_layers=int(scr_seq_layers),
                out_dim=int(scr_seq_out_dim),
                dropout=dropout,
            )
        else:
            self.scr_seq_encoder = None

        head_in = int(fused_dim) + self.scr_seq_out_dim
        self.reg_head = nn.Sequential(
            nn.Linear(head_in, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, n_horizons),
        )

    def forward(self, batch: Dict) -> Dict:
        out = self.backbone(batch)
        fused = out["fused_repr"]
        if self.use_scr_seq and self.scr_seq_encoder is not None:
            seq_repr = self.scr_seq_encoder(batch["scr_seq"], batch["scr_seq_mask"])
            seq_present = batch["scr_seq_present"].unsqueeze(1)
            seq_repr = seq_repr * seq_present
            fused = torch.cat([fused, seq_repr], dim=1)
            out["scr_seq_repr"] = seq_repr
        pred = self.reg_head(fused)
        out["pred"] = pred
        return out


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = pred - target
    abs_e = torch.abs(err)
    quad = torch.minimum(abs_e, torch.tensor(delta, device=abs_e.device))
    lin = abs_e - quad
    huber = 0.5 * quad * quad + delta * lin
    num = (huber * mask).sum()
    den = torch.clamp(mask.sum(), min=1.0)
    return num / den


def evaluate(model, loader, device, horizons: List[int]) -> Dict:
    model.eval()
    per_h_abs = {h: [] for h in horizons}
    per_h_sq = {h: [] for h in horizons}
    pred_rows = []

    with torch.no_grad():
        for batch in loader:
            b = _move_to_device(batch, device)
            out = model(b)
            pred = out["pred"].detach().cpu()
            tgt = b["target"].detach().cpu()
            msk = b["target_mask"].detach().cpu()

            sids = batch["sample_ids"]
            splits = batch["splits"]
            for i in range(pred.shape[0]):
                row = {"sample_id": sids[i], "split": splits[i]}
                for j, h in enumerate(horizons):
                    p = float(pred[i, j].item())
                    y = float(tgt[i, j].item())
                    m = float(msk[i, j].item())
                    row[f"pred_d{h}"] = p
                    row[f"target_d{h}"] = y if m > 0.5 else ""
                    row[f"mask_d{h}"] = int(m)
                    if m > 0.5:
                        ae = abs(p - y)
                        per_h_abs[h].append(ae)
                        per_h_sq[h].append((p - y) ** 2)
                pred_rows.append(row)

    metrics = {}
    maes = []
    for h in horizons:
        n = len(per_h_abs[h])
        mae = sum(per_h_abs[h]) / n if n > 0 else float("nan")
        rmse = (sum(per_h_sq[h]) / n) ** 0.5 if n > 0 else float("nan")
        metrics[f"d{h}"] = {"n": n, "mae": mae, "rmse": rmse}
        if n > 0:
            maes.append(mae)
    metrics["mae_mean"] = (sum(maes) / len(maes)) if maes else float("inf")
    return {"metrics": metrics, "predictions": pred_rows}


def main():
    p = argparse.ArgumentParser("Train multimodal short-horizon Blood Creatinine predictor")
    p.add_argument("--cohort-csv", default="data/processed/scr_prediction_cohort.csv")
    p.add_argument("--feature-manifest", default="data/processed/prognosis_feature_manifest.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--scr-seq-features", default="data/processed/scr_sequence_features.csv")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--horizons-days", default="14,30,90")
    p.add_argument("--target-type", choices=["delta_log", "log", "raw"], default="delta_log")
    p.add_argument("--out-dir", default="outputs/scr_prediction")

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
    p.add_argument("--profile", choices=["baseline_v1", "full_v1"], default="full_v1")
    p.add_argument("--use-scr-seq", action="store_true")
    p.add_argument("--scr-seq-max-len", type=int, default=32)
    p.add_argument("--scr-seq-hidden-dim", type=int, default=64)
    p.add_argument("--scr-seq-layers", type=int, default=1)
    p.add_argument("--scr-seq-out-dim", type=int, default=64)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    horizons = _parse_ints(args.horizons_days)
    if not horizons:
        raise ValueError("--horizons-days is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_clinicopath_encoder = args.profile in {"full_v1"}

    manifest_paths = [args.train_manifest, args.val_manifest, args.test_manifest]
    ds_train = SCRPredictionDataset(
        cohort_csv=args.cohort_csv,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        horizons_days=horizons,
        target_type=args.target_type,
        feature_manifest_json=args.feature_manifest,
        scr_seq_feature_csv=args.scr_seq_features,
        scr_seq_max_len=args.scr_seq_max_len,
        split="train",
        max_patches_per_stain=args.max_patches,
    )
    ds_val = SCRPredictionDataset(
        cohort_csv=args.cohort_csv,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        horizons_days=horizons,
        target_type=args.target_type,
        feature_manifest_json=args.feature_manifest,
        scr_seq_feature_csv=args.scr_seq_features,
        scr_seq_max_len=args.scr_seq_max_len,
        split="val",
        max_patches_per_stain=args.max_patches,
    )
    ds_test = SCRPredictionDataset(
        cohort_csv=args.cohort_csv,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        horizons_days=horizons,
        target_type=args.target_type,
        feature_manifest_json=args.feature_manifest,
        scr_seq_feature_csv=args.scr_seq_features,
        scr_seq_max_len=args.scr_seq_max_len,
        split="test",
        max_patches_per_stain=args.max_patches,
    )

    if len(ds_train) == 0 or len(ds_val) == 0:
        raise RuntimeError(f"Empty split after filtering: train={len(ds_train)} val={len(ds_val)}")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_scr_prediction_batch)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_scr_prediction_batch)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_scr_prediction_batch)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = stain_vocab.get("HE", 0)

    one = ds_train[0]
    tab_in = int(one["tabular"].shape[0])
    lab_in = int(one["lab"].shape[0])

    model = SCRMultiHorizonModel(
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        stain_vocab_size=len(stain_vocab),
        stain_emb_dim=args.stain_emb_dim,
        tab_in_dim=tab_in,
        lab_in_dim=lab_in,
        n_horizons=len(horizons),
        tab_dim=args.tab_dim,
        lab_dim=args.lab_dim,
        fused_dim=args.fused_dim,
        dropout=args.dropout,
        he_stain_id=he_stain_id,
        use_clinicopath_encoder=use_clinicopath_encoder,
        use_scr_seq=args.use_scr_seq,
        scr_seq_hidden_dim=args.scr_seq_hidden_dim,
        scr_seq_layers=args.scr_seq_layers,
        scr_seq_out_dim=args.scr_seq_out_dim,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early = EarlyStopper(patience=args.early_stop_patience, mode="min")

    cfg = dict(vars(args))
    cfg["tab_in_dim"] = tab_in
    cfg["lab_in_dim"] = lab_in
    _write_json(out_dir / "config.json", cfg)

    best = 1e18
    best_path = out_dir / "best_scr_model.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"train-scr {ep}/{args.epochs}")
        for batch in pbar:
            b = _move_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(b)
                loss = masked_huber_loss(out["pred"], b["target"], b["target_mask"], delta=1.0)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{(sum(losses)/max(1,len(losses))):.4f}")

        val_out = evaluate(model, val_loader, device, horizons)
        val_m = val_out["metrics"]
        score = float(val_m["mae_mean"])
        _write_json(out_dir / f"val_epoch_{ep:03d}.json", val_m)
        _write_predictions(out_dir / f"val_pred_epoch_{ep:03d}.csv", val_out["predictions"])
        print(f"[ep {ep}] train_loss={(sum(losses)/max(1,len(losses))):.4f} val_mae_mean={score:.5f}")

        if score < best:
            best = score
            torch.save(model.state_dict(), best_path)
            _write_json(out_dir / "val_best.json", val_m)
            _write_predictions(out_dir / "val_best_predictions.csv", val_out["predictions"])

        if early.step(score):
            print("[early-stop] triggered")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_out = evaluate(model, test_loader, device, horizons)
    _write_json(out_dir / "test_metrics.json", test_out["metrics"])
    _write_predictions(out_dir / "test_predictions.csv", test_out["predictions"])
    print("[done] best_val_mae_mean:", float(best), "test_mae_mean:", float(test_out["metrics"]["mae_mean"]))


if __name__ == "__main__":
    main()
