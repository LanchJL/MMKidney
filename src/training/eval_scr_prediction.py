import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.scr_prediction_dataset import SCRPredictionDataset, collate_scr_prediction_batch
from src.models.prognosis_teacher_model import MMKidneyPrognosisTeacher
from src.models.scr_seq_encoder import SCRSeqEncoder


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


def _write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["sample_id"])
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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


def _reg_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
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
            "within_0_1": float("nan"),
            "within_0_2": float("nan"),
            "within_0_3": float("nan"),
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

    my = sum(y_true) / n
    sst = sum((y - my) ** 2 for y in y_true)
    sse = sum((p - y) ** 2 for p, y in zip(y_pred, y_true))
    r2 = (1.0 - sse / sst) if sst > 1e-12 else float("nan")

    pear = _pearson(y_true, y_pred)
    rx = _rankdata(y_true)
    ry = _rankdata(y_pred)
    spear = _pearson(rx, ry)

    ae_sorted = sorted(abs_e)
    idx90 = min(n - 1, max(0, int(math.ceil(0.9 * n)) - 1))
    p90_ae = ae_sorted[idx90]

    within_01 = sum(1 for e in abs_e if e <= 0.1) / n
    within_02 = sum(1 for e in abs_e if e <= 0.2) / n
    within_03 = sum(1 for e in abs_e if e <= 0.3) / n

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
        "within_0_1": within_01,
        "within_0_2": within_02,
        "within_0_3": within_03,
    }


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
        out["pred"] = self.reg_head(fused)
        return out


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


def evaluate(model, loader, device, horizons: List[int], target_type: str = "delta_log") -> Dict:
    model.eval()
    per_h_true = {h: [] for h in horizons}
    per_h_pred = {h: [] for h in horizons}
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
                        err = p - y
                        ae = abs(err)
                        row[f"error_d{h}"] = err
                        row[f"abs_error_d{h}"] = ae
                        row[f"sq_error_d{h}"] = err * err
                        row[f"ape_d{h}"] = (ae / abs(y)) if abs(y) > 1e-8 else ""
                        if target_type == "delta_log":
                            row[f"pred_change_pct_d{h}"] = 100.0 * (math.exp(p) - 1.0)
                            row[f"target_change_pct_d{h}"] = 100.0 * (math.exp(y) - 1.0)
                            row[f"abs_change_pct_error_d{h}"] = abs(row[f"pred_change_pct_d{h}"] - row[f"target_change_pct_d{h}"])
                        per_h_true[h].append(y)
                        per_h_pred[h].append(p)
                    else:
                        row[f"error_d{h}"] = ""
                        row[f"abs_error_d{h}"] = ""
                        row[f"sq_error_d{h}"] = ""
                        row[f"ape_d{h}"] = ""
                        if target_type == "delta_log":
                            row[f"pred_change_pct_d{h}"] = ""
                            row[f"target_change_pct_d{h}"] = ""
                            row[f"abs_change_pct_error_d{h}"] = ""
                pred_rows.append(row)

    metrics = {"by_horizon": {}}
    maes = []
    all_true = []
    all_pred = []
    for h in horizons:
        m = _reg_metrics(per_h_true[h], per_h_pred[h])
        metrics["by_horizon"][f"d{h}"] = m
        if m["n"] > 0 and not math.isnan(m["mae"]):
            maes.append(m["mae"])
            all_true.extend(per_h_true[h])
            all_pred.extend(per_h_pred[h])
    metrics["mae_mean"] = (sum(maes) / len(maes)) if maes else float("inf")
    metrics["global_micro"] = _reg_metrics(all_true, all_pred)

    long_rows = []
    for r in pred_rows:
        sid = r["sample_id"]
        sp = r.get("split", "")
        for h in horizons:
            mk = int(r.get(f"mask_d{h}", 0) or 0)
            long_rows.append(
                {
                    "sample_id": sid,
                    "split": sp,
                    "horizon_days": h,
                    "mask": mk,
                    "pred": r.get(f"pred_d{h}", ""),
                    "target": r.get(f"target_d{h}", ""),
                    "error": r.get(f"error_d{h}", ""),
                    "abs_error": r.get(f"abs_error_d{h}", ""),
                    "sq_error": r.get(f"sq_error_d{h}", ""),
                    "ape": r.get(f"ape_d{h}", ""),
                    "pred_change_pct": r.get(f"pred_change_pct_d{h}", ""),
                    "target_change_pct": r.get(f"target_change_pct_d{h}", ""),
                    "abs_change_pct_error": r.get(f"abs_change_pct_error_d{h}", ""),
                }
            )

    return {"metrics": metrics, "predictions": pred_rows, "predictions_long": long_rows}


def _load_run_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_norm_stats(run_dir: Path) -> Dict:
    p = run_dir / "normalization_stats.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser("Evaluate trained short-horizon sCr model on one split")
    p.add_argument("--run-dir", default="")
    p.add_argument("--ckpt", default="")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--out-dir", default="")

    # fallback/manual args
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
    p.add_argument("--profile", choices=["baseline_v1", "full_v1"], default="full_v1")

    p.add_argument("--use-scr-seq", action="store_true")
    p.add_argument("--scr-seq-max-len", type=int, default=32)
    p.add_argument("--scr-seq-hidden-dim", type=int, default=64)
    p.add_argument("--scr-seq-layers", type=int, default=1)
    p.add_argument("--scr-seq-out-dim", type=int, default=64)
    args = p.parse_args()

    norm_stats = {}
    if args.run_dir:
        run_dir = Path(args.run_dir)
        cfg = _load_run_config(run_dir)
        norm_stats = _load_norm_stats(run_dir)
        for k in [
            "cohort_csv",
            "feature_manifest",
            "tabular_features",
            "lab_features",
            "scr_seq_features",
            "stain_vocab",
            "train_manifest",
            "val_manifest",
            "test_manifest",
            "horizons_days",
            "target_type",
            "max_patches",
            "feat_dim",
            "proj_dim",
            "stain_emb_dim",
            "tab_dim",
            "lab_dim",
            "fused_dim",
            "dropout",
            "profile",
            "use_scr_seq",
            "scr_seq_max_len",
            "scr_seq_hidden_dim",
            "scr_seq_layers",
            "scr_seq_out_dim",
        ]:
            if k in cfg:
                setattr(args, k, cfg[k])
        if not args.ckpt:
            args.ckpt = str(run_dir / "best_scr_model.pt")
        if not args.out_dir:
            args.out_dir = str(run_dir / f"eval_{args.split}")

    if not args.ckpt:
        raise ValueError("--ckpt is required (or provide --run-dir with best_scr_model.pt)")
    if not args.out_dir:
        args.out_dir = "outputs/scr_prediction_eval"

    horizons = _parse_ints(args.horizons_days)
    if not horizons:
        raise ValueError("--horizons-days is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_clinicopath_encoder = args.profile in {"full_v1"}

    manifest_paths = [args.train_manifest, args.val_manifest, args.test_manifest]
    ds = SCRPredictionDataset(
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
        split=args.split,
        max_patches_per_stain=args.max_patches,
    )
    if norm_stats:
        ds.set_normalization_stats(norm_stats)

    if len(ds) == 0:
        raise RuntimeError(f"Empty split after filtering: split={args.split}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_scr_prediction_batch)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = stain_vocab.get("HE", 0)

    one = ds[0]
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
        use_scr_seq=bool(args.use_scr_seq),
        scr_seq_hidden_dim=int(args.scr_seq_hidden_dim),
        scr_seq_layers=int(args.scr_seq_layers),
        scr_seq_out_dim=int(args.scr_seq_out_dim),
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    out = evaluate(model, loader, device, horizons, target_type=args.target_type)
    _write_json(out_dir / f"{args.split}_metrics.json", out["metrics"])
    _write_csv(out_dir / f"{args.split}_predictions_wide.csv", out["predictions"])
    _write_csv(out_dir / f"{args.split}_predictions_long.csv", out["predictions_long"])

    run_meta = {
        "split": args.split,
        "n_samples": len(ds),
        "horizons": horizons,
        "target_type": args.target_type,
        "ckpt": args.ckpt,
        "cohort_csv": args.cohort_csv,
        "tabular_features": args.tabular_features,
        "lab_features": args.lab_features,
        "scr_seq_features": args.scr_seq_features,
        "use_scr_seq": bool(args.use_scr_seq),
        "norm_stats_loaded": bool(norm_stats),
    }
    _write_json(out_dir / f"{args.split}_run_meta.json", run_meta)

    print(f"[eval] split={args.split} n={len(ds)}")
    print("[eval] out_dir:", out_dir)
    print("[eval] mae_mean:", out["metrics"].get("mae_mean"))


if __name__ == "__main__":
    main()
