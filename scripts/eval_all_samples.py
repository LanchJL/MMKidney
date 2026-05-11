#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.collate import collate_patient_batch
from src.datasets.patient_dataset import MultiStainPatientDataset
from src.models.wsi_model import MMKidneyWSIModel
from src.training.common import move_to_device


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _flatten(xs):
    out = []
    for row in xs:
        out.extend(row)
    return out


def _binary_f1(y_true, y_pred):
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


def _compute_metrics(y_true, y_prob, thr=0.5):
    # y_*: list[list[float]]
    n = len(y_true)
    c = len(y_true[0]) if n > 0 else 0

    y_pred = [[1 if v >= thr else 0 for v in row] for row in y_prob]
    y_true_flat = _flatten(y_true)
    y_pred_flat = _flatten(y_pred)

    out = {
        "n_samples": n,
        "n_classes": c,
        "f1@0.5": _safe_float(_binary_f1(y_true_flat, y_pred_flat)),
        "class_positive_rate": [],
        "class_pred_positive_rate@0.5": [],
        "macro_auroc": None,
        "micro_auroc": None,
        "macro_auprc": None,
        "micro_auprc": None,
    }

    for j in range(c):
        yt = [row[j] for row in y_true]
        yp = [row[j] for row in y_pred]
        out["class_positive_rate"].append(sum(yt) / max(1, len(yt)))
        out["class_pred_positive_rate@0.5"].append(sum(yp) / max(1, len(yp)))

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        class_aurocs = []
        class_auprcs = []
        for j in range(c):
            yt = [row[j] for row in y_true]
            yp = [row[j] for row in y_prob]
            if len(set(yt)) < 2:
                class_aurocs.append(None)
                class_auprcs.append(None)
                continue
            class_aurocs.append(float(roc_auc_score(yt, yp)))
            class_auprcs.append(float(average_precision_score(yt, yp)))

        valid_auc = [x for x in class_aurocs if x is not None]
        valid_ap = [x for x in class_auprcs if x is not None]
        out["class_auroc"] = class_aurocs
        out["class_auprc"] = class_auprcs
        out["macro_auroc"] = (sum(valid_auc) / len(valid_auc)) if valid_auc else None
        out["macro_auprc"] = (sum(valid_ap) / len(valid_ap)) if valid_ap else None

        yf = y_true_flat
        pf = _flatten(y_prob)
        if len(set(yf)) >= 2:
            out["micro_auroc"] = float(roc_auc_score(yf, pf))
            out["micro_auprc"] = float(average_precision_score(yf, pf))
    except Exception:
        out["note"] = "scikit-learn not available: AUROC/AUPRC are omitted."

    return out


def main():
    p = argparse.ArgumentParser("Evaluate all samples and export per-sample + summary")
    p.add_argument("--manifest", required=True)
    p.add_argument("--stain-vocab", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-patches", type=int, default=512)
    p.add_argument("--feat-dim", type=int, default=768)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--stain-emb-dim", type=int, default=64)
    p.add_argument("--thr", type=float, default=0.5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    he_id = int(vocab.get("HE", 0))

    ds = MultiStainPatientDataset(
        manifest_path=args.manifest,
        stain_vocab_path=args.stain_vocab,
        train_mode=False,
        max_patches_per_stain=args.max_patches,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_patient_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMKidneyWSIModel(
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        stain_vocab_size=len(vocab),
        stain_emb_dim=args.stain_emb_dim,
        label_dims=(4, 5, 8),
        he_stain_id=he_id,
        use_he_adapter=True,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    preds = {"L1": [], "L2": [], "L3": []}
    trues = {"L1": [], "L2": [], "L3": []}
    rows = []

    with torch.no_grad():
        for batch in loader:
            b = move_to_device(batch, device)
            out = model(b)
            p1 = torch.sigmoid(out["logits_L1"]).detach().cpu()
            p2 = torch.sigmoid(out["logits_L2"]).detach().cpu()
            p3 = torch.sigmoid(out["logits_L3"]).detach().cpu()
            t1 = b["labels"]["L1"].detach().cpu()
            t2 = b["labels"]["L2"].detach().cpu()
            t3 = b["labels"]["L3"].detach().cpu()

            preds["L1"].extend(p1.tolist())
            preds["L2"].extend(p2.tolist())
            preds["L3"].extend(p3.tolist())
            trues["L1"].extend(t1.tolist())
            trues["L2"].extend(t2.tolist())
            trues["L3"].extend(t3.tolist())

            for i, sid in enumerate(batch["sample_ids"]):
                rows.append(
                    {
                        "sample_id": sid,
                        "prob_L1": p1[i].tolist(),
                        "prob_L2": p2[i].tolist(),
                        "prob_L3": p3[i].tolist(),
                        "true_L1": t1[i].tolist(),
                        "true_L2": t2[i].tolist(),
                        "true_L3": t3[i].tolist(),
                    }
                )

    with (out_dir / "per_sample_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with (out_dir / "per_sample_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "prob_L1", "prob_L2", "prob_L3", "true_L1", "true_L2", "true_L3"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "sample_id": r["sample_id"],
                    "prob_L1": json.dumps(r["prob_L1"], ensure_ascii=False),
                    "prob_L2": json.dumps(r["prob_L2"], ensure_ascii=False),
                    "prob_L3": json.dumps(r["prob_L3"], ensure_ascii=False),
                    "true_L1": json.dumps(r["true_L1"], ensure_ascii=False),
                    "true_L2": json.dumps(r["true_L2"], ensure_ascii=False),
                    "true_L3": json.dumps(r["true_L3"], ensure_ascii=False),
                }
            )

    summary = {
        "L1": _compute_metrics(trues["L1"], preds["L1"], thr=args.thr),
        "L2": _compute_metrics(trues["L2"], preds["L2"], thr=args.thr),
        "L3": _compute_metrics(trues["L3"], preds["L3"], thr=args.thr),
    }
    macro_vals = [summary[k].get("macro_auroc") for k in ("L1", "L2", "L3")]
    macro_vals = [x for x in macro_vals if x is not None]
    summary["mean_macro_auroc"] = (sum(macro_vals) / len(macro_vals)) if macro_vals else None

    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] wrote:")
    print("-", out_dir / "per_sample_predictions.json")
    print("-", out_dir / "per_sample_predictions.csv")
    print("-", out_dir / "summary_metrics.json")


if __name__ == "__main__":
    main()

