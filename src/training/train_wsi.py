import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.collate import collate_patient_batch
from src.datasets.patient_dataset import MultiStainPatientDataset
from src.losses.asymmetric_loss import AsymmetricLossMultiLabel
from src.losses.distill_loss import compute_supervised_loss
from src.losses.hierarchical_consistency import HierarchicalConsistencyLoss
from src.models.wsi_model import MMKidneyWSIModel
from src.training.callbacks import EarlyStopper
from src.training.common import move_to_device, save_json
from src.training.metrics import summarize_levels
from src.training.optimizer import build_optimizer
from src.training.schedulers import build_scheduler
from src.utils.seed import set_seed


def evaluate(model, loader, device):
    model.eval()
    preds = {"L1": [], "L2": [], "L3": []}
    trues = {"L1": [], "L2": [], "L3": []}
    with torch.no_grad():
        for batch in loader:
            b = move_to_device(batch, device)
            out = model(b)
            for lv in ["L1", "L2", "L3"]:
                preds[lv].append(torch.sigmoid(out[f"logits_{lv}"]).detach().cpu())
                trues[lv].append(b["labels"][lv].detach().cpu())
    preds = {k: torch.cat(v, dim=0).tolist() for k, v in preds.items()}
    trues = {k: torch.cat(v, dim=0).tolist() for k, v in trues.items()}
    return summarize_levels(preds, trues)


def main():
    p = argparse.ArgumentParser("Train WSI-only MMKidney student")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--out-dir", default="outputs/wsi")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stain-dropout", type=float, default=0.2)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-patches", type=int, default=512)
    p.add_argument("--feat-dim", type=int, default=768)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--stain-emb-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-set-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = MultiStainPatientDataset(
        manifest_path=args.train_manifest,
        stain_vocab_path=args.stain_vocab,
        stain_dropout_p=args.stain_dropout,
        train_mode=True,
        max_patches_per_stain=args.max_patches,
    )
    ds_val = MultiStainPatientDataset(
        manifest_path=args.val_manifest,
        stain_vocab_path=args.stain_vocab,
        train_mode=False,
        max_patches_per_stain=args.max_patches,
    )
    ds_test = MultiStainPatientDataset(
        manifest_path=args.test_manifest,
        stain_vocab_path=args.stain_vocab,
        train_mode=False,
        max_patches_per_stain=args.max_patches,
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_patient_batch)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_patient_batch)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_patient_batch)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = stain_vocab.get("HE", 0)

    model = MMKidneyWSIModel(
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        stain_vocab_size=len(stain_vocab),
        stain_emb_dim=args.stain_emb_dim,
        label_dims=(4, 5, 8),
        num_heads=args.num_heads,
        num_set_layers=args.num_set_layers,
        dropout=args.dropout,
        he_stain_id=he_stain_id,
        use_he_adapter=True,
    ).to(device)

    asl = AsymmetricLossMultiLabel()
    hier = HierarchicalConsistencyLoss()
    optim = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(optim, epochs=args.epochs, warmup=5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early = EarlyStopper(patience=args.early_stop_patience, mode="max")

    best = -1e9
    best_path = out_dir / "best_wsi.pt"
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"train-wsi {ep}/{args.epochs}")
        for batch in pbar:
            b = move_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(b)
                loss, logs = compute_supervised_loss(out, b["labels"], asl_fn=asl, hier_fn=hier)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{(sum(losses)/max(1,len(losses))):.4f}")

        sched.step()
        val_m = evaluate(model, val_loader, device)
        score = float(val_m["mean_macro_auroc"])
        save_json(str(out_dir / f"val_epoch_{ep:03d}.json"), val_m)
        print(f"[ep {ep}] train_loss={(sum(losses)/max(1,len(losses))):.4f} val_mean_macro_auroc={score:.4f}")

        if score > best:
            best = score
            torch.save(model.state_dict(), best_path)
            save_json(str(out_dir / "val_best.json"), val_m)

        if early.step(score):
            print("[early-stop] triggered")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_m = evaluate(model, test_loader, device)
    save_json(str(out_dir / "test_metrics.json"), test_m)
    print("[done] best:", float(best), "test_mean_macro_auroc:", float(test_m["mean_macro_auroc"]))


if __name__ == "__main__":
    main()
