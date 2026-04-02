import argparse
import json
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from data import make_dataloader
from models import HierMultiLabelNet, HierMultiLabelMIL, HierMultiLabelMIL_MultiStain
from losses import HierMultiLabelLoss, attention_diversity_loss
from hierarchy import build_parent_map_auto
from utils import TrainConfig, set_seed, evaluate, save_json, move_labels_to_device


def train_loop(
    train_loader,
    val_loader,
    model,
    loss_fn,
    device,
    epochs=50,
    lr=2e-4,
    weight_decay=1e-4,
    out_dir="./output",
    lambda_div=0.0,
    is_mil=False,
    is_mil_ms=False,
):
    os.makedirs(out_dir, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_score = -1.0
    best_path = os.path.join(out_dir, "best.pt")

    parent_map, _ = build_parent_map_auto()

    for ep in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Train {ep+1}/{epochs}", ncols=120, leave=False)

        for batch in pbar:
            if is_mil or is_mil_ms:
                xs, labels, ids, extra = batch
                if is_mil:
                    xs = [x.to(device) for x in xs]
                else:
                    xs = [{k: v.to(device) for k, v in x.items()} for x in xs]
                labels = move_labels_to_device(labels, device)
                extra = extra.to(device) if extra is not None else None
                logits, aux = model(xs, extra=extra, return_attn=True)
                loss = loss_fn(logits, labels, parent_map=parent_map)
                if lambda_div > 0.0 and aux is not None and "attn" in aux:
                    loss = loss + attention_diversity_loss(aux["attn"], lambda_div=lambda_div)
            else:
                feats, labels, ids, extra = batch
                feats = feats.to(device)
                labels = move_labels_to_device(labels, device)
                extra = extra.to(device) if extra is not None else None
                logits = model(feats, extra=extra)
                loss = loss_fn(logits, labels, parent_map=parent_map)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        scheduler.step()

        if is_mil or is_mil_ms:
            def forward_fn_eval(xs, extra):
                out = model(xs, extra=extra, return_attn=False)
                return out[0] if isinstance(out, tuple) else out
        else:
            def forward_fn_eval(x, extra):
                out = model(x, extra=extra)
                return out[0] if isinstance(out, tuple) else out

        metrics = evaluate(model, val_loader, forward_fn_eval, device)
        score = metrics.get("L3", {}).get("AUROC", float("nan"))
        if score != score:
            score = -running

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            save_json(os.path.join(out_dir, "val_best_metrics.json"), metrics)

        save_json(os.path.join(out_dir, f"val_epoch_{ep + 1:03d}.json"), metrics)

        line = (
            f"[{ep + 1:03d}/{epochs}] loss={running / len(train_loader):.4f} | "
            f"L1 AUC={metrics.get('L1', {}).get('AUROC', 'nan'):.4f} "
            f"AUPRC={metrics.get('L1', {}).get('AUPRC', 'nan'):.4f} "
            f"F1={metrics.get('L1', {}).get('F1@0.5', 'nan'):.4f} | "
            f"L2 AUC={metrics.get('L2', {}).get('AUROC', 'nan'):.4f} "
            f"AUPRC={metrics.get('L2', {}).get('AUPRC', 'nan'):.4f} "
            f"F1={metrics.get('L2', {}).get('F1@0.5', 'nan'):.4f} | "
            f"L3 AUC={metrics.get('L3', {}).get('AUROC', 'nan'):.4f} "
            f"AUPRC={metrics.get('L3', {}).get('AUPRC', 'nan'):.4f} "
            f"F1={metrics.get('L3', {}).get('F1@0.5', 'nan'):.4f}"
        )
        print(line)

    return best_path


def run_training(conf: TrainConfig, train_samples, val_samples, test_samples=None):
    set_seed(1449)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mil = conf.mode == "mil"
    is_mil_ms = conf.mode == "mil_ms"

    train_loader = make_dataloader(
        train_samples,
        mode=conf.mode,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        max_patches=conf.max_patches,
    )
    val_loader = make_dataloader(
        val_samples,
        mode=conf.mode,
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        max_patches=conf.max_patches,
    )

    if is_mil:
        model = HierMultiLabelMIL(extra_dim=conf.extra_dim).to(device)
    elif is_mil_ms:
        model = HierMultiLabelMIL_MultiStain(extra_dim=conf.extra_dim).to(device)
    else:
        model = HierMultiLabelNet(extra_dim=conf.extra_dim).to(device)

    loss_fn = HierMultiLabelLoss(lambda_hier=conf.lambda_hier)

    best_path = train_loop(
        train_loader,
        val_loader,
        model,
        loss_fn,
        device,
        epochs=conf.epochs,
        lr=conf.lr,
        weight_decay=conf.weight_decay,
        out_dir=conf.out_dir,
        lambda_div=conf.lambda_div,
        is_mil=is_mil,
        is_mil_ms=is_mil_ms,
    )
    print(f"[OK] best model saved at: {best_path}")

    if test_samples:
        test_loader = make_dataloader(
            test_samples,
            mode=conf.mode,
            batch_size=conf.batch_size,
            shuffle=False,
            num_workers=conf.num_workers,
            max_patches=conf.max_patches,
        )
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state, strict=True)

        if is_mil or is_mil_ms:
            def forward_fn_eval(xs, extra):
                out = model(xs, extra=extra, return_attn=False)
                return out[0] if isinstance(out, tuple) else out
        else:
            def forward_fn_eval(x, extra):
                out = model(x, extra=extra)
                return out[0] if isinstance(out, tuple) else out

        test_metrics = evaluate(model, test_loader, forward_fn_eval, device)
        save_json(os.path.join(conf.out_dir, "test_metrics.json"), test_metrics)

        print(
            "[TEST] "
            f"L1 AUC={test_metrics.get('L1', {}).get('AUROC', 'nan'):.4f} "
            f"AUPRC={test_metrics.get('L1', {}).get('AUPRC', 'nan'):.4f} "
            f"F1={test_metrics.get('L1', {}).get('F1@0.5', 'nan'):.4f} | "
            f"L2 AUC={test_metrics.get('L2', {}).get('AUROC', 'nan'):.4f} "
            f"AUPRC={test_metrics.get('L2', {}).get('AUPRC', 'nan'):.4f} "
            f"F1={test_metrics.get('L2', {}).get('F1@0.5', 'nan'):.4f} | "
            f"L3 AUC={test_metrics.get('L3', {}).get('AUROC', 'nan'):.4f} "
            f"AUPRC={test_metrics.get('L3', {}).get('AUPRC', 'nan'):.4f} "
            f"F1={test_metrics.get('L3', {}).get('F1@0.5', 'nan'):.4f}"
        )


def main():
    p = argparse.ArgumentParser("Clean hierarchical multi-label training")
    p.add_argument("--mode", choices=["vector", "mil", "mil_ms"], default="vector")
    p.add_argument("--train-json", required=True)
    p.add_argument("--val-json", required=True)
    p.add_argument("--test-json", default=None)
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--lambda-hier", type=float, default=0.2)
    p.add_argument("--lambda-div", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--extra-dim", type=int, default=0, help="optional extra feature dimension for multimodal fusion")
    p.add_argument("--max-patches", type=int, default=None, help="optional cap per bag/stain")
    args = p.parse_args()

    with open(args.train_json, "r") as f:
        train_samples = json.load(f)
    with open(args.val_json, "r") as f:
        val_samples = json.load(f)
    test_samples = None
    if args.test_json:
        with open(args.test_json, "r") as f:
            test_samples = json.load(f)

    conf = TrainConfig(
        mode=args.mode,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        lambda_hier=args.lambda_hier,
        lambda_div=args.lambda_div,
        num_workers=args.num_workers,
        extra_dim=args.extra_dim,
        max_patches=args.max_patches,
    )

    run_training(conf, train_samples, val_samples, test_samples=test_samples)


if __name__ == "__main__":
    main()
