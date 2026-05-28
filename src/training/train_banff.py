import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.banff.schema import BANFF_TASKS, CI_CT_STAIN_MODES, resolve_ci_ct_stains
from src.datasets.banff_dataset import BanffPatientDataset
from src.datasets.collate import collate_banff_patient_batch
from src.losses.banff_loss import compute_banff_loss
from src.models.banff_model import BanffFirstWSIModel
from src.training.banff_metrics import summarize_banff_metrics
from src.training.callbacks import EarlyStopper
from src.training.common import save_json
from src.training.optimizer import build_optimizer
from src.training.schedulers import build_scheduler
from src.utils.seed import set_seed


DEFAULT_CORE_TASKS = ["ci", "ct", "c4d", "pvl", "cg", "g"]
DEFAULT_BINARY_FIRST_TASKS = ["c4d", "pvl", "cg", "g"]


def parse_tasks(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(BANFF_TASKS)
    tasks = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in tasks if x not in BANFF_TASKS]
    if unknown:
        raise ValueError(f"Unknown Banff tasks: {unknown}")
    return tasks


def parse_binary_tasks(value: str, tasks: List[str]) -> List[str]:
    value = value.strip()
    if not value or value.lower() == "none":
        return []
    if value.lower() == "auto":
        return [task for task in DEFAULT_BINARY_FIRST_TASKS if task in tasks]
    binary_tasks = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in binary_tasks if x not in tasks]
    if unknown:
        raise ValueError(f"Binary tasks must be included in --tasks. Unknown: {unknown}")
    return binary_tasks


def banff_task_config(
    tasks: List[str],
    binary_tasks: List[str] = None,
    ci_ct_stain_mode: str = "masson_he",
) -> Dict[str, Dict]:
    binary_set = set(binary_tasks or [])
    ci_ct_stains = resolve_ci_ct_stains(ci_ct_stain_mode)
    config = {}
    for name in tasks:
        stains = ci_ct_stains if name in {"ci", "ct"} else list(BANFF_TASKS[name].stains)
        config[name] = {
            "num_classes": 2 if name in binary_set else BANFF_TASKS[name].num_classes,
            "stains": stains,
        }
    return config


def move_banff_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = dict(batch)
    out["banff_labels"] = {k: v.to(device) for k, v in batch["banff_labels"].items()}
    out["banff_masks"] = {k: v.to(device) for k, v in batch["banff_masks"].items()}
    wsi = []
    for patient in batch["wsi"]:
        stains = []
        for stain in patient["stains"]:
            item = dict(stain)
            item["features"] = stain["features"].to(device)
            if stain.get("coords") is not None:
                item["coords"] = stain["coords"].to(device)
            stains.append(item)
        wsi.append({"stains": stains})
    out["wsi"] = wsi
    return out


def effective_masks(batch: Dict, out: Dict) -> Dict[str, torch.Tensor]:
    masks = {}
    for task, mask in batch["banff_masks"].items():
        task_present = out["banff_task_masks"].get(task)
        masks[task] = mask if task_present is None else mask * task_present
    return masks


def evaluate(model, loader, device):
    model.eval()
    probs = {}
    labels = {}
    masks = {}
    with torch.no_grad():
        for batch in loader:
            b = move_banff_batch_to_device(batch, device)
            out = model(b)
            eff_masks = effective_masks(b, out)
            for task, logits in out["banff_logits"].items():
                probs.setdefault(task, []).append(torch.softmax(logits, dim=1).detach().cpu())
                labels.setdefault(task, []).append(b["banff_labels"][task].detach().cpu())
                masks.setdefault(task, []).append(eff_masks[task].detach().cpu())
    probs = {k: torch.cat(v, dim=0) for k, v in probs.items()}
    labels = {k: torch.cat(v, dim=0) for k, v in labels.items()}
    masks = {k: torch.cat(v, dim=0) for k, v in masks.items()}
    return summarize_banff_metrics(probs, labels, masks)


def main():
    parser = argparse.ArgumentParser("Train Banff-first multitask WSI model")
    parser.add_argument("--train-manifest", default="data/processed_banff/manifests/train_banff_manifest.jsonl")
    parser.add_argument("--val-manifest", default="data/processed_banff/manifests/val_banff_manifest.jsonl")
    parser.add_argument("--test-manifest", default="data/processed_banff/manifests/test_banff_manifest.jsonl")
    parser.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    parser.add_argument("--out-dir", default="outputs/banff_first")
    parser.add_argument("--tasks", default=",".join(DEFAULT_CORE_TASKS))
    parser.add_argument("--binary-tasks", default="auto")
    parser.add_argument("--ci-ct-stain-mode", choices=sorted(CI_CT_STAIN_MODES), default="masson_he")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--stain-dropout", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--feat-dim", type=int, default=768)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--stain-emb-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-set-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    binary_tasks = parse_binary_tasks(args.binary_tasks, tasks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_config = banff_task_config(tasks, binary_tasks, ci_ct_stain_mode=args.ci_ct_stain_mode)
    save_json(
        str(out_dir / "tasks.json"),
        {
            "tasks": tasks,
            "binary_tasks": binary_tasks,
            "ci_ct_stain_mode": args.ci_ct_stain_mode,
            "task_config": task_config,
        },
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = BanffPatientDataset(
        manifest_path=args.train_manifest,
        stain_vocab_path=args.stain_vocab,
        tasks=tasks,
        binary_tasks=binary_tasks,
        stain_dropout_p=args.stain_dropout,
        train_mode=True,
        max_patches_per_stain=args.max_patches,
    )
    ds_val = BanffPatientDataset(
        manifest_path=args.val_manifest,
        stain_vocab_path=args.stain_vocab,
        tasks=tasks,
        binary_tasks=binary_tasks,
        train_mode=False,
        max_patches_per_stain=args.max_patches,
    )
    ds_test = BanffPatientDataset(
        manifest_path=args.test_manifest,
        stain_vocab_path=args.stain_vocab,
        tasks=tasks,
        binary_tasks=binary_tasks,
        train_mode=False,
        max_patches_per_stain=args.max_patches,
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_banff_patient_batch)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_banff_patient_batch)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_banff_patient_batch)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = int(stain_vocab.get("HE", 0))

    model = BanffFirstWSIModel(
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        stain_vocab_size=len(stain_vocab),
        stain_emb_dim=args.stain_emb_dim,
        num_heads=args.num_heads,
        num_set_layers=args.num_set_layers,
        dropout=args.dropout,
        he_stain_id=he_stain_id,
        use_he_adapter=True,
        tasks=task_config,
    ).to(device)

    optim = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(optim, epochs=args.epochs, warmup=5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early = EarlyStopper(patience=args.early_stop_patience, mode="max")

    best = -1e9
    best_path = out_dir / "best_banff.pt"
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"train-banff {ep}/{args.epochs}")
        for batch in pbar:
            b = move_banff_batch_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(b)
                masks = effective_masks(b, out)
                loss, logs = compute_banff_loss(out["banff_logits"], b["banff_labels"], masks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{sum(losses) / max(1, len(losses)):.4f}")

        sched.step()
        val_m = evaluate(model, val_loader, device)
        score = val_m.get("mean_macro_recall") or val_m.get("mean_accuracy") or 0.0
        save_json(str(out_dir / f"val_epoch_{ep:03d}.json"), val_m)
        print(f"[ep {ep}] train_loss={sum(losses)/max(1,len(losses)):.4f} val_score={float(score):.4f}")

        if float(score) > best:
            best = float(score)
            torch.save(model.state_dict(), best_path)
            save_json(str(out_dir / "val_best.json"), val_m)

        if early.step(float(score)):
            print("[early-stop] triggered")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_m = evaluate(model, test_loader, device)
    save_json(str(out_dir / "test_metrics.json"), test_m)
    print("[done] best:", float(best), "test_mean_macro_recall:", test_m.get("mean_macro_recall"))


if __name__ == "__main__":
    main()
