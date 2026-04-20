import argparse
import json
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.collate import collate_patient_batch
from src.datasets.patient_dataset import MultiStainPatientDataset
from src.losses.asymmetric_loss import AsymmetricLossMultiLabel
from src.losses.distill_loss import compute_supervised_loss, distill_kl
from src.losses.hierarchical_consistency import HierarchicalConsistencyLoss
from src.models.teacher_model import MMKidneyTeacherModel
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
    p = argparse.ArgumentParser("Distill tri-modal teacher into WSI student")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--trimodal-manifest", default="data/processed/manifests/trimodal_manifest.jsonl")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--student-init-ckpt", default="")
    p.add_argument("--out-dir", default="outputs/distill")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
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
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = stain_vocab.get("HE", 0)

    ds_train = MultiStainPatientDataset(
        manifest_path=args.train_manifest,
        stain_vocab_path=args.stain_vocab,
        train_mode=True,
        stain_dropout_p=0.2,
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
    ds_trimodal = MultiStainPatientDataset(
        manifest_path=args.trimodal_manifest,
        stain_vocab_path=args.stain_vocab,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        train_mode=True,
        max_patches_per_stain=args.max_patches,
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_patient_batch)
    trimodal_loader = DataLoader(ds_trimodal, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_patient_batch)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_patient_batch)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_patient_batch)

    student = MMKidneyWSIModel(
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        stain_vocab_size=len(stain_vocab),
        stain_emb_dim=args.stain_emb_dim,
        label_dims=(4, 5, 8),
        he_stain_id=he_stain_id,
        use_he_adapter=True,
    ).to(device)
    if args.student_init_ckpt and Path(args.student_init_ckpt).exists():
        student.load_state_dict(torch.load(args.student_init_ckpt, map_location=device), strict=False)

    # infer tab/lab dims from trimodal
    one = ds_trimodal[0]
    tab_in = int(one["tabular"].shape[0]) if one.get("tabular") is not None else 1
    lab_in = int(one["lab"].shape[0]) if one.get("lab") is not None else 1

    teacher = MMKidneyTeacherModel(
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        stain_vocab_size=len(stain_vocab),
        stain_emb_dim=args.stain_emb_dim,
        tab_in_dim=tab_in,
        lab_in_dim=lab_in,
        tab_dim=args.tab_dim,
        lab_dim=args.lab_dim,
        fused_dim=args.fused_dim,
        label_dims=(4, 5, 8),
        he_stain_id=he_stain_id,
        use_he_adapter=True,
    ).to(device)
    t_state = torch.load(args.teacher_ckpt, map_location=device)
    model_state = teacher.state_dict()
    copied = {k: v for k, v in t_state.items() if k in model_state and model_state[k].shape == v.shape}
    model_state.update(copied)
    teacher.load_state_dict(model_state)
    print(f"[distill] loaded {len(copied)} compatible teacher params from {args.teacher_ckpt}")
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False

    # projection for repr alignment
    student_proj = torch.nn.Linear(args.proj_dim, args.fused_dim).to(device)

    asl = AsymmetricLossMultiLabel()
    hier = HierarchicalConsistencyLoss()
    params = list(student.parameters()) + list(student_proj.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = build_scheduler(optim, epochs=args.epochs, warmup=5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early = EarlyStopper(patience=15, mode="max")

    best = -1e9
    best_path = out_dir / "best_distilled_student.pt"

    for ep in range(1, args.epochs + 1):
        student.train()
        student_proj.train()
        losses = []

        tri_iter = cycle(trimodal_loader)
        pbar = tqdm(train_loader, desc=f"train-distill {ep}/{args.epochs}")
        for batch_wsi in pbar:
            b_wsi = move_to_device(batch_wsi, device)
            b_tri = move_to_device(next(tri_iter), device)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                # supervised on full WSI
                out_wsi = student(b_wsi)
                loss_sup_wsi, _ = compute_supervised_loss(out_wsi, b_wsi["labels"], asl_fn=asl, hier_fn=hier)

                # supervised + distill on trimodal bridge
                out_s_tri = student(b_tri)
                with torch.no_grad():
                    out_t_tri = teacher(b_tri)
                loss_sup_tri, _ = compute_supervised_loss(out_s_tri, b_tri["labels"], asl_fn=asl, hier_fn=hier)
                loss_kl = distill_kl(out_s_tri["logits_L3"], out_t_tri["logits_L3"], T=args.T)
                loss_mse = F.mse_loss(student_proj(out_s_tri["patient_repr"]), out_t_tri["fused_repr"].detach())

                loss = args.gamma * (loss_sup_wsi + loss_sup_tri) + args.alpha * loss_kl + args.beta * loss_mse

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{(sum(losses)/max(1,len(losses))):.4f}")

        sched.step()
        val_m = evaluate(student, val_loader, device)
        score = float(val_m["mean_macro_auroc"])
        save_json(str(out_dir / f"val_epoch_{ep:03d}.json"), val_m)
        if score > best:
            best = score
            torch.save(student.state_dict(), best_path)
            save_json(str(out_dir / "val_best.json"), val_m)
        print(f"[ep {ep}] loss={(sum(losses)/max(1,len(losses))):.4f} val={score:.4f}")
        if early.step(score):
            print("[early-stop] triggered")
            break

    student.load_state_dict(torch.load(best_path, map_location=device))
    test_m = evaluate(student, test_loader, device)
    save_json(str(out_dir / "test_metrics.json"), test_m)
    print("[done] best:", float(best), "test:", float(test_m["mean_macro_auroc"]))


if __name__ == "__main__":
    main()
