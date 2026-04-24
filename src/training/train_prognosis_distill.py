import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.prognosis_dataset import PrognosisDataset, collate_prognosis_batch
from src.models.prognosis_heads import cox_ph_loss, discrete_time_nll
from src.models.prognosis_teacher_model import MMKidneyPrognosisTeacher
from src.training.callbacks import EarlyStopper
from src.training.train_prognosis import _move_to_device, _parse_bins, _write_json, _write_predictions, evaluate
from src.utils.seed import set_seed


def _kl_discrete(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    ps = F.log_softmax(student_logits / T, dim=-1)
    pt = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(ps, pt, reduction="batchmean") * (T * T)


def _build_dataset(args, split: str, exclude_feature_prefixes: List[str]):
    manifest_paths = [args.train_manifest, args.val_manifest, args.test_manifest]
    return PrognosisDataset(
        prognosis_cohort_csv=args.prognosis_cohort,
        feature_manifest_json=args.feature_manifest,
        tabular_feature_csv=args.tabular_features,
        lab_feature_csv=args.lab_features,
        stain_vocab_path=args.stain_vocab,
        manifest_paths=manifest_paths,
        split=split,
        max_patches_per_stain=args.max_patches,
        exclude_feature_prefixes=exclude_feature_prefixes,
    )


def main():
    p = argparse.ArgumentParser("Distill prognosis teacher into student")
    p.add_argument("--prognosis-cohort", default="data/processed/prognosis_cohort.csv")
    p.add_argument("--feature-manifest", default="data/processed/prognosis_feature_manifest.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--train-manifest", default="data/processed/manifests/train_manifest.jsonl")
    p.add_argument("--val-manifest", default="data/processed/manifests/val_manifest.jsonl")
    p.add_argument("--test-manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--out-dir", default="outputs/prognosis_distill")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
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
    p.add_argument("--teacher-profile", choices=["full_v1", "full_v2"], default="full_v2")
    p.add_argument("--student-profile", choices=["baseline_v1", "full_v1"], default="baseline_v1")
    p.add_argument("--strict-treatment-history", action="store_true")
    p.add_argument("--alpha-kd", type=float, default=1.0)
    p.add_argument("--beta-embed", type=float, default=0.5)
    p.add_argument("--gamma-sup", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bins_days = _parse_bins(args.time_bins_days)
    horizons = _parse_bins(args.horizons_days)
    n_bins = len(bins_days) + 1

    strict_treat = bool(args.strict_treatment_history or args.teacher_profile == "full_v2")
    ex_teacher = ["treatment_"] if strict_treat else None

    ds_train = _build_dataset(args, "train", ex_teacher)
    ds_val = _build_dataset(args, "val", ex_teacher)
    ds_test = _build_dataset(args, "test", ex_teacher)
    if len(ds_train) == 0:
        raise RuntimeError("Empty train split.")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_prognosis_batch)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_prognosis_batch)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_prognosis_batch)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    he_stain_id = stain_vocab.get("HE", 0)
    one = ds_train[0]
    tab_in = int(one["tabular"].shape[0])
    lab_in = int(one["lab"].shape[0])

    teacher = MMKidneyPrognosisTeacher(
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
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device), strict=True)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False

    use_cp_student = args.student_profile == "full_v1"
    student = MMKidneyPrognosisTeacher(
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
        use_clinicopath_encoder=use_cp_student,
    ).to(device)

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early = EarlyStopper(patience=args.early_stop_patience, mode="max")

    best = -1e9
    best_path = out_dir / "best_distilled_student.pt"
    _write_json(
        out_dir / "config.json",
        {
            **vars(args),
            "strict_treatment_history_effective": strict_treat,
            "student_use_clinicopath_encoder": use_cp_student,
        },
    )

    for ep in range(1, args.epochs + 1):
        student.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"train-prognosis-distill {ep}/{args.epochs}")
        for batch in pbar:
            b = _move_to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out_s = student(b)
                with torch.no_grad():
                    out_t = teacher(b)

                if args.survival_head == "cox":
                    sup = cox_ph_loss(out_s["risk"], b["event"], b["time_days"])
                    kd = F.mse_loss(out_s["risk"], out_t["risk"])
                else:
                    sup = discrete_time_nll(out_s["hazard_logits"], b["event"], b["time_days"], bins_days=bins_days)
                    kd = _kl_discrete(out_s["hazard_logits"], out_t["hazard_logits"], T=args.temperature)
                emb = F.mse_loss(out_s["fused_repr"], out_t["fused_repr"].detach())
                loss = args.gamma_sup * sup + args.alpha_kd * kd + args.beta_embed * emb

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{(sum(losses)/max(1,len(losses))):.4f}")

        val_out = evaluate(student, val_loader, device, args.survival_head, bins_days=bins_days, horizons=horizons)
        score = float(val_out["metrics"]["c_index"])
        _write_json(out_dir / f"val_epoch_{ep:03d}.json", val_out["metrics"])
        if score > best:
            best = score
            torch.save(student.state_dict(), best_path)
            _write_json(out_dir / "val_best.json", val_out["metrics"])
            _write_predictions(out_dir / "val_best_predictions.csv", val_out["predictions"])
        print(f"[ep {ep}] loss={(sum(losses)/max(1,len(losses))):.4f} val_c_index={score:.4f}")
        if early.step(score):
            print("[early-stop] triggered")
            break

    student.load_state_dict(torch.load(best_path, map_location=device))
    test_out = evaluate(student, test_loader, device, args.survival_head, bins_days=bins_days, horizons=horizons)
    _write_json(out_dir / "test_metrics.json", test_out["metrics"])
    _write_predictions(out_dir / "test_predictions.csv", test_out["predictions"])
    print("[done] best_val_c_index:", float(best), "test_c_index:", float(test_out["metrics"]["c_index"]))


if __name__ == "__main__":
    main()
