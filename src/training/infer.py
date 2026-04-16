import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.collate import collate_patient_batch
from src.datasets.patient_dataset import MultiStainPatientDataset
from src.models.teacher_model import MMKidneyTeacherModel
from src.models.wsi_model import MMKidneyWSIModel


def _to_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def _build_model(args, stain_vocab, sample):
    he_stain_id = stain_vocab.get("HE", 0)
    if args.model_type == "wsi":
        model = MMKidneyWSIModel(
            feat_dim=args.feat_dim,
            proj_dim=args.proj_dim,
            stain_vocab_size=len(stain_vocab),
            stain_emb_dim=args.stain_emb_dim,
            label_dims=(4, 5, 8),
            he_stain_id=he_stain_id,
            use_he_adapter=True,
        )
    else:
        tab_in = int(sample["tabular"].shape[0]) if sample.get("tabular") is not None else 1
        lab_in = int(sample["lab"].shape[0]) if sample.get("lab") is not None else 1
        model = MMKidneyTeacherModel(
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
        )
    return model


def main():
    p = argparse.ArgumentParser("Inference + explanation export")
    p.add_argument("--model-type", choices=["wsi", "teacher"], default="wsi")
    p.add_argument("--manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", default="outputs/explanations")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--feat-dim", type=int, default=768)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--stain-emb-dim", type=int, default=64)
    p.add_argument("--tab-dim", type=int, default=128)
    p.add_argument("--lab-dim", type=int, default=128)
    p.add_argument("--fused-dim", type=int, default=256)
    p.add_argument("--topk", type=int, default=20)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)

    ds = MultiStainPatientDataset(
        manifest_path=args.manifest,
        stain_vocab_path=args.stain_vocab,
        tabular_feature_csv=args.tabular_features if args.model_type == "teacher" else None,
        lab_feature_csv=args.lab_features if args.model_type == "teacher" else None,
        train_mode=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_patient_batch)

    model = _build_model(args, stain_vocab, ds[0])
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.to(device).eval()

    with torch.no_grad():
        for batch in loader:
            # move nested
            b = dict(batch)
            b["labels"] = {k: v.to(device) for k, v in batch["labels"].items()}
            b["modality_mask"] = batch["modality_mask"].to(device)
            if batch.get("tabular") is not None:
                b["tabular"] = batch["tabular"].to(device)
            if batch.get("tabular_mask") is not None:
                b["tabular_mask"] = batch["tabular_mask"].to(device)
            if batch.get("tabular_group_ids") is not None:
                b["tabular_group_ids"] = batch["tabular_group_ids"].to(device)
            if batch.get("lab") is not None:
                b["lab"] = batch["lab"].to(device)
            if batch.get("lab_mask") is not None:
                b["lab_mask"] = batch["lab_mask"].to(device)
            wsi = []
            for p_ in batch["wsi"]:
                stains = []
                for s in p_["stains"]:
                    ss = dict(s)
                    ss["features"] = s["features"].to(device)
                    if s.get("coords") is not None:
                        ss["coords"] = s["coords"].to(device)
                    stains.append(ss)
                wsi.append({"stains": stains})
            b["wsi"] = wsi

            out = model(b)

            probs = {
                "L1": torch.sigmoid(out["logits_L1"]).cpu(),
                "L2": torch.sigmoid(out["logits_L2"]).cpu(),
                "L3": torch.sigmoid(out["logits_L3"]).cpu(),
            }

            for i, sid in enumerate(batch["sample_ids"]):
                aux_i = out.get("aux", [None] * len(batch["sample_ids"]))[i]
                patch_topk = {}
                stain_weights = {}
                if aux_i is not None:
                    for sname, attn in aux_i.get("patch_attn", {}).items():
                        if attn is None:
                            continue
                        topv, topi = torch.topk(attn, k=min(args.topk, attn.shape[0]))
                        patch_topk[sname] = {
                            "topk_index": topi.detach().cpu().tolist(),
                            "topk_weight": topv.detach().cpu().tolist(),
                        }
                    sf = aux_i.get("stain_fusion_attn")
                    if sf is not None:
                        names = aux_i.get("stain_names", [])
                        vals = sf.detach().cpu().tolist()
                        stain_weights = {names[j]: vals[j] for j in range(min(len(names), len(vals)))}

                gates = None
                if "gates" in out:
                    gates = out["gates"][i].detach().cpu().tolist()

                payload = {
                    "sample_id": sid,
                    "prob_L1": probs["L1"][i].tolist(),
                    "prob_L2": probs["L2"][i].tolist(),
                    "prob_L3": probs["L3"][i].tolist(),
                    "stain_patch_attention_topk": patch_topk,
                    "stain_fusion_weights": stain_weights,
                    "modality_gates": gates,
                }
                with (out_dir / f"{sid}.json").open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

    print("[infer] outputs:", out_dir)


if __name__ == "__main__":
    main()
