import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.datasets.collate import collate_patient_batch
from src.datasets.patient_dataset import MultiStainPatientDataset
from src.models.teacher_model import MMKidneyTeacherModel
from src.models.wsi_model import MMKidneyWSIModel

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    HAVE_SK = True
except Exception:
    HAVE_SK = False


def _safe_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_auroc(y_true, y_prob):
    if not HAVE_SK:
        return float("nan")
    try:
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _safe_auprc(y_true, y_prob):
    if not HAVE_SK:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _build_model(args, stain_vocab, sample):
    he_stain_id = stain_vocab.get("HE", 0)
    if args.model_type == "wsi":
        return MMKidneyWSIModel(
            feat_dim=args.feat_dim,
            proj_dim=args.proj_dim,
            stain_vocab_size=len(stain_vocab),
            stain_emb_dim=args.stain_emb_dim,
            label_dims=(4, 5, 8),
            he_stain_id=he_stain_id,
            use_he_adapter=True,
        )

    tab_in = int(sample["tabular"].shape[0]) if sample.get("tabular") is not None else 1
    lab_in = int(sample["lab"].shape[0]) if sample.get("lab") is not None else 1
    return MMKidneyTeacherModel(
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


def _move_nested_batch_to_device(batch: Dict, device: torch.device) -> Dict:
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
    return b


def _model_class_mapping(class_name_path: str) -> Dict[str, List[Dict]]:
    with open(class_name_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}
    normal = "Normal Biopsy Or Nonspecific Changes"
    for lv in ["L1", "L2", "L3"]:
        raw_names = list(raw.get(lv, []))
        model_names = [x for x in raw_names if x != normal]
        mapping = []
        for i, name in enumerate(model_names):
            raw_idx = raw_names.index(name) if name in raw_names else -1
            mapping.append({"model_index": i, "class_name": name, "raw_index": raw_idx})
        out[lv] = mapping
    return out


def _classwise_metrics(y_true: List[List[int]], y_prob: List[List[float]], names: List[str], thr: float = 0.5):
    # y shape: [N, C]
    n = len(y_true)
    c = len(y_true[0]) if n > 0 else 0
    out = {}
    for i in range(c):
        yt = [int(r[i]) for r in y_true]
        yp = [float(r[i]) for r in y_prob]
        yh = [1 if p >= thr else 0 for p in yp]

        tp = sum(1 for a, b in zip(yt, yh) if a == 1 and b == 1)
        tn = sum(1 for a, b in zip(yt, yh) if a == 0 and b == 0)
        fp = sum(1 for a, b in zip(yt, yh) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(yt, yh) if a == 1 and b == 0)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        acc = (tp + tn) / max(1, len(yt))

        name = names[i] if i < len(names) else f"class_{i}"
        out[str(i)] = {
            "class_name": name,
            "support_pos": int(sum(yt)),
            "support_neg": int(len(yt) - sum(yt)),
            "prevalence": float(sum(yt) / max(1, len(yt))),
            "auroc": _safe_auroc(yt, yp),
            "auprc": _safe_auprc(yt, yp),
            "precision@thr": float(prec),
            "recall@thr": float(rec),
            "f1@thr": float(f1),
            "accuracy@thr": float(acc),
            "confusion@thr": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        }
    return out


def _nanmean(xs: List[float]) -> float:
    ys = [x for x in xs if not math.isnan(x)]
    if not ys:
        return float("nan")
    return float(sum(ys) / len(ys))


def _availability_summary(stain_counter: Counter, stain_per_sample: List[int]):
    return {
        "num_samples": int(len(stain_per_sample)),
        "avg_stains_per_sample": float(sum(stain_per_sample) / max(1, len(stain_per_sample))),
        "max_stains_per_sample": int(max(stain_per_sample) if stain_per_sample else 0),
        "min_stains_per_sample": int(min(stain_per_sample) if stain_per_sample else 0),
        "stain_presence_count": dict(sorted(stain_counter.items(), key=lambda x: x[0])),
    }


def main():
    p = argparse.ArgumentParser("Export complete raw analysis bundle (predictions, AUCs, attentions, mappings).")
    p.add_argument("--model-type", choices=["wsi", "teacher"], default="wsi")
    p.add_argument("--manifest", default="data/processed/manifests/test_manifest.jsonl")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--tabular-features", default="data/processed/tabular_features.csv")
    p.add_argument("--lab-features", default="data/processed/lab_features.csv")
    p.add_argument("--class-name-json", default="data/class_names.json")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", default="outputs/analysis_bundle")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--feat-dim", type=int, default=768)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--stain-emb-dim", type=int, default=64)
    p.add_argument("--tab-dim", type=int, default=128)
    p.add_argument("--lab-dim", type=int, default=128)
    p.add_argument("--fused-dim", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save-per-sample-files", action="store_true")
    p.add_argument("--topk", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_dir = out_dir / "sample_details"
    if args.save_per_sample_files:
        details_dir.mkdir(parents=True, exist_ok=True)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)
    stain_id_to_name = {int(v): k for k, v in stain_vocab.items()}

    ds = MultiStainPatientDataset(
        manifest_path=args.manifest,
        stain_vocab_path=args.stain_vocab,
        tabular_feature_csv=args.tabular_features if args.model_type == "teacher" else None,
        lab_feature_csv=args.lab_features if args.model_type == "teacher" else None,
        train_mode=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_patient_batch)

    model = _build_model(args, stain_vocab, ds[0])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    class_mapping = _model_class_mapping(args.class_name_json)
    class_names_by_level = {lv: [x["class_name"] for x in class_mapping[lv]] for lv in ["L1", "L2", "L3"]}

    preds = {"L1": [], "L2": [], "L3": []}
    trues = {"L1": [], "L2": [], "L3": []}
    logits_all = {"L1": [], "L2": [], "L3": []}
    sample_rows = []
    label_support = {lv: defaultdict(int) for lv in ["L1", "L2", "L3"]}
    stain_counter = Counter()
    stain_per_sample = []

    with torch.no_grad():
        for batch in loader:
            b = _move_nested_batch_to_device(batch, device)
            out = model(b)
            probs = {
                "L1": torch.sigmoid(out["logits_L1"]).detach().cpu(),
                "L2": torch.sigmoid(out["logits_L2"]).detach().cpu(),
                "L3": torch.sigmoid(out["logits_L3"]).detach().cpu(),
            }
            logits = {
                "L1": out["logits_L1"].detach().cpu(),
                "L2": out["logits_L2"].detach().cpu(),
                "L3": out["logits_L3"].detach().cpu(),
            }
            tr = {k: batch["labels"][k].detach().cpu() for k in ["L1", "L2", "L3"]}

            for lv in ["L1", "L2", "L3"]:
                preds[lv].extend(probs[lv].tolist())
                logits_all[lv].extend(logits[lv].tolist())
                trues[lv].extend(tr[lv].tolist())

            for i, sid in enumerate(batch["sample_ids"]):
                # sample-level label support
                for lv in ["L1", "L2", "L3"]:
                    y_i = tr[lv][i].tolist()
                    for c, v in enumerate(y_i):
                        if int(v) == 1:
                            label_support[lv][str(c)] += 1

                patient_stains = batch["wsi"][i]["stains"]
                stain_per_sample.append(len(patient_stains))
                for s in patient_stains:
                    stain_counter[s.get("name", "UNKNOWN")] += 1

                aux_i = out.get("aux", [None] * len(batch["sample_ids"]))[i]
                patch_attn_full = {}
                patch_topk = {}
                stain_fusion_weights = {}
                patch_coord_map = {}
                patch_feat_shape = {}

                # Build name -> coords/features from the original batch for precise patch index mapping.
                for s in patient_stains:
                    sname = s.get("name", "UNKNOWN")
                    coords = s.get("coords")
                    patch_coord_map[sname] = coords.tolist() if coords is not None else None
                    patch_feat_shape[sname] = list(s["features"].shape)

                if aux_i is not None:
                    for sname, attn in aux_i.get("patch_attn", {}).items():
                        if attn is None:
                            continue
                        av = attn.detach().cpu().tolist()
                        patch_attn_full[sname] = av
                        topv, topi = torch.topk(attn, k=min(args.topk, attn.shape[0]))
                        patch_topk[sname] = {
                            "topk_index": topi.detach().cpu().tolist(),
                            "topk_weight": topv.detach().cpu().tolist(),
                        }
                    sf = aux_i.get("stain_fusion_attn")
                    if sf is not None:
                        names = aux_i.get("stain_names", [])
                        vals = sf.detach().cpu().tolist()
                        stain_fusion_weights = {names[j]: vals[j] for j in range(min(len(names), len(vals)))}

                gates = out["gates"][i].detach().cpu().tolist() if "gates" in out else None
                cp_branch_gates = None
                cp_branch_present = None
                if out.get("clinicopath_aux") is not None:
                    cpa = out["clinicopath_aux"]
                    if cpa.get("clinicopath_branch_gates") is not None:
                        cp_branch_gates = cpa["clinicopath_branch_gates"][i].detach().cpu().tolist()
                    if cpa.get("clinicopath_branch_present") is not None:
                        cp_branch_present = cpa["clinicopath_branch_present"][i].detach().cpu().tolist()

                row = {
                    "sample_id": sid,
                    "split": batch["splits"][i],
                    "modality_mask": batch["modality_mask"][i].tolist(),
                    "label_true": {lv: tr[lv][i].tolist() for lv in ["L1", "L2", "L3"]},
                    "logits": {lv: logits[lv][i].tolist() for lv in ["L1", "L2", "L3"]},
                    "prob": {lv: probs[lv][i].tolist() for lv in ["L1", "L2", "L3"]},
                    "pred@thr": {lv: [1 if p >= args.threshold else 0 for p in probs[lv][i].tolist()] for lv in ["L1", "L2", "L3"]},
                    "stain_names_in_batch": [s.get("name", "UNKNOWN") for s in patient_stains],
                    "stain_ids_in_batch": [int(s.get("id", -1)) for s in patient_stains],
                    "stain_id_to_name_vocab": stain_id_to_name,
                    "patch_feature_shape_by_stain": patch_feat_shape,
                    "patch_index_to_coord_by_stain": patch_coord_map,
                    "patch_attention_full_by_stain": patch_attn_full,
                    "patch_attention_topk_by_stain": patch_topk,
                    "stain_fusion_weights": stain_fusion_weights,
                    "modality_gates": gates,
                    "clinicopath_branch_gates": cp_branch_gates,
                    "clinicopath_branch_present": cp_branch_present,
                }
                sample_rows.append(row)

                if args.save_per_sample_files:
                    _safe_json(details_dir / f"{sid}.json", row)

    # Metrics and summaries.
    metrics = {}
    for lv in ["L1", "L2", "L3"]:
        cls_metrics = _classwise_metrics(trues[lv], preds[lv], class_names_by_level[lv], thr=args.threshold)
        macro_auroc = _nanmean([v["auroc"] for v in cls_metrics.values()])
        macro_auprc = _nanmean([v["auprc"] for v in cls_metrics.values()])
        metrics[lv] = {
            "macro_auroc": float(macro_auroc),
            "macro_auprc": float(macro_auprc),
            "per_class": cls_metrics,
        }
    metrics["mean_macro_auroc"] = _nanmean([metrics["L1"]["macro_auroc"], metrics["L2"]["macro_auroc"], metrics["L3"]["macro_auroc"]])

    label_count_summary = {}
    for lv in ["L1", "L2", "L3"]:
        level_map = class_names_by_level[lv]
        label_count_summary[lv] = []
        for i, name in enumerate(level_map):
            pos = int(label_support[lv][str(i)])
            total = len(sample_rows)
            label_count_summary[lv].append(
                {
                    "class_index": i,
                    "class_name": name,
                    "positive_samples": pos,
                    "negative_samples": int(total - pos),
                    "prevalence": float(pos / max(1, total)),
                }
            )

    # Save bundle files.
    _safe_json(
        out_dir / "run_config.json",
        {
            "model_type": args.model_type,
            "manifest": args.manifest,
            "stain_vocab": args.stain_vocab,
            "tabular_features": args.tabular_features,
            "lab_features": args.lab_features,
            "ckpt": args.ckpt,
            "threshold": args.threshold,
            "device": str(device),
            "num_samples": len(sample_rows),
        },
    )
    _safe_json(out_dir / "class_index_mapping.json", class_mapping)
    _safe_json(out_dir / "metrics_detailed.json", metrics)
    _safe_json(out_dir / "class_sample_counts.json", label_count_summary)
    _safe_json(out_dir / "stain_availability_summary.json", _availability_summary(stain_counter, stain_per_sample))

    # jsonl with all raw per-sample outputs for analysis pipelines.
    with (out_dir / "sample_predictions_all_raw.jsonl").open("w", encoding="utf-8") as f:
        for r in sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[export] done")
    print("[export] out_dir:", out_dir)
    print("[export] samples:", len(sample_rows))
    print("[export] mean_macro_auroc:", metrics["mean_macro_auroc"])


if __name__ == "__main__":
    main()
