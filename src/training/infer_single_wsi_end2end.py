import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from src.datasets.h5_reader import H5BagReader
from src.datasets.stain_utils import normalize_stain_name
from src.models.wsi_model import MMKidneyWSIModel


def _run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _build_model(args, stain_vocab):
    he_stain_id = stain_vocab.get("HE", 0)
    model = MMKidneyWSIModel(
        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        stain_vocab_size=len(stain_vocab),
        stain_emb_dim=args.stain_emb_dim,
        label_dims=(4, 5, 8),
        he_stain_id=he_stain_id,
        use_he_adapter=True,
    )
    return model


def _find_feature_h5(job_dir: Path, mag: float, patch_size: int, overlap: int, patch_encoder: str, slide_stem: str) -> Path:
    mag_str = f"{float(mag):g}"
    feat_dir = job_dir / f"{mag_str}x_{patch_size}px_{overlap}px_overlap" / f"features_{patch_encoder}"
    if not feat_dir.exists():
        raise FileNotFoundError(f"Feature dir not found: {feat_dir}")

    exact = feat_dir / f"{slide_stem}.h5"
    if exact.exists():
        return exact

    cands = sorted(feat_dir.glob("*.h5"))
    if not cands:
        raise FileNotFoundError(f"No feature h5 found in: {feat_dir}")
    if len(cands) == 1:
        return cands[0]

    stem_low = slide_stem.lower()
    matched = [p for p in cands if p.stem.lower() == stem_low]
    if len(matched) == 1:
        return matched[0]
    raise RuntimeError(f"Multiple feature h5 found in {feat_dir}, cannot decide automatically: {[p.name for p in cands[:10]]}")


def main():
    p = argparse.ArgumentParser("End2End single-WSI inference: TRIDENT + CONCHv1.5 + MMKidney")
    p.add_argument("--slide-path", required=True, help="Input HE WSI path, e.g. .svs")
    p.add_argument("--ckpt", required=True, help="Trained MMKidney WSI checkpoint")
    p.add_argument("--stain-vocab", default="data/processed/stain_vocab.json")
    p.add_argument("--out-dir", default="outputs/end2end_single_wsi")
    p.add_argument("--sample-id", default="", help="Optional output sample_id; default=slide stem")
    p.add_argument("--stain-name", default="HE", help="Stain name for this single slide, default HE")
    p.add_argument("--topk", type=int, default=20)

    p.add_argument("--skip-trident", action="store_true", help="Skip TRIDENT run, read precomputed features from --trident-job-dir")
    p.add_argument("--trident-dir", default="external/TRIDENT")
    p.add_argument("--trident-job-dir", default="outputs/trident_single_wsi")
    p.add_argument("--trident-python", default=sys.executable, help="Python executable used to run TRIDENT")
    p.add_argument("--gpu", type=int, default=0, help="GPU index used by TRIDENT")
    p.add_argument("--segmenter", choices=["hest", "grandqc", "otsu"], default="hest")
    p.add_argument("--reader-type", choices=["openslide", "image", "cucim", "sdpc", "omezarr", "czi"], default=None)
    p.add_argument("--mag", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=512, help="CONCHv1.5 expects 512")
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--patch-encoder", default="conch_v15")
    p.add_argument("--trident-batch-size", type=int, default=32)
    p.add_argument("--seg-conf-thresh", type=float, default=0.5)
    p.add_argument("--remove-holes", action="store_true", default=False)
    p.add_argument("--remove-artifacts", action="store_true", default=False)
    p.add_argument("--remove-penmarks", action="store_true", default=False)

    p.add_argument("--feat-dim", type=int, default=768)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--stain-emb-dim", type=int, default=64)
    args = p.parse_args()

    slide_path = Path(args.slide_path).resolve()
    if not slide_path.exists():
        raise FileNotFoundError(f"slide not found: {slide_path}")
    sample_id = args.sample_id.strip() or slide_path.stem

    trident_job_dir = Path(args.trident_job_dir).resolve()
    trident_job_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_trident:
        run_single = Path(args.trident_dir).resolve() / "run_single_slide.py"
        if not run_single.exists():
            raise FileNotFoundError(f"TRIDENT entry not found: {run_single}")
        cmd = [
            args.trident_python,
            str(run_single),
            "--slide_path",
            str(slide_path),
            "--job_dir",
            str(trident_job_dir),
            "--gpu",
            str(args.gpu),
            "--patch_encoder",
            args.patch_encoder,
            "--mag",
            str(args.mag),
            "--patch_size",
            str(args.patch_size),
            "--segmenter",
            args.segmenter,
            "--seg_conf_thresh",
            str(args.seg_conf_thresh),
            "--overlap",
            str(args.overlap),
            "--batch_size",
            str(args.trident_batch_size),
        ]
        if args.reader_type:
            cmd += ["--reader_type", args.reader_type]
        if args.remove_holes:
            cmd.append("--remove_holes")
        if args.remove_artifacts:
            cmd.append("--remove_artifacts")
        if args.remove_penmarks:
            cmd.append("--remove_penmarks")
        _run_cmd(cmd)

    feat_h5 = _find_feature_h5(
        job_dir=trident_job_dir,
        mag=args.mag,
        patch_size=args.patch_size,
        overlap=args.overlap,
        patch_encoder=args.patch_encoder,
        slide_stem=slide_path.stem,
    )
    print("[feature_h5]", feat_h5)

    with open(args.stain_vocab, "r", encoding="utf-8") as f:
        stain_vocab = json.load(f)

    stain_name = normalize_stain_name(args.stain_name)
    if stain_name not in stain_vocab:
        raise KeyError(f"stain '{stain_name}' not found in stain vocab: {args.stain_vocab}")
    stain_id = int(stain_vocab[stain_name])

    reader = H5BagReader(cache_size=0)
    item = reader.read(str(feat_h5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args, stain_vocab)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.to(device).eval()

    batch = {
        "wsi": [
            {
                "stains": [
                    {
                        "name": stain_name,
                        "id": stain_id,
                        "features": item["features"].to(device),
                        "coords": item["coords"].to(device) if item["coords"] is not None else None,
                    }
                ]
            }
        ]
    }

    with torch.no_grad():
        out = model(batch)
        prob_l1 = torch.sigmoid(out["logits_L1"])[0].detach().cpu().tolist()
        prob_l2 = torch.sigmoid(out["logits_L2"])[0].detach().cpu().tolist()
        prob_l3 = torch.sigmoid(out["logits_L3"])[0].detach().cpu().tolist()

        aux = out.get("aux", [None])[0]
        patch_topk = {}
        stain_weights = {}
        if aux is not None:
            for sname, attn in aux.get("patch_attn", {}).items():
                if attn is None:
                    continue
                topv, topi = torch.topk(attn, k=min(args.topk, attn.shape[0]))
                patch_topk[sname] = {
                    "topk_index": topi.detach().cpu().tolist(),
                    "topk_weight": topv.detach().cpu().tolist(),
                }
            sf = aux.get("stain_fusion_attn")
            if sf is not None:
                names = aux.get("stain_names", [])
                vals = sf.detach().cpu().tolist()
                stain_weights = {names[j]: vals[j] for j in range(min(len(names), len(vals)))}

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{sample_id}.json"
    payload = {
        "sample_id": sample_id,
        "slide_path": str(slide_path),
        "feature_h5": str(feat_h5),
        "stain_name": stain_name,
        "prob_L1": prob_l1,
        "prob_L2": prob_l2,
        "prob_L3": prob_l3,
        "stain_patch_attention_topk": patch_topk,
        "stain_fusion_weights": stain_weights,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("[ok] output:", out_json)


if __name__ == "__main__":
    main()
