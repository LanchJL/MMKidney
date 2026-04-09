import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import torch

from src.datasets.h5_reader import H5BagReader


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _apply_affine(coords: torch.Tensor, mat: Optional[List[List[float]]]) -> torch.Tensor:
    """coords [N,2], mat 2x3"""
    if mat is None:
        return coords
    m = torch.tensor(mat, dtype=torch.float32, device=coords.device)
    if m.shape != (2, 3):
        return coords
    ones = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=coords.device)
    xy1 = torch.cat([coords.float(), ones], dim=1)  # [N,3]
    out = (m @ xy1.T).T  # [N,2]
    return out


def _knn_match_features(
    he_coords: torch.Tensor,
    st_coords: torch.Tensor,
    st_feats: torch.Tensor,
    k: int,
    max_radius: float,
    chunk_size: int,
):
    """Return matched feature mean [N,D], valid_count [N,1]."""
    n, d = he_coords.shape[0], st_feats.shape[1]
    out_feat = torch.zeros((n, d), dtype=torch.float32)
    out_cnt = torch.zeros((n, 1), dtype=torch.float32)

    st_coords_f = st_coords.float()
    he_coords_f = he_coords.float()

    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        c = he_coords_f[s:e]  # [b,2]
        dist = torch.cdist(c, st_coords_f, p=2)  # [b,m]

        kk = min(k, dist.shape[1])
        d_k, i_k = torch.topk(dist, k=kk, dim=1, largest=False)

        # gather and mean neighbors
        neigh = st_feats[i_k]  # [b,kk,d]

        if max_radius > 0:
            valid = (d_k <= max_radius).float().unsqueeze(-1)  # [b,kk,1]
            wsum = valid.sum(dim=1).clamp(min=1.0)
            agg = (neigh * valid).sum(dim=1) / wsum
            cnt = valid.sum(dim=1)  # [b,1]
        else:
            agg = neigh.mean(dim=1)
            cnt = torch.full((e - s, 1), float(kk), dtype=torch.float32)

        out_feat[s:e] = agg
        out_cnt[s:e] = cnt

    return out_feat, out_cnt


def main():
    p = argparse.ArgumentParser("Build HE-anchored pseudo patch-level fusion features")
    p.add_argument("--manifest", default="data/processed/manifests/trimodal_manifest.jsonl")
    p.add_argument("--output-dir", default="data/processed/pseudo_patch_fusion")
    p.add_argument("--he-name", default="HE")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5, help="fused=(1-alpha)*HE + alpha*other")
    p.add_argument("--max-radius", type=float, default=-1.0, help="coord radius filter; <=0 means no filter")
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--max-he-patches", type=int, default=0, help="0 means keep all")
    p.add_argument("--affine-json", default="", help="optional per sample/stain affine mapping json")
    args = p.parse_args()

    rows = _read_jsonl(Path(args.manifest))
    out_dir = Path(args.output_dir)
    out_h5_dir = out_dir / "h5"
    out_h5_dir.mkdir(parents=True, exist_ok=True)

    affine_map = {}
    if args.affine_json:
        with Path(args.affine_json).open("r", encoding="utf-8") as f:
            affine_map = json.load(f)

    reader = H5BagReader(cache_size=32)

    out_manifest = []
    summary = []

    for idx, rec in enumerate(rows, 1):
        sid = rec.get("sample_id") or rec.get("id")
        h5s = rec.get("h5s", {}) or {}
        if args.he_name not in h5s:
            continue

        he_item = reader.read(h5s[args.he_name])
        he_feat = he_item["features"].float()
        he_coords = he_item["coords"]
        if he_coords is None:
            # if no coords, cannot cross-stain match -> keep HE
            he_coords = torch.arange(he_feat.shape[0], dtype=torch.float32).unsqueeze(1).repeat(1, 2)
        else:
            he_coords = he_coords.float()

        if args.max_he_patches > 0 and he_feat.shape[0] > args.max_he_patches:
            perm = torch.randperm(he_feat.shape[0])[: args.max_he_patches]
            he_feat = he_feat[perm]
            he_coords = he_coords[perm]

        other_feats = []
        other_cnts = []
        used_stains = []

        for stain, path in h5s.items():
            if stain == args.he_name:
                continue
            try:
                st_item = reader.read(path)
            except Exception:
                continue
            st_feat = st_item["features"].float()
            st_coords = st_item["coords"]
            if st_coords is None:
                continue
            st_coords = st_coords.float()

            mat = None
            if sid in affine_map and stain in affine_map[sid]:
                mat = affine_map[sid][stain]
            st_coords = _apply_affine(st_coords, mat)

            m_feat, m_cnt = _knn_match_features(
                he_coords=he_coords,
                st_coords=st_coords,
                st_feats=st_feat,
                k=args.k,
                max_radius=args.max_radius,
                chunk_size=args.chunk_size,
            )
            other_feats.append(m_feat)
            other_cnts.append((m_cnt > 0).float())
            used_stains.append(stain)

        if other_feats:
            stack_feat = torch.stack(other_feats, dim=0)  # [S,N,D]
            stack_cnt = torch.stack(other_cnts, dim=0)  # [S,N,1]
            denom = stack_cnt.sum(dim=0).clamp(min=1.0)
            agg_other = (stack_feat * stack_cnt).sum(dim=0) / denom
            fused = (1.0 - args.alpha) * he_feat + args.alpha * agg_other
            matched_ratio = float((denom > 0).float().mean().item())
        else:
            fused = he_feat
            matched_ratio = 0.0

        out_h5 = out_h5_dir / f"{sid}_fused.h5"
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("features", data=fused.cpu().numpy())
            f.create_dataset("coords", data=he_coords.cpu().numpy())
            f.create_dataset("features_he", data=he_feat.cpu().numpy())

        out_rec = {
            "sample_id": sid,
            "split": rec.get("split", ""),
            "h5": str(out_h5),
            "labels": rec.get("labels", {}),
            "source_h5s": h5s,
            "used_stains": used_stains,
            "matched_ratio": matched_ratio,
        }
        out_manifest.append(out_rec)
        summary.append(
            {
                "sample_id": sid,
                "split": rec.get("split", ""),
                "n_he_patches": int(he_feat.shape[0]),
                "n_used_stains": int(len(used_stains)),
                "used_stains": used_stains,
                "matched_ratio": matched_ratio,
            }
        )

        if idx % 20 == 0:
            print(f"[fusion] processed {idx}/{len(rows)}")

    manifest_path = out_dir / "fused_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in out_manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = out_dir / "fused_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples": len(out_manifest),
                "manifest": str(manifest_path),
                "k": args.k,
                "alpha": args.alpha,
                "max_radius": args.max_radius,
                "max_he_patches": args.max_he_patches,
                "details": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[done] fused samples={len(out_manifest)}")
    print(f"[saved] {manifest_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
