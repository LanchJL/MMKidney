import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .cluster_hierarchical_patches import (
    _assign_prototypes,
    _cluster_level1,
    _fit_prototypes,
    _normalize_manifest_rows,
    _read_jsonl,
    _refine_next_level,
    _save_table_auto,
    _spatial_smooth_majority,
)


def _load_table_auto(base: Path) -> pd.DataFrame:
    cands = [base.with_suffix('.parquet'), base.with_suffix('.csv'), base.with_suffix('.pkl')]
    for p in cands:
        if p.exists():
            if p.suffix == '.parquet':
                return pd.read_parquet(p)
            if p.suffix == '.csv':
                return pd.read_csv(p)
            if p.suffix == '.pkl':
                return pd.read_pickle(p)
    raise FileNotFoundError(f'Cannot find table for base={base}, tried: {cands}')


def _load_or_read_manifest(manifest: str, stain_name: str) -> List[Dict]:
    rows = _read_jsonl(Path(manifest))
    rows = _normalize_manifest_rows(rows, stain_name=stain_name)
    return rows


def _step_fit(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_or_read_manifest(args.manifest, args.stain_name)

    centers = _fit_prototypes(
        manifest_rows=rows,
        n_prototypes=args.n_prototypes,
        feature_key=args.feature_key,
        max_patches_per_sample=args.max_patches_per_sample,
        prototype_patches_per_sample=args.prototype_patches_per_sample,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    np.save(str(out_dir / 'prototype_centers.npy'), centers)
    with (out_dir / 'step_fit_meta.json').open('w', encoding='utf-8') as f:
        json.dump(
            {
                'manifest': args.manifest,
                'stain_name': args.stain_name,
                'n_samples': len(rows),
                'n_prototypes': int(args.n_prototypes),
                'feature_key': args.feature_key,
                'centers': str(out_dir / 'prototype_centers.npy'),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print('[done] step fit')


def _step_assign(args):
    out_dir = Path(args.output_dir)
    rows = _load_or_read_manifest(args.manifest, args.stain_name)
    centers = np.load(str(out_dir / 'prototype_centers.npy')).astype(np.float32, copy=False)

    patch_df = _assign_prototypes(
        manifest_rows=rows,
        centers=centers,
        feature_key=args.feature_key,
        max_patches_per_sample=args.max_patches_per_sample,
    )
    p1 = _save_table_auto(patch_df, out_dir / 'patch_assignments')
    with (out_dir / 'step_assign_meta.json').open('w', encoding='utf-8') as f:
        json.dump(
            {
                'n_rows': int(len(patch_df)),
                'n_samples': int(patch_df['sample_id'].nunique()),
                'patch_assignments': p1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print('[done] step assign')


def _step_l1(args):
    out_dir = Path(args.output_dir)
    centers = np.load(str(out_dir / 'prototype_centers.npy')).astype(np.float32, copy=False)

    l1_labels = _cluster_level1(
        centers=centers,
        method=args.l1_method,
        n_l1_clusters=args.n_l1_clusters,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        resolution=args.l1_resolution,
        random_state=args.random_state,
    )
    proto_l1 = pd.DataFrame(
        {
            'proto_id': np.arange(centers.shape[0], dtype=np.int32),
            'proto_cluster_L1': l1_labels.astype(str),
        }
    )
    p = _save_table_auto(proto_l1, out_dir / 'prototype_clusters_L1')
    with (out_dir / 'step_l1_meta.json').open('w', encoding='utf-8') as f:
        json.dump({'n_l1_clusters': int(len(set(l1_labels.tolist()))), 'prototype_clusters_L1': p}, f, ensure_ascii=False, indent=2)
    print('[done] step l1')


def _step_l2(args):
    out_dir = Path(args.output_dir)
    centers = np.load(str(out_dir / 'prototype_centers.npy')).astype(np.float32, copy=False)
    patch_df = _load_table_auto(out_dir / 'patch_assignments')
    proto_l1 = _load_table_auto(out_dir / 'prototype_clusters_L1')

    l1_labels = proto_l1.sort_values('proto_id')['proto_cluster_L1'].astype(str).to_numpy()
    l2_labels, l2_logs = _refine_next_level(
        centers=centers,
        parent_labels=l1_labels,
        method=args.l1_method,
        level_name='L2',
        min_prototypes_to_refine=args.min_prototypes_to_refine,
        min_child_prototypes=args.min_l2_prototypes,
        min_child_slides=args.min_l2_slides,
        min_child_patch_frac=args.min_l2_patch_frac,
        min_silhouette=args.min_l2_silhouette,
        refine_resolution=args.refine_resolution,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
        stability_n_seeds=args.stability_n_seeds,
        min_stability_ari=args.min_stability_ari,
        patch_df=patch_df,
    )

    l3_labels = np.array(['NA'] * centers.shape[0], dtype=object)
    l3_logs = []
    if args.enable_l3:
        l3_labels, l3_logs = _refine_next_level(
            centers=centers,
            parent_labels=l2_labels.astype(str),
            method=args.l1_method,
            level_name='L3',
            min_prototypes_to_refine=args.min_l2_to_refine_l3,
            min_child_prototypes=args.min_l3_prototypes,
            min_child_slides=args.min_l3_slides,
            min_child_patch_frac=args.min_l3_patch_frac,
            min_silhouette=args.min_l3_silhouette,
            refine_resolution=args.l3_resolution,
            n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state,
            stability_n_seeds=args.stability_n_seeds,
            min_stability_ari=args.min_stability_ari,
            patch_df=patch_df,
        )

    proto_map = pd.DataFrame(
        {
            'proto_id': np.arange(centers.shape[0], dtype=np.int32),
            'proto_cluster_L1': l1_labels.astype(str),
            'proto_cluster_L2': l2_labels.astype(str),
            'proto_cluster_L3': l3_labels.astype(str),
        }
    )

    if args.enable_l3:
        proto_map['final_cluster'] = proto_map['proto_cluster_L3']
        m = proto_map['final_cluster'].isna() | (proto_map['final_cluster'] == 'NA')
        proto_map.loc[m, 'final_cluster'] = proto_map.loc[m, 'proto_cluster_L2']
        m2 = proto_map['final_cluster'].isna() | (proto_map['final_cluster'] == 'NA')
        proto_map.loc[m2, 'final_cluster'] = proto_map.loc[m2, 'proto_cluster_L1']
    else:
        proto_map['final_cluster'] = proto_map['proto_cluster_L2']
        m = proto_map['final_cluster'].isna() | (proto_map['final_cluster'] == 'NA')
        proto_map.loc[m, 'final_cluster'] = proto_map.loc[m, 'proto_cluster_L1']

    p = _save_table_auto(proto_map, out_dir / 'prototype_clusters_L1L2L3')
    with (out_dir / 'refine_decisions.json').open('w', encoding='utf-8') as f:
        json.dump({'L2': l2_logs, 'L3': l3_logs}, f, ensure_ascii=False, indent=2)
    with (out_dir / 'step_l2_meta.json').open('w', encoding='utf-8') as f:
        json.dump({'prototype_clusters_L1L2L3': p}, f, ensure_ascii=False, indent=2)
    print('[done] step l2/l3')


def _step_final(args):
    out_dir = Path(args.output_dir)
    patch_df = _load_table_auto(out_dir / 'patch_assignments')
    proto_map = _load_table_auto(out_dir / 'prototype_clusters_L1L2L3')

    patch_final = patch_df.merge(proto_map, on='proto_id', how='left', validate='many_to_one')
    if args.spatial_smooth_k > 1:
        patch_final = _spatial_smooth_majority(
            patch_final=patch_final,
            label_col='final_cluster',
            n_neighbors=args.spatial_smooth_k,
            n_iter=max(1, int(args.spatial_smooth_iter)),
        )

    slide_hist = patch_final.groupby(['sample_id', 'final_cluster']).size().reset_index(name='count')
    slide_total = patch_final.groupby('sample_id').size().reset_index(name='sample_total')
    slide_hist = slide_hist.merge(slide_total, on='sample_id', how='left')
    slide_hist['fraction'] = slide_hist['count'] / slide_hist['sample_total']

    p1 = _save_table_auto(patch_final, out_dir / 'patch_final_clusters')
    p2 = _save_table_auto(slide_hist, out_dir / 'slide_final_cluster_hist')

    meta = {
        'patch_final_clusters': p1,
        'slide_final_cluster_hist': p2,
        'n_patches': int(len(patch_final)),
        'n_samples': int(patch_final['sample_id'].nunique()),
        'n_final_clusters': int(patch_final['final_cluster'].astype(str).nunique()),
    }
    with (out_dir / 'step_final_meta.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('[done] step final')


def _step_check(args):
    out_dir = Path(args.output_dir)
    reports = {}
    try:
        centers = np.load(str(out_dir / 'prototype_centers.npy'))
        reports['centers_shape'] = list(centers.shape)
    except Exception as e:
        reports['centers_shape'] = f'missing: {e}'

    for name in ['patch_assignments', 'prototype_clusters_L1', 'prototype_clusters_L1L2L3', 'patch_final_clusters', 'slide_final_cluster_hist']:
        try:
            df = _load_table_auto(out_dir / name)
            reports[name] = {'rows': int(len(df)), 'cols': list(df.columns)}
        except Exception as e:
            reports[name] = f'missing: {e}'

    if isinstance(reports.get('patch_final_clusters'), dict):
        df = _load_table_auto(out_dir / 'patch_final_clusters')
        reports['patch_final_missing_final_cluster'] = int(df['final_cluster'].isna().sum()) if 'final_cluster' in df.columns else 'no_final_cluster_col'

    print(json.dumps(reports, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser('Step-wise HE hierarchical clustering (CLUSTER-like)')
    p.add_argument('--step', required=True, choices=['fit', 'assign', 'l1', 'l2', 'final', 'check'])
    p.add_argument('--manifest', default='data/processed/manifests/train_manifest.jsonl')
    p.add_argument('--output-dir', default='data/processed/hier_cluster_he_steps')
    p.add_argument('--stain-name', default='HE')

    p.add_argument('--feature-key', default='features')
    p.add_argument('--n-prototypes', type=int, default=2048)
    p.add_argument('--max-patches-per-sample', type=int, default=0)
    p.add_argument('--prototype-patches-per-sample', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=10000)
    p.add_argument('--random-state', type=int, default=0)

    p.add_argument('--l1-method', choices=['leiden', 'kmeans'], default='leiden')
    p.add_argument('--n-l1-clusters', type=int, default=12)
    p.add_argument('--n-pcs', type=int, default=50)
    p.add_argument('--n-neighbors', type=int, default=15)
    p.add_argument('--l1-resolution', type=float, default=0.45)

    p.add_argument('--refine-resolution', type=float, default=0.60)
    p.add_argument('--min-prototypes-to-refine', type=int, default=30)
    p.add_argument('--min-l2-prototypes', type=int, default=15)
    p.add_argument('--min-l2-slides', type=int, default=3)
    p.add_argument('--min-l2-patch-frac', type=float, default=0.005)
    p.add_argument('--min-l2-silhouette', type=float, default=0.05)
    p.add_argument('--stability-n-seeds', type=int, default=1)
    p.add_argument('--min-stability-ari', type=float, default=0.0)

    p.add_argument('--enable-l3', action='store_true')
    p.add_argument('--l3-resolution', type=float, default=0.75)
    p.add_argument('--min-l2-to-refine-l3', type=int, default=35)
    p.add_argument('--min-l3-prototypes', type=int, default=10)
    p.add_argument('--min-l3-slides', type=int, default=3)
    p.add_argument('--min-l3-patch-frac', type=float, default=0.003)
    p.add_argument('--min-l3-silhouette', type=float, default=0.06)

    p.add_argument('--spatial-smooth-k', type=int, default=0)
    p.add_argument('--spatial-smooth-iter', type=int, default=1)

    args = p.parse_args()

    if args.step == 'fit':
        _step_fit(args)
    elif args.step == 'assign':
        _step_assign(args)
    elif args.step == 'l1':
        _step_l1(args)
    elif args.step == 'l2':
        _step_l2(args)
    elif args.step == 'final':
        _step_final(args)
    elif args.step == 'check':
        _step_check(args)


if __name__ == '__main__':
    main()
