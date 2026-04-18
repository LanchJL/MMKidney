import argparse
import os

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    if path.endswith('.csv'):
        return pd.read_csv(path)
    if path.endswith('.pkl'):
        return pd.read_pickle(path)
    raise ValueError(f'Unsupported format: {path}')


def save_table(df: pd.DataFrame, path_base: str) -> str:
    try:
        out = path_base if path_base.endswith('.parquet') else path_base + '.parquet'
        df.to_parquet(out, index=False)
        return out
    except Exception:
        out = path_base if path_base.endswith('.csv') else path_base + '.csv'
        df.to_csv(out, index=False)
        return out


def main():
    p = argparse.ArgumentParser('Apply suggested cluster merges to patch_final_clusters')
    p.add_argument('--patch-final', default='prototype_pipeline_output/patch_final_clusters_HE_2048.parquet')
    p.add_argument('--merge-csv', default='prototype_pipeline_output/cluster_merge_suggestions_HE_2048.csv')
    p.add_argument('--out-prefix', default='prototype_pipeline_output/patch_final_clusters_HE_2048_merged')
    p.add_argument('--overwrite-final', action='store_true', help='Overwrite final_cluster with merged result')
    args = p.parse_args()

    patch_df = load_table(args.patch_final).copy()
    map_df = pd.read_csv(args.merge_csv).copy()

    patch_df['final_cluster'] = patch_df['final_cluster'].astype(str)
    map_df['small_cluster'] = map_df['small_cluster'].astype(str)
    map_df['suggested_target'] = map_df['suggested_target'].astype(str)

    if 'enabled' in map_df.columns:
        map_df = map_df[map_df['enabled'].fillna(1).astype(int) == 1].copy()

    merge_map = dict(zip(map_df['small_cluster'], map_df['suggested_target']))

    patch_df['final_cluster_merged'] = patch_df['final_cluster'].map(lambda x: merge_map.get(str(x), str(x)))

    if args.overwrite_final:
        patch_df['final_cluster'] = patch_df['final_cluster_merged']

    out_patch = save_table(patch_df, args.out_prefix)

    size_df = (
        patch_df['final_cluster_merged']
        .value_counts()
        .rename_axis('final_cluster')
        .reset_index(name='n_patches')
        .sort_values('n_patches', ascending=False)
    )
    out_size = save_table(size_df, args.out_prefix + '_sizes')

    print('[apply] merged cluster map applied')
    print(f'[saved] {out_patch}')
    print(f'[saved] {out_size}')
    print(f'[stats] n_clusters_before={patch_df["final_cluster"].astype(str).nunique()} n_clusters_after={patch_df["final_cluster_merged"].astype(str).nunique()}')


if __name__ == '__main__':
    main()
