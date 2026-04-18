#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-prototype_pipeline_output}"
TAG="${2:-$(date +%Y%m%d_%H%M%S)}"
DST="${OUT_DIR}/checkpoints/${TAG}"

mkdir -p "$DST"

echo "[save] baseline checkpoint -> $DST"

copy_if_exists() {
  local p="$1"
  if [[ -f "$p" ]]; then
    cp -f "$p" "$DST/"
    echo "  + $(basename "$p")"
  else
    echo "  - missing: $p"
  fi
}

copy_if_exists "${OUT_DIR}/final_cluster_sizes_HE_2048.csv"
copy_if_exists "${OUT_DIR}/overlay_final_clusters_HE_2048/final_cluster_colors.csv"
copy_if_exists "${OUT_DIR}/patch_final_clusters_HE_2048.parquet"
copy_if_exists "${OUT_DIR}/patch_final_clusters_HE_2048.pkl"
copy_if_exists "${OUT_DIR}/patch_final_clusters_HE_2048.csv"
copy_if_exists "${OUT_DIR}/prototype_clusters_L1L2_HE_2048.csv"
copy_if_exists "${OUT_DIR}/prototype_centers_HE_2048.npy"
copy_if_exists "${OUT_DIR}/prototype_usage_HE_2048.csv"
copy_if_exists "${OUT_DIR}/l1_cluster_check_report.json"
copy_if_exists "${OUT_DIR}/final_cluster_check_report.json"
copy_if_exists "CLUSTER/05b_refine_prototype_clusters.py"

echo "[done] baseline saved"
