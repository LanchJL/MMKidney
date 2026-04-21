#!/usr/bin/env bash
set -euo pipefail

PATCH_FINAL="${1:-}"
WSI_DIR="${2:-}"
OUT_DIR="${3:-}"

if [[ -z "${PATCH_FINAL}" || -z "${WSI_DIR}" ]]; then
  echo "Usage: bash scripts/generate_l2_review_pack.sh <patch_final.parquet> <wsi_dir> [output_dir]"
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="$(dirname "${PATCH_FINAL}")/l2_review_pack"
fi

python3 CLUSTER/19_visualize_final_cluster_patches.py \
  --patch-final "${PATCH_FINAL}" \
  --wsi-dir "${WSI_DIR}" \
  --output-dir "${OUT_DIR}" \
  --cluster-col proto_cluster_L2 \
  --review-mode both \
  --n-per-cluster 25 \
  --n-cols 5 \
  --core-quantile 0.30 \
  --consistency-max-per-slide 0 \
  --max-per-slide 3 \
  --min-center-dist 512 \
  --drop-black-white \
  --patch-size 512

