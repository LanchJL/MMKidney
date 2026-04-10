#!/usr/bin/env bash
set -euo pipefail

# Overlay hierarchical final cluster labels back onto WSI thumbnails.
# Example:
# bash scripts/overlay_hier_clusters_wsi.sh \
#   --patch-final data/processed/hier_cluster_fused/patch_final_clusters.parquet \
#   --wsi-dir /path/to/wsi \
#   --output-dir data/processed/hier_cluster_fused_overlay \
#   --legend

python3 -m src.datasets.overlay_hier_clusters_wsi "$@"
