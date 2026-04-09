#!/usr/bin/env bash
set -euo pipefail

# Hierarchical clustering (L1 -> L2 refinement) for HE or fused patch features
# Example (fused):
# bash scripts/cluster_hierarchical_patches.sh \
#   --manifest data/processed/pseudo_patch_fusion/fused_manifest.jsonl \
#   --output-dir data/processed/hier_cluster_fused \
#   --l1-method leiden --n-prototypes 2048

python3 -m src.datasets.cluster_hierarchical_patches "$@"
