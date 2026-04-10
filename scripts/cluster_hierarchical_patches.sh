#!/usr/bin/env bash
set -euo pipefail

# Hierarchical clustering (L1 -> L2, optional L3 refinement) for HE or fused patch features
# Example (fused):
# bash scripts/cluster_hierarchical_patches.sh \
#   --manifest data/processed/pseudo_patch_fusion/fused_manifest.jsonl \
#   --output-dir data/processed/hier_cluster_fused \
#   --l1-method leiden --n-prototypes 2048
#
# Example (HE-only + enable L3):
# bash scripts/cluster_hierarchical_patches.sh \
#   --manifest data/processed/he_only_manifest.jsonl \
#   --output-dir data/processed/hier_cluster_he_l3 \
#   --l1-method leiden \
#   --n-prototypes 2048 \
#   --enable-l3

python3 -m src.datasets.cluster_hierarchical_patches "$@"
