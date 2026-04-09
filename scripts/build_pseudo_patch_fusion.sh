#!/usr/bin/env bash
set -euo pipefail

# HE-anchored pseudo patch-level fusion features for clustering comparison
# Example:
# bash scripts/build_pseudo_patch_fusion.sh \
#   --manifest data/processed/manifests/trimodal_manifest.jsonl \
#   --output-dir data/processed/pseudo_patch_fusion \
#   --k 3 --alpha 0.5 --max-radius -1

python3 -m src.datasets.build_pseudo_patch_fusion "$@"
