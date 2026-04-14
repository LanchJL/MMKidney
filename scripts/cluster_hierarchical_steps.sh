#!/usr/bin/env bash
set -euo pipefail

# CLUSTER-like step-wise pipeline for HE hierarchical clustering
# Steps: fit -> assign -> l1 -> l2 -> final -> check

python3 -m src.datasets.cluster_hierarchical_steps "$@"
