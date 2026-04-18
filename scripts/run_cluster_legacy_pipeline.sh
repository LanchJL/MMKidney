#!/usr/bin/env bash
set -euo pipefail

# Strictly reproduce CLUSTER legacy flow:
# 02 -> 03 -> 04 -> 05 -> 05b -> 06 -> 07
# with explicit manual checkpoint before 05b REFINE_CONFIG.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-until_manual}"

run_step() {
  local step_py="$1"
  echo ""
  echo "[RUN] ${step_py}"
  "$PYTHON_BIN" "$step_py"
  echo "[OK ] ${step_py}"
}

ensure_step01_artifact() {
  local need_file="prototype_pipeline_output/patch_index_raw.parquet"
  if [[ -f "$need_file" ]]; then
    return 0
  fi
  echo "[INFO] Missing prerequisite: $need_file"
  echo "[INFO] Auto-running CLUSTER/01_build_patch_index.py first."
  echo "[NOTE] If it fails, check H5_DIR in CLUSTER/01_build_patch_index.py."
  run_step "CLUSTER/01_build_patch_index.py"
  if [[ ! -f "$need_file" ]]; then
    echo "[ERROR] Still missing: $need_file"
    echo "[ERROR] Please verify H5_DIR and output path in CLUSTER/01_build_patch_index.py."
    exit 1
  fi
}

manual_notice() {
  cat << 'TXT'

================ MANUAL CHECKPOINT (REQUIRED) ================
Please edit REFINE_CONFIG in:
  CLUSTER/05b_refine_prototype_clusters.py

Recommended manual actions before continuing:
1) Inspect L1 outputs from step 05:
   - prototype_clusters_HE_2048.csv
   - prototype_cluster_sizes_HE_2048.csv
   - prototype_umap_HE_2048.png
2) Decide which L1 clusters should be refined (enabled=true/false).
3) Tune per-cluster params in REFINE_CONFIG:
   - resolution
   - min_prototypes_to_refine
   - min_l2_prototypes

After editing, continue with:
  bash scripts/run_cluster_legacy_pipeline.sh resume
==============================================================
TXT
}

case "$MODE" in
  until_manual)
    ensure_step01_artifact
    run_step "CLUSTER/02_compute_tissue_qc.py"
    run_step "CLUSTER/03_fit_prototypes.py"
    run_step "CLUSTER/04_assign_prototypes.py"
    run_step "CLUSTER/05_cluster_prototypes.py"
    manual_notice
    ;;
  resume)
    run_step "CLUSTER/05b_refine_prototype_clusters.py"
    run_step "CLUSTER/06_assign_final_clusters.py"
    run_step "CLUSTER/07_overlay_final_clusters.py"
    echo ""
    echo "[DONE] Legacy CLUSTER pipeline finished (through step 07)."
    ;;
  all)
    ensure_step01_artifact
    run_step "CLUSTER/02_compute_tissue_qc.py"
    run_step "CLUSTER/03_fit_prototypes.py"
    run_step "CLUSTER/04_assign_prototypes.py"
    run_step "CLUSTER/05_cluster_prototypes.py"
    echo "[WARN] Running 05b without manual REFINE_CONFIG review."
    run_step "CLUSTER/05b_refine_prototype_clusters.py"
    run_step "CLUSTER/06_assign_final_clusters.py"
    run_step "CLUSTER/07_overlay_final_clusters.py"
    echo ""
    echo "[DONE] Legacy CLUSTER pipeline finished (through step 07)."
    ;;
  02|03|04|05|05b|06|07)
    case "$MODE" in
      02) ensure_step01_artifact; run_step "CLUSTER/02_compute_tissue_qc.py" ;;
      03) ensure_step01_artifact; run_step "CLUSTER/03_fit_prototypes.py" ;;
      04) ensure_step01_artifact; run_step "CLUSTER/04_assign_prototypes.py" ;;
      05) ensure_step01_artifact; run_step "CLUSTER/05_cluster_prototypes.py" ;;
      05b) run_step "CLUSTER/05b_refine_prototype_clusters.py" ;;
      06) run_step "CLUSTER/06_assign_final_clusters.py" ;;
      07) run_step "CLUSTER/07_overlay_final_clusters.py" ;;
    esac
    ;;
  *)
    cat << 'USAGE'
Usage:
  bash scripts/run_cluster_legacy_pipeline.sh until_manual   # default, run 02-05 then stop for REFINE_CONFIG
  bash scripts/run_cluster_legacy_pipeline.sh resume         # run 05b-07 after manual edit
  bash scripts/run_cluster_legacy_pipeline.sh all            # run 02-07 without pause (not recommended)
  bash scripts/run_cluster_legacy_pipeline.sh 02|03|04|05|05b|06|07

Env:
  PYTHON_BIN=python3  (default)
USAGE
    exit 1
    ;;
esac
