#!/usr/bin/env bash
set -euo pipefail

# Batch run local L3 refinement for multiple L2 targets.
# Example:
# bash scripts/batch_refine_single_cluster_l3.sh \
#   --targets 2_1,2_2,2_3 \
#   --output-dir prototype_pipeline_output \
#   --resolution 0.60 \
#   --min-l3-prototypes 12

OUTPUT_DIR="prototype_pipeline_output"
TARGET_STAIN="HE"
N_PROTOTYPES=2048
TARGETS="2_1,2_2,2_3"
METHOD="leiden"
RESOLUTION="0.60"
N_PCS=30
N_NEIGHBORS=12
MIN_PROTOS=30
MIN_L3_PROTOS=12
RANDOM_STATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --target-stain) TARGET_STAIN="${2:-}"; shift 2 ;;
    --n-prototypes) N_PROTOTYPES="${2:-}"; shift 2 ;;
    --targets) TARGETS="${2:-}"; shift 2 ;;
    --method) METHOD="${2:-}"; shift 2 ;;
    --resolution) RESOLUTION="${2:-}"; shift 2 ;;
    --n-pcs) N_PCS="${2:-}"; shift 2 ;;
    --n-neighbors) N_NEIGHBORS="${2:-}"; shift 2 ;;
    --min-prototypes-to-refine) MIN_PROTOS="${2:-}"; shift 2 ;;
    --min-l3-prototypes) MIN_L3_PROTOS="${2:-}"; shift 2 ;;
    --random-state) RANDOM_STATE="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/batch_l3_runs/${STAMP}"
mkdir -p "$RUN_DIR"

SUMMARY_CSV="${RUN_DIR}/batch_l3_summary.csv"
echo "target_l2,status,n_l3_kept,top1_cluster,top1_ratio,n_clusters" > "$SUMMARY_CSV"

IFS=',' read -r -a TARR <<< "$TARGETS"

echo "[batch] targets=${TARGETS}"
echo "[batch] run_dir=${RUN_DIR}"

for TARGET in "${TARR[@]}"; do
  TARGET="$(echo "$TARGET" | xargs)"
  [[ -z "$TARGET" ]] && continue
  echo ""
  echo "[batch] running target=${TARGET}"

  bash scripts/refine_single_cluster_l3.sh \
    --output-dir "$OUTPUT_DIR" \
    --target-stain "$TARGET_STAIN" \
    --n-prototypes "$N_PROTOTYPES" \
    --target-l2 "$TARGET" \
    --method "$METHOD" \
    --resolution "$RESOLUTION" \
    --n-pcs "$N_PCS" \
    --n-neighbors "$N_NEIGHBORS" \
    --min-prototypes-to-refine "$MIN_PROTOS" \
    --min-l3-prototypes "$MIN_L3_PROTOS" \
    --random-state "$RANDOM_STATE"

  META_JSON="${OUTPUT_DIR}/l3_refine_meta_target_${TARGET}_${TARGET_STAIN}_${N_PROTOTYPES}.json"
  SIZE_CSV="${OUTPUT_DIR}/final_cluster_sizes_${TARGET_STAIN}_${N_PROTOTYPES}_l3_target_${TARGET}.csv"

  # copy artifacts for this target
  TDIR="${RUN_DIR}/${TARGET}"
  mkdir -p "$TDIR"
  cp -f "$META_JSON" "$TDIR/" 2>/dev/null || true
  cp -f "$SIZE_CSV" "$TDIR/" 2>/dev/null || true
  cp -f "${OUTPUT_DIR}/patch_final_clusters_${TARGET_STAIN}_${N_PROTOTYPES}_l3_target_${TARGET}.parquet" "$TDIR/" 2>/dev/null || true
  cp -f "${OUTPUT_DIR}/slide_final_cluster_hist_${TARGET_STAIN}_${N_PROTOTYPES}_l3_target_${TARGET}.csv" "$TDIR/" 2>/dev/null || true

  python3 - << PY
import json
import pandas as pd
from pathlib import Path

meta_path = Path("$META_JSON")
size_path = Path("$SIZE_CSV")
target = "$TARGET"
out_csv = Path("$SUMMARY_CSV")

status = "missing_meta"
n_l3 = 0
if meta_path.exists():
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    status = str(m.get("status", "unknown"))
    n_l3 = int(m.get("n_l3_kept", 0))

top1 = ""
top1_ratio = 0.0
n_clusters = 0
if size_path.exists():
    df = pd.read_csv(size_path)
    n_clusters = int(df["final_cluster"].nunique()) if len(df) else 0
    total = float(df["n_patches"].sum()) if len(df) else 0.0
    if len(df):
        i = df["n_patches"].idxmax()
        top1 = str(df.loc[i, "final_cluster"])
        top1_ratio = float(df.loc[i, "n_patches"] / max(1.0, total))

with out_csv.open("a", encoding="utf-8") as f:
    f.write(f"{target},{status},{n_l3},{top1},{top1_ratio:.6f},{n_clusters}\\n")
PY
done

echo ""
echo "[batch] done"
echo "[batch] summary: ${SUMMARY_CSV}"
cat "$SUMMARY_CSV"

