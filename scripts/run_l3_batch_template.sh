#!/usr/bin/env bash
set -euo pipefail

# End-to-end template:
# 1) save baseline
# 2) auto plan targets (or use provided --targets)
# 3) run batch local L3
# 4) run batch checks
# 5) generate manual review checklist

OUTPUT_DIR="prototype_pipeline_output"
TARGET_STAIN="HE"
N_PROTOTYPES=2048
TARGETS=""
MIN_PATCHES=100000
TOP_N=4
ONLY_L2="yes"
RESOLUTION=0.60
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
    --min-patches) MIN_PATCHES="${2:-}"; shift 2 ;;
    --top-n) TOP_N="${2:-}"; shift 2 ;;
    --only-l2) ONLY_L2="yes"; shift ;;
    --include-l1) ONLY_L2="no"; shift ;;
    --resolution) RESOLUTION="${2:-}"; shift 2 ;;
    --n-pcs) N_PCS="${2:-}"; shift 2 ;;
    --n-neighbors) N_NEIGHBORS="${2:-}"; shift 2 ;;
    --min-prototypes-to-refine) MIN_PROTOS="${2:-}"; shift 2 ;;
    --min-l3-prototypes) MIN_L3_PROTOS="${2:-}"; shift 2 ;;
    --random-state) RANDOM_STATE="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[template] Step 1/5: save baseline"
bash scripts/save_cluster_baseline.sh "$OUTPUT_DIR" "template_before_batch_${STAMP}"

if [[ -z "$TARGETS" ]]; then
  echo "[template] Step 2/5: auto plan targets"
  PLAN_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --target-stain "$TARGET_STAIN"
    --n-prototypes "$N_PROTOTYPES"
    --min-patches "$MIN_PATCHES"
    --top-n "$TOP_N"
    --out-json "${OUTPUT_DIR}/l3_target_plan_${STAMP}.json"
  )
  if [[ "$ONLY_L2" == "yes" ]]; then
    PLAN_ARGS+=(--only-l2)
  fi
  python3 CLUSTER/15_plan_l3_targets.py "${PLAN_ARGS[@]}"

  TARGETS="$(python3 - << PY
import json
p='${OUTPUT_DIR}/l3_target_plan_${STAMP}.json'
with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
print(d.get('targets_csv',''))
PY
)"
fi

if [[ -z "$TARGETS" ]]; then
  echo "[template] No targets found, exit."
  exit 0
fi

echo "[template] targets=$TARGETS"

echo "[template] Step 3/5: run batch L3"
bash scripts/batch_refine_single_cluster_l3.sh \
  --output-dir "$OUTPUT_DIR" \
  --target-stain "$TARGET_STAIN" \
  --n-prototypes "$N_PROTOTYPES" \
  --targets "$TARGETS" \
  --resolution "$RESOLUTION" \
  --n-pcs "$N_PCS" \
  --n-neighbors "$N_NEIGHBORS" \
  --min-prototypes-to-refine "$MIN_PROTOS" \
  --min-l3-prototypes "$MIN_L3_PROTOS" \
  --random-state "$RANDOM_STATE"

echo "[template] Step 4/5: check batch results"
bash scripts/check_batch_l3_results.sh --output-dir "$OUTPUT_DIR"

echo "[template] Step 5/5: build manual review checklist"
python3 CLUSTER/16_make_l3_review_pack.py --output-dir "$OUTPUT_DIR"

echo "[template] done"
echo "[template] Please open latest batch dir under: ${OUTPUT_DIR}/batch_l3_runs"
