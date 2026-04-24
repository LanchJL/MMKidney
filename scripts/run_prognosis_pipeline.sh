#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline_v1}"

COMMON_ARGS=(
  --prognosis-cohort data/processed/prognosis_cohort.csv
  --feature-manifest data/processed/prognosis_feature_manifest.json
  --tabular-features data/processed/tabular_features.csv
  --lab-features data/processed/lab_features.csv
  --stain-vocab data/processed/stain_vocab.json
  --train-manifest data/processed/manifests/train_manifest.jsonl
  --val-manifest data/processed/manifests/val_manifest.jsonl
  --test-manifest data/processed/manifests/test_manifest.jsonl
  --survival-head discrete
  --time-bins-days 365,1095,1825
  --horizons-days 365,1095,1825
)

if [[ "${MODE}" == "baseline_v1" ]]; then
  python3 -m src.training.train_prognosis \
    "${COMMON_ARGS[@]}" \
    --profile baseline_v1 \
    --out-dir outputs/prognosis_baseline_v1
elif [[ "${MODE}" == "full_v1" ]]; then
  python3 -m src.training.train_prognosis \
    "${COMMON_ARGS[@]}" \
    --profile full_v1 \
    --save-plots \
    --out-dir outputs/prognosis_full_v1
elif [[ "${MODE}" == "full_v2" ]]; then
  python3 -m src.training.train_prognosis \
    "${COMMON_ARGS[@]}" \
    --profile full_v2 \
    --strict-treatment-history \
    --save-plots \
    --out-dir outputs/prognosis_full_v2
elif [[ "${MODE}" == "distill_v1" ]]; then
  python3 -m src.training.train_prognosis_distill \
    "${COMMON_ARGS[@]}" \
    --teacher-ckpt outputs/prognosis_full_v2/best_prognosis.pt \
    --teacher-profile full_v2 \
    --student-profile baseline_v1 \
    --strict-treatment-history \
    --out-dir outputs/prognosis_distill_v1
else
  echo "Unknown mode: ${MODE}"
  echo "Usage: bash scripts/run_prognosis_pipeline.sh [baseline_v1|full_v1|full_v2|distill_v1]"
  exit 1
fi
