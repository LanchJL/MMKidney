#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_pipeline.sh prepare
#   bash scripts/run_pipeline.sh stage_a
#   bash scripts/run_pipeline.sh stage_b
#   bash scripts/run_pipeline.sh stage_c
#   bash scripts/run_pipeline.sh infer
#   bash scripts/run_pipeline.sh all

STEP="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Paths (override by env vars if needed)
DATA_DIR="${DATA_DIR:-data}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
OUT_ROOT="${OUT_ROOT:-outputs}"
XLSX_NAME="${XLSX_NAME:-pathology_reports_v7.3_with_GT2__20260414  Censored.xlsx}"
MAIN_SHEET="${MAIN_SHEET:-移植肾汇总-筛选后}"
SLICE_NOTE_SHEET="${SLICE_NOTE_SHEET:-切片备注}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROCESSED_DIR}/manifests/train_manifest.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${PROCESSED_DIR}/manifests/val_manifest.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-${PROCESSED_DIR}/manifests/test_manifest.jsonl}"
TRIMODAL_MANIFEST="${TRIMODAL_MANIFEST:-${PROCESSED_DIR}/manifests/trimodal_manifest.jsonl}"

STAIN_VOCAB="${STAIN_VOCAB:-${PROCESSED_DIR}/stain_vocab.json}"
TAB_FEATS="${TAB_FEATS:-${PROCESSED_DIR}/tabular_features.csv}"
LAB_FEATS="${LAB_FEATS:-${PROCESSED_DIR}/lab_features.csv}"

WSI_OUT="${WSI_OUT:-${OUT_ROOT}/wsi}"
TEACHER_OUT="${TEACHER_OUT:-${OUT_ROOT}/teacher}"
DISTILL_OUT="${DISTILL_OUT:-${OUT_ROOT}/distill}"
EXPLAIN_OUT="${EXPLAIN_OUT:-${OUT_ROOT}/explanations}"

run_prepare() {
  echo "[run] prepare"
  "${PYTHON_BIN}" -m src.datasets.build_cohort \
    --data-dir "${DATA_DIR}" \
    --out-dir "${PROCESSED_DIR}" \
    --xlsx-name "${XLSX_NAME}" \
    --main-sheet "${MAIN_SHEET}" \
    --slice-note-sheet "${SLICE_NOTE_SHEET}"
  "${PYTHON_BIN}" -m src.datasets.build_tabular_features --cohort-tabular "${PROCESSED_DIR}/cohort_tabular.csv" --out-dir "${PROCESSED_DIR}"
  "${PYTHON_BIN}" -m src.datasets.build_lab_features --cohort-trimodal "${PROCESSED_DIR}/cohort_trimodal.csv" --lab-csv "${DATA_DIR}/Merged_Lab_Results_Final.csv" --out-dir "${PROCESSED_DIR}"
}

run_stage_a() {
  echo "[run] stage_a wsi student"
  "${PYTHON_BIN}" -m src.training.train_wsi \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --test-manifest "${TEST_MANIFEST}" \
    --stain-vocab "${STAIN_VOCAB}" \
    --out-dir "${WSI_OUT}"
}

run_stage_b() {
  echo "[run] stage_b teacher"
  "${PYTHON_BIN}" -m src.training.train_teacher \
    --trimodal-manifest "${TRIMODAL_MANIFEST}" \
    --stain-vocab "${STAIN_VOCAB}" \
    --tabular-features "${TAB_FEATS}" \
    --lab-features "${LAB_FEATS}" \
    --init-wsi-ckpt "${WSI_OUT}/best_wsi.pt" \
    --out-dir "${TEACHER_OUT}"
}

run_stage_c() {
  echo "[run] stage_c distill"
  "${PYTHON_BIN}" -m src.training.train_distill \
    --teacher-ckpt "${TEACHER_OUT}/best_teacher.pt" \
    --student-init-ckpt "${WSI_OUT}/best_wsi.pt" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --test-manifest "${TEST_MANIFEST}" \
    --trimodal-manifest "${TRIMODAL_MANIFEST}" \
    --stain-vocab "${STAIN_VOCAB}" \
    --tabular-features "${TAB_FEATS}" \
    --lab-features "${LAB_FEATS}" \
    --out-dir "${DISTILL_OUT}"
}

run_infer() {
  echo "[run] infer student"
  "${PYTHON_BIN}" -m src.training.infer \
    --model-type wsi \
    --manifest "${TEST_MANIFEST}" \
    --stain-vocab "${STAIN_VOCAB}" \
    --ckpt "${DISTILL_OUT}/best_distilled_student.pt" \
    --out-dir "${EXPLAIN_OUT}"
}

case "${STEP}" in
  prepare)
    run_prepare
    ;;
  stage_a)
    run_stage_a
    ;;
  stage_b)
    run_stage_b
    ;;
  stage_c)
    run_stage_c
    ;;
  infer)
    run_infer
    ;;
  all)
    run_prepare
    run_stage_a
    run_stage_b
    run_stage_c
    run_infer
    ;;
  *)
    echo "Unknown step: ${STEP}"
    echo "Usage: bash scripts/run_pipeline.sh [prepare|stage_a|stage_b|stage_c|infer|all]"
    exit 1
    ;;
esac

echo "[ok] done: ${STEP}"
