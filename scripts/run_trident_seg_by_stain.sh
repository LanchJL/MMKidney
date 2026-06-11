#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/a6000/3E5C99E35C9995ED/jcy/MMKidney"
TRIDENT_ROOT="${ROOT}/external/TRIDENT"
WSI_DIR="/media/a6000/3E5C99E35C9995ED/jcy/dataset/medical/WSIs_merged"

STAIN="${1:-HE}"
ROI_MODE="${2:-roi}"
SEGMENTER="${3:-hest}"
GPU="${GPU:-0}"
MAX_WORKERS="${MAX_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-}"
SEG_CONF_THRESH="${SEG_CONF_THRESH:-0.5}"
LIST_CSV_OVERRIDE="${LIST_CSV:-}"
OUT_DIR_OVERRIDE="${OUT_DIR:-}"

if [[ "${ROI_MODE}" != "roi" && "${ROI_MODE}" != "all" ]]; then
  echo "Usage: $0 [STAIN] [roi|all] [hest|grandqc|otsu]" >&2
  exit 2
fi

ROI_FLAG=()
ROI_SUFFIX="all"
if [[ "${ROI_MODE}" == "roi" ]]; then
  ROI_FLAG=(--roi-only)
  ROI_SUFFIX="roi"
fi

LIST_DIR="${ROOT}/data/trident_lists"
THRESH_SUFFIX=""
if [[ "${SEG_CONF_THRESH}" != "0.5" ]]; then
  THRESH_SUFFIX="_thr${SEG_CONF_THRESH//./p}"
fi
OUT_DIR_DEFAULT="${ROOT}/outputs/trident_seg_${STAIN,,}_${ROI_SUFFIX}_${SEGMENTER}${THRESH_SUFFIX}"
LIST_CSV_DEFAULT="${LIST_DIR}/wsi_merged_${STAIN,,}_${ROI_SUFFIX}.csv"
OUT_DIR="${OUT_DIR_OVERRIDE:-${OUT_DIR_DEFAULT}}"
LIST_CSV="${LIST_CSV_OVERRIDE:-${LIST_CSV_DEFAULT}}"

mkdir -p "$(dirname "${LIST_CSV}")" "$(dirname "${OUT_DIR}")"
if [[ -n "${LIST_CSV_OVERRIDE}" ]]; then
  LIST_CSV="$(realpath "${LIST_CSV}")"
fi
OUT_DIR="$(realpath -m "${OUT_DIR}")"

BUILD_ARGS=(
  python "${ROOT}/scripts/build_trident_stain_list.py"
  --wsi-dir "${WSI_DIR}"
  --stain "${STAIN}"
  --out "${LIST_CSV}"
  "${ROI_FLAG[@]}"
)

if [[ -n "${LIMIT}" ]]; then
  BUILD_ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${LIST_CSV_OVERRIDE}" ]]; then
  echo "Using existing WSI list: ${LIST_CSV}"
else
  "${BUILD_ARGS[@]}"
fi

echo "Output directory: ${OUT_DIR}"
echo "WSI list: ${LIST_CSV}"
echo "Starting TRIDENT segmentation..."

cd "${TRIDENT_ROOT}"
conda run -n trident python run_batch_of_slides.py \
  --task seg \
  --wsi_dir "${WSI_DIR}" \
  --wsi_ext .mrxs \
  --custom_list_of_wsis "${LIST_CSV}" \
  --job_dir "${OUT_DIR}" \
  --gpus "${GPU}" \
  --segmenter "${SEGMENTER}" \
  --seg_conf_thresh "${SEG_CONF_THRESH}" \
  --skip_errors \
  --clear_dead_locks \
  --max_workers "${MAX_WORKERS}" \
  --batch_size "${BATCH_SIZE}"
