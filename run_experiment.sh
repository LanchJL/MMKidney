#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
RUNS_DIR="${ROOT_DIR}/runs"

MODE="mil_ms"           # vector | mil | mil_ms
PRESET="quick"          # quick | full
EXTRA_DIM="0"
LAMBDA_DIV="0.0"
NO_TEST="0"
OUT_DIR=""

EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage:
  bash run_experiment.sh [options] [-- extra_train_args]

Options:
  --mode <vector|mil|mil_ms>   Training mode (default: mil_ms)
  --preset <quick|full>        Quick smoke run or full run (default: quick)
  --extra-dim <int>            Extra feature dimension (default: 0)
  --lambda-div <float>         Attention diversity weight (default: 0.0)
  --out-dir <path>             Output directory (default: auto under ./runs)
  --no-test                    Do not run test split evaluation
  -h, --help                   Show this message

Examples:
  bash run_experiment.sh
  bash run_experiment.sh --mode mil --preset quick
  bash run_experiment.sh --mode mil_ms --preset full -- --lr 1e-4 --batch-size 4
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --extra-dim)
      EXTRA_DIM="$2"
      shift 2
      ;;
    --lambda-div)
      LAMBDA_DIV="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --no-test)
      NO_TEST="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "[ERR] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "${MODE}" != "vector" && "${MODE}" != "mil" && "${MODE}" != "mil_ms" ]]; then
  echo "[ERR] --mode must be one of: vector | mil | mil_ms"
  exit 1
fi

if [[ "${PRESET}" != "quick" && "${PRESET}" != "full" ]]; then
  echo "[ERR] --preset must be one of: quick | full"
  exit 1
fi

if [[ "${MODE}" == "mil_ms" ]]; then
  TRAIN_JSON="${DATA_DIR}/train_ms.json"
  VAL_JSON="${DATA_DIR}/val_ms.json"
  TEST_JSON="${DATA_DIR}/test_ms.json"
else
  TRAIN_JSON="${DATA_DIR}/train.json"
  VAL_JSON="${DATA_DIR}/val.json"
  TEST_JSON="${DATA_DIR}/test.json"
fi

for f in "${TRAIN_JSON}" "${VAL_JSON}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERR] Required file not found: ${f}"
    exit 1
  fi
done

# Fast preflight: check whether at least one referenced H5 file exists.
FIRST_H5="$(rg -m1 '\"h5\"\\s*:' "${TRAIN_JSON}" | sed -E 's/.*\"h5\"\\s*:\\s*\"([^\"]+)\".*/\1/' || true)"
if [[ -z "${FIRST_H5}" ]]; then
  FIRST_H5="$(rg -m1 '\\.h5\"' "${TRAIN_JSON}" | sed -E 's/.*\"([^\"]+\\.h5)\".*/\1/' || true)"
fi
if [[ -n "${FIRST_H5}" && ! -f "${FIRST_H5}" ]]; then
  echo "[WARN] Referenced H5 not found: ${FIRST_H5}"
  echo "[WARN] JSON exists under ./data, but embedded H5 paths may still point to old locations."
fi

if [[ "${PRESET}" == "quick" ]]; then
  EPOCHS=3
  BATCH_SIZE=4
  NUM_WORKERS=0
  MAX_PATCHES=128
else
  EPOCHS=50
  BATCH_SIZE=8
  NUM_WORKERS=4
  MAX_PATCHES=""
fi

if [[ -z "${OUT_DIR}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="${RUNS_DIR}/${MODE}_${PRESET}_${TS}"
fi

mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[ERR] python/python3 not found in PATH"
    exit 1
  fi
fi

CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/train.py"
  --mode "${MODE}"
  --train-json "${TRAIN_JSON}"
  --val-json "${VAL_JSON}"
  --out-dir "${OUT_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --extra-dim "${EXTRA_DIM}"
  --lambda-div "${LAMBDA_DIV}"
)

if [[ -n "${MAX_PATCHES}" ]]; then
  CMD+=(--max-patches "${MAX_PATCHES}")
fi

if [[ "${NO_TEST}" == "0" && -f "${TEST_JSON}" ]]; then
  CMD+=(--test-json "${TEST_JSON}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Mode      : ${MODE}"
echo "[INFO] Preset    : ${PRESET}"
echo "[INFO] Train JSON: ${TRAIN_JSON}"
echo "[INFO] Val JSON  : ${VAL_JSON}"
if [[ "${NO_TEST}" == "0" && -f "${TEST_JSON}" ]]; then
  echo "[INFO] Test JSON : ${TEST_JSON}"
else
  echo "[INFO] Test JSON : (skipped)"
fi
echo "[INFO] Output Dir: ${OUT_DIR}"
echo "[INFO] Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[DONE] Finished. Outputs are in: ${OUT_DIR}"
