#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install torch h5py tqdm scikit-learn

# Optional parquet support
if [[ "${INSTALL_PARQUET:-1}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install pandas pyarrow
fi

echo "[ok] dependencies installed"
