#!/usr/bin/env bash
set -euo pipefail
python3 -m src.training.infer_single_wsi_end2end "$@"
