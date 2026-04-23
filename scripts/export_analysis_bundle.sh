#!/usr/bin/env bash
set -euo pipefail
python3 -m src.training.export_analysis_bundle "$@"
