#!/usr/bin/env bash
set -euo pipefail

python -m src.datasets.build_banff_first \
  --excel "data/pathology_reports_v7.3_with_GT2__20260414  Censored.xlsx" \
  --sheet "移植肾汇总-筛选后" \
  --wsi-manifest "data/processed/manifests/all_manifest.jsonl" \
  --out-dir "data/processed_banff/manifests"
