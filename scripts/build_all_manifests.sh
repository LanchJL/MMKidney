#!/usr/bin/env bash
set -euo pipefail
python3 -m src.datasets.build_cohort \
  --data-dir data \
  --out-dir data/processed \
  --xlsx-name "pathology_reports_v7.3_with_GT2__20260414  Censored.xlsx" \
  --main-sheet "移植肾汇总-筛选后" \
  --slice-note-sheet "切片备注"
python3 -m src.datasets.build_tabular_features --cohort-tabular data/processed/cohort_tabular.csv --out-dir data/processed
python3 -m src.datasets.build_lab_features --cohort-trimodal data/processed/cohort_trimodal.csv --lab-csv data/Merged_Lab_Results_Final.csv --out-dir data/processed
