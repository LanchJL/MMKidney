#!/usr/bin/env bash
set -euo pipefail
python3 -m src.training.train_distill "$@"
