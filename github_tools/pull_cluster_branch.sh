#!/bin/bash
set -euo pipefail

source ./github_tools/config.sh

TARGET="${CLUSTER_BRANCH:-cluster}"

echo "拉取 cluster 分支: $TARGET"
bash ./github_tools/pull_latest_changes.sh --branch "$TARGET"
