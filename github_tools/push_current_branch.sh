#!/bin/bash
set -euo pipefail

# 引入配置文件
source ./github_tools/config.sh

usage() {
  cat <<'USAGE'
用法:
  bash github_tools/push_current_branch.sh [--current | --branch <name> | --all] [--no-commit]

说明:
  --current         推送当前分支
  --branch <name>   推送指定本地分支
  --all             推送所有本地分支
  --no-commit       跳过提交步骤（默认会询问是否提交当前分支改动）

不带参数时会进入交互选择。
USAGE
}

MODE=""
TARGET_BRANCH=""
DO_COMMIT="ask"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --current)
      MODE="current"
      shift
      ;;
    --branch)
      MODE="branch"
      TARGET_BRANCH="${2:-}"
      shift 2
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --no-commit)
      DO_COMMIT="no"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

CURRENT_BRANCH="$(git symbolic-ref --short HEAD)"

if [[ -z "$MODE" ]]; then
  echo "请选择推送模式:"
  echo "1) 当前分支 ($CURRENT_BRANCH)"
  echo "2) 指定分支"
  echo "3) 所有本地分支"
  read -r -p "输入 1/2/3: " CHOICE
  case "$CHOICE" in
    1) MODE="current" ;;
    2)
      MODE="branch"
      read -r -p "请输入要推送的本地分支名: " TARGET_BRANCH
      ;;
    3) MODE="all" ;;
    *)
      echo "无效选择"
      exit 1
      ;;
  esac
fi

if [[ "$DO_COMMIT" == "ask" ]]; then
  read -r -p "是否先提交当前分支改动后再推送? [y/N]: " ANS
  case "${ANS:-N}" in
    y|Y) DO_COMMIT="yes" ;;
    *) DO_COMMIT="no" ;;
  esac
fi

if [[ "$DO_COMMIT" == "yes" ]]; then
  git add -A
  if git diff --cached --quiet; then
    echo "暂存区无变更，跳过 commit。"
  else
    read -r -p "请输入提交信息: " COMMIT_MSG
    if [[ -z "${COMMIT_MSG:-}" ]]; then
      echo "提交信息为空，已取消。"
      exit 1
    fi
    git commit -m "$COMMIT_MSG"
  fi
fi

if [[ "$MODE" == "current" ]]; then
  echo "推送当前分支: $CURRENT_BRANCH"
  git push -u origin "$CURRENT_BRANCH"
elif [[ "$MODE" == "branch" ]]; then
  if [[ -z "$TARGET_BRANCH" ]]; then
    echo "未提供分支名。"
    exit 1
  fi
  if ! git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
    echo "本地分支不存在: $TARGET_BRANCH"
    exit 1
  fi
  echo "推送指定分支: $TARGET_BRANCH"
  git push -u origin "$TARGET_BRANCH"
elif [[ "$MODE" == "all" ]]; then
  echo "推送所有本地分支..."
  git push origin --all
else
  echo "未知模式: $MODE"
  exit 1
fi
