#!/bin/bash
set -euo pipefail

# 引入配置文件
source ./github_tools/config.sh

usage() {
  cat <<'USAGE'
用法:
  bash github_tools/pull_latest_changes.sh [--current | --branch <name> | --all]

说明:
  --current         拉取当前分支
  --branch <name>   拉取指定分支（会切换到该分支执行 pull 后切回）
  --all             依次拉取所有本地分支（需要干净工作区）

不带参数时会进入交互选择。
USAGE
}

is_clean_worktree() {
  git diff --quiet && git diff --cached --quiet
}

MODE=""
TARGET_BRANCH=""

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
  echo "请选择拉取模式:"
  echo "1) 当前分支 ($CURRENT_BRANCH)"
  echo "2) 指定分支"
  echo "3) 所有本地分支"
  read -r -p "输入 1/2/3: " CHOICE
  case "$CHOICE" in
    1) MODE="current" ;;
    2)
      MODE="branch"
      read -r -p "请输入要拉取的分支名: " TARGET_BRANCH
      ;;
    3) MODE="all" ;;
    *)
      echo "无效选择"
      exit 1
      ;;
  esac
fi

if [[ "$MODE" == "current" ]]; then
  echo "拉取当前分支: $CURRENT_BRANCH"
  git pull --ff-only origin "$CURRENT_BRANCH"
elif [[ "$MODE" == "branch" ]]; then
  if [[ -z "$TARGET_BRANCH" ]]; then
    echo "未提供分支名。"
    exit 1
  fi
  if ! git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
    echo "本地分支不存在: $TARGET_BRANCH"
    exit 1
  fi

  if [[ "$CURRENT_BRANCH" == "$TARGET_BRANCH" ]]; then
    echo "拉取当前分支: $TARGET_BRANCH"
    git pull --ff-only origin "$TARGET_BRANCH"
  else
    if ! is_clean_worktree; then
      echo "当前工作区有未提交变更，无法安全切换分支。请先提交或暂存。"
      exit 1
    fi
    echo "切换并拉取分支: $TARGET_BRANCH"
    git checkout "$TARGET_BRANCH"
    git pull --ff-only origin "$TARGET_BRANCH"
    git checkout "$CURRENT_BRANCH"
  fi
elif [[ "$MODE" == "all" ]]; then
  if ! is_clean_worktree; then
    echo "当前工作区有未提交变更，无法安全执行全部分支拉取。请先提交或暂存。"
    exit 1
  fi

  echo "拉取所有本地分支..."
  git fetch --all --prune
  ORIG_BRANCH="$CURRENT_BRANCH"

  while IFS= read -r BR; do
    [[ -z "$BR" ]] && continue
    echo "-> $BR"
    git checkout "$BR"
    # 仅对存在同名远程分支的本地分支执行 pull
    if git ls-remote --exit-code --heads origin "$BR" >/dev/null 2>&1; then
      git pull --ff-only origin "$BR"
    else
      echo "   跳过（远程不存在同名分支）"
    fi
  done < <(git for-each-ref --format='%(refname:short)' refs/heads)

  git checkout "$ORIG_BRANCH"
  echo "全部分支拉取完成。"
else
  echo "未知模式: $MODE"
  exit 1
fi
