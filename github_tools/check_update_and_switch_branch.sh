#!/bin/bash

set -e

# 引入配置文件
source ./github_tools/config.sh

echo "正在获取远程仓库分支信息..."
git fetch --all --prune

echo ""
echo "远程分支列表："
git branch -r | sed 's/^ *//'

echo ""
echo "请输入要切换的分支名称（例如: main 或 gt2）:"
read TARGET_BRANCH

if [ -z "$TARGET_BRANCH" ]; then
  echo "未输入分支名称，脚本退出。"
  exit 1
fi

if ! git show-ref --verify --quiet "refs/remotes/origin/$TARGET_BRANCH"; then
  echo "远程不存在分支: origin/$TARGET_BRANCH"
  exit 1
fi

# 如果本地已存在该分支，直接切换；否则基于远程创建跟踪分支
if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  git checkout "$TARGET_BRANCH"
else
  git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
fi

echo "正在更新当前分支..."
git pull --ff-only origin "$TARGET_BRANCH"

echo ""
echo "已切换并更新到分支: $TARGET_BRANCH"
