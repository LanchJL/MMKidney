#!/bin/bash

# 引入配置文件
source ./github_tools/config.sh

# 获取当前分支名称
CURRENT_BRANCH=$(git symbolic-ref --short HEAD)

# 添加所有更改到暂存区
git add .

# 提交更改
echo "请输入提交信息:"
read COMMIT_MSG
git commit -m "$COMMIT_MSG"

# 推送当前分支到 GitHub
git push origin $CURRENT_BRANCH