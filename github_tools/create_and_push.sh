#!/bin/bash

# 引入配置文件
source ./github_tools/config.sh

# 初始化 Git 仓库
git init

# 添加所有文件
git add .

# 提交更改
echo "请输入提交信息:"
read COMMIT_MSG
git commit -m "$COMMIT_MSG"

# 远程仓库 URL 使用 REPO_NAME
git remote add origin $REPO_URL

# 推送到 GitHub
git push -u origin $BRANCH_NAME