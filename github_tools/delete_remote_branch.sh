#!/bin/bash

# 引入配置文件
source ./github_tools/config.sh

# 提示用户输入要删除的远程分支名称
echo "请输入要删除的远程分支名称:"
read BRANCH_NAME

# 删除远程分支
git push origin --delete $BRANCH_NAME