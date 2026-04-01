#!/bin/bash

# 引入配置文件
source ./github_tools/config.sh

# 拉取远程仓库的更改
git pull origin $BRANCH_NAME        