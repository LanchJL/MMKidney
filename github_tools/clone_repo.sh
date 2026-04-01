#!/bin/bash

# 引入配置文件
source ./github_tools/config.sh

# 克隆远程仓库
git clone git@github.com:$GITHUB_USER/$REPO_NAME.git