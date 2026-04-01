#!/bin/bash

# 引入配置文件
source ./github_tools/config.sh

# 提示用户输入实验分支的名称
echo "请输入要创建的实验分支名称:"
read EXPERIMENT_BRANCH
EXPERIMENT_BRANCH=${EXPERIMENT_BRANCH:-"experiment-$RANDOM"}  # 如果没有输入，生成随机名称

# 创建并切换到新分支
git checkout -b $EXPERIMENT_BRANCH

echo "你现在已经切换到分支 '$EXPERIMENT_BRANCH'，可以开始实验了。"