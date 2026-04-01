#!/bin/bash

# 定义脚本所在的目录
TOOL_DIR="./github_tools"

# 检查目录是否存在
if [ ! -d "$TOOL_DIR" ]; then
  echo "目录 $TOOL_DIR 不存在！"
  exit 1
fi

# 遍历目录中的所有 .sh 文件并赋予执行权限
for sh_file in "$TOOL_DIR"/*.sh; do
  if [ -f "$sh_file" ]; then
    chmod +x "$sh_file"
    echo "已为 $sh_file 添加执行权限"
  fi
done

echo "所有 .sh 文件的权限已更新。"