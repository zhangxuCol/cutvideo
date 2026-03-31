#!/bin/bash
# 提交除jpg文件以外的所有代码到远端

echo "开始提交代码..."

# 添加所有非jpg文件
git add -- ':!*.jpg'

# 提交更改
git commit -m "提交代码更新"

# 推送到远程仓库
git push

echo "代码提交完成！"
