#!/bin/bash
# 将代码合并到主分支并推送到远端

echo "开始合并代码到主分支..."

# 获取当前分支名
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# 切换到主分支
git checkout main || git checkout master || {
    echo "错误: 无法切换到主分支"
    exit 1
}

# 获取主分支名
MAIN_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "当前主分支: $MAIN_BRANCH"
echo "正在合并分支 $CURRENT_BRANCH 到 $MAIN_BRANCH..."

# 合并当前分支到主分支
git merge $CURRENT_BRANCH

# 添加所有非jpg文件
git add -- ':!*.jpg'

# 提交更改
git commit -m "合并 $CURRENT_BRANCH 到 $MAIN_BRANCH"

# 推送到远程仓库
git push

echo "代码已成功合并到主分支并推送到远端！"

# 切换回原分支
git checkout $CURRENT_BRANCH
