---
name: project-fix-autologger
description: 项目修复日志自动记录技能。用于本项目内所有代码修复、缺陷排查、参数调整、视频裁剪/时间轴修正等任务。只要发生文件修改或问题修复，就自动记录问题现象、根因分析、修改方法、验证结果、涉及文件，以及视频文件与具体时间点，不需要用户显式触发。
---

# 项目修复自动记录技能

在任何“修复类”工作结束时，自动写入详细记录，确保可追溯、可复盘。

## 被动触发要求

这不是一个“用户提醒后再用”的技能，而是一个默认闭环：

1. 只要当前任务属于修复、排障、稳定性改进、参数修正、视频时间轴修正、错段回收、音频缺失修正，就必须自动使用本技能
2. 即使用户没有提到“记录”“日志”“补录”，也要在发现 bug 后先自动落一条 `待修复`
3. 修复完成后，再用同一个问题编号自动补一条 `已修复`
4. 如果最终回复前没有完成“待修复 + 已修复”闭环，说明本技能执行失败，必须先补记录再回复
5. 如果本次没有真正修复代码或逻辑，只做了观察分析，则可以不写 `已修复`，但若已确认存在 bug，仍应先写 `待修复`

## 强制规则

1. 只要本次任务产生了修复行为，就必须写日志
2. 一次任务内有多个问题时，按问题拆成多条记录
3. 每个问题都必须先写 `待修复`，再写 `已修复`
4. 同一个问题的两条记录必须复用同一个“问题编号”
5. 每条记录开头必须包含时间戳，格式：`[YYYY-MM-DD HH:MM:SS]`
6. 若涉及视频，必须写明视频文件名和问题出现时间点（如 `00:03:25`）
7. 所有记录正文和示例内容必须使用中文，避免使用英文表达（路径、命令行参数名除外）
8. 记录脚本默认检查正文是否包含英文；若确需英文，才可显式使用 `--allow-english`
9. 在项目修复流程中，创建进程与终止进程命令视为已获统一授权，默认直接执行，不逐条询问
10. 终止进程后仅简要回显：`已由<技能名>终止进程：<进程标识>`

## 记录目录

默认写入：`project_fix_records/`

1. `fix-history.md`：按时间顺序汇总
2. `fix-history.jsonl`：机器可读结构化记录
3. `entries/<timestamp>.md`：单条详细记录
4. `by-date/YYYY-MM-DD.md`：按天归档

## 执行流程

1. 先收集 bug 信息并写入 `待修复`  
最少包含：问题现象、根因初判、涉及文件、视频时间点。

2. 收集视频维度信息（如适用）  
包含：视频路径、出现时间点、裁剪区间、问题描述。

3. 调用记录脚本写入 `待修复`  
```bash
python3 skills/project-fix-autologger/scripts/log_fix_record.py \
  --issue-id "20260422_170000" \
  --status "待修复" \
  --title "修复标题" \
  --problem "问题现象" \
  --root-cause "初步根因" \
  --solution "计划修复方法" \
  --result "待修复" \
  --verification "待修复" \
  --files fast_v7.py pipeline_config.py \
  --video-file "/绝对路径/input.mp4" \
  --issue-time "00:03:25" \
  --clip-range "00:03:20-00:03:40" \
  --issue-detail "字幕与音频错位"
```

4. 完成代码修复后，再写入 `已修复`  
```bash
python3 skills/project-fix-autologger/scripts/log_fix_record.py \
  --issue-id "20260422_170000" \
  --status "已修复" \
  --title "修复标题" \
  --problem "问题现象" \
  --solution "修改方法" \
  --result "修复结果" \
  --verification "验证命令与结果" \
  --files fast_v7.py pipeline_config.py \
  --video-file "/绝对路径/input.mp4" \
  --issue-time "00:03:25" \
  --clip-range "00:03:20-00:03:40" \
  --issue-detail "字幕与音频错位"
```

5. 在最终回复前确认记录已落盘  
至少给出 `entries/*.md` 路径。

6. 在最终回复前执行一次“记录合规检查”  
确认最近一条 `已修复` 记录已经覆盖本次修复涉及的文件；如果检查失败，必须补录后再结束。

## 自动补全规则

1. 若未显式传 `--files`，脚本会尝试从 `git status --short` 自动抓取改动文件
2. 若未显式传 `--issue-id`，脚本会自动生成问题编号，并在输出里打印
3. 若本次不涉及视频字段，脚本会写 `无`
4. 记录内容不允许空泛描述，必须可复现

## 收尾自检

在每次修复类任务结束前，按下面顺序执行，不允许跳过：

1. 调用 `scripts/log_fix_record.py` 写入 `待修复`
2. 完成修复后，再调用 `scripts/log_fix_record.py` 写入 `已修复`
3. 调用 `scripts/assert_fix_record.py` 检查最近一条 `已修复` 记录是否覆盖本次修复文件
4. 只有当检查通过后，才可以在最终回复里说“已记录”

示例：

```bash
python3 skills/project-fix-autologger/scripts/assert_fix_record.py \
  --workspace . \
  --log-dir project_fix_records \
  --expect-files fast_v7.py scripts/run_clip_one.sh
```

## 模板参考

见 `references/log-template.md`。
