---
name: goal-constrained-self-improver
description: 目标驱动的代码持续优化技能。用于用户明确给出目标功能、限制约束、验收命令时，执行“实现/触发自动修改-验证-诊断-再实现”闭环，直到验收通过或触发安全停止条件。适用于“只要没达到目标就继续优化”的任务，如性能优化、缺陷修复、重构后行为保持、测试达标和稳定性提升。
---

# Goal Constrained Self Improver

把“优化”变成可验证闭环，而不是主观判断。每轮必须用验收命令说话，未达标就继续迭代。

## 输入契约

先收集并确认以下 4 项，再开始改代码：

1. `Goal`：目标功能和成功定义（可观察、可验证）
2. `Constraints`：不可破坏的限制（接口兼容、性能下限、禁改目录、风格要求等）
3. `Checks`：验收命令清单（测试、lint、类型检查、基准测试）
4. `Stop Rules`：最大迭代次数、单轮超时、人工介入阈值

优先使用模板：`references/spec-template.json`。

若要自动触发修改，`spec.json` 中必须配置 `auto_modify.command`。

## 执行循环

### Step 0: 建立验收基线（只验证）

把目标和约束写入 `spec.json`，然后执行：

```bash
python skills/goal-constrained-self-improver/scripts/goal_gate.py \
  --spec /abs/path/spec.json \
  --report /abs/path/goal_gate_report.json
```

记录当前失败项作为基线，不跳过这一步。

### Step 1: 自动修改模式（二选一）

1. `Agent 模式`：由 Codex 直接改代码，每轮后跑 `goal_gate.py`
2. `Command 模式`：运行自动循环脚本 `auto_improve.py`，由 `auto_modify.command` 触发外部自动改码器

Command 模式：

```bash
python skills/goal-constrained-self-improver/scripts/auto_improve.py \
  --spec /abs/path/spec.json \
  --out-dir /abs/path/runtime/temp_outputs/goal_self_improve
```

### Step 2: 运行验收门

每轮修改后都必须运行 `goal_gate.py`。只有 `passed_all=true` 才算达标。

### Step 3: 诊断与策略切换

若失败，按顺序处理：

1. 修复第一个失败的硬门（hard gate）
2. 若同类失败重复 2 轮以上，切换策略（重构边界、补测试、收窄改动面）
3. 若出现约束冲突，优先保约束并向用户报告冲突点

### Step 4: 继续迭代直到停止条件

持续循环 `改动 -> 验收 -> 诊断`，直到：

1. 全部检查通过
2. 触发停止规则（如达到 max_iterations）
3. 发现不可解冲突（目标和约束互斥）

## `auto_modify.command` 占位符

在 `spec.auto_modify.command` 里可用：

1. `{iteration}`：当前迭代次数（从 1 开始）
2. `{spec}` / `{spec_q}`：spec 路径（原始/带 shell quote）
3. `{last_gate_report}` / `{last_gate_report_q}`：上一轮 gate 报告
4. `{gate_report}` / `{gate_report_q}`：本轮 gate 报告目标路径
5. `{run_dir}` / `{run_dir_q}`：本次自动循环输出目录
6. `{workspace}` / `{workspace_q}`：当前工作目录

## 输出要求

结束时必须输出：

1. 最终状态：`PASS` 或 `STOPPED`
2. 已满足目标列表
3. 未满足项与阻塞原因
4. 最后一次验收报告路径
5. 下一步最小建议动作（若未通过）

## 行为守则

1. 不以“看起来没问题”作为完成标准
2. 不在未跑验收时宣称完成
3. 不为通过验收而破坏约束
4. 不在失败时空转；每轮都给出可执行改进动作
