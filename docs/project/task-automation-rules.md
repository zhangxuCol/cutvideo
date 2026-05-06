# 任务自动化规则（当前主链路）

本文档只描述当前仍在使用的自动化规则，历史规则已归档。

## 任务 1：二次裁剪

触发条件：
- 新素材待处理，或需要全量重跑。

执行动作：
1. 调用 `fast_v7.py` 进行重构输出。
2. 记录单视频裁剪耗时（`*.quality_report.json` 中的 `timing` 字段）。

完成标准：
- 产出 `*_V3_FAST.mp4`；
- 单视频耗时不超过当前配置上限（默认 300 秒）。

## 任务 2：每 3 秒 AI 审片

触发条件：
- 裁剪视频生成完成。

执行动作：
1. 对输出视频运行 `build_ai_video_audit_bundle.py`（单条）或 `run_batch_ai_audit_3s.py`（批量）。
2. 生成 `audit_manifest.json` 与 `comparison_report.html`。

完成标准：
- 报告产出完整；
- 关键问题点可定位到具体时间点与证据（画面/音频/字幕）。

## 任务 3：不一致自动修复闭环

触发条件：
- 审片存在 `明显不一致`。

执行动作：
1. 运行 `run_batch_ai_audit_3s.py --optimize-on-mismatch`。
2. 按脚本内策略自动执行音轨快修、局部画面覆盖快修或重构优化。
3. 自动复审并输出新报告。

完成标准：
- 不一致点消除或显著下降；
- 修复不允许比原裁剪更慢（`--optimize-max-clip-increase-ratio 0.0`）。

## 执行记录建议

每轮自动化记录建议包含：
- 任务名
- 输入目录
- 输出目录
- 单视频耗时
- 总耗时
- mismatch 数量变化
- 是否触发自动修复及修复轮次

## 已确认执行约定

- 项目素材根目录统一使用 `/Volumes/学习资料/cutvideo_project_videos`。
- 批量二次裁剪、暂停修复、回归护栏、批量恢复、状态检查需要读写该目录及其项目级缓存时，视为本项目已确认操作。
- 后续同类操作不再作为需要人工决策的问题中断流程；若运行环境弹出系统级授权限制，应继续按既定流程请求/使用相同权限并恢复执行。
- 修复后必须先跑外置回归护栏，再按 `failed_materials -> source_pool_gap_failed_materials -> inflight_materials_at_pause -> remaining_materials` 恢复批量；回归不通过不得恢复批量。
- 回归护栏允许“重新生成结果”或“重新验证已有合格报告”两种模式；对耗时很长的稳定样本，优先重新验证最近合格报告，缺少合格报告才重构，重构必须有超时和日志，避免恢复流程被单条护栏样本长时间卡住。
- “最终失败累计 3 条即暂停修复”的阈值统计本轮恢复后新增的所有最终失败，包括 `insufficient_coverage_in_source_pool`；这类失败统一归类为 `source_pool_gap`，不再叫不可修复，暂停后自动走带标记的目标素材兜底修复路径。
- `target_sequence_low_confidence` 若表现为整条无命中（`available_segments=0` 且缺段数覆盖全部分段），归入 `all_unmatched`，直接进入目标素材兜底输出，并在质量报告中显式标记，不能伪装成源片匹配成功。
- 真正不可修复失败仍要保留在批量汇总和状态文件中，明确给出素材、失败阶段、缺段数量和原因，但不进入恢复重跑队列。

## 说明

- 旧链路中提到的 `v6_precision.py`、`av_consistency_checker.py`、`auto_fix_loop.sh` 不再作为自动化入口。
