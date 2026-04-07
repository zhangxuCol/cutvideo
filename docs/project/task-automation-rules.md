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

## 说明

- 旧链路中提到的 `v6_precision.py`、`av_consistency_checker.py`、`auto_fix_loop.sh` 不再作为自动化入口。
