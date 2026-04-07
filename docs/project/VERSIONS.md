# 项目版本与主链路说明

## 当前结论

当前有效主链路为：

1. `fast_v7.py`：二次裁剪主脚本（推荐）
2. `skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py`：单视频 AI 审片
3. `skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py`：批量 3 秒审片与自动修复闭环

## 版本演进

### 历史阶段（已归档）

- 早期 V6 系列、`v6_precision.py`、`av_consistency_checker.py`、手工循环修复脚本等方案已归档到 `archive/`。
- 这些方案不再作为日常生产链路，保留仅用于历史追溯。

### 当前阶段（生产可用）

- 主裁剪：`fast_v7.py`
- 抽检标准：默认 3 秒间隔（可通过 CLI 覆盖）
- 修复策略：以批量脚本自动闭环为主（音轨快修/局部覆盖/重构优化）

## 配置兼容说明

- 配置文件仍使用 section 名 `v6_fast`，用于兼容历史配置；
- 该 section 现由 `fast_v7.py` 读取并执行，不影响当前链路。

## 推荐工作流

1. 先出片：运行 `fast_v7.py` 批量/单条二次裁剪。
2. 再审片：运行 3 秒 AI 抽检报告。
3. 有差异即修复：运行批量脚本的 `--optimize-on-mismatch` 闭环能力。
4. 复核通过后再提交。

## 不建议继续使用

- `v6_precision.py`
- `av_consistency_checker.py`
- `auto_fix_loop.sh`
- 旧测试配置和旧下载脚本

这些文件已归档，避免继续污染主链路。
