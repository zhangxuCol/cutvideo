# Docs 导航

当前 `docs/` 目录按用途分为两类：

## 1. 项目文档（`docs/project`）

- `docs/project/README.md`：项目使用说明（二次裁剪、3 秒审片、修复闭环）
- `docs/project/VERSIONS.md`：版本演进与当前主链路说明
- `docs/project/task-automation-rules.md`：自动化任务规则（当前有效版）

## 2. 参考文档（`docs/reference`）

- `docs/reference/CLI_CONFIG_REFERENCE.md`：CLI 与配置参数总手册
- `docs/reference/coding_standards.md`：通用编码规范参考

## 一致性说明

- 生产主链路统一为：
  - `fast_v7.py`
  - `skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py`
  - `skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py`
- 历史方案与日志统一归档在 `archive/`，不作为当前执行依据。
