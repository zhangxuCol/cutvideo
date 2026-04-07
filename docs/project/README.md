# 视频二次裁剪与 AI 审片工具 (CutVideo)

当前项目主链路围绕 3 件事：

1. 二次裁剪（重构视频）
2. 每 3 秒内容一致性验证（画面 + 音频 + 字幕）
3. 自动修复（先审片，再按问题类型快修/重构）

## 环境要求

- Python 3.8+
- `ffmpeg`
- `ffprobe`
- `tesseract`
- Whisper 可用环境（本机已装可直接用）

## 安装

```bash
pip install -r requirements.txt
```

## 核心脚本

- `fast_v7.py`：主链路二次裁剪脚本（推荐）
- `skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py`：单视频 AI 审片报告
- `skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py`：批量 3 秒审片 + 自动修复闭环

## 配置文件

统一配置文件：`configurations/ai_pipeline.defaults.json`

- `v6_fast`：`fast_v7.py` 默认参数 section
- `build_ai_video_audit_bundle`：单视频审片默认参数
- `batch_ai_audit_3s`：批量审片与修复默认参数

可通过环境变量指定配置：

```bash
export CUTVIDEO_CONFIG=/abs/path/ai_pipeline.local.json
```

全量参数说明见：`docs/reference/CLI_CONFIG_REFERENCE.md`

## 使用方法

### 1) 二次裁剪

单视频二次裁剪（推荐）：

```bash
python fast_v7.py \
  --config configurations/ai_pipeline.defaults.json \
  --target /abs/material.mp4 \
  --source-dir /abs/source_dir \
  --output /abs/output/xxx_V3_FAST.mp4 \
  --cache /abs/output/cache \
  --no-run-evidence-validation \
  --no-run-ai-verify-snapshots
```

说明：

- `--target`：要重构的素材
- `--source-dir`：源剧集目录
- `--output`：二次裁剪输出视频
- `--cache`：缓存目录（建议放到 output 下）

### 2) 生成每 3 秒内容验证报告

单视频 3 秒审片：

```bash
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --config configurations/ai_pipeline.defaults.json \
  --target /abs/material.mp4 \
  --candidate /abs/output/xxx_V3_FAST.mp4 \
  --interval 3 \
  --clip-duration 2 \
  --max-points 1200 \
  --output-dir /abs/output/reports/xxx_3s
```

批量 3 秒审片（只审片，不重构）：

```bash
python skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py \
  --config configurations/ai_pipeline.defaults.json \
  --material-dir /abs/material_dir \
  --candidate-dir /abs/output \
  --output-root /abs/output/ai_audit_3s_batch
```

### 3) 修复脚本（推荐）

推荐使用批量脚本的自动优化闭环能力，而不是旧的手写循环脚本：

```bash
python skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py \
  --config configurations/ai_pipeline.defaults.json \
  --material-dir /abs/material_dir \
  --candidate-dir /abs/output \
  --source-dir /abs/source_dir \
  --reconstruct-all \
  --clip-jobs 2 \
  --audit-jobs 2 \
  --optimize-on-mismatch \
  --optimize-max-retries 1 \
  --optimize-max-clip-increase-ratio 0.0 \
  --optimize-max-clip-seconds-cap 300 \
  --output-root /abs/output/ai_audit_3s_batch
```

说明：

- `--reconstruct-all`：先统一重构，再统一审片
- `--optimize-on-mismatch`：对不一致样本自动进入修复流程
- `--optimize-max-clip-increase-ratio 0.0`：修复不允许比原裁剪更慢
- `--optimize-max-clip-seconds-cap 300`：单视频裁剪上限 5 分钟

历史脚本 `auto_fix_loop.sh` 仅作归档参考，不建议继续使用（包含旧路径和旧链路）。

## 输出产物

常见输出：

- 二次裁剪视频：`*_V3_FAST.mp4`
- 裁剪质量报告：`*_V3_FAST.quality_report.json`
- 单视频审片清单：`audit_manifest.json`
- 单视频审片页面：`comparison_report.html`
- 批量汇总报告目录：`ai_audit_3s_batch_*`

## 许可证

MIT
