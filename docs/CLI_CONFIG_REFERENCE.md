# 配置与 CLI 参数总手册

本文档覆盖主链路当前 3 个可执行脚本的**全部 CLI 参数**、与配置文件 `06_configurations/ai_pipeline.defaults.json` 的映射关系、推荐用法，以及最终采用的技术方案。

## 最终技术方案

采用统一的参数治理方案：

1. 使用统一 JSON 配置文件作为默认参数源：`06_configurations/ai_pipeline.defaults.json`
2. 三个脚本都支持 `--config` 指定配置文件
3. 支持环境变量 `CUTVIDEO_CONFIG` 指定配置文件
4. CLI 参数始终可以覆盖配置文件默认值
5. 若配置文件缺失，则回退脚本内置默认值

配置解析优先级：

1. 显式 `--config /path/to/file.json`
2. 环境变量 `CUTVIDEO_CONFIG`
3. 默认候选文件：
   `06_configurations/ai_pipeline.defaults.json`
   `06_configurations/ai_pipeline.local.json`

参数值优先级：

1. CLI 显式传参
2. 配置文件 section 值
3. 脚本内置默认值

## 配置文件结构

配置文件包含 3 个 section：

- `v6_fast`
- `build_ai_video_audit_bundle`
- `batch_ai_audit_3s`

示例：

```json
{
  "v6_fast": {
    "segment_duration": 5.0
  },
  "build_ai_video_audit_bundle": {
    "interval": 8.0
  },
  "batch_ai_audit_3s": {
    "interval": 3.0
  }
}
```

## 使用方法

### 1) 主链路重构 + 证据验证

```bash
python v6_fast.py \
  --config 06_configurations/ai_pipeline.defaults.json \
  --target /abs/target.mp4 \
  --source-dir /abs/source_dir \
  --output /abs/output.mp4
```

### 2) 单视频证据审片

```bash
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --config 06_configurations/ai_pipeline.defaults.json \
  --target /abs/source.mp4 \
  --candidate /abs/candidate.mp4
```

### 3) 批量 3 秒审片

```bash
python skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py \
  --config 06_configurations/ai_pipeline.defaults.json \
  --material-dir /abs/materials \
  --candidate-dir /abs/output
```

### 4) 使用环境变量统一配置

```bash
export CUTVIDEO_CONFIG=/abs/path/ai_pipeline.local.json
```

## 参数总表

### A. `v6_fast.py`

| CLI 参数 | 类型 | 配置键 (`v6_fast`) | 说明 |
|---|---|---|---|
| `--config` | string | - | 配置文件路径（JSON） |
| `--target` | string | - | 目标视频路径（必填） |
| `--source-dir` | string | - | 源视频目录（必填） |
| `--output` | string | - | 输出视频路径 |
| `--cache` | string | `cache`（可选） | 缓存目录 |
| `--segment-duration` | float | `segment_duration` | 分段时长（秒） |
| `--workers` | int | `workers` | 并行线程数（0=自动） |
| `--low-score-threshold` | float | `low_score_threshold` | 低分段重匹配阈值 |
| `--rematch-window` | int | `rematch_window` | 局部重匹配起始窗口（秒） |
| `--rematch-max-window` | int | `rematch_max_window` | 局部重匹配最大窗口（秒） |
| `--continuity-weight` | float | `continuity_weight` | 连续性奖励权重 |
| `--strict-visual-verify` / `--no-strict-visual-verify` | bool | `strict_visual_verify` | 启用/关闭严格画面核验 |
| `--strict-verify-min-sim` | float | `strict_verify_min_sim` | 严格核验最低相似度阈值 |
| `--tail-guard-seconds` | float | `tail_guard_seconds` | 尾段守卫时长（秒） |
| `--tail-verify-min-avg` | float | `tail_verify_min_avg` | 尾段守卫平均相似度阈值 |
| `--tail-verify-min-floor` | float | `tail_verify_min_floor` | 尾段守卫最低点相似度阈值 |
| `--adjacent-overlap-trigger` | float | `adjacent_overlap_trigger` | 相邻段重叠触发阈值（秒） |
| `--adjacent-lag-trigger` | float | `adjacent_lag_trigger` | 相邻段慢进触发阈值（秒） |
| `--isolated-drift-trigger` | float | `isolated_drift_trigger` | 孤立段漂移触发阈值（秒） |
| `--cross-source-mapping-jump-trigger` | float | `cross_source_mapping_jump_trigger` | 跨源切换映射跳变阈值（秒） |
| `--boundary-glitch-fix` / `--no-boundary-glitch-fix` | bool | `boundary_glitch_fix` | 启用/关闭段边界单帧突刺修复 |
| `--boundary-glitch-hi-threshold` | float | `boundary_glitch_hi_threshold` | 段边界突刺修复：prev/next 高相似阈值 |
| `--boundary-glitch-lo-threshold` | float | `boundary_glitch_lo_threshold` | 段边界突刺修复：current 低相似阈值 |
| `--boundary-glitch-gap-threshold` | float | `boundary_glitch_gap_threshold` | 段边界突刺修复：current 相对掉分阈值 |
| `--use-audio-matching` / `--no-use-audio-matching` | bool | `use_audio_matching` | 匹配阶段是否使用音频指纹 |
| `--force-target-audio` / `--no-force-target-audio` | bool | `force_target_audio` | 封装阶段是否强制目标音轨 |
| `--verify-interval` | float | `verify_interval` | 证据验证抽检间隔（秒） |
| `--verify-clip-duration` | float | `verify_clip_duration` | 证据验证音频切片时长（秒） |
| `--verify-max-points` | int | `verify_max_points` | 证据验证最大检查点数 |
| `--verify-asr-mode` | enum | `verify_asr_mode` | `auto/none/faster_whisper/whisper` |
| `--verify-target-sub` | string | `verify_target_sub` | 目标字幕文件（可选） |
| `--verify-output-sub` | string | `verify_output_sub` | 输出字幕文件（可选） |
| `--verify-asr-cmd` | string | `verify_asr_cmd` | Whisper 命令路径（可选） |
| `--verify-asr-python` | string | `verify_asr_python` | Whisper Python 路径（可选） |
| `--verify-asr-model` | string | `verify_asr_model` | ASR 模型 |
| `--verify-language` | string | `verify_language` | ASR 语种 |
| `--verify-output-root` | string | `verify_output_root` | 证据验证报告输出根目录（默认输出视频同目录） |
| `--verify-whisper-candidates` | string(csv) | `verify_whisper_python_candidates` | Whisper Python 候选列表 |
| `--allow-numeric-fallback` / `--no-allow-numeric-fallback` | bool | `allow_numeric_fallback` | 证据验证失败后是否回退数值校验 |

示例：

```bash
python v6_fast.py \
  --config 06_configurations/ai_pipeline.defaults.json \
  --target /abs/target.mp4 \
  --source-dir /abs/source_dir \
  --verify-asr-mode auto \
  --verify-interval 3 \
  --verify-clip-duration 2
```

### B. `build_ai_video_audit_bundle.py`

| CLI 参数 | 类型 | 配置键 (`build_ai_video_audit_bundle`) | 说明 |
|---|---|---|---|
| `--config` | string | - | 配置文件路径（JSON） |
| `--target` | string | - | 基准视频（必填） |
| `--candidate` | string | - | 候选视频（必填） |
| `--target-sub` | string | - | 基准字幕文件（可选） |
| `--candidate-sub` | string | - | 候选字幕文件（可选） |
| `--interval` | float | `interval` | 抽检间隔（秒） |
| `--clip-duration` | float | `clip_duration` | 每点音频切片时长（秒） |
| `--max-points` | int | `max_points` | 最大检查点数量 |
| `--ocr-lang` | string | `ocr_lang` | Tesseract 语言 |
| `--language` | string | `language` | ASR 语种 |
| `--asr-model` | string | `asr_model` | ASR 模型 |
| `--asr` | enum | `asr` | `auto/none/faster_whisper/whisper` |
| `--asr-cmd` | string | `asr_cmd` | Whisper 命令路径 |
| `--asr-python` | string | `asr_python` | Whisper Python 路径 |
| `--clip-elapsed-sec` | float | - | 二次裁剪耗时（秒，通常由批处理注入） |
| `--candidate-mode` | string | - | 候选来源（`existing` / `reconstructed`） |
| `--output-dir` | string | `output_dir` | 输出目录 |

示例：

```bash
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --config 06_configurations/ai_pipeline.defaults.json \
  --target /abs/source.mp4 \
  --candidate /abs/candidate.mp4 \
  --interval 3 \
  --clip-duration 2 \
  --asr auto
```

### C. `run_batch_ai_audit_3s.py`

| CLI 参数 | 类型 | 配置键 (`batch_ai_audit_3s`) | 说明 |
|---|---|---|---|
| `--config` | string | - | 配置文件路径（JSON） |
| `--material-dir` | string | - | 原素材目录（必填） |
| `--candidate-dir` | string | - | 二剪目录（必填） |
| `--source-dir` | string | `source_dir` | 源剧集目录（用于缺失候选时重构） |
| `--reconstruct-missing` | bool(flag) | - | 候选缺失时自动调用 `v6_fast.py` 重构 |
| `--interval` | float | `interval` | 抽检间隔（秒） |
| `--clip-duration` | float | `clip_duration` | 每点音频切片时长（秒） |
| `--max-points` | int | `max_points` | 每条视频最大检查点 |
| `--asr` | enum | `asr` | `auto/none/faster_whisper/whisper` |
| `--asr-cmd` | string | `asr_cmd` | Whisper 命令路径 |
| `--asr-python` | string | `asr_python` | Whisper Python 路径 |
| `--asr-model` | string | `asr_model` | ASR 模型 |
| `--language` | string | `language` | ASR 语种 |
| `--target-sub` | string | `target_sub` | 原素材字幕文件（可选） |
| `--candidate-sub` | string | `candidate_sub` | 二剪字幕文件（可选） |
| `--output-root` | string | `output_root` | 报告输出根目录 |
| `--cache-dir` | string | `cache_dir` | v6_fast 缓存目录 |
| `--reconstruct-workers` | int | `reconstruct_workers` | 缺失候选时 v6_fast 线程数 |
| `--reconstruct-segment-duration` | float | `reconstruct_segment_duration` | 缺失候选时分段时长 |
| `--reconstruct-low-score-threshold` | float | `reconstruct_low_score_threshold` | 缺失候选时低分阈值 |
| `--reconstruct-force-target-audio` / `--no-reconstruct-force-target-audio` | bool | `reconstruct_force_target_audio` | 缺失候选时是否强制目标音轨 |

示例：

```bash
python skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py \
  --config 06_configurations/ai_pipeline.defaults.json \
  --material-dir /abs/materials \
  --candidate-dir /abs/output \
  --source-dir /abs/source \
  --reconstruct-missing
```

## 注意事项

1. `--config` 指向的文件必须是 JSON，且根节点必须是对象。
2. 配置文件里若某个键类型错误（如把 `float` 写成不可解析字符串）会直接报错退出。
3. 布尔参数建议显式写出：例如 `--force-target-audio` 或 `--no-force-target-audio`，避免默认值歧义。
4. `--reconstruct-missing` 是纯 CLI 行为开关，目前不从配置文件读取。
5. 若没有外部字幕文件，字幕一致性会基于 OCR；OCR 对浮水印/花字较敏感，可能带来误报。
6. 若 ASR 未成功初始化（模型不可用或命令不可用），音频对比会降级，报告会在 `asr_error` 给出原因。
7. `v6_fast` 的 `--verify-whisper-candidates` 为逗号分隔路径列表；仅在未显式提供 `--verify-asr-cmd/--verify-asr-python` 时用于自动探测。
8. 推荐保留一份 `ai_pipeline.local.json`（不入库）用于机器本地路径与模型偏好。
9. `v6_fast` 的 `*.quality_report.json` 已包含 `timing`、`render_metrics` 与逐段 `render_extract_elapsed_sec`，可直接查看“二次裁剪耗时/渲染耗时”。

## 推荐落地方式

1. 仓库保留通用基线：`ai_pipeline.defaults.json`
2. 机器本地维护覆盖文件：`ai_pipeline.local.json`
3. 生产/批处理命令统一带 `--config`，重要参数在命令行二次覆盖

```bash
python v6_fast.py --config 06_configurations/ai_pipeline.local.json ...
```
