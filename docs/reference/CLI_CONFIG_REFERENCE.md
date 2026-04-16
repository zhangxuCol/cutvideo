# 配置与 CLI 参数总手册

本文档覆盖主链路当前 3 个可执行脚本的**全部 CLI 参数**、与配置文件 `configurations/ai_pipeline.defaults.json` 的映射关系、推荐用法，以及最终采用的技术方案。

## 最终技术方案

采用统一的参数治理方案：

1. 使用统一 JSON 配置文件作为默认参数源：`configurations/ai_pipeline.defaults.json`
2. 三个脚本都支持 `--config` 指定配置文件
3. 支持环境变量 `CUTVIDEO_CONFIG` 指定配置文件
4. CLI 参数始终可以覆盖配置文件默认值
5. 若配置文件缺失，则回退脚本内置默认值

配置解析优先级：

1. 显式 `--config /path/to/file.json`
2. 环境变量 `CUTVIDEO_CONFIG`
3. 默认候选文件：
   `configurations/ai_pipeline.defaults.json`
   `configurations/ai_pipeline.local.json`

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
    "segment_duration": 5.0,
    "frame_index_sample_interval": 0.3333333333
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
python fast_v7.py \
  --config configurations/ai_pipeline.defaults.json \
  --target /abs/target.mp4 \
  --source-dir /abs/source_dir \
  --output /abs/output.mp4
```

### 2) 单视频证据审片

```bash
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --config configurations/ai_pipeline.defaults.json \
  --target /abs/source.mp4 \
  --candidate /abs/candidate.mp4
```

### 3) 批量 3 秒审片

```bash
python skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py \
  --config configurations/ai_pipeline.defaults.json \
  --material-dir /abs/materials \
  --candidate-dir /abs/output
```

### 4) 使用环境变量统一配置

```bash
export CUTVIDEO_CONFIG=/abs/path/ai_pipeline.local.json
```

## 参数总表

### A. `fast_v7.py`

| CLI 参数 | 类型 | 配置键 (`v6_fast`) | 说明 |
|---|---|---|---|
| `--config` | string | - | 配置文件路径（JSON） |
| `--target` | string | - | 目标视频路径（必填） |
| `--source-dir` | string | - | 源视频目录（必填） |
| `--output` | string | - | 输出视频路径 |
| `--cache` | string | `cache`（可选） | 缓存目录 |
| `--segment-duration` | float | `segment_duration` | 分段时长（秒） |
| `--frame-index-sample-interval` | float | `frame_index_sample_interval` | pHash 帧索引采样间隔（秒） |
| `--workers` | int | `workers` | 并行线程数（0=自动） |
| `--render-workers` | int | `render_workers` | 渲染分段提取线程数（0=自动） |
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
| `--audio-guard-enabled` / `--no-audio-guard-enabled` | bool | `audio_guard_enabled` | 启用/关闭轻量音频守卫（拦截画面对但台词错位） |
| `--audio-guard-score-trigger` | float | `audio_guard_score_trigger` | 音频守卫触发阈值（combined 低于该值触发） |
| `--audio-guard-sample-duration` | float | `audio_guard_sample_duration` | 音频守卫采样时长（秒） |
| `--audio-guard-min-similarity` | float | `audio_guard_min_similarity` | 音频守卫对齐相似度阈值（低于阈值判为可疑） |
| `--audio-guard-hard-floor` | float | `audio_guard_hard_floor` | 音频守卫硬阈值（低于即回退目标段） |
| `--audio-guard-shift-margin` | float | `audio_guard_shift_margin` | 音频偏移收益阈值（用于判定“疑似音频错位”） |
| `--audio-segment-accurate-seek` / `--no-audio-segment-accurate-seek` | bool | `audio_segment_accurate_seek` | 带音轨分段提取使用精确寻址（降低词尾重复/音画错位） |
| `--segment-shortfall-pad` / `--no-segment-shortfall-pad` | bool | `segment_shortfall_pad` | 分段素材不足时补齐末帧与静音，避免局部掉音/黑帧 |
| `--source-tail-safety-enabled` / `--no-source-tail-safety-enabled` | bool | `source_tail_safety_enabled` | 启用源片尾安全边距（避开片尾高风险黑场/尾音异常） |
| `--source-tail-safety-margin` | float | `source_tail_safety_margin` | 源片尾安全边距（秒） |
| `--source-tail-safety-target-tail-ignore-sec` | float | `source_tail_safety_target_tail_ignore_sec` | 目标视频尾段忽略源片尾安全边距检查范围（秒） |
| `--source-tail-safety-switch-min-gain` | float | `source_tail_safety_switch_min_gain` | 源片尾安全切换最小收益阈值（verify_avg 提升量） |
| `--cross-source-head-nudge-enabled` / `--no-cross-source-head-nudge-enabled` | bool | `cross_source_head_nudge_enabled` | 启用跨源边界头部微调（减少接缝重复词/黑一下） |
| `--cross-source-head-nudge-prev-tail-window` | float | `cross_source_head_nudge_prev_tail_window` | 跨源微调触发：前段接近源片尾阈值（秒） |
| `--cross-source-head-nudge-curr-head-window` | float | `cross_source_head_nudge_curr_head_window` | 跨源微调触发：后段接近源片头阈值（秒） |
| `--cross-source-head-nudge-max-offset` | float | `cross_source_head_nudge_max_offset` | 跨源边界后段最大前移量（秒） |
| `--cross-source-head-nudge-score-bias` | float | `cross_source_head_nudge_score_bias` | 跨源微调评分偏置（越大越偏向更大前移） |
| `--cross-source-head-nudge-max-verify-drop` | float | `cross_source_head_nudge_max_verify_drop` | 跨源微调允许的画面核验下降上限 |
| `--backprop-overlap-fix-no-target` / `--no-backprop-overlap-fix-no-target` | bool | `no_target_backprop_overlap_fix` | 禁兜底模式下启用同源尾段重叠反向回推修复 |
| `--no-target-backprop-max-shift` | float | `no_target_backprop_max_shift` | 禁兜底尾段重叠回推最大允许修正量（秒） |
| `--no-target-backprop-min-quality` | float | `no_target_backprop_min_quality` | 禁兜底尾段回推最小置信门限（combined） |
| `--boundary-rematch-no-target` / `--no-boundary-rematch-no-target` | bool | `no_target_boundary_rematch_enabled` | 禁兜底模式下对未收敛边界启用定点重匹配 |
| `--no-target-boundary-rematch-max-attempts` | int | `no_target_boundary_rematch_max_attempts` | 禁兜底未收敛边界定点重匹配最大尝试次数 |
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
| `--run-evidence-validation` / `--no-run-evidence-validation` | bool | `run_evidence_validation` | 重构完成后是否自动执行证据级验证（默认关闭） |
| `--run-ai-verify-snapshots` / `--no-run-ai-verify-snapshots` | bool | `run_ai_verify_snapshots` | 渲染完成后是否执行 AI 抽样帧核验（默认关闭） |
| `--verify-whisper-candidates` | string(csv) | `verify_whisper_python_candidates` | Whisper Python 候选列表 |
| `--allow-numeric-fallback` / `--no-allow-numeric-fallback` | bool | `allow_numeric_fallback` | 证据验证失败后是否回退数值校验 |

示例：

```bash
python fast_v7.py \
  --config configurations/ai_pipeline.defaults.json \
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
| `--asr-python-candidates` | string(csv) | `asr_python_candidates` | 自动探测 Whisper 的 Python 候选列表 |
| `--clip-elapsed-sec` | float | - | 二次裁剪耗时（秒，通常由批处理注入） |
| `--candidate-mode` | string | - | 候选来源（`existing` / `reconstructed`） |
| `--output-dir` | string | `output_dir` | 输出目录 |

示例：

```bash
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --config configurations/ai_pipeline.defaults.json \
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
| `--source-dir` | string | `source_dir` | 源剧集目录（重构与优化时必需） |
| `--reconstruct-missing` | bool(flag) | - | 候选缺失时自动重构 |
| `--reconstruct-all` | bool(flag) | - | 忽略现有候选，全部重构后再审片 |
| `--jobs` | int | `jobs` | 并发总开关（未指定 clip/audit 时作为默认） |
| `--clip-jobs` | int | `clip_jobs` | 并发裁剪任务数 |
| `--audit-jobs` | int | `audit_jobs` | 并发审片任务数 |
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
| `--cache-dir` | string | `cache_dir` | 重构缓存目录 |
| `--reconstruct-script` | string | `reconstruct_script` | 重构脚本（默认 `fast_v7.py`） |
| `--reconstruct-workers` | int | `reconstruct_workers` | 重构匹配并发数 |
| `--reconstruct-render-workers` | int | `reconstruct_render_workers` | 重构渲染并发数 |
| `--reconstruct-segment-duration` | float | `reconstruct_segment_duration` | 重构分段时长 |
| `--reconstruct-low-score-threshold` | float | `reconstruct_low_score_threshold` | 重构低分阈值 |
| `--reconstruct-force-target-audio` / `--no-reconstruct-force-target-audio` | bool | `reconstruct_force_target_audio` | 重构是否强制目标音轨 |
| `--reconstruct-strict-visual-verify` / `--no-reconstruct-strict-visual-verify` | bool | `reconstruct_strict_visual_verify` | 重构是否启用严格画面核验 |
| `--reconstruct-boundary-glitch-fix` / `--no-reconstruct-boundary-glitch-fix` | bool | `reconstruct_boundary_glitch_fix` | 重构是否启用边界单帧修复 |
| `--optimize-on-mismatch` / `--no-optimize-on-mismatch` | bool | `optimize_on_mismatch` | 不一致时自动优化重构并复审 |
| `--optimize-max-retries` | int | `optimize_max_retries` | 自动优化最大重试次数 |
| `--optimize-max-clip-increase-ratio` | float | `optimize_max_clip_increase_ratio` | 自动优化允许的裁剪耗时增幅比例（`0` 表示不允许变慢） |
| `--optimize-max-clip-seconds-cap` | float | `optimize_max_clip_seconds_cap` | 自动优化单次重构绝对耗时上限（秒，`<=0` 表示不限制） |
| `--optimize-audio-remux-on-audio-mismatch` / `--no-optimize-audio-remux-on-audio-mismatch` | bool | `optimize_audio_remux_on_audio_mismatch` | 仅音频硬不一致时优先做目标音轨快修 |
| `--optimize-audio-remux-min-visual` | float | `optimize_audio_remux_min_visual` | 音轨快修触发的最低视觉一致度 |
| `--optimize-visual-overlay-on-visual-mismatch` / `--no-optimize-visual-overlay-on-visual-mismatch` | bool | `optimize_visual_overlay_on_visual_mismatch` | 少量视觉硬不一致时优先做局部画面覆盖快修 |
| `--optimize-visual-overlay-window-sec` | float | `optimize_visual_overlay_window_sec` | 视觉快修单点前后覆盖窗口（秒） |
| `--optimize-visual-overlay-merge-gap-sec` | float | `optimize_visual_overlay_merge_gap_sec` | 视觉快修相邻窗口合并间隔（秒） |
| `--optimize-visual-overlay-max-points` | int | `optimize_visual_overlay_max_points` | 视觉快修可处理的最大视觉硬不一致点数 |
| `--optimize-visual-overlay-max-total-window-sec` | float | `optimize_visual_overlay_max_total_window_sec` | 视觉快修允许的覆盖总时长上限（秒） |
| `--optimize-visual-overlay-crf` | int | `optimize_visual_overlay_crf` | 视觉快修重编码 CRF（越大体积越小） |
| `--optimize-visual-overlay-preset` | string | `optimize_visual_overlay_preset` | 视觉快修重编码 preset（速度/压缩率权衡） |
| `--optimize-adjacent-overlap-trigger` | float | `optimize_adjacent_overlap_trigger` | 自动优化相邻重叠触发阈值 |
| `--optimize-adjacent-lag-trigger` | float | `optimize_adjacent_lag_trigger` | 自动优化相邻慢进触发阈值 |
| `--optimize-isolated-drift-trigger` | float | `optimize_isolated_drift_trigger` | 自动优化孤立漂移触发阈值 |
| `--optimize-cross-source-mapping-jump-trigger` | float | `optimize_cross_source_mapping_jump_trigger` | 自动优化跨源跳变触发阈值 |

示例：

```bash
python skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py \
  --config configurations/ai_pipeline.defaults.json \
  --material-dir /abs/materials \
  --candidate-dir /abs/output \
  --source-dir /abs/source \
  --reconstruct-all \
  --clip-jobs 2 \
  --audit-jobs 2 \
  --optimize-on-mismatch \
  --optimize-max-clip-seconds-cap 180 \
  --optimize-audio-remux-on-audio-mismatch \
  --optimize-max-clip-increase-ratio 0.0
```

## 注意事项

1. `--config` 指向的文件必须是 JSON，且根节点必须是对象。
2. 配置文件里若某个键类型错误（如把 `float` 写成不可解析字符串）会直接报错退出。
3. 布尔参数建议显式写出：例如 `--force-target-audio` 或 `--no-force-target-audio`，避免默认值歧义。
4. `--reconstruct-missing` / `--reconstruct-all` 是纯 CLI 行为开关，不从配置文件读取。
5. 若没有外部字幕文件，字幕一致性会基于 OCR；OCR 对浮水印/花字较敏感，可能带来误报。
6. 若 ASR 未成功初始化（模型不可用或命令不可用），音频对比会降级，报告会在 `asr_error` 给出原因。
7. `fast_v7.py` 的 `--verify-whisper-candidates` 为逗号分隔路径列表；仅在未显式提供 `--verify-asr-cmd/--verify-asr-python` 时用于自动探测。
8. 推荐保留一份 `ai_pipeline.local.json`（不入库）用于机器本地路径与模型偏好。
9. `fast_v7.py` 的 `*.quality_report.json` 已包含 `timing`、`render_metrics` 与逐段 `render_extract_elapsed_sec`，可直接查看“二次裁剪耗时/渲染耗时”。
10. 追求“先出片再审片”时，建议默认关闭 `run_evidence_validation` 与 `run_ai_verify_snapshots`，在批量出片完成后再单独跑审片脚本。
11. 批量脚本在自动优化阶段会把“允许耗时上限”同时用于超时中断；并且可用 `optimize_max_clip_seconds_cap` 额外限制绝对时长。超时即判定本次优化无效并跳过替换。
12. 当未提供外部字幕文件时，字幕通道默认走 OCR；OCR 仅作为软证据，硬不一致优先由画面与音频判定，减少角标/水印误报。
13. `build_ai_video_audit_bundle.py` 在 `asr=auto` 下会自动探测 `whisper` 命令与 `asr_python_candidates`，确保可用环境下真正执行语音转写。
14. 批量优化阶段默认有三层快修策略：`音轨快修`（audio-only）→ `局部画面覆盖快修`（少量视觉硬不一致）→ `重构优化`；每一步都受“不能慢于预算”的速度守护约束。

## 推荐落地方式

1. 仓库保留通用基线：`ai_pipeline.defaults.json`
2. 机器本地维护覆盖文件：`ai_pipeline.local.json`
3. 生产/批处理命令统一带 `--config`，重要参数在命令行二次覆盖

```bash
python fast_v7.py --config configurations/ai_pipeline.local.json ...
```
