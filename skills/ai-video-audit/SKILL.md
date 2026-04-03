---
name: ai-video-audit
description: 用于“AI真看画面、真听音频、真核对字幕”的视频抽检技能。适用于验证原视频与候选视频在内容、音频语义、字幕文本是否一致；会生成审片包（抽帧、音频切片、OCR字幕、可选ASR文本）供AI逐点审阅。
---

# AI Video Audit

## 什么时候用

当用户提出以下需求时使用：
- "帮我验证视频内容、音频、字幕是否一致"
- "不要算法分数，要 AI 真看真听"
- "抽检这个视频有没有内容错位/字幕不对/音画不一致"

## 核心原则

- 先产出“可审阅证据包”，再给结论。
- 画面结论基于 AI 对抽帧逐点查看。
- 音频结论基于 AI-ASR 文本（本地 ASR 可用时）+ 人工复听建议。
- 字幕结论基于两路信息：字幕文件文本 + 画面底部 OCR。

## 执行步骤

1. 生成审片包

```bash
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --target /abs/path/source.mp4 \
  --candidate /abs/path/candidate.mp4 \
  --interval 8 \
  --clip-duration 6
```

2. 如有外部字幕，追加字幕文件（推荐）

```bash
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --target /abs/path/source.mp4 \
  --candidate /abs/path/candidate.mp4 \
  --target-sub /abs/path/source.srt \
  --candidate-sub /abs/path/candidate.srt
```

2.1 如果 Whisper 在其他环境，显式指定命令（你当前机器示例）

```bash
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --target /abs/path/source.mp4 \
  --candidate /abs/path/candidate.mp4 \
  --asr-cmd "/Users/zhangxu/Library/Python/3.9/bin/whisper" \
  --asr-model base
```

3. AI 逐点审阅
- 读取 `audit_manifest.json`
- 对 `checkpoints` 中每个时间点：
  - 查看 `target_frame` / `candidate_frame`
  - 对比 `ocr_target` / `ocr_candidate`
  - 对比 `subtitle_target_from_file` / `subtitle_candidate_from_file`
  - 若存在 `asr_target` / `asr_candidate`，检查语义是否一致

4. 输出结论格式（必须）
- 总体结论：`通过/不通过`
- 风险等级：`高/中/低`
- 问题清单：`时间点 + 问题类型(画面/音频/字幕) + 证据`
- 建议动作：`重做片段/替换字幕/人工复听区间`

## 音频“真听”说明

- 本技能优先调用本地 ASR（Whisper 系列）将音频切片转文本，作为 AI 听觉证据。
- 若本机未安装 ASR，`asr_*` 字段会标记 `unavailable`，此时需要：
  - 安装本地 ASR 后重跑，或
  - 对问题时间点做人工复听。
- 若 Whisper 装在其他 Python 环境，使用以下参数强制接入：
  - `--asr-cmd "/path/to/whisper"`（直接指定命令）
  - `--asr-python "/path/to/python"`（用该解释器执行 `-m whisper`）

## 产物

默认输出目录：`temp_outputs/ai_video_audit/<timestamp>/`

- `audit_manifest.json`：核心审片清单
- `frames/*.jpg`：抽帧对照
- `audio/*.wav`：音频切片
- `ocr/*.txt`：OCR 文本
- `asr/*.txt`：ASR 文本（若可用）

## 限制

- 硬字幕（烧录字幕）无法直接读取文本流，需依赖 OCR。
- 若无本地 ASR，无法给出“AI听写级”音频文本证据。
