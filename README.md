# 视频重构工具 (CutVideo)

批量视频重构工具，用于在源视频中匹配并重构裁剪视频片段。

## 功能特点

- 🎬 批量处理多个裁剪视频
- 🔍 音频指纹快速定位 + 视频帧精修的混合匹配算法
- 📊 生成详细的处理报告
- ⚙️ 可配置的匹配参数和阈值
- 🔄 支持多源视频匹配

## 项目结构

```
cutvideo/
├── v6_fast.py                      # 主链路：视频重构
├── av_consistency_checker.py       # 一致性检查
├── skills/ai-video-audit/          # AI 每 3 秒抽检与报告
├── 01_test_data_generation/        # 测试数据与样本生成脚本
├── 02_video_download/              # 下载脚本
├── 03_reconstruction_algorithms/   # 研究型算法目录（非主链路）
├── 04_comparison_validation/       # 对比验证占位目录
├── 05_utilities/                   # 工具占位目录
├── 06_configurations/              # 配置文件
├── docs/                           # 项目文档
├── logs/                           # 历史运行日志
├── runtime/                        # 运行时产物（缓存/输出）
│   ├── cache/
│   └── temp_outputs/
├── archive/                        # 历史归档
│   ├── deprecated_project_20260403/
│   └── backups/
└── memory/                         # 会话与维护记录
```

## 环境要求

- Python 3.8+
- FFmpeg
- FFprobe

## 安装步骤

### 1. 安装系统依赖

#### macOS
```bash
brew install ffmpeg
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### 2. 创建并激活虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 安装Python依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 批量重构视频

```bash
python batch_reconstructor.py
```

### 自定义路径

修改 `batch_reconstructor.py` 中的路径配置：

```python
CUT_VIDEOS_DIR = 'path/to/cut/videos'
SOURCE_VIDEOS_DIR = 'path/to/source/videos'
OUTPUT_DIR = 'path/to/output'
```

### 使用单个重构算法

```bash
python 03_reconstruction_algorithms/video_reconstructor_hybrid.py
```

## 配置说明

主要配置文件位于 `06_configurations/cut_reconstruction_config.yaml`：

- `fps`: 帧率设置（默认5帧/秒）
- `similarity_threshold`: 相似度阈值（默认0.3）
- `match_threshold`: 匹配阈值（默认0.3）
- `audio_match_threshold`: 音频匹配阈值（默认0.6）
- `max_workers`: 并行线程数（默认8）
- `use_target_audio`: 是否使用目标视频的音频（默认true）
- `min_segment_duration`: 最小片段时长（默认0.2秒）
- `max_segment_gap`: 最大片段间隔（默认3.0秒）

### 主链路统一配置（推荐）

新增统一 JSON 配置文件：`06_configurations/ai_pipeline.defaults.json`

- `v6_fast`：主重构链路默认参数
- `build_ai_video_audit_bundle`：单视频证据审片默认参数
- `batch_ai_audit_3s`：批量 3 秒审片与缺失重构默认参数
- 全量参数备注与示例：`docs/CLI_CONFIG_REFERENCE.md`

使用方式（配置文件默认 + CLI 覆盖）：

```bash
# 主链路重构 + 证据验证
python v6_fast.py --config 06_configurations/ai_pipeline.defaults.json \
  --target /abs/target.mp4 \
  --source-dir /abs/source_dir \
  --output /abs/output.mp4

# 单视频证据审片
python skills/ai-video-audit/scripts/build_ai_video_audit_bundle.py \
  --config 06_configurations/ai_pipeline.defaults.json \
  --target /abs/source.mp4 \
  --candidate /abs/candidate.mp4

# 批量 3 秒审片
python skills/ai-video-audit/scripts/run_batch_ai_audit_3s.py \
  --config 06_configurations/ai_pipeline.defaults.json \
  --material-dir /abs/materials \
  --candidate-dir /abs/output
```

也可通过环境变量指定配置文件：

```bash
export CUTVIDEO_CONFIG=/abs/path/ai_pipeline.local.json
```

## 输出说明

处理完成后，会在输出目录生成：

1. 重构后的视频文件（`*_cut.mp4`）
2. 批量处理报告（`batch_reconstruction_report.json`）

报告包含：
- 总视频数
- 成功/失败数量
- 成功率
- 每个视频的详细处理结果

## 注意事项

1. 确保已安装FFmpeg并可在命令行中使用
2. 源视频和裁剪视频的路径必须正确
3. 输出目录会自动创建
4. 匹配阈值可根据实际情况调整

## 许可证

MIT License
