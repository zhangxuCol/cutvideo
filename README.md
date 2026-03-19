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
├── 01_test_data_generation/       # 测试数据生成
│   ├── create_clip1_v2.sh
│   ├── create_clip2_correct.sh
│   ├── create_real_clip1.py
│   ├── create_real_clip2.py
│   ├── generate_clip2_v2.sh
│   ├── generate_test_videos.py
│   ├── quick_generate_clip2.sh
│   ├── real_videos/
│   ├── source_videos/
│   └── test_videos/
├── 02_video_download/             # 视频下载
│   ├── download_video.py
│   ├── download_with_ffmpeg.sh
│   └── download_with_playwright.py
├── 03_reconstruction_algorithms/  # 重构算法
│   ├── auto_reconstruct.py
│   ├── multi_source_reconstructor.py
│   ├── multi_source_reconstructor_config.py
│   ├── reconstruct_clip2_high_threshold.py
│   ├── reconstruct_clip2_improved.py
│   ├── reconstruct_clip2_intelligent.py
│   ├── reconstruct_improved_algorithm.py
│   ├── reconstruct_true_content_match.py
│   ├── reconstruct_video.sh
│   ├── video_reconstructor_fixed.py
│   ├── video_reconstructor_hybrid.py
│   ├── video_reconstructor_optimized.py
│   └── video_reconstructor_parallel.py
├── 04_comparison_validation/      # 比较验证
│   ├── compare_content.py
│   ├── compare_precise.py
│   ├── compare_real_videos.py
│   ├── compare_v2.py
│   └── compare_videos.py
├── 05_utilities/                   # 工具
│   ├── concat_list.txt
│   └── video_timestamp_finder.py
├── 06_configurations/              # 配置文件
│   ├── clip2_reconstructed_reconstruction_log.json
│   ├── cut_reconstruction_config.yaml
│   ├── video_reconstruct_config.yaml
│   └── video_reconstruct_real_config.yaml
├── batch_reconstructor.py          # 批量重构主程序
├── requirements.txt                # Python依赖
└── README.md                       # 项目说明
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
