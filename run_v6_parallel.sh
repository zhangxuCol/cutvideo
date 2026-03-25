#!/bin/bash
# V6 简化并发 - 同时处理 3 个视频

cd /Users/zhangxu/work/项目/cutvideo

OUTPUT_DIR="01_test_data_generation/source_videos/南城以北/output_v6_parallel"
mkdir -p "$OUTPUT_DIR"

# 视频列表
VIDEOS=(
    "01_test_data_generation/source_videos/南城以北/adx原/115192-1-363935817413439491.mp4"
    "01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    "01_test_data_generation/source_videos/南城以北/adx原/115200-1-363935820852768771.mp4"
)

# 同时启动 3 个进程
for video in "${VIDEOS[@]}"; do
    name=$(basename "$video" .mp4)
    echo "启动: $name"
    python3 -c "
import sys
sys.path.insert(0, '03_reconstruction_algorithms')
from video_reconstructor_hybrid_v6_final import VideoReconstructorHybridV6

sources = [
    '01_test_data_generation/source_videos/南城以北/剧集/1.mp4',
    '01_test_data_generation/source_videos/南城以北/剧集/2.mp4',
    '01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977611190734848_363977611040526336.mp4',
    '01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977666966589440_363977611271213056.mp4',
    '01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977673945911296_363977667063844864.mp4',
]

config = {'fps': 2, 'similarity_threshold': 0.85, 'match_threshold': 0.6, 'audio_weight': 0.4, 'video_weight': 0.6}

reconstructor = VideoReconstructorHybridV6('$video', sources, config)
reconstructor.reconstruct('$OUTPUT_DIR/${name}_cut.mp4', use_target_audio=True)
print('完成: $name')
" &
done

wait
echo "所有任务完成"
