#!/usr/bin/env python3
"""
使用 V6.1 优化版处理 115196
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v6_optimized import VideoReconstructorHybridV6Optimized
from pathlib import Path

# 只处理 115196
target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
sources = [
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/2.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977611190734848_363977611040526336.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977666966589440_363977611271213056.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977673945911296_363977667063844864.mp4",
]
output = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output/115196-1-363935819124715523_cut_v6_optimized.mp4"

print("🎬 V6.1 优化版测试 - 115196")
print("="*60)

config = {
    'fps': 0.5,
    'similarity_threshold': 0.85,
    'match_threshold': 0.6,
    'audio_weight': 0.4,
    'video_weight': 0.6
}

reconstructor = VideoReconstructorHybridV6Optimized(target, sources, config)
segments = reconstructor.reconstruct(output, use_target_audio=True)

if segments:
    print(f"\n✅ 处理完成！生成了 {len(segments)} 个片段")
    
    # 验证
    print(f"\n🔍 验证结果...")
    sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/04_comparison_validation')
    from compare_videos import compare_videos
    
    result = compare_videos(target, output, 0.90)
    print(f"\n最终相似度: {result['overall_similarity']:.1%}")
else:
    print(f"\n❌ 处理失败")
