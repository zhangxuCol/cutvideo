#!/usr/bin/env python3
"""
测试 V5.1 修复版
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v5_fixed import VideoReconstructorHybridV5Fixed
from pathlib import Path

target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
sources = [
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/2.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977611190734848_363977611040526336.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977666966589440_363977611271213056.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977673945911296_363977667063844864.mp4",
]
output = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output/115196_v5_fixed.mp4"

config = {
    'fps': 2,
    'similarity_threshold': 0.85,
    'audio_weight': 0.4,
    'video_weight': 0.6,
    'match_threshold': 0.6
}

print("🎬 测试 V5.1 修复版")
print("="*60)

reconstructor = VideoReconstructorHybridV5Fixed(target, sources, config)
segments = reconstructor.reconstruct(output, use_target_audio=True)

if segments:
    print(f"\n✅ 完成! 生成了 {len(segments)} 个片段")
    
    # 验证
    sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/04_comparison_validation')
    from compare_videos import compare_videos
    
    print(f"\n🔍 验证结果...")
    result = compare_videos(target, output, 0.90)
    print(f"   相似度: {result['overall_similarity']:.1%}")
    print(f"   通过: {'✅ 是' if result['passed'] else '❌ 否'}")
else:
    print(f"\n❌ 处理失败")
