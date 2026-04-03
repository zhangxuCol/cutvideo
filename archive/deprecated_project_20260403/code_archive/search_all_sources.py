#!/usr/bin/env python3
"""
在所有源视频中搜索 115196
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from pathlib import Path
from video_reconstructor_hybrid_v6 import VideoReconstructorHybridV6

# 目标视频
target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"

# 所有源视频
source_videos = [
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4",
    "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/2.mp4",
]

print("=" * 60)
print("在所有源视频中搜索 115196")
print("=" * 60)

# 创建重构器
reconstructor = VideoReconstructorHybridV6(
    target_video=target,
    source_videos=source_videos,
    config={}
)

import tempfile
reconstructor.temp_dir = Path(tempfile.mkdtemp())

try:
    # 提取目标音频
    print("\n提取目标音频指纹...")
    target_audio = reconstructor._extract_audio_fingerprint(Path(target))
    print(f"  目标指纹: {len(target_audio)} 帧")
    
    # 在每个源视频中搜索
    print("\n搜索所有源视频...")
    results = []
    
    for source_path in source_videos:
        source = Path(source_path)
        print(f"\n搜索 {source.name}...")
        
        audio_score, start_time = reconstructor._find_audio_match(target_audio, source)
        
        print(f"  最佳匹配: @{start_time}s")
        print(f"  音频评分: {audio_score:.1%}")
        
        results.append({
            'source': source.name,
            'start_time': start_time,
            'audio_score': audio_score
        })
    
    # 找出最佳结果
    best = max(results, key=lambda x: x['audio_score'])
    
    print(f"\n{'='*60}")
    print("搜索结果")
    print("=" * 60)
    for r in results:
        marker = "⭐" if r['source'] == best['source'] else "  "
        print(f"{marker} {r['source']}: @{r['start_time']}s ({r['audio_score']:.1%})")
    
    print(f"\n最佳匹配: {best['source']} @ {best['start_time']}s")
    
finally:
    import shutil
    if reconstructor.temp_dir and reconstructor.temp_dir.exists():
        shutil.rmtree(reconstructor.temp_dir)
