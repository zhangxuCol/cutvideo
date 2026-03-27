#!/usr/bin/env python3
"""
快速分析 115196 的片段来源
"""
import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')
from video_reconstructor_hybrid_v6_optimized import VideoReconstructorHybridV6Optimized
from pathlib import Path
import tempfile

ADX_DIR = Path('/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原')
SOURCE_DIR = Path('/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集')

target_video = ADX_DIR / '115196-1-363935819124715523.mp4'
source_videos = [f for f in SOURCE_DIR.iterdir() if f.suffix == '.mp4']

print(f"分析: {target_video.name}")
print(f"源视频数: {len(source_videos)}")
print("="*60)

reconstructor = VideoReconstructorHybridV6Optimized(
    target_video=str(target_video),
    source_videos=[str(s) for s in source_videos],
    config={'validation_level': 'normal', 'single_source_threshold': 0.85}
)

reconstructor.temp_dir = Path(tempfile.mkdtemp())
target_duration = reconstructor.get_video_duration(target_video)
print(f"目标时长: {target_duration:.1f}s")

# 检查单源匹配
print("\n单源匹配检查...")
single_match = reconstructor.find_best_single_source_match(target_duration)
if single_match:
    print(f"  最佳匹配: {single_match['source'].name}")
    print(f"  综合得分: {single_match['combined_score']:.1%}")
    print(f"  阈值: 85%")
    print(f"  结果: {'通过' if single_match['combined_score'] > 0.85 else '失败'}")
else:
    print("  无匹配")

# 检查多源片段
print("\n多源片段分析...")
segments = reconstructor.find_multi_source_segments(target_duration)
print(f"  总片段数: {len(segments)}")

# 统计每个源的使用情况
source_usage = {}
for seg in segments:
    name = seg.source_video.name
    if name not in source_usage:
        source_usage[name] = []
    source_usage[name].append({
        'start': seg.start_time,
        'end': seg.end_time,
        'target_start': seg.target_start,
        'target_end': seg.target_end,
        'duration': seg.end_time - seg.start_time
    })

print(f"\n使用的源视频:")
for name, segs in source_usage.items():
    total_duration = sum(s['duration'] for s in segs)
    print(f"  {name}: {len(segs)}个片段, 总时长{total_duration:.1f}s")

print(f"\n片段详情:")
for i, seg in enumerate(segments):
    print(f"  片段{i+1}: {seg.source_video.name}")
    print(f"    源位置: {seg.start_time:.1f}s ~ {seg.end_time:.1f}s (时长{seg.end_time-seg.start_time:.1f}s)")
    print(f"    目标位置: {seg.target_start:.1f}s ~ {seg.target_end:.1f}s")
