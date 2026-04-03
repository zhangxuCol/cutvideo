#!/usr/bin/env python3
"""
测试 V6 在 115196 上的匹配效果
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from pathlib import Path
import subprocess

def get_duration(video_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

# 测试文件
target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")

print("=" * 60)
print("V6 快速测试 - 115196")
print("=" * 60)

print(f"\n目标: {target.name}")
print(f"源: {source.name}")

target_duration = get_duration(target)
source_duration = get_duration(source)

print(f"目标时长: {target_duration:.1f}s")
print(f"源时长: {source_duration:.1f}s")

# 导入 V6
from video_reconstructor_hybrid_v6 import VideoReconstructorHybridV6

# 创建重构器
reconstructor = VideoReconstructorHybridV6(
    target_video=str(target),
    source_videos=[str(source)],
    config={}
)

# 提取音频指纹并搜索
import tempfile
reconstructor.temp_dir = Path(tempfile.mkdtemp())

try:
    print("\n提取目标音频指纹...")
    target_audio = reconstructor._extract_audio_fingerprint(target)
    print(f"  目标指纹: {len(target_audio)} 帧 (8kHz)")
    
    print("\n搜索最佳音频匹配...")
    audio_score, start_time = reconstructor._find_audio_match(target_audio, source)
    
    print(f"\n音频匹配结果:")
    print(f"  位置: @{start_time}s")
    print(f"  音频评分: {audio_score:.1%}")
    
    # 视频验证
    print("\n视频验证 (3个采样点)...")
    sample_times = [0, target_duration * 0.25, target_duration * 0.5]
    video_scores = []
    
    for t in sample_times:
        target_frame = reconstructor.temp_dir / f"target_{t:.0f}.jpg"
        source_frame = reconstructor.temp_dir / f"source_{t:.0f}.jpg"
        
        reconstructor.extract_frame_at(target, t, target_frame)
        reconstructor.extract_frame_at(source, start_time + t, source_frame)
        
        if target_frame.exists() and source_frame.exists():
            sim = reconstructor.calculate_frame_similarity(target_frame, source_frame)
            video_scores.append(sim)
            print(f"  @{t:.0f}s: {sim:.1%}")
    
    video_score = sum(video_scores) / len(video_scores) if video_scores else 0
    combined = 0.4 * audio_score + 0.6 * video_score
    
    print(f"\n{'='*60}")
    print("V6 结果:")
    print(f"  匹配位置: @{start_time:.0f}s")
    print(f"  音频评分: {audio_score:.1%}")
    print(f"  视频评分: {video_score:.1%}")
    print(f"  综合评分: {combined:.1%}")
    
    if 35 <= start_time <= 45:
        print(f"  状态: ✅ 在 @40s 附近")
    elif 160 <= start_time <= 170:
        print(f"  状态: ✅ 在 @165s 附近")
    else:
        print(f"  状态: ⚠️ 位置 @{start_time:.0f}s 需要验证")
        
finally:
    import shutil
    if reconstructor.temp_dir and reconstructor.temp_dir.exists():
        shutil.rmtree(reconstructor.temp_dir)
