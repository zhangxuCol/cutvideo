#!/usr/bin/env python3
"""
快速测试 V7 的单源匹配 - 只测试 115196
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
# 传入目录，V7 会自动扫描所有视频
source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"

print("=" * 60)
print("V7 快速测试 - 115196")
print("=" * 60)

print(f"\n目标: {target.name}")
print(f"源视频目录: {source_dir}")

# 计算总时长
target_duration = get_duration(target)
print(f"\n目标时长: {target_duration:.1f}s")

# 导入 V7
from video_reconstructor_hybrid_v7 import VideoReconstructorHybridV7

# 创建重构器
reconstructor = VideoReconstructorHybridV7(
    target_video=str(target),
    source_videos=[source_dir],  # 传入目录
    config={'audio_match_threshold': 0.6}
)

try:
    # 获取重构结果（V7 会自动遍历所有源视频）
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v7")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target.stem}_reconstructed.mp4"
    segments = reconstructor.reconstruct(str(output_path), use_target_audio=False)
    
    if segments and len(segments) > 0:
        print(f"\n{'='*60}")
        print("结果:")
        
        if len(segments) == 1:
            # 单源匹配
            seg = segments[0]
            print(f"  类型: 单源完整匹配")
            print(f"  源视频: {seg.source_video.name}")
            print(f"  匹配位置: @{seg.start_time:.0f}s")
            print(f"  相似度: {seg.similarity_score:.1%}")
            
            if 35 <= seg.start_time <= 45:
                print(f"  状态: ✅ 在 @40s 附近")
            elif 160 <= seg.start_time <= 170:
                print(f"  状态: ✅ 在 @165s 附近")
            else:
                print(f"  状态: ⚠️ 位置 @{seg.start_time:.0f}s 可能需要验证")
        else:
            # 多源拼接
            print(f"  类型: 多源拼接 ({len(segments)} 段)")
            total_duration = sum(seg.end_time - seg.start_time for seg in segments)
            print(f"  总覆盖: {total_duration:.1f}s / {target_duration:.1f}s")
            for i, seg in enumerate(segments):
                print(f"    段{i+1}: {seg.source_video.name} @{seg.start_time:.0f}s "
                      f"({seg.end_time - seg.start_time:.1f}s)")
    else:
        print("\n❌ 未找到匹配")
        
finally:
    import shutil
    import tempfile
    if reconstructor.temp_dir and reconstructor.temp_dir.exists():
        shutil.rmtree(reconstructor.temp_dir)
