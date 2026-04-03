#!/usr/bin/env python3
"""
V6 重新运行 - 修复花屏问题（使用重新编码）
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v6_final import VideoReconstructorHybridV6
from pathlib import Path
import subprocess

def get_video_duration(video_path: Path) -> float:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def main():
    cut_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
    source_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_reencoded")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source_videos = sorted(source_dir.glob("*.mp4"))
    source_paths = [str(sv) for sv in source_videos]
    
    cut_videos = sorted(cut_dir.glob("*.mp4"))
    
    print(f"🎬 V6 重新运行（修复花屏）")
    print(f"   使用重新编码确保帧连续")
    print(f"   预计时间: {len(cut_videos) * 8} 分钟\n")
    
    config = {
        'fps': 2,
        'similarity_threshold': 0.85,
        'match_threshold': 0.6,
        'audio_weight': 0.4,
        'video_weight': 0.6
    }
    
    for i, cut_video in enumerate(cut_videos, 1):
        output_path = output_dir / f"{cut_video.stem}_cut.mp4"
        
        print(f"[{i}/{len(cut_videos)}] {cut_video.name}")
        
        try:
            reconstructor = VideoReconstructorHybridV6(
                str(cut_video),
                source_paths,
                config
            )
            
            segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
            
            if segments and output_path.exists():
                out_duration = get_video_duration(output_path)
                orig_duration = get_video_duration(cut_video)
                coverage = out_duration / orig_duration
                print(f"   ✅ 完成! 覆盖 {coverage:.1%}\n")
            else:
                print(f"   ❌ 失败\n")
                
        except Exception as e:
            print(f"   ❌ 错误: {e}\n")

if __name__ == '__main__':
    main()
