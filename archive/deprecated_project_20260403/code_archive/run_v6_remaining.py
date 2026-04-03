#!/usr/bin/env python3
"""
V6 批量处理剩余视频
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v6_final import VideoReconstructorHybridV6
from pathlib import Path
import json
import subprocess

def get_video_duration(video_path: Path) -> float:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def main():
    cut_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
    source_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output")
    
    # 已完成的视频
    done = [
        "115192-1-363935817413439491_cut.mp4",
        "115196-1-363935819124715523_cut.mp4",
        "115200-1-363935820852768771_cut.mp4",
        "115204-1-363935821733572611_cut.mp4",
        "115208-1-363935826842234883_cut.mp4",
    ]
    
    source_videos = sorted(source_dir.glob("*.mp4"))
    source_paths = [str(sv) for sv in source_videos]
    
    print(f"🎬 V6 继续处理剩余视频")
    
    # 获取未完成的视频
    cut_videos = sorted(cut_dir.glob("*.mp4"))
    pending = [v for v in cut_videos if f"{v.stem}_cut.mp4" not in done]
    
    print(f"   待处理: {len(pending)} 个\n")
    
    config = {
        'fps': 2,
        'similarity_threshold': 0.85,
        'match_threshold': 0.6,
        'audio_weight': 0.4,
        'video_weight': 0.6
    }
    
    for i, cut_video in enumerate(pending, 1):
        output_path = output_dir / f"{cut_video.stem}_cut.mp4"
        
        print(f"[{i}/{len(pending)}] {cut_video.name}")
        
        try:
            reconstructor = VideoReconstructorHybridV6(
                str(cut_video),
                source_paths,
                config
            )
            
            segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
            
            if segments and output_path.exists():
                print(f"   ✅ 完成\n")
            else:
                print(f"   ❌ 失败\n")
                
        except Exception as e:
            print(f"   ❌ 错误: {e}\n")

if __name__ == '__main__':
    main()
