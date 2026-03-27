#!/usr/bin/env python3
"""
实际对比 V7 找到的 @20s 位置 vs 原素材
"""

from pathlib import Path
import subprocess
import tempfile
import shutil

def extract_frame(video_path, time_sec, output_path):
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', 
           '-q:v', '2', str(output_path)]
    subprocess.run(cmd, capture_output=True)

def main():
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    
    temp_dir = Path(tempfile.mkdtemp())
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/comparison_frames")
    output_dir.mkdir(exist_ok=True)
    
    try:
        print("=" * 60)
        print("提取对比帧 - V7 找到的 @20s 位置")
        print("=" * 60)
        
        # 对比时间点
        compare_times = [0, 30, 60, 100, 150, 200]
        
        for t in compare_times:
            if t >= 217:  # 跳过超出时长的
                continue
                
            print(f"\n提取 @{t}s:")
            
            # 原素材帧
            target_frame = output_dir / f"target_{t:03d}s.jpg"
            extract_frame(target, t, target_frame)
            
            # V7 找到的 @20s 对应位置
            source_frame = output_dir / f"source_20s_{t:03d}s.jpg"
            extract_frame(source, 20 + t, source_frame)
            
            print(f"  原素材: {target_frame.name}")
            print(f"  源@20s: {source_frame.name}")
        
        print(f"\n{'='*60}")
        print("对比帧已保存到:")
        print(f"  {output_dir}")
        print("\n请查看这些图片，确认 @20s 是否正确:")
        
        for t in compare_times:
            if t >= 217:
                continue
            print(f"  - target_{t:03d}s.jpg vs source_20s_{t:03d}s.jpg")
        
        # 同时提取 @165s 的对比帧（验证脚本找到的位置）
        print(f"\n同时提取 @165s 位置作为对比:")
        for t in [0, 30, 60]:
            source_frame = output_dir / f"source_165s_{t:03d}s.jpg"
            extract_frame(source, 165 + t, source_frame)
            print(f"  - source_165s_{t:03d}s.jpg")
            
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
