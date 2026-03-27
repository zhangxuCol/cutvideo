#!/usr/bin/env python3
"""
验证 V6 找到的 @5s 位置
"""

from pathlib import Path
import subprocess
from PIL import Image

def extract_frame(video_path, time_sec, output_path):
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', 
           '-q:v', '2', str(output_path)]
    subprocess.run(cmd, capture_output=True)

def main():
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/v6_validation")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("验证 V6 找到的 @5s 位置")
    print("=" * 60)
    
    # 对比时间点
    compare_times = [0, 30, 60]
    
    for t in compare_times:
        print(f"\n@{t}s:")
        
        # 原素材
        target_frame = output_dir / f"target_{t:03d}s.jpg"
        extract_frame(target, t, target_frame)
        print(f"  原素材: {target_frame.name}")
        
        # V6 的 @5s
        source_frame = output_dir / f"v6_5s_{t:03d}s.jpg"
        extract_frame(source, 5 + t, source_frame)
        print(f"  V6@5s:  {source_frame.name}")
    
    print(f"\n{'='*60}")
    print(f"对比帧已保存到: {output_dir}")
    print("请查看图片确认 @5s 是否正确")

if __name__ == '__main__':
    main()
