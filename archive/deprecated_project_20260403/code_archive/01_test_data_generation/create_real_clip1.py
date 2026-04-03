#!/usr/bin/env python3
"""
真实视频混剪脚本 - 从三个短剧视频中裁剪片段并合并
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import json
from datetime import datetime

# 配置
SOURCE_DIR = Path("/Users/zhangxu/.openclaw/workspace/test_videos/source_videos")
OUTPUT_DIR = Path("/Users/zhangxu/.openclaw/workspace/test_videos")

def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }

def create_clip1():
    """
    创建混剪视频1 (clip1_real.mp4)
    从三个视频中各取一段：
    - video1: 0-15秒
    - video2: 30-45秒  
    - video3: 60-75秒
    总时长: 45秒
    """
    print("="*60)
    print("🎬 创建混剪视频1 (clip1_real.mp4)")
    print("="*60)
    
    video1 = SOURCE_DIR / "video1_天赋变异后我无敌了_ep1.mp4"
    video2 = SOURCE_DIR / "video2_开局饕餮血统我吞噬一切_ep1.mp4"
    video3 = SOURCE_DIR / "video3_咒术反噬我有无限血条_ep1.mp4"
    
    output = OUTPUT_DIR / "clip1_real.mp4"
    
    # 检查视频信息
    info1 = get_video_info(video1)
    info2 = get_video_info(video2)
    info3 = get_video_info(video3)
    
    print(f"\n视频1: {video1.name}")
    print(f"  时长: {info1['duration']:.2f}秒, 分辨率: {info1['width']}x{info1['height']}")
    
    print(f"\n视频2: {video2.name}")
    print(f"  时长: {info2['duration']:.2f}秒, 分辨率: {info2['width']}x{info2['height']}")
    
    print(f"\n视频3: {video3.name}")
    print(f"  时长: {info3['duration']:.2f}秒, 分辨率: {info3['width']}x{info3['height']}")
    
    # 使用FFmpeg裁剪并合并
    temp_dir = OUTPUT_DIR / "temp_clips"
    temp_dir.mkdir(exist_ok=True)
    
    print("\n1. 裁剪视频1 (0-15秒)...")
    seg1 = temp_dir / "seg1.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(video1), '-ss', '0', '-t', '15',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(seg1)
    ], check=True)
    
    print("2. 裁剪视频2 (30-45秒)...")
    seg2 = temp_dir / "seg2.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(video2), '-ss', '30', '-t', '15',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(seg2)
    ], check=True)
    
    print("3. 裁剪视频3 (60-75秒)...")
    seg3 = temp_dir / "seg3.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(video3), '-ss', '60', '-t', '15',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(seg3)
    ], check=True)
    
    # 合并片段
    print("4. 合并片段...")
    concat_list = temp_dir / "concat_list.txt"
    with open(concat_list, 'w') as f:
        f.write(f"file '{seg1}'\n")
        f.write(f"file '{seg2}'\n")
        f.write(f"file '{seg3}'\n")
    
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'concat', '-safe', '0', '-i', str(concat_list),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(output)
    ], check=True)
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ 混剪视频1创建完成: {output}")
    
    # 返回视频信息
    output_info = get_video_info(output)
    print(f"   时长: {output_info['duration']:.2f}秒")
    print(f"   分辨率: {output_info['width']}x{output_info['height']}")
    
    return output

def main():
    print("\n" + "="*60)
    print("🎬 真实视频混剪工具")
    print("="*60 + "\n")
    
    # 创建混剪视频1
    clip1_path = create_clip1()
    
    print("\n" + "="*60)
    print("✅ 完成！")
    print("="*60)
    print(f"\n混剪视频1: {clip1_path}")
    print("\n现在可以运行重构脚本生成 clip2_reconstructed_real.mp4")

if __name__ == "__main__":
    main()
