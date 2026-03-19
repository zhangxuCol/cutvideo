#!/usr/bin/env python3
"""
真实视频重构脚本 - 从源视频重新裁剪生成clip2
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import shutil

# 配置
SOURCE_DIR = Path("/Users/zhangxu/.openclaw/workspace/test_videos/source_videos")
OUTPUT_DIR = Path("/Users/zhangxu/.openclaw/workspace/test_videos")
CLIP1_PATH = OUTPUT_DIR / "clip1_real.mp4"

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

def create_clip2():
    """
    创建裁剪视频2 (clip2_reconstructed_real.mp4)
    从源视频重新裁剪，应该与clip1内容一致：
    - video1: 0-15秒
    - video2: 30-45秒  
    - video3: 60-75秒
    """
    print("="*60)
    print("🎬 创建裁剪视频2 (clip2_reconstructed_real.mp4)")
    print("="*60)
    
    video1 = SOURCE_DIR / "video1_天赋变异后我无敌了_ep1.mp4"
    video2 = SOURCE_DIR / "video2_开局饕餮血统我吞噬一切_ep1.mp4"
    video3 = SOURCE_DIR / "video3_咒术反噬我有无限血条_ep1.mp4"
    
    output = OUTPUT_DIR / "clip2_reconstructed_real.mp4"
    
    print(f"\n源视频:")
    print(f"  - {video1.name}")
    print(f"  - {video2.name}")
    print(f"  - {video3.name}")
    
    # 使用FFmpeg裁剪并合并
    temp_dir = OUTPUT_DIR / "temp_clips2"
    temp_dir.mkdir(exist_ok=True)
    
    print("\n1. 从 video1 裁剪 0-15秒...")
    seg1 = temp_dir / "seg1.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(video1), '-ss', '0', '-t', '15',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(seg1)
    ], check=True)
    
    print("2. 从 video2 裁剪 30-45秒...")
    seg2 = temp_dir / "seg2.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(video2), '-ss', '30', '-t', '15',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(seg2)
    ], check=True)
    
    print("3. 从 video3 裁剪 60-75秒...")
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
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ 裁剪视频2创建完成: {output}")
    
    # 返回视频信息
    output_info = get_video_info(output)
    print(f"   时长: {output_info['duration']:.2f}秒")
    print(f"   分辨率: {output_info['width']}x{output_info['height']}")
    
    return output

def main():
    print("\n" + "="*60)
    print("🎬 真实视频重构工具")
    print("="*60 + "\n")
    
    # 创建裁剪视频2
    clip2_path = create_clip2()
    
    print("\n" + "="*60)
    print("✅ 完成！")
    print("="*60)
    print(f"\n裁剪视频2: {clip2_path}")

if __name__ == "__main__":
    main()
