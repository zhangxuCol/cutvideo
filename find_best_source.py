#!/usr/bin/env python3
"""
检查所有源视频寻找 115196 的最佳匹配
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import wave
import struct

def get_video_duration(video_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def extract_frame(video_path, time_sec, temp_dir):
    frame_path = temp_dir / f"frame_{video_path.stem}_{time_sec}.jpg"
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', str(time_sec), '-i', str(video_path),
        '-vframes', '1', str(frame_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return frame_path if frame_path.exists() else None

def compare_frames(frame1_path, frame2_path):
    img1 = cv2.imread(str(frame1_path))
    img2 = cv2.imread(str(frame2_path))
    if img1 is None or img2 is None:
        return 0.0
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (320, 180))
    gray2 = cv2.resize(gray2, (320, 180))
    
    # 直方图
    hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
    hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 模板匹配
    result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    template_sim = np.max(result)
    
    return 0.5 * max(0, hist_sim) + 0.5 * template_sim

def main():
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    sources_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
    
    target_duration = get_video_duration(target)
    print(f"目标视频: {target.name} ({target_duration:.1f}s)")
    print(f"检查时间点: 0s, 30s, 60s, 120s")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 提取目标视频的关键帧
        target_times = [0, 30, 60, 120]
        target_frames = {}
        for t in target_times:
            if t < target_duration:
                frame = extract_frame(target, t, temp_dir)
                if frame:
                    target_frames[t] = frame
        
        print(f"\n目标视频关键帧已提取: {list(target_frames.keys())}")
        
        # 检查每个源视频
        source_videos = list(sources_dir.glob("*.mp4"))
        
        for source in sorted(source_videos):
            source_duration = get_video_duration(source)
            if source_duration < target_duration:
                print(f"\n{source.name}: 时长不足 ({source_duration:.1f}s < {target_duration:.1f}s)")
                continue
            
            print(f"\n{source.name}: ({source_duration:.1f}s)")
            
            # 在不同时间点检查
            best_score = 0
            best_time = 0
            
            for start in [0, 10, 20, 30, 40, 50]:
                scores = []
                for t in target_frames.keys():
                    source_time = start + t
                    if source_time < source_duration:
                        sf = extract_frame(source, source_time, temp_dir)
                        if sf:
                            sim = compare_frames(target_frames[t], sf)
                            scores.append(sim)
                
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_time = start
            
            print(f"   最佳匹配 @ {best_time}s: 相似度 {best_score:.1%}")
            
            if best_score > 0.9:
                print(f"   ✅ 高度匹配！")
            elif best_score > 0.7:
                print(f"   ⚠️ 可能匹配")
            else:
                print(f"   ❌ 不匹配")
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
