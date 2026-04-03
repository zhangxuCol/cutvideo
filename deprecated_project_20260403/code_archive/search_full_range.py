#!/usr/bin/env python3
"""
在 1.mp4 全范围内搜索 115196 的最佳匹配
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil

def get_video_duration(video_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def extract_frame(video_path, time_sec, output_path):
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', str(time_sec), '-i', str(video_path),
        '-vframes', '1', str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path.exists()

def compare_frames(frame1_path, frame2_path):
    img1 = cv2.imread(str(frame1_path))
    img2 = cv2.imread(str(frame2_path))
    if img1 is None or img2 is None:
        return 0.0
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (320, 180))
    gray2 = cv2.resize(gray2, (320, 180))
    
    hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
    hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    template_sim = np.max(result)
    
    return 0.5 * max(0, hist_sim) + 0.5 * template_sim

def main():
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    
    target_duration = get_video_duration(target)
    source_duration = get_video_duration(source)
    
    print(f"🎬 全范围搜索 115196 在 1.mp4 中的位置")
    print(f"   目标时长: {target_duration:.1f}s")
    print(f"   源视频时长: {source_duration:.1f}s")
    print(f"   搜索范围: 0s ~ {source_duration - target_duration:.1f}s")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 提取目标视频的关键帧（起始、25%、50%、75%）
        target_times = [0, target_duration * 0.25, target_duration * 0.5, target_duration * 0.75]
        target_frames = {}
        for t in target_times:
            frame_path = temp_dir / f"target_{t:.0f}.jpg"
            if extract_frame(target, t, frame_path):
                target_frames[t] = frame_path
        
        print(f"\n目标视频关键帧: {list(target_frames.keys())}")
        print(f"开始搜索（每10秒检查一次）...\n")
        
        best_score = 0
        best_start = 0
        results = []
        
        # 每10秒搜索一次
        for start in range(0, int(source_duration - target_duration), 10):
            scores = []
            for offset, target_frame in target_frames.items():
                source_time = start + offset
                if source_time < source_duration:
                    source_frame = temp_dir / f"source_{start:.0f}_{offset:.0f}.jpg"
                    if extract_frame(source, source_time, source_frame):
                        sim = compare_frames(target_frame, source_frame)
                        scores.append(sim)
            
            if scores:
                avg_score = np.mean(scores)
                results.append((start, avg_score))
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_start = start
                
                if avg_score > 0.8:
                    print(f"   @{start:3d}s: {avg_score:.1%} ⭐")
                elif avg_score > 0.6:
                    print(f"   @{start:3d}s: {avg_score:.1%}")
        
        print(f"\n{'='*50}")
        print(f"📊 搜索结果")
        print(f"{'='*50}")
        print(f"   最佳匹配: @{best_start}s")
        print(f"   相似度: {best_score:.1%}")
        
        if best_score > 0.9:
            print(f"   状态: ✅ 高度匹配")
        elif best_score > 0.8:
            print(f"   状态: ⚠️ 可能匹配")
        else:
            print(f"   状态: ❌ 匹配度较低")
        
        # 显示前5个匹配
        print(f"\n   前5个匹配位置:")
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
        for start, score in sorted_results:
            marker = "⭐" if score > 0.8 else "  "
            print(f"      {marker} @{start:3d}s: {score:.1%}")
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
