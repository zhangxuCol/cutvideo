#!/usr/bin/env python3
"""
对比 @25s 和 @165s 哪个是真正的匹配
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil

def extract_frame(video_path, time_sec, output_path):
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', str(output_path)]
    subprocess.run(cmd, capture_output=True)

def calculate_similarity(frame1_path, frame2_path):
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
    v7_output = Path("/tmp/test_115196.mp4")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        print("=" * 60)
        print("对比不同位置的匹配质量")
        print("=" * 60)
        
        # 测试 @25s vs @165s
        test_positions = [25, 165]
        
        for pos in test_positions:
            print(f"\n测试 @{pos}s:")
            print("-" * 40)
            
            # 提取目标视频的关键帧
            target_times = [0, 30, 60, 120, 180]
            similarities = []
            
            for t in target_times:
                if t < 217:  # 目标视频时长
                    target_frame = temp_dir / f"target_{t}.jpg"
                    source_frame = temp_dir / f"source_{pos}_{t}.jpg"
                    
                    extract_frame(target, t, target_frame)
                    extract_frame(source, pos + t, source_frame)
                    
                    if target_frame.exists() and source_frame.exists():
                        sim = calculate_similarity(target_frame, source_frame)
                        similarities.append(sim)
                        print(f"  @{t}s vs @{pos+t}s: {sim:.1%}")
            
            avg_sim = np.mean(similarities) if similarities else 0
            print(f"  平均相似度: {avg_sim:.1%}")
        
        # 对比 V7 输出
        print(f"\n\n对比 V7 输出 (@25s):")
        print("-" * 40)
        v7_similarities = []
        for t in target_times:
            if t < 217:
                target_frame = temp_dir / f"target_{t}.jpg"
                v7_frame = temp_dir / f"v7_{t}.jpg"
                
                extract_frame(target, t, target_frame)
                extract_frame(v7_output, t, v7_frame)
                
                if target_frame.exists() and v7_frame.exists():
                    sim = calculate_similarity(target_frame, v7_frame)
                    v7_similarities.append(sim)
                    print(f"  @{t}s: {sim:.1%}")
        
        v7_avg = np.mean(v7_similarities) if v7_similarities else 0
        print(f"  V7 输出平均相似度: {v7_avg:.1%}")
        
        print(f"\n{'='*60}")
        print("结论:")
        if v7_avg > 0.85:
            print(f"  ✅ V7 输出 (@25s) 是正确的，平均相似度 {v7_avg:.1%}")
        elif v7_avg > 0.7:
            print(f"  ⚠️  V7 输出 (@25s) 可能正确，但相似度只有 {v7_avg:.1%}")
        else:
            print(f"  ❌ V7 输出 (@25s) 很可能是错误的，相似度只有 {v7_avg:.1%}")
            
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
