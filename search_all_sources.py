#!/usr/bin/env python3
"""
在所有源视频中搜索 115196 的最佳匹配
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

def search_in_video(target, source, temp_dir):
    """在单个源视频中搜索"""
    target_duration = get_video_duration(target)
    source_duration = get_video_duration(source)
    
    if source_duration < target_duration:
        return None
    
    # 提取目标视频的关键帧
    target_times = [0, target_duration * 0.25, target_duration * 0.5, target_duration * 0.75]
    target_frames = {}
    for t in target_times:
        if t < target_duration:
            frame_path = temp_dir / f"target_{t:.0f}.jpg"
            if extract_frame(target, t, frame_path):
                target_frames[t] = frame_path
    
    best_score = 0
    best_start = 0
    
    # 每10秒搜索一次
    for start in range(0, int(source_duration - target_duration), 10):
        scores = []
        for offset, target_frame in target_frames.items():
            source_time = start + offset
            if source_time < source_duration:
                source_frame = temp_dir / f"source_{source.stem}_{start}_{offset:.0f}.jpg"
                if extract_frame(source, source_time, source_frame):
                    sim = compare_frames(target_frame, source_frame)
                    scores.append(sim)
        
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_start = start
    
    return {'source': source.name, 'start': best_start, 'score': best_score}

def main():
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    sources_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
    
    target_duration = get_video_duration(target)
    
    print(f"🎬 在所有源视频中搜索 115196")
    print(f"   目标时长: {target_duration:.1f}s")
    print(f"   源视频目录: {sources_dir}\n")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        results = []
        
        for source in sorted(sources_dir.glob("*.mp4")):
            source_duration = get_video_duration(source)
            print(f"📹 检查 {source.name} ({source_duration:.1f}s)...")
            
            result = search_in_video(target, source, temp_dir)
            if result:
                results.append(result)
                print(f"   最佳匹配: @{result['start']}s, 相似度 {result['score']:.1%}")
            else:
                print(f"   时长不足，跳过")
        
        # 排序结果
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"📊 搜索结果汇总")
        print(f"{'='*60}")
        
        for i, r in enumerate(results, 1):
            marker = "⭐" if r['score'] > 0.85 else "⚠️" if r['score'] > 0.7 else "❌"
            print(f"   {i}. {r['source']}: @{r['start']}s, {r['score']:.1%} {marker}")
        
        if results and results[0]['score'] > 0.85:
            print(f"\n✅ 找到高度匹配！")
            print(f"   推荐: {results[0]['source']} @{results[0]['start']}s")
        elif results and results[0]['score'] > 0.7:
            print(f"\n⚠️ 找到可能匹配")
            print(f"   推荐: {results[0]['source']} @{results[0]['start']}s")
        else:
            print(f"\n❌ 未找到良好匹配")
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
