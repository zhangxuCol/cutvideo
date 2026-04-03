#!/usr/bin/env python3
"""
验证 115196 的正确匹配位置 - 音频+视频双重验证
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

def extract_audio_fingerprint(video_path, temp_dir):
    temp_wav = temp_dir / f"audio.wav"
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '8000', '-ac', '1',
           str(temp_wav)]
    subprocess.run(cmd, capture_output=True)
    
    if not temp_wav.exists():
        return np.array([])
    
    with wave.open(str(temp_wav), 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        samples = struct.unpack(f'{n_frames}h', audio_data)
    
    samples_per_sec = 8000
    n_blocks = len(samples) // samples_per_sec
    features = []
    
    for i in range(min(n_blocks, 500)):
        block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
        fft = np.fft.rfft(block)
        magnitude = np.abs(fft)
        bands = np.array([np.mean(magnitude[j:j+len(magnitude)//20]) 
                        for j in range(0, len(magnitude), len(magnitude)//20)])
        features.append(bands[:20])
    
    return np.array(features)

def extract_frame(video_path, time_sec, output_path):
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', str(output_path)]
    subprocess.run(cmd, capture_output=True)

def calculate_frame_similarity(frame1_path, frame2_path):
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
    
    print(f"🎬 音频+视频双重验证 115196")
    print(f"   目标: {target_duration:.1f}s")
    print(f"   源: {source_duration:.1f}s\n")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 提取音频指纹
        print("提取音频指纹...")
        target_fp = extract_audio_fingerprint(target, temp_dir)
        source_fp = extract_audio_fingerprint(source, temp_dir)
        
        print(f"   目标音频: {len(target_fp)}s")
        print(f"   源音频: {len(source_fp)}s\n")
        
        # 提取目标视频的验证帧
        print("提取目标视频验证帧...")
        verify_times = [0, 30, 60, 120, 180]
        target_frames = {}
        for t in verify_times:
            if t < target_duration:
                frame_path = temp_dir / f"target_{t:.0f}.jpg"
                extract_frame(target, t, frame_path)
                if frame_path.exists():
                    target_frames[t] = frame_path
                    print(f"   @{t}s: OK")
        
        # 滑动窗口搜索 + 视频验证
        print(f"\n滑动窗口搜索 + 视频验证 (每5秒)...")
        best_result = None
        best_combined = 0
        
        for start in range(0, len(source_fp) - len(target_fp) + 1, 5):
            # 音频相似度
            end = start + len(target_fp)
            source_segment = source_fp[start:end]
            correlations = []
            for t, s in zip(target_fp, source_segment):
                if len(t) == len(s) and np.std(t) > 0 and np.std(s) > 0:
                    corr = np.corrcoef(t, s)[0,1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
            audio_score = np.mean(correlations)
            
            if audio_score < 0.5:  # 跳过低音频匹配
                continue
            
            # 视频验证
            video_scores = []
            for target_time, target_frame in target_frames.items():
                source_time = start + target_time
                if source_time < source_duration:
                    source_frame = temp_dir / f"source_{start}_{target_time:.0f}.jpg"
                    extract_frame(source, source_time, source_frame)
                    if source_frame.exists():
                        sim = calculate_frame_similarity(target_frame, source_frame)
                        video_scores.append(sim)
            
            video_score = np.mean(video_scores) if video_scores else 0
            combined = 0.4 * audio_score + 0.6 * video_score
            
            if combined > best_combined:
                best_combined = combined
                best_result = {
                    'start': start,
                    'audio': audio_score,
                    'video': video_score,
                    'combined': combined
                }
            
            if combined > 0.8:
                print(f"   @{start:3d}s: 音频{audio_score:.1%} 视频{video_score:.1%} 综合{combined:.1%} ⭐")
            elif combined > 0.6:
                print(f"   @{start:3d}s: 音频{audio_score:.1%} 视频{video_score:.1%} 综合{combined:.1%}")
        
        print(f"\n{'='*50}")
        print(f"📊 最佳匹配结果")
        print(f"{'='*50}")
        if best_result:
            print(f"   位置: @{best_result['start']}s")
            print(f"   音频相似度: {best_result['audio']:.1%}")
            print(f"   视频相似度: {best_result['video']:.1%}")
            print(f"   综合评分: {best_result['combined']:.1%}")
            
            if best_result['combined'] > 0.85:
                print(f"   状态: ✅ 高度匹配")
            elif best_result['combined'] > 0.7:
                print(f"   状态: ⚠️ 可能匹配")
            else:
                print(f"   状态: ❌ 匹配度低")
        else:
            print(f"   未找到匹配")
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
