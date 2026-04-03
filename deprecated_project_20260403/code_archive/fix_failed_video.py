#!/usr/bin/env python3
"""
重新处理失败视频的脚本 - 强制视频帧验证
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
    """获取视频时长"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def extract_audio_fingerprint(video_path, temp_dir):
    """提取音频指纹"""
    temp_wav = temp_dir / f"audio.wav"
    
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '8000',
        '-ac', '1',
        str(temp_wav)
    ]
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
    
    for i in range(min(n_blocks, 300)):
        block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
        fft = np.fft.rfft(block)
        magnitude = np.abs(fft)
        bands = np.array([np.mean(magnitude[j:j+len(magnitude)//20]) 
                        for j in range(0, len(magnitude), len(magnitude)//20)])
        features.append(bands[:20])
    
    return np.array(features)

def extract_keyframe_at(video_path, timestamp, temp_dir):
    """提取指定时间点的关键帧"""
    frame_path = temp_dir / f"frame_{timestamp:.2f}.jpg"
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', str(timestamp),
        '-i', str(video_path),
        '-vframes', '1',
        '-q:v', '2',
        str(frame_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return frame_path if frame_path.exists() else None

def calculate_frame_similarity(frame1_path, frame2_path):
    """计算两帧相似度"""
    img1 = cv2.imread(str(frame1_path))
    img2 = cv2.imread(str(frame2_path))
    
    if img1 is None or img2 is None:
        return 0.0
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 调整大小一致
    gray1 = cv2.resize(gray1, (320, 180))
    gray2 = cv2.resize(gray2, (320, 180))
    
    # 直方图相似度
    hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
    hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 模板匹配
    result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    template_sim = np.max(result)
    
    return 0.5 * max(0, hist_sim) + 0.5 * template_sim

def find_best_match_with_video_verify(target_video, source_video, temp_dir):
    """结合音频匹配和视频帧验证找到最佳匹配"""
    
    print(f"   提取音频指纹...")
    target_fp = extract_audio_fingerprint(target_video, temp_dir)
    source_fp = extract_audio_fingerprint(source_video, temp_dir)
    
    target_duration = get_video_duration(target_video)
    source_duration = get_video_duration(source_video)
    
    print(f"   目标时长: {target_duration:.1f}s, 源视频时长: {source_duration:.1f}s")
    
    if len(target_fp) == 0 or len(source_fp) == 0:
        return None, 0
    
    # 提取目标视频的关键帧（用于验证）
    print(f"   提取目标视频验证帧...")
    verify_times = [0, target_duration * 0.25, target_duration * 0.5, target_duration * 0.75, target_duration - 1]
    target_frames = {}
    for t in verify_times:
        if t < target_duration:
            frame_path = extract_keyframe_at(target_video, t, temp_dir)
            if frame_path:
                target_frames[t] = frame_path
    
    print(f"   滑动窗口搜索 + 视频帧验证...")
    best_result = None
    best_combined_score = 0
    
    # 滑动窗口搜索
    for start in range(0, len(source_fp) - len(target_fp) + 1, 1):
        end = start + len(target_fp)
        source_segment = source_fp[start:end]
        
        # 音频相似度
        audio_score = np.mean([np.corrcoef(t, s)[0,1] if len(t) == len(s) and np.std(t) > 0 and np.std(s) > 0 else 0 
                               for t, s in zip(target_fp, source_segment)])
        
        if audio_score < 0.5:  # 跳过低音频匹配
            continue
        
        # 视频帧验证
        video_scores = []
        for target_time, target_frame_path in target_frames.items():
            source_time = start + target_time
            if source_time < source_duration:
                source_frame_path = extract_keyframe_at(source_video, source_time, temp_dir)
                if source_frame_path:
                    sim = calculate_frame_similarity(target_frame_path, source_frame_path)
                    video_scores.append(sim)
        
        video_score = np.mean(video_scores) if video_scores else 0
        
        # 综合评分
        combined_score = 0.4 * audio_score + 0.6 * video_score
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_result = {
                'start': start,
                'audio_score': audio_score,
                'video_score': video_score,
                'combined_score': combined_score
            }
    
    return best_result, best_combined_score

def reconstruct_video(target_video, source_video, output_path):
    """重新重构视频"""
    print(f"\n🎬 重新处理: {target_video.name}")
    print(f"   源视频: {source_video.name}")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 找到最佳匹配
        best_match, score = find_best_match_with_video_verify(target_video, source_video, temp_dir)
        
        if not best_match or score < 0.6:
            print(f"   ❌ 未找到良好匹配")
            return False
        
        start_time = best_match['start']
        target_duration = get_video_duration(target_video)
        
        print(f"   ✅ 找到匹配 @ {start_time:.1f}s")
        print(f"      音频相似度: {best_match['audio_score']:.2%}")
        print(f"      视频相似度: {best_match['video_score']:.2%}")
        print(f"      综合评分: {best_match['combined_score']:.2%}")
        
        # 生成输出
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(start_time),
            '-t', str(target_duration),
            '-i', str(source_video),
            '-c', 'copy',
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        if output_path.exists():
            print(f"   ✅ 生成成功: {output_path}")
            return True
        else:
            print(f"   ❌ 生成失败")
            return False
            
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    """处理失败视频"""
    target_video = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source_video = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    output_path = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output/115196-1-363935819124715523_cut_fixed.mp4")
    
    success = reconstruct_video(target_video, source_video, output_path)
    
    if success:
        print(f"\n✅ 处理完成！")
        print(f"   输出: {output_path}")
        
        # 验证结果
        print(f"\n🔍 验证新输出...")
        import sys
        sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/04_comparison_validation')
        from compare_videos import compare_videos
        
        result = compare_videos(target_video, output_path, 0.90)
        
        if result['passed']:
            print(f"\n✅ 验证通过！用新文件替换旧文件")
            # 替换原文件
            import shutil as sh
            old_file = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output/115196-1-363935819124715523_cut.mp4")
            sh.move(str(output_path), str(old_file))
            print(f"   已替换: {old_file}")
    else:
        print(f"\n❌ 处理失败")

if __name__ == '__main__':
    main()
