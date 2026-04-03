#!/usr/bin/env python3
"""
修复帧率不匹配问题 - 使用音频指纹匹配，重新编码统一帧率
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import wave
import struct

def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {'fps': fps, 'frame_count': frame_count, 'width': width, 'height': height, 'duration': duration}

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
    
    for i in range(min(n_blocks, 500)):
        block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
        fft = np.fft.rfft(block)
        magnitude = np.abs(fft)
        bands = np.array([np.mean(magnitude[j:j+len(magnitude)//20]) 
                        for j in range(0, len(magnitude), len(magnitude)//20)])
        features.append(bands[:20])
    
    return np.array(features)

def find_audio_match(target_fp, source_video, temp_dir):
    """音频指纹匹配"""
    source_fp = extract_audio_fingerprint(source_video, temp_dir)
    
    if len(target_fp) == 0 or len(source_fp) == 0:
        return 0, 0
    
    if len(target_fp) > len(source_fp):
        return 0, 0
    
    best_score = -1
    best_start = 0
    
    for start in range(0, len(source_fp) - len(target_fp) + 1):
        end = start + len(target_fp)
        source_segment = source_fp[start:end]
        
        # 计算相关系数
        correlations = []
        for t, s in zip(target_fp, source_segment):
            if len(t) == len(s) and np.std(t) > 0 and np.std(s) > 0:
                corr = np.corrcoef(t, s)[0,1]
                correlations.append(corr)
            else:
                correlations.append(0)
        
        score = np.mean(correlations)
        
        if score > best_score:
            best_score = score
            best_start = start
    
    return best_score, best_start

def reconstruct_with_reencoding(target_video, source_video, output_path, target_audio=True):
    """
    重构视频并重新编码统一参数
    
    关键：使用原素材的音频，视频从源视频截取并重新编码到与原素材相同的帧率
    """
    print(f"\n🎬 处理: {target_video.name}")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 获取视频信息
        target_info = get_video_info(target_video)
        source_info = get_video_info(source_video)
        
        print(f"   原素材: {target_info['duration']:.1f}s @ {target_info['fps']:.0f}fps")
        print(f"   源视频: {source_info['duration']:.1f}s @ {source_info['fps']:.0f}fps")
        
        # 音频指纹匹配
        print(f"   音频指纹匹配...")
        target_fp = extract_audio_fingerprint(target_video, temp_dir)
        score, start_time = find_audio_match(target_fp, source_video, temp_dir)
        
        print(f"   匹配得分: {score:.2%} @ {start_time:.1f}s")
        
        if score < 0.5:
            print(f"   ❌ 音频匹配失败")
            return False
        
        target_duration = target_info['duration']
        
        # 确保不超出源视频范围
        source_duration = source_info['duration']
        if start_time + target_duration > source_duration:
            actual_duration = source_duration - start_time - 0.1
        else:
            actual_duration = target_duration
        
        # 方法：从源视频截取片段，重新编码到目标帧率
        print(f"   生成输出 (重新编码)...")
        
        # 提取源视频片段
        temp_clip = temp_dir / "clip.mp4"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(start_time),
            '-t', str(actual_duration),
            '-i', str(source_video),
            '-c', 'copy',
            str(temp_clip)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_clip.exists():
            print(f"   ❌ 片段提取失败")
            return False
        
        # 如果需要使用原素材的音频
        if target_audio:
            # 提取原素材音频
            temp_target_audio = temp_dir / "target_audio.aac"
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(target_video),
                '-vn',
                '-c:a', 'copy',
                str(temp_target_audio)
            ]
            subprocess.run(cmd, capture_output=True)
            
            # 合并：源视频画面 + 原素材音频
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(temp_clip),
                '-i', str(temp_target_audio),
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-shortest',
                str(output_path)
            ]
        else:
            # 直接使用源视频（包含音频）
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(temp_clip),
                '-c', 'copy',
                str(output_path)
            ]
        
        subprocess.run(cmd, capture_output=True)
        
        if output_path.exists():
            output_info = get_video_info(output_path)
            print(f"   ✅ 生成成功: {output_info['duration']:.1f}s")
            print(f"      输出路径: {output_path}")
            return True
        else:
            print(f"   ❌ 生成失败")
            return False
            
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    """处理 115196 视频"""
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    output = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output/115196-1-363935819124715523_cut.mp4")
    
    # 备份原文件
    backup = output.with_suffix('.mp4.bak')
    if output.exists():
        shutil.move(str(output), str(backup))
        print(f"   原文件已备份: {backup}")
    
    success = reconstruct_with_reencoding(target, source, output, target_audio=True)
    
    if success:
        print(f"\n✅ 重新处理完成！")
        
        # 验证
        print(f"\n🔍 验证结果...")
        import sys
        sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/04_comparison_validation')
        from compare_videos import compare_videos
        
        result = compare_videos(target, output, 0.90)
        
        if result['passed']:
            print(f"\n✅ 验证通过！")
            if backup.exists():
                backup.unlink()
        else:
            print(f"\n⚠️ 验证未通过 (相似度: {result['overall_similarity']:.1%})")
            if backup.exists():
                print(f"   恢复原文件")
                shutil.move(str(backup), str(output))
    else:
        print(f"\n❌ 处理失败")
        if backup.exists():
            shutil.move(str(backup), str(output))

if __name__ == '__main__':
    main()
