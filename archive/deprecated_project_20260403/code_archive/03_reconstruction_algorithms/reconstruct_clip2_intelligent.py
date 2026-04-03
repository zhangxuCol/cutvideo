#!/usr/bin/env python3
"""
真正的视频重构脚本 - 根据clip1的内容在源视频中查找匹配
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import json

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

def extract_keyframes(video_path, num_frames=10):
    """提取关键帧"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []
    
    # 均匀采样
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 缩小以加速处理
            frame_small = cv2.resize(frame, (320, 180))
            frames.append((idx, frame_small))
    
    cap.release()
    return frames

def find_best_match(target_frame, source_video_path):
    """
    在源视频中查找与目标帧最匹配的帧
    返回: (最佳匹配时间戳, 相似度)
    """
    cap = cv2.VideoCapture(str(source_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    best_sim = 0
    best_time = 0
    
    # 每5帧采样一次（加速）
    sample_interval = 5
    
    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 缩小以匹配目标帧尺寸
        frame_small = cv2.resize(frame, (320, 180))
        
        # 计算相似度
        diff = np.abs(target_frame.astype(float) - frame_small.astype(float))
        sim = 1 - (np.mean(diff) / 255)
        
        if sim > best_sim:
            best_sim = sim
            best_time = frame_idx / fps
    
    cap.release()
    return best_time, best_sim

def find_segments_in_source(clip1_path, source_videos):
    """
    分析clip1，在源视频中查找每个片段的来源
    返回: [(源视频路径, 开始时间, 结束时间, 相似度), ...]
    """
    print("="*60)
    print("🔍 分析clip1并在源视频中查找匹配...")
    print("="*60)
    
    # 提取clip1的关键帧
    print("\n1. 提取clip1的关键帧...")
    clip1_frames = extract_keyframes(clip1_path, 12)  # 提取12个关键帧
    print(f"   提取了 {len(clip1_frames)} 个关键帧")
    
    # 将关键帧分成3组（对应3个片段）
    segment_size = len(clip1_frames) // 3
    segments = [
        clip1_frames[0:segment_size],                    # 第1个片段
        clip1_frames[segment_size:2*segment_size],       # 第2个片段
        clip1_frames[2*segment_size:]                    # 第3个片段
    ]
    
    results = []
    
    for seg_idx, segment_frames in enumerate(segments):
        print(f"\n2.{seg_idx+1} 查找第{seg_idx+1}个片段的来源...")
        
        # 取片段的中间帧作为代表
        mid_frame_idx = len(segment_frames) // 2
        target_frame = segment_frames[mid_frame_idx][1]
        
        # 在每个源视频中查找最佳匹配
        best_source = None
        best_time = 0
        best_sim = 0
        
        for source_path in source_videos:
            time, sim = find_best_match(target_frame, source_path)
            print(f"   在 {source_path.name}: 时间={time:.2f}s, 相似度={sim:.2%}")
            
            if sim > best_sim:
                best_sim = sim
                best_time = time
                best_source = source_path
        
        # 计算片段时长（根据关键帧数估算）
        seg_duration = 15  # 每个片段约15秒
        
        print(f"   ✅ 最佳匹配: {best_source.name}")
        print(f"      时间: {best_time:.2f}s - {best_time + seg_duration:.2f}s")
        print(f"      相似度: {best_sim:.2%}")
        
        results.append((best_source, best_time, best_time + seg_duration, best_sim))
    
    return results

def create_clip2_from_segments(segments_info):
    """
    根据查找结果从源视频裁剪片段并合并
    """
    print("\n" + "="*60)
    print("🎬 创建裁剪视频2...")
    print("="*60)
    
    output = OUTPUT_DIR / "clip2_reconstructed_real.mp4"
    temp_dir = OUTPUT_DIR / "temp_reconstruct"
    temp_dir.mkdir(exist_ok=True)
    
    segment_files = []
    
    for idx, (source_path, start_time, end_time, sim) in enumerate(segments_info):
        print(f"\n{idx+1}. 从 {source_path.name} 裁剪 {start_time:.2f}s - {end_time:.2f}s...")
        
        seg_file = temp_dir / f"seg{idx+1}.mp4"
        duration = end_time - start_time
        
        subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(source_path), '-ss', str(start_time), '-t', str(duration),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', str(seg_file)
        ], check=True)
        
        segment_files.append(seg_file)
    
    # 合并片段
    print("\n4. 合并片段...")
    concat_list = temp_dir / "concat_list.txt"
    with open(concat_list, 'w') as f:
        for seg_file in segment_files:
            f.write(f"file '{seg_file}'\n")
    
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'concat', '-safe', '0', '-i', str(concat_list),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(output)
    ], check=True)
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ 裁剪视频2创建完成: {output}")
    return output

def main():
    print("\n" + "="*60)
    print("🎬 智能视频重构工具")
    print("   根据clip1内容在源视频中查找匹配")
    print("="*60 + "\n")
    
    # 源视频列表
    source_videos = [
        SOURCE_DIR / "video1_天赋变异后我无敌了_ep1.mp4",
        SOURCE_DIR / "video2_开局饕餮血统我吞噬一切_ep1.mp4",
        SOURCE_DIR / "video3_咒术反噬我有无限血条_ep1.mp4"
    ]
    
    print("源视频:")
    for sv in source_videos:
        print(f"  - {sv.name}")
    
    print(f"\n目标视频 (clip1): {CLIP1_PATH.name}")
    
    # 步骤1: 在源视频中查找clip1的每个片段
    segments_info = find_segments_in_source(CLIP1_PATH, source_videos)
    
    # 步骤2: 根据查找结果创建clip2
    clip2_path = create_clip2_from_segments(segments_info)
    
    # 显示结果
    print("\n" + "="*60)
    print("📊 重构结果")
    print("="*60)
    for idx, (source, start, end, sim) in enumerate(segments_info):
        print(f"\n片段{idx+1}:")
        print(f"  来源: {source.name}")
        print(f"  时间: {start:.2f}s - {end:.2f}s")
        print(f"  匹配相似度: {sim:.2%}")
    
    print(f"\n✅ 裁剪视频2: {clip2_path}")

if __name__ == "__main__":
    main()
