#!/usr/bin/env python3
"""
改进的视频重构脚本 - 更精确的片段查找算法
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

# 相似度阈值
HIGH_THRESHOLD = 0.98      # 高置信度匹配
MEDIUM_THRESHOLD = 0.95    # 中等置信度匹配
MIN_THRESHOLD = 0.90       # 最低接受阈值

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

def extract_segment_frames(video_path, start_time, end_time, num_frames=5):
    """提取指定时间段的帧"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    
    if total_frames <= 0:
        cap.release()
        return []
    
    # 在时间段内均匀采样
    indices = [start_frame + int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 缩小以加速处理
            frame_small = cv2.resize(frame, (320, 180))
            frames.append((idx / fps, frame_small))
    
    cap.release()
    return frames

def find_segment_in_source(segment_frames, source_video_path, search_window=None):
    """
    在源视频中查找与片段最匹配的位置
    
    segment_frames: 片段的帧列表 [(time, frame), ...]
    source_video_path: 源视频路径
    search_window: 搜索时间窗口 (start, end)，None表示全视频搜索
    
    返回: (最佳开始时间, 最佳结束时间, 平均相似度, 匹配帧数/总帧数)
    """
    cap = cv2.VideoCapture(str(source_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 确定搜索范围
    if search_window:
        search_start = int(search_window[0] * fps)
        search_end = int(search_window[1] * fps)
    else:
        search_start = 0
        search_end = total_frames
    
    # 确保搜索范围有效
    search_start = max(0, search_start)
    search_end = min(total_frames, search_end)
    
    best_match = {
        'start_time': 0,
        'end_time': 0,
        'avg_sim': 0,
        'matched_frames': 0
    }
    
    # 使用滑动窗口，步长为1秒
    window_size = len(segment_frames)
    step_frames = int(fps)  # 每秒移动一次
    
    for window_start in range(search_start, search_end - window_size * 3, step_frames):
        similarities = []
        
        for i, (seg_time, seg_frame) in enumerate(segment_frames):
            # 在源视频中对应位置读取帧
            source_frame_idx = window_start + int(i * fps)  # 假设每秒一个采样点
            
            if source_frame_idx >= total_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
            ret, source_frame = cap.read()
            
            if not ret:
                break
            
            # 缩小并计算相似度
            source_frame_small = cv2.resize(source_frame, (320, 180))
            diff = np.abs(seg_frame.astype(float) - source_frame_small.astype(float))
            sim = 1 - (np.mean(diff) / 255)
            similarities.append(sim)
        
        if len(similarities) == len(segment_frames):
            avg_sim = np.mean(similarities)
            matched_frames = sum(1 for s in similarities if s >= MEDIUM_THRESHOLD)
            
            # 更新最佳匹配
            if avg_sim > best_match['avg_sim']:
                best_match = {
                    'start_time': window_start / fps,
                    'end_time': (window_start + len(segment_frames) * fps) / fps,
                    'avg_sim': avg_sim,
                    'matched_frames': matched_frames,
                    'similarities': similarities
                }
    
    cap.release()
    
    return (best_match['start_time'], best_match['end_time'], 
            best_match['avg_sim'], best_match['matched_frames'])

def find_all_segments(clip1_path, source_videos):
    """
    分析clip1，在源视频中查找所有片段
    使用改进的算法确保准确性
    """
    print("="*60)
    print("🔍 改进的片段查找算法")
    print("="*60)
    
    # 获取clip1信息
    clip1_info = get_video_info(clip1_path)
    print(f"\nclip1信息:")
    print(f"  时长: {clip1_info['duration']:.2f}秒")
    print(f"  分辨率: {clip1_info['width']}x{clip1_info['height']}")
    print(f"  帧数: {clip1_info['frame_count']}")
    
    # 将clip1分成3个片段（根据时长估算）
    segment_duration = clip1_info['duration'] / 3
    print(f"\n预计每个片段时长: {segment_duration:.2f}秒")
    
    segments_time = [
        (0, segment_duration),                           # 片段1
        (segment_duration, 2 * segment_duration),        # 片段2
        (2 * segment_duration, 3 * segment_duration)     # 片段3
    ]
    
    results = []
    
    for seg_idx, (start_time, end_time) in enumerate(segments_time):
        print(f"\n{'='*60}")
        print(f"查找第{seg_idx+1}个片段 ({start_time:.2f}s - {end_time:.2f}s)")
        print(f"{'='*60}")
        
        # 提取片段的帧
        segment_frames = extract_segment_frames(clip1_path, start_time, end_time, num_frames=5)
        print(f"  提取了 {len(segment_frames)} 个采样帧")
        
        # 在每个源视频中查找
        best_source = None
        best_start = 0
        best_end = 0
        best_sim = 0
        best_matched = 0
        
        for source_path in source_videos:
            print(f"\n  搜索 {source_path.name}...")
            
            # 在全视频中搜索（也可以限制搜索窗口）
            s_start, s_end, s_sim, s_matched = find_segment_in_source(
                segment_frames, source_path, search_window=None
            )
            
            print(f"    最佳匹配: {s_start:.2f}s - {s_end:.2f}s")
            print(f"    平均相似度: {s_sim:.2%}")
            print(f"    高相似度帧: {s_matched}/{len(segment_frames)}")
            
            # 选择最佳匹配
            if s_sim > best_sim and s_sim >= MIN_THRESHOLD:
                best_sim = s_sim
                best_start = s_start
                best_end = s_end
                best_source = source_path
                best_matched = s_matched
        
        if best_source:
            print(f"\n  ✅ 选择: {best_source.name}")
            print(f"     时间: {best_start:.2f}s - {best_end:.2f}s")
            print(f"     相似度: {best_sim:.2%}")
            results.append((best_source, best_start, best_end, best_sim, best_matched))
        else:
            print(f"\n  ❌ 未找到足够匹配的源，使用默认")
            # 使用默认
            default_source = source_videos[seg_idx % len(source_videos)]
            default_start = seg_idx * 15  # 0, 15, 30
            default_end = default_start + 15
            results.append((default_source, default_start, default_end, 0.0, 0))
    
    return results

def create_clip2_from_segments(segments_info):
    """根据查找结果创建clip2"""
    print("\n" + "="*60)
    print("🎬 创建裁剪视频2")
    print("="*60)
    
    output = OUTPUT_DIR / "clip2_reconstructed_real_v2.mp4"
    temp_dir = OUTPUT_DIR / "temp_reconstruct_v2"
    temp_dir.mkdir(exist_ok=True)
    
    for idx, (source_path, start_time, end_time, sim, matched) in enumerate(segments_info):
        print(f"\n{idx+1}. 从 {source_path.name}")
        print(f"   裁剪 {start_time:.2f}s - {end_time:.2f}s")
        
        seg_file = temp_dir / f"seg{idx+1}.mp4"
        duration = end_time - start_time
        
        subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(source_path), '-ss', str(start_time), '-t', str(duration),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', str(seg_file)
        ], check=True)
    
    # 合并
    print("\n合并片段...")
    concat_list = temp_dir / "concat_list.txt"
    with open(concat_list, 'w') as f:
        for idx in range(len(segments_info)):
            f.write(f"file 'seg{idx+1}.mp4'\n")
    
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'concat', '-safe', '0', '-i', str(concat_list),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(output)
    ], check=True)
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ 创建完成: {output}")
    return output

def main():
    print("\n" + "="*60)
    print("🎬 改进的视频重构工具")
    print(f"   阈值: 高{HIGH_THRESHOLD:.0%} / 中{MEDIUM_THRESHOLD:.0%} / 低{MIN_THRESHOLD:.0%}")
    print("="*60 + "\n")
    
    source_videos = [
        SOURCE_DIR / "video1_天赋变异后我无敌了_ep1.mp4",
        SOURCE_DIR / "video2_开局饕餮血统我吞噬一切_ep1.mp4",
        SOURCE_DIR / "video3_咒术反噬我有无限血条_ep1.mp4"
    ]
    
    print("源视频:")
    for sv in source_videos:
        info = get_video_info(sv)
        print(f"  - {sv.name} ({info['duration']:.0f}s)")
    
    # 查找片段
    segments_info = find_all_segments(CLIP1_PATH, source_videos)
    
    # 创建clip2
    clip2_path = create_clip2_from_segments(segments_info)
    
    # 显示结果
    print("\n" + "="*60)
    print("📊 重构结果汇总")
    print("="*60)
    for idx, (source, start, end, sim, matched) in enumerate(segments_info):
        print(f"\n片段{idx+1}:")
        print(f"  来源: {source.name}")
        print(f"  时间: {start:.2f}s - {end:.2f}s")
        print(f"  相似度: {sim:.2%}")
        print(f"  匹配帧: {matched}/5")
    
    print(f"\n✅ clip2: {clip2_path}")
    
    # 验证
    print("\n" + "="*60)
    print("🔍 验证clip1和clip2...")
    print("="*60)
    
    info1 = get_video_info(CLIP1_PATH)
    info2 = get_video_info(clip2_path)
    
    print(f"\nclip1: {info1['duration']:.2f}s, {info1['width']}x{info1['height']}")
    print(f"clip2: {info2['duration']:.2f}s, {info2['width']}x{info2['height']}")
    
    duration_diff = abs(info1['duration'] - info2['duration'])
    print(f"\n时长差异: {duration_diff:.2f}s")
    
    if duration_diff < 1.0:
        print("✅ 时长匹配")
    else:
        print("⚠️ 时长差异较大")

if __name__ == "__main__":
    main()
