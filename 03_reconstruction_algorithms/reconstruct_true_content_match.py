#!/usr/bin/env python3
"""
真正的逐帧内容查找重构脚本
根据clip1的每一帧在源视频中精确查找匹配位置
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
from collections import deque

# 配置
SOURCE_DIR = Path("/Users/zhangxu/.openclaw/workspace/test_videos/source_videos")
OUTPUT_DIR = Path("/Users/zhangxu/.openclaw/workspace/test_videos")
CLIP1_PATH = OUTPUT_DIR / "clip1_real.mp4"

# 匹配阈值
MATCH_THRESHOLD = 0.95  # 单帧匹配阈值
CONTINUITY_THRESHOLD = 0.90  # 连续性阈值（连续N帧都超过此值才认为是真正匹配）
MIN_CONSECUTIVE_MATCHES = 10  # 最少连续匹配帧数

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

def extract_all_frames(video_path, resize=(320, 180)):
    """提取视频的所有帧（缩小尺寸以加速）"""
    print(f"  提取所有帧: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 缩小尺寸
        frame_small = cv2.resize(frame, resize)
        frames.append(frame_small)
    
    cap.release()
    print(f"    共 {len(frames)} 帧")
    return frames

def calculate_frame_similarity(frame1, frame2):
    """计算两帧的相似度"""
    diff = np.abs(frame1.astype(float) - frame2.astype(float))
    return 1 - (np.mean(diff) / 255)

def find_best_match_with_continuity(target_frames, source_frames, source_name):
    """
    在源视频中查找与目标片段最匹配的位置（使用连续性验证）
    
    返回: (开始帧索引, 平均相似度, 连续匹配帧数)
    """
    print(f"\n  在 {source_name} 中查找...")
    
    best_match = {
        'start_idx': -1,
        'avg_sim': 0,
        'consecutive_matches': 0
    }
    
    target_len = len(target_frames)
    source_len = len(source_frames)
    
    # 滑动窗口，步长为1帧（最精确）
    for start_idx in range(source_len - target_len + 1):
        similarities = []
        consecutive_count = 0
        
        for i in range(target_len):
            sim = calculate_frame_similarity(target_frames[i], source_frames[start_idx + i])
            similarities.append(sim)
            
            # 检查连续性
            if sim >= CONTINUITY_THRESHOLD:
                consecutive_count += 1
            else:
                # 如果不匹配，中断连续性检查
                if i < MIN_CONSECUTIVE_MATCHES:
                    consecutive_count = 0
                    break
        
        # 如果连续匹配帧数足够，计算平均相似度
        if consecutive_count >= MIN_CONSECUTIVE_MATCHES:
            avg_sim = np.mean(similarities)
            
            if avg_sim > best_match['avg_sim']:
                best_match = {
                    'start_idx': start_idx,
                    'avg_sim': avg_sim,
                    'consecutive_matches': consecutive_count
                }
        
        # 每1000帧输出一次进度
        if start_idx > 0 and start_idx % 1000 == 0:
            print(f"    已检查 {start_idx}/{source_len - target_len} 帧...")
    
    return best_match['start_idx'], best_match['avg_sim'], best_match['consecutive_matches']

def find_segment_boundaries(clip1_frames, source_videos_info):
    """
    分析clip1的所有帧，在源视频中查找片段边界
    
    策略：
    1. 先在每个源视频中找到最佳匹配位置
    2. 根据匹配质量确定片段边界
    """
    print("="*60)
    print("🔍 逐帧内容查找（使用连续性验证）")
    print("="*60)
    
    clip1_len = len(clip1_frames)
    print(f"\nclip1 总帧数: {clip1_len}")
    
    # 假设有3个片段，每个片段约1/3
    # 但我们需要自动检测边界
    
    # 方法：滑动窗口，在每个源视频中查找最佳匹配
    # 片段1：从clip1的第0帧开始
    # 片段2：从clip1的约1/3处开始
    # 片段3：从clip1的约2/3处开始
    
    segment_starts = [0, clip1_len // 3, 2 * clip1_len // 3]
    segment_names = ['片段1', '片段2', '片段3']
    
    results = []
    
    for seg_idx, (seg_name, start_frame) in enumerate(zip(segment_names, segment_starts)):
        print(f"\n{'='*60}")
        print(f"查找 {seg_name} (从clip1第{start_frame}帧开始)")
        print(f"{'='*60}")
        
        # 提取这个片段的帧（假设每个片段约1/3）
        if seg_idx < 2:
            end_frame = segment_starts[seg_idx + 1]
        else:
            end_frame = clip1_len
        
        segment_frames = clip1_frames[start_frame:end_frame]
        print(f"  片段帧数: {len(segment_frames)}")
        
        # 在每个源视频中查找
        best_source_idx = -1
        best_start_idx = -1
        best_sim = 0
        best_consecutive = 0
        
        for idx, (source_name, source_frames) in enumerate(source_videos_info):
            start_idx, avg_sim, consecutive = find_best_match_with_continuity(
                segment_frames, source_frames, source_name
            )
            
            if start_idx >= 0:
                print(f"    匹配: 帧{start_idx}, 相似度{avg_sim:.2%}, 连续{consecutive}帧")
                
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_start_idx = start_idx
                    best_source_idx = idx
                    best_consecutive = consecutive
            else:
                print(f"    未找到足够连续的匹配")
        
        if best_source_idx >= 0:
            source_name, source_frames = source_videos_info[best_source_idx]
            fps = 30  # 假设30fps
            start_time = best_start_idx / fps
            end_time = (best_start_idx + len(segment_frames)) / fps
            
            print(f"\n  ✅ 最佳匹配: {source_name}")
            print(f"     帧范围: {best_start_idx} - {best_start_idx + len(segment_frames)}")
            print(f"     时间: {start_time:.2f}s - {end_time:.2f}s")
            print(f"     相似度: {best_sim:.2%}")
            print(f"     连续匹配: {best_consecutive}帧")
            
            results.append((
                source_videos_info[best_source_idx][0],  # source_name
                source_videos_info[best_source_idx][1],  # source_frames
                start_time,
                end_time,
                best_sim,
                best_consecutive
            ))
        else:
            print(f"\n  ❌ 未找到匹配，使用默认")
            # 使用默认
            default_idx = seg_idx % len(source_videos_info)
            default_source_name, default_source_frames = source_videos_info[default_idx]
            default_start = seg_idx * 15
            default_end = default_start + 15
            results.append((
                default_source_name,
                default_source_frames,
                default_start,
                default_end,
                0.0,
                0
            ))
    
    return results

def create_clip2_from_results(results, output_path):
    """根据查找结果创建clip2"""
    print("\n" + "="*60)
    print("🎬 创建裁剪视频2")
    print("="*60)
    
    temp_dir = OUTPUT_DIR / "temp_final"
    temp_dir.mkdir(exist_ok=True)
    
    # 获取源视频路径
    source_paths = [
        SOURCE_DIR / "video1_天赋变异后我无敌了_ep1.mp4",
        SOURCE_DIR / "video2_开局饕餮血统我吞噬一切_ep1.mp4",
        SOURCE_DIR / "video3_咒术反噬我有无限血条_ep1.mp4"
    ]
    
    for idx, (source_name, _, start_time, end_time, sim, consecutive) in enumerate(results):
        # 找到对应的源视频路径
        source_path = None
        for sp in source_paths:
            if sp.name == source_name:
                source_path = sp
                break
        
        if not source_path:
            source_path = source_paths[idx % len(source_paths)]
        
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
        for idx in range(len(results)):
            f.write(f"file 'seg{idx+1}.mp4'\n")
    
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'concat', '-safe', '0', '-i', str(concat_list),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(output_path)
    ], check=True)
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ 创建完成: {output_path}")

def main():
    print("\n" + "="*60)
    print("🎬 真正的逐帧内容查找重构工具")
    print(f"   匹配阈值: {MATCH_THRESHOLD:.0%}")
    print(f"   连续性阈值: {CONTINUITY_THRESHOLD:.0%}")
    print(f"   最少连续匹配: {MIN_CONSECUTIVE_MATCHES}帧")
    print("="*60 + "\n")
    
    # 1. 提取clip1的所有帧
    print("1. 提取clip1的所有帧...")
    clip1_frames = extract_all_frames(CLIP1_PATH)
    
    # 2. 提取所有源视频的帧
    print("\n2. 提取源视频的帧...")
    source_videos_info = []
    
    source_paths = [
        ("video1_天赋变异后我无敌了_ep1.mp4", SOURCE_DIR / "video1_天赋变异后我无敌了_ep1.mp4"),
        ("video2_开局饕餮血统我吞噬一切_ep1.mp4", SOURCE_DIR / "video2_开局饕餮血统我吞噬一切_ep1.mp4"),
        ("video3_咒术反噬我有无限血条_ep1.mp4", SOURCE_DIR / "video3_咒术反噬我有无限血条_ep1.mp4")
    ]
    
    for name, path in source_paths:
        frames = extract_all_frames(path)
        source_videos_info.append((name, frames))
    
    # 3. 查找片段边界
    print("\n3. 查找片段边界...")
    results = find_segment_boundaries(clip1_frames, source_videos_info)
    
    # 4. 创建clip2
    output_path = OUTPUT_DIR / "clip2_true_reconstructed.mp4"
    create_clip2_from_results(results, output_path)
    
    # 5. 显示结果
    print("\n" + "="*60)
    print("📊 重构结果")
    print("="*60)
    for idx, (source_name, _, start, end, sim, consecutive) in enumerate(results):
        print(f"\n片段{idx+1}:")
        print(f"  来源: {source_name}")
        print(f"  时间: {start:.2f}s - {end:.2f}s")
        print(f"  相似度: {sim:.2%}")
        print(f"  连续匹配: {consecutive}帧")
    
    # 6. 验证
    print("\n" + "="*60)
    print("🔍 验证...")
    print("="*60)
    
    info1 = get_video_info(CLIP1_PATH)
    info2 = get_video_info(output_path)
    
    print(f"\nclip1: {info1['duration']:.2f}s, {info1['frame_count']}帧")
    print(f"clip2: {info2['duration']:.2f}s, {info2['frame_count']}帧")
    
    duration_diff = abs(info1['duration'] - info2['duration'])
    print(f"\n时长差异: {duration_diff:.2f}s")
    
    if duration_diff < 1.0:
        print("✅ 时长匹配，可以进行逐帧比对")
    else:
        print("⚠️ 时长差异较大")

if __name__ == "__main__":
    main()
