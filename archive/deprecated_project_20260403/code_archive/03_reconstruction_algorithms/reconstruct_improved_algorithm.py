#!/usr/bin/env python3
"""
改进的逐帧内容查找重构脚本 - 多特征融合版本
使用颜色直方图、边缘特征等多种特征提高匹配准确性
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
PIXEL_THRESHOLD = 0.90      # 像素级相似度阈值
HIST_THRESHOLD = 0.85       # 直方图相似度阈值
EDGE_THRESHOLD = 0.80       # 边缘特征阈值
COMBINED_THRESHOLD = 0.88   # 综合相似度阈值
MIN_CONSECUTIVE_MATCHES = 15  # 最少连续匹配帧数

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

def extract_frame_features(frame):
    """
    提取帧的多维特征
    返回: (缩小帧, 颜色直方图, 边缘图)
    """
    # 1. 缩小帧（用于像素级比较）
    frame_small = cv2.resize(frame, (320, 180))
    
    # 2. 颜色直方图特征（HSV空间）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = hist / (np.sum(hist) + 1e-7)  # 归一化
    
    # 3. 边缘特征（Canny边缘检测）
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    return frame_small, hist, edges

def extract_all_features(video_path, resize=(320, 180)):
    """提取视频所有帧的多维特征"""
    print(f"  提取特征: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    features = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_small, hist, edges = extract_frame_features(frame)
        features.append({
            'frame': frame_small,
            'hist': hist,
            'edges': edges,
            'idx': frame_idx
        })
        
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    已处理 {frame_idx} 帧...")
    
    cap.release()
    print(f"    共 {len(features)} 帧")
    return features

def calculate_combined_similarity(feat1, feat2):
    """
    计算综合相似度（融合多种特征）
    """
    # 1. 像素级相似度
    diff = np.abs(feat1['frame'].astype(float) - feat2['frame'].astype(float))
    pixel_sim = 1 - (np.mean(diff) / 255)
    
    # 2. 颜色直方图相似度（相关性）
    hist_sim = cv2.compareHist(
        feat1['hist'].astype(np.float32).reshape(-1, 1),
        feat2['hist'].astype(np.float32).reshape(-1, 1),
        cv2.HISTCMP_CORREL
    )
    hist_sim = max(0, hist_sim)  # 确保非负
    
    # 3. 边缘特征相似度
    edge_diff = np.abs(feat1['edges'].astype(float) - feat2['edges'].astype(float))
    edge_sim = 1 - (np.mean(edge_diff) / 255)
    
    # 综合相似度（加权平均）
    # 像素级最重要，其次是颜色，边缘作为辅助
    combined_sim = 0.5 * pixel_sim + 0.35 * hist_sim + 0.15 * edge_sim
    
    return {
        'pixel': pixel_sim,
        'hist': hist_sim,
        'edge': edge_sim,
        'combined': combined_sim
    }

def find_best_match_improved(target_features, source_features, source_name):
    """
    改进的匹配算法（使用多特征融合和连续性验证）
    """
    print(f"\n  在 {source_name} 中查找...")
    
    best_match = {
        'start_idx': -1,
        'avg_combined': 0,
        'avg_pixel': 0,
        'avg_hist': 0,
        'consecutive_matches': 0
    }
    
    target_len = len(target_features)
    source_len = len(source_features)
    
    # 滑动窗口
    for start_idx in range(source_len - target_len + 1):
        combined_sims = []
        pixel_sims = []
        hist_sims = []
        consecutive_count = 0
        
        for i in range(target_len):
            sims = calculate_combined_similarity(
                target_features[i],
                source_features[start_idx + i]
            )
            
            combined_sims.append(sims['combined'])
            pixel_sims.append(sims['pixel'])
            hist_sims.append(sims['hist'])
            
            # 连续性检查（综合相似度必须超过阈值）
            if sims['combined'] >= COMBINED_THRESHOLD:
                consecutive_count += 1
            else:
                if i < MIN_CONSECUTIVE_MATCHES:
                    consecutive_count = 0
                    break
        
        # 如果连续匹配足够，评估整体质量
        if consecutive_count >= MIN_CONSECUTIVE_MATCHES:
            avg_combined = np.mean(combined_sims)
            avg_pixel = np.mean(pixel_sims)
            avg_hist = np.mean(hist_sims)
            
            # 更新最佳匹配（优先考虑综合相似度）
            if avg_combined > best_match['avg_combined']:
                best_match = {
                    'start_idx': start_idx,
                    'avg_combined': avg_combined,
                    'avg_pixel': avg_pixel,
                    'avg_hist': avg_hist,
                    'consecutive_matches': consecutive_count
                }
        
        # 进度输出
        if start_idx > 0 and start_idx % 1000 == 0:
            print(f"    已检查 {start_idx}/{source_len - target_len} 帧...")
    
    return best_match

def find_segments_improved(clip1_features, source_videos_features):
    """
    改进的片段查找（使用多特征）
    """
    print("="*60)
    print("🔍 改进的逐帧内容查找（多特征融合）")
    print("="*60)
    
    clip1_len = len(clip1_features)
    print(f"\nclip1 总帧数: {clip1_len}")
    
    # 分成3个片段
    segment_size = clip1_len // 3
    segment_starts = [0, segment_size, 2 * segment_size]
    segment_names = ['片段1', '片段2', '片段3']
    
    results = []
    
    for seg_idx, (seg_name, start_frame) in enumerate(zip(segment_names, segment_starts)):
        print(f"\n{'='*60}")
        print(f"查找 {seg_name} (clip1第{start_frame}-{start_frame+segment_size}帧)")
        print(f"{'='*60}")
        
        segment_features = clip1_features[start_frame:start_frame+segment_size]
        print(f"  片段帧数: {len(segment_features)}")
        
        # 在每个源视频中查找
        best_match_info = None
        best_source_idx = -1
        
        for idx, (source_name, source_features) in enumerate(source_videos_features):
            match_info = find_best_match_improved(
                segment_features, source_features, source_name
            )
            
            if match_info['start_idx'] >= 0:
                print(f"\n    ✅ 匹配成功:")
                print(f"       帧位置: {match_info['start_idx']}")
                print(f"       综合相似度: {match_info['avg_combined']:.2%}")
                print(f"       像素相似度: {match_info['avg_pixel']:.2%}")
                print(f"       颜色相似度: {match_info['avg_hist']:.2%}")
                print(f"       连续匹配: {match_info['consecutive_matches']}帧")
                
                if best_match_info is None or match_info['avg_combined'] > best_match_info['avg_combined']:
                    best_match_info = match_info
                    best_source_idx = idx
            else:
                print(f"\n    ❌ 未找到足够连续的匹配")
        
        if best_source_idx >= 0:
            source_name, source_features = source_videos_features[best_source_idx]
            fps = 30
            start_time = best_match_info['start_idx'] / fps
            end_time = (best_match_info['start_idx'] + len(segment_features)) / fps
            
            print(f"\n  🎯 最终选择: {source_name}")
            print(f"     时间: {start_time:.2f}s - {end_time:.2f}s")
            print(f"     综合相似度: {best_match_info['avg_combined']:.2%}")
            
            results.append((
                source_name,
                source_videos_features[best_source_idx][1],
                start_time,
                end_time,
                best_match_info
            ))
        else:
            print(f"\n  ❌ 未找到匹配，使用默认")
            default_idx = seg_idx % len(source_videos_features)
            default_name, default_features = source_videos_features[default_idx]
            default_start = seg_idx * 15
            default_end = default_start + 15
            results.append((
                default_name,
                default_features,
                default_start,
                default_end,
                {'avg_combined': 0, 'avg_pixel': 0, 'avg_hist': 0}
            ))
    
    return results

def create_clip2(results, output_path):
    """创建clip2"""
    print("\n" + "="*60)
    print("🎬 创建裁剪视频2")
    print("="*60)
    
    temp_dir = OUTPUT_DIR / "temp_improved"
    temp_dir.mkdir(exist_ok=True)
    
    source_paths = [
        SOURCE_DIR / "video1_天赋变异后我无敌了_ep1.mp4",
        SOURCE_DIR / "video2_开局饕餮血统我吞噬一切_ep1.mp4",
        SOURCE_DIR / "video3_咒术反噬我有无限血条_ep1.mp4"
    ]
    
    for idx, (source_name, _, start_time, end_time, match_info) in enumerate(results):
        source_path = None
        for sp in source_paths:
            if sp.name == source_name:
                source_path = sp
                break
        
        if not source_path:
            source_path = source_paths[idx % len(source_paths)]
        
        print(f"\n{idx+1}. 从 {source_path.name}")
        print(f"   时间: {start_time:.2f}s - {end_time:.2f}s")
        if match_info['avg_combined'] > 0:
            print(f"   相似度: {match_info['avg_combined']:.2%}")
        
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
    
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ 创建完成: {output_path}")

def main():
    print("\n" + "="*60)
    print("🎬 改进的逐帧内容查找重构工具")
    print("   特征: 像素 + 颜色直方图 + 边缘")
    print(f"   综合阈值: {COMBINED_THRESHOLD:.0%}")
    print("="*60 + "\n")
    
    # 1. 提取clip1特征
    print("1. 提取clip1的多维特征...")
    clip1_features = extract_all_features(CLIP1_PATH)
    
    # 2. 提取源视频特征
    print("\n2. 提取源视频的多维特征...")
    source_videos_features = []
    
    source_paths = [
        ("video1_天赋变异后我无敌了_ep1.mp4", SOURCE_DIR / "video1_天赋变异后我无敌了_ep1.mp4"),
        ("video2_开局饕餮血统我吞噬一切_ep1.mp4", SOURCE_DIR / "video2_开局饕餮血统我吞噬一切_ep1.mp4"),
        ("video3_咒术反噬我有无限血条_ep1.mp4", SOURCE_DIR / "video3_咒术反噬我有无限血条_ep1.mp4")
    ]
    
    for name, path in source_paths:
        features = extract_all_features(path)
        source_videos_features.append((name, features))
    
    # 3. 查找片段
    print("\n3. 使用多特征查找片段...")
    results = find_segments_improved(clip1_features, source_videos_features)
    
    # 4. 创建clip2
    output_path = OUTPUT_DIR / "clip2_improved_reconstructed.mp4"
    create_clip2(results, output_path)
    
    # 5. 显示结果
    print("\n" + "="*60)
    print("📊 重构结果汇总")
    print("="*60)
    for idx, (source_name, _, start, end, match_info) in enumerate(results):
        print(f"\n片段{idx+1}:")
        print(f"  来源: {source_name}")
        print(f"  时间: {start:.2f}s - {end:.2f}s")
        if match_info['avg_combined'] > 0:
            print(f"  综合相似度: {match_info['avg_combined']:.2%}")
            print(f"  像素相似度: {match_info['avg_pixel']:.2%}")
            print(f"  颜色相似度: {match_info['avg_hist']:.2%}")
    
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
        print("✅ 时长匹配")
    else:
        print(f"⚠️ 时长差异: {duration_diff:.2f}s")

if __name__ == "__main__":
    main()
