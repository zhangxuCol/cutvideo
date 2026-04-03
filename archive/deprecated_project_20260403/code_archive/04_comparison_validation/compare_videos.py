#!/usr/bin/env python3
"""
比对两个视频的相似度
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil

def get_video_duration(video_path):
    """获取视频时长"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, fps, frame_count

def extract_frames(video_path, output_dir, fps=2):
    """提取视频帧"""
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    if frame_interval < 1:
        frame_interval = 1
    
    frames = []
    frame_idx = 0
    extracted_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps if video_fps > 0 else 0
            frame_path = output_dir / f"frame_{extracted_idx:06d}.jpg"
            frame_resized = cv2.resize(frame, (480, 270))
            cv2.imwrite(str(frame_path), frame_resized)
            frames.append((frame_path, timestamp))
            extracted_idx += 1
        
        frame_idx += 1
    
    cap.release()
    return frames

def calculate_similarity(frame1_path, frame2_path):
    """计算两帧相似度"""
    img1 = cv2.imread(str(frame1_path))
    img2 = cv2.imread(str(frame2_path))
    
    if img1 is None or img2 is None:
        return 0.0
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 直方图相似度
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 模板匹配
    result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    template_sim = np.max(result)
    
    # 感知哈希
    size = (32, 32)
    img1_small = cv2.resize(gray1, size)
    img2_small = cv2.resize(gray2, size)
    img1_float = np.float32(img1_small)
    img2_float = np.float32(img2_small)
    dct1 = cv2.dct(img1_float)
    dct2 = cv2.dct(img2_float)
    low_freq1 = dct1[:8, :8].flatten()
    low_freq2 = dct2[:8, :8].flatten()
    mean1 = np.mean(low_freq1)
    mean2 = np.mean(low_freq2)
    hash1 = (low_freq1 > mean1).astype(int)
    hash2 = (low_freq2 > mean2).astype(int)
    hamming_dist = np.sum(hash1 != hash2)
    phash_sim = 1 - (hamming_dist / 64.0)
    
    # 综合评分
    final_sim = 0.4 * max(0, hist_sim) + 0.3 * template_sim + 0.3 * phash_sim
    return final_sim

def compare_videos(video1_path, video2_path, similarity_threshold=0.90):
    """比对两个视频"""
    print("="*60)
    print("🎬 视频比对工具")
    print("="*60)
    
    video1 = Path(video1_path)
    video2 = Path(video2_path)
    
    print(f"\n视频1 (参考): {video1.name}")
    print(f"视频2 (生成): {video2.name}")
    
    # 获取视频信息
    dur1, fps1, frames1 = get_video_duration(video1)
    dur2, fps2, frames2 = get_video_duration(video2)
    
    print(f"\n📊 基本信息:")
    print(f"   视频1: {dur1:.2f}秒, {fps1:.2f}fps, {frames1}帧")
    print(f"   视频2: {dur2:.2f}秒, {fps2:.2f}fps, {frames2}帧")
    
    duration_diff = abs(dur1 - dur2)
    duration_match = duration_diff < 0.5
    
    print(f"\n   时长差异: {duration_diff:.2f}秒 {'✅' if duration_match else '❌'}")
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    try:
        dir1 = temp_dir / "video1"
        dir2 = temp_dir / "video2"
        dir1.mkdir()
        dir2.mkdir()
        
        # 提取帧
        print(f"\n🎬 提取视频帧 (2 fps)...")
        frames1_list = extract_frames(video1, dir1, fps=2)
        frames2_list = extract_frames(video2, dir2, fps=2)
        
        print(f"   视频1: {len(frames1_list)}帧")
        print(f"   视频2: {len(frames2_list)}帧")
        
        # 逐帧比对
        min_frames = min(len(frames1_list), len(frames2_list))
        print(f"\n🔍 逐帧比对 ({min_frames}帧)...")
        
        similarities = []
        low_sim_frames = []
        
        for i in range(min_frames):
            sim = calculate_similarity(frames1_list[i][0], frames2_list[i][0])
            similarities.append(sim)
            
            if sim < similarity_threshold:
                low_sim_frames.append((i, sim))
            
            if (i + 1) % 20 == 0 or i == min_frames - 1:
                print(f"   进度: {i+1}/{min_frames} ({(i+1)/min_frames*100:.1f}%) - 当前相似度: {sim:.2%}")
        
        # 计算统计信息
        overall_similarity = np.mean(similarities)
        min_similarity = min(similarities)
        max_similarity = max(similarities)
        low_sim_count = len(low_sim_frames)
        low_sim_percent = low_sim_count / len(similarities) * 100
        
        print(f"\n" + "="*60)
        print("📊 比对结果")
        print("="*60)
        print(f"   整体相似度: {overall_similarity:.2%}")
        print(f"   最低相似度: {min_similarity:.2%}")
        print(f"   最高相似度: {max_similarity:.2%}")
        print(f"   低相似度帧: {low_sim_count}/{len(similarities)} ({low_sim_percent:.1f}%)")
        
        if low_sim_frames[:5]:
            print(f"\n   低相似度帧示例:")
            for idx, sim in low_sim_frames[:5]:
                time_sec = idx / 2
                print(f"      第{idx}帧 ({time_sec:.1f}s): {sim:.2%}")
        
        # 判断是否通过
        passed = (overall_similarity >= similarity_threshold and 
                 duration_match and 
                 low_sim_percent < 10)
        
        print(f"\n" + "="*60)
        print("✅ 验证结果" if passed else "❌ 验证失败")
        print("="*60)
        print(f"   相似度阈值: {similarity_threshold:.0%}")
        print(f"   实际相似度: {overall_similarity:.2%} {'✅' if overall_similarity >= similarity_threshold else '❌'}")
        print(f"   时长匹配: {'✅' if duration_match else '❌'}")
        print(f"   低相似度比例: {low_sim_percent:.1f}% {'✅' if low_sim_percent < 10 else '❌'}")
        print(f"\n   最终结果: {'✅ 通过' if passed else '❌ 未通过'}")
        
        return {
            'passed': passed,
            'overall_similarity': overall_similarity,
            'duration_match': duration_match,
            'low_sim_percent': low_sim_percent,
            'similarities': similarities
        }
        
    finally:
        shutil.rmtree(temp_dir)

def main():
    import sys
    
    video1 = "/Users/zhangxu/.openclaw/workspace/test_videos/clip1.mp4"
    video2 = "/Users/zhangxu/.openclaw/workspace/test_videos/clip2_reconstructed.mp4"
    
    threshold = 0.90
    if len(sys.argv) > 1:
        threshold = float(sys.argv[1]) / 100
    
    result = compare_videos(video1, video2, threshold)
    
    if not result['passed']:
        print(f"\n⚠️ 相似度未达标，建议:")
        print(f"   1. 检查源视频是否包含目标素材")
        print(f"   2. 降低相似度阈值重试")
        print(f"   3. 使用更高质量的源视频")
    
    return 0 if result['passed'] else 1

if __name__ == "__main__":
    exit(main())
