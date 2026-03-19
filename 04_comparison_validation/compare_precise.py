#!/usr/bin/env python3
"""
精确视频比对 - 逐帧比对
"""

import cv2
import numpy as np
from pathlib import Path

def compare_videos_frame_by_frame(video1_path, video2_path, sample_interval=30):
    """
    逐帧比对两个视频
    sample_interval: 每隔多少帧比对一次（默认30帧=1秒）
    """
    print("="*60)
    print("🎬 精确视频比对")
    print("="*60)
    
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    
    # 获取视频信息
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n视频1: {total1}帧, {fps1:.2f}fps")
    print(f"视频2: {total2}帧, {fps2:.2f}fps")
    
    # 逐帧比对
    similarities = []
    frame_idx = 0
    
    print(f"\n🔍 逐帧比对中 (每{sample_interval}帧采样)...")
    
    while True:
        # 设置位置
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # 计算相似度
        diff = np.abs(frame1.astype(float) - frame2.astype(float))
        sim = 1 - (np.mean(diff) / 255)
        similarities.append(sim)
        
        if len(similarities) % 10 == 0:
            print(f"   已比对 {len(similarities)} 帧, 当前相似度: {sim:.2%}")
        
        frame_idx += sample_interval
        
        # 防止无限循环
        if frame_idx > min(total1, total2):
            break
    
    cap1.release()
    cap2.release()
    
    # 统计结果
    if similarities:
        avg_sim = np.mean(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
        
        # 低于阈值的帧
        threshold = 0.90
        low_count = sum(1 for s in similarities if s < threshold)
        
        print(f"\n" + "="*60)
        print("📊 比对结果")
        print("="*60)
        print(f"   总比对帧数: {len(similarities)}")
        print(f"   平均相似度: {avg_sim:.2%}")
        print(f"   最低相似度: {min_sim:.2%}")
        print(f"   最高相似度: {max_sim:.2%}")
        print(f"   低于{threshold:.0%}的帧: {low_count}/{len(similarities)} ({low_count/len(similarities)*100:.1f}%)")
        
        # 判断结果
        passed = avg_sim >= 0.95 and low_count / len(similarities) < 0.1
        
        print(f"\n" + "="*60)
        if passed:
            print("✅ 验证通过 - 视频高度相似")
        else:
            print("❌ 验证未通过 - 视频差异较大")
        print("="*60)
        
        return passed, avg_sim
    else:
        print("❌ 无法比对视频")
        return False, 0.0

if __name__ == "__main__":
    video1 = Path("/Users/zhangxu/.openclaw/workspace/test_videos/clip1.mp4")
    video2 = Path("/Users/zhangxu/.openclaw/workspace/test_videos/clip2_reconstructed.mp4")
    
    passed, sim = compare_videos_frame_by_frame(video1, video2)
    exit(0 if passed else 1)
