#!/usr/bin/env python3
"""
比对 clip1_v2 和 clip2_reconstructed_real_v2
"""

import cv2
import numpy as np
from pathlib import Path

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

def compare_videos(video1_path, video2_path, sample_interval=30):
    """逐帧比对两个视频"""
    print("="*60)
    print("🎬 视频比对")
    print("="*60)
    
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n视频1: {total1}帧, {fps1:.2f}fps")
    print(f"视频2: {total2}帧, {fps2:.2f}fps")
    
    similarities = []
    frame_idx = 0
    
    print(f"\n🔍 逐帧比对中...")
    
    while True:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        diff = np.abs(frame1.astype(float) - frame2.astype(float))
        sim = 1 - (np.mean(diff) / 255)
        similarities.append(sim)
        
        if len(similarities) % 10 == 0:
            print(f"   已比对 {len(similarities)} 帧, 当前相似度: {sim:.2%}")
        
        frame_idx += sample_interval
        
        if frame_idx > min(total1, total2):
            break
    
    cap1.release()
    cap2.release()
    
    if similarities:
        avg_sim = np.mean(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)
        low_count = sum(1 for s in similarities if s < 0.90)
        
        print(f"\n" + "="*60)
        print("📊 比对结果")
        print("="*60)
        print(f"   总比对帧数: {len(similarities)}")
        print(f"   平均相似度: {avg_sim:.2%}")
        print(f"   最低相似度: {min_sim:.2%}")
        print(f"   最高相似度: {max_sim:.2%}")
        print(f"   低于90%的帧: {low_count}/{len(similarities)}")
        
        passed = avg_sim >= 0.95 and low_count / len(similarities) < 0.1
        
        print(f"\n" + "="*60)
        if passed:
            print("✅ 验证通过")
        else:
            print("❌ 验证未通过")
        print("="*60)
        
        return passed, avg_sim
    else:
        print("❌ 无法比对")
        return False, 0.0

def main():
    video1 = Path("/Users/zhangxu/.openclaw/workspace/test_videos/clip1_real_v2.mp4")
    video2 = Path("/Users/zhangxu/.openclaw/workspace/test_videos/clip2_reconstructed_real_v2.mp4")
    
    print("\n📊 基本信息:")
    info1 = get_video_info(video1)
    info2 = get_video_info(video2)
    
    print(f"\n视频1 (clip1_v2):")
    print(f"   时长: {info1['duration']:.2f}秒")
    print(f"   分辨率: {info1['width']}x{info1['height']}")
    
    print(f"\n视频2 (clip2_v2):")
    print(f"   时长: {info2['duration']:.2f}秒")
    print(f"   分辨率: {info2['width']}x{info2['height']}")
    
    duration_diff = abs(info1['duration'] - info2['duration'])
    print(f"\n时长差异: {duration_diff:.2f}秒")
    
    if duration_diff > 1.0:
        print("❌ 时长差异过大")
        return 1
    
    passed, sim = compare_videos(video1, video2)
    return 0 if passed else 1

if __name__ == "__main__":
    exit(main())
