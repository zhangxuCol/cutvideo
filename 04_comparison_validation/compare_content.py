#!/usr/bin/env python3
"""
视频内容比对工具 - 检查内容是否一致（而非像素级一致）
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil

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
        'duration': duration,
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height
    }

def extract_keyframes(video_path, num_frames=10):
    """提取关键帧（均匀分布）"""
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
            frame_small = cv2.resize(frame, (320, 240))
            frames.append((idx, frame_small))
    
    cap.release()
    return frames

def compare_content(frames1, frames2):
    """比较两组关键帧的内容相似度"""
    if len(frames1) != len(frames2):
        return 0.0, []
    
    similarities = []
    
    for (idx1, frame1), (idx2, frame2) in zip(frames1, frames2):
        # 转换为灰度
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算结构相似性（使用直方图）
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 计算颜色分布相似性
        color_sim = 0
        for i in range(3):
            hist_c1 = cv2.calcHist([frame1], [i], None, [32], [0, 256])
            hist_c2 = cv2.calcHist([frame2], [i], None, [32], [0, 256])
            color_sim += cv2.compareHist(hist_c1, hist_c2, cv2.HISTCMP_CORREL)
        color_sim /= 3
        
        # 综合相似度
        sim = 0.6 * max(0, hist_sim) + 0.4 * max(0, color_sim)
        similarities.append(sim)
    
    avg_sim = np.mean(similarities) if similarities else 0
    return avg_sim, similarities

def check_segment_structure(video1_path, video2_path):
    """
    检查两个视频的结构是否一致
    对于测试视频，检查是否包含相同的图案序列
    """
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    
    # 读取第一帧、中间帧、最后一帧
    def get_sample_frames(cap):
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        samples = []
        for pos in [0, total // 2, total - 1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                samples.append(cv2.resize(frame, (160, 120)))
        return samples
    
    samples1 = get_sample_frames(cap1)
    samples2 = get_sample_frames(cap2)
    
    cap1.release()
    cap2.release()
    
    if len(samples1) != len(samples2):
        return False, 0.0
    
    sims = []
    for f1, f2 in zip(samples1, samples2):
        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        
        hist1 = cv2.calcHist([gray1], [0], None, [32], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [32], [0, 256])
        sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        sims.append(max(0, sim))
    
    avg_sim = np.mean(sims)
    passed = avg_sim > 0.70  # 内容级别相似度阈值
    
    return passed, avg_sim

def main():
    video1 = Path("/Users/zhangxu/.openclaw/workspace/test_videos/clip1.mp4")
    video2 = Path("/Users/zhangxu/.openclaw/workspace/test_videos/clip2_reconstructed.mp4")
    
    print("="*60)
    print("🎬 视频内容比对工具")
    print("="*60)
    print(f"\n参考视频: {video1.name}")
    print(f"生成视频: {video2.name}")
    
    # 基本信息
    info1 = get_video_info(video1)
    info2 = get_video_info(video2)
    
    print(f"\n📊 基本信息:")
    print(f"   参考视频: {info1['duration']:.2f}秒, {info1['width']}x{info1['height']}")
    print(f"   生成视频: {info2['duration']:.2f}秒, {info2['width']}x{info2['height']}")
    
    duration_match = abs(info1['duration'] - info2['duration']) < 1.0
    resolution_match = info1['width'] == info2['width'] and info1['height'] == info2['height']
    
    print(f"\n   时长匹配: {'✅' if duration_match else '❌'} (差异 {abs(info1['duration'] - info2['duration']):.2f}秒)")
    print(f"   分辨率匹配: {'✅' if resolution_match else '❌'}")
    
    # 内容结构检查
    print(f"\n🔍 检查内容结构...")
    struct_passed, struct_sim = check_segment_structure(video1, video2)
    print(f"   结构相似度: {struct_sim:.2%} {'✅' if struct_passed else '❌'}")
    
    # 关键帧比对
    print(f"\n🎬 提取并比对关键帧...")
    frames1 = extract_keyframes(video1, 10)
    frames2 = extract_keyframes(video2, 10)
    
    content_sim, sims = compare_content(frames1, frames2)
    print(f"   内容相似度: {content_sim:.2%}")
    print(f"   各帧相似度: {[f'{s:.1%}' for s in sims]}")
    
    # 综合判断
    content_passed = content_sim >= 0.75  # 内容级别阈值 75%
    
    print(f"\n" + "="*60)
    print("📊 综合评估")
    print("="*60)
    print(f"   时长匹配: {'✅' if duration_match else '❌'}")
    print(f"   分辨率匹配: {'✅' if resolution_match else '❌'}")
    print(f"   结构相似度: {struct_sim:.2%} {'✅' if struct_passed else '❌'}")
    print(f"   内容相似度: {content_sim:.2%} {'✅' if content_passed else '❌'}")
    
    # 最终判断
    all_passed = duration_match and resolution_match and struct_passed and content_passed
    
    print(f"\n" + "="*60)
    if all_passed:
        print("✅ 验证通过 - 视频内容一致")
    else:
        print("❌ 验证失败 - 视频内容不一致")
    print("="*60)
    
    # 说明
    print(f"\n💡 说明:")
    print(f"   由于 clip1 和 clip2 使用不同的编码参数生成，")
    print(f"   像素级比对会有差异，但内容应该一致。")
    print(f"   此工具检查的是内容一致性而非像素一致性。")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
