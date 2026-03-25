#!/usr/bin/env python3
"""
快速二次校验脚本 - 验证重构视频与原素材内容是否一致
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys

def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, fps, frame_count

def extract_sample_frames(video_path, num_samples=20):
    """均匀采样提取关键帧"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []
    
    # 均匀采样
    indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_small = cv2.resize(frame, (320, 180))
            frames.append(frame_small)
    
    cap.release()
    return frames

def calculate_frame_similarity(frame1, frame2):
    """计算两帧相似度"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 直方图相似度
    hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
    hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 模板匹配
    result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    template_sim = np.max(result)
    
    # 综合
    return 0.5 * max(0, hist_sim) + 0.5 * template_sim

def quick_verify(original_path, reconstructed_path):
    """快速验证两个视频内容是否一致"""
    orig = Path(original_path)
    recon = Path(reconstructed_path)
    
    if not orig.exists():
        return {'error': f'原素材不存在: {orig}'}
    if not recon.exists():
        return {'error': f'重构视频不存在: {recon}'}
    
    # 获取时长信息
    dur1, fps1, _ = get_video_info(orig)
    dur2, fps2, _ = get_video_info(recon)
    
    duration_diff = abs(dur1 - dur2)
    duration_match = duration_diff < 2.0
    
    # 提取样本帧
    frames1 = extract_sample_frames(orig, 15)
    frames2 = extract_sample_frames(recon, 15)
    
    if len(frames1) != len(frames2):
        return {
            'original_duration': dur1,
            'reconstructed_duration': dur2,
            'duration_diff': duration_diff,
            'error': '采样帧数不一致'
        }
    
    # 计算相似度
    similarities = []
    for f1, f2 in zip(frames1, frames2):
        sim = calculate_frame_similarity(f1, f2)
        similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    # 判断标准：平均相似度 >= 85% 且 最低相似度 >= 70%
    content_match = avg_sim >= 0.85 and min_sim >= 0.70
    
    return {
        'original_duration': dur1,
        'reconstructed_duration': dur2,
        'duration_diff': duration_diff,
        'duration_match': duration_match,
        'avg_similarity': avg_sim,
        'min_similarity': min_sim,
        'max_similarity': max_sim,
        'content_match': content_match,
        'all_similarities': similarities
    }

def main():
    """批量验证所有重构视频"""
    import argparse
    
    parser = argparse.ArgumentParser(description='快速二次校验工具')
    parser.add_argument('--report', type=str, default='01_test_data_generation/source_videos/南城以北/output/reconstruction_report_v5.json',
                        help='重构报告路径')
    parser.add_argument('--adx-dir', type=str, default='01_test_data_generation/source_videos/南城以北/adx原',
                        help='原素材目录')
    parser.add_argument('--output-dir', type=str, default='01_test_data_generation/source_videos/南城以北/output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 读取重构报告
    with open(args.report, 'r') as f:
        report = json.load(f)
    
    print("="*70)
    print("🎬 二次校验 - 验证重构视频内容一致性")
    print("="*70)
    
    results = []
    passed_count = 0
    failed_count = 0
    
    for item in report['results']:
        cut_name = item['cut_video_name']
        output_name = cut_name.replace('.mp4', '_cut.mp4')
        
        orig_path = Path(args.adx_dir) / cut_name
        recon_path = Path(args.output_dir) / output_name
        
        print(f"\n📹 验证: {cut_name}")
        
        result = quick_verify(orig_path, recon_path)
        result['video_name'] = cut_name
        results.append(result)
        
        if 'error' in result:
            print(f"   ❌ 错误: {result['error']}")
            failed_count += 1
        else:
            status = "✅" if result['content_match'] else "❌"
            print(f"   {status} 平均相似度: {result['avg_similarity']:.1%}")
            print(f"      最低相似度: {result['min_similarity']:.1%}")
            print(f"      时长差异: {result['duration_diff']:.2f}s")
            
            if result['content_match']:
                passed_count += 1
            else:
                failed_count += 1
    
    # 打印汇总
    print("\n" + "="*70)
    print("📊 二次校验汇总")
    print("="*70)
    print(f"   总视频数: {len(results)}")
    print(f"   内容匹配: {passed_count} ({passed_count/len(results)*100:.1f}%)")
    print(f"   内容不匹配: {failed_count} ({failed_count/len(results)*100:.1f}%)")
    
    # 保存校验报告
    verify_report = {
        'total': len(results),
        'passed': passed_count,
        'failed': failed_count,
        'pass_rate': passed_count / len(results) * 100 if results else 0,
        'results': results
    }
    
    report_path = Path(args.output_dir) / 'verify_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(verify_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n   报告已保存: {report_path}")
    
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    exit(main())
