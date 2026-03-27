#!/usr/bin/env python3
"""
新的验证逻辑设计 - 内容一致性检查

问题分析：
1. V7 的音频指纹提取可能有问题（16kHz + 滤波器 vs 8kHz）
2. 视频验证采样点太少，容易被局部相似误导
3. 评分高不代表内容一致

新方案：
1. 多维度内容验证（人脸、场景、文字等）
2. 时序一致性检查（连续帧的连贯性）
3. 关键帧强制匹配（开头、结尾、中间）
4. 人工反馈循环（确认正确位置后学习）
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from typing import List, Tuple, Dict

class ContentAwareValidator:
    """内容感知验证器 - 不只是评分，还要检查内容一致性"""
    
    def __init__(self):
        self.temp_dir = None
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path):
        """提取帧"""
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', 
               '-q:v', '2', str(output_path)]
        subprocess.run(cmd, capture_output=True)
    
    def calculate_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        """计算帧相似度（多种方法）"""
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))
        if img1 is None or img2 is None:
            return 0.0
        
        # 统一尺寸
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.resize(gray1, (320, 180))
        gray2 = cv2.resize(gray2, (320, 180))
        
        # 1. 直方图相似度（颜色分布）
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 2. 模板匹配（结构相似）
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        # 3. SSIM（结构相似性指数）
        # 简化版：使用边缘检测后比较
        edges1 = cv2.Canny(gray1, 100, 200)
        edges2 = cv2.Canny(gray2, 100, 200)
        edge_sim = np.sum(edges1 == edges2) / edges1.size
        
        # 综合评分
        return 0.4 * max(0, hist_sim) + 0.4 * template_sim + 0.2 * edge_sim
    
    def validate_content_consistency(self, 
                                     target_video: Path, 
                                     source_video: Path, 
                                     start_time: float,
                                     duration: float) -> Dict:
        """
        内容一致性验证
        
        返回:
            {
                'overall_score': float,      # 综合评分
                'is_consistent': bool,       # 是否一致
                'key_frames_match': bool,    # 关键帧是否匹配
                'temporal_coherence': float, # 时序连贯性
                'details': {                 # 详细结果
                    'frame_scores': [],
                    'min_score': float,
                    'max_score': float,
                    'variance': float
                }
            }
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # 1. 密集采样（每10秒）
            sample_interval = 10
            sample_times = np.arange(0, duration, sample_interval)
            
            frame_scores = []
            frame_details = []
            
            print(f"  密集采样验证 ({len(sample_times)} 个时间点)...")
            
            for t in sample_times:
                if t >= duration:
                    break
                
                target_frame = self.temp_dir / f"target_{t:.0f}.jpg"
                source_frame = self.temp_dir / f"source_{t:.0f}.jpg"
                
                self.extract_frame(target_video, t, target_frame)
                self.extract_frame(source_video, start_time + t, source_frame)
                
                if target_frame.exists() and source_frame.exists():
                    sim = self.calculate_similarity(target_frame, source_frame)
                    frame_scores.append(sim)
                    frame_details.append({'time': t, 'score': sim})
                    
                    status = "✓" if sim > 0.7 else "✗" if sim < 0.5 else "~"
                    print(f"    @{t:3.0f}s: {sim:.1%} {status}")
            
            if not frame_scores:
                return {'overall_score': 0, 'is_consistent': False, 'details': {}}
            
            # 2. 统计分析
            scores = np.array(frame_scores)
            mean_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            variance = np.var(scores)
            
            # 3. 关键帧强制检查（开头、中间、结尾）
            key_times = [0, duration * 0.5, duration * 0.9]
            key_scores = []
            
            print(f"\n  关键帧检查...")
            for t in key_times:
                target_frame = self.temp_dir / f"key_target_{t:.0f}.jpg"
                source_frame = self.temp_dir / f"key_source_{t:.0f}.jpg"
                
                self.extract_frame(target_video, t, target_frame)
                self.extract_frame(source_video, start_time + t, source_frame)
                
                if target_frame.exists() and source_frame.exists():
                    sim = self.calculate_similarity(target_frame, source_frame)
                    key_scores.append(sim)
                    print(f"    @{t:3.0f}s: {sim:.1%}")
            
            key_frames_match = all(s > 0.6 for s in key_scores) if key_scores else False
            
            # 4. 时序连贯性检查（分数变化是否平滑）
            if len(frame_scores) > 2:
                diffs = np.diff(frame_scores)
                temporal_coherence = 1.0 - min(1.0, np.std(diffs) * 2)
            else:
                temporal_coherence = 0.5
            
            # 5. 综合判断
            # 不能只看平均分，还要看最低分和方差
            consistency_score = (
                0.3 * mean_score +      # 平均分
                0.3 * min_score +       # 最低分（不能有明显错误的帧）
                0.2 * (1 - variance) +  # 稳定性
                0.2 * temporal_coherence  # 时序连贯性
            )
            
            is_consistent = (
                mean_score > 0.7 and      # 平均分要高
                min_score > 0.5 and       # 不能有明显错误的帧
                key_frames_match and      # 关键帧必须匹配
                variance < 0.1            # 分数波动不能太大
            )
            
            return {
                'overall_score': consistency_score,
                'is_consistent': is_consistent,
                'key_frames_match': key_frames_match,
                'temporal_coherence': temporal_coherence,
                'details': {
                    'frame_scores': frame_details,
                    'mean_score': mean_score,
                    'min_score': min_score,
                    'max_score': max_score,
                    'variance': variance,
                    'key_scores': key_scores
                }
            }
            
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)


def test_validation():
    """测试新的验证逻辑"""
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    
    validator = ContentAwareValidator()
    
    # 测试 V7 找到的 @20s
    print("=" * 60)
    print("测试 V7 找到的 @20s 位置")
    print("=" * 60)
    
    result_20s = validator.validate_content_consistency(target, source, 20, 217.7)
    
    print(f"\n结果:")
    print(f"  综合评分: {result_20s['overall_score']:.1%}")
    print(f"  内容一致: {result_20s['is_consistent']}")
    print(f"  关键帧匹配: {result_20s['key_frames_match']}")
    print(f"  时序连贯性: {result_20s['temporal_coherence']:.1%}")
    print(f"  平均分: {result_20s['details']['mean_score']:.1%}")
    print(f"  最低分: {result_20s['details']['min_score']:.1%}")
    print(f"  方差: {result_20s['details']['variance']:.4f}")
    
    # 测试 @165s（验证脚本找到的位置）
    print(f"\n{'='*60}")
    print("测试 @165s 位置（验证脚本找到的）")
    print("=" * 60)
    
    result_165s = validator.validate_content_consistency(target, source, 165, 217.7)
    
    print(f"\n结果:")
    print(f"  综合评分: {result_165s['overall_score']:.1%}")
    print(f"  内容一致: {result_165s['is_consistent']}")
    print(f"  关键帧匹配: {result_165s['key_frames_match']}")
    print(f"  时序连贯性: {result_165s['temporal_coherence']:.1%}")
    print(f"  平均分: {result_165s['details']['mean_score']:.1%}")
    print(f"  最低分: {result_165s['details']['min_score']:.1%}")
    print(f"  方差: {result_165s['details']['variance']:.4f}")
    
    # 对比
    print(f"\n{'='*60}")
    print("对比")
    print("=" * 60)
    print(f"  @20s:  {'✅ 通过' if result_20s['is_consistent'] else '❌ 失败'} "
          f"(评分: {result_20s['overall_score']:.1%})")
    print(f"  @165s: {'✅ 通过' if result_165s['is_consistent'] else '❌ 失败'} "
          f"(评分: {result_165s['overall_score']:.1%})")


if __name__ == '__main__':
    test_validation()
