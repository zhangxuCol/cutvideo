#!/usr/bin/env python3
"""
使用 OpenCV DNN 进行深度学习特征提取

使用 OpenCV 自带的 DNN 模块，无需额外下载模型
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from typing import Dict, List

class OpenCVDeepValidator:
    """基于 OpenCV DNN 的图像内容验证器"""
    
    def __init__(self):
        print("初始化 OpenCV DNN 验证器...")
        # 使用 OpenCV 的 DNN 模块
        # 尝试加载预训练的模型
        self.net = None
        self._load_model()
        self.temp_dir = None
    
    def _load_model(self):
        """加载预训练模型"""
        # 使用 OpenCV 自带的模型或者简单的特征提取
        # 这里使用 SIFT 特征作为替代（虽然不如深度学习，但比直方图好）
        try:
            self.sift = cv2.SIFT_create()
            print("  使用 SIFT 特征提取")
        except:
            # 如果 SIFT 不可用，使用 ORB
            self.sift = cv2.ORB_create(nfeatures=500)
            print("  使用 ORB 特征提取")
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path):
        """提取视频帧"""
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', 
               '-q:v', '2', str(output_path)]
        subprocess.run(cmd, capture_output=True)
    
    def get_image_features(self, image_path: Path) -> np.ndarray:
        """获取图像特征（使用 SIFT/ORB + 颜色直方图）"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. SIFT/ORB 特征
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # 2. 颜色直方图特征
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # 如果有关键点描述符，计算描述符的统计特征
        if descriptors is not None and len(descriptors) > 0:
            # 计算描述符的均值和标准差作为特征
            desc_mean = np.mean(descriptors, axis=0)
            desc_std = np.std(descriptors, axis=0)
            sift_features = np.concatenate([desc_mean, desc_std])
            
            # 限制特征维度，避免过大
            sift_features = sift_features[:128]
        else:
            sift_features = np.zeros(128)
        
        # 组合特征
        combined_features = np.concatenate([sift_features, hist])
        
        # 归一化
        norm = np.linalg.norm(combined_features)
        if norm > 0:
            combined_features = combined_features / norm
        
        return combined_features
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算余弦相似度"""
        if features1 is None or features2 is None:
            return 0.0
        
        similarity = np.dot(features1, features2)
        # 转换为 0-1 范围
        return (similarity + 1) / 2
    
    def compare_existing_frames(self, frame_dir: Path) -> Dict:
        """对比已存在的对比帧"""
        print("=" * 60)
        print("使用 SIFT + 颜色特征对比")
        print("=" * 60)
        
        compare_times = [0, 30, 60, 100, 150, 200]
        
        results = {
            '20s': {'scores': [], 'frames': []},
            '165s': {'scores': [], 'frames': []}
        }
        
        for t in compare_times:
            target_frame = frame_dir / f"target_{t:03d}s.jpg"
            frame_20s = frame_dir / f"source_20s_{t:03d}s.jpg"
            frame_165s = frame_dir / f"source_165s_{t:03d}s.jpg"
            
            if not target_frame.exists():
                continue
            
            print(f"\n@{t}s:")
            
            target_feat = self.get_image_features(target_frame)
            
            # 对比 @20s
            if frame_20s.exists():
                feat_20s = self.get_image_features(frame_20s)
                sim_20s = self.calculate_similarity(target_feat, feat_20s)
                results['20s']['scores'].append(sim_20s)
                results['20s']['frames'].append(t)
                status = "✓" if sim_20s > 0.7 else "✗" if sim_20s < 0.5 else "~"
                print(f"  @20s:  {sim_20s:.1%} {status}")
            
            # 对比 @165s
            if frame_165s.exists():
                feat_165s = self.get_image_features(frame_165s)
                sim_165s = self.calculate_similarity(target_feat, feat_165s)
                results['165s']['scores'].append(sim_165s)
                results['165s']['frames'].append(t)
                status = "✓" if sim_165s > 0.7 else "✗" if sim_165s < 0.5 else "~"
                print(f"  @165s: {sim_165s:.1%} {status}")
        
        # 总结
        print(f"\n{'='*60}")
        print("总结")
        print("=" * 60)
        
        if results['20s']['scores']:
            avg_20s = np.mean(results['20s']['scores'])
            min_20s = np.min(results['20s']['scores'])
            pass_20s = "✅" if avg_20s > 0.7 and min_20s > 0.5 else "❌"
            print(f"@20s:  平均 {avg_20s:.1%}, 最低 {min_20s:.1%} {pass_20s}")
        
        if results['165s']['scores']:
            avg_165s = np.mean(results['165s']['scores'])
            min_165s = np.min(results['165s']['scores'])
            pass_165s = "✅" if avg_165s > 0.7 and min_165s > 0.5 else "❌"
            print(f"@165s: 平均 {avg_165s:.1%}, 最低 {min_165s:.1%} {pass_165s}")
        
        return results


def main():
    """测试 OpenCV DNN 验证器"""
    validator = OpenCVDeepValidator()
    
    # 对比已有的对比帧
    frame_dir = Path("/Users/zhangxu/work/项目/cutvideo/temp_outputs/comparison_frames")
    
    if frame_dir.exists():
        validator.compare_existing_frames(frame_dir)
    else:
        print("对比帧目录不存在")


if __name__ == '__main__':
    main()
