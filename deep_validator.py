#!/usr/bin/env python3
"""
使用深度学习特征提取进行内容验证（无需下载 CLIP）

使用 torchvision 的预训练 ResNet 模型提取图像特征，
比传统算法更能理解内容。
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from typing import Dict, List

class DeepContentValidator:
    """基于深度学习的图像内容验证器"""
    
    def __init__(self):
        print("加载 ResNet 模型...")
        # 使用预训练的 ResNet50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # 去掉最后的分类层
        self.model.to(self.device)
        self.model.eval()
        print(f"  使用设备: {self.device}")
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        self.temp_dir = None
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path):
        """提取视频帧"""
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', 
               '-q:v', '2', str(output_path)]
        subprocess.run(cmd, capture_output=True)
    
    def get_image_features(self, image_path: Path) -> torch.Tensor:
        """获取图像的深度学习特征向量"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image_tensor)
            features = features.squeeze()
            # 归一化
            features = features / features.norm()
        
        return features
    
    def calculate_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """计算余弦相似度"""
        similarity = torch.nn.functional.cosine_similarity(
            features1.unsqueeze(0), 
            features2.unsqueeze(0)
        ).item()
        
        # 转换为 0-1 范围
        return (similarity + 1) / 2
    
    def validate_position(self, 
                         target_video: Path, 
                         source_video: Path, 
                         start_time: float,
                         duration: float,
                         sample_interval: float = 10.0) -> Dict:
        """验证特定位置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            print(f"\n验证 @{start_time}s...")
            
            sample_times = np.arange(0, duration, sample_interval)
            similarities = []
            details = []
            
            print(f"  采样 {len(sample_times)} 个时间点...")
            
            for t in sample_times:
                if t >= duration:
                    break
                
                target_frame = self.temp_dir / f"target_{t:.0f}.jpg"
                source_frame = self.temp_dir / f"source_{t:.0f}.jpg"
                
                self.extract_frame(target_video, t, target_frame)
                self.extract_frame(source_video, start_time + t, source_frame)
                
                if target_frame.exists() and source_frame.exists():
                    target_features = self.get_image_features(target_frame)
                    source_features = self.get_image_features(source_frame)
                    
                    sim = self.calculate_similarity(target_features, source_features)
                    similarities.append(sim)
                    details.append({'time': t, 'score': sim})
                    
                    status = "✓" if sim > 0.8 else "✗" if sim < 0.6 else "~"
                    print(f"    @{t:3.0f}s: {sim:.1%} {status}")
            
            if not similarities:
                return {'overall_score': 0, 'is_consistent': False}
            
            scores = np.array(similarities)
            mean_score = np.mean(scores)
            min_score = np.min(scores)
            
            # ResNet 特征的阈值
            is_consistent = mean_score > 0.75 and min_score > 0.60
            
            return {
                'overall_score': mean_score,
                'is_consistent': is_consistent,
                'mean_score': mean_score,
                'min_score': min_score,
                'details': details
            }
            
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def compare_existing_frames(self, frame_dir: Path) -> Dict:
        """对比已存在的对比帧"""
        print("=" * 60)
        print("使用深度学习特征对比已有帧")
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
            
            # 对比 @20s
            if frame_20s.exists():
                target_feat = self.get_image_features(target_frame)
                feat_20s = self.get_image_features(frame_20s)
                sim_20s = self.calculate_similarity(target_feat, feat_20s)
                results['20s']['scores'].append(sim_20s)
                results['20s']['frames'].append(t)
                print(f"  @20s: {sim_20s:.1%}")
            
            # 对比 @165s
            if frame_165s.exists():
                feat_165s = self.get_image_features(frame_165s)
                sim_165s = self.calculate_similarity(target_feat, feat_165s)
                results['165s']['scores'].append(sim_165s)
                results['165s']['frames'].append(t)
                print(f"  @165s: {sim_165s:.1%}")
        
        # 总结
        print(f"\n{'='*60}")
        print("总结")
        print("=" * 60)
        
        if results['20s']['scores']:
            avg_20s = np.mean(results['20s']['scores'])
            min_20s = np.min(results['20s']['scores'])
            print(f"@20s:  平均 {avg_20s:.1%}, 最低 {min_20s:.1%}")
        
        if results['165s']['scores']:
            avg_165s = np.mean(results['165s']['scores'])
            min_165s = np.min(results['165s']['scores'])
            print(f"@165s: 平均 {avg_165s:.1%}, 最低 {min_165s:.1%}")
        
        return results


def main():
    """测试深度学习验证器"""
    validator = DeepContentValidator()
    
    # 对比已有的对比帧
    frame_dir = Path("/Users/zhangxu/work/项目/cutvideo/comparison_frames")
    
    if frame_dir.exists():
        validator.compare_existing_frames(frame_dir)
    else:
        print("对比帧目录不存在，运行 extract_comparison.py 生成")


if __name__ == '__main__':
    main()
