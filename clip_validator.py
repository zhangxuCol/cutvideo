#!/usr/bin/env python3
"""
使用 CLIP 多模态 AI 进行真正的内容理解验证

CLIP (Contrastive Language-Image Pre-training) 可以将图像和文本映射到同一语义空间，
通过比较图像特征的相似度来判断内容是否一致。
"""

import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from typing import Dict, List

class CLIPContentValidator:
    """基于 CLIP 的内容验证器"""
    
    def __init__(self):
        # 加载 CLIP 模型
        print("加载 CLIP 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"  使用设备: {self.device}")
        self.temp_dir = None
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path):
        """提取视频帧"""
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', 
               '-q:v', '2', str(output_path)]
        subprocess.run(cmd, capture_output=True)
    
    def get_image_features(self, image_path: Path) -> torch.Tensor:
        """获取图像的 CLIP 特征向量"""
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def calculate_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """计算两个特征向量的余弦相似度"""
        similarity = (features1 @ features2.T).item()
        # 转换为 0-1 范围
        return (similarity + 1) / 2
    
    def validate_with_clip(self, 
                          target_video: Path, 
                          source_video: Path, 
                          start_time: float,
                          duration: float,
                          sample_interval: float = 10.0) -> Dict:
        """
        使用 CLIP 进行内容验证
        
        Args:
            target_video: 目标视频（原素材）
            source_video: 源视频
            start_time: 在源视频中的起始时间
            duration: 视频时长
            sample_interval: 采样间隔（秒）
        
        Returns:
            验证结果字典
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            print(f"\n使用 CLIP 验证 @{start_time}s...")
            print(f"  采样间隔: {sample_interval}s")
            
            # 采样时间点
            sample_times = np.arange(0, duration, sample_interval)
            
            similarities = []
            details = []
            
            print(f"  提取 {len(sample_times)} 个时间点...")
            
            for t in sample_times:
                if t >= duration:
                    break
                
                # 提取目标帧
                target_frame = self.temp_dir / f"target_{t:.0f}.jpg"
                self.extract_frame(target_video, t, target_frame)
                
                # 提取源帧
                source_frame = self.temp_dir / f"source_{t:.0f}.jpg"
                self.extract_frame(source_video, start_time + t, source_frame)
                
                if target_frame.exists() and source_frame.exists():
                    # 获取 CLIP 特征
                    target_features = self.get_image_features(target_frame)
                    source_features = self.get_image_features(source_frame)
                    
                    # 计算相似度
                    sim = self.calculate_similarity(target_features, source_features)
                    similarities.append(sim)
                    details.append({'time': t, 'score': sim})
                    
                    status = "✓" if sim > 0.8 else "✗" if sim < 0.6 else "~"
                    print(f"    @{t:3.0f}s: {sim:.1%} {status}")
            
            if not similarities:
                return {
                    'overall_score': 0,
                    'is_consistent': False,
                    'mean_score': 0,
                    'min_score': 0,
                    'max_score': 0,
                    'details': []
                }
            
            # 统计分析
            scores = np.array(similarities)
            mean_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            # CLIP 的阈值更严格
            # 因为 CLIP 真正理解内容，不像传统算法只看颜色
            is_consistent = (
                mean_score > 0.85 and  # 平均分要高
                min_score > 0.70       # 最低分不能太
            )
            
            return {
                'overall_score': mean_score,
                'is_consistent': is_consistent,
                'mean_score': mean_score,
                'min_score': min_score,
                'max_score': max_score,
                'details': details
            }
            
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def compare_positions(self, 
                         target_video: Path,
                         source_video: Path,
                         positions: List[float],
                         duration: float) -> Dict:
        """对比多个位置，找出最佳匹配"""
        results = {}
        
        print("=" * 60)
        print("CLIP 多位置对比")
        print("=" * 60)
        
        for pos in positions:
            result = self.validate_with_clip(target_video, source_video, pos, duration)
            results[pos] = result
            
            status = "✅" if result['is_consistent'] else "❌"
            print(f"\n{status} @{pos}s: 平均 {result['mean_score']:.1%}, "
                  f"最低 {result['min_score']:.1%}")
        
        # 找出最佳位置
        best_pos = max(results.keys(), key=lambda p: results[p]['overall_score'])
        
        print(f"\n{'='*60}")
        print("最佳匹配")
        print("=" * 60)
        print(f"  位置: @{best_pos}s")
        print(f"  评分: {results[best_pos]['overall_score']:.1%}")
        print(f"  通过: {results[best_pos]['is_consistent']}")
        
        return {
            'best_position': best_pos,
            'best_result': results[best_pos],
            'all_results': results
        }


def main():
    """测试 CLIP 验证器"""
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    
    validator = CLIPContentValidator()
    
    # 对比 @20s 和 @165s
    result = validator.compare_positions(
        target, source, 
        positions=[20, 165, 40, 100],  # 测试多个位置
        duration=217.7
    )
    
    print(f"\n{'='*60}")
    print("详细对比")
    print("=" * 60)
    for pos, res in result['all_results'].items():
        print(f"@{pos:3}s: {res['mean_score']:.1%} (min: {res['min_score']:.1%}) "
              f"{'✅' if res['is_consistent'] else '❌'}")


if __name__ == '__main__':
    main()
