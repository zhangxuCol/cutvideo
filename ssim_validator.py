#!/usr/bin/env python3
"""
使用 SSIM (结构相似性指数) 进行内容验证

SSIM 比传统算法更能感知结构变化，对内容差异更敏感
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from PIL import Image

class SSIMValidator:
    """基于 SSIM 的图像内容验证器"""
    
    def __init__(self):
        print("初始化 SSIM 验证器...")
    
    def compare_frames(self, img1_path: Path, img2_path: Path) -> float:
        """计算两张图的 SSIM"""
        # 读取图片
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 统一尺寸
        gray1 = cv2.resize(gray1, (320, 180))
        gray2 = cv2.resize(gray2, (320, 180))
        
        # 计算 SSIM
        score, _ = ssim(gray1, gray2, full=True)
        
        return score
    
    def compare_with_visualization(self, img1_path: Path, img2_path: Path, output_path: Path = None):
        """对比并可视化差异"""
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        gray1 = cv2.resize(gray1, (320, 180))
        gray2 = cv2.resize(gray2, (320, 180))
        
        # 计算 SSIM 和差异图
        score, diff = ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")
        
        # 阈值化差异图
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # 找到轮廓
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在图片上标记差异区域
        img1_vis = cv2.resize(img1, (320, 180))
        img2_vis = cv2.resize(img2, (320, 180))
        
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img1_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img2_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 拼接显示
        result = np.hstack([img1_vis, img2_vis])
        
        if output_path:
            cv2.imwrite(str(output_path), result)
        
        return score
    
    def compare_all_frames(self, frame_dir: Path):
        """对比所有帧"""
        print("=" * 60)
        print("使用 SSIM 对比所有帧")
        print("=" * 60)
        
        compare_times = [0, 30, 60]
        
        results = {
            '20s': {'scores': [], 'frames': []},
            '165s': {'scores': [], 'frames': []}
        }
        
        output_dir = frame_dir / "ssim_comparison"
        output_dir.mkdir(exist_ok=True)
        
        for t in compare_times:
            target_frame = frame_dir / f"target_{t:03d}s.jpg"
            frame_20s = frame_dir / f"source_20s_{t:03d}s.jpg"
            frame_165s = frame_dir / f"source_165s_{t:03d}s.jpg"
            
            if not target_frame.exists():
                continue
            
            print(f"\n@{t}s:")
            
            # 对比 @20s
            if frame_20s.exists():
                score_20s = self.compare_frames(target_frame, frame_20s)
                results['20s']['scores'].append(score_20s)
                results['20s']['frames'].append(t)
                
                # 生成对比图
                vis_path = output_dir / f"compare_20s_{t:03d}s.jpg"
                self.compare_with_visualization(target_frame, frame_20s, vis_path)
                
                status = "✓" if score_20s > 0.7 else "✗" if score_20s < 0.5 else "~"
                print(f"  @20s:  {score_20s:.1%} {status} (差异图: {vis_path.name})")
            
            # 对比 @165s
            if frame_165s.exists():
                score_165s = self.compare_frames(target_frame, frame_165s)
                results['165s']['scores'].append(score_165s)
                results['165s']['frames'].append(t)
                
                # 生成对比图
                vis_path = output_dir / f"compare_165s_{t:03d}s.jpg"
                self.compare_with_visualization(target_frame, frame_165s, vis_path)
                
                status = "✓" if score_165s > 0.7 else "✗" if score_165s < 0.5 else "~"
                print(f"  @165s: {score_165s:.1%} {status} (差异图: {vis_path.name})")
        
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
        
        print(f"\n差异图已保存到: {output_dir}")
        
        return results


def main():
    """测试 SSIM 验证器"""
    validator = SSIMValidator()
    
    frame_dir = Path("/Users/zhangxu/work/项目/cutvideo/comparison_frames")
    
    if frame_dir.exists():
        validator.compare_all_frames(frame_dir)
    else:
        print("对比帧目录不存在")


if __name__ == '__main__':
    main()
