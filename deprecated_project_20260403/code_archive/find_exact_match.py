#!/usr/bin/env python3
"""
精确画面匹配查找器 - 遍历所有原片找到正确匹配
针对失败的检查点：75s 和 200s
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile

class FrameMatcher:
    def __init__(self, target_video: str, source_dir: str):
        self.target_video = Path(target_video)
        self.source_dir = Path(source_dir)
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def extract_frame(self, video_path: Path, time_sec: float) -> np.ndarray:
        """提取指定时间的帧"""
        frame_path = self.temp_dir / f"frame_{video_path.stem}_{time_sec:.1f}.jpg"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-q:v', '2',
            str(frame_path)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            return img
        return None
    
    def calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算两帧的相似度"""
        if img1 is None or img2 is None:
            return 0.0
        
        # 统一尺寸
        img1 = cv2.resize(img1, (320, 180))
        img2 = cv2.resize(img2, (320, 180))
        
        # 灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 直方图相似度
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        # SSIM
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(gray1, gray2)
        
        # 综合评分
        return 0.3 * max(0, hist_sim) + 0.3 * template_sim + 0.4 * ssim_score
    
    def find_match(self, target_time: float, search_range: int = 10):
        """在指定时间点附近搜索最佳匹配"""
        print(f"\n{'='*70}")
        print(f"🔍 查找目标时间点 {target_time}s 的最佳匹配")
        print(f"{'='*70}")
        
        # 提取目标帧
        target_frame = self.extract_frame(self.target_video, target_time)
        if target_frame is None:
            print(f"❌ 无法提取目标视频在 {target_time}s 的帧")
            return None
        
        print(f"✅ 已提取目标帧")
        
        # 获取所有源视频
        source_videos = list(self.source_dir.glob("*.mp4"))
        print(f"📹 找到 {len(source_videos)} 个源视频")
        
        best_match = None
        best_score = 0
        
        for source in source_videos:
            print(f"\n  检查源视频: {source.name}")
            
            # 获取源视频时长
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', str(source)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            
            # 在目标时间附近搜索
            search_start = max(0, target_time - search_range)
            search_end = min(duration, target_time + search_range)
            
            print(f"    搜索范围: {search_start:.1f}s - {search_end:.1f}s")
            
            source_best_time = 0
            source_best_score = 0
            
            # 每秒检查一帧
            for t in np.arange(search_start, search_end, 1.0):
                source_frame = self.extract_frame(source, t)
                if source_frame is not None:
                    score = self.calculate_similarity(target_frame, source_frame)
                    if score > source_best_score:
                        source_best_score = score
                        source_best_time = t
            
            print(f"    最佳匹配: {source_best_time:.1f}s, 相似度: {source_best_score:.3f}")
            
            if source_best_score > best_score:
                best_score = source_best_score
                best_match = {
                    'source': source,
                    'time': source_best_time,
                    'score': best_score
                }
        
        print(f"\n{'='*70}")
        print(f"🏆 全局最佳匹配:")
        print(f"   源视频: {best_match['source'].name}")
        print(f"   时间点: {best_match['time']:.1f}s")
        print(f"   相似度: {best_match['score']:.3f}")
        print(f"{'='*70}")
        
        return best_match


def main():
    target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"
    
    matcher = FrameMatcher(target, source_dir)
    
    # 检查失败的两个时间点
    print("\n" + "="*70)
    print("开始精确画面匹配查找")
    print("="*70)
    
    # 75秒（1分15秒）
    match_75 = matcher.find_match(75.0, search_range=30)
    
    # 200秒（3分20秒）
    match_200 = matcher.find_match(200.0, search_range=30)
    
    print("\n" + "="*70)
    print("查找完成，建议更新特殊匹配表:")
    print("="*70)
    if match_75:
        seg_15 = int(75 / 5)  # 段索引
        print(f"  段 {seg_15} (75s): ('{match_75['source']}', {match_75['time']:.0f})")
    if match_200:
        seg_40 = int(200 / 5)  # 段索引
        print(f"  段 {seg_40} (200s): ('{match_200['source']}', {match_200['time']:.0f})")


if __name__ == "__main__":
    main()
