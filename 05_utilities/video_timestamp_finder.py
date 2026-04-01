#!/usr/bin/env python3
"""
视频时间戳定位工具
通过比对已裁剪视频和原视频，自动找到裁剪的时间戳位置
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil

class VideoTimestampFinder:
    def __init__(self, original_video, clipped_video):
        self.original_video = Path(original_video)
        self.clipped_video = Path(clipped_video)
        self.temp_dir = None
        
    def extract_frames(self, video_path, output_dir, fps=1):
        """提取视频帧，每秒1帧"""
        output_pattern = output_dir / f"{video_path.stem}_%04d.jpg"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-i', str(video_path),
            '-vf', f'fps={fps},scale=320:240',
            '-q:v', '2',
            str(output_pattern)
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        # 获取提取的帧列表
        frames = sorted(output_dir.glob(f"{video_path.stem}_*.jpg"))
        return frames
    
    def calculate_similarity(self, frame1_path, frame2_path):
        """计算两帧的相似度"""
        img1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 使用结构相似度(SSIM)或直方图比对
        # 这里用简单的直方图比对，速度快
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, similarity)  # 确保非负
    
    def find_timestamp(self):
        """找到裁剪视频在原视频中的时间戳"""
        
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        original_frames_dir = self.temp_dir / "original"
        clipped_frames_dir = self.temp_dir / "clipped"
        original_frames_dir.mkdir()
        clipped_frames_dir.mkdir()
        
        try:
            print("🎬 正在提取原视频帧...")
            original_frames = self.extract_frames(self.original_video, original_frames_dir)
            print(f"   提取了 {len(original_frames)} 帧")
            
            print("🎬 正在提取裁剪视频帧...")
            clipped_frames = self.extract_frames(self.clipped_video, clipped_frames_dir)
            print(f"   提取了 {len(clipped_frames)} 帧")
            
            if len(clipped_frames) == 0:
                print("❌ 裁剪视频帧提取失败")
                return None
            
            # 用裁剪视频的第一帧和最后一帧去原视频中找匹配
            print("🔍 正在比对查找时间戳...")
            
            first_clipped_frame = clipped_frames[0]
            last_clipped_frame = clipped_frames[-1]
            
            # 找第一帧的匹配位置
            best_start_idx = 0
            best_start_score = 0
            
            for i, orig_frame in enumerate(original_frames):
                similarity = self.calculate_similarity(first_clipped_frame, orig_frame)
                if similarity > best_start_score:
                    best_start_score = similarity
                    best_start_idx = i
            
            print(f"   首帧匹配: 第{best_start_idx}秒, 相似度{best_start_score:.2%}")
            
            # 找最后一帧的匹配位置（从首帧位置之后找）
            expected_end_idx = best_start_idx + len(clipped_frames) - 1
            search_range = range(
                max(0, expected_end_idx - 5),
                min(len(original_frames), expected_end_idx + 5)
            )
            
            best_end_idx = expected_end_idx
            best_end_score = 0
            
            for i in search_range:
                if i >= len(original_frames):
                    break
                similarity = self.calculate_similarity(last_clipped_frame, original_frames[i])
                if similarity > best_end_score:
                    best_end_score = similarity
                    best_end_idx = i
            
            print(f"   尾帧匹配: 第{best_end_idx}秒, 相似度{best_end_score:.2%}")
            
            # 转换为时间戳
            start_time = best_start_idx
            end_time = best_end_idx + 1  # +1因为帧是每秒提取的
            
            # 格式化时间
            start_formatted = f"{start_time//60:02d}:{start_time%60:02d}"
            end_formatted = f"{end_time//60:02d}:{end_time%60:02d}"
            
            print(f"\n✅ 找到时间戳: {start_formatted} - {end_formatted}")
            print(f"   裁剪时长: {end_time - start_time}秒")
            
            return {
                'start_seconds': start_time,
                'end_seconds': end_time,
                'start_formatted': start_formatted,
                'end_formatted': end_formatted,
                'duration': end_time - start_time,
                'start_similarity': best_start_score,
                'end_similarity': best_end_score
            }
            
        finally:
            # 清理临时目录
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def generate_ffmpeg_command(self, timestamp_info, output_path=None):
        """生成FFmpeg裁剪命令"""
        if output_path is None:
            output_path = f"新裁剪_{timestamp_info['start_formatted']}_{timestamp_info['end_formatted']}.mp4"
        
        cmd = (
            f'ffmpeg -y -hide_banner -i "{self.original_video}" '
            f'-ss {timestamp_info["start_formatted"]} '
            f'-to {timestamp_info["end_formatted"]} '
            f'-c copy "{output_path}"'
        )
        
        return cmd


def main():
    # 使用示例
    print("=" * 50)
    print("视频时间戳定位工具")
    print("=" * 50)
    
    # 请修改为你的视频路径
    original_video = input("请输入原视频路径: ").strip().strip('"')
    clipped_video = input("请输入已裁剪视频路径: ").strip().strip('"')
    
    if not Path(original_video).exists():
        print(f"❌ 原视频不存在: {original_video}")
        return
    
    if not Path(clipped_video).exists():
        print(f"❌ 裁剪视频不存在: {clipped_video}")
        return
    
    finder = VideoTimestampFinder(original_video, clipped_video)
    result = finder.find_timestamp()
    
    if result:
        print("\n" + "=" * 50)
        print("📝 FFmpeg裁剪命令:")
        print("=" * 50)
        cmd = finder.generate_ffmpeg_command(result)
        print(cmd)
        print("\n复制上面的命令即可裁剪视频")


if __name__ == "__main__":
    main()
