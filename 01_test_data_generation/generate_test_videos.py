#!/usr/bin/env python3
"""
生成测试视频用于演示视频重构脚本
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_video(output_path, duration=10, fps=30, width=640, height=480, pattern='color_bars'):
    """生成测试视频"""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        if pattern == 'color_bars':
            # 彩条测试图
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            bar_width = width // 8
            colors = [
                (255, 255, 255),  # 白
                (255, 255, 0),    # 黄
                (0, 255, 255),    # 青
                (0, 255, 0),      # 绿
                (255, 0, 255),    # 紫
                (255, 0, 0),      # 红
                (0, 0, 255),      # 蓝
                (0, 0, 0),        # 黑
            ]
            for i, color in enumerate(colors):
                frame[:, i*bar_width:(i+1)*bar_width] = color
                
        elif pattern == 'gradient':
            # 渐变图
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                intensity = int(255 * (y / height))
                frame[y, :] = [intensity, intensity//2, 255-intensity]
                
        elif pattern == 'checkerboard':
            # 棋盘格
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            square_size = 40
            for y in range(0, height, square_size):
                for x in range(0, width, square_size):
                    if ((y // square_size) + (x // square_size)) % 2 == 0:
                        frame[y:y+square_size, x:x+square_size] = (255, 255, 255)
                    else:
                        frame[y:y+square_size, x:x+square_size] = (0, 0, 0)
                        
        elif pattern == 'moving_circle':
            # 移动圆形
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            center_x = int(width * (frame_num / total_frames))
            center_y = height // 2
            radius = 50
            color = (0, 255, 0)
            cv2.circle(frame, (center_x, center_y), radius, color, -1)
            
        # 添加帧编号
        cv2.putText(frame, f'Frame {frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Time {frame_num/fps:.2f}s', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ 已生成测试视频: {output_path}")
    print(f"   时长: {duration}s, 分辨率: {width}x{height}, FPS: {fps}")


def create_clip_from_source(source_path, output_path, start_time, end_time):
    """从源视频裁剪片段"""
    
    cap = cv2.VideoCapture(source_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"✅ 已裁剪视频: {output_path}")
    print(f"   来源: {source_path}")
    print(f"   时间段: {start_time}s - {end_time}s")


def main():
    print("="*60)
    print("生成测试视频")
    print("="*60)
    
    # 创建测试视频目录
    test_dir = Path("test_videos")
    test_dir.mkdir(exist_ok=True)
    
    # 1. 生成3个源视频（各30秒，不同图案）
    print("\n1. 生成源视频...")
    create_test_video(test_dir / "source_A.mp4", duration=30, pattern='color_bars')
    create_test_video(test_dir / "source_B.mp4", duration=30, pattern='gradient')
    create_test_video(test_dir / "source_C.mp4", duration=30, pattern='checkerboard')
    
    # 2. 生成裁剪视频1（混剪：A的前10秒 + B的10-20秒 + C的15-25秒）
    print("\n2. 生成裁剪视频1（混剪）...")
    
    # 从A裁剪0-10秒
    create_clip_from_source(test_dir / "source_A.mp4", test_dir / "temp1.mp4", 0, 10)
    # 从B裁剪10-20秒
    create_clip_from_source(test_dir / "source_B.mp4", test_dir / "temp2.mp4", 10, 20)
    # 从C裁剪15-25秒
    create_clip_from_source(test_dir / "source_C.mp4", test_dir / "temp3.mp4", 15, 25)
    
    # 合并片段（简单复制文件方式）
    print("\n3. 合并片段生成 clip1.mp4...")
    
    # 读取并合并
    cap1 = cv2.VideoCapture(str(test_dir / "temp1.mp4"))
    cap2 = cv2.VideoCapture(str(test_dir / "temp2.mp4"))
    cap3 = cv2.VideoCapture(str(test_dir / "temp3.mp4"))
    
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(test_dir / "clip1.mp4"), fourcc, fps, (width, height))
    
    for cap in [cap1, cap2, cap3]:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    
    out.release()
    
    # 清理临时文件
    for f in ["temp1.mp4", "temp2.mp4", "temp3.mp4"]:
        (test_dir / f).unlink()
    
    print(f"✅ 已生成 clip1.mp4 (时长约30秒)")
    
    # 4. 更新配置文件
    print("\n4. 更新配置文件...")
    config_content = f"""# 视频重构测试配置

# 裁剪视频1路径（参考视频）
target_video: "{test_dir.absolute()}/clip1.mp4"

# 原视频路径列表（素材库）
source_videos:
  - "{test_dir.absolute()}/source_A.mp4"
  - "{test_dir.absolute()}/source_B.mp4"
  - "{test_dir.absolute()}/source_C.mp4"

# 裁剪视频2输出路径
output_video: "{test_dir.absolute()}/clip2_reconstructed.mp4"

# 每秒提取帧数（1-10，建议5）
fps: 5

# 相似度阈值（0-1，建议0.90）
similarity_threshold: 0.90

# 最大重试次数
max_retries: 3

# 是否使用目标视频的音频
use_target_audio: true

# 最小片段时长（秒）
min_segment_duration: 0.5

# 单帧匹配阈值
match_threshold: 0.6

# 提取帧分辨率
scale: "480:270"

# 缺失片段处理策略
missing_segment_strategy: "smart_fill"

# 智能填充的最低相似度阈值
smart_fill_threshold: 0.5
"""
    
    with open("video_reconstruct_config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ 配置文件已更新")
    
    print("\n" + "="*60)
    print("测试视频生成完成！")
    print("="*60)
    print(f"\n文件位置: {test_dir.absolute()}/")
    print("\n生成的文件:")
    print("  - source_A.mp4 (彩条图案)")
    print("  - source_B.mp4 (渐变图案)")
    print("  - source_C.mp4 (棋盘图案)")
    print("  - clip1.mp4 (混剪视频，由A/B/C各10秒组成)")
    print("\n现在可以运行: python3 multi_source_reconstructor_config.py")


if __name__ == "__main__":
    main()