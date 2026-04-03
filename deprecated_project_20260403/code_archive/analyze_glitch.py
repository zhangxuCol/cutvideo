#!/usr/bin/env python3
"""
分析 115200 花屏问题 - 5分44秒处
"""

import subprocess
from pathlib import Path
import tempfile

target = Path('/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115200-1-363935820852768771.mp4')
output = Path('/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output/115200-1-363935820852768771_cut.mp4')

print('🔍 分析 115200 花屏问题')
print('='*60)

# 提取 5分40秒 - 5分50秒 的帧
temp_dir = Path(tempfile.mkdtemp())

print('\n📹 提取目标视频帧（原素材）:')
for t in [340, 342, 344, 345, 346, 348, 350]:
    frame = temp_dir / f'target_{t}s.jpg'
    subprocess.run(['ffmpeg', '-y', '-ss', str(t), '-i', str(target), '-vframes', '1', str(frame)], 
                   capture_output=True)
    status = "✓" if frame.exists() else "✗"
    print(f'   原素材 @{t}s: {status}')

print('\n📹 提取输出视频帧（二次裁剪）:')
for t in [340, 342, 344, 345, 346, 348, 350]:
    frame = temp_dir / f'output_{t}s.jpg'
    subprocess.run(['ffmpeg', '-y', '-ss', str(t), '-i', str(output), '-vframes', '1', str(frame)], 
                   capture_output=True)
    status = "✓" if frame.exists() else "✗"
    print(f'   输出 @{t}s: {status}')

# 检查输出视频的片段信息
print('\n📊 检查输出视频信息:')
cmd = ['ffprobe', '-v', 'error', '-show_entries', 'frame=pkt_pts_time,pict_type', 
       '-select_streams', 'v', '-of', 'csv', str(output)]
result = subprocess.run(cmd, capture_output=True, text=True)
frames = result.stdout.strip().split('\n')

# 找到 5分44秒附近的帧
print(f'   总帧数: {len(frames)}')
for i, frame in enumerate(frames):
    if '344' in frame or '345' in frame or '346' in frame:
        print(f'   帧 {i}: {frame}')

print('\n💡 分析完成')
print('   花屏原因可能是片段边界处帧不连续')
print('   建议: 使用重新编码而非直接复制')
