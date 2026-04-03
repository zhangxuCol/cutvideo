#!/usr/bin/env python3
"""
V6 并发处理版本 - 使用多进程同时处理多个视频
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v6_final import VideoReconstructorHybridV6
from pathlib import Path
import subprocess
from multiprocessing import Pool, cpu_count
import os

def get_video_duration(video_path: Path) -> float:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def process_single_video(args):
    """处理单个视频"""
    cut_video_path, source_paths, output_dir, config = args
    
    cut_video = Path(cut_video_path)
    output_path = output_dir / f"{cut_video.stem}_cut.mp4"
    
    # 如果已存在，跳过
    if output_path.exists():
        return {'name': cut_video.name, 'status': 'skipped', 'coverage': 0}
    
    try:
        reconstructor = VideoReconstructorHybridV6(
            str(cut_video),
            source_paths,
            config
        )
        
        segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
        
        if segments and output_path.exists():
            out_duration = get_video_duration(output_path)
            orig_duration = get_video_duration(cut_video)
            coverage = out_duration / orig_duration if orig_duration > 0 else 0
            return {'name': cut_video.name, 'status': 'success', 'coverage': coverage}
        else:
            return {'name': cut_video.name, 'status': 'failed', 'coverage': 0}
            
    except Exception as e:
        return {'name': cut_video.name, 'status': 'error', 'error': str(e), 'coverage': 0}

def main():
    cut_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
    source_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_concurrent")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source_videos = sorted(source_dir.glob("*.mp4"))
    source_paths = [str(sv) for sv in source_videos]
    
    cut_videos = sorted(cut_dir.glob("*.mp4"))
    
    # 确定并发数（CPU核心数 - 1，保留一个核心给系统）
    num_workers = max(1, cpu_count() - 1)
    print(f"🎬 V6 并发处理（{num_workers} 个进程）")
    print(f"   视频数: {len(cut_videos)}")
    print(f"   预计时间: 约 {len(cut_videos) * 5 / num_workers:.0f} 分钟\n")
    
    config = {
        'fps': 2,
        'similarity_threshold': 0.85,
        'match_threshold': 0.6,
        'audio_weight': 0.4,
        'video_weight': 0.6
    }
    
    # 准备参数
    args_list = [(str(cv), source_paths, output_dir, config) for cv in cut_videos]
    
    # 并发处理
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_video, args_list)
    
    # 统计结果
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"\n{'='*60}")
    print(f"📊 完成!")
    print(f"   成功: {success}")
    print(f"   失败: {failed}")
    print(f"   跳过: {skipped}")
    
    # 打印详细结果
    print(f"\n详细结果:")
    for r in results:
        status_icon = {'success': '✅', 'failed': '❌', 'skipped': '⏭️', 'error': '💥'}
        print(f"   {status_icon.get(r['status'], '?')} {r['name']}: {r['status']}")

if __name__ == '__main__':
    main()
