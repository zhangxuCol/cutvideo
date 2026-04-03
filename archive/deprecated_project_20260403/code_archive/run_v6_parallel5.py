#!/usr/bin/env python3
"""
V6 并发处理 5 个视频 - 独立进程方式
"""

import subprocess
import sys
from pathlib import Path
import time

def run_v6(video_path, output_path, log_file):
    """运行 V6 处理单个视频"""
    script = f'''
import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')
from video_reconstructor_hybrid_v6_final import VideoReconstructorHybridV6

sources = [
    '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4',
    '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/2.mp4',
    '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977611190734848_363977611040526336.mp4',
    '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977666966589440_363977611271213056.mp4',
    '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/363747886756540416_363977673945911296_363977667063844864.mp4',
]

config = {{'fps': 2, 'similarity_threshold': 0.85, 'match_threshold': 0.6, 'audio_weight': 0.4, 'video_weight': 0.6}}

reconstructor = VideoReconstructorHybridV6('{video_path}', sources, config)
segments = reconstructor.reconstruct('{output_path}', use_target_audio=True)
print('DONE: {Path(video_path).name}')
'''
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            [sys.executable, '-c', script],
            stdout=f,
            stderr=subprocess.STDOUT
        )
    return process

def main():
    cut_dir = Path('/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原')
    output_dir = Path('/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_parallel5')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cut_videos = sorted(cut_dir.glob('*.mp4'))
    
    print(f'🎬 V6 并发处理（5 个进程）')
    print(f'   总视频: {len(cut_videos)}')
    print(f'   并发数: 5')
    print(f'   预计批次: {(len(cut_videos) + 4) // 5} 批\n')
    
    # 分批处理，每批 5 个
    for batch_idx in range(0, len(cut_videos), 5):
        batch = cut_videos[batch_idx:batch_idx+5]
        print(f'批次 {batch_idx//5 + 1}: 处理 {len(batch)} 个视频')
        
        processes = []
        for i, video in enumerate(batch):
            output = output_dir / f'{video.stem}_cut.mp4'
            log = output_dir / f'{video.stem}.log'
            
            if output.exists():
                print(f'  跳过: {video.name}')
                continue
            
            print(f'  启动: {video.name}')
            p = run_v6(str(video), str(output), str(log))
            processes.append((video.name, p))
        
        if processes:
            print(f'  等待 {len(processes)} 个任务完成...')
            for name, p in processes:
                p.wait()
                print(f'    ✓ {name}')
        
        print()
    
    print('✅ 全部完成!')

if __name__ == '__main__':
    main()
