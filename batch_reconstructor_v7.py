#!/usr/bin/env python3
"""
批量视频重构工具 V7 - 性能优化版
输出原裁剪视频和二次裁剪视频的对比报告
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v7 import VideoReconstructorHybridV7
from pathlib import Path
import json
import subprocess
import time

def get_video_duration(video_path: Path) -> float:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def get_video_frame_count(video_path: Path) -> int:
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
           '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1',
           str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except:
        return 0

def compare_videos(original_path: Path, output_path: Path) -> dict:
    """对比原裁剪视频和二次裁剪视频"""
    orig_duration = get_video_duration(original_path)
    output_duration = get_video_duration(output_path)
    
    orig_frames = get_video_frame_count(original_path)
    output_frames = get_video_frame_count(output_path)
    
    return {
        'original_duration': orig_duration,
        'output_duration': output_duration,
        'duration_diff': orig_duration - output_duration,
        'duration_diff_pct': (orig_duration - output_duration) / orig_duration * 100 if orig_duration > 0 else 0,
        'original_frames': orig_frames,
        'output_frames': output_frames,
        'frame_diff': orig_frames - output_frames if orig_frames > 0 and output_frames > 0 else 0,
        'frame_diff_pct': (orig_frames - output_frames) / orig_frames * 100 if orig_frames > 0 else 0
    }

def main():
    cut_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
    source_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v7")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取源视频
    source_videos = list(source_dir.glob("*.mp4"))
    source_paths = [str(sv) for sv in source_videos]
    
    print(f"🎬 V7 批量处理 - 音频预筛选 + 动态帧间隔 + 并行")
    print(f"   源视频: {len(source_videos)} 个")
    
    # 获取裁剪视频
    cut_videos = sorted(cut_dir.glob("*.mp4"))
    print(f"   待处理: {len(cut_videos)} 个\n")
    
    config = {
        'fps': 2,
        'similarity_threshold': 0.85,
        'match_threshold': 0.4,
        'audio_weight': 0.4,
        'video_weight': 0.6,
        'max_workers': 8,
        'audio_sample_rate': 16000,
        'coarse_interval': 2.0,
        'fine_interval': 1.0
    }
    
    results = []
    success = 0
    perfect = 0
    total_start_time = time.time()
    
    for i, cut_video in enumerate(cut_videos, 1):
        print(f"[{i}/{len(cut_videos)}] 处理: {cut_video.name}")
        
        output_path = output_dir / f"{cut_video.stem}_cut.mp4"
        
        try:
            start_time = time.time()
            
            reconstructor = VideoReconstructorHybridV7(
                str(cut_video),
                source_paths,
                config
            )
            
            segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
            
            elapsed = time.time() - start_time
            
            if segments and output_path.exists():
                out_duration = get_video_duration(output_path)
                orig_duration = get_video_duration(cut_video)
                diff = abs(out_duration - orig_duration)
                coverage = out_duration / orig_duration if orig_duration > 0 else 0
                
                # 对比原裁剪和二次裁剪
                comparison = compare_videos(cut_video, output_path)
                
                is_success = coverage > 0.9
                is_perfect = coverage > 0.95 and len(segments) == 1
                
                result = {
                    'cut_video': str(cut_video),
                    'cut_video_name': cut_video.name,
                    'output_video': str(output_path),
                    'cut_duration': orig_duration,
                    'output_duration': out_duration,
                    'duration_diff': diff,
                    'segments': len(segments),
                    'success': is_success,
                    'is_perfect': is_perfect,
                    'mode': 'single' if len(segments) == 1 else 'multi',
                    'processing_time': elapsed,
                    'comparison': comparison
                }
                
                if is_success:
                    success += 1
                    if is_perfect:
                        perfect += 1
                    print(f"   ✅ 成功! 覆盖 {coverage:.1%}, 耗时 {elapsed:.1f}s")
                    print(f"      丢帧: {comparison.get('frame_diff', 0)} 帧 ({comparison.get('frame_diff_pct', 0):.1f}%)")
                else:
                    print(f"   ⚠️ 覆盖不足: {coverage:.1%}")
            else:
                result = {
                    'cut_video': str(cut_video),
                    'cut_video_name': cut_video.name,
                    'success': False,
                    'error': '未生成输出',
                    'processing_time': elapsed
                }
                print(f"   ❌ 失败")
            
            results.append(result)
            print()
            
        except Exception as e:
            result = {
                'cut_video': str(cut_video),
                'cut_video_name': cut_video.name,
                'success': False,
                'error': str(e)
            }
            results.append(result)
            print(f"   ❌ 异常: {e}\n")
    
    total_elapsed = time.time() - total_start_time
    
    # 生成报告
    report = {
        'version': 'V7',
        'config': config,
        'total': len(cut_videos),
        'success': success,
        'perfect': perfect,
        'success_rate': success / len(cut_videos) * 100 if cut_videos else 0,
        'perfect_rate': perfect / len(cut_videos) * 100 if cut_videos else 0,
        'total_processing_time': total_elapsed,
        'avg_processing_time': total_elapsed / len(cut_videos) if cut_videos else 0,
        'results': results
    }
    
    report_path = output_dir / 'report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 V7 处理完成")
    print(f"{'='*60}")
    print(f"   总计: {len(cut_videos)} 个视频")
    print(f"   成功: {success} ({report['success_rate']:.1f}%)")
    print(f"   完美: {perfect} ({report['perfect_rate']:.1f}%)")
    print(f"   总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")
    print(f"   平均: {report['avg_processing_time']:.1f}s/视频")
    print(f"   报告: {report_path}")

if __name__ == '__main__':
    main()
