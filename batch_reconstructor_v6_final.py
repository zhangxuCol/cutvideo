#!/usr/bin/env python3
"""
批量视频重构工具 V6 - 最终版
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v6_final import VideoReconstructorHybridV6, load_config
from pathlib import Path
import json
import subprocess

def get_video_duration(video_path: Path) -> float:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def main():
    cut_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
    source_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取源视频
    source_videos = list(source_dir.glob("*.mp4"))
    source_paths = [str(sv) for sv in source_videos]
    
    print(f"🎬 V6 最终版批量处理")
    print(f"   源视频: {len(source_videos)} 个")
    
    # 获取裁剪视频
    cut_videos = sorted(cut_dir.glob("*.mp4"))
    print(f"   待处理: {len(cut_videos)} 个\n")
    
    config = {
        'fps': 2,
        'similarity_threshold': 0.85,
        'match_threshold': 0.6,
        'audio_weight': 0.4,
        'video_weight': 0.6
    }
    
    results = []
    success = 0
    perfect = 0
    
    for i, cut_video in enumerate(cut_videos, 1):
        print(f"[{i}/{len(cut_videos)}] 处理: {cut_video.name}")
        
        output_path = output_dir / f"{cut_video.stem}_cut.mp4"
        
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
                diff = abs(out_duration - orig_duration)
                coverage = out_duration / orig_duration if orig_duration > 0 else 0
                
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
                    'mode': 'single' if len(segments) == 1 else 'multi'
                }
                
                if is_success:
                    success += 1
                    if is_perfect:
                        perfect += 1
                    print(f"   ✅ 成功! 覆盖 {coverage:.1%}\n")
                else:
                    print(f"   ⚠️ 覆盖不足: {coverage:.1%}\n")
            else:
                result = {
                    'cut_video': str(cut_video),
                    'cut_video_name': cut_video.name,
                    'success': False,
                    'error': '未生成输出'
                }
                print(f"   ❌ 失败\n")
                
        except Exception as e:
            result = {
                'cut_video': str(cut_video),
                'cut_video_name': cut_video.name,
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ 错误: {e}\n")
        
        results.append(result)
    
    # 保存报告
    report = {
        'total': len(cut_videos),
        'success': success,
        'perfect': perfect,
        'success_rate': success / len(cut_videos) * 100,
        'perfect_rate': perfect / len(cut_videos) * 100,
        'results': results
    }
    
    report_path = output_dir / 'report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"{'='*60}")
    print(f"📊 完成: {success}/{len(cut_videos)} 成功 ({success/len(cut_videos)*100:.1f}%)")
    print(f"   完美: {perfect}")
    print(f"   报告: {report_path}")

if __name__ == '__main__':
    main()
