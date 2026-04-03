#!/usr/bin/env python3
"""
V6 批量处理所有原素材
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v6_final import VideoReconstructorHybridV6
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
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取源视频
    source_videos = sorted(source_dir.glob("*.mp4"))
    source_paths = [str(sv) for sv in source_videos]
    
    print(f"🎬 V6 批量处理")
    print(f"   源视频: {len(source_videos)} 个")
    print(f"   输出目录: {output_dir}\n")
    
    # 获取所有原素材
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
    failed = 0
    
    for i, cut_video in enumerate(cut_videos, 1):
        # 输出文件名: 原素材名_cut.mp4
        output_path = output_dir / f"{cut_video.stem}_cut.mp4"
        
        print(f"[{i}/{len(cut_videos)}] {cut_video.name} → {output_path.name}")
        
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
                
                if is_success:
                    success += 1
                    print(f"   ✅ 成功! 覆盖 {coverage:.1%}, 片段数 {len(segments)}\n")
                else:
                    failed += 1
                    print(f"   ⚠️ 覆盖不足: {coverage:.1%}\n")
                
                results.append({
                    'cut_video': cut_video.name,
                    'output': output_path.name,
                    'segments': len(segments),
                    'coverage': coverage,
                    'success': is_success
                })
            else:
                failed += 1
                print(f"   ❌ 失败: 未生成输出\n")
                results.append({
                    'cut_video': cut_video.name,
                    'success': False,
                    'error': '未生成输出'
                })
                
        except Exception as e:
            failed += 1
            print(f"   ❌ 错误: {e}\n")
            results.append({
                'cut_video': cut_video.name,
                'success': False,
                'error': str(e)
            })
    
    # 保存报告
    report = {
        'total': len(cut_videos),
        'success': success,
        'failed': failed,
        'results': results
    }
    
    report_path = output_dir / 'v6_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"{'='*60}")
    print(f"📊 完成!")
    print(f"   成功: {success}/{len(cut_videos)}")
    print(f"   失败: {failed}")
    print(f"   报告: {report_path}")

if __name__ == '__main__':
    main()
