#!/usr/bin/env python3
"""
V6 Optimized 批量视频重构系统
基于V6核心算法，添加可配置验证级别
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from pathlib import Path
import json
from datetime import datetime
from video_reconstructor_hybrid_v6_optimized import VideoReconstructorHybridV6Optimized, ValidationLevel

# 配置路径
ADX_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
SOURCE_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
OUTPUT_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_optimized")
LOG_DIR = OUTPUT_DIR / "logs"

# 确保目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_duration(video_path):
    """获取视频时长"""
    import subprocess
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

def process_single_video(target_video, source_videos, output_path, validation_level='normal'):
    """
    处理单个素材视频
    """
    print(f"\n{'='*60}")
    print(f"处理: {target_video.name}")
    print(f"{'='*60}")
    
    target_duration = get_duration(target_video)
    print(f"目标时长: {target_duration:.1f}s")
    print(f"验证级别: {validation_level}")
    
    # 创建重构器
    reconstructor = VideoReconstructorHybridV6Optimized(
        target_video=str(target_video),
        source_videos=[str(s) for s in source_videos],
        config={
            'validation_level': validation_level,
            'audio_match_threshold': 0.6,
            'single_source_threshold': 0.85
        }
    )
    
    try:
        # 执行重构
        segments, validation = reconstructor.reconstruct(str(output_path), use_target_audio=True)
        
        return {
            "success": validation.success,
            "target": target_video.name,
            "segments": len(segments),
            "match_rate": validation.match_rate,
            "duration_diff": validation.duration_diff,
            "reasons": validation.reasons,
            "suggestions": validation.suggestions,
            "output_path": str(output_path) if validation.success else None
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "target": target_video.name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def batch_process(validation_level='normal', limit=None):
    """批量处理所有素材视频"""
    
    print("="*60)
    print("V6 Optimized 批量视频重构系统")
    print("="*60)
    print(f"素材目录: {ADX_DIR}")
    print(f"源视频目录: {SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"验证级别: {validation_level}")
    
    # 扫描所有素材视频
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    target_videos = [
        f for f in ADX_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    target_videos.sort()
    
    if limit:
        target_videos = target_videos[:limit]
    
    print(f"\n发现 {len(target_videos)} 个素材视频")
    
    # 扫描源视频
    source_videos = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    print(f"发现 {len(source_videos)} 个源视频")
    
    # 批量处理
    results = []
    total = len(target_videos)
    
    for i, target_video in enumerate(target_videos, 1):
        print(f"\n\n[{i}/{total}] 处理: {target_video.name}")
        
        output_path = OUTPUT_DIR / f"{target_video.stem}_reconstructed.mp4"
        
        # 检查是否已处理过
        if output_path.exists():
            print(f"  ⚠️  输出文件已存在，跳过")
            results.append({
                "target": target_video.name,
                "status": "skipped",
                "output_path": str(output_path)
            })
            continue
        
        # 处理视频
        result = process_single_video(target_video, source_videos, output_path, validation_level)
        result["status"] = "success" if result.get("success") else "failed"
        results.append(result)
        
        # 保存中间结果
        save_progress(results, i, total)
    
    # 生成最终报告
    generate_report(results, validation_level)
    
    return results

def save_progress(results, current, total):
    """保存处理进度"""
    progress_file = LOG_DIR / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "current": current,
        "total": total,
        "results": results
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

def generate_report(results, validation_level):
    """生成处理报告"""
    report_file = LOG_DIR / f"report_{validation_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"V6 Optimized 批量处理报告 [{validation_level}]\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计处理: {len(results)} 个视频\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {failed_count}\n")
        f.write(f"跳过: {skipped_count}\n")
        f.write(f"成功率: {success_count/(len(results)-skipped_count)*100:.1f}%\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"\n素材: {result['target']}\n")
            f.write(f"状态: {result['status']}\n")
            
            if result.get("success"):
                f.write(f"片段数: {result.get('segments', 0)}\n")
                f.write(f"匹配率: {result.get('match_rate', 0):.1%}\n")
                f.write(f"时长差异: {result.get('duration_diff', 0):.1%}\n")
            elif "error" in result:
                f.write(f"错误: {result['error']}\n")
            
            if result.get('reasons'):
                f.write(f"原因: {', '.join(result['reasons'])}\n")
            if result.get('suggestions'):
                f.write(f"建议: {', '.join(result['suggestions'])}\n")
    
    print(f"\n{'='*60}")
    print("批量处理完成!")
    print(f"{'='*60}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"跳过: {skipped_count}")
    print(f"成功率: {success_count/(len(results)-skipped_count)*100:.1f}%")
    print(f"报告: {report_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='V6 Optimized 批量视频重构')
    parser.add_argument('--validation', type=str, default='normal',
                       choices=['strict', 'normal', 'loose', 'best_effort'],
                       help='验证级别 (默认: normal)')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理数量（用于测试）')
    
    args = parser.parse_args()
    
    batch_process(validation_level=args.validation, limit=args.limit)
