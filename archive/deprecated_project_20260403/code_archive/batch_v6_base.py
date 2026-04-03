#!/usr/bin/env python3
"""
V6 原版批量处理 - 目标5分钟/视频
使用最基础的 V6 算法，不做额外优化
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from pathlib import Path
import json
from datetime import datetime
import time
from video_reconstructor_hybrid_v6 import VideoReconstructorHybridV6

# 配置路径
ADX_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
SOURCE_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
OUTPUT_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base")
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

def process_single_video(target_video, source_videos, output_path):
    """处理单个视频"""
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"处理: {target_video.name}")
    print(f"{'='*60}")
    
    target_duration = get_duration(target_video)
    print(f"目标时长: {target_duration:.1f}s")
    
    # 使用 V6 原版
    reconstructor = VideoReconstructorHybridV6(
        target_video=str(target_video),
        source_videos=[str(s) for s in source_videos],
        config={
            'fps': 5,
            'similarity_threshold': 0.85,
            'match_threshold': 0.6,
            'audio_weight': 0.4,
            'video_weight': 0.6
        }
    )
    
    try:
        segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
        elapsed = time.time() - start_time
        
        return {
            "success": len(segments) > 0,
            "target": target_video.name,
            "segments": len(segments),
            "time": elapsed,
            "output_path": str(output_path) if segments else None
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "target": target_video.name,
            "error": str(e),
            "time": elapsed
        }

def batch_process():
    """批量处理"""
    print("="*60)
    print("V6 原版批量处理")
    print("目标: 5分钟/视频")
    print("="*60)
    
    # 扫描素材视频
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    target_videos = [
        f for f in ADX_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    target_videos.sort()
    
    # 扫描源视频
    source_videos = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    print(f"\n素材视频: {len(target_videos)} 个")
    print(f"源视频: {len(source_videos)} 个")
    
    # 批量处理
    results = []
    total = len(target_videos)
    
    for i, target_video in enumerate(target_videos, 1):
        output_path = OUTPUT_DIR / f"{target_video.stem}_reconstructed.mp4"
        
        # 跳过已处理的
        if output_path.exists():
            print(f"\n[{i}/{total}] {target_video.name} - 已存在，跳过")
            results.append({
                "target": target_video.name,
                "status": "skipped",
                "output_path": str(output_path)
            })
            continue
        
        # 处理
        result = process_single_video(target_video, source_videos, output_path)
        result["status"] = "success" if result.get("success") else "failed"
        results.append(result)
        
        # 显示时间
        print(f"⏱️  处理时间: {result['time']:.1f}秒 ({result['time']/60:.1f}分钟)")
        
        # 保存进度
        save_progress(results, i, total)
    
    # 生成报告
    generate_report(results)
    
    return results

def save_progress(results, current, total):
    """保存进度"""
    progress_file = LOG_DIR / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "current": current,
        "total": total,
        "results": results
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

def generate_report(results):
    """生成报告"""
    report_file = LOG_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    
    total_time = sum(r.get('time', 0) for r in results if 'time' in r)
    avg_time = total_time / (len(results) - skipped_count) if (len(results) - skipped_count) > 0 else 0
    
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("V6 原版批量处理报告\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计: {len(results)} 个视频\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {failed_count}\n")
        f.write(f"跳过: {skipped_count}\n")
        f.write(f"总时间: {total_time/60:.1f}分钟\n")
        f.write(f"平均时间: {avg_time/60:.1f}分钟/视频\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"\n{result['target']}\n")
            f.write(f"  状态: {result['status']}\n")
            if 'time' in result:
                f.write(f"  时间: {result['time']:.1f}秒\n")
            if result.get('segments'):
                f.write(f"  片段: {result['segments']}\n")
            if result.get('error'):
                f.write(f"  错误: {result['error']}\n")
    
    print(f"\n{'='*60}")
    print("处理完成!")
    print(f"{'='*60}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"跳过: {skipped_count}")
    print(f"平均时间: {avg_time/60:.1f}分钟/视频")
    print(f"报告: {report_file}")

if __name__ == "__main__":
    batch_process()
