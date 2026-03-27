#!/usr/bin/env python3
"""
V7 批量视频重构系统
自动处理 adx原 目录下的所有素材视频
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from pathlib import Path
import subprocess
import json
from datetime import datetime
from video_reconstructor_hybrid_v7 import VideoReconstructorHybridV7

# 配置路径
ADX_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
SOURCE_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
OUTPUT_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v7")
LOG_DIR = OUTPUT_DIR / "logs"

# 确保目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_duration(video_path):
    """获取视频时长"""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

def compare_videos(video1, video2):
    """
    对比两个视频的内容一致性
    返回相似度分数和详细信息
    """
    # 1. 提取音频指纹对比
    # 2. 提取关键帧对比（SSIM）
    # 3. 综合评分
    
    duration1 = get_duration(video1)
    duration2 = get_duration(video2)
    
    # 时长差异检查
    duration_diff = abs(duration1 - duration2)
    if duration_diff > 1.0:  # 超过1秒差异
        return {
            "match": False,
            "score": 0,
            "reason": f"时长不匹配: {duration1:.1f}s vs {duration2:.1f}s",
            "duration_diff": duration_diff
        }
    
    # TODO: 实现音频和视频的详细对比
    # 暂时返回基础信息
    return {
        "match": True,
        "score": 100,
        "reason": "基础检查通过（需要完善详细对比）",
        "duration_diff": duration_diff
    }

def process_single_video(target_video, source_videos, output_path):
    """
    处理单个素材视频
    返回处理结果和验证信息
    """
    print(f"\n{'='*60}")
    print(f"处理: {target_video.name}")
    print(f"{'='*60}")
    
    target_duration = get_duration(target_video)
    print(f"目标时长: {target_duration:.1f}s")
    
    # 创建重构器
    reconstructor = VideoReconstructorHybridV7(
        target_video=str(target_video),
        source_videos=[str(SOURCE_DIR)],  # 传入目录，自动扫描
        config={
            'audio_match_threshold': 0.6,
            'video_verify_threshold': 0.6
        }
    )
    
    try:
        # 执行重构
        segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
        
        if not segments:
            return {
                "success": False,
                "error": "未找到匹配片段",
                "segments": 0,
                "coverage": 0
            }
        
        # 计算覆盖率
        total_covered = sum(seg.end_time - seg.start_time for seg in segments)
        coverage = total_covered / target_duration if target_duration > 0 else 0
        
        print(f"\n重构完成:")
        print(f"  片段数: {len(segments)}")
        print(f"  覆盖率: {coverage:.1%}")
        print(f"  输出: {output_path}")
        
        # 验证输出质量
        print(f"\n验证输出质量...")
        verification = compare_videos(target_video, output_path)
        
        return {
            "success": True,
            "segments": len(segments),
            "coverage": coverage,
            "verification": verification,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "segments": 0,
            "coverage": 0
        }

def batch_process():
    """批量处理所有素材视频"""
    
    print("="*60)
    print("V7 批量视频重构系统")
    print("="*60)
    print(f"素材目录: {ADX_DIR}")
    print(f"源视频目录: {SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 扫描所有素材视频
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    target_videos = [
        f for f in ADX_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    target_videos.sort()
    
    print(f"\n发现 {len(target_videos)} 个素材视频")
    
    # 批量处理
    results = []
    total = len(target_videos)
    
    for i, target_video in enumerate(target_videos, 1):
        print(f"\n\n[{i}/{total}] 处理: {target_video.name}")
        
        output_path = OUTPUT_DIR / f"{target_video.stem}_reconstructed.mp4"
        
        # 检查是否已处理过
        if output_path.exists():
            print(f"  ⚠️  输出文件已存在，跳过")
            # 仍然验证
            verification = compare_videos(target_video, output_path)
            results.append({
                "target": target_video.name,
                "status": "skipped",
                "verification": verification
            })
            continue
        
        # 处理视频
        result = process_single_video(target_video, SOURCE_DIR, output_path)
        result["target"] = target_video.name
        result["status"] = "success" if result["success"] else "failed"
        results.append(result)
        
        # 保存中间结果
        save_progress(results, i, total)
    
    # 生成最终报告
    generate_report(results)
    
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

def generate_report(results):
    """生成处理报告"""
    report_file = LOG_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("V7 批量处理报告\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计处理: {len(results)} 个视频\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {failed_count}\n")
        f.write(f"跳过: {skipped_count}\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"\n素材: {result['target']}\n")
            f.write(f"状态: {result['status']}\n")
            
            if result.get("success"):
                f.write(f"片段数: {result.get('segments', 0)}\n")
                f.write(f"覆盖率: {result.get('coverage', 0):.1%}\n")
            elif "error" in result:
                f.write(f"错误: {result['error']}\n")
            
            if "verification" in result:
                v = result["verification"]
                f.write(f"验证: {v.get('reason', 'N/A')}\n")
                f.write(f"评分: {v.get('score', 0)}\n")
    
    print(f"\n{'='*60}")
    print("批量处理完成!")
    print(f"{'='*60}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"跳过: {skipped_count}")
    print(f"报告: {report_file}")

if __name__ == "__main__":
    batch_process()
