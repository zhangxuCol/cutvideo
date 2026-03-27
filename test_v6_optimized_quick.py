#!/usr/bin/env python3
"""
快速测试脚本 - 只测试第一个视频的前30秒处理
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from pathlib import Path
import subprocess
import tempfile
import time
from video_reconstructor_hybrid_v6_optimized import VideoReconstructorHybridV6Optimized

# 路径配置
ADX_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
SOURCE_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
OUTPUT_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_optimized")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_duration(video_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

def create_short_test_video(input_path, output_path, duration=30):
    """创建测试用的短视频（前30秒）"""
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(input_path), '-t', str(duration),
        '-c', 'copy', str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def main():
    print("="*60)
    print("V6 Optimized 快速测试")
    print("="*60)
    
    # 获取第一个素材视频
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    target_videos = [f for f in ADX_DIR.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    target_videos.sort()
    
    if not target_videos:
        print("未找到素材视频")
        return
    
    target_video = target_videos[0]
    print(f"测试视频: {target_video.name}")
    print(f"原始时长: {get_duration(target_video):.1f}s")
    
    # 创建30秒测试版本
    test_video = OUTPUT_DIR / f"test_{target_video.name}"
    print(f"\n创建30秒测试版本...")
    if not create_short_test_video(target_video, test_video, 30):
        print("创建测试视频失败")
        return
    print(f"测试视频: {test_video}")
    print(f"测试时长: {get_duration(test_video):.1f}s")
    
    # 扫描源视频
    source_videos = [f for f in SOURCE_DIR.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    print(f"\n源视频数量: {len(source_videos)}")
    for s in source_videos:
        print(f"  - {s.name} ({get_duration(s):.1f}s)")
    
    # 测试不同验证级别
    validation_levels = ['strict', 'normal', 'loose']
    results = []
    
    for level in validation_levels:
        output_path = OUTPUT_DIR / f"test_{level}_{target_video.stem}_reconstructed.mp4"
        
        print(f"\n{'='*60}")
        print(f"测试验证级别: {level}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        reconstructor = VideoReconstructorHybridV6Optimized(
            target_video=str(test_video),
            source_videos=[str(s) for s in source_videos],
            config={
                'validation_level': level,
                'audio_match_threshold': 0.6,
                'single_source_threshold': 0.85
            }
        )
        
        try:
            segments, validation = reconstructor.reconstruct(str(output_path), use_target_audio=True)
            elapsed = time.time() - start_time
            
            result = {
                'level': level,
                'success': validation.success,
                'time': elapsed,
                'segments': len(segments),
                'match_rate': validation.match_rate,
                'duration_diff': validation.duration_diff,
                'reasons': validation.reasons,
                'output_exists': output_path.exists()
            }
            results.append(result)
            
            print(f"\n⏱️  处理时间: {elapsed:.1f}秒")
            print(f"✅ 验证通过: {validation.success}")
            print(f"📊 匹配率: {validation.match_rate:.1%}")
            print(f"📊 时长差异: {validation.duration_diff:.1%}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ 错误: {e}")
            results.append({
                'level': level,
                'success': False,
                'time': elapsed,
                'error': str(e)
            })
    
    # 输出对比报告
    print(f"\n{'='*60}")
    print("测试结果对比")
    print(f"{'='*60}")
    print(f"{'级别':<10} {'成功':<6} {'时间':<8} {'匹配率':<10} {'差异':<10} {'片段数'}")
    print("-"*60)
    for r in results:
        success = "✅" if r.get('success') else "❌"
        match_rate = f"{r.get('match_rate', 0):.1%}" if r.get('match_rate') is not None else "N/A"
        duration_diff = f"{r.get('duration_diff', 0):.1%}" if r.get('duration_diff') is not None else "N/A"
        print(f"{r['level']:<10} {success:<6} {r['time']:.1f}s    {match_rate:<10} {duration_diff:<10} {r.get('segments', 'N/A')}")
    
    # 清理测试文件
    print(f"\n清理测试文件...")
    test_video.unlink(missing_ok=True)
    
    print(f"\n✅ 测试完成")
    print(f"输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
