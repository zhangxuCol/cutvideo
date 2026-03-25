#!/usr/bin/env python3
"""
使用正确的匹配位置 (40s) 重新处理 115196
"""

import subprocess
from pathlib import Path
import shutil

def get_video_duration(video_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def main():
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    output = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output/115196-1-363935819124715523_cut.mp4")
    
    target_duration = get_video_duration(target)
    source_duration = get_video_duration(source)
    
    print(f"🎬 重新处理 115196")
    print(f"   原素材: {target_duration:.1f}s")
    print(f"   源视频: {source_duration:.1f}s")
    print(f"   匹配位置: @ 40s")
    
    # 从 @40s 开始截取
    start_time = 40
    
    # 确保不超出源视频范围
    if start_time + target_duration > source_duration:
        actual_duration = source_duration - start_time - 0.1
    else:
        actual_duration = target_duration
    
    print(f"   截取: {start_time}s ~ {start_time + actual_duration:.1f}s")
    
    # 截取视频
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', str(start_time),
        '-t', str(actual_duration),
        '-i', str(source),
        '-c', 'copy',
        str(output)
    ]
    
    subprocess.run(cmd, capture_output=True)
    
    if output.exists():
        output_duration = get_video_duration(output)
        print(f"   ✅ 生成成功: {output_duration:.1f}s")
        print(f"   输出: {output}")
        
        # 验证
        print(f"\n🔍 验证结果...")
        import sys
        sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/04_comparison_validation')
        from compare_videos import compare_videos
        
        result = compare_videos(target, output, 0.90)
        
        if result['passed']:
            print(f"\n✅ 验证通过！")
        else:
            print(f"\n⚠️ 验证结果: 相似度 {result['overall_similarity']:.1%}")
            
    else:
        print(f"   ❌ 生成失败")

if __name__ == '__main__':
    main()
