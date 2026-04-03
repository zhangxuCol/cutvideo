#!/usr/bin/env python3
"""
验证脚本：测试视频重构是否能正确匹配 115196
正确位置：1.mp4 @40s
"""

import sys
import subprocess
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/04_comparison_validation')

def get_video_duration(video_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def verify_115196(reconstructor_class, config=None):
    """
    验证 115196 是否能正确匹配到 @40s 位置
    返回: (是否通过, 匹配位置, 综合评分)
    """
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    source = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集/1.mp4")
    
    target_duration = get_video_duration(target)
    
    # 创建重构器
    reconstructor = reconstructor_class(
        target_video=str(target),
        source_videos=[str(source)],
        config=config or {}
    )
    
    # 获取匹配结果
    temp_output = "/tmp/test_115196.mp4"
    segments = reconstructor.reconstruct(temp_output, use_target_audio=False)
    
    if not segments:
        return False, None, 0.0, "未找到匹配"
    
    # 检查是否是单源完整匹配
    if len(segments) == 1:
        seg = segments[0]
        start_time = seg.start_time
        score = seg.similarity_score
        
        # 检查是否在 @40s 附近（允许 ±5s 误差）
        if 35 <= start_time <= 45:
            return True, start_time, score, "正确匹配到 @40s 附近"
        else:
            return False, start_time, score, f"匹配到了错误位置 @{start_time:.0f}s，应该是 @40s"
    else:
        return False, None, 0.0, f"多源拼接 ({len(segments)} 段)，应该单源完整匹配"

if __name__ == '__main__':
    # 测试 V7
    try:
        from video_reconstructor_hybrid_v7 import VideoReconstructorHybridV7
        
        print("=" * 60)
        print("测试 V7 重构器")
        print("=" * 60)
        
        passed, start_time, score, message = verify_115196(VideoReconstructorHybridV7)
        
        print(f"\n结果:")
        print(f"  通过: {passed}")
        print(f"  匹配位置: @{start_time:.1f}s" if start_time else "  匹配位置: None")
        print(f"  评分: {score:.1%}" if score else "  评分: N/A")
        print(f"  详情: {message}")
        
        if passed:
            print("\n✅ 验证通过")
            sys.exit(0)
        else:
            print("\n❌ 验证失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
