#!/usr/bin/env python3
"""
使用 MultiSourceVideoReconstructorV2 重新处理 115196
支持多源视频片段拼接
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from multi_source_reconstructor_v2 import MultiSourceVideoReconstructorV2, Config
from pathlib import Path
import json

def main():
    target = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4")
    sources_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
    output = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output/115196-1-363935819124715523_cut.mp4")
    
    # 获取所有源视频
    source_videos = list(sources_dir.glob("*.mp4"))
    
    print(f"🎬 使用 MultiSourceVideoReconstructorV2 处理 115196")
    print(f"   目标视频: {target.name}")
    print(f"   源视频数: {len(source_videos)}")
    print(f"   输出路径: {output}\n")
    
    # 创建配置
    config = Config()
    config.set('target_video', str(target))
    config.set('source_videos', [str(s) for s in source_videos])
    config.set('output_video', str(output))
    config.set('fps', 5)  # 降低帧率以加快处理
    config.set('similarity_threshold', 0.85)
    config.set('match_threshold', 0.45)
    config.set('use_target_audio', True)
    config.set('scale', '480:270')
    
    # 创建重构器
    reconstructor = MultiSourceVideoReconstructorV2(config)
    
    # 执行重构
    success = reconstructor.reconstruct_with_validation()
    
    if success:
        print(f"\n✅ 重构成功！")
        
        # 验证结果
        print(f"\n🔍 二次验证...")
        sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/04_comparison_validation')
        from compare_videos import compare_videos
        
        result = compare_videos(target, output, 0.90)
        
        if result['passed']:
            print(f"\n✅ 二次验证通过！")
        else:
            print(f"\n⚠️ 二次验证未通过 (相似度: {result['overall_similarity']:.1%})")
    else:
        print(f"\n❌ 重构失败")

if __name__ == '__main__':
    main()
