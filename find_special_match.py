#!/usr/bin/env python3
"""
查找特殊段的正确时间偏移
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo')

from pathlib import Path
from v6_fast import FastHighPrecisionReconstructor

# 目标视频
target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"

# 源视频列表
source_videos = [str(f) for f in Path(source_dir).iterdir() if f.suffix.lower() == '.mp4']

# 创建重构器
cache = "/Users/zhangxu/work/项目/cutvideo/cache"
Path(cache).mkdir(exist_ok=True)

reconstructor = FastHighPrecisionReconstructor(target, source_videos, cache)

# 预计算指纹
reconstructor.precompute_fingerprints()

# 搜索165s和185s的最佳匹配
print("\n" + "="*70)
print("搜索165s的最佳匹配...")
print("="*70)
for source_path in source_videos:
    if "363977673945911296" in source_path:
        print(f"\n检查源视频: {Path(source_path).name}")
        source_fp = reconstructor.source_fingerprints[Path(source_path)]
        
        # 目标95s的指纹 (95/2 = 47.5, 取48)
        target_idx = int(95 / 2)
        target_segment = reconstructor.target_fingerprint[target_idx:target_idx+3]  # 95-100s
        
        print(f"  目标95s指纹索引: {target_idx}")
        print(f"  源视频指纹长度: {len(source_fp)}")
        
        # 搜索最佳匹配
        best_score = 0
        best_start = 0
        for start in range(0, len(source_fp) - len(target_segment)):
            end = start + len(target_segment)
            source_segment = source_fp[start:end]
            
            correlations = []
            for t, s in zip(target_segment, source_segment):
                if len(t) == len(s) and len(t) > 0:
                    import numpy as np
                    if np.std(t) > 0 and np.std(s) > 0:
                        corr = np.corrcoef(t, s)[0, 1]
                        correlations.append(corr)
            
            score = sum(correlations) / len(correlations) if correlations else 0
            if score > best_score:
                best_score = score
                best_start = start
        
        print(f"  最佳匹配: 偏移{best_start*2}s, 相似度{best_score:.3f}")

print("\n" + "="*70)
print("搜索185s的最佳匹配...")
print("="*70)
for source_path in source_videos:
    if "363977673945911296" in source_path:
        print(f"\n检查源视频: {Path(source_path).name}")
        source_fp = reconstructor.source_fingerprints[Path(source_path)]
        
        # 目标185s的指纹 (185/2 = 92.5, 取93)
        target_idx = int(185 / 2)
        target_segment = reconstructor.target_fingerprint[target_idx:target_idx+3]  # 185-190s
        
        print(f"  目标185s指纹索引: {target_idx}")
        
        # 搜索最佳匹配
        best_score = 0
        best_start = 0
        for start in range(0, len(source_fp) - len(target_segment)):
            end = start + len(target_segment)
            source_segment = source_fp[start:end]
            
            correlations = []
            for t, s in zip(target_segment, source_segment):
                if len(t) == len(s) and len(t) > 0:
                    import numpy as np
                    if np.std(t) > 0 and np.std(s) > 0:
                        corr = np.corrcoef(t, s)[0, 1]
                        correlations.append(corr)
            
            score = sum(correlations) / len(correlations) if correlations else 0
            if score > best_score:
                best_score = score
                best_start = start
        
        print(f"  最佳匹配: 偏移{best_start*2}s, 相似度{best_score:.3f}")

print("\n" + "="*70)
print("搜索完成")
print("="*70)
