#!/usr/bin/env python3
"""
查找95s的正确匹配
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo')

from pathlib import Path
from v6_fast import FastHighPrecisionReconstructor
import numpy as np

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

# 搜索95s的最佳匹配
print("\n" + "="*70)
print("搜索95s的最佳匹配...")
print("="*70)

# 目标95s的指纹
# 指纹步长是2秒，所以95s对应的指纹索引是 95/2 = 47.5 -> 47或48
# 但指纹提取时可能从0秒开始，每2秒一个特征
target_idx = int(95 / 2)  # 47
target_segment = reconstructor.target_fingerprint[target_idx:target_idx+3]  # 95-100s对应约3个特征

print(f"目标95s指纹索引范围: {target_idx}-{target_idx+2}")
print(f"目标段长度: {len(target_segment)} 个特征")

for source_path in source_videos:
    source = Path(source_path)
    source_fp = reconstructor.source_fingerprints[source]
    
    if len(source_fp) < len(target_segment):
        continue
    
    # 搜索最佳匹配
    best_score = 0
    best_start = 0
    
    for start in range(0, len(source_fp) - len(target_segment)):
        end = start + len(target_segment)
        source_segment = source_fp[start:end]
        
        correlations = []
        for t, s in zip(target_segment, source_segment):
            if len(t) == len(s) and len(t) > 0:
                if np.std(t) > 0 and np.std(s) > 0:
                    corr = np.corrcoef(t, s)[0, 1]
                    correlations.append(corr)
        
        score = sum(correlations) / len(correlations) if correlations else 0
        if score > best_score:
            best_score = score
            best_start = start
    
    if best_score > 0.5:  # 只显示相似度较高的
        print(f"  {source.name}: 偏移{best_start*2}s, 相似度{best_score:.3f}")

print("\n" + "="*70)
print("搜索完成")
print("="*70)
