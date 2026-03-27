#!/usr/bin/env python3
"""
视频重构性能优化补丁
用于优化多源片段匹配的速度
"""

import cv2
import numpy as np
from pathlib import Path

def optimize_multi_source_segments(original_method):
    """
    优化多源片段匹配：
    1. 目标视频每2秒采样一次（而不是每秒）
    2. 源视频每5秒采样一次（而不是每秒）
    """
    
    def optimized_method(self, target_duration: float):
        print(f"\n🔍 阶段2: 多源片段拼接... [优化版]")
        
        # 优化：每2秒采样一帧，减少计算量
        sample_interval = 2.0
        target_times = np.arange(0, target_duration, sample_interval)
        target_frames = {}
        
        print(f"   提取目标视频 {len(target_times)} 帧 (间隔{sample_interval}s)...")
        for t in target_times:
            frame_path = self.temp_dir / f"target_{t:.1f}.jpg"
            self.extract_frame_at(self.target_video, t, frame_path)
            if frame_path.exists():
                target_frames[t] = frame_path
        
        # 优化：源视频每5秒采样一帧
        source_frames = {}
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            source_times = np.arange(0, source_duration, 5.0)  # 5秒间隔
            source_frames[source] = {}
            
            for t in source_times:
                frame_path = self.temp_dir / f"source_{source.stem}_{t:.1f}.jpg"
                self.extract_frame_at(source, t, frame_path)
                if frame_path.exists():
                    source_frames[source][t] = frame_path
        
        print(f"   搜索最佳匹配...")
        
        # 为每个目标时间点找到最佳源
        matches = []
        for target_time, target_frame in target_frames.items():
            best_source = None
            best_time = 0
            best_score = 0
            
            for source, frames in source_frames.items():
                for source_time, source_frame in frames.items():
                    sim = self.calculate_frame_similarity(target_frame, source_frame)
                    if sim > best_score:
                        best_score = sim
                        best_source = source
                        best_time = source_time
            
            if best_score > self.match_threshold:
                matches.append({
                    'target_time': target_time,
                    'source': best_source,
                    'source_time': best_time,
                    'score': best_score
                })
        
        print(f"   找到 {len(matches)}/{len(target_frames)} 个匹配点")
        
        # 组织成连续片段（保持原逻辑）
        from video_reconstructor_hybrid_v6_optimized import VideoSegment
        segments = []
        current_segment = None
        
        for match in sorted(matches, key=lambda x: x['target_time']):
            if current_segment is None:
                current_segment = {
                    'source': match['source'],
                    'start_time': match['source_time'],
                    'end_time': match['source_time'] + sample_interval,
                    'target_start': match['target_time'],
                    'scores': [match['score']]
                }
            elif (current_segment['source'] == match['source'] and 
                  abs(match['source_time'] - current_segment['end_time']) < 5):
                current_segment['end_time'] = match['source_time'] + sample_interval
                current_segment['scores'].append(match['score'])
            else:
                if len(current_segment['scores']) >= self.min_segment_duration:
                    segments.append(VideoSegment(
                        source_video=current_segment['source'],
                        start_time=current_segment['start_time'],
                        end_time=current_segment['end_time'],
                        similarity_score=np.mean(current_segment['scores']),
                        target_start=current_segment['target_start'],
                        target_end=current_segment['target_start'] + len(current_segment['scores']) * sample_interval
                    ))
                
                current_segment = {
                    'source': match['source'],
                    'start_time': match['source_time'],
                    'end_time': match['source_time'] + sample_interval,
                    'target_start': match['target_time'],
                    'scores': [match['score']]
                }
        
        if current_segment and len(current_segment['scores']) >= self.min_segment_duration:
            segments.append(VideoSegment(
                source_video=current_segment['source'],
                start_time=current_segment['start_time'],
                end_time=current_segment['end_time'],
                similarity_score=np.mean(current_segment['scores']),
                target_start=current_segment['target_start'],
                target_end=current_segment['target_start'] + len(current_segment['scores']) * sample_interval
            ))
        
        print(f"   生成 {len(segments)} 个连续片段")
        
        return segments
    
    return optimized_method

print("优化补丁已加载")
print("优化内容：")
print("  - 目标视频采样间隔: 2秒 (原1秒)")
print("  - 源视频采样间隔: 5秒 (原1秒)")
print("  - 预期速度提升: 5-10倍")
