#!/usr/bin/env python3
"""
视频混剪重构工具 V7 - 性能优化版
优化逻辑：
1. 音频预筛选 + 视频精修（V2方式）
2. 并行处理（8 workers）
3. 动态帧间隔 - 先粗筛（2秒），再精修（1秒）
4. 保留V6多源拼接能力
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import wave
import struct

@dataclass
class VideoSegment:
    """视频片段"""
    source_video: Path
    start_time: float
    end_time: float
    similarity_score: float
    target_start: float = 0
    target_end: float = 0

class VideoReconstructorHybridV7:
    """V7 性能优化版"""
    
    def __init__(self, target_video: str, source_videos: List[str], config: dict = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.config = config or {}
        
        # 优化参数 - V7融合配置
        self.fps = config.get('fps', 2) if config else 2
        self.similarity_threshold = config.get('similarity_threshold', 0.85) if config else 0.85
        self.match_threshold = config.get('match_threshold', 0.4) if config else 0.4
        self.audio_weight = config.get('audio_weight', 0.4) if config else 0.4
        self.video_weight = config.get('video_weight', 0.6) if config else 0.6
        self.search_step = 5
        
        # V7新增：动态帧间隔
        self.coarse_interval = config.get('coarse_interval', 2.0) if config else 2.0  # 粗筛间隔
        self.fine_interval = config.get('fine_interval', 1.0) if config else 1.0     # 精修间隔
        
        # V7新增：并行处理
        self.max_workers = config.get('max_workers', 8) if config else 8
        self.use_audio_first = config.get('use_audio_first', True) if config else True
        self.audio_sample_rate = config.get('audio_sample_rate', 16000) if config else 16000
        
        self.temp_dir = None
        
    def get_video_duration(self, video_path: Path) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def extract_frame_at(self, video_path: Path, time_sec: float, output_path: Path):
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(time_sec), '-i', str(video_path), '-vframes', '1', str(output_path)]
        subprocess.run(cmd, capture_output=True)

    def calculate_frame_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))
        if img1 is None or img2 is None:
            return 0.0
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.resize(gray1, (320, 180))
        gray2 = cv2.resize(gray2, (320, 180))
        
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        return 0.5 * max(0, hist_sim) + 0.5 * template_sim

    def _extract_audio_fingerprint_v7(self, video_path: Path) -> np.ndarray:
        """V7: 高质量音频指纹提取（16kHz，50%重叠）"""
        temp_wav = self.temp_dir / f"{video_path.stem}_audio.wav"
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', 
               '-ar', str(self.audio_sample_rate), '-ac', '1',
               '-af', 'highpass=200,lowpass=4000',
               str(temp_wav)]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_wav.exists():
            return np.array([])
        
        with wave.open(str(temp_wav), 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            samples = struct.unpack(f'{n_frames}h', audio_data)
        
        temp_wav.unlink(missing_ok=True)
        
        # 50%重叠窗口
        segment_size = self.audio_sample_rate
        fingerprints = []
        
        for i in range(0, len(samples) - segment_size, segment_size // 2):
            segment = samples[i:i + segment_size]
            if len(segment) < segment_size:
                break
            
            # 汉宁窗
            window = np.hanning(len(segment))
            segment = np.array(segment) * window
            
            # FFT + 分桶
            fft = np.abs(np.fft.fft(segment))[:segment_size // 2]
            buckets = 32
            bucket_size = len(fft) // buckets
            fingerprint = []
            for b in range(buckets):
                bucket = fft[b * bucket_size:(b + 1) * bucket_size]
                fingerprint.append(np.mean(bucket))
            
            # 归一化
            fingerprint = np.array(fingerprint)
            fingerprint = fingerprint / (np.linalg.norm(fingerprint) + 1e-10)
            fingerprints.append(fingerprint)
        
        return np.array(fingerprints)

    def _find_audio_matches_v7(self, target_fp: np.ndarray, source_video: Path) -> List[Tuple[float, float, float]]:
        """V7: 多尺度音频匹配，返回 (source_time, score, confidence)"""
        source_fp = self._extract_audio_fingerprint_v7(source_video)
        
        if len(target_fp) == 0 or len(source_fp) == 0 or len(target_fp) > len(source_fp):
            return []
        
        matches = []
        window_sizes = [5, 10, 15]  # 多尺度窗口
        audio_threshold = self.config.get('audio_match_threshold', 0.7)  # 提高阈值到0.7
        
        for target_idx in range(len(target_fp)):
            best_match = None
            best_score = 0.0
            
            for window_size in window_sizes:
                if target_idx + window_size > len(target_fp):
                    continue
                
                target_window = target_fp[target_idx:target_idx + window_size]
                
                for source_idx in range(0, len(source_fp) - window_size, 2):  # 步长2提高效率
                    source_window = source_fp[source_idx:source_idx + window_size]
                    
                    # 余弦相似度
                    cosine_sim = np.mean([
                        np.dot(t, s) / (np.linalg.norm(t) * np.linalg.norm(s) + 1e-10)
                        for t, s in zip(target_window, source_window)
                    ])
                    
                    # 欧氏距离转相似度
                    euclidean_dist = np.mean([
                        np.linalg.norm(t - s) for t, s in zip(target_window, source_window)
                    ])
                    euclidean_sim = 1 / (1 + euclidean_dist)
                    
                    similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim
                    
                    if similarity > best_score and similarity > audio_threshold:
                        best_score = similarity
                        best_match = (source_idx * 0.5, best_score)  # 0.5因为50%重叠
            
            if best_match:
                matches.append((target_idx * 0.5, best_match[0], best_match[1]))
        
        # 去重
        if matches:
            matches.sort(key=lambda x: x[0])
            deduplicated = [matches[0]]
            for match in matches[1:]:
                if match[0] - deduplicated[-1][0] > 1.0:
                    deduplicated.append(match)
                elif match[2] > deduplicated[-1][2]:
                    deduplicated[-1] = match
            matches = deduplicated
        
        return matches

    def verify_with_video(self, source: Path, start_time: float, target_duration: float) -> float:
        """视频帧验证"""
        sample_times = [0, target_duration * 0.5, target_duration * 0.9]
        video_scores = []
        
        for t in sample_times:
            if t < target_duration:
                target_frame = self.temp_dir / f"v_target_{t:.0f}.jpg"
                source_frame = self.temp_dir / f"v_source_{t:.0f}.jpg"
                self.extract_frame_at(self.target_video, t, target_frame)
                self.extract_frame_at(source, start_time + t, source_frame)
                
                if target_frame.exists() and source_frame.exists():
                    sim = self.calculate_frame_similarity(target_frame, source_frame)
                    video_scores.append(sim)
        
        return np.mean(video_scores) if video_scores else 0

    def find_multi_source_segments_v7(self, target_duration: float) -> List[VideoSegment]:
        """V7: 音频预筛选 + 动态帧间隔 + 并行处理"""
        print(f"\n🔍 V7 多源片段搜索 (音频预筛选 + 动态帧间隔)...")
        
        # 阶段1: 音频预筛选 - 快速定位候选区域
        print(f"   阶段1: 音频预筛选...")
        target_audio = self._extract_audio_fingerprint_v7(self.target_video)
        
        all_audio_matches = []
        for source in self.source_videos:
            matches = self._find_audio_matches_v7(target_audio, source)
            for target_time, source_time, score in matches:
                all_audio_matches.append({
                    'target_time': target_time,
                    'source': source,
                    'source_time': source_time,
                    'audio_score': score
                })
        
        all_audio_matches.sort(key=lambda x: x['target_time'])
        print(f"   音频预筛选: {len(all_audio_matches)} 个候选点")
        
        if not all_audio_matches:
            print(f"   ⚠️ 音频预筛选无结果，降级到纯视频匹配")
            return self._fallback_video_only_match(target_duration)
        
        # 阶段2: 粗筛 (2秒间隔) - 快速筛选高置信度区域
        print(f"   阶段2: 粗筛 (2秒间隔)...")
        coarse_matches = self._coarse_video_match(all_audio_matches, target_duration)
        print(f"   粗筛结果: {len(coarse_matches)} 个匹配点")
        
        # 阶段3: 精修 (1秒间隔) - 在粗筛结果附近精细匹配
        print(f"   阶段3: 精修 (1秒间隔)...")
        fine_matches = self._fine_video_match(coarse_matches, target_duration)
        print(f"   精修结果: {len(fine_matches)} 个匹配点")
        
        # 阶段4: 组织成连续片段
        segments = self._aggregate_segments(fine_matches)
        print(f"   最终: {len(segments)} 个连续片段")
        
        return segments
    
    def _coarse_video_match(self, audio_matches: List[Dict], target_duration: float) -> List[Dict]:
        """粗筛: 2秒间隔快速匹配 + 强制视频验证"""
        print(f"   粗筛: 强制视频验证每个音频候选点...")
        
        verified_matches = []
        video_verify_threshold = 0.6  # 视频验证阈值
        
        # 对每个音频候选点进行视频验证
        for match in audio_matches:
            target_time = match['target_time']
            source = match['source']
            source_time = match['source_time']
            audio_score = match['audio_score']
            
            # 提取目标帧和源帧进行验证
            target_frame = self.temp_dir / f"verify_target_{target_time:.1f}.jpg"
            source_frame = self.temp_dir / f"verify_{source.stem}_{source_time:.1f}.jpg"
            
            self.extract_frame_at(self.target_video, target_time, target_frame)
            self.extract_frame_at(source, source_time, source_frame)
            
            if not target_frame.exists() or not source_frame.exists():
                continue
            
            video_score = self.calculate_frame_similarity(target_frame, source_frame)
            combined_score = self.audio_weight * audio_score + self.video_weight * video_score
            
            # 强制视频验证: 视频分数必须超过阈值
            if video_score >= video_verify_threshold:
                verified_matches.append({
                    'target_time': target_time,
                    'source': source,
                    'source_time': source_time,
                    'audio_score': audio_score,
                    'video_score': video_score,
                    'combined_score': combined_score,
                    'verified': True
                })
                print(f"      ✓ @{target_time:.1f}s -> {source.name}@{source_time:.1f}s "
                      f"音频{audio_score:.1%} 视频{video_score:.1%}")
            else:
                print(f"      ✗ @{target_time:.1f}s -> {source.name}@{source_time:.1f}s "
                      f"音频{audio_score:.1%} 视频{video_score:.1%} (视频验证失败)")
        
        if not verified_matches:
            print(f"   ⚠️ 无通过视频验证的匹配点")
            return []
        
        print(f"   通过视频验证: {len(verified_matches)} 个")
        
        # 基于验证通过的匹配点，2秒间隔提取更多帧进行匹配
        target_times = set()
        for match in verified_matches:
            for offset in [-2, 0, 2]:
                t = match['target_time'] + offset
                if 0 <= t < target_duration:
                    target_times.add(round(t, 1))
        
        target_times = sorted(target_times)
        
        # 并行提取目标帧
        target_frames = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_frame_cached, self.target_video, t): t
                for t in target_times
            }
            for future in futures:
                t = futures[future]
                frame_path = future.result()
                if frame_path:
                    target_frames[t] = frame_path
        
        # 只从验证通过的源视频区域搜索
        source_candidates = {}
        for match in verified_matches:
            source = match['source']
            if source not in source_candidates:
                source_candidates[source] = set()
            for offset in range(-4, 5, 2):
                t = match['source_time'] + offset
                if t >= 0:
                    source_candidates[source].add(round(t, 1))
        
        # 并行提取源帧
        source_frames = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for source, times in source_candidates.items():
                source_frames[source] = {}
                for t in times:
                    future = executor.submit(self._extract_frame_cached, source, t)
                    futures[future] = (source, t)
            
            for future in futures:
                source, t = futures[future]
                frame_path = future.result()
                if frame_path:
                    source_frames[source][t] = frame_path
        
        # 匹配
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
        
        return matches
    
    def _fine_video_match(self, coarse_matches: List[Dict], target_duration: float) -> List[Dict]:
        """精修: 1秒间隔精细匹配"""
        if not coarse_matches:
            return []
        
        # 在粗筛结果附近，1秒间隔精细匹配
        fine_target_times = set()
        for match in coarse_matches:
            for offset in [-0.5, 0, 0.5]:
                t = match['target_time'] + offset
                if 0 <= t < target_duration:
                    fine_target_times.add(round(t, 1))
        
        fine_target_times = sorted(fine_target_times)
        
        # 并行提取精细帧
        target_frames = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_frame_cached, self.target_video, t): t
                for t in fine_target_times
            }
            for future in futures:
                t = futures[future]
                frame_path = future.result()
                if frame_path:
                    target_frames[t] = frame_path
        
        # 构建源候选区域（更密集）
        source_candidates = {}
        for match in coarse_matches:
            source = match['source']
            if source not in source_candidates:
                source_candidates[source] = set()
            for offset in np.arange(-1, 1.1, 0.5):
                t = match['source_time'] + offset
                if t >= 0:
                    source_candidates[source].add(round(t, 1))
        
        # 并行提取源帧
        source_frames = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for source, times in source_candidates.items():
                source_frames[source] = {}
                for t in times:
                    future = executor.submit(self._extract_frame_cached, source, t)
                    futures[future] = (source, t)
            
            for future in futures:
                source, t = futures[future]
                frame_path = future.result()
                if frame_path:
                    source_frames[source][t] = frame_path
        
        # 精细匹配
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
        
        return matches
    
    def _extract_frame_cached(self, video_path: Path, time_sec: float) -> Path:
        """提取帧并缓存"""
        cache_key = f"{video_path.stem}_{time_sec:.1f}"
        frame_path = self.temp_dir / f"cache_{cache_key}.jpg"
        
        if frame_path.exists():
            return frame_path
        
        self.extract_frame_at(video_path, time_sec, frame_path)
        return frame_path if frame_path.exists() else None
    
    def _aggregate_segments(self, matches: List[Dict]) -> List[VideoSegment]:
        """将匹配点聚合成连续片段"""
        if not matches:
            return []
        
        matches = sorted(matches, key=lambda x: x['target_time'])
        
        segments = []
        current = None
        interval = 0.5  # 精修间隔
        
        for match in matches:
            if current is None:
                current = {
                    'source': match['source'],
                    'start_time': match['source_time'],
                    'end_time': match['source_time'] + interval,
                    'target_start': match['target_time'],
                    'scores': [match['score']]
                }
            elif (current['source'] == match['source'] and 
                  abs(match['source_time'] - current['end_time']) < 10):
                current['end_time'] = match['source_time'] + interval
                current['scores'].append(match['score'])
            else:
                segments.append(VideoSegment(
                    source_video=current['source'],
                    start_time=current['start_time'],
                    end_time=current['end_time'],
                    similarity_score=np.mean(current['scores']),
                    target_start=current['target_start'],
                    target_end=current['target_start'] + len(current['scores']) * interval
                ))
                current = {
                    'source': match['source'],
                    'start_time': match['source_time'],
                    'end_time': match['source_time'] + interval,
                    'target_start': match['target_time'],
                    'scores': [match['score']]
                }
        
        if current:
            segments.append(VideoSegment(
                source_video=current['source'],
                start_time=current['start_time'],
                end_time=current['end_time'],
                similarity_score=np.mean(current['scores']),
                target_start=current['target_start'],
                target_end=current['target_start'] + len(current['scores']) * interval
            ))
        
        return segments
    
    def _fallback_video_only_match(self, target_duration: float) -> List[VideoSegment]:
        """降级: 纯视频匹配（V6方式）"""
        print(f"   使用纯视频匹配降级方案...")
        
        # 2秒间隔提取
        target_times = np.arange(0, target_duration, 2.0)
        target_frames = {}
        
        for t in target_times:
            frame_path = self._extract_frame_cached(self.target_video, t)
            if frame_path:
                target_frames[t] = frame_path
        
        # 提取所有源视频帧
        source_frames = {}
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            source_times = np.arange(0, source_duration, 2.0)
            source_frames[source] = {}
            
            for t in source_times:
                frame_path = self._extract_frame_cached(source, t)
                if frame_path:
                    source_frames[source][t] = frame_path
        
        # 匹配
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
        
        return self._aggregate_segments(matches)

    def reconstruct(self, output_path: str, use_target_audio: bool = True) -> List[VideoSegment]:
        """V7: 重构视频 - 音频预筛选 + 动态帧间隔 + 并行处理"""
        print(f"\n{'='*60}")
        print(f"🎬 V7 音频预筛选 + 动态帧间隔 + 并行处理")
        print(f"{'='*60}")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            target_duration = self.get_video_duration(self.target_video)
            print(f"   目标: {self.target_video.name} ({target_duration:.1f}s)")
            print(f"   源视频: {len(self.source_videos)} 个")
            print(f"   并行workers: {self.max_workers}")
            
            # 阶段1: 尝试单源完整匹配
            print(f"\n🔍 阶段1: 单源完整匹配...")
            
            target_audio = self._extract_audio_fingerprint_v7(self.target_video)
            
            best_result = None
            best_score = 0
            
            for source in self.source_videos:
                source_duration = self.get_video_duration(source)
                if source_duration < target_duration:
                    continue
                
                # 音频匹配找候选
                audio_matches = self._find_audio_matches_v7(target_audio, source)
                
                for target_time, source_time, audio_score in audio_matches[:5]:  # 只验证前5个
                    # 视频验证
                    video_score = self.verify_with_video(source, source_time, target_duration)
                    combined = self.audio_weight * audio_score + self.video_weight * video_score
                    
                    print(f"   {source.name} @{source_time:.0f}s: 音频{audio_score:.1%} 视频{video_score:.1%} 综合{combined:.1%}")
                    
                    if combined > best_score:
                        best_score = combined
                        best_result = {
                            'source': source,
                            'start_time': source_time,
                            'audio_score': audio_score,
                            'video_score': video_score,
                            'combined': combined
                        }
                    
                    if combined > 0.9:
                        break
                if best_result and best_result['combined'] > 0.9:
                    break
            
            # 检查单源是否足够
            if best_result and best_result['combined'] > self.similarity_threshold:
                print(f"\n✅ 单源匹配成功: {best_result['source'].name} @{best_result['start_time']:.1f}s")
                
                segment = VideoSegment(
                    source_video=best_result['source'],
                    start_time=best_result['start_time'],
                    end_time=best_result['start_time'] + target_duration,
                    similarity_score=best_result['combined']
                )
                
                self._generate_single_output(segment, output_path, use_target_audio)
                return [segment]
            
            # 阶段2: V7 多源片段拼接
            print(f"\n⚠️ 单源不足，切换到 V7 多源拼接...")
            if best_result:
                print(f"   最佳单源: {best_result['combined']:.1%}")
            
            segments = self.find_multi_source_segments_v7(target_duration)
            
            if not segments:
                print(f"   ❌ 未找到匹配片段")
                return []
            
            total_covered = sum(seg.target_end - seg.target_start for seg in segments)
            coverage = total_covered / target_duration
            print(f"   覆盖: {total_covered:.1f}s / {target_duration:.1f}s ({coverage:.1%})")
            
            if coverage < 0.5:
                print(f"   ❌ 覆盖不足")
                return []
            
            self._generate_multi_output(segments, output_path, use_target_audio)
            return segments
            
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    def _generate_single_output(self, segment: VideoSegment, output_path: str, use_target_audio: bool):
        """生成单源输出"""
        source_duration = self.get_video_duration(segment.source_video)
        actual_end = min(segment.end_time, source_duration - 0.1)
        
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(segment.start_time), '-t', str(actual_end - segment.start_time),
               '-i', str(segment.source_video), '-c', 'copy', output_path]
        subprocess.run(cmd, capture_output=True)
        
        if use_target_audio:
            target_audio = self.temp_dir / "audio.aac"
            subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                          '-i', str(self.target_video), '-vn', '-c:a', 'copy', str(target_audio)],
                         capture_output=True)
            
            if target_audio.exists():
                temp_out = output_path + ".tmp"
                subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                              '-i', output_path, '-i', str(target_audio),
                              '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                              '-shortest', temp_out], capture_output=True)
                Path(temp_out).replace(output_path)
        
        print(f"   ✅ 单源生成: {output_path}")

    def _generate_multi_output(self, segments: List[VideoSegment], output_path: str, use_target_audio: bool):
        """生成多源拼接输出 - 修复音画不同步问题"""
        print(f"\n🎬 生成多源拼接...")

        segments = sorted(segments, key=lambda x: x.target_start)

        # 简单拼接，不做复杂重叠（避免时间范围无效）
        concat_lines = []
        for seg in segments:
            concat_lines.append(f"file '{seg.source_video}'")
            concat_lines.append(f"inpoint {seg.start_time:.3f}")
            concat_lines.append(f"outpoint {seg.end_time:.3f}")

        concat_file = self.temp_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            f.write('\n'.join(concat_lines))

        # 使用重新编码确保帧连续（避免花屏）
        temp_concat = self.temp_dir / "concat.mp4"
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                       '-f', 'concat', '-safe', '0', '-i', str(concat_file),
                       '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                       '-r', '25',  # 统一帧率
                       str(temp_concat)], capture_output=True)

        if not temp_concat.exists():
            print(f"   ❌ 拼接失败")
            return

        if use_target_audio:
            # 提取与视频等长的音频（解决音画不同步）
            target_audio = self.temp_dir / "audio.aac"
            video_duration = self.get_video_duration(temp_concat)

            # 从原裁剪视频中截取与生成视频等长的音频
            subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                          '-i', str(self.target_video), '-vn', '-c:a', 'aac',
                          '-t', str(video_duration),  # 限制音频时长与视频一致
                          str(target_audio)], capture_output=True)

            if target_audio.exists():
                subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                              '-i', str(temp_concat), '-i', str(target_audio),
                              '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                              '-shortest', output_path], capture_output=True)
            else:
                Path(temp_concat).replace(output_path)
        else:
            Path(temp_concat).replace(output_path)

        if Path(output_path).exists():
            duration = self.get_video_duration(output_path)
            print(f"   ✅ 多源生成: {output_path} ({duration:.1f}s)")
        else:
            print(f"   ❌ 生成失败")


def load_config(config_path: str) -> dict:
    import yaml
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}
