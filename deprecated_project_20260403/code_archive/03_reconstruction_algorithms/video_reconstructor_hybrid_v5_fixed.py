#!/usr/bin/env python3
"""
视频混剪重构工具 V5.1 - 修复版
修复问题：
1. 支持从所有源视频中查找匹配片段并拼接
2. 增加视频帧验证
3. 不再假设"素材一定比源视频短"
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional
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

class VideoReconstructorHybridV5Fixed:
    """V5.1: 修复版 - 支持多源片段拼接"""
    
    def __init__(self, target_video: str, source_videos: List[str], config: dict = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.config = config or {}
        
        # 参数
        self.fps = config.get('fps', 2) if config else 2
        self.similarity_threshold = config.get('similarity_threshold', 0.85) if config else 0.85
        self.min_segment_duration = config.get('min_segment_duration', 0.5) if config else 0.5
        self.match_threshold = config.get('match_threshold', 0.6) if config else 0.6
        self.audio_weight = config.get('audio_weight', 0.4) if config else 0.4
        self.video_weight = config.get('video_weight', 0.6) if config else 0.6
        
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

    def _extract_audio_fingerprint(self, video_path: Path) -> np.ndarray:
        """提取音频指纹"""
        temp_wav = self.temp_dir / f"{video_path.stem}_audio.wav"
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '8000', '-ac', '1',
               str(temp_wav)]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_wav.exists():
            return np.array([])
        
        with wave.open(str(temp_wav), 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            samples = struct.unpack(f'{n_frames}h', audio_data)
        
        samples_per_sec = 8000
        n_blocks = len(samples) // samples_per_sec
        features = []
        
        for i in range(min(n_blocks, 500)):
            block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
            fft = np.fft.rfft(block)
            magnitude = np.abs(fft)
            bands = np.array([np.mean(magnitude[j:j+len(magnitude)//20]) 
                            for j in range(0, len(magnitude), len(magnitude)//20)])
            features.append(bands[:20])
        
        return np.array(features)

    def _find_audio_match(self, target_fp: np.ndarray, source_video: Path) -> Tuple[float, float]:
        """在源视频中搜索最佳音频匹配"""
        source_fp = self._extract_audio_fingerprint(source_video)
        
        if len(target_fp) == 0 or len(source_fp) == 0 or len(target_fp) > len(source_fp):
            return 0, 0
        
        best_score = 0
        best_start = 0
        
        # 步长2秒搜索
        for start in range(0, len(source_fp) - len(target_fp) + 1, 2):
            end = start + len(target_fp)
            source_segment = source_fp[start:end]
            
            correlations = []
            for t, s in zip(target_fp, source_segment):
                if len(t) == len(s) and np.std(t) > 0 and np.std(s) > 0:
                    corr = np.corrcoef(t, s)[0,1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
            
            score = np.mean(correlations)
            if score > best_score:
                best_score = score
                best_start = start
        
        return best_score, best_start

    def verify_with_video_frames(self, source: Path, start_time: float, target_duration: float) -> float:
        """视频帧验证"""
        sample_times = [0, target_duration * 0.3, target_duration * 0.6, target_duration * 0.9]
        video_scores = []
        
        for t in sample_times:
            if t < target_duration:
                target_frame = self.temp_dir / f"verify_target_{t:.0f}.jpg"
                source_frame = self.temp_dir / f"verify_source_{t:.0f}.jpg"
                self.extract_frame_at(self.target_video, t, target_frame)
                self.extract_frame_at(source, start_time + t, source_frame)
                
                if target_frame.exists() and source_frame.exists():
                    sim = self.calculate_frame_similarity(target_frame, source_frame)
                    video_scores.append(sim)
        
        return np.mean(video_scores) if video_scores else 0

    def find_multi_source_segments(self, target_duration: float) -> List[VideoSegment]:
        """从所有源视频中查找匹配片段并拼接"""
        print(f"\n🔍 多源片段搜索...")
        
        # 提取目标视频帧（每2秒一帧）
        target_times = np.arange(0, target_duration, 2.0)
        target_frames = {}
        
        for t in target_times:
            frame_path = self.temp_dir / f"target_{t:.1f}.jpg"
            self.extract_frame_at(self.target_video, t, frame_path)
            if frame_path.exists():
                target_frames[t] = frame_path
        
        # 提取所有源视频帧
        source_frames = {}
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            source_times = np.arange(0, source_duration, 2.0)
            source_frames[source] = {}
            
            for t in source_times:
                frame_path = self.temp_dir / f"source_{source.stem}_{t:.1f}.jpg"
                self.extract_frame_at(source, t, frame_path)
                if frame_path.exists():
                    source_frames[source][t] = frame_path
        
        # 为每个目标时间点找最佳源
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
        
        # 组织成连续片段
        if not matches:
            return []
        
        matches = sorted(matches, key=lambda x: x['target_time'])
        
        segments = []
        current = None
        
        for match in matches:
            if current is None:
                current = {
                    'source': match['source'],
                    'start_time': match['source_time'],
                    'end_time': match['source_time'] + 2,
                    'target_start': match['target_time'],
                    'scores': [match['score']]
                }
            elif (current['source'] == match['source'] and 
                  abs(match['source_time'] - current['end_time']) < 5):
                current['end_time'] = match['source_time'] + 2
                current['scores'].append(match['score'])
            else:
                segments.append(VideoSegment(
                    source_video=current['source'],
                    start_time=current['start_time'],
                    end_time=current['end_time'],
                    similarity_score=np.mean(current['scores']),
                    target_start=current['target_start'],
                    target_end=current['target_start'] + len(current['scores']) * 2
                ))
                current = {
                    'source': match['source'],
                    'start_time': match['source_time'],
                    'end_time': match['source_time'] + 2,
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
                target_end=current['target_start'] + len(current['scores']) * 2
            ))
        
        print(f"   找到 {len(matches)} 个匹配点，{len(segments)} 个连续片段")
        return segments

    def reconstruct(self, output_path: str, use_target_audio: bool = True) -> List[VideoSegment]:
        """重构视频"""
        print(f"\n{'='*60}")
        print(f"🎬 V5.1 修复版 - 多源片段拼接")
        print(f"{'='*60}")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            target_duration = self.get_video_duration(self.target_video)
            print(f"   目标: {self.target_video.name} ({target_duration:.1f}s)")
            print(f"   源视频: {len(self.source_videos)} 个")
            
            # 阶段1: 尝试单源完整匹配 + 视频验证
            print(f"\n🔍 阶段1: 单源匹配 + 视频验证...")
            
            target_audio = self._extract_audio_fingerprint(self.target_video)
            
            best_result = None
            best_score = 0
            
            for source in self.source_videos:
                source_duration = self.get_video_duration(source)
                
                # 音频匹配
                audio_score, start = self._find_audio_match(target_audio, source)
                
                if audio_score > 0.3:
                    # 视频验证
                    video_score = self.verify_with_video_frames(source, start, target_duration)
                    combined = self.audio_weight * audio_score + self.video_weight * video_score
                    
                    print(f"   {source.name}: 音频{audio_score:.1%} 视频{video_score:.1%} 综合{combined:.1%}")
                    
                    if combined > best_score:
                        best_score = combined
                        best_result = {
                            'source': source,
                            'start_time': start,
                            'audio_score': audio_score,
                            'video_score': video_score,
                            'combined': combined
                        }
            
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
            
            # 阶段2: 多源片段拼接
            print(f"\n⚠️ 单源不足，切换到多源拼接模式...")
            if best_result:
                print(f"   最佳单源: {best_result['combined']:.1%} (需要>{self.similarity_threshold:.1%})")
            
            segments = self.find_multi_source_segments(target_duration)
            
            if not segments:
                print(f"   ❌ 未找到匹配片段")
                return []
            
            # 计算覆盖
            total_covered = sum(seg.target_end - seg.target_start for seg in segments)
            coverage = total_covered / target_duration
            print(f"   覆盖: {total_covered:.1f}s / {target_duration:.1f}s ({coverage:.1%})")
            
            if coverage < 0.5:
                print(f"   ❌ 覆盖不足 50%，无法生成")
                return []
            
            # 生成拼接视频
            self._generate_multi_output(segments, output_path, use_target_audio, target_duration)
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
        
        if use_target_audio:
            target_audio = self.temp_dir / "target_audio.aac"
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
        
        print(f"   ✅ 生成: {output_path}")

    def _generate_multi_output(self, segments: List[VideoSegment], output_path: str, 
                               use_target_audio: bool, target_duration: float):
        """生成多源拼接输出"""
        print(f"\n🎬 生成多源拼接...")
        
        # 按目标时间排序
        segments = sorted(segments, key=lambda x: x.target_start)
        
        # 创建 concat 列表
        concat_lines = []
        for seg in segments:
            concat_lines.append(f"file '{seg.source_video}'")
            concat_lines.append(f"inpoint {seg.start_time:.3f}")
            concat_lines.append(f"outpoint {seg.end_time:.3f}")
        
        concat_file = self.temp_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            f.write('\n'.join(concat_lines))
        
        # 拼接
        temp_concat = self.temp_dir / "concat.mp4"
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                       '-f', 'concat', '-safe', '0', '-i', str(concat_file),
                       '-c', 'copy', str(temp_concat)], capture_output=True)
        
        if not temp_concat.exists():
            print(f"   ❌ 拼接失败")
            return
        
        # 添加音频
        if use_target_audio:
            target_audio = self.temp_dir / "target_audio.aac"
            subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                          '-i', str(self.target_video), '-vn', '-c:a', 'copy', str(target_audio)],
                         capture_output=True)
            
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
            print(f"   ✅ 生成: {output_path} ({duration:.1f}s)")
        else:
            print(f"   ❌ 生成失败")


def load_config(config_path: str) -> dict:
    import yaml
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}
