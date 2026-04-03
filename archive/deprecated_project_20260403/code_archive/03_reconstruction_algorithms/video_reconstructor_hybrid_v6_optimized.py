#!/usr/bin/env python3
"""
视频混剪重构工具 V6 Optimized - 基于V6的优化版本
优化内容：
1. 可配置的验证级别（严格/正常/宽松）
2. 更灵活的阈值配置
3. 改进的日志输出
4. 保持V6的性能优势
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import wave
import struct

class ValidationLevel(Enum):
    """验证级别"""
    STRICT = "strict"      # 严格: 匹配率>50%, 时长差异<20%
    NORMAL = "normal"      # 正常: 匹配率>30%, 时长差异<30%
    LOOSE = "loose"        # 宽松: 匹配率>10%, 时长差异<50%
    BEST_EFFORT = "best_effort"  # 尽力: 只要有输出就算成功

@dataclass
class VideoSegment:
    """视频片段"""
    source_video: Path
    start_time: float
    end_time: float
    similarity_score: float
    target_start: float = 0
    target_end: float = 0

@dataclass
class ValidationResult:
    """验证结果"""
    success: bool
    match_rate: float
    duration_diff: float
    reasons: List[str]
    suggestions: List[str]

class VideoReconstructorHybridV6Optimized:
    """V6 Optimized: 基于V6的优化版本，改进验证逻辑"""
    
    def __init__(self, target_video: str, source_videos: List[str], config: dict = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.config = config or {}
        self.temp_dir = None
        
        # 验证级别
        validation_level = config.get('validation_level', 'normal')
        self.validation_level = ValidationLevel(validation_level) if isinstance(validation_level, str) else ValidationLevel.NORMAL
        
        # 根据验证级别设置阈值
        self.thresholds = self._get_thresholds_by_level(self.validation_level)
        
        # 覆盖配置中的阈值
        self.thresholds.update({
            'fps': config.get('fps', 5),
            'similarity_threshold': config.get('similarity_threshold', 0.85),
            'min_segment_duration': config.get('min_segment_duration', 0.5),
            'match_threshold': config.get('match_threshold', 0.6),
            'audio_weight': config.get('audio_weight', 0.4),
            'video_weight': config.get('video_weight', 0.6),
            'single_source_threshold': config.get('single_source_threshold', 0.85),
        })
        
        self.fps = self.thresholds['fps']
        self.similarity_threshold = self.thresholds['similarity_threshold']
        self.min_segment_duration = self.thresholds['min_segment_duration']
        self.match_threshold = self.thresholds['match_threshold']
        self.audio_weight = self.thresholds['audio_weight']
        self.video_weight = self.thresholds['video_weight']
        self.single_source_threshold = self.thresholds['single_source_threshold']

    def _get_thresholds_by_level(self, level: ValidationLevel) -> Dict:
        """根据验证级别获取阈值"""
        thresholds = {
            ValidationLevel.STRICT: {
                'min_match_rate': 0.50,
                'max_duration_diff': 0.20,
                'min_coverage': 0.80,
            },
            ValidationLevel.NORMAL: {
                'min_match_rate': 0.30,
                'max_duration_diff': 0.30,
                'min_coverage': 0.60,
            },
            ValidationLevel.LOOSE: {
                'min_match_rate': 0.10,
                'max_duration_diff': 0.50,
                'min_coverage': 0.40,
            },
            ValidationLevel.BEST_EFFORT: {
                'min_match_rate': 0.0,
                'max_duration_diff': 1.0,
                'min_coverage': 0.0,
            }
        }
        return thresholds.get(level, thresholds[ValidationLevel.NORMAL])

    def validate_result(self, target_duration: float, segments: List[VideoSegment], output_path: str) -> ValidationResult:
        """验证重构结果"""
        reasons = []
        suggestions = []
        
        # 检查输出文件
        if not Path(output_path).exists():
            return ValidationResult(
                success=False,
                match_rate=0.0,
                duration_diff=1.0,
                reasons=["输出文件不存在"],
                suggestions=["检查磁盘空间", "检查FFmpeg命令"]
            )
        
        # 获取输出时长
        output_duration = self.get_video_duration(Path(output_path))
        duration_diff = abs(output_duration - target_duration) / target_duration if target_duration > 0 else 1.0
        
        # 计算匹配率
        if segments:
            total_covered = sum(seg.target_end - seg.target_start for seg in segments)
            match_rate = total_covered / target_duration if target_duration > 0 else 0.0
            avg_score = np.mean([seg.similarity_score for seg in segments])
        else:
            match_rate = 0.0
            avg_score = 0.0
        
        # 根据验证级别判断
        min_match_rate = self.thresholds.get('min_match_rate', 0.3)
        max_duration_diff = self.thresholds.get('max_duration_diff', 0.3)
        min_coverage = self.thresholds.get('min_coverage', 0.6)
        
        success = True
        
        if match_rate < min_match_rate:
            success = False
            reasons.append(f"匹配率过低 ({match_rate:.1%} < {min_match_rate:.0%})")
            suggestions.append("尝试放宽匹配阈值")
            suggestions.append("检查源视频是否包含目标内容")
        
        if duration_diff > max_duration_diff:
            success = False
            reasons.append(f"时长差异过大 ({duration_diff:.1%} > {max_duration_diff:.0%})")
            suggestions.append("检查片段拼接逻辑")
            suggestions.append("尝试使用不同的源视频")
        
        if match_rate < min_coverage:
            success = False
            reasons.append(f"覆盖率不足 ({match_rate:.1%} < {min_coverage:.0%})")
        
        return ValidationResult(
            success=success,
            match_rate=match_rate,
            duration_diff=duration_diff,
            reasons=reasons,
            suggestions=suggestions
        )

    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 0.0

    def extract_frame_at(self, video_path: Path, time_sec: float, output_path: Path):
        """提取指定时间点的帧"""
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)

    def calculate_frame_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        """计算两帧相似度"""
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))
        if img1 is None or img2 is None:
            return 0.0
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)
        if gray2 is None:
            return 0.0
        
        gray1 = cv2.resize(gray1, (320, 180))
        gray2 = cv2.resize(gray2, (320, 180))
        
        # 直方图相似度
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        return 0.5 * max(0, hist_sim) + 0.5 * template_sim

    def _extract_audio_fingerprint(self, video_path: Path) -> np.ndarray:
        """提取音频指纹 - 保持V6的8kHz采样率，保证速度"""
        temp_wav = self.temp_dir / f"{video_path.stem}_audio.wav"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(video_path), '-vn',
            '-acodec', 'pcm_s16le', '-ar', '8000', '-ac', '1',
            str(temp_wav)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_wav.exists():
            return np.array([])
        
        with wave.open(str(temp_wav), 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            if n_frames == 0:
                return np.array([])
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
        """音频指纹匹配 - 保持V6的简单滑动窗口"""
        source_fp = self._extract_audio_fingerprint(source_video)
        
        if len(target_fp) == 0 or len(source_fp) == 0 or len(target_fp) > len(source_fp):
            return 0, 0
        
        best_score = -1
        best_start = 0
        
        # 使用步长2加速搜索
        step = 2
        for start in range(0, len(source_fp) - len(target_fp) + 1, step):
            end = start + len(target_fp)
            source_segment = source_fp[start:end]
            
            correlations = []
            for t, s in zip(target_fp, source_segment):
                if len(t) == len(s) and np.std(t) > 0 and np.std(s) > 0:
                    corr = np.corrcoef(t, s)[0,1]
                    correlations.append(corr)
                else:
                    correlations.append(0)
            
            score = np.mean(correlations)
            if score > best_score:
                best_score = score
                best_start = start
        
        return best_score, best_start

    def find_best_single_source_match(self, target_duration: float):
        """找到最佳单源匹配"""
        print(f"\n🔍 阶段1: 单源完整匹配...")
        
        target_audio = self._extract_audio_fingerprint(self.target_video)
        
        best_result = None
        best_score = 0
        
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            if source_duration < target_duration:
                continue
            
            audio_score, start_block = self._find_audio_match(target_audio, source)
            start_time = start_block * 0.5  # 转换为秒
            
            if audio_score > 0.3:  # 音频预筛选
                # 视频帧验证 - 只检查片段开始位置（关键帧）
                target_frame = self.temp_dir / f"target_start.jpg"
                source_frame = self.temp_dir / f"source_{source.stem}_start.jpg"
                self.extract_frame_at(self.target_video, 0, target_frame)
                self.extract_frame_at(source, start_time, source_frame)
                
                video_score = 0
                if target_frame.exists() and source_frame.exists():
                    video_score = self.calculate_frame_similarity(target_frame, source_frame)
                
                # 如果开始位置匹配好，再检查中间和结尾
                if video_score > 0.7:
                    sample_times = [target_duration * 0.5, target_duration * 0.9]
                    video_scores = [video_score]
                    
                    for t in sample_times:
                        if t < target_duration:
                            target_frame = self.temp_dir / f"target_{t:.0f}.jpg"
                            source_frame = self.temp_dir / f"source_{source.stem}_{t:.0f}.jpg"
                            self.extract_frame_at(self.target_video, t, target_frame)
                            self.extract_frame_at(source, start_time + t, source_frame)
                            
                            if target_frame.exists() and source_frame.exists():
                                sim = self.calculate_frame_similarity(target_frame, source_frame)
                                video_scores.append(sim)
                    
                    video_score = np.mean(video_scores)
                
                combined_score = self.audio_weight * audio_score + self.video_weight * video_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = {
                        'source': source,
                        'start_time': start_time,
                        'audio_score': audio_score,
                        'video_score': video_score,
                        'combined_score': combined_score
                    }
                
                print(f"   {source.name}: 音频{audio_score:.1%} 视频{video_score:.1%} 综合{combined_score:.1%} @ {start_time:.1f}s")
        
        return best_result

    def find_multi_source_segments(self, target_duration: float) -> List[VideoSegment]:
        """多源片段匹配 - 保持V6的实现"""
        print(f"\n🔍 阶段2: 多源片段拼接...")
        
        # 提取目标视频的关键帧
        target_times = np.arange(0, target_duration, 1.0)
        target_frames = {}
        
        print(f"   提取目标视频 {len(target_times)} 帧...")
        for t in target_times:
            frame_path = self.temp_dir / f"target_{t:.1f}.jpg"
            self.extract_frame_at(self.target_video, t, frame_path)
            if frame_path.exists():
                target_frames[t] = frame_path
        
        # 提取所有源视频的帧
        source_frames = {}
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            source_times = np.arange(0, source_duration, 1.0)
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
        
        # 组织成连续片段
        segments = []
        current_segment = None
        
        for match in sorted(matches, key=lambda x: x['target_time']):
            if current_segment is None:
                current_segment = {
                    'source': match['source'],
                    'start_time': match['source_time'],
                    'end_time': match['source_time'] + 1,
                    'target_start': match['target_time'],
                    'scores': [match['score']]
                }
            elif (current_segment['source'] == match['source'] and 
                  abs(match['source_time'] - current_segment['end_time']) < 2):
                current_segment['end_time'] = match['source_time'] + 1
                current_segment['scores'].append(match['score'])
            else:
                if len(current_segment['scores']) >= self.min_segment_duration:
                    segments.append(VideoSegment(
                        source_video=current_segment['source'],
                        start_time=current_segment['start_time'],
                        end_time=current_segment['end_time'],
                        similarity_score=np.mean(current_segment['scores']),
                        target_start=current_segment['target_start'],
                        target_end=current_segment['target_start'] + len(current_segment['scores'])
                    ))
                
                current_segment = {
                    'source': match['source'],
                    'start_time': match['source_time'],
                    'end_time': match['source_time'] + 1,
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
                target_end=current_segment['target_start'] + len(current_segment['scores'])
            ))
        
        print(f"   生成 {len(segments)} 个连续片段")
        for i, seg in enumerate(segments[:5], 1):
            print(f"      片段{i}: {seg.source_video.name} @{seg.start_time:.1f}s~{seg.end_time:.1f}s")
        
        if len(segments) > 5:
            print(f"      ... 还有 {len(segments)-5} 个片段")
        
        return segments

    def generate_multi_source_output(self, segments: List[VideoSegment], output_path: str, use_target_audio: bool = True):
        """生成多源拼接输出"""
        print(f"\n🎬 生成多源拼接视频...")
        
        if not segments:
            print(f"   ❌ 没有可用片段")
            return False
        
        segments = sorted(segments, key=lambda x: x.target_start)
        
        concat_lines = []
        for seg in segments:
            concat_lines.append(f"file '{seg.source_video}'")
            concat_lines.append(f"inpoint {seg.start_time:.3f}")
            concat_lines.append(f"outpoint {seg.end_time:.3f}")
        
        concat_file = self.temp_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            f.write('\n'.join(concat_lines))
        
        temp_concat = self.temp_dir / "temp_concat.mp4"
        cmd1 = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-c', 'copy', str(temp_concat)
        ]
        subprocess.run(cmd1, capture_output=True)
        
        if not temp_concat.exists():
            print(f"   ❌ 拼接失败")
            return False
        
        if use_target_audio:
            target_audio = self.temp_dir / "target_audio.aac"
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                   '-i', str(self.target_video), '-vn', '-c:a', 'copy', str(target_audio)]
            subprocess.run(cmd, capture_output=True)
            
            if target_audio.exists():
                cmd2 = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(temp_concat), '-i', str(target_audio),
                    '-c:v', 'copy', '-c:a', 'aac',
                    '-map', '0:v:0', '-map', '1:a:0', '-shortest',
                    str(output_path)
                ]
            else:
                cmd2 = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-i', str(temp_concat), '-c', 'copy', str(output_path)]
        else:
            cmd2 = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(temp_concat), '-c', 'copy', str(output_path)]
        
        subprocess.run(cmd2, capture_output=True)
        
        if Path(output_path).exists():
            output_duration = self.get_video_duration(Path(output_path))
            print(f"   ✅ 生成成功: {output_duration:.1f}s")
            return True
        else:
            print(f"   ❌ 生成失败")
            return False

    def reconstruct(self, output_path: str, use_target_audio: bool = True) -> Tuple[List[VideoSegment], ValidationResult]:
        """智能重构 - 自动选择单源或多源模式，带验证"""
        print(f"\n{'='*60}")
        print(f"🎬 V6 Optimized 智能重构 [验证级别: {self.validation_level.value}]")
        print(f"{'='*60}")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        segments = []
        
        try:
            target_duration = self.get_video_duration(self.target_video)
            print(f"   目标视频: {self.target_video.name} ({target_duration:.1f}s)")
            print(f"   源视频数: {len(self.source_videos)}")
            
            # 阶段1: 尝试单源完整匹配
            single_match = self.find_best_single_source_match(target_duration)
            
            if single_match and single_match['combined_score'] > self.single_source_threshold:
                print(f"\n✅ 单源匹配成功!")
                print(f"   来源: {single_match['source'].name}")
                print(f"   位置: @{single_match['start_time']:.1f}s")
                print(f"   综合得分: {single_match['combined_score']:.1%}")
                
                segment = VideoSegment(
                    source_video=single_match['source'],
                    start_time=single_match['start_time'],
                    end_time=single_match['start_time'] + target_duration,
                    similarity_score=single_match['combined_score'],
                    target_start=0,
                    target_end=target_duration
                )
                segments = [segment]
                
                self._generate_single_output(segment, output_path, use_target_audio)
            else:
                print(f"\n⚠️ 单源匹配不足，切换到多源拼接模式")
                if single_match:
                    print(f"   最佳单源得分: {single_match['combined_score']:.1%} (需要>{self.single_source_threshold:.0%})")
                
                segments = self.find_multi_source_segments(target_duration)
                
                if segments:
                    total_covered = sum(seg.target_end - seg.target_start for seg in segments)
                    coverage = total_covered / target_duration
                    
                    print(f"\n   覆盖时长: {total_covered:.1f}s / {target_duration:.1f}s ({coverage:.1%})")
                    
                    min_coverage = self.thresholds.get('min_coverage', 0.6)
                    if coverage >= min_coverage or self.validation_level == ValidationLevel.BEST_EFFORT:
                        self.generate_multi_source_output(segments, output_path, use_target_audio)
                    else:
                        print(f"   ❌ 覆盖不足 ({coverage:.1%} < {min_coverage:.0%})，跳过生成")
                        segments = []
                else:
                    print(f"   ❌ 未找到可用片段")
            
            # 验证结果
            validation = self.validate_result(target_duration, segments, output_path)
            
            print(f"\n📊 验证结果:")
            print(f"   匹配率: {validation.match_rate:.1%}")
            print(f"   时长差异: {validation.duration_diff:.1%}")
            print(f"   状态: {'✅ 通过' if validation.success else '❌ 失败'}")
            
            if not validation.success:
                print(f"\n   失败原因:")
                for reason in validation.reasons:
                    print(f"      - {reason}")
                print(f"\n   建议:")
                for suggestion in validation.suggestions:
                    print(f"      - {suggestion}")
            
            return segments, validation
        
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    def _generate_single_output(self, segment: VideoSegment, output_path: str, use_target_audio: bool):
        """生成单源输出"""
        target_duration = self.get_video_duration(self.target_video)
        source_duration = self.get_video_duration(segment.source_video)
        
        actual_end = min(segment.end_time, source_duration - 0.1)
        actual_duration = actual_end - segment.start_time
        
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(segment.start_time), '-t', str(actual_duration),
               '-i', str(segment.source_video), '-c', 'copy', output_path]
        
        subprocess.run(cmd, capture_output=True)
        
        if Path(output_path).exists():
            print(f"   ✅ 生成: {output_path}")
        else:
            print(f"   ❌ 生成失败")


def load_config(config_path: str) -> dict:
    """加载配置"""
    import yaml
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}
