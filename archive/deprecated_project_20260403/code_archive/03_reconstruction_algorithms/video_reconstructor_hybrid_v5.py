#!/usr/bin/env python3
"""
视频混剪重构工具 V5 - 轻量版
优化：使用关键帧提取 + 音频指纹快速匹配
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

class VideoReconstructorHybridV5:
    """V5: 轻量级多源重构器"""
    
    def __init__(self, target_video: str, source_videos: List[str], config: dict = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.config = config or {}
        self.temp_dir = None

    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def reconstruct(self, output_path: str, use_target_audio: bool = True) -> List[VideoSegment]:
        """重构视频"""
        print(f"\n🎬 V5 轻量重构...")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # 快速方法：使用音频指纹粗匹配 + 视频帧精匹配
            target_duration = self.get_video_duration(self.target_video)
            print(f"   目标时长: {target_duration:.1f}s")
            
            # 提取目标音频指纹
            print(f"   提取音频指纹...")
            target_audio = self._extract_audio_fingerprint(self.target_video)
            
            # 在所有源视频中搜索最佳匹配区域
            print(f"   搜索源视频...")
            best_source = None
            best_start = 0
            best_score = 0
            
            for source in self.source_videos:
                score, start = self._find_audio_match(target_audio, source)
                if score > best_score:
                    best_score = score
                    best_source = source
                    best_start = start
                    print(f"      {source.name}: 匹配度 {score:.2f} @ {start:.1f}s")
            
            if not best_source or best_score < 0.3:
                print(f"   ⚠️ 未找到良好匹配")
                return []
            
            print(f"   ✅ 最佳匹配: {best_source.name} @ {best_start:.1f}s (得分: {best_score:.2f})")
            
            # 创建片段
            segment = VideoSegment(
                source_video=best_source,
                start_time=best_start,
                end_time=best_start + target_duration,
                similarity_score=best_score,
                target_start=0,
                target_end=target_duration
            )
            
            # 生成输出
            self._generate_output(segment, output_path, use_target_audio)
            
            return [segment]
            
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    def _extract_audio_fingerprint(self, video_path: Path) -> np.ndarray:
        """提取音频指纹 - 简化版"""
        temp_wav = self.temp_dir / f"{video_path.stem}_audio.wav"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '8000',
            '-ac', '1',
            str(temp_wav)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_wav.exists():
            return np.array([])
        
        # 读取音频数据
        with wave.open(str(temp_wav), 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            samples = struct.unpack(f'{n_frames}h', audio_data)
        
        # 分块计算特征（每秒一个块）
        samples_per_sec = 8000
        n_blocks = len(samples) // samples_per_sec
        features = []
        
        for i in range(min(n_blocks, 300)):  # 最多300秒
            block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
            # 计算频谱特征（简化）
            fft = np.fft.rfft(block)
            magnitude = np.abs(fft)
            # 取前20个频段的平均值
            bands = np.array([np.mean(magnitude[j:j+len(magnitude)//20]) 
                            for j in range(0, len(magnitude), len(magnitude)//20)])
            features.append(bands[:20])
        
        return np.array(features)

    def _find_audio_match(self, target_fp: np.ndarray, source_video: Path) -> Tuple[float, float]:
        """在源视频中搜索最佳匹配"""
        source_fp = self._extract_audio_fingerprint(source_video)
        
        if len(target_fp) == 0 or len(source_fp) == 0:
            return 0, 0
        
        if len(target_fp) > len(source_fp):
            return 0, 0
        
        # 滑动窗口匹配
        best_score = 0
        best_start = 0
        
        for start in range(0, len(source_fp) - len(target_fp) + 1, 2):  # 步长2秒
            end = start + len(target_fp)
            source_segment = source_fp[start:end]
            
            # 计算相似度
            score = np.mean([np.corrcoef(t, s)[0,1] if len(t) == len(s) else 0 
                           for t, s in zip(target_fp, source_segment)])
            
            if score > best_score:
                best_score = score
                best_start = start
        
        return best_score, best_start

    def _generate_output(self, segment: VideoSegment, output_path: str, use_target_audio: bool):
        """生成输出视频"""
        target_duration = self.get_video_duration(self.target_video)
        source_duration = self.get_video_duration(segment.source_video)
        
        # 确保不超出源视频范围
        actual_end = min(segment.end_time, source_duration - 0.1)
        actual_duration = actual_end - segment.start_time
        
        if actual_duration < target_duration * 0.5:
            print(f"   ⚠️ 匹配片段过短 ({actual_duration:.1f}s)")
        
        # 提取视频片段
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(segment.start_time),
            '-t', str(actual_duration),
            '-i', str(segment.source_video),
            '-c', 'copy',
            output_path
        ]
        
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
