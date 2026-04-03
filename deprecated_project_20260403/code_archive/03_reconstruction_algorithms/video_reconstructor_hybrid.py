#!/usr/bin/env python3
"""
视频混剪重构工具 - 音频+视频混合对齐版
先用音频指纹快速定位，再用视频帧精修
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import json
import yaml
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import hashlib

@dataclass
class VideoSegment:
    """视频片段信息"""
    source_video: Path
    start_time: float
    end_time: float
    similarity_score: float
    segment_idx: int
    match_type: str = "video"  # 'audio' 或 'video'

@dataclass
class FrameMatch:
    """帧匹配结果"""
    target_frame_idx: int
    source_video: Path
    source_frame_idx: int
    similarity: float
    timestamp: float

@dataclass
class AudioMatch:
    """音频匹配结果"""
    target_time: float
    source_video: Path
    source_time: float
    confidence: float

@dataclass
class FrameFeature:
    """帧特征数据"""
    frame_path: Path
    timestamp: float
    hist: np.ndarray
    phash: np.ndarray
    template_key: np.ndarray

class VideoReconstructorHybrid:
    def __init__(self, target_video: str, source_videos: List[str], config: dict = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.config = config or {}
        self.temp_dir = None
        self.max_workers = config.get('max_workers', 8)
        self.use_audio_first = config.get('use_audio_first', True)
        self.audio_sample_rate = config.get('audio_sample_rate', 8000)  # 降采样加速
        
    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def extract_audio_fingerprint(self, video_path: Path) -> Tuple[np.ndarray, float]:
        """提取音频指纹（简化版频谱特征）"""
        temp_audio = self.temp_dir / f"{video_path.stem}_audio.wav"
        
        # 提取音频（降采样加速）
        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-i', str(video_path),
            '-vn',  # 无视频
            '-acodec', 'pcm_s16le',
            '-ar', str(self.audio_sample_rate),  # 降采样
            '-ac', '1',  # 单声道
            str(temp_audio)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_audio.exists():
            return None, 0
        
        # 读取音频数据
        import wave
        with wave.open(str(temp_audio), 'rb') as wav:
            n_frames = wav.getnframes()
            audio_data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16)
        
        # 计算音频指纹（分段的频谱特征）
        segment_size = self.audio_sample_rate  # 1秒一段
        fingerprints = []
        
        for i in range(0, len(audio_data) - segment_size, segment_size // 2):  # 50%重叠
            segment = audio_data[i:i + segment_size]
            if len(segment) < segment_size:
                break
            
            # FFT获取频谱
            fft = np.abs(np.fft.fft(segment))[:segment_size // 2]
            
            # 取对数并分桶降维
            buckets = 32
            bucket_size = len(fft) // buckets
            fingerprint = []
            for b in range(buckets):
                bucket = fft[b * bucket_size:(b + 1) * bucket_size]
                fingerprint.append(np.mean(bucket))
            
            fingerprints.append(np.array(fingerprint))
        
        temp_audio.unlink(missing_ok=True)
        
        return np.array(fingerprints), self.audio_sample_rate
    
    def find_audio_matches(self, target_fingerprints: np.ndarray, 
                          source_fingerprints_dict: Dict[Path, np.ndarray]) -> List[AudioMatch]:
        """音频指纹匹配"""
        print(f"🔊 正在音频指纹匹配...")
        
        matches = []
        window_size = 10  # 对比窗口大小（约5秒）
        
        for target_idx in range(len(target_fingerprints) - window_size):
            target_window = target_fingerprints[target_idx:target_idx + window_size]
            
            best_match = None
            best_score = 0.0
            
            for source_video, source_fps in source_fingerprints_dict.items():
                if len(source_fps) < window_size:
                    continue
                
                # 滑动窗口匹配
                for source_idx in range(len(source_fps) - window_size):
                    source_window = source_fps[source_idx:source_idx + window_size]
                    
                    # 计算相似度（余弦相似度）
                    similarity = np.mean([
                        np.dot(t, s) / (np.linalg.norm(t) * np.linalg.norm(s) + 1e-10)
                        for t, s in zip(target_window, source_window)
                    ])
                    
                    if similarity > best_score and similarity > 0.85:  # 音频匹配阈值
                        best_score = similarity
                        best_match = AudioMatch(
                            target_time=target_idx * 0.5,  # 50%重叠
                            source_video=source_video,
                            source_time=source_idx * 0.5,
                            confidence=best_score
                        )
            
            if best_match:
                matches.append(best_match)
        
        print(f"   找到 {len(matches)} 个音频匹配点")
        return matches
    
    def extract_frames(self, video_path: Path, output_dir: Path, fps: int = 2) -> List[Tuple[Path, float]]:
        """提取视频帧"""
        output_pattern = output_dir / f"{video_path.stem}_%06d.jpg"
        scale = self.config.get('scale', '320:180')
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-i', str(video_path),
            '-vf', f'fps={fps},scale={scale}',
            '-q:v', '5',
            str(output_pattern)
        ]
        subprocess.run(cmd, capture_output=True)
        
        frames = []
        frame_files = sorted(output_dir.glob(f"{video_path.stem}_*.jpg"))
        for idx, frame_path in enumerate(frame_files):
            timestamp = idx / fps
            frames.append((frame_path, timestamp))
        
        return frames
    
    def extract_frame_features(self, frame_path: Path, timestamp: float) -> FrameFeature:
        """提取单帧特征"""
        img = cv2.imread(str(frame_path))
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 直方图特征
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # pHash
        small = cv2.resize(gray, (16, 16))
        dct = cv2.dct(np.float32(small))
        phash = (dct[:4, :4].flatten() > np.mean(dct[:4, :4])).astype(int)
        
        # 模板关键点
        template_key = cv2.resize(gray, (64, 64)).flatten()[:256]
        
        return FrameFeature(
            frame_path=frame_path,
            timestamp=timestamp,
            hist=hist,
            phash=phash,
            template_key=template_key
        )
    
    def extract_features_parallel(self, frames: List[Tuple[Path, float]], desc: str = "提取特征") -> List[FrameFeature]:
        """并行提取帧特征"""
        print(f"🔍 {desc}...")
        features = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.extract_frame_features, frame_path, ts): idx 
                for idx, (frame_path, ts) in enumerate(frames)
            }
            
            for future in futures:
                result = future.result()
                if result:
                    features.append(result)
        
        print(f"   完成: {len(features)} 帧")
        return features
    
    def calculate_similarity_fast(self, feat1: FrameFeature, feat2: FrameFeature) -> float:
        """快速相似度计算"""
        hist_sim = cv2.compareHist(
            feat1.hist.reshape(-1, 1), 
            feat2.hist.reshape(-1, 1), 
            cv2.HISTCMP_CORREL
        )
        
        hamming = np.sum(feat1.phash != feat2.phash)
        phash_sim = 1 - (hamming / 16.0)
        
        template_sim = np.dot(feat1.template_key, feat2.template_key) / (
            np.linalg.norm(feat1.template_key) * np.linalg.norm(feat2.template_key) + 1e-10
        )
        
        return 0.4 * max(0, hist_sim) + 0.3 * phash_sim + 0.3 * max(0, template_sim)
    
    def verify_matches_with_video(self, audio_matches: List[AudioMatch],
                                   target_features: List[FrameFeature],
                                   source_features_dict: Dict[Path, List[FrameFeature]]) -> List[FrameMatch]:
        """用视频帧验证音频匹配结果 - 改进版：音频对齐点前10帧开始对比"""
        print(f"🎬 用视频帧精修音频匹配结果（音频对齐点前10帧开始）...")
        
        verified_matches = [None] * len(target_features)
        fps = self.config.get('fps', 2)
        match_threshold = self.config.get('match_threshold', 0.6)
        pre_search_frames = self.config.get('pre_search_frames', 10)  # 音频对齐点前10帧开始
        
        # 从音频匹配构建候选区域（按目标时间分组）
        candidate_regions = {}
        for match in audio_matches:
            source = match.source_video
            if source not in candidate_regions:
                candidate_regions[source] = []
            # 调整搜索起点：从音频对齐点前10帧开始
            adjusted_time = max(0, match.source_time - (pre_search_frames / fps))
            candidate_regions[source].append({
                'original_time': match.source_time,
                'search_start_time': adjusted_time,
                'confidence': match.confidence
            })
        
        print(f"   音频对齐点数量: {len(audio_matches)}")
        print(f"   每点前{pre_search_frames}帧开始搜索（约{pre_search_frames/fps:.1f}秒）")
        
        # 为每个目标帧找到最佳匹配
        for target_idx, target_feat in enumerate(target_features):
            target_time = target_idx / fps
            
            best_match = None
            best_score = 0.0
            
            for source_video, source_features in source_features_dict.items():
                if source_video not in candidate_regions:
                    continue
                
                # 在调整后的搜索范围内查找
                for region in candidate_regions[source_video]:
                    search_start_idx = int(region['search_start_time'] * fps)
                    original_idx = int(region['original_time'] * fps)
                    
                    # 搜索范围：从调整起点到音频对齐点后5帧
                    search_end_idx = min(original_idx + 5, len(source_features))
                    search_start_idx = max(0, search_start_idx)
                    
                    for source_idx in range(search_start_idx, search_end_idx):
                        source_feat = source_features[source_idx]
                        similarity = self.calculate_similarity_fast(target_feat, source_feat)
                        
                        if similarity > best_score and similarity > match_threshold:
                            best_score = similarity
                            best_match = FrameMatch(
                                target_frame_idx=target_idx,
                                source_video=source_video,
                                source_frame_idx=source_idx,
                                similarity=best_score,
                                timestamp=source_feat.timestamp
                            )
            
            verified_matches[target_idx] = best_match
            
            if target_idx % 50 == 0:
                matched = sum(1 for m in verified_matches[:target_idx+1] if m is not None)
                print(f"   处理中: {target_idx}/{len(target_features)} 帧 (已匹配: {matched})")
        
        # 统计结果
        matched_count = sum(1 for m in verified_matches if m is not None)
        print(f"   视频帧验证完成: {matched_count}/{len(target_features)} 帧匹配成功")
        
        return verified_matches
    
    def segment_matches(self, matches: List[Optional[FrameMatch]]) -> List[VideoSegment]:
        """将连续的匹配帧聚合成片段 - 改进版：按目标视频时间顺序排列"""
        segments = []
        current_segment = []
        
        fps = self.config.get('fps', 5)
        min_duration = self.config.get('min_segment_duration', 0.2)  # 降低最小片段时长
        max_gap = self.config.get('max_segment_gap', 3.0)  # 允许最大3秒间隔
        
        print(f"\n📦 聚合片段参数: fps={fps}, min_duration={min_duration}s, max_gap={max_gap}s")
        
        last_target_time = None
        
        for idx, match in enumerate(matches):
            if match is None:
                # 检查当前片段是否满足最小时长
                if len(current_segment) >= min_duration * fps:
                    self._save_segment(current_segment, segments)
                    print(f"   保存片段: {len(current_segment)/fps:.1f}秒 ({len(current_segment)}帧)")
                elif current_segment:
                    print(f"   丢弃片段: 太短 {len(current_segment)/fps:.1f}秒")
                current_segment = []
                last_target_time = None
            else:
                # 检查时间间隔（基于目标视频时间）
                current_target_time = idx / fps
                if last_target_time is not None:
                    gap = current_target_time - last_target_time
                    if gap > max_gap:
                        # 间隔太大，保存当前片段并开始新片段
                        if len(current_segment) >= min_duration * fps:
                            self._save_segment(current_segment, segments)
                            print(f"   保存片段(间隔{gap:.1f}s): {len(current_segment)/fps:.1f}秒")
                        elif current_segment:
                            print(f"   丢弃片段(间隔{gap:.1f}s): 太短 {len(current_segment)/fps:.1f}秒")
                        current_segment = []
                
                current_segment.append(match)
                last_target_time = current_target_time
        
        # 保存最后一个片段
        if len(current_segment) >= min_duration * fps:
            self._save_segment(current_segment, segments)
            print(f"   保存最后片段: {len(current_segment)/fps:.1f}秒 ({len(current_segment)}帧)")
        elif current_segment:
            print(f"   丢弃最后片段: 太短 {len(current_segment)/fps:.1f}秒")
        
        print(f"\n📊 共找到 {len(segments)} 个有效片段")
        for i, seg in enumerate(segments):
            print(f"   片段{i+1}: {seg.source_video.name} {seg.start_time:.1f}s - {seg.end_time:.1f}s ({seg.end_time-seg.start_time:.1f}s)")
        
        return segments
    
    def _save_segment(self, segment_matches: List[FrameMatch], segments: List[VideoSegment]):
        """保存一个片段"""
        if not segment_matches:
            return
        
        first_match = segment_matches[0]
        last_match = segment_matches[-1]
        avg_similarity = np.mean([m.similarity for m in segment_matches])
        
        segment = VideoSegment(
            source_video=first_match.source_video,
            start_time=min(first_match.timestamp, last_match.timestamp),
            end_time=max(first_match.timestamp, last_match.timestamp) + 0.5,
            similarity_score=avg_similarity,
            segment_idx=len(segments),
            match_type="hybrid"  # 音频+视频混合
        )
        segments.append(segment)
    
    def generate_ffmpeg_commands(self, segments: List[VideoSegment], output_path: str, 
                                 use_target_audio: bool = True) -> List[str]:
        """生成FFmpeg命令列表 - 按时间顺序合并所有片段"""
        commands = []
        temp_files = []
        
        if not segments:
            print("❌ 没有找到任何片段")
            return []
        
        # 按开始时间排序所有片段
        sorted_segments = sorted(segments, key=lambda s: s.segment_idx)
        
        print(f"\n📋 合并 {len(sorted_segments)} 个片段（按目标视频时间顺序）:")
        total_duration = 0
        for i, seg in enumerate(sorted_segments):
            duration = seg.end_time - seg.start_time
            total_duration += duration
            print(f"   片段{i+1}: {seg.source_video.name} {seg.start_time:.1f}s - {seg.end_time:.1f}s ({duration:.1f}s)")
        print(f"   总时长: {total_duration:.1f}s")
        
        # 合并所有片段
        for idx, seg in enumerate(sorted_segments):
            temp_file = f"temp_segment_{idx:03d}.mp4"
            temp_files.append(temp_file)
            
            cmd = f'ffmpeg -y -hide_banner -ss {seg.start_time:.3f} -to {seg.end_time:.3f} -i "{seg.source_video}" -an -c:v copy "{temp_file}"'
            commands.append(cmd)
        
        concat_content = "\n".join([f"file '{f}'" for f in temp_files])
        with open("concat_list.txt", "w") as f:
            f.write(concat_content)
        
        temp_video = "temp_video_no_audio.mp4"
        concat_cmd = f'ffmpeg -y -hide_banner -f concat -safe 0 -i concat_list.txt -an -c:v copy "{temp_video}"'
        commands.append(concat_cmd)
        temp_files.append(temp_video)
        
        if use_target_audio:
            temp_audio = "temp_audio.aac"
            extract_cmd = f'ffmpeg -y -hide_banner -i "{self.target_video}" -vn -c:a aac "{temp_audio}"'
            commands.append(extract_cmd)
            temp_files.append(temp_audio)
            
            final_cmd = f'ffmpeg -y -hide_banner -i "{temp_video}" -i "{temp_audio}" -c:v copy -c:a copy -shortest "{output_path}"'
            commands.append(final_cmd)
        else:
            commands.append(f'mv "{temp_video}" "{output_path}"')
        
        cleanup_cmd = f"rm -f {' '.join(temp_files)} concat_list.txt"
        commands.append(cleanup_cmd)
        
        return commands
    
    def reconstruct(self, output_path: str = "reconstructed.mp4", use_target_audio: bool = True):
        """主流程：音频+视频混合对齐重构"""
        import time
        start_time = time.time()
        
        target_duration = self.get_video_duration(self.target_video)
        print(f"🎯 目标视频时长: {target_duration:.2f}秒")
        print(f"🎬 源视频数量: {len(self.source_videos)}个")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            fps = self.config.get('fps', 2)
            
            # 1. 音频指纹匹配（快速定位）
            if self.use_audio_first:
                print("\n🔊 步骤1: 提取音频指纹...")
                step_start = time.time()
                
                target_audio_fp, _ = self.extract_audio_fingerprint(self.target_video)
                source_audio_fp_dict = {}
                for source_video in self.source_videos:
                    fp, _ = self.extract_audio_fingerprint(source_video)
                    if fp is not None:
                        source_audio_fp_dict[source_video] = fp
                
                print(f"   耗时: {time.time() - step_start:.1f}秒")
                
                # 音频匹配
                print("\n🔊 步骤2: 音频指纹匹配...")
                step_start = time.time()
                audio_matches = self.find_audio_matches(target_audio_fp, source_audio_fp_dict)
                print(f"   耗时: {time.time() - step_start:.1f}秒")
            else:
                audio_matches = []
            
            # 2. 提取视频帧特征
            print("\n🎬 步骤3: 提取视频帧特征...")
            step_start = time.time()
            
            target_frames_dir = self.temp_dir / "target"
            target_frames_dir.mkdir()
            target_frames = self.extract_frames(self.target_video, target_frames_dir, fps)
            target_features = self.extract_features_parallel(target_frames, "提取目标视频特征")
            
            source_frames_dirs = {}
            source_features_dict = {}
            for source_video in self.source_videos:
                source_dir = self.temp_dir / f"source_{source_video.stem}"
                source_dir.mkdir()
                frames = self.extract_frames(source_video, source_dir, fps)
                features = self.extract_features_parallel(frames, f"提取 {source_video.name} 特征")
                source_features_dict[source_video] = features
            
            print(f"   耗时: {time.time() - step_start:.1f}秒")
            
            # 3. 视频帧验证（精修）
            print("\n🎬 步骤4: 视频帧精修匹配...")
            step_start = time.time()
            
            if audio_matches and self.use_audio_first:
                # 使用音频匹配结果缩小搜索范围
                matches = self.verify_matches_with_video(audio_matches, target_features, source_features_dict)
            else:
                # 纯视频匹配
                matches = self.full_video_match(target_features, source_features_dict)
            
            print(f"   耗时: {time.time() - step_start:.1f}秒")
            
            matched_frames = sum(1 for m in matches if m is not None)
            match_rate = matched_frames / len(matches) * 100
            print(f"\n📊 匹配率: {matched_frames}/{len(matches)} 帧 ({match_rate:.1f}%)")
            
            # 4. 聚合成片段
            segments = self.segment_matches(matches)
            print(f"\n📦 找到 {len(segments)} 个连续片段:")
            
            total_duration = 0
            for seg in segments:
                duration = seg.end_time - seg.start_time
                total_duration += duration
                print(f"   片段{seg.segment_idx+1}: {seg.source_video.name}")
                print(f"      时间: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
                print(f"      相似度: {seg.similarity_score:.2%}")
                print(f"      类型: {seg.match_type}")
            
            print(f"\n⏱️ 总时长: {total_duration:.2f}s / 目标: {target_duration:.2f}s")
            
            # 5. 生成视频
            print("\n⚙️ 步骤5: 生成重构视频...")
            step_start = time.time()
            commands = self.generate_ffmpeg_commands(segments, output_path, use_target_audio)
            
            for cmd in commands[:-1]:
                result = subprocess.run(cmd, shell=True, capture_output=True)
                if result.returncode != 0:
                    print(f"❌ 命令失败: {result.stderr.decode()[:200]}")
                    return None
            
            subprocess.run(commands[-1], shell=True)
            print(f"   耗时: {time.time() - step_start:.1f}秒")
            
            total_time = time.time() - start_time
            print(f"\n✅ 视频重构完成: {output_path}")
            print(f"⏱️ 总耗时: {total_time:.1f}秒")
            
            # 保存元数据
            metadata = {
                'target_video': str(self.target_video),
                'source_videos': [str(v) for v in self.source_videos],
                'output_video': output_path,
                'audio_matches': len(audio_matches) if self.use_audio_first else 0,
                'segments': [
                    {
                        'source_video': str(seg.source_video),
                        'source_video_name': seg.source_video.name,
                        'start_time': seg.start_time,
                        'end_time': seg.end_time,
                        'duration': seg.end_time - seg.start_time,
                        'similarity_score': float(seg.similarity_score),
                        'match_type': seg.match_type,
                        'segment_idx': seg.segment_idx
                    }
                    for seg in segments
                ],
                'total_segments': len(segments),
                'total_duration': total_duration,
                'target_duration': target_duration,
                'processing_time': total_time
            }
            
            metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"📝 元数据已保存: {metadata_path}")
            
            return segments
            
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def full_video_match(self, target_features: List[FrameFeature],
                        source_features_dict: Dict[Path, List[FrameFeature]]) -> List[Optional[FrameMatch]]:
        """纯视频完整匹配（备用方案）"""
        print(f"🔍 执行完整视频匹配...")
        
        matches = []
        match_threshold = self.config.get('match_threshold', 0.6)
        
        all_source_features = []
        for source_video, features in source_features_dict.items():
            for idx, feat in enumerate(features):
                all_source_features.append((source_video, idx, feat))
        
        for target_idx, target_feat in enumerate(target_features):
            if target_idx % 50 == 0:
                print(f"   处理中: {target_idx}/{len(target_features)} 帧")
            
            best_match = None
            best_score = 0.0
            
            for source_video, source_idx, source_feat in all_source_features:
                similarity = self.calculate_similarity_fast(target_feat, source_feat)
                
                if similarity > best_score and similarity > match_threshold:
                    best_score = similarity
                    best_match = FrameMatch(
                        target_frame_idx=target_idx,
                        source_video=source_video,
                        source_frame_idx=source_idx,
                        similarity=best_score,
                        timestamp=source_feat.timestamp
                    )
            
            matches.append(best_match)
        
        return matches


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='视频混剪重构工具 - 音频+视频混合对齐版')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--target', '-t', help='目标视频路径')
    parser.add_argument('--sources', '-s', nargs='+', help='源视频路径列表')
    parser.add_argument('--output', '-o', help='输出视频路径')
    parser.add_argument('--fps', type=int, help='帧提取率')
    parser.add_argument('--threshold', type=float, help='匹配阈值')
    parser.add_argument('--workers', '-w', type=int, help='并行线程数')
    parser.add_argument('--no-audio', action='store_true', help='禁用音频对齐，只用视频')
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        config = load_config(args.config)
    
    if args.fps is not None:
        config['fps'] = args.fps
    if args.threshold is not None:
        config['match_threshold'] = args.threshold
    if args.workers is not None:
        config['max_workers'] = args.workers
    if args.no_audio:
        config['use_audio_first'] = False
    
    if args.config:
        target_video = config.get('target_video')
        source_videos = config.get('source_videos', [])
        output_video = config.get('output_video', 'reconstructed.mp4')
    else:
        target_video = args.target
        source_videos = args.sources
        output_video = args.output or 'reconstructed.mp4'
    
    if not target_video or not source_videos:
        print("❌ 请提供目标视频和源视频")
        parser.print_help()
        return
    
    if not Path(target_video).exists():
        print(f"❌ 目标视频不存在: {target_video}")
        return
    
    for sv in source_videos:
        if not Path(sv).exists():
            print(f"❌ 源视频不存在: {sv}")
            return
    
    print("=" * 60)
    print("🎬 视频混剪重构工具 - 音频+视频混合对齐版")
    print("=" * 60)
    print(f"目标视频: {target_video}")
    print(f"源视频: {len(source_videos)}个")
    print(f"输出视频: {output_video}")
    print(f"音频对齐: {'启用' if config.get('use_audio_first', True) else '禁用'}")
    print("=" * 60)
    
    reconstructor = VideoReconstructorHybrid(target_video, source_videos, config)
    segments = reconstructor.reconstruct(output_video, use_target_audio=True)
    
    if segments:
        print("\n✅ 重构成功!")
    else:
        print("\n❌ 重构失败")


if __name__ == "__main__":
    main()
