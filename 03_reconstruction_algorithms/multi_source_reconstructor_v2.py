#!/usr/bin/env python3
"""
多源视频混剪重构工具 V2（改进版）
从多个原视频中找到素材，重新混剪出与目标视频相同长度、相似内容的新版本
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import json
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

@dataclass
class FrameFeature:
    """帧特征""" 
    frame_path: Path
    timestamp: float
    hist: np.ndarray
    phash: np.ndarray
    template_key: np.ndarray

@dataclass
class VideoSegment:
    source_video: Path
    start_time: float
    end_time: float
    similarity_score: float
    segment_idx: int
    is_filled: bool = False

@dataclass
class FrameMatch:
    target_frame_idx: int
    source_video: Path
    source_frame_idx: int
    similarity: float
    timestamp: float

@dataclass
class ValidationResult:
    overall_similarity: float
    frame_by_frame_similarity: List[float]
    duration_match: bool
    duration_diff: float
    passed: bool

class Config:
    def __init__(self, config_file: str = "video_reconstruct_config_v2.yaml"):
        self.config_file = Path(config_file)
        self.config = self.load_config()

    def load_config(self) -> dict:
        default_config = {
            'target_video': '',
            'source_videos': [],
            'output_video': 'reconstructed_v2.mp4',
            'fps': 8,
            'similarity_threshold': 0.85,
            'max_retries': 3,
            'use_target_audio': True,
            'min_segment_duration': 0.3,
            'match_threshold': 0.45,
            'scale': '480:270',
            'missing_segment_strategy': 'smart_fill',
            'smart_fill_threshold': 0.4,
            'enable_post_processing': True,
            'post_process_threshold': 0.5,
            'max_gap_fill_duration': 2.0,
        }

        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)

        return default_config

    def save_config(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value):
        self.config[key] = value

class MultiSourceVideoReconstructorV2:
    def __init__(self, config: Config):
        self.config = config
        self.target_video = Path(config.get('target_video'))
        self.source_videos = [Path(v) for v in config.get('source_videos', [])]
        self.output_video = Path(config.get('output_video', 'reconstructed_v2.mp4'))
        self.fps = config.get('fps', 8)
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        self.max_retries = config.get('max_retries', 3)
        self.use_target_audio = config.get('use_target_audio', True)
        self.min_segment_duration = config.get('min_segment_duration', 0.3)
        self.match_threshold = config.get('match_threshold', 0.45)
        self.scale = config.get('scale', '480:270')
        self.missing_strategy = config.get('missing_segment_strategy', 'smart_fill')
        self.smart_fill_threshold = config.get('smart_fill_threshold', 0.4)
        self.enable_post_processing = config.get('enable_post_processing', True)
        self.post_process_threshold = config.get('post_process_threshold', 0.5)
        self.max_gap_fill_duration = config.get('max_gap_fill_duration', 2.0)
        self.max_workers = config.get('max_workers', 4)

        self.temp_dir = None
        self.target_duration = 0
        self.reconstruction_log = []
        self.frame_cache = {}

    def get_video_duration(self, video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果FPS为None或0，使用默认值
        if fps is None or fps <= 0:
            fps = 30.0
            print(f"⚠️  {video_path.name} 无法获取FPS，使用默认值 {fps}")
        
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration

    def extract_frame_features(self, frame_path: Path, timestamp: float) -> Optional[FrameFeature]:
        """提取单帧特征（用于并行处理）"""
        img = cv2.imread(str(frame_path))
        if img is None:
            return None

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # 计算感知哈希（简化版）
        small = cv2.resize(gray, (8, 8))
        small_float = np.float32(small)
        dct = cv2.dct(small_float)
        low_freq = dct[:4, :4].flatten()
        mean = np.mean(low_freq)
        phash = (low_freq > mean).astype(int)

        # 计算模板关键点（简化版）
        template_key = cv2.resize(gray, (16, 16)).flatten()
        template_key = template_key / (np.linalg.norm(template_key) + 1e-10)

        return FrameFeature(
            frame_path=frame_path,
            timestamp=timestamp,
            hist=hist,
            phash=phash,
            template_key=template_key
        )

    def extract_features_parallel(self, frames: List[Tuple[Path, float]], desc: str = "提取特征") -> List[FrameFeature]:
        """并行提取帧特征"""
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
        return features

    def calculate_similarity_fast(self, feat1: FrameFeature, feat2: FrameFeature) -> float:
        """快速相似度计算"""
        # 直方图相似度
        hist_sim = cv2.compareHist(
            feat1.hist.reshape(-1, 1),
            feat2.hist.reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )

        # 感知哈希相似度
        hamming = np.sum(feat1.phash != feat2.phash)
        phash_sim = 1 - (hamming / 16.0)

        # 模板关键点相似度
        template_sim = np.dot(feat1.template_key, feat2.template_key)

        # 综合评分
        return 0.4 * max(0, hist_sim) + 0.3 * phash_sim + 0.3 * max(0, template_sim)

    def extract_frames_with_timestamp(self, video_path: Path, output_dir: Path) -> List[Tuple[Path, float]]:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 如果FPS为None或0，使用默认值
        if fps is None or fps <= 0:
            fps = 30.0  # 使用默认FPS
            print(f"⚠️  {video_path.name} 无法获取FPS，使用默认值 {fps}")

        # 如果self.fps为None或0，使用默认值
        if self.fps is None or self.fps <= 0:
            self.fps = 8.0  # 使用默认FPS
            print(f"⚠️  配置中fps为None，使用默认值 {self.fps}")

        frame_interval = int(fps / self.fps) if fps > 0 else 1
        if frame_interval < 1:
            frame_interval = 1

        frames = []
        frame_idx = 0
        extracted_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps if fps > 0 else 0
                frame_path = output_dir / f"{video_path.stem}_{extracted_idx:06d}.jpg"
                frame_resized = cv2.resize(frame, (480, 270))
                cv2.imwrite(str(frame_path), frame_resized)
                frames.append((frame_path, timestamp))
                extracted_idx += 1

            frame_idx += 1

        cap.release()
        return frames

    def calculate_frame_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        cache_key = f"{frame1_path}_{frame2_path}"
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]

        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))

        if img1 is None or img2 is None:
            return 0.0

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)


        phash_sim = self._phash_similarity(gray1, gray2)

        ssim_sim = self._ssim_similarity(gray1, gray2)

        final_sim = 0.25 * max(0, hist_sim) + 0.2 * template_sim + 0.25 * phash_sim + 0.3 * ssim_sim

        self.frame_cache[cache_key] = final_sim
        return final_sim

    def _phash_similarity(self, img1, img2) -> float:
        size = (32, 32)
        img1_small = cv2.resize(img1, size)
        img2_small = cv2.resize(img2, size)
        img1_float = np.float32(img1_small)
        img2_float = np.float32(img2_small)
        dct1 = cv2.dct(img1_float)
        dct2 = cv2.dct(img2_float)
        low_freq1 = dct1[:8, :8].flatten()
        low_freq2 = dct2[:8, :8].flatten()
        mean1 = np.mean(low_freq1)
        mean2 = np.mean(low_freq2)
        hash1 = (low_freq1 > mean1).astype(int)
        hash2 = (low_freq2 > mean2).astype(int)
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1 - (hamming_dist / 64.0)
        return max(0, similarity)

    def _ssim_similarity(self, img1, img2) -> float:
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)

    def find_best_matches(self, target_frames: List[Tuple[Path, float]],
                         source_frames_dict: dict) -> List[FrameMatch]:
        matches = []
        print(f"🔍 正在比对 {len(target_frames)} 帧目标画面...")
        
        # 并行提取目标帧特征
        print("📊 提取目标帧特征...")
        target_features = self.extract_features_parallel(target_frames, "目标帧")
        
        # 并行提取源帧特征
        print("📊 提取源帧特征...")
        source_features_dict = {}
        for source_video, source_frames in source_frames_dict.items():
            source_features_dict[source_video] = self.extract_features_parallel(source_frames, f"{source_video.name}")

        for target_idx, target_feat in enumerate(target_features):
            if target_idx % (int(self.fps) * 2 if self.fps else 16) == 0:
                print(f"   处理中: {target_idx}/{len(target_frames)} 帧 ({target_idx/len(target_frames)*100:.1f}%)")

            best_match = None
            best_score = 0.0

            for source_video, source_features in source_features_dict.items():
                # 提前退出：如果已经找到足够好的匹配，跳过剩余的源视频
                if best_score > 0.95:
                    break
                    
                for source_idx, source_feat in enumerate(source_features):
                    similarity = self.calculate_similarity_fast(target_feat, source_feat)

                    if similarity > best_score and similarity > self.match_threshold:
                        best_score = similarity
                        best_match = FrameMatch(
                            target_frame_idx=target_idx,
                            source_video=source_video,
                            source_frame_idx=source_idx,
                            similarity=best_score,
                            timestamp=source_feat.timestamp
                        )
                        
                        # 提前退出：如果找到非常高的相似度，跳过剩余的帧
                        if best_score > 0.95:
                            break

            if best_match:
                matches.append(best_match)
            else:
                matches.append(None)

        return matches

    def segment_matches(self, matches: List[FrameMatch]) -> List[VideoSegment]:
        segments = []
        current_segment = []
        
        # 确保fps不为None
        fps = self.fps if self.fps else 8.0

        for idx, match in enumerate(matches):
            if match is None:
                if len(current_segment) >= int(self.min_segment_duration * fps):
                    self._save_segment(current_segment, segments)
                current_segment = []
            else:
                if current_segment and self._is_continuous(current_segment[-1], match):
                    current_segment.append(match)
                else:
                    if len(current_segment) >= int(self.min_segment_duration * fps):
                        self._save_segment(current_segment, segments)
                    current_segment = [match]

        if len(current_segment) >= int(self.min_segment_duration * fps):
            self._save_segment(current_segment, segments)

        return segments

    def _is_continuous(self, match1: FrameMatch, match2: FrameMatch) -> bool:
        if match1.source_video != match2.source_video:
            return False
        time_diff = abs(match2.timestamp - match1.timestamp)
        fps = self.fps if self.fps else 8.0
        return time_diff <= (2.0 / fps)

    def _save_segment(self, segment_matches: List[FrameMatch], segments: List[VideoSegment]):
        if not segment_matches:
            return

        first_match = segment_matches[0]
        last_match = segment_matches[-1]
        avg_similarity = np.mean([m.similarity for m in segment_matches])
        
        fps = self.fps if self.fps else 8.0

        segment = VideoSegment(
            source_video=first_match.source_video,
            start_time=first_match.timestamp,
            end_time=last_match.timestamp + (1.0 / fps),
            similarity_score=avg_similarity,
            segment_idx=len(segments),
            is_filled=False
        )
        segments.append(segment)

    def reconstruct_with_validation(self) -> bool:
        """重构视频并进行验证
        
        Returns:
            bool: 是否成功
        """
        try:
            # 创建临时目录
            self.temp_dir = Path(tempfile.mkdtemp(prefix='video_reconstruct_'))
            print(f"📁 临时目录: {self.temp_dir}")

            # 获取目标视频时长
            self.target_duration = self.get_video_duration(self.target_video)
            print(f"⏱️ 目标视频时长: {self.target_duration:.2f}秒")

            # 提取目标视频帧
            target_frames_dir = self.temp_dir / "target_frames"
            target_frames_dir.mkdir(parents=True, exist_ok=True)
            print("📸 正在提取目标视频帧...")
            target_frames = self.extract_frames_with_timestamp(self.target_video, target_frames_dir)
            print(f"✅ 提取了 {len(target_frames)} 帧目标画面")

            # 提取源视频帧
            source_frames_dir = self.temp_dir / "source_frames"
            source_frames_dir.mkdir(parents=True, exist_ok=True)
            print("📸 正在提取源视频帧...")
            source_frames_dict = {}
            for source_video in self.source_videos:
                # 为每个源视频创建单独的子目录
                video_frames_dir = source_frames_dir / source_video.stem
                video_frames_dir.mkdir(parents=True, exist_ok=True)
                source_frames = self.extract_frames_with_timestamp(source_video, video_frames_dir)
                source_frames_dict[source_video] = source_frames
                print(f"✅ {source_video.name}: {len(source_frames)} 帧")

            # 查找最佳匹配
            matches = self.find_best_matches(target_frames, source_frames_dict)
            matched_count = sum(1 for m in matches if m is not None)
            print(f"📊 匹配率: {matched_count}/{len(matches)} ({matched_count/len(matches)*100:.1f}%)")

            # 分段
            segments = self.segment_matches(matches)
            print(f"📊 找到 {len(segments)} 个有效片段")

            if not segments:
                print("❌ 未找到任何有效片段")
                return False

            # 生成FFmpeg命令
            ffmpeg_cmd = self._generate_ffmpeg_command(segments)
            print(f"🎬 正在生成重构视频...")

            # 执行FFmpeg
            result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ FFmpeg执行失败: {result.stderr}")
                return False

            print(f"✅ 重构视频已保存到: {self.output_video}")

            # 验证重构结果
            validation = self._validate_reconstruction()
            print(f"📊 验证结果: 相似度={validation.overall_similarity:.2f}, 通过={validation.passed}")

            # 清理临时文件
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

            return validation.passed

        except Exception as e:
            print(f"❌ 重构过程出错: {str(e)}")
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            return False

    def _generate_ffmpeg_command(self, segments: List[VideoSegment]) -> str:
        """生成FFmpeg拼接命令 - 修复版"""
        concat_list = []
        for seg in segments:
            concat_list.append(f"file '{seg.source_video}'")
            concat_list.append(f"inpoint {seg.start_time:.3f}")
            concat_list.append(f"outpoint {seg.end_time:.3f}")

        concat_file = self.temp_dir / "concat_list.txt"
        with open(concat_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(concat_list))

        scale = self.scale.split(':')
        scale_width, scale_height = scale[0], scale[1]

        fps = self.fps if self.fps else 8.0
        
        # 分两步处理：先concat，再处理视频
        temp_concat = self.temp_dir / "temp_concat.mp4"
        
        # 第一步：只concat，不处理视频
        cmd1 = f"ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i '{concat_file}' -c copy '{temp_concat}'"
        
        # 第二步：处理视频（缩放、fps）并添加音频
        if self.use_target_audio:
            cmd2 = (
                f"ffmpeg -y -hide_banner -loglevel error "
                f"-i '{temp_concat}' -i '{self.target_video}' "
                f"-vf 'scale={scale_width}:{scale_height}:force_original_aspect_ratio=decrease,pad={scale_width}:{scale_height}:(ow-iw)/2:(oh-ih)/2' "
                f"-r {fps} -c:v libx264 -preset fast -crf 23 "
                f"-c:a aac -map 0:v:0 -map 1:a:0 -shortest "
                f"'{self.output_video}'"
            )
        else:
            cmd2 = (
                f"ffmpeg -y -hide_banner -loglevel error "
                f"-i '{temp_concat}' "
                f"-vf 'scale={scale_width}:{scale_height}:force_original_aspect_ratio=decrease,pad={scale_width}:{scale_height}:(ow-iw)/2:(oh-ih)/2' "
                f"-r {fps} -c:v libx264 -preset fast -crf 23 "
                f"-c:a copy "
                f"'{self.output_video}'"
            )
        
        # 返回组合命令
        return f"{cmd1} && {cmd2}"

    def _validate_reconstruction(self) -> ValidationResult:
        """验证重构结果"""
        output_duration = self.get_video_duration(self.output_video)
        duration_diff = abs(output_duration - self.target_duration)
        duration_match = duration_diff < 1.0

        # 计算相似度
        target_frames_dir = self.temp_dir / "target_frames"
        output_frames_dir = self.temp_dir / "output_frames"
        target_frames_dir.mkdir(parents=True, exist_ok=True)
        output_frames_dir.mkdir(parents=True, exist_ok=True)

        target_frames = self.extract_frames_with_timestamp(self.target_video, target_frames_dir)
        output_frames = self.extract_frames_with_timestamp(self.output_video, output_frames_dir)

        frame_similarities = []
        min_frames = min(len(target_frames), len(output_frames))

        for i in range(min_frames):
            sim = self.calculate_frame_similarity(target_frames[i][0], output_frames[i][0])
            frame_similarities.append(sim)

        overall_similarity = np.mean(frame_similarities) if frame_similarities else 0.0
        passed = overall_similarity >= self.similarity_threshold and duration_match

        return ValidationResult(
            overall_similarity=overall_similarity,
            frame_by_frame_similarity=frame_similarities,
            duration_match=duration_match,
            duration_diff=duration_diff,
            passed=passed
        )
