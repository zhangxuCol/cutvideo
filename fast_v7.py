#!/usr/bin/env python3
"""
极速高精度重构器 V3 - 3分钟/视频目标
优化策略：
1. 预计算音频指纹缓存
2. 并行处理分段
3. 智能候选预筛选
4. 快速画面验证（3个时间点）
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import wave
import struct
import json
import pickle
import re
import hashlib
import threading
from datetime import datetime
from collections import OrderedDict
from PIL import Image
import imagehash
from pipeline_config import (
    load_section_config,
    cfg_str,
    cfg_int,
    cfg_float,
    cfg_bool,
    cfg_str_list,
    split_csv,
)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


def extract_chromaprint(video_path: Path, start: float = 0, duration: float = None) -> list:
    """使用 Chromaprint (fpcalc) 提取音频指纹"""
    try:
        cmd = ['fpcalc', '-raw']
        if start > 0:
            cmd.extend(['-start', str(int(start))])
        if duration is not None:
            cmd.extend(['-length', str(int(duration))])
        cmd.append(str(video_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return []
        
        # 解析 fpcalc 输出
        # 格式: DURATION=xxx\nFINGERPRINT=xxx
        output = result.stdout
        fingerprint_match = re.search(r'FINGERPRINT=(.+)', output)
        
        if fingerprint_match:
            # 指纹是以逗号分隔的整数列表
            fingerprint_str = fingerprint_match.group(1)
            fingerprint = [int(x) for x in fingerprint_str.split(',')]
            return fingerprint
        return []
    except Exception as e:
        print(f"   Chromaprint 提取失败: {e}")
        return []

def compare_chromaprint(fp1: list, fp2: list) -> float:
    """比较两个 Chromaprint 指纹的相似度"""
    if not fp1 or not fp2:
        return 0.0
    
    # 使用汉明距离计算相似度
    # Chromaprint 指纹是 32-bit 整数列表
    min_len = min(len(fp1), len(fp2))
    if min_len == 0:
        return 0.0
    
    matches = 0
    for i in range(min_len):
        # 计算汉明距离
        xor = fp1[i] ^ fp2[i]
        # 统计不同的 bit 数
        diff_bits = bin(xor).count('1')
        # 相似度 = 1 - (不同bit数 / 32)
        sim = 1.0 - (diff_bits / 32.0)
        matches += sim
    
    return matches / min_len

@dataclass
class SegmentTask:
    index: int
    target_start: float
    duration: float
    
@dataclass
class SegmentResult:
    index: int
    success: bool
    source: Path = None
    source_start: float = 0
    quality: dict = None

class FastHighPrecisionReconstructor:
    """
    极速高精度重构器
    """
    
    def __init__(self, target_video: str, source_videos: List[str], cache_dir: str = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = Path(cache_dir) if cache_dir else self.temp_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 配置
        self.match_threshold = 0.95
        self.segment_duration = 5.0  # 降低分段时长提高精度
        cpu_count = os.cpu_count() or 8
        # v7: 提高默认并发（在不改算法的前提下优先提速）
        auto_workers = max(1, cpu_count - 1)
        self.max_workers = min(16, max(6, auto_workers))
        self.render_workers = min(4, max(1, cpu_count // 4))
        self.low_score_threshold = 0.82
        self.rematch_window = 2
        self.rematch_max_window = 12
        self.continuity_weight = 0.05
        self.total_segments = 0
        self.use_audio_matching = False
        self.force_target_audio = False
        self.strict_visual_verify = True
        self.strict_verify_min_sim = 0.78
        self.tail_guard_seconds = 20.0
        self.tail_verify_min_avg = 0.84
        self.tail_verify_min_floor = 0.78
        self.target_duration = 0.0
        self.output_fps = 24.0
        # 相邻段时间轴守卫阈值：用于拦截“轻微慢进导致的缺内容+重复画面”
        self.adjacent_overlap_trigger = 0.6
        self.adjacent_lag_trigger = 0.8
        # 孤立段（无同源邻段）允许的最大时间漂移；超出则回退目标段
        self.isolated_drift_trigger = 0.8
        # 跨源切换时允许的映射偏移突变；超出则判定为边界时间回跳风险
        self.cross_source_mapping_jump_trigger = 0.75
        # 绝对映射漂移守卫：拦截“整体看着像对，但时间轴提前/滞后约 1 秒”的段
        self.max_mapping_drift_trigger = 0.75
        self.max_mapping_drift_combined_floor = 0.95
        # 段边界单帧突刺修复（A-B-A 单帧跳帧）
        self.boundary_glitch_fix = True
        self.boundary_glitch_hi_threshold = 0.965
        self.boundary_glitch_lo_threshold = 0.94
        self.boundary_glitch_gap_threshold = 0.03
        self.boundary_glitch_use_videotoolbox = True
        self.run_ai_verify_snapshots = False
        # 轻量音频守卫：拦截“画面对了但台词错位”
        # 只在可疑段触发，避免显著增加整体耗时。
        self.audio_guard_enabled = True
        self.audio_guard_score_trigger = 0.93
        self.audio_guard_sample_duration = 1.8
        self.audio_guard_min_similarity = 0.34
        self.audio_guard_hard_floor = 0.18
        self.audio_guard_shift_margin = 0.16
        self.audio_guard_shift_candidates = [-2.0, -1.0, 1.0, 2.0]

        # 运行期统计（用于质量报告）
        self.match_elapsed_sec = 0.0
        self.timeline_guard_elapsed_sec = 0.0
        self.total_elapsed_sec = 0.0
        self.guard_stats: Dict[str, int] = {}
        self.last_render_metrics: Dict[str, object] = {}

        # pHash 帧索引
        self.frame_index = {}  # {video_path: [(time, phash), ...]}

        # 缓存
        self.source_fingerprints = {}
        self.target_fingerprint = None
        self.audio_fp_cache_max_items = 6000
        self.audio_fp_cache: "OrderedDict[Tuple[str, float, float], List[int]]" = OrderedDict()
        self.audio_fp_cache_lock = threading.Lock()

        # v7: 帧提取缓存（减少重复 ffmpeg 抽帧）
        self.frame_cache_dir = self.temp_dir / "frame_cache"
        self.frame_cache_dir.mkdir(exist_ok=True)
        self.frame_cache_max_items = 4000
        self.frame_cache_index: "OrderedDict[Tuple[str, float], Path]" = OrderedDict()
        self.frame_cache_lock = threading.Lock()
        self.frame_build_locks: Dict[Tuple[str, float], threading.Lock] = {}
        self.frame_build_locks_guard = threading.Lock()
        self.output_path_locks: Dict[str, threading.Lock] = {}
        self.output_path_locks_guard = threading.Lock()
        self.meta_cache_lock = threading.Lock()
        self.video_duration_cache: Dict[str, float] = {}
        self.audio_duration_cache: Dict[str, float] = {}
        self.video_fps_cache: Dict[str, float] = {}

        # v7.3: 帧特征缓存（减少重复 imread/resize/cvtColor）
        self.frame_feature_cache_max_items = 1500
        self.frame_feature_cache: "OrderedDict[str, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self.frame_feature_lock = threading.Lock()
        self._ffmpeg_encoder_cache: Dict[str, bool] = {}

    def continuity_bonus(self, source_start: float, target_start: float, duration: float) -> float:
        """连续性奖励（软约束），鼓励时间映射更平滑，不做硬性限制。"""
        tolerance = max(duration * 8, 20.0)
        delta = abs(source_start - target_start)
        return max(0.0, 1.0 - (delta / tolerance)) * self.continuity_weight
        
    def get_video_duration(self, video_path: Path) -> float:
        key = str(Path(video_path).resolve())
        with self.meta_cache_lock:
            cached = self.video_duration_cache.get(key)
            if cached is not None:
                return float(cached)

        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            duration = max(0.0, float((result.stdout or "").strip()))
        except Exception:
            duration = 0.0
        with self.meta_cache_lock:
            self.video_duration_cache[key] = float(duration)
        return float(duration)

    def get_audio_duration(self, video_path: Path) -> float:
        """读取首条音轨时长；无音轨或解析失败时返回 0。"""
        key = str(Path(video_path).resolve())
        with self.meta_cache_lock:
            cached = self.audio_duration_cache.get(key)
            if cached is not None:
                return float(cached)

        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        value = (result.stdout or "").strip().splitlines()
        if not value:
            duration = 0.0
            with self.meta_cache_lock:
                self.audio_duration_cache[key] = duration
            return duration
        raw = value[0].strip()
        if not raw or raw.upper() == "N/A":
            duration = 0.0
            with self.meta_cache_lock:
                self.audio_duration_cache[key] = duration
            return duration
        try:
            dur = float(raw)
            duration = max(0.0, dur)
        except Exception:
            duration = 0.0
        with self.meta_cache_lock:
            self.audio_duration_cache[key] = float(duration)
        return float(duration)

    def _audio_dbfs_sample(self, video_path: Path, start_sec: float, duration_sec: float = 8.0) -> Optional[float]:
        """抽样解码音频并计算 dBFS；失败时返回 None。"""
        cmd = [
            'ffmpeg', '-v', 'error',
            '-ss', f"{max(0.0, float(start_sec)):.3f}",
            '-t', f"{max(0.2, float(duration_sec)):.3f}",
            '-i', str(video_path),
            '-vn',
            '-ac', '1',
            '-ar', '16000',
            '-f', 's16le',
            '-',
        ]
        try:
            result = subprocess.run(cmd, capture_output=True)
            raw = result.stdout or b""
            if not raw:
                return None
            pcm = np.frombuffer(raw, dtype=np.int16)
            if pcm.size <= 0:
                return None
            rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float64)))))
            if rms <= 1e-9:
                return -120.0
            return float(20.0 * np.log10(rms / 32768.0))
        except Exception:
            return None

    def _check_output_audio_audible(self, video_path: Path, target_duration: float) -> Tuple[bool, List[Optional[float]]]:
        """
        快速可听性检查：
        - 在头/中/尾三段采样 dBFS
        - 任一点 > -45dBFS 认为“有可听音频”
        """
        duration = max(0.0, float(target_duration))
        if duration <= 0.1:
            duration = self.get_video_duration(video_path)
        mid = max(0.0, duration * 0.5 - 4.0)
        tail = max(0.0, duration - 12.0)
        points = [0.0, mid, tail]
        db_samples: List[Optional[float]] = [self._audio_dbfs_sample(video_path, t, 8.0) for t in points]
        audible = any((v is not None) and (float(v) > -45.0) for v in db_samples)
        return bool(audible), db_samples

    def _remux_with_target_audio(self, video_path: Path, output_path: Path, target_duration: float) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(video_path),
            '-i', str(self.target_video),
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            '-t', f"{max(0.0, float(target_duration)):.3f}",
            '-shortest',
            str(output_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 and output_path.exists()
        except Exception:
            return False

    def get_video_fps(self, video_path: Path) -> float:
        """读取视频平均帧率，失败时回退 24fps。"""
        key = str(Path(video_path).resolve())
        with self.meta_cache_lock:
            cached = self.video_fps_cache.get(key)
            if cached is not None:
                return float(cached)

        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=avg_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        rate = (result.stdout or "").strip()
        if not rate:
            fps = 24.0
            with self.meta_cache_lock:
                self.video_fps_cache[key] = fps
            return fps
        try:
            if '/' in rate:
                num, den = rate.split('/', 1)
                fps = float(num) / float(den)
            else:
                fps = float(rate)
            if fps <= 0:
                fps = 24.0
            fps = min(120.0, max(12.0, fps))
        except Exception:
            fps = 24.0
        with self.meta_cache_lock:
            self.video_fps_cache[key] = float(fps)
        return float(fps)

    def _ffmpeg_has_encoder(self, encoder_name: str) -> bool:
        name = (encoder_name or "").strip()
        if not name:
            return False
        cached = self._ffmpeg_encoder_cache.get(name)
        if cached is not None:
            return bool(cached)
        try:
            probe = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
            )
            text = (probe.stdout or "") + "\n" + (probe.stderr or "")
            ok = (probe.returncode == 0) and (name in text)
        except Exception:
            ok = False
        self._ffmpeg_encoder_cache[name] = bool(ok)
        return bool(ok)

    def _normalize_frame_time(self, time_sec: float) -> float:
        return round(max(0.0, float(time_sec)), 3)

    def _frame_cache_key(self, video_path: Path, time_sec: float) -> Tuple[str, float]:
        return str(Path(video_path).resolve()), self._normalize_frame_time(time_sec)

    def _frame_cache_file(self, key: Tuple[str, float]) -> Path:
        digest = hashlib.sha1(f"{key[0]}|{key[1]:.3f}".encode("utf-8")).hexdigest()
        return self.frame_cache_dir / f"{digest}.jpg"

    def _remember_cached_frame(self, key: Tuple[str, float], cache_path: Path) -> None:
        if not cache_path.exists():
            return
        with self.frame_cache_lock:
            self.frame_cache_index[key] = cache_path
            self.frame_cache_index.move_to_end(key)
            while len(self.frame_cache_index) > self.frame_cache_max_items:
                _, old_path = self.frame_cache_index.popitem(last=False)
                old_path.unlink(missing_ok=True)

    def _lookup_cached_frame(self, key: Tuple[str, float]) -> Optional[Path]:
        with self.frame_cache_lock:
            cached = self.frame_cache_index.get(key)
            if cached and cached.exists():
                self.frame_cache_index.move_to_end(key)
                return cached
            if cached and not cached.exists():
                self.frame_cache_index.pop(key, None)
        return None

    def _get_frame_build_lock(self, key: Tuple[str, float]) -> threading.Lock:
        with self.frame_build_locks_guard:
            lk = self.frame_build_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self.frame_build_locks[key] = lk
            return lk

    def _get_output_path_lock(self, output_path: Path) -> threading.Lock:
        key = str(Path(output_path).resolve())
        with self.output_path_locks_guard:
            lk = self.output_path_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self.output_path_locks[key] = lk
            return lk

    def _extract_single_frame_raw(self, video_path: Path, time_sec: float, output_path: Path) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', f"{self._normalize_frame_time(time_sec):.3f}", '-i', str(video_path),
            '-vframes', '1', '-vf', 'scale=360:202',
            str(output_path)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode == 0 and output_path.exists()

    def get_cached_frame_path(self, video_path: Path, time_sec: float) -> Optional[Path]:
        """确保缓存中存在该帧，返回缓存路径。"""
        key = self._frame_cache_key(video_path, time_sec)
        cached = self._lookup_cached_frame(key)
        if cached and cached.exists():
            return cached

        cache_path = self._frame_cache_file(key)
        if cache_path.exists():
            self._remember_cached_frame(key, cache_path)
            return cache_path

        # 按帧键串行构建，避免多线程同时写同一文件导致 JPEG 损坏。
        build_lock = self._get_frame_build_lock(key)
        with build_lock:
            if cache_path.exists():
                self._remember_cached_frame(key, cache_path)
                return cache_path

            tmp_path = self.frame_cache_dir / (
                f"{cache_path.stem}.tmp_{os.getpid()}_{threading.get_ident()}.jpg"
            )
            tmp_path.unlink(missing_ok=True)
            ok = self._extract_single_frame_raw(video_path, time_sec, tmp_path)
            if ok and tmp_path.exists():
                os.replace(tmp_path, cache_path)
                self._remember_cached_frame(key, cache_path)
                return cache_path
            tmp_path.unlink(missing_ok=True)
        return None

    def _materialize_frame(self, source_path: Path, output_path: Path) -> None:
        """将缓存帧映射到输出路径，优先使用软链接以减少 IO。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            try:
                if output_path.samefile(source_path):
                    return
            except Exception:
                pass
        output_path.unlink(missing_ok=True)
        try:
            os.symlink(str(source_path), str(output_path))
        except Exception:
            try:
                if output_path.exists() and output_path.samefile(source_path):
                    return
            except Exception:
                pass
            shutil.copyfile(source_path, output_path)

    def extract_frames_batch(self, video_path: Path, time_points: List[float], output_paths: List[Path]) -> None:
        """批量抽帧：逐点命中缓存（v7.2，避免重型拼接命令导致慢速）。"""
        if not time_points or not output_paths or len(time_points) != len(output_paths):
            return

        for idx, (time_sec, out_path) in enumerate(zip(time_points, output_paths)):
            _ = idx
            cached = self.get_cached_frame_path(video_path, time_sec)
            if cached and cached.exists():
                self._materialize_frame(cached, out_path)
    
    def extract_audio_fingerprint(self, video_path: Path, start: float = 0, duration: float = None) -> list:
        """提取音频指纹 - 使用 Chromaprint"""
        duration_str = f"{duration:.0f}" if duration is not None else "full"
        cache_key = f"{video_path.stem}_{start:.0f}_{duration_str}"
        cache_file = self.cache_dir / f"{cache_key}_chromaprint.pkl"
        
        # 检查缓存
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 使用 Chromaprint 提取
        fingerprint = extract_chromaprint(video_path, start, duration)
        
        # 保存缓存
        if fingerprint:
            with open(cache_file, 'wb') as f:
                pickle.dump(fingerprint, f)
        
        return fingerprint

    def _get_audio_fp_cached(self, video_path: Path, start: float, duration: float) -> list:
        """
        读取/缓存短音频 Chromaprint，避免段级音频守卫重复调用 fpcalc。
        """
        clip_start = round(max(0.0, float(start)), 3)
        clip_duration = round(max(0.2, float(duration)), 3)
        key = (str(Path(video_path).resolve()), clip_start, clip_duration)
        with self.audio_fp_cache_lock:
            cached = self.audio_fp_cache.get(key)
            if cached is not None:
                self.audio_fp_cache.move_to_end(key)
                return cached

        fp = extract_chromaprint(video_path, clip_start, clip_duration)

        with self.audio_fp_cache_lock:
            self.audio_fp_cache[key] = fp
            self.audio_fp_cache.move_to_end(key)
            while len(self.audio_fp_cache) > self.audio_fp_cache_max_items:
                self.audio_fp_cache.popitem(last=False)
        return fp

    def quick_verify_audio(
        self,
        source: Path,
        source_start: float,
        target_start: float,
        duration: float,
        combined_score: float,
    ) -> Tuple[bool, Dict]:
        """
        轻量音频一致性守卫：
        - 默认仅对可疑段触发（低分或明显时间漂移）
        - 对齐点相似度过低时，再做 ±1/2 秒偏移探测
        """
        meta: Dict[str, object] = {
            "enabled": bool(self.audio_guard_enabled),
            "checked": False,
        }
        if not self.audio_guard_enabled:
            meta["skipped_reason"] = "disabled"
            return True, meta
        if source == self.target_video:
            meta["skipped_reason"] = "target_segment"
            return True, meta

        drift = abs(float(source_start) - float(target_start))
        drift_trigger = max(0.9, float(duration) * 0.25)
        need_check = (
            float(combined_score) < float(self.audio_guard_score_trigger)
            or drift > drift_trigger
        )
        if not need_check:
            meta["skipped_reason"] = "high_confidence"
            meta["drift"] = float(drift)
            return True, meta

        clip_dur = max(0.8, min(float(duration), float(self.audio_guard_sample_duration)))
        offset = max(0.0, (float(duration) - clip_dur) * 0.5)
        tgt_clip_start = float(target_start) + offset
        src_clip_start = float(source_start) + offset

        target_fp = self._get_audio_fp_cached(self.target_video, tgt_clip_start, clip_dur)
        source_fp = self._get_audio_fp_cached(source, src_clip_start, clip_dur)

        meta.update({
            "checked": True,
            "clip_duration": float(clip_dur),
            "clip_offset": float(offset),
            "drift": float(drift),
            "aligned_start": float(src_clip_start),
        })

        if len(target_fp) < 10 or len(source_fp) < 10:
            # 短静音/无声段容易无有效指纹；此处不做硬拦截，交给画面守卫与时间轴守卫。
            meta["available"] = False
            meta["skipped_reason"] = "fingerprint_unavailable"
            return True, meta

        aligned_sim = float(compare_chromaprint(target_fp, source_fp))
        best_shift = 0.0
        best_shift_sim = aligned_sim

        # 仅在对齐相似度偏低时才探测偏移，控制额外开销。
        if aligned_sim < float(self.audio_guard_min_similarity):
            for shift in self.audio_guard_shift_candidates:
                shifted_start = src_clip_start + float(shift)
                if shifted_start < 0:
                    continue
                shifted_fp = self._get_audio_fp_cached(source, shifted_start, clip_dur)
                if len(shifted_fp) < 10:
                    continue
                shifted_sim = float(compare_chromaprint(target_fp, shifted_fp))
                if shifted_sim > best_shift_sim:
                    best_shift_sim = shifted_sim
                    best_shift = float(shift)

        meta.update({
            "available": True,
            "aligned_similarity": float(aligned_sim),
            "best_shift_sec": float(best_shift),
            "best_shift_similarity": float(best_shift_sim),
        })

        fail_reason = ""
        if aligned_sim < float(self.audio_guard_hard_floor):
            fail_reason = "audio_guard_hard_floor"
        elif (
            aligned_sim < float(self.audio_guard_min_similarity)
            and abs(best_shift) >= 1.0
            and (best_shift_sim - aligned_sim) >= float(self.audio_guard_shift_margin)
        ):
            fail_reason = "audio_guard_shift_bias"
        elif (
            aligned_sim < float(self.audio_guard_min_similarity)
            and best_shift == 0.0
            and best_shift_sim < (float(self.audio_guard_min_similarity) + 0.05)
        ):
            fail_reason = "audio_guard_low_similarity"

        if fail_reason:
            meta["passed"] = False
            meta["reason"] = fail_reason
            return False, meta

        meta["passed"] = True
        return True, meta

    def extract_frame_to_pil(self, video_path: Path, time_sec: float) -> Image.Image:
        """提取帧并转换为 PIL Image"""
        frame_path = self.get_cached_frame_path(video_path, time_sec)
        if frame_path and frame_path.exists():
            try:
                with Image.open(str(frame_path)) as img:
                    return img.copy()
            except Exception:
                return None
        return None

    def compute_phash(self, img: Image.Image) -> imagehash.ImageHash:
        """计算感知哈希"""
        return imagehash.phash(img, hash_size=8)

    def build_frame_index(self, sample_interval: float = 1.0):
        """预建帧索引：提取所有源视频的帧 pHash"""
        index_file = self.cache_dir / "frame_index_v6.pkl"

        if index_file.exists():
            print(f"\n📂 加载已有帧索引: {index_file}")
            with open(index_file, 'rb') as f:
                self.frame_index = pickle.load(f)
            total_frames = sum(len(v) for v in self.frame_index.values())
            print(f"   ✅ 已索引 {len(self.frame_index)} 个视频，共 {total_frames} 帧")
            return

        print(f"\n🔨 构建帧索引 (采样间隔: {sample_interval}s)...")
        for i, video_path in enumerate(self.source_videos):
            print(f"   [{i+1}/{len(self.source_videos)}] {video_path.name}")
            duration = self.get_video_duration(video_path)
            frames = []
            for t in np.arange(0, duration, sample_interval):
                img = self.extract_frame_to_pil(video_path, t)
                if img:
                    frames.append((t, self.compute_phash(img)))
            self.frame_index[video_path] = frames
            print(f"      提取了 {len(frames)} 帧")

        with open(index_file, 'wb') as f:
            pickle.dump(self.frame_index, f)
        total_frames = sum(len(v) for v in self.frame_index.values())
        print(f"\n✅ 索引构建完成: {len(self.frame_index)} 个视频，共 {total_frames} 帧")

    def find_match_by_phash(self, target_start: float, duration: float,
                            seg_index: int = 0, top_k: int = 10) -> List[Tuple[Path, float, float]]:
        """使用 pHash 快速查找匹配候选，返回 [(source, start_time, similarity), ...]"""
        target_img = self.extract_frame_to_pil(self.target_video, target_start + duration * 0.5)
        if not target_img:
            return []
        target_phash = self.compute_phash(target_img)

        candidates = []
        for video_path, frames in self.frame_index.items():
            for time_sec, phash in frames:
                distance = target_phash - phash
                if distance <= 18:
                    similarity = 1.0 - (distance / 64.0)
                    candidates.append((video_path, time_sec, similarity))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def precompute_fingerprints(self):
        """预计算所有音频指纹"""
        print("\n🔄 预计算音频指纹...")
        
        # 目标视频
        print("   目标视频...")
        self.target_fingerprint = self.extract_audio_fingerprint(self.target_video)
        
        # 源视频
        print(f"   {len(self.source_videos)} 个源视频...")
        for i, source in enumerate(self.source_videos):
            print(f"   {i+1}/{len(self.source_videos)}: {source.name}")
            fp = self.extract_audio_fingerprint(source)
            self.source_fingerprints[source] = fp
        
        print("   ✅ 预计算完成")
    
    def find_match_combined(self, target_start: float, duration: float, seg_index: int = 0) -> Tuple[Path, float, float]:
        """三阶段匹配：pHash 预筛选 → (可选音频) → 画面精细定位"""
        best_source = None
        best_start = 0
        best_score = 0.0

        # 第一步：pHash 快速预筛选候选
        phash_candidates = self.find_match_by_phash(target_start, duration, seg_index, top_k=10)

        if phash_candidates:
            # 第二步：对 pHash 候选做精细验证（默认纯画面，可选音频）
            target_fp = extract_chromaprint(self.target_video, target_start, duration) if self.use_audio_matching else []
            target_frames = []
            check_offsets = [0, duration * 0.3, duration * 0.7]
            for offset in check_offsets:
                tf = self.get_cached_frame_path(self.target_video, target_start + offset)
                if tf and tf.exists():
                    target_frames.append((offset, tf))

            for source, phash_time, phash_sim in phash_candidates:
                # 在 pHash 匹配时间点前后 ±2 秒精细搜索
                source_duration = self.get_video_duration(source)
                search_start = max(0, int(phash_time) - 2)
                search_end = min(int(source_duration - duration), int(phash_time) + 2)

                for start_sec in range(search_start, search_end + 1, 1):
                    # 画面验证
                    visual_sim = 0.0
                    if target_frames:
                        total_sim = 0
                        valid = 0
                        for offset, target_frame in target_frames:
                            sf = self.get_cached_frame_path(source, start_sec + offset)
                            if sf and sf.exists():
                                total_sim += self.calculate_frame_similarity(target_frame, sf)
                                valid += 1
                        visual_sim = total_sim / valid if valid > 0 else 0

                    # 音频验证
                    audio_sim = 0.0
                    if self.use_audio_matching and target_fp and len(target_fp) >= 10:
                        source_fp = extract_chromaprint(source, start_sec, duration)
                        if source_fp and len(source_fp) >= 10:
                            audio_sim = compare_chromaprint(target_fp, source_fp)

                    if self.use_audio_matching:
                        # 综合评分：pHash 20% + 音频 40% + 画面 40%
                        combined_score = 0.2 * phash_sim + 0.4 * audio_sim + 0.4 * visual_sim
                    else:
                        # 默认纯画面：pHash 35% + 画面 65%
                        combined_score = 0.35 * phash_sim + 0.65 * visual_sim
                    if combined_score > best_score:
                        best_score = combined_score
                        best_start = start_sec
                        best_source = source

                    if best_score > 0.92:
                        break

            if best_source and best_score >= 0.70:
                return best_source, best_start, best_score

        # Fallback：无 pHash 候选时，默认回退纯画面搜索
        if not self.use_audio_matching:
            return self.find_best_match_by_visual(target_start, duration, seg_index)

        # 启用音频匹配时，回退到音频+画面搜索
        target_fp = extract_chromaprint(self.target_video, target_start, duration)
        if not target_fp or len(target_fp) < 10:
            return self.find_best_match_by_visual(target_start, duration, seg_index)

        audio_candidates = []
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            for start_sec in range(0, int(source_duration - duration), 1):
                source_fp = extract_chromaprint(source, start_sec, duration)
                if not source_fp or len(source_fp) < 10:
                    continue
                score = compare_chromaprint(target_fp, source_fp)
                if score > 0.40:
                    audio_candidates.append((source, start_sec, score))

        if not audio_candidates:
            return self.find_best_match_by_visual(target_start, duration, seg_index)

        audio_candidates.sort(key=lambda x: x[2], reverse=True)
        top_audio = audio_candidates[:20]

        target_frames = []
        check_offsets = [0, duration * 0.3, duration * 0.7]
        for offset in check_offsets:
            tf = self.get_cached_frame_path(self.target_video, target_start + offset)
            if tf and tf.exists():
                target_frames.append((offset, tf))

        if not target_frames:
            best = top_audio[0]
            return best[0], best[1], best[2]

        for source, audio_start, audio_score in top_audio:
            search_start = max(0, audio_start - 3)
            search_end = min(int(self.get_video_duration(source) - duration), audio_start + 3)
            for start_sec in range(search_start, search_end, 1):
                total_sim = 0
                valid_frames = 0
                for offset, target_frame in target_frames:
                    sf = self.get_cached_frame_path(source, start_sec + offset)
                    if sf and sf.exists():
                        total_sim += self.calculate_frame_similarity(target_frame, sf)
                        valid_frames += 1
                if valid_frames > 0:
                    avg_sim = total_sim / valid_frames
                    combined_score = 0.5 * audio_score + 0.5 * avg_sim
                    if combined_score > best_score:
                        best_score = combined_score
                        best_start = start_sec
                        best_source = source
                if best_score > 0.90:
                    break

        return best_source, best_start, best_score

    def refine_start_by_visual(
        self,
        source: Path,
        initial_start: float,
        target_start: float,
        duration: float,
        window: float = 1.2,
        step: float = 0.2,
    ) -> Tuple[float, float]:
        """
        在初始匹配点附近做细粒度视觉对齐，降低整秒步进导致的轻微时间偏移。
        返回 (最佳起点, 对齐分数)。
        """
        try:
            source_duration = self.get_video_duration(source)
        except Exception:
            return float(initial_start), 0.0

        max_start = max(0.0, source_duration - duration)
        left = max(0.0, float(initial_start) - window)
        right = min(max_start, float(initial_start) + window)
        if right < left:
            return float(initial_start), 0.0

        offsets = [0.1, duration * 0.25, duration * 0.5, duration * 0.75, max(0.1, duration - 0.1)]
        best_start = float(initial_start)
        best_score = -1.0

        # v7.2: 目标帧直接读取缓存路径，避免临时文件拷贝/删除开销
        valid_targets: List[Tuple[float, Path]] = []
        for off in offsets:
            tf = self.get_cached_frame_path(self.target_video, target_start + off)
            if tf and tf.exists():
                valid_targets.append((off, tf))
        if not valid_targets:
            return float(initial_start), 0.0

        cursor = left
        while cursor <= right + 1e-6:
            sims: List[float] = []
            for off, target_frame in valid_targets:
                source_frame = self.get_cached_frame_path(source, cursor + off)
                if source_frame and source_frame.exists():
                    sims.append(float(self.calculate_frame_similarity(target_frame, source_frame)))
            if sims:
                score = float(np.mean(sims))
                if score > best_score:
                    best_score = score
                    best_start = float(cursor)
            cursor += step

        if best_score < 0:
            return float(initial_start), 0.0
        return best_start, best_score
    
    def find_best_match_by_visual(self, target_start: float, duration: float, seg_index: int = 0) -> Tuple[Path, float, float]:
        """当音频匹配失败时，遍历所有源视频进行画面匹配（更精细的搜索）"""
        best_source = None
        best_start = 0
        best_score = 0.0
        
        # 提取目标帧（多个时间点）
        target_frames = []
        check_offsets = [0, duration * 0.3, duration * 0.7]  # 3个时间点

        for offset in check_offsets:
            target_frame = self.get_cached_frame_path(self.target_video, target_start + offset)
            if target_frame and target_frame.exists():
                target_frames.append((offset, target_frame))
        
        if not target_frames:
            return None, 0, 0
        
        # 遍历所有源视频
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            
            # 扩大搜索范围：在目标时间点附近搜索（±60秒）
            search_start = max(0, int(target_start - 60))
            search_end = min(int(source_duration - duration), int(target_start + 60))
            
            # 滑动搜索（步长2秒，更精细）
            for start_sec in range(search_start, search_end, 2):
                total_sim = 0
                valid_frames = 0

                for offset, target_frame in target_frames:
                    source_frame = self.get_cached_frame_path(source, start_sec + offset)
                    if source_frame and source_frame.exists():
                        sim = self.calculate_frame_similarity(target_frame, source_frame)
                        total_sim += sim
                        valid_frames += 1
                
                if valid_frames > 0:
                    avg_sim = total_sim / valid_frames
                    if avg_sim > best_score:
                        best_score = avg_sim
                        best_start = start_sec
                        best_source = source
                
                # 提前退出条件
                if best_score > 0.90:
                    break
            
            if best_score > 0.90:
                break
        
        # 只有相似度足够高才返回
        if best_score >= 0.75:
            return best_source, best_start, best_score
        return None, 0, 0

    def rematch_low_confidence_segment(
        self,
        target_start: float,
        duration: float,
        seg_index: int,
        base_source: Optional[Path],
        base_start: float,
        base_score: float,
    ) -> Tuple[Optional[Path], float, float, Dict]:
        """
        低置信度段局部重匹配：
        - 动态触发，无固定时间点
        - 以候选中心做逐步扩窗搜索
        """
        target_fp = []
        if self.use_audio_matching:
            target_fp = extract_chromaprint(self.target_video, target_start, duration)
            if not target_fp or len(target_fp) < 10:
                return base_source, base_start, base_score, {"triggered": True, "improved": False, "reason": "no_target_fp"}

        target_frames = []
        check_offsets = [0, duration * 0.5, max(0.0, duration - 0.2)]
        for offset in check_offsets:
            tf = self.get_cached_frame_path(self.target_video, target_start + offset)
            if tf and tf.exists():
                target_frames.append((offset, tf))
        if not target_frames:
            return base_source, base_start, base_score, {"triggered": True, "improved": False, "reason": "no_target_frames"}

        best_source = base_source
        best_start = base_start
        best_score = base_score
        best_meta = {"triggered": True, "improved": False, "window": self.rematch_window}

        candidates = self.find_match_by_phash(target_start, duration, seg_index, top_k=12)
        if base_source is not None:
            candidates.insert(0, (base_source, base_start + duration * 0.5, min(1.0, base_score + 0.05)))

        seen = set()
        for source, center_time, phash_sim in candidates:
            source_duration = self.get_video_duration(source)
            for window in range(self.rematch_window, self.rematch_max_window + 1, self.rematch_window):
                left = max(0, int(center_time - window))
                right = min(int(source_duration - duration), int(center_time + window))
                if right < left:
                    continue
                for start_sec in range(left, right + 1):
                    key = (str(source), start_sec)
                    if key in seen:
                        continue
                    seen.add(key)

                    audio_sim = 0.0
                    if self.use_audio_matching:
                        source_fp = extract_chromaprint(source, start_sec, duration)
                        if not source_fp or len(source_fp) < 10:
                            continue
                        audio_sim = compare_chromaprint(target_fp, source_fp)
                        if audio_sim < 0.40:
                            continue

                    total_sim = 0.0
                    valid = 0
                    for offset, target_frame in target_frames:
                        sf = self.get_cached_frame_path(source, start_sec + offset)
                        if sf and sf.exists():
                            total_sim += self.calculate_frame_similarity(target_frame, sf)
                            valid += 1
                    visual_sim = (total_sim / valid) if valid > 0 else 0.0

                    if self.use_audio_matching:
                        score = (
                            0.15 * phash_sim
                            + 0.45 * audio_sim
                            + 0.35 * visual_sim
                            + self.continuity_bonus(start_sec, target_start, duration)
                        )
                    else:
                        score = (
                            0.35 * phash_sim
                            + 0.60 * visual_sim
                            + self.continuity_bonus(start_sec, target_start, duration)
                        )
                    if score > best_score:
                        best_source = source
                        best_start = start_sec
                        best_score = score
                        best_meta = {
                            "triggered": True,
                            "improved": True,
                            "window": window,
                            "visual": visual_sim,
                            "phash": phash_sim,
                        }
                        if self.use_audio_matching:
                            best_meta["audio"] = audio_sim

        return best_source, best_start, best_score, best_meta
    
    def _find_alternative_match(self, target_start: float, duration: float, 
                                 source: Path, min_start: float, seg_index: int) -> float:
        """
        在同一源视频中向后搜索，找一个不重叠的替代匹配点
        返回: 新的开始时间，或 None（如果没找到）
        """
        # 提取目标帧（只取2个关键点，减少计算）
        target_frames = []
        check_offsets = [0, duration * 0.5]  # 减少到2个时间点
        
        for offset in check_offsets:
            target_frame = self.get_cached_frame_path(self.target_video, target_start + offset)
            if target_frame and target_frame.exists():
                target_frames.append((offset, target_frame))
        
        if not target_frames:
            return None
        
        source_duration = self.get_video_duration(source)
        best_start = None
        best_score = 0.0
        
        # 限制搜索范围：最多向后搜索30秒，步长1秒（减少计算量）
        search_end = min(int(source_duration - duration), int(min_start + 30))
        
        for start_sec in range(int(min_start), search_end, 1):
            total_sim = 0
            valid_frames = 0
            
            for offset, target_frame in target_frames:
                source_frame = self.get_cached_frame_path(source, start_sec + offset)
                if source_frame and source_frame.exists():
                    sim = self.calculate_frame_similarity(target_frame, source_frame)
                    total_sim += sim
                    valid_frames += 1
            
            if valid_frames > 0:
                avg_sim = total_sim / valid_frames
                # 降低相似度阈值，提高找到替代点的概率
                if avg_sim > best_score and avg_sim >= 0.70:
                    best_score = avg_sim
                    best_start = start_sec
                    
                    # 如果找到足够好的匹配，提前退出
                    if avg_sim >= 0.85:
                        break
        
        return best_start

    def quick_verify(self, source: Path, source_start: float, target_start: float, duration: float) -> Tuple[bool, float]:
        """快速画面验证 + 时间偏移偏置检测（防“短暂快进/慢进后恢复”）。"""

        def frame_sim_with_target_shift(offset: float, target_shift: float = 0.0) -> float:
            """比较 source(t) 与 target(t + shift) 的相似度。"""
            src_t = source_start + offset
            tgt_t = target_start + offset + target_shift
            target_frame = self.get_cached_frame_path(self.target_video, tgt_t)
            source_frame = self.get_cached_frame_path(source, src_t)
            if target_frame and source_frame and target_frame.exists() and source_frame.exists():
                sim = self.calculate_frame_similarity(target_frame, source_frame)
                return float(sim)
            return 0.0

        # 原有段级核验：增加采样点，降低“边界错位漏检”概率
        check_times = [0.0, duration * 0.25, duration * 0.5, duration * 0.75, max(0.0, duration - 0.1)]
        similarities = [frame_sim_with_target_shift(offset, 0.0) for offset in check_times]

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        min_sim = float(np.min(similarities)) if similarities else 0.0

        min_avg = self.strict_verify_min_sim
        min_floor = max(0.0, min_avg - 0.08)
        passed = avg_sim >= min_avg and min_sim >= min_floor
        if not passed:
            return False, avg_sim

        # 新增：时间偏移偏置检测
        # 如果同一段在多个采样点都“对齐 +1/+2 秒或 -1/-2 秒更像”，说明存在系统性快进/慢进错位。
        core_offsets = [duration * 0.2, duration * 0.5, duration * 0.8]
        bias_votes: List[int] = []
        for offset in core_offsets:
            base_sim = frame_sim_with_target_shift(offset, 0.0)
            best_shift = 0
            best_sim = base_sim
            for shift in (-2.0, -1.0, 1.0, 2.0):
                shifted_sim = frame_sim_with_target_shift(offset, shift)
                if shifted_sim > best_sim:
                    best_sim = shifted_sim
                    best_shift = int(shift)

            # 只有“偏移显著优于对齐”才记票，避免静态画面导致误判。
            if best_shift != 0 and (best_sim - base_sim) >= 0.10 and base_sim < 0.93:
                bias_votes.append(best_shift)

        if len(bias_votes) >= 2:
            forward_votes = sum(1 for v in bias_votes if v > 0)
            backward_votes = sum(1 for v in bias_votes if v < 0)
            if forward_votes >= 2 or backward_votes >= 2:
                return False, avg_sim

        return True, avg_sim

    def verify_segment_visual(
        self,
        source: Path,
        source_start: float,
        target_start: float,
        duration: float,
        offsets: List[float],
        min_avg: float,
        min_floor: float,
    ) -> Tuple[bool, float]:
        """通用多点画面核验。"""
        similarities = []
        for offset in offsets:
            clamped_offset = max(0.0, min(duration, offset))
            target_frame = self.get_cached_frame_path(self.target_video, target_start + clamped_offset)
            source_frame = self.get_cached_frame_path(source, source_start + clamped_offset)
            if target_frame and source_frame and target_frame.exists() and source_frame.exists():
                similarities.append(self.calculate_frame_similarity(target_frame, source_frame))

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        min_sim = float(np.min(similarities)) if similarities else 0.0
        return avg_sim >= min_avg and min_sim >= min_floor, avg_sim
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path):
        """提取帧"""
        cached = self.get_cached_frame_path(video_path, time_sec)
        if cached and cached.exists():
            out_lock = self._get_output_path_lock(output_path)
            with out_lock:
                self._materialize_frame(cached, output_path)
    
    def calculate_frame_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        """计算帧相似度"""
        f1 = self._get_frame_features(frame1_path)
        f2 = self._get_frame_features(frame2_path)
        if f1 is None or f2 is None:
            return 0.0
        gray1, hist1 = f1
        gray2, hist2 = f2
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        return 0.5 * max(0, hist_sim) + 0.5 * template_sim

    def _get_frame_features(self, frame_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        key = str(Path(frame_path).resolve())
        with self.frame_feature_lock:
            cached = self.frame_feature_cache.get(key)
            if cached is not None:
                self.frame_feature_cache.move_to_end(key)
                return cached

        img = cv2.imread(str(frame_path))
        if img is None:
            return None

        img = cv2.resize(img, (320, 180))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        features = (gray, hist)

        with self.frame_feature_lock:
            self.frame_feature_cache[key] = features
            self.frame_feature_cache.move_to_end(key)
            while len(self.frame_feature_cache) > self.frame_feature_cache_max_items:
                self.frame_feature_cache.popitem(last=False)
        return features
    
    def process_segment(self, task: SegmentTask) -> SegmentResult:
        """处理单个段 - 音频+画面结合匹配，失败时用目标视频兜底"""

        # 音频+画面结合匹配
        source, source_start, combined_score = self.find_match_combined(
            task.target_start, task.duration, task.index
        )

        quality = {
            "combined": combined_score,
            "low_confidence": combined_score < self.low_score_threshold,
            "rematch_triggered": False,
            "rematch_improved": False,
        }

        # 对低置信度段做局部重匹配
        if source and combined_score < self.low_score_threshold:
            rematch_source, rematch_start, rematch_score, rematch_meta = self.rematch_low_confidence_segment(
                task.target_start,
                task.duration,
                task.index,
                source,
                source_start,
                combined_score,
            )
            quality["rematch_triggered"] = rematch_meta.get("triggered", False)
            quality["rematch_improved"] = rematch_meta.get("improved", False)
            quality["rematch"] = rematch_meta
            if rematch_source and rematch_score > combined_score:
                source, source_start, combined_score = rematch_source, rematch_start, rematch_score
                quality["combined"] = combined_score

        # 对候选段做局部起点微调，降低整秒匹配带来的 1~2 秒偏移风险
        if source and source != self.target_video:
            refined_start, refined_score = self.refine_start_by_visual(
                source=source,
                initial_start=float(source_start),
                target_start=float(task.target_start),
                duration=float(task.duration),
            )
            quality["start_refine"] = {
                "before": float(source_start),
                "after": float(refined_start),
                "score": float(refined_score),
            }
            source_start = float(refined_start)

        if not source or combined_score < 0.70:
            # 兜底：使用目标视频本身对应时间段
            print(f"   段 {task.index + 1}/{self.total_segments} ⚠️ 匹配失败 (score={combined_score:.2f})，使用目标视频兜底")
            return SegmentResult(
                index=task.index,
                success=True,
                source=self.target_video,
                source_start=task.target_start,
                quality={**quality, "combined": 0.0, "fallback": True}
            )

        # 通用防误匹配：段级画面快速核验，失败则回退目标片段
        if self.strict_visual_verify:
            passed, verify_avg = self.quick_verify(source, source_start, task.target_start, task.duration)
            quality["strict_verify"] = {"passed": bool(passed), "avg": float(verify_avg)}
            if not passed:
                print(
                    f"   段 {task.index + 1}/{self.total_segments} ⚠️ 画面核验失败 "
                    f"(avg={verify_avg:.2f})，回退目标视频"
                )
                return SegmentResult(
                    index=task.index,
                    success=True,
                    source=self.target_video,
                    source_start=task.target_start,
                    quality={
                        **quality,
                        "combined": 0.0,
                        "fallback": True,
                        "fallback_reason": "strict_visual_verify_failed",
                    },
                )

        # 轻量音频守卫：拦截“画面相似但语音错位”的段。
        if source != self.target_video:
            audio_passed, audio_meta = self.quick_verify_audio(
                source=source,
                source_start=source_start,
                target_start=task.target_start,
                duration=task.duration,
                combined_score=float(quality.get("combined", combined_score)),
            )
            quality["audio_guard"] = audio_meta
            if not audio_passed:
                aligned_sim = float(audio_meta.get("aligned_similarity", 0.0))
                best_shift = float(audio_meta.get("best_shift_sec", 0.0))
                reason = str(audio_meta.get("reason", "audio_guard_failed"))
                print(
                    f"   段 {task.index + 1}/{self.total_segments} ⚠️ 音频守卫失败 "
                    f"(sim={aligned_sim:.2f}, shift={best_shift:+.1f}s, reason={reason})，回退目标视频"
                )
                return SegmentResult(
                    index=task.index,
                    success=True,
                    source=self.target_video,
                    source_start=task.target_start,
                    quality={
                        **quality,
                        "combined": 0.0,
                        "fallback": True,
                        "fallback_reason": reason,
                    },
                )

        # 通用尾段守卫：末尾更严格核验，防止尾段误匹配导致“内容回跳”
        if self.target_duration > 0 and task.target_start >= (self.target_duration - self.tail_guard_seconds):
            offsets = [0.0, task.duration * 0.25, task.duration * 0.5, task.duration * 0.75, max(0.0, task.duration - 0.1)]
            tail_passed, tail_avg = self.verify_segment_visual(
                source,
                source_start,
                task.target_start,
                task.duration,
                offsets=offsets,
                min_avg=self.tail_verify_min_avg,
                min_floor=self.tail_verify_min_floor,
            )
            quality["tail_verify"] = {"passed": bool(tail_passed), "avg": float(tail_avg)}
            if not tail_passed:
                print(
                    f"   段 {task.index + 1}/{self.total_segments} ⚠️ 尾段守卫失败 "
                    f"(avg={tail_avg:.2f})，回退目标视频"
                )
                return SegmentResult(
                    index=task.index,
                    success=True,
                    source=self.target_video,
                    source_start=task.target_start,
                    quality={
                        **quality,
                        "combined": 0.0,
                        "fallback": True,
                        "fallback_reason": "tail_guard_failed",
                    },
                )

        print(f"   段 {task.index + 1}/{self.total_segments} ✅ {source.name} @ {source_start}s (综合: {combined_score:.2f})")

        return SegmentResult(
            index=task.index,
            success=True,
            source=source,
            source_start=source_start,
            quality=quality
        )
    
    def reconstruct_fast(self, output_path: str) -> bool:
        """极速重构"""
        import time

        print(f"\n{'='*70}")
        print(f"🚀 极速高精度重构 V3 + pHash")
        print(f"{'='*70}")

        start_wall = time.time()
        overall_perf = time.perf_counter()

        # 预建 pHash 帧索引（首次运行后缓存，后续秒速加载）
        self.build_frame_index(sample_interval=1.0)

        target_duration = self.get_video_duration(self.target_video)
        self.target_duration = target_duration
        self.output_fps = self.get_video_fps(self.target_video)
        print(f"\n📹 目标视频: {target_duration:.1f}s")
        print(f"🎞️ 输出帧率: {self.output_fps:.3f} fps (自动读取目标视频)")
        
        # 创建任务列表
        tasks = []
        num_segments = int(target_duration / self.segment_duration) + 1
        
        for i in range(num_segments):
            start = i * self.segment_duration
            duration = min(self.segment_duration, target_duration - start)
            if duration > 0:
                tasks.append(SegmentTask(index=i, target_start=start, duration=duration))

        self.total_segments = len(tasks)
        
        print(f"\n🔄 并行处理 {len(tasks)} 个段 (线程数: {self.max_workers})...")
        match_perf = time.perf_counter()

        # 并行处理
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.process_segment, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.index] = result
                    status = "✅" if result.success else "❌"
                    print(f"   段 {task.index+1}/{len(tasks)} {status}")
                except Exception as e:
                    print(f"   段 {task.index+1} 错误: {e}")
                    results[task.index] = SegmentResult(index=task.index, success=False)
        
        # 整理结果：优先保全所有段，任何缺失段都用目标视频对应时间兜底
        confirmed_by_index = {}
        for r in results:
            if r and r.success:
                confirmed_by_index[r.index] = {
                    'index': r.index,
                    'source': r.source,
                    'start': r.source_start,
                    'duration': tasks[r.index].duration,
                    'target_start': tasks[r.index].target_start,
                    'quality': r.quality or {}
                }

        missing_count = 0
        confirmed_segments = []
        for task in tasks:
            seg = confirmed_by_index.get(task.index)
            if seg is None:
                missing_count += 1
                seg = {
                    'index': task.index,
                    'source': self.target_video,
                    'start': task.target_start,
                    'duration': task.duration,
                    'target_start': task.target_start,
                    'quality': {'combined': 0.0, 'fallback': True, 'reason': 'missing_result'}
                }
            confirmed_segments.append(seg)

        if missing_count > 0:
            print(f"   ⚠️ 自动补齐缺失段: {missing_count} 段")

        self.match_elapsed_sec = time.perf_counter() - match_perf

        guard_perf = time.perf_counter()
        overlap_adjusted, overlap_fallback = self.smooth_adjacent_overlaps(confirmed_segments)
        if overlap_adjusted > 0:
            print(f"   🔧 去重叠平滑: 修正 {overlap_adjusted} 个相邻段")
        if overlap_fallback > 0:
            print(f"   🛟 反重复兜底: 切换 {overlap_fallback} 个段到目标素材")
        global_repeat_fallback = self.suppress_temporal_loops(confirmed_segments)
        if global_repeat_fallback > 0:
            print(f"   🧯 全局反重复: 切换 {global_repeat_fallback} 个段到目标素材")
        step_guard_fallback = self.enforce_temporal_step_consistency(confirmed_segments)
        if step_guard_fallback > 0:
            print(f"   🧷 步长一致性守卫: 切换 {step_guard_fallback} 个段到目标素材")
        alignment_bias_fallback = self.enforce_target_alignment_bias(confirmed_segments)
        if alignment_bias_fallback > 0:
            print(f"   🎯 对齐偏置守卫: 切换 {alignment_bias_fallback} 个段到目标素材")

        self.timeline_guard_elapsed_sec = time.perf_counter() - guard_perf
        self.guard_stats = {
            "missing_filled": int(missing_count),
            "overlap_adjusted": int(overlap_adjusted),
            "overlap_fallback": int(overlap_fallback),
            "global_repeat_fallback": int(global_repeat_fallback),
            "step_guard_fallback": int(step_guard_fallback),
            "alignment_bias_fallback": int(alignment_bias_fallback),
        }

        print(f"\n✅ 可用段: {len(confirmed_segments)}/{len(tasks)} 段 (完整保留)")
        
        # 生成输出
        success = False
        if confirmed_segments:
            print(f"\n🎬 生成视频...")
            success = self._generate_output(confirmed_segments, output_path, target_duration)

        self.total_elapsed_sec = time.perf_counter() - overall_perf
        self.save_quality_report(confirmed_segments, output_path)

        elapsed = time.time() - start_wall
        print(f"\n{'='*70}")
        print(f"✅ 完成!")
        print(f"   耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
        print(f"   输出: {output_path}")
        print(f"{'='*70}")

        return success

    def save_quality_report(self, segments: List[dict], output_path: str):
        """输出段级质量统计，便于排查局部错位。"""
        def to_jsonable(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, np.bool_):
                return bool(value)
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, dict):
                return {str(k): to_jsonable(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return [to_jsonable(v) for v in value]
            if isinstance(value, list):
                return [to_jsonable(v) for v in value]
            return value

        low_segments = []
        rematch_triggered = 0
        rematch_improved = 0
        fallback_count = 0
        combined_scores = []
        segment_details = []

        for seg in segments:
            quality = seg.get("quality", {}) or {}
            combined = float(quality.get("combined", 0.0))
            combined_scores.append(combined)
            if quality.get("rematch_triggered"):
                rematch_triggered += 1
            if quality.get("rematch_improved"):
                rematch_improved += 1
            if quality.get("fallback"):
                fallback_count += 1
            if combined < self.low_score_threshold:
                low_segments.append({
                    "index": seg["index"],
                    "target_start": seg["target_start"],
                    "source": str(seg["source"]),
                    "source_start": seg["start"],
                    "combined": combined,
                    "fallback": bool(quality.get("fallback", False)),
                })

            segment_details.append({
                "index": int(seg["index"]),
                "target_start": float(seg["target_start"]),
                "duration": float(seg["duration"]),
                "source": str(seg["source"]),
                "source_start": float(seg["start"]),
                "mapping_offset": float(seg["start"] - seg["target_start"]),
                "combined": combined,
                "fallback": bool(quality.get("fallback", False)),
                "fallback_reason": quality.get("fallback_reason", quality.get("reason", "")),
                "render_extract_elapsed_sec": float(seg.get("render_extract_elapsed_sec", 0.0)),
                "quality": to_jsonable(quality),
            })

        avg_score = float(np.mean(combined_scores)) if combined_scores else 0.0
        output_path_obj = Path(output_path)
        output_duration = None
        if output_path_obj.exists():
            try:
                output_duration = float(self.get_video_duration(output_path_obj))
            except Exception:
                output_duration = None

        render_metrics = self.last_render_metrics if isinstance(self.last_render_metrics, dict) else {}

        report = {
            "target_video": str(self.target_video),
            "output_video": str(output_path_obj),
            "target_duration": float(self.target_duration),
            "output_duration": output_duration,
            "total_segments": len(segments),
            "avg_combined_score": avg_score,
            "low_score_threshold": self.low_score_threshold,
            "low_score_segments": len(low_segments),
            "rematch_triggered": rematch_triggered,
            "rematch_improved": rematch_improved,
            "fallback_segments": fallback_count,
            "timing": {
                "match_elapsed_sec": round(float(self.match_elapsed_sec), 3),
                "timeline_guard_elapsed_sec": round(float(self.timeline_guard_elapsed_sec), 3),
                "render_elapsed_sec": round(float(render_metrics.get("render_total_sec", 0.0)), 3),
                "total_elapsed_sec": round(float(self.total_elapsed_sec), 3),
            },
            "guard_stats": to_jsonable(self.guard_stats),
            "render_metrics": to_jsonable(render_metrics),
            "low_segments": low_segments,
            "segments": segment_details,
        }

        report_path = Path(output_path).with_suffix(".quality_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"   📋 质量报告: {report_path}")
        print(
            f"   统计: 平均分 {avg_score:.3f}, 低分段 {len(low_segments)}, "
            f"重匹配触发 {rematch_triggered}, 提升 {rematch_improved}, 兜底 {fallback_count}"
        )

    def smooth_adjacent_overlaps(self, segments: List[dict]) -> Tuple[int, int]:
        """
        全局时间轴连续性修正（避免连锁推移）：
        - 仅处理“严重回跳/重叠/前跳”异常
        - 优先在同源内重找可用起点，失败则切回目标段
        - 不再对全部轻微重叠做连续平移，防止后半段整体被推偏
        """
        adjusted = 0
        fallback_switched = 0

        def fallback_segment(seg: dict, reason: str, metric: float):
            nonlocal fallback_switched
            seg["source"] = self.target_video
            seg["start"] = seg["target_start"]
            q = seg.get("quality", {}) or {}
            q["fallback"] = True
            q["fallback_reason"] = reason
            q["timeline_guard_metric"] = float(metric)
            seg["quality"] = q
            fallback_switched += 1

        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            curr_start = float(curr["start"])
            curr_target = float(curr["target_start"])
            prev_quality = prev.get("quality", {}) or {}
            prev_fallback_reason = str(prev_quality.get("fallback_reason", "") or "")
            prev_guard_trigger_reasons = {
                "timeline_guard_overlap",
                "timeline_guard_backjump",
                "timeline_guard_fast_forward_gap",
                "timeline_guard_lagging_step",
                "timeline_cross_source_after_target",
                "timeline_cross_source_mapping_jump",
            }
            prev_guard_fallback = bool(
                prev["source"] == self.target_video
                and bool(prev_quality.get("fallback", False))
                and prev_fallback_reason in prev_guard_trigger_reasons
            )

            # 守卫冷却：上一段刚发生 timeline_guard 兜底时，
            # 下一段禁止立即切回外部源，避免出现“1 段修正后立刻回跳”的局部重复。
            if prev_guard_fallback and curr["source"] != self.target_video:
                fallback_segment(curr, "timeline_guard_cooldown_after_fallback", 0.0)
                continue

            # 跨源边界守卫：如果前段已是目标兜底，后段又是孤立非目标段且时间漂移明显，
            # 极易出现“某 1 秒错位后又恢复”的现象，直接回退当前段到目标素材。
            if curr["source"] != self.target_video and prev["source"] != curr["source"]:
                cross_drift = abs(curr_start - curr_target)
                if prev["source"] == self.target_video and cross_drift > self.isolated_drift_trigger:
                    fallback_segment(curr, "timeline_cross_source_after_target", cross_drift)
                    continue
                # 跨源切换守卫：若映射偏移突变，极易出现 1 秒级“回放/倒放样式”错觉。
                if prev["source"] != self.target_video:
                    prev_mapping = float(prev["start"]) - float(prev["target_start"])
                    curr_mapping = curr_start - curr_target
                    mapping_jump = abs(curr_mapping - prev_mapping)
                    if mapping_jump > self.cross_source_mapping_jump_trigger:
                        fallback_segment(curr, "timeline_cross_source_mapping_jump", mapping_jump)
                continue

            if curr["source"] == self.target_video:
                continue

            prev_end = float(prev["start"]) + float(prev["duration"])
            overlap = prev_end - curr_start
            source_delta = curr_start - float(prev["start"])
            target_delta = float(curr["target_start"]) - float(prev["target_start"])
            gap_allowance = max(1.0, float(curr["duration"]) * 0.25)

            severe_overlap = overlap > max(self.adjacent_overlap_trigger, float(curr["duration"]) * 0.12)
            backjump = source_delta < -0.5
            fast_forward_gap = source_delta > target_delta + gap_allowance
            lagging_step = (target_delta - source_delta) > max(self.adjacent_lag_trigger, float(curr["duration"]) * 0.12)

            # 轻度边界重叠定点矫正：
            # 命中“刚好贴阈值”的回放风险时，优先将当前段起点吸附到前段末尾，
            # 避免 0.5~1s 的局部重复/倒放观感，同时不触发大范围兜底。
            overlap_floor = max(self.adjacent_overlap_trigger * 0.95, float(curr["duration"]) * 0.12)
            eps = 1e-6
            mild_overlap_nudge = (
                overlap > 0.0
                and overlap + eps >= overlap_floor
                and overlap <= max(1.2, float(curr["duration"]) * 0.30)
                and not backjump
                and not fast_forward_gap
                and not lagging_step
            )
            if mild_overlap_nudge:
                adjusted_start = prev_end
                step_tolerance = max(0.9, float(curr["duration"]) * 0.30)
                adjusted_source_delta = float(adjusted_start) - float(prev["start"])
                if abs(adjusted_source_delta - target_delta) <= step_tolerance:
                    curr["start"] = float(adjusted_start)
                    q = curr.get("quality", {}) or {}
                    q["timeline_guard_overlap_nudged"] = True
                    q["timeline_guard_overlap_before"] = float(overlap)
                    q["timeline_guard_source_delta_before"] = float(source_delta)
                    curr["quality"] = q
                    adjusted += 1
                    continue

            if not (severe_overlap or backjump or fast_forward_gap or lagging_step):
                continue

            # 只在确有异常时尝试替代匹配，避免对正常段做连锁平移。
            min_start = max(curr_start, prev_end - 0.05)
            alt_start = self._find_alternative_match(
                curr["target_start"],
                curr["duration"],
                curr["source"],
                min_start,
                curr["index"],
            )

            if alt_start is not None and alt_start >= prev_end - 0.2:
                step_tolerance = max(0.9, float(curr["duration"]) * 0.30)
                adjusted_source_delta = float(alt_start) - float(prev["start"])
                # 硬约束：禁止重匹配后出现明显“快进/慢进”步长偏差，避免局部短暂错位后又恢复。
                if abs(adjusted_source_delta - target_delta) > step_tolerance:
                    alt_start = None

            if alt_start is not None and alt_start >= prev_end - 0.2:
                passed, verify_avg = self.quick_verify(
                    curr["source"],
                    alt_start,
                    curr["target_start"],
                    curr["duration"],
                )
                if passed:
                    curr["start"] = float(alt_start)
                    q = curr.get("quality", {}) or {}
                    q["timeline_guard_rematched"] = True
                    q["timeline_guard_verify_avg"] = float(verify_avg)
                    q["timeline_guard_overlap"] = float(overlap)
                    q["timeline_guard_source_delta"] = float(source_delta)
                    curr["quality"] = q
                    adjusted += 1
                    continue

            reason = "timeline_guard_overlap"
            metric = overlap
            if backjump:
                reason = "timeline_guard_backjump"
                metric = source_delta
            elif fast_forward_gap:
                reason = "timeline_guard_fast_forward_gap"
                metric = source_delta - target_delta
            elif lagging_step:
                reason = "timeline_guard_lagging_step"
                metric = target_delta - source_delta
            fallback_segment(curr, reason, metric)

        return adjusted, fallback_switched

    def suppress_temporal_loops(self, segments: List[dict]) -> int:
        """
        轻量全局反重复（仅基于时间轴，避免额外帧提取开销）：
        - 同源回跳：目标时间前进但源时间倒退
        - 同源前跳：源时间前进过快导致中间内容被跳过（丢帧/缺内容）
        - 同源缓慢前进：连续多段几乎停在同一时间附近
        - 远距复用：隔很远的目标段复用同一源时间桶
        """
        switched = 0

        def fallback_segment(seg: dict, reason: str):
            nonlocal switched
            if seg["source"] == self.target_video:
                return
            seg["source"] = self.target_video
            seg["start"] = seg["target_start"]
            q = seg.get("quality", {}) or {}
            q["fallback"] = True
            q["fallback_reason"] = reason
            seg["quality"] = q
            switched += 1

        # 规则1：逐段回跳/停滞检测
        slow_streak = 0
        for i, seg in enumerate(segments):
            if seg["source"] == self.target_video:
                slow_streak = 0
                continue

            fallback_reason: Optional[str] = None
            duration = float(seg["duration"])
            if i > 0:
                prev = segments[i - 1]
                if prev["source"] == seg["source"]:
                    source_delta = float(seg["start"]) - float(prev["start"])
                    target_delta = float(seg["target_start"]) - float(prev["target_start"])

                    # 同源回跳：目标时间前进，但源时间倒退，容易出现内容回放/重复。
                    if target_delta > duration * 0.8 and source_delta < -0.5:
                        fallback_reason = "temporal_loop_step_backjump"

                    # 同源前跳：源时间推进明显快于目标时间，容易跳过中间内容（用户感知为丢帧/缺内容）。
                    gap_allowance = max(1.0, duration * 0.25)
                    if source_delta > target_delta + gap_allowance:
                        fallback_reason = "temporal_gap_step"

                    if source_delta < max(0.8, duration * 0.35):
                        slow_streak += 1
                    else:
                        slow_streak = 0

                    if slow_streak >= 2 and fallback_reason is None:
                        fallback_reason = "temporal_loop_step_stall"
                else:
                    slow_streak = 0

            if fallback_reason is not None:
                fallback_segment(seg, fallback_reason)
                slow_streak = 0

        # 规则2：连续同源段“时间压缩”检测（强制）
        i = 0
        while i < len(segments):
            src = segments[i]["source"]
            if src == self.target_video:
                i += 1
                continue

            j = i + 1
            while j < len(segments) and segments[j]["source"] == src:
                j += 1

            run = segments[i:j]
            if len(run) >= 3:
                d = float(run[0]["duration"])
                target_span = float(run[-1]["target_start"] - run[0]["target_start"] + run[-1]["duration"])
                src_starts = [float(s["start"]) for s in run]
                source_span = max(src_starts) - min(src_starts) + d
                compression_ratio = source_span / target_span if target_span > 1e-6 else 1.0

                # 目标覆盖明显更长，但源时间只在小范围抖动，判定为循环重复
                if target_span >= d * 3 and compression_ratio < 0.60:
                    for seg in run[1:]:
                        fallback_segment(seg, "temporal_loop_run_compression")

            i = j

        return switched

    def enforce_target_alignment_bias(self, segments: List[dict]) -> int:
        """
        目标时间对齐偏置守卫：
        - 检测某段是否在多个采样点都更像 target 的 +/-1~3 秒位置
        - 若存在稳定同向偏置（例如整体领先 2 秒），直接回退目标段
        用于拦截“局部画面提前/滞后，音频仍正常”的错位问题。
        """
        switched = 0

        def fallback_segment(seg: dict, reason: str, metric: float):
            nonlocal switched
            if seg["source"] == self.target_video:
                return
            seg["source"] = self.target_video
            seg["start"] = seg["target_start"]
            q = seg.get("quality", {}) or {}
            q["fallback"] = True
            q["fallback_reason"] = reason
            q["timeline_bias_metric"] = float(metric)
            seg["quality"] = q
            switched += 1

        def frame_sim(source: Path, source_time: float, target_time: float) -> float:
            source_frame = self.get_cached_frame_path(source, source_time)
            target_frame = self.get_cached_frame_path(self.target_video, target_time)
            if source_frame and target_frame and source_frame.exists() and target_frame.exists():
                sim = self.calculate_frame_similarity(source_frame, target_frame)
                return float(sim)
            return 0.0

        for seg in segments:
            if seg["source"] == self.target_video:
                continue

            duration = float(seg["duration"])
            start = float(seg["start"])
            target_start = float(seg["target_start"])
            if duration <= 0.5:
                continue

            raw_offsets = [duration * 0.2, duration * 0.4, duration * 0.6, duration * 0.8]
            offsets = [max(0.1, min(duration - 0.1, o)) for o in raw_offsets]

            bias_votes: List[int] = []
            gains: List[float] = []
            for offset in offsets:
                src_t = start + offset
                tgt_t = target_start + offset
                aligned = frame_sim(seg["source"], src_t, tgt_t)
                best_shift = 0
                best_sim = aligned

                for shift in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0):
                    shifted_sim = frame_sim(seg["source"], src_t, tgt_t + shift)
                    if shifted_sim > best_sim:
                        best_sim = shifted_sim
                        best_shift = int(shift)

                if best_shift != 0:
                    gain = best_sim - aligned
                    if gain >= 0.08:
                        bias_votes.append(best_shift)
                        gains.append(gain)

            if len(bias_votes) < 2:
                continue

            forward_votes = sum(1 for v in bias_votes if v > 0)
            backward_votes = sum(1 for v in bias_votes if v < 0)
            dominant = max(forward_votes, backward_votes)
            avg_gain = float(np.mean(gains)) if gains else 0.0

            if dominant >= 2 and avg_gain >= 0.09:
                reason = (
                    "timeline_target_alignment_bias_forward"
                    if forward_votes >= backward_votes
                    else "timeline_target_alignment_bias_backward"
                )
                fallback_segment(seg, reason, avg_gain)

        return switched

    def enforce_temporal_step_consistency(self, segments: List[dict]) -> int:
        """
        同源步长一致性硬守卫：
        - 对同源相邻段（前后双向）要求 source 步长与 target 步长近似一致
        - 如果出现突变（短暂错位后恢复的常见根因），直接兜底到目标段
        """
        switched = 0

        def fallback_segment(seg: dict, reason: str, metric: float):
            nonlocal switched
            if seg["source"] == self.target_video:
                return
            seg["source"] = self.target_video
            seg["start"] = seg["target_start"]
            q = seg.get("quality", {}) or {}
            q["fallback"] = True
            q["fallback_reason"] = reason
            q["timeline_step_metric"] = float(metric)
            seg["quality"] = q
            switched += 1

        for i, curr in enumerate(segments):
            if curr["source"] == self.target_video:
                continue

            curr_start = float(curr["start"])
            curr_target = float(curr["target_start"])
            curr_quality = curr.get("quality", {}) or {}
            curr_combined = float(curr_quality.get("combined", 0.0))
            step_tolerance = max(0.9, float(curr["duration"]) * 0.30)
            hard_tolerance = max(2.0, step_tolerance * 2.2)
            step_deviations: List[float] = []
            mapping_jumps: List[float] = []
            same_source_neighbor_count = 0

            def collect_with_neighbor(nei: dict):
                nonlocal same_source_neighbor_count
                if nei["source"] != curr["source"]:
                    return
                same_source_neighbor_count += 1
                nei_start = float(nei["start"])
                nei_target = float(nei["target_start"])
                target_delta = abs(curr_target - nei_target)
                source_delta = abs(curr_start - nei_start)
                if target_delta <= 0:
                    return
                step_deviations.append(abs(source_delta - target_delta))
                curr_mapping = curr_start - curr_target
                nei_mapping = nei_start - nei_target
                mapping_jumps.append(abs(curr_mapping - nei_mapping))

            if i > 0:
                collect_with_neighbor(segments[i - 1])
            if i + 1 < len(segments):
                collect_with_neighbor(segments[i + 1])

            isolated_drift = abs(curr_start - curr_target)
            isolated_limit = max(self.isolated_drift_trigger, float(curr["duration"]) * 0.16)

            # 绝对漂移守卫：
            # 1) 单侧同源（常见边界段）一旦漂移超阈值，直接回退目标段（不再依赖分数）
            # 2) 其余情况仍按 combined 上限判断，避免过度回退
            if isolated_drift > float(self.max_mapping_drift_trigger):
                if same_source_neighbor_count <= 1:
                    fallback_segment(curr, "timeline_absolute_mapping_drift", isolated_drift)
                    continue
            if (
                isolated_drift > float(self.max_mapping_drift_trigger)
                and curr_combined < float(self.max_mapping_drift_combined_floor)
            ):
                fallback_segment(curr, "timeline_absolute_mapping_drift", isolated_drift)
                continue

            # 孤立段守卫：前后都非同源时，限制 target/source 的绝对偏移，避免“单段跳错后又恢复”。
            if same_source_neighbor_count == 0 and isolated_drift > isolated_limit:
                fallback_segment(curr, "timeline_isolated_drift", isolated_drift)
                continue

            # 单侧同源守卫：仅一侧同源时也限制漂移，拦截“边界 1 秒回跳后下一段恢复”的问题。
            if same_source_neighbor_count == 1:
                one_side_limit = max(isolated_limit, float(curr["duration"]) * 0.18)
                if isolated_drift > one_side_limit:
                    fallback_segment(curr, "timeline_single_neighbor_drift", isolated_drift)
                    continue
                # 单侧同源步长/映射跳变守卫：
                # 仅一侧有同源邻居时，若 source 步长或 mapping 出现 1s 级突变，
                # 很容易表现为“局部倒放/回放感”，直接回退当前段。
                one_side_step_limit = max(self.adjacent_lag_trigger, float(curr["duration"]) * 0.16)
                one_side_map_limit = max(self.cross_source_mapping_jump_trigger, float(curr["duration"]) * 0.15)
                if step_deviations:
                    one_side_step_dev = max(step_deviations)
                    if one_side_step_dev > one_side_step_limit:
                        fallback_segment(curr, "timeline_single_neighbor_step_jump", one_side_step_dev)
                        continue
                if mapping_jumps:
                    one_side_map_jump = max(mapping_jumps)
                    if one_side_map_jump > one_side_map_limit:
                        fallback_segment(curr, "timeline_single_neighbor_mapping_jump", one_side_map_jump)
                        continue

            if not step_deviations and not mapping_jumps:
                continue

            max_step_dev = max(step_deviations) if step_deviations else 0.0
            max_map_jump = max(mapping_jumps) if mapping_jumps else 0.0
            bad_step_pairs = sum(1 for d in step_deviations if d > step_tolerance)
            bad_map_pairs = sum(1 for m in mapping_jumps if m > step_tolerance)
            severe = (max_step_dev > hard_tolerance) or (max_map_jump > hard_tolerance)

            if severe:
                reason = "timeline_step_deviation_hard"
                metric = max_step_dev
                if max_map_jump > max_step_dev:
                    reason = "timeline_mapping_jump_hard"
                    metric = max_map_jump
                fallback_segment(curr, reason, metric)
                continue

            if (bad_step_pairs + bad_map_pairs) >= 2:
                metric = max(max_step_dev, max_map_jump)
                fallback_segment(curr, "timeline_step_bi_mismatch", metric)
                continue

        return switched

    def _extract_av_clip(
        self,
        source: Path,
        start: float,
        duration: float,
        output_clip: Path,
        fps_expr: str,
        expected_frames: int = 0,
        include_audio: bool = True,
    ) -> Tuple[bool, str]:
        """提取单段 AV 片段，返回 (是否成功, 错误信息)。"""
        frame_count = max(1, int(expected_frames)) if expected_frames else 0
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(source),
            '-ss', str(start),
            '-t', str(duration),
            '-vf', fps_expr,
            '-reset_timestamps', '1',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        ]
        if include_audio:
            cmd.extend([
                '-af', f'aresample=async=1:first_pts=0,atrim=0:{duration:.6f},asetpts=PTS-STARTPTS',
                '-c:a', 'aac', '-b:a', '128k',
            ])
        else:
            cmd.extend(['-an'])
        if frame_count > 0:
            cmd.extend(['-frames:v', str(frame_count)])
        cmd.append(str(output_clip))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, (result.stderr or "").strip()
        if not output_clip.exists() or output_clip.stat().st_size <= 0:
            return False, "empty_output"
        return True, ""

    def _quick_frame_similarity(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """轻量帧相似度（1-归一化绝对差），用于单帧突刺检测。"""
        if frame_a is None or frame_b is None:
            return 0.0
        mad = np.mean(np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))) / 255.0
        return float(1.0 - mad)

    def _repair_boundary_single_frame_glitches(
        self,
        input_video: Path,
        segments: List[dict],
        fps: float,
    ) -> Tuple[Path, int]:
        """
        修复段边界附近的单帧突刺（A-B-A）：
        - 仅在段边界前后 1 帧检测
        - 仅当 prev/next 高相似且 current 同时与两侧低相似时替换
        """
        if not self.boundary_glitch_fix or len(segments) < 2:
            return input_video, 0
        if not input_video.exists():
            return input_video, 0

        boundary_indices: Set[int] = set()
        fps_safe = max(1.0, float(fps))
        for seg in segments[:-1]:
            boundary_t = float(seg["target_start"]) + float(seg["duration"])
            center = int(round(boundary_t * fps_safe))
            for delta in (-1, 0, 1):
                idx = center + delta
                if idx > 0:
                    boundary_indices.add(idx)
        if not boundary_indices:
            return input_video, 0

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            return input_video, 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps_safe)
        if width <= 0 or height <= 0:
            cap.release()
            return input_video, 0

        features: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
            features.append(small)
        cap.release()

        frame_count = len(features)
        if frame_count < 3:
            return input_video, 0

        replace_indices: Set[int] = set()
        hi = float(self.boundary_glitch_hi_threshold)
        lo = float(self.boundary_glitch_lo_threshold)
        gap_threshold = float(self.boundary_glitch_gap_threshold)
        for idx in sorted(boundary_indices):
            if idx <= 0 or idx >= frame_count - 1:
                continue
            prev_frame = features[idx - 1]
            curr_frame = features[idx]
            next_frame = features[idx + 1]
            sim_prev_next = self._quick_frame_similarity(prev_frame, next_frame)
            sim_prev_curr = self._quick_frame_similarity(prev_frame, curr_frame)
            sim_curr_next = self._quick_frame_similarity(curr_frame, next_frame)
            drop_gap = sim_prev_next - max(sim_prev_curr, sim_curr_next)
            if (
                sim_prev_next >= hi
                and (
                    (sim_prev_curr <= lo and sim_curr_next <= lo)
                    or (drop_gap >= gap_threshold)
                )
            ):
                replace_indices.add(idx)

        if not replace_indices:
            return input_video, 0

        repaired_video_only = self.temp_dir / "temp_output_boundary_repaired_video.mp4"
        encoder_args = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
        if self.boundary_glitch_use_videotoolbox and self._ffmpeg_has_encoder('h264_videotoolbox'):
            # Mac 硬编优先：显著降低单帧突刺修复耗时；高码率保证观感。
            encoder_args = [
                '-c:v', 'h264_videotoolbox',
                '-b:v', '10M',
                '-maxrate', '14M',
                '-bufsize', '20M',
            ]
        encode_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{source_fps:.6f}",
            "-i", "-",
            "-an",
            *encoder_args,
            str(repaired_video_only),
        ]

        proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cap = cv2.VideoCapture(str(input_video))
        prev_full_frame = None
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                out_frame = frame
                if frame_idx in replace_indices and prev_full_frame is not None:
                    out_frame = prev_full_frame
                if proc.stdin is None:
                    break
                proc.stdin.write(out_frame.tobytes())
                prev_full_frame = frame
                frame_idx += 1
            if proc.stdin is not None:
                proc.stdin.close()
            ret = proc.wait()
        finally:
            cap.release()
            if proc.poll() is None:
                proc.kill()

        if ret != 0 or not repaired_video_only.exists():
            return input_video, 0

        # 若后续会强制覆盖目标音轨，这里不再回灌输入音轨，
        # 避免输入音轨时长异常时把视频错误截短。
        if self.force_target_audio:
            return repaired_video_only, len(replace_indices)

        input_duration = self.get_video_duration(input_video)
        repaired_output = self.temp_dir / "temp_output_boundary_repaired.mp4"
        mux_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(repaired_video_only),
            "-i", str(input_video),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-t", f"{input_duration:.3f}",
            str(repaired_output),
        ]
        mux = subprocess.run(mux_cmd, capture_output=True, text=True)
        if mux.returncode != 0 or not repaired_output.exists():
            return input_video, 0

        return repaired_output, len(replace_indices)

    def _generate_output(self, segments: List[dict], output_path: str, target_duration: float) -> bool:
        """生成输出 - 同步音视频"""
        import time

        av_clips = []
        output_fps = max(12.0, float(self.output_fps))
        fps_expr = f"fps={output_fps:.6f},format=yuv420p"
        fps_out = f"{output_fps:.6f}"
        render_fallbacks = 0
        render_target_audio_only = bool(self.force_target_audio)
        render_perf = time.perf_counter()
        clip_extract_total = 0.0
        concat_elapsed = 0.0
        concat_mode = ""
        trim_elapsed = 0.0
        mux_elapsed = 0.0
        boundary_glitch_fixed_frames = 0
        stitched_audio_duration = 0.0
        final_audio_duration = 0.0
        auto_target_audio_fallback = False
        audio_audible_check_passed = True
        audio_audible_samples_dbfs: List[Optional[float]] = []
        audio_silence_auto_fixed = False
        self.last_render_metrics = {
            "status": "running",
            "segments_total": len(segments),
            "render_fallback_segments": 0,
            "clip_extract_total_sec": 0.0,
            "concat_elapsed_sec": 0.0,
            "trim_elapsed_sec": 0.0,
            "mux_elapsed_sec": 0.0,
            "concat_mode": "",
            "render_target_audio_only": bool(render_target_audio_only),
            "boundary_glitch_fixed_frames": 0,
            "stitched_audio_duration": 0.0,
            "final_audio_duration": 0.0,
            "auto_target_audio_fallback": False,
            "audio_audible_check_passed": True,
            "audio_audible_samples_dbfs": [],
            "audio_silence_auto_fixed": False,
            "render_total_sec": 0.0,
        }
        
        render_worker_count = max(1, int(getattr(self, "render_workers", 1)))

        def extract_one_segment(seg: dict) -> Dict[str, object]:
            seg_source = seg['source']
            seg_start = seg['start']
            seg_expected_frames = max(1, int(round(float(seg['duration']) * output_fps)))
            av_clip = self.temp_dir / f"seg_{seg['index']:03d}_av.mp4"
            seg_extract_perf = time.perf_counter()

            ok, err = self._extract_av_clip(
                seg_source,
                seg_start,
                seg['duration'],
                av_clip,
                fps_expr,
                expected_frames=seg_expected_frames,
                include_audio=(not render_target_audio_only),
            )

            fallback_used = False
            if not ok and seg_source != self.target_video:
                fallback_used = True
                q = seg.get("quality", {}) or {}
                q["fallback"] = True
                q["fallback_reason"] = "render_extract_failed"
                q["render_extract_error"] = err[:240]
                seg["quality"] = q
                seg["source"] = self.target_video
                seg["start"] = seg["target_start"]
                ok, err = self._extract_av_clip(
                    seg["source"],
                    seg["start"],
                    seg["duration"],
                    av_clip,
                    fps_expr,
                    expected_frames=seg_expected_frames,
                    include_audio=(not render_target_audio_only),
                )

            seg_elapsed = time.perf_counter() - seg_extract_perf
            return {
                "seg": seg,
                "clip": av_clip,
                "ok": bool(ok),
                "err": str(err),
                "elapsed": float(seg_elapsed),
                "fallback_used": bool(fallback_used),
            }

        clip_by_index: Dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=render_worker_count) as pool:
            future_map = {pool.submit(extract_one_segment, seg): seg for seg in segments}
            for future in as_completed(future_map):
                item = future.result()
                seg = item["seg"]
                av_clip = item["clip"]
                ok = bool(item["ok"])
                err = str(item["err"])
                seg_elapsed = float(item["elapsed"])
                fallback_used = bool(item["fallback_used"])

                if fallback_used:
                    render_fallbacks += 1
                clip_extract_total += seg_elapsed
                seg["render_extract_elapsed_sec"] = round(float(seg_elapsed), 3)

                if not ok:
                    self.last_render_metrics = {
                        "status": "failed",
                        "error": f"segment_extract_failed:{seg['index']}",
                        "error_detail": err[:240],
                        "segments_total": len(segments),
                        "render_fallback_segments": render_fallbacks,
                        "clip_extract_total_sec": round(float(clip_extract_total), 3),
                        "concat_elapsed_sec": round(float(concat_elapsed), 3),
                        "trim_elapsed_sec": round(float(trim_elapsed), 3),
                        "mux_elapsed_sec": round(float(mux_elapsed), 3),
                        "concat_mode": concat_mode,
                        "render_target_audio_only": bool(render_target_audio_only),
                        "boundary_glitch_fixed_frames": int(boundary_glitch_fixed_frames),
                        "render_total_sec": round(float(time.perf_counter() - render_perf), 3),
                    }
                    print(f"❌ 段 {seg['index']+1} 提取失败，无法生成完整时间轴: {err}")
                    return False

                clip_by_index[int(seg["index"])] = Path(av_clip)

        for seg in sorted(segments, key=lambda x: int(x["index"])):
            clip = clip_by_index.get(int(seg["index"]))
            if clip is not None:
                av_clips.append(clip)

        if not av_clips:
            print("❌ 没有有效的 AV 片段")
            return False

        print(f"   AV片段: {len(av_clips)}")
        if render_fallbacks > 0:
            print(f"   🛟 渲染期自动兜底: {render_fallbacks} 段（防止静默丢段）")

        # 直接拼接 AV 片段
        av_concat = self.temp_dir / "av_concat.txt"
        with open(av_concat, 'w') as f:
            for clip in av_clips:
                f.write(f"file '{clip}'\n")

        temp_output = self.temp_dir / "temp_output.mp4"
        concat_mode = ""
        concat_perf = time.perf_counter()

        copy_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(av_concat),
            '-t', f"{target_duration:.3f}",
            '-c', 'copy',
            str(temp_output)
        ]
        concat_proc = subprocess.run(copy_cmd, capture_output=True, text=True)
        if concat_proc.returncode == 0 and temp_output.exists():
            concat_mode = "copy"
        else:
            reencode_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'concat', '-safe', '0',
                '-i', str(av_concat),
                '-t', f"{target_duration:.3f}",
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            ]
            if render_target_audio_only:
                reencode_cmd.extend(['-an'])
            else:
                reencode_cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            reencode_cmd.append(str(temp_output))
            concat_proc = subprocess.run(reencode_cmd, capture_output=True, text=True)
            if concat_proc.returncode == 0 and temp_output.exists():
                concat_mode = "reencode"
        concat_elapsed = time.perf_counter() - concat_perf

        if not concat_mode:
            self.last_render_metrics = {
                "status": "failed",
                "error": "concat_failed",
                "error_detail": (concat_proc.stderr or "").strip()[:240],
                "segments_total": len(segments),
                "render_fallback_segments": render_fallbacks,
                "clip_extract_total_sec": round(float(clip_extract_total), 3),
                "concat_elapsed_sec": round(float(concat_elapsed), 3),
                "trim_elapsed_sec": round(float(trim_elapsed), 3),
                "mux_elapsed_sec": round(float(mux_elapsed), 3),
                "concat_mode": concat_mode,
                "render_target_audio_only": bool(render_target_audio_only),
                "boundary_glitch_fixed_frames": int(boundary_glitch_fixed_frames),
                "stitched_audio_duration": round(float(stitched_audio_duration), 3),
                "final_audio_duration": round(float(final_audio_duration), 3),
                "auto_target_audio_fallback": bool(auto_target_audio_fallback),
                "render_total_sec": round(float(time.perf_counter() - render_perf), 3),
            }
            print("❌ AV 拼接失败")
            return False

        current_duration = self.get_video_duration(temp_output)
        print(f"   当前视频时长: {current_duration:.2f}s, 目标: {target_duration:.2f}s")

        # concat 已经带 -t，默认不再做二次全片重编码 trim。
        # 仅在极少数时间戳异常导致明显超长时，用 stream copy 快速截断。
        if current_duration > (target_duration + 0.12):
            trimmed_output = self.temp_dir / "temp_output_trimmed_copy.mp4"
            trim_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(temp_output),
                '-t', f"{target_duration:.3f}",
                '-c', 'copy',
                str(trimmed_output)
            ]
            trim_perf = time.perf_counter()
            trim_proc = subprocess.run(trim_cmd, capture_output=True, text=True)
            trim_elapsed = time.perf_counter() - trim_perf
            if trim_proc.returncode == 0 and trimmed_output.exists():
                temp_output = trimmed_output

        repaired_output, boundary_glitch_fixed_frames = self._repair_boundary_single_frame_glitches(
            temp_output,
            segments,
            output_fps,
        )
        if boundary_glitch_fixed_frames > 0:
            temp_output = repaired_output
            print(f"   🩹 边界单帧突刺修复: {boundary_glitch_fixed_frames} 帧")

        stitched_audio_duration = self.get_audio_duration(temp_output)
        min_audio_required = max(5.0, float(target_duration) * 0.90)
        auto_target_audio_fallback = (
            stitched_audio_duration <= 0.0
            or (stitched_audio_duration + 0.5) < min_audio_required
        )
        if auto_target_audio_fallback and not self.force_target_audio:
            print(
                f"   ⚠️ 拼接音轨偏短: {stitched_audio_duration:.2f}s/{target_duration:.2f}s，"
                "自动切换目标音轨封装"
            )

        use_target_audio_mux = bool(self.force_target_audio or auto_target_audio_fallback)

        if use_target_audio_mux:
            # 使用目标音轨进行最终封装（显式启用或音轨偏短自动兜底）
            muxed_output = self.temp_dir / "temp_output_target_audio.mp4"
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(temp_output),
                '-i', str(self.target_video),
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '128k',
                '-t', f"{target_duration:.3f}",
                '-shortest',
                str(muxed_output)
            ]
            mux_perf = time.perf_counter()
            subprocess.run(cmd, capture_output=True)
            mux_elapsed = time.perf_counter() - mux_perf

            if muxed_output.exists():
                shutil.copy(muxed_output, output_path)
                final_audio_duration = self.get_audio_duration(muxed_output)
                print("   🔊 已使用目标音轨封装")
            else:
                if auto_target_audio_fallback:
                    self.last_render_metrics = {
                        "status": "failed",
                        "error": "target_audio_mux_failed_after_short_audio",
                        "segments_total": len(segments),
                        "render_fallback_segments": render_fallbacks,
                        "clip_extract_total_sec": round(float(clip_extract_total), 3),
                        "concat_elapsed_sec": round(float(concat_elapsed), 3),
                        "trim_elapsed_sec": round(float(trim_elapsed), 3),
                        "mux_elapsed_sec": round(float(mux_elapsed), 3),
                        "concat_mode": concat_mode,
                        "render_target_audio_only": bool(render_target_audio_only),
                        "boundary_glitch_fixed_frames": int(boundary_glitch_fixed_frames),
                        "stitched_audio_duration": round(float(stitched_audio_duration), 3),
                        "final_audio_duration": round(float(final_audio_duration), 3),
                        "auto_target_audio_fallback": bool(auto_target_audio_fallback),
                        "render_total_sec": round(float(time.perf_counter() - render_perf), 3),
                    }
                    print("❌ 目标音轨封装失败，且拼接音轨明显偏短，终止输出")
                    return False
                shutil.copy(temp_output, output_path)
                final_audio_duration = stitched_audio_duration
                print("   ⚠️ 目标音轨封装失败，回退为原拼接音轨")
        else:
            shutil.copy(temp_output, output_path)
            final_audio_duration = stitched_audio_duration
            print("   🔊 保留拼接片段原音轨（未使用目标音轨封装）")

        if Path(output_path).exists():
            final_duration = self.get_video_duration(Path(output_path))
            print(f"   最终视频时长: {final_duration:.2f}s")
            self.last_render_metrics = {
                "status": "ok",
                "segments_total": len(segments),
                "render_fallback_segments": render_fallbacks,
                "clip_extract_total_sec": round(float(clip_extract_total), 3),
                "concat_elapsed_sec": round(float(concat_elapsed), 3),
                "trim_elapsed_sec": round(float(trim_elapsed), 3),
                "mux_elapsed_sec": round(float(mux_elapsed), 3),
                "concat_mode": concat_mode,
                "render_target_audio_only": bool(render_target_audio_only),
                "boundary_glitch_fixed_frames": int(boundary_glitch_fixed_frames),
                "stitched_audio_duration": round(float(stitched_audio_duration), 3),
                "final_audio_duration": round(float(final_audio_duration), 3),
                "auto_target_audio_fallback": bool(auto_target_audio_fallback),
                "render_total_sec": round(float(time.perf_counter() - render_perf), 3),
                "final_duration": round(float(final_duration), 3),
            }
            if self.run_ai_verify_snapshots:
                # 可选：AI 抽样帧人工核验（默认关闭以优先保证出片速度）
                self.ai_verify_video(output_path)
            return True
        self.last_render_metrics = {
            "status": "failed",
            "error": "output_not_found",
            "segments_total": len(segments),
            "render_fallback_segments": render_fallbacks,
            "clip_extract_total_sec": round(float(clip_extract_total), 3),
            "concat_elapsed_sec": round(float(concat_elapsed), 3),
            "trim_elapsed_sec": round(float(trim_elapsed), 3),
            "mux_elapsed_sec": round(float(mux_elapsed), 3),
            "concat_mode": concat_mode,
            "render_target_audio_only": bool(render_target_audio_only),
            "boundary_glitch_fixed_frames": int(boundary_glitch_fixed_frames),
            "stitched_audio_duration": round(float(stitched_audio_duration), 3),
            "final_audio_duration": round(float(final_audio_duration), 3),
            "auto_target_audio_fallback": bool(auto_target_audio_fallback),
            "render_total_sec": round(float(time.perf_counter() - render_perf), 3),
        }
        return False
    
    def ai_verify_video(self, output_path: str) -> bool:
        """
        【强制】AI亲自查看视频关键时间点
        必须实际查看每一帧，不能依赖自动工具的百分比
        """
        import os
        
        print("\n" + "="*70)
        print("🤖 【强制】AI亲自查看视频内容")
        print("="*70)
        print("⚠️  警告：必须实际查看每一帧，不能只看百分比！")
        
        # 创建检查目录
        project_root = Path(__file__).resolve().parent
        check_dir = project_root / "runtime" / "temp_outputs" / ("ai_check_round" + str(getattr(self, 'round_num', 'X')))
        check_dir.mkdir(parents=True, exist_ok=True)
        
        # 关键时间点检查（动态采样，适配任意时长）
        output_duration = self.get_video_duration(Path(output_path))
        if output_duration <= 1:
            check_points = [0]
        else:
            raw_points = np.linspace(0, output_duration - 1, num=5)
            check_points = sorted(set(int(p) for p in raw_points))
        
        frame_pairs = []  # 存储帧路径用于后续查看
        
        for time_sec in check_points:
            print(f"\n🔍 提取时间点 {time_sec}s 的帧...")
            
            # 提取原视频帧
            original_frame = check_dir / f"orig_{time_sec:03d}s.jpg"
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(time_sec), '-i', str(self.target_video),
                '-vframes', '1', '-q:v', '2', str(original_frame)
            ]
            subprocess.run(cmd, capture_output=True)
            
            # 提取生成视频帧
            generated_frame = check_dir / f"gen_{time_sec:03d}s.jpg"
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(time_sec), '-i', str(output_path),
                '-vframes', '1', '-q:v', '2', str(generated_frame)
            ]
            subprocess.run(cmd, capture_output=True)
            
            if original_frame.exists() and generated_frame.exists():
                frame_pairs.append({
                    'time': time_sec,
                    'original': original_frame,
                    'generated': generated_frame
                })
                print(f"   ✅ 原视频帧: {original_frame}")
                print(f"   ✅ 生成帧: {generated_frame}")
            else:
                print(f"   ❌ 帧提取失败")
        
        # 保存检查目录路径供外部查看
        self.last_check_dir = check_dir
        
        print("\n" + "="*70)
        print("📋 【下一步】请使用 read 工具查看以下帧对：")
        print("="*70)
        for pair in frame_pairs:
            print(f"\n时间点 {pair['time']}s:")
            print(f"  read {pair['original']}")
            print(f"  read {pair['generated']}")
        
        print("\n" + "="*70)
        print("⚠️  必须完成以下步骤：")
        print("1. 使用 read 工具查看每一对帧")
        print("2. 对比画面内容是否一致")
        print("3. 输出查看报告")
        print("4. 等待用户确认")
        print("="*70)
        
        return True


def run_evidence_validation(
    root: Path,
    target_video: Path,
    candidate_video: Path,
    interval: float = 3.0,
    clip_duration: float = 2.0,
    max_points: int = 1200,
    asr_mode: str = "auto",
    target_sub: str = "",
    candidate_sub: str = "",
    asr_cmd: str = "",
    asr_python: str = "",
    asr_model: str = "base",
    language: str = "zh",
    whisper_python_candidates: Optional[List[str]] = None,
    clip_elapsed_sec: Optional[float] = None,
    candidate_mode: str = "reconstructed",
    report_output_root: Optional[str] = None,
) -> Optional[Dict]:
    """
    证据级一致性验证：
    - 抽帧画面
    - OCR 字幕
    - ASR 音频文本（可用时）
    """
    script = root / "skills" / "ai-video-audit" / "scripts" / "build_ai_video_audit_bundle.py"
    if not script.exists():
        print(f"⚠️ 未找到证据验证脚本: {script}")
        return None

    def detect_whisper_python() -> str:
        candidates: List[str] = []

        def add_candidate(raw: Optional[str]) -> None:
            p = (raw or "").strip()
            if p:
                candidates.append(p)

        add_candidate(sys.executable)
        add_candidate(shutil.which("python3"))
        add_candidate(shutil.which("python"))
        for item in (whisper_python_candidates or []):
            add_candidate(item)

        seen = set()
        for p in candidates:
            if not p or p in seen:
                continue
            seen.add(p)
            pp = Path(p)
            if not pp.exists():
                continue
            try:
                probe = subprocess.run([p, "-m", "whisper", "--help"], capture_output=True, text=True)
            except Exception:
                continue
            if probe.returncode == 0:
                return p
        return ""

    if not asr_cmd and not asr_python:
        auto_asr_python = detect_whisper_python()
        if auto_asr_python:
            asr_python = auto_asr_python
            print(f"   🔎 自动探测到 Whisper Python: {asr_python}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if report_output_root and str(report_output_root).strip():
        base_report_root = Path(str(report_output_root).strip()).expanduser()
    else:
        # 默认与输出视频同目录，避免报告散落在 runtime 临时目录
        base_report_root = candidate_video.parent / "ai_evidence_check"
    out_dir = base_report_root / f"{target_video.stem}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--target", str(target_video),
        "--candidate", str(candidate_video),
        "--interval", f"{interval:.3f}",
        "--clip-duration", f"{max(0.2, float(clip_duration)):.3f}",
        "--max-points", str(max(1, int(max_points))),
        "--asr", str(asr_mode),
        "--asr-model", str(asr_model),
        "--language", str(language),
        "--output-dir", str(out_dir),
    ]
    if target_sub:
        cmd.extend(["--target-sub", target_sub])
    if candidate_sub:
        cmd.extend(["--candidate-sub", candidate_sub])
    if asr_cmd:
        cmd.extend(["--asr-cmd", asr_cmd])
    if asr_python:
        cmd.extend(["--asr-python", asr_python])
    if clip_elapsed_sec is not None:
        cmd.extend(["--clip-elapsed-sec", f"{max(0.0, float(clip_elapsed_sec)):.3f}"])
    if candidate_mode:
        cmd.extend(["--candidate-mode", str(candidate_mode)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "evidence_validation_failed").strip()
        print(f"⚠️ 证据级验证执行失败: {err[:500]}")
        return None

    manifest_path = out_dir / "audit_manifest.json"
    if not manifest_path.exists():
        print(f"⚠️ 证据级验证缺少清单: {manifest_path}")
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"⚠️ 证据清单解析失败: {exc}")
        return {
            "output_dir": str(out_dir),
            "manifest_path": str(manifest_path),
        }

    return {
        "output_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "report_html": manifest.get("comparison_report_html"),
        "summary": manifest.get("summary", {}),
        "asr_backend": manifest.get("asr_backend"),
        "asr_error": manifest.get("asr_error"),
    }


def main():
    root = Path(__file__).resolve().parent
    default_cache = root / "runtime" / "cache"

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", help="配置文件路径（JSON）")
    pre_args, _ = pre_parser.parse_known_args()

    try:
        cfg, cfg_path = load_section_config(root, "v6_fast", explicit_path=pre_args.config)
        cache_default = cfg_str(cfg, "cache", str(default_cache))
        segment_duration_default = cfg_float(cfg, "segment_duration", 5.0)
        workers_default = cfg_int(cfg, "workers", 0)
        render_workers_default = cfg_int(cfg, "render_workers", 0)
        low_score_threshold_default = cfg_float(cfg, "low_score_threshold", 0.82)
        rematch_window_default = cfg_int(cfg, "rematch_window", 2)
        rematch_max_window_default = cfg_int(cfg, "rematch_max_window", 12)
        continuity_weight_default = cfg_float(cfg, "continuity_weight", 0.05)
        strict_visual_verify_default = cfg_bool(cfg, "strict_visual_verify", True)
        strict_verify_min_sim_default = cfg_float(cfg, "strict_verify_min_sim", 0.78)
        tail_guard_seconds_default = cfg_float(cfg, "tail_guard_seconds", 20.0)
        tail_verify_min_avg_default = cfg_float(cfg, "tail_verify_min_avg", 0.84)
        tail_verify_min_floor_default = cfg_float(cfg, "tail_verify_min_floor", 0.78)
        adjacent_overlap_trigger_default = cfg_float(cfg, "adjacent_overlap_trigger", 0.6)
        adjacent_lag_trigger_default = cfg_float(cfg, "adjacent_lag_trigger", 0.8)
        isolated_drift_trigger_default = cfg_float(cfg, "isolated_drift_trigger", 0.8)
        cross_source_mapping_jump_trigger_default = cfg_float(cfg, "cross_source_mapping_jump_trigger", 0.75)
        max_mapping_drift_trigger_default = cfg_float(cfg, "max_mapping_drift_trigger", 0.75)
        max_mapping_drift_combined_floor_default = cfg_float(cfg, "max_mapping_drift_combined_floor", 0.95)
        use_audio_matching_default = cfg_bool(cfg, "use_audio_matching", False)
        force_target_audio_default = cfg_bool(cfg, "force_target_audio", False)
        verify_interval_default = cfg_float(cfg, "verify_interval", 3.0)
        verify_clip_duration_default = cfg_float(cfg, "verify_clip_duration", 2.0)
        verify_max_points_default = cfg_int(cfg, "verify_max_points", 1200)
        verify_asr_mode_default = cfg_str(cfg, "verify_asr_mode", "auto")
        verify_target_sub_default = cfg_str(cfg, "verify_target_sub", "")
        verify_output_sub_default = cfg_str(cfg, "verify_output_sub", "")
        verify_asr_cmd_default = cfg_str(cfg, "verify_asr_cmd", "")
        verify_asr_python_default = cfg_str(cfg, "verify_asr_python", "")
        verify_asr_model_default = cfg_str(cfg, "verify_asr_model", "base")
        verify_language_default = cfg_str(cfg, "verify_language", "zh")
        verify_whisper_candidates_default = cfg_str_list(cfg, "verify_whisper_python_candidates", [])
        verify_output_root_default = cfg_str(cfg, "verify_output_root", "")
        run_evidence_validation_default = cfg_bool(cfg, "run_evidence_validation", False)
        run_ai_verify_snapshots_default = cfg_bool(cfg, "run_ai_verify_snapshots", False)
        boundary_glitch_fix_default = cfg_bool(cfg, "boundary_glitch_fix", True)
        boundary_glitch_hi_threshold_default = cfg_float(cfg, "boundary_glitch_hi_threshold", 0.965)
        boundary_glitch_lo_threshold_default = cfg_float(cfg, "boundary_glitch_lo_threshold", 0.94)
        boundary_glitch_gap_threshold_default = cfg_float(cfg, "boundary_glitch_gap_threshold", 0.03)
        audio_guard_enabled_default = cfg_bool(cfg, "audio_guard_enabled", True)
        audio_guard_score_trigger_default = cfg_float(cfg, "audio_guard_score_trigger", 0.93)
        audio_guard_sample_duration_default = cfg_float(cfg, "audio_guard_sample_duration", 1.8)
        audio_guard_min_similarity_default = cfg_float(cfg, "audio_guard_min_similarity", 0.34)
        audio_guard_hard_floor_default = cfg_float(cfg, "audio_guard_hard_floor", 0.18)
        audio_guard_shift_margin_default = cfg_float(cfg, "audio_guard_shift_margin", 0.16)
        allow_numeric_fallback_default = cfg_bool(cfg, "allow_numeric_fallback", False)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    parser = argparse.ArgumentParser(description="通用极速高精度重构器 V3 + pHash")
    parser.add_argument("--config", default=str(cfg_path) if cfg_path else "", help="配置文件路径（JSON）")
    parser.add_argument("--target", required=True, help="目标视频路径")
    parser.add_argument("--source-dir", required=True, help="源视频目录")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--cache", default=cache_default, help="缓存目录")
    parser.add_argument("--segment-duration", type=float, default=segment_duration_default, help="分段时长（秒）")
    parser.add_argument("--workers", type=int, default=workers_default, help="并行线程数（0=自动）")
    parser.add_argument("--render-workers", type=int, default=render_workers_default, help="渲染分段提取线程数（0=自动）")
    parser.add_argument("--low-score-threshold", type=float, default=low_score_threshold_default, help="低分段重匹配阈值")
    parser.add_argument("--rematch-window", type=int, default=rematch_window_default, help="局部重匹配起始窗口（秒）")
    parser.add_argument("--rematch-max-window", type=int, default=rematch_max_window_default, help="局部重匹配最大窗口（秒）")
    parser.add_argument("--continuity-weight", type=float, default=continuity_weight_default, help="连续性奖励权重（0~0.2）")
    add_bool_arg(parser, "--strict-visual-verify", strict_visual_verify_default, "启用严格画面核验")
    parser.add_argument("--strict-verify-min-sim", type=float, default=strict_verify_min_sim_default, help="严格画面核验最低相似度阈值")
    parser.add_argument("--tail-guard-seconds", type=float, default=tail_guard_seconds_default, help="尾段守卫时长（秒）")
    parser.add_argument("--tail-verify-min-avg", type=float, default=tail_verify_min_avg_default, help="尾段守卫平均相似度阈值")
    parser.add_argument("--tail-verify-min-floor", type=float, default=tail_verify_min_floor_default, help="尾段守卫最低点相似度阈值")
    parser.add_argument("--adjacent-overlap-trigger", type=float, default=adjacent_overlap_trigger_default, help="相邻段重叠触发阈值（秒）")
    parser.add_argument("--adjacent-lag-trigger", type=float, default=adjacent_lag_trigger_default, help="相邻段慢进触发阈值（秒）")
    parser.add_argument("--isolated-drift-trigger", type=float, default=isolated_drift_trigger_default, help="孤立段时间漂移触发阈值（秒）")
    parser.add_argument("--cross-source-mapping-jump-trigger", type=float, default=cross_source_mapping_jump_trigger_default, help="跨源切换映射跳变触发阈值（秒）")
    parser.add_argument("--max-mapping-drift-trigger", type=float, default=max_mapping_drift_trigger_default, help="绝对映射漂移触发阈值（秒，超出将优先回退目标段）")
    parser.add_argument("--max-mapping-drift-combined-floor", type=float, default=max_mapping_drift_combined_floor_default, help="绝对映射漂移守卫置信度上限（combined 低于该值才触发）")
    add_bool_arg(parser, "--use-audio-matching", use_audio_matching_default, "启用音频指纹参与匹配")
    add_bool_arg(parser, "--force-target-audio", force_target_audio_default, "最终封装强制使用目标音轨")
    parser.add_argument("--verify-interval", type=float, default=verify_interval_default, help="证据级验证间隔（秒）")
    parser.add_argument("--verify-clip-duration", type=float, default=verify_clip_duration_default, help="证据级验证音频切片时长（秒）")
    parser.add_argument("--verify-max-points", type=int, default=verify_max_points_default, help="证据级验证最大检查点数量")
    parser.add_argument(
        "--verify-asr-mode",
        choices=["auto", "none", "faster_whisper", "whisper"],
        default=verify_asr_mode_default,
        help="证据级验证 ASR 模式",
    )
    parser.add_argument("--verify-target-sub", default=verify_target_sub_default, help="证据级验证：目标字幕文件（可选）")
    parser.add_argument("--verify-output-sub", default=verify_output_sub_default, help="证据级验证：输出字幕文件（可选）")
    parser.add_argument("--verify-asr-cmd", default=verify_asr_cmd_default, help="证据级验证：whisper 命令（可选）")
    parser.add_argument("--verify-asr-python", default=verify_asr_python_default, help="证据级验证：whisper 所在 python（可选）")
    parser.add_argument("--verify-asr-model", default=verify_asr_model_default, help="证据级验证：ASR 模型")
    parser.add_argument("--verify-language", default=verify_language_default, help="证据级验证：ASR 语种")
    parser.add_argument("--verify-output-root", default=verify_output_root_default, help="证据级验证：报告输出根目录（默认输出视频同目录）")
    add_bool_arg(parser, "--run-evidence-validation", run_evidence_validation_default, "重构完成后是否自动运行证据级验证")
    add_bool_arg(parser, "--run-ai-verify-snapshots", run_ai_verify_snapshots_default, "渲染完成后是否执行 AI 抽样帧核验")
    add_bool_arg(parser, "--boundary-glitch-fix", boundary_glitch_fix_default, "启用段边界单帧突刺修复")
    parser.add_argument("--boundary-glitch-hi-threshold", type=float, default=boundary_glitch_hi_threshold_default, help="边界突刺修复：prev/next 高相似阈值")
    parser.add_argument("--boundary-glitch-lo-threshold", type=float, default=boundary_glitch_lo_threshold_default, help="边界突刺修复：current 低相似阈值")
    parser.add_argument("--boundary-glitch-gap-threshold", type=float, default=boundary_glitch_gap_threshold_default, help="边界突刺修复：当前帧相对掉分阈值")
    add_bool_arg(parser, "--audio-guard-enabled", audio_guard_enabled_default, "启用轻量音频守卫（拦截画面对但台词错位）")
    parser.add_argument("--audio-guard-score-trigger", type=float, default=audio_guard_score_trigger_default, help="音频守卫触发分数阈值（combined 低于该值触发）")
    parser.add_argument("--audio-guard-sample-duration", type=float, default=audio_guard_sample_duration_default, help="音频守卫采样时长（秒）")
    parser.add_argument("--audio-guard-min-similarity", type=float, default=audio_guard_min_similarity_default, help="音频守卫最小对齐相似度阈值")
    parser.add_argument("--audio-guard-hard-floor", type=float, default=audio_guard_hard_floor_default, help="音频守卫硬兜底阈值（低于即回退）")
    parser.add_argument("--audio-guard-shift-margin", type=float, default=audio_guard_shift_margin_default, help="音频守卫偏移收益阈值（用于判定音频错位）")
    parser.add_argument(
        "--verify-whisper-candidates",
        default=",".join(verify_whisper_candidates_default),
        help="证据级验证：Whisper Python 候选列表（逗号分隔）",
    )
    add_bool_arg(parser, "--allow-numeric-fallback", allow_numeric_fallback_default, "证据级验证失败时是否允许回退到数值一致性检查")
    args = parser.parse_args()

    target = Path(args.target)
    source_dir = Path(args.source_dir)
    if args.output:
        output = Path(args.output)
    else:
        output = root / "runtime" / "temp_outputs" / f"{target.stem}_reconstructed_v3.mp4"
    cache = Path(args.cache)

    if not target.exists():
        print(f"❌ 目标视频不存在: {target}")
        return
    if not source_dir.exists():
        print(f"❌ 源视频目录不存在: {source_dir}")
        return

    source_videos = [str(f) for f in source_dir.iterdir() if f.suffix.lower() == '.mp4']
    if not source_videos:
        print(f"❌ 源视频目录中未找到 mp4: {source_dir}")
        return

    print("="*70)
    print("通用极速高精度重构 V3")
    print("="*70)
    if args.config:
        print(f"🧩 配置文件: {args.config}")

    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    reconstructor = FastHighPrecisionReconstructor(str(target), source_videos, str(cache))
    reconstructor.segment_duration = max(1.0, args.segment_duration)
    if args.workers and args.workers > 0:
        reconstructor.max_workers = max(1, args.workers)
    if args.render_workers and args.render_workers > 0:
        reconstructor.render_workers = max(1, args.render_workers)
    reconstructor.low_score_threshold = min(0.95, max(0.60, args.low_score_threshold))
    reconstructor.rematch_window = max(1, args.rematch_window)
    reconstructor.rematch_max_window = max(reconstructor.rematch_window, args.rematch_max_window)
    reconstructor.continuity_weight = min(0.2, max(0.0, args.continuity_weight))
    reconstructor.strict_visual_verify = bool(args.strict_visual_verify)
    reconstructor.strict_verify_min_sim = min(1.0, max(0.0, args.strict_verify_min_sim))
    reconstructor.tail_guard_seconds = max(0.0, float(args.tail_guard_seconds))
    reconstructor.tail_verify_min_avg = min(1.0, max(0.0, float(args.tail_verify_min_avg)))
    reconstructor.tail_verify_min_floor = min(1.0, max(0.0, float(args.tail_verify_min_floor)))
    reconstructor.adjacent_overlap_trigger = max(0.0, float(args.adjacent_overlap_trigger))
    reconstructor.adjacent_lag_trigger = max(0.0, float(args.adjacent_lag_trigger))
    reconstructor.isolated_drift_trigger = max(0.0, float(args.isolated_drift_trigger))
    reconstructor.cross_source_mapping_jump_trigger = max(0.0, float(args.cross_source_mapping_jump_trigger))
    reconstructor.max_mapping_drift_trigger = max(0.0, float(args.max_mapping_drift_trigger))
    reconstructor.max_mapping_drift_combined_floor = min(1.0, max(0.0, float(args.max_mapping_drift_combined_floor)))
    reconstructor.boundary_glitch_fix = bool(args.boundary_glitch_fix)
    reconstructor.boundary_glitch_hi_threshold = min(1.0, max(0.0, float(args.boundary_glitch_hi_threshold)))
    reconstructor.boundary_glitch_lo_threshold = min(1.0, max(0.0, float(args.boundary_glitch_lo_threshold)))
    reconstructor.boundary_glitch_gap_threshold = min(1.0, max(0.0, float(args.boundary_glitch_gap_threshold)))
    reconstructor.audio_guard_enabled = bool(args.audio_guard_enabled)
    reconstructor.audio_guard_score_trigger = min(1.0, max(0.0, float(args.audio_guard_score_trigger)))
    reconstructor.audio_guard_sample_duration = max(0.2, float(args.audio_guard_sample_duration))
    reconstructor.audio_guard_min_similarity = min(1.0, max(0.0, float(args.audio_guard_min_similarity)))
    reconstructor.audio_guard_hard_floor = min(
        reconstructor.audio_guard_min_similarity,
        max(0.0, float(args.audio_guard_hard_floor)),
    )
    reconstructor.audio_guard_shift_margin = min(1.0, max(0.0, float(args.audio_guard_shift_margin)))
    reconstructor.use_audio_matching = bool(args.use_audio_matching)
    reconstructor.force_target_audio = bool(args.force_target_audio)
    reconstructor.run_ai_verify_snapshots = bool(args.run_ai_verify_snapshots)
    verify_whisper_candidates = split_csv(args.verify_whisper_candidates)

    try:
        success = reconstructor.reconstruct_fast(str(output))

        if success:
            print("\n🎉 极速重构完成!")

            if args.run_evidence_validation:
                # 可选：证据级验证（抽帧 + OCR + ASR），默认关闭以优先出片速度
                print("\n正在进行证据级一致性验证（抽帧+OCR+ASR）...")
                evidence = run_evidence_validation(
                    root=root,
                    target_video=target,
                    candidate_video=output,
                    interval=max(0.5, float(args.verify_interval)),
                    clip_duration=max(0.2, float(args.verify_clip_duration)),
                    max_points=max(1, int(args.verify_max_points)),
                    asr_mode=args.verify_asr_mode,
                    target_sub=args.verify_target_sub,
                    candidate_sub=args.verify_output_sub,
                    asr_cmd=args.verify_asr_cmd,
                    asr_python=args.verify_asr_python,
                    asr_model=args.verify_asr_model,
                    language=args.verify_language,
                    whisper_python_candidates=verify_whisper_candidates,
                    clip_elapsed_sec=round(float(reconstructor.total_elapsed_sec), 3),
                    candidate_mode="reconstructed",
                    report_output_root=args.verify_output_root,
                )
                if evidence:
                    summary = evidence.get("summary", {}) or {}
                    print("\n✅ 证据级验证完成")
                    print(f"   证据目录: {evidence.get('output_dir')}")
                    print(f"   清单文件: {evidence.get('manifest_path')}")
                    if evidence.get("report_html"):
                        print(f"   详细报告: {evidence.get('report_html')}")
                    print(f"   asr_backend: {evidence.get('asr_backend')}")
                    if evidence.get("asr_error"):
                        print(f"   asr_note: {evidence.get('asr_error')}")
                    if summary:
                        print(
                            f"   汇总: 检查点 {summary.get('total_points')}, "
                            f"总体 {summary.get('overall_verdict')}, "
                            f"不一致点 {summary.get('mismatch_points')}"
                        )
                else:
                    if args.allow_numeric_fallback:
                        print("\n⚠️ 证据级验证失败，按参数回退到数值一致性检查...")
                        from av_consistency_checker import AVConsistencyChecker
                        checker = AVConsistencyChecker(str(target), str(output))
                        results = checker.check_consistency(interval=5.0)
                        if results['statistics']['poor'] == 0:
                            print("\n✅✅✅ 100%通过一致性检查！✅✅✅")
                    else:
                        print("\n⚠️ 证据级验证失败，已禁用数值回退。")
                        print("   若需临时启用旧逻辑，可增加参数: --allow-numeric-fallback")
            else:
                print("\n⏭️ 已跳过证据级验证（可加 --run-evidence-validation 重新开启）")
        else:
            print("\n❌ 重构失败")
    finally:
        print(f"\n📁 临时文件: {reconstructor.temp_dir}")


if __name__ == "__main__":
    main()
