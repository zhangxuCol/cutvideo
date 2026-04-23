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
import uuid
from datetime import datetime
from collections import OrderedDict
from PIL import Image
import imagehash
from pipeline_config import (
    load_section_config,
    cfg_req_str,
    cfg_req_int,
    cfg_req_float,
    cfg_req_bool,
    cfg_req_str_list,
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
    def extract_waveform_fallback() -> list:
        """
        fpcalc 不可用时的后备指纹：
        - 用 ffmpeg 抽取单声道 PCM；
        - 做短时频谱并量化为 32-bit 位码序列（可复用现有汉明距离比较）。
        """
        try:
            sample_rate = 8000
            frame_size = 1024
            hop = 512
            band_count = 32
            max_codes = 320
            clip_dur = max(0.3, float(duration)) if duration is not None else None

            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
            if float(start) > 0:
                cmd.extend(['-ss', f"{float(start):.3f}"])
            cmd.extend(['-i', str(video_path)])
            if clip_dur is not None:
                cmd.extend(['-t', f"{clip_dur:.3f}"])
            cmd.extend([
                '-vn',
                '-ac', '1',
                '-ar', str(sample_rate),
                '-f', 's16le',
                '-',
            ])
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0 or (not result.stdout):
                return []

            samples = np.frombuffer(result.stdout, dtype=np.int16)
            if samples.size < max(sample_rate // 3, frame_size):
                return []

            x = samples.astype(np.float32)
            x -= float(np.mean(x))
            std = float(np.std(x))
            if std < 2.0:
                return []
            x /= std

            window = np.hanning(frame_size).astype(np.float32)
            fingerprint: List[int] = []
            total = int(x.size)
            for pos in range(0, max(1, total - frame_size + 1), hop):
                if pos + frame_size > total:
                    break
                seg = x[pos:pos + frame_size] * window
                spec = np.abs(np.fft.rfft(seg))
                if spec.size <= band_count:
                    continue
                bands = np.array_split(spec[1:], band_count)
                vals = np.array([float(np.mean(b)) for b in bands], dtype=np.float32)
                vals = np.log1p(vals)
                med = float(np.median(vals))
                bits = (vals >= med).astype(np.uint32)
                code = 0
                for bit in bits:
                    code = ((code << 1) | int(bit)) & 0xFFFFFFFF
                fingerprint.append(int(code))
                if len(fingerprint) >= max_codes:
                    break

            return fingerprint
        except Exception:
            return []

    try:
        fpcalc_path = shutil.which('fpcalc')
        if not fpcalc_path:
            return extract_waveform_fallback()

        cmd = ['fpcalc', '-raw']
        if start > 0:
            cmd.extend(['-start', str(int(start))])
        if duration is not None:
            cmd.extend(['-length', str(int(duration))])
        cmd.append(str(video_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return extract_waveform_fallback()
        
        # 解析 fpcalc 输出
        # 格式: DURATION=xxx\nFINGERPRINT=xxx
        output = result.stdout
        fingerprint_match = re.search(r'FINGERPRINT=(.+)', output)
        
        if fingerprint_match:
            # 指纹是以逗号分隔的整数列表
            fingerprint_str = fingerprint_match.group(1)
            fingerprint = [int(x) for x in fingerprint_str.split(',')]
            return fingerprint
        return extract_waveform_fallback()
    except Exception as e:
        print(f"   Chromaprint 提取失败: {e}")
        return extract_waveform_fallback()

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
    
    def __init__(
        self,
        target_video: str,
        source_videos: List[str],
        cache_dir: str = None,
        frame_index_cache_dir: str = None,
    ):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = Path(cache_dir) if cache_dir else self.temp_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.frame_index_cache_dir = (
            Path(frame_index_cache_dir) if frame_index_cache_dir else self.cache_dir
        )
        self.frame_index_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置
        self.match_threshold = 0.95
        self.segment_duration = 5.0  # 降低分段时长提高精度
        self.frame_index_sample_interval = 1.0 / 3.0
        self.phash_preprocess_version = 2
        self.phash_match_frame_count = 20
        self.phash_match_max_distance = 48
        self.phash_match_min_window_score = 0.48
        self.phash_match_min_frame_score = 0.30
        self.phash_match_min_strong_ratio = 0.0
        self.phash_match_candidate_margin = 0.02
        self.phash_match_dedupe_sec = 0.6
        # 当前策略：禁止回退到目标素材，避免输出混入原视频内容。
        self.enable_target_video_fallback = False
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
        # 音频守卫自动纠偏：检测到稳定偏移时，自动微调段起点并复核。
        self.audio_guard_auto_shift_enabled = True
        self.audio_guard_auto_shift_min_abs_sec = 0.45
        self.audio_guard_auto_shift_min_gain = 0.045
        self.audio_guard_auto_shift_max_verify_drop = 0.02
        # 分段提取精确寻址（含音轨段）：减少关键帧回退导致的词尾重复/音画错位
        self.audio_segment_accurate_seek = True
        # 当命中源片尾导致分段素材天然不足时，自动补齐静音与末帧，避免局部掉音/黑帧
        self.segment_shortfall_pad = True
        # 分段短缺硬阈值：短缺过大时不再静音补齐，改为让提取策略重试。
        self.segment_shortfall_pad_hard_fail_sec = 0.90
        self.segment_shortfall_pad_hard_fail_ratio = 0.22
        # 跨源短缺拼桥：当当前段命中源尾且下一段从新源片头接入时，
        # 允许把下一源的开头借到当前段尾部，避免“前面重复、后面缺字”。
        self.cross_source_shortfall_bridge_enabled = True
        self.cross_source_shortfall_bridge_min_sec = 0.12
        self.cross_source_shortfall_bridge_max_sec = 1.20
        self.cross_source_shortfall_bridge_next_head_window = 0.95
        self.cross_source_shortfall_bridge_base_search_cap = 0.40
        self.cross_source_shortfall_bridge_extra_trim_cap = 0.90
        # 轻微音频前移补偿（秒）：用于抵消分段编码/拼接带来的主观音频滞后感。
        self.audio_sync_advance_sec = 0.05
        # 中间分段默认沿用 AAC，优先稳定可听性；PCM 仅按需手动开启。
        self.segment_intermediate_pcm_audio = False
        # 桥接恢复动态守卫：避免恢复后出现“音频在走、画面像不动”的观感问题
        self.bridge_motion_guard_enabled = True
        self.bridge_motion_min_target_motion = 0.010
        self.bridge_motion_min_ratio = 0.60
        self.bridge_motion_samples = 7
        # 桥接恢复起始保护：避免开头几段被“桥接”后出现主观卡顿观感
        self.bridge_recover_min_target_start = 12.0
        # 源片尾安全边距：避免命中分片尾部黑场/尾音异常（非目标尾段）
        self.source_tail_safety_enabled = True
        self.source_tail_safety_margin = 0.6
        self.source_tail_safety_target_tail_ignore_sec = 10.0
        self.source_tail_safety_switch_min_gain = 0.02
        # 近尾软切换：当剩余尾量极小且候选不显著更差时，允许切到非尾段，降低“尾字被切/单帧”风险。
        self.source_tail_safety_soft_force_margin = 0.5
        self.source_tail_safety_soft_force_max_drop = 0.03
        # 跨源边界头部微调：处理“前段到片尾 + 后段从片头起”时的轻微重复词/接缝感
        self.cross_source_head_nudge_enabled = True
        self.cross_source_head_nudge_prev_tail_window = 1.5
        self.cross_source_head_nudge_curr_head_window = 1.2
        self.cross_source_head_nudge_max_offset = 1.0
        self.cross_source_head_nudge_score_bias = 0.004
        self.cross_source_head_nudge_max_verify_drop = 0.015
        # 跨源片头保护：限制“向前挪头”过大，避免切掉句头/词尾。
        self.cross_source_head_nudge_forward_cap = 0.35
        self.cross_source_head_nudge_forward_gain_trigger = 0.06
        # 跨源片头回拉：允许将片头段轻微向前回拉到更接近句首，降低“上一句被吃掉”的风险。
        self.cross_source_head_nudge_allow_backward = True
        self.cross_source_head_nudge_backward_cap = 0.8
        self.cross_source_head_nudge_backward_gain_trigger = 0.015
        self.cross_source_head_nudge_boundary_audio_weight = 0.06
        # 禁兜底模式下的同源尾段重叠回推修复：
        # 当当前段已贴到源尾无法后移时，把重叠量向前分摊，避免词尾重复/回放感。
        self.no_target_backprop_overlap_fix = True
        self.no_target_backprop_max_shift = 3.0
        self.no_target_backprop_min_quality = 0.78
        self.no_target_backprop_neg_trigger_floor = 0.04
        # 禁兜底同源严重重叠 run 前推修复：
        # 对 1~3 秒级负重叠做整段链前推，避免后段重复/倒放观感。
        self.no_target_severe_overlap_run_fix = True
        self.no_target_severe_overlap_trigger = 0.85
        self.no_target_severe_overlap_safe_tol = 0.02
        self.no_target_severe_overlap_force_combined_max = 0.92
        # 禁兜底尾段容忍短缺：尾部若仍存在重复边界，优先裁掉前段尾巴，宁可整片略短也不回放补时长。
        self.no_target_tail_shortfall_tolerance_sec = 1.0
        # 禁兜底尾部缺段容忍：若未恢复缺段全部集中在尾部，则允许直接输出略短成片。
        self.no_target_missing_tail_tolerance_sec = 4.5
        # 同源 run 尾部容量不足时，允许更大幅度的均摊，避免把重叠集中到尾部个别边界。
        self.no_target_run_overflow_max_per_boundary = 0.32
        self.no_target_run_overflow_tiny_forward_freeze = 0.03
        self.no_target_boundary_rematch_enabled = True
        self.no_target_boundary_rematch_max_attempts = 4
        # 边界硬约束 + 边界音频探针（用于自动发现重读/丢词/局部静音风险）
        self.boundary_hard_max_negative_overlap = 0.01
        self.boundary_hard_max_positive_gap = 0.06
        # 边界硬约束单次最大位移：防止一次性大幅硬夹紧造成后续内容缺失/跨段错位。
        self.boundary_hard_max_shift_sec = 1.2
        self.boundary_audio_probe_window_sec = 0.45
        self.boundary_audio_expected_gain_trigger = 0.08
        self.boundary_audio_repair_max_offset = 0.5
        self.boundary_audio_repair_max_passes = 2
        # 源目录下的 mp4 视频一律视为同级原视频。
        # 保留这些字段仅为兼容旧质量报告/旧分支逻辑，不再按文件名做主次分级。
        self.secondary_source_name_rescue_only = False
        self.secondary_source_numeric_stem_max_len = 0

        # 运行期统计（用于质量报告）
        self.match_elapsed_sec = 0.0
        self.timeline_guard_elapsed_sec = 0.0
        self.total_elapsed_sec = 0.0
        self.guard_stats: Dict[str, int] = {}
        self.last_render_metrics: Dict[str, object] = {}
        self.last_boundary_repair_stats: Dict[str, int] = {}

        # pHash 帧索引
        self.frame_index = {}  # {video_path: [(time, phash), ...]}
        self.phash_cache_max_items = 12000
        self.phash_cache: "OrderedDict[Tuple[str, float, int], imagehash.ImageHash]" = OrderedDict()
        self.phash_cache_lock = threading.Lock()

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
        self.primary_source_videos: List[Path] = []
        self.secondary_source_videos: List[Path] = []
        self.primary_source_video_keys: Set[str] = set()
        self.secondary_source_video_keys: Set[str] = set()

        # v7.3: 帧特征缓存（减少重复 imread/resize/cvtColor）
        self.frame_feature_cache_max_items = 1500
        self.frame_feature_cache: "OrderedDict[str, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self.frame_feature_lock = threading.Lock()
        self._ffmpeg_encoder_cache: Dict[str, bool] = {}
        self._rebuild_source_priority_sets()

    def _source_key(self, video_path: Path) -> str:
        return str(Path(video_path).resolve())

    def _is_secondary_source(self, video_path: Optional[Path]) -> bool:
        if video_path is None:
            return False
        path = Path(video_path)
        key = self._source_key(path)
        return key in self.secondary_source_video_keys

    def _is_secondary_source_filename(self, video_path: Path) -> bool:
        return False

    def _rebuild_source_priority_sets(self) -> None:
        self.primary_source_videos = list(self.source_videos)
        self.secondary_source_videos = []
        self.primary_source_video_keys = {self._source_key(p) for p in self.primary_source_videos}
        self.secondary_source_video_keys = {self._source_key(p) for p in self.secondary_source_videos}

    def _candidate_source_videos(self, include_secondary: bool = False) -> List[Path]:
        if include_secondary or not self.secondary_source_videos:
            return list(self.source_videos)
        return list(self.primary_source_videos)

    def _no_target_boundary_thresholds(self, curr: dict) -> Tuple[float, float]:
        duration = max(0.0, float(curr.get("duration", 0.0)))
        q = curr.get("quality", {}) or {}
        recover_mode = str(q.get("recover_mode", "") or "")
        neg_trigger = max(0.12, duration * 0.02)
        pos_trigger = max(0.20, duration * 0.04)
        if recover_mode == "neighbors_last_resort_no_target":
            neg_trigger = min(neg_trigger, max(0.06, duration * 0.012))
            pos_trigger = min(pos_trigger, max(0.12, duration * 0.024))
        return float(neg_trigger), float(pos_trigger)

    def _no_target_boundary_unresolved_meta(self, prev: dict, curr: dict) -> Tuple[bool, float, Dict[str, float]]:
        prev_end = float(prev["start"]) + float(prev["duration"])
        gap = float(curr["start"]) - prev_end
        neg_trigger, pos_trigger = self._no_target_boundary_thresholds(curr)
        q = curr.get("quality", {}) or {}
        repeat_risk_unresolved = bool(q.get("boundary_audio_repeat_risk_unresolved_after", False))
        recover_mode = str(q.get("recover_mode", "") or "")
        recover_boundary_excess = max(0.0, float(q.get("recover_boundary_excess", 0.0) or 0.0))
        next_penalty_after = max(0.0, float(q.get("boundary_audio_next_penalty_after", 0.0) or 0.0))
        recover_last_resort_unresolved = bool(
            recover_mode == "neighbors_last_resort_no_target"
            and (
                recover_boundary_excess > max(0.045, float(curr["duration"]) * 0.008)
                or next_penalty_after > max(0.04, float(curr["duration"]) * 0.008)
            )
        )
        unresolved = bool(
            (gap < -neg_trigger)
            or (gap > pos_trigger)
            or repeat_risk_unresolved
            or recover_last_resort_unresolved
        )
        meta = {
            "neg_trigger": float(neg_trigger),
            "pos_trigger": float(pos_trigger),
            "recover_boundary_excess": float(recover_boundary_excess),
            "boundary_audio_next_penalty_after": float(next_penalty_after),
            "repeat_risk_unresolved": 1.0 if repeat_risk_unresolved else 0.0,
            "recover_last_resort_unresolved": 1.0 if recover_last_resort_unresolved else 0.0,
        }
        return unresolved, float(gap), meta

    def _is_tail_sensitive_last_resort_segment(self, seg: dict) -> bool:
        if not seg:
            return False
        q = seg.get("quality", {}) or {}
        recover_mode = str(q.get("recover_mode", "") or "")
        if recover_mode not in {"neighbors_last_resort_no_target", "neighbors_forced_no_verify"}:
            return False
        if self.target_duration <= 0.0:
            return True
        tail_window = max(float(getattr(self, "tail_guard_seconds", 18.0)), 25.0)
        return float(seg.get("target_start", 0.0)) >= max(0.0, float(self.target_duration - tail_window))

    def _is_tail_shortfall_relax_boundary(self, boundary_index: int, curr: dict, total_segments: int) -> bool:
        if self.enable_target_video_fallback:
            return False
        tol = max(0.0, float(getattr(self, "no_target_tail_shortfall_tolerance_sec", 0.0)))
        if tol <= 1e-6:
            return False
        if boundary_index >= max(1, int(total_segments) - 3):
            return True
        if self.target_duration <= 0.0:
            return False
        tail_window = max(12.0, float(tol) + 8.0)
        return float(curr.get("target_start", 0.0)) >= max(0.0, float(self.target_duration - tail_window))

    def _resolve_missing_tail_shortfall_no_target(
        self,
        missing_indices: List[int],
        tasks: List[SegmentTask],
        extra_shortfall_sec: float = 0.0,
    ) -> Tuple[bool, float]:
        if self.enable_target_video_fallback:
            return False, 0.0
        tol = max(0.0, float(getattr(self, "no_target_missing_tail_tolerance_sec", 0.0)))
        if tol <= 1e-6 or (not tasks):
            return False, 0.0

        if missing_indices:
            expected_tail = list(range(missing_indices[0], len(tasks)))
            if missing_indices != expected_tail:
                return False, 0.0

        missing_duration = float(sum(max(0.0, float(tasks[idx].duration)) for idx in missing_indices))
        missing_duration += max(0.0, float(extra_shortfall_sec))
        if missing_duration <= 1e-6:
            return False, 0.0
        if missing_duration - tol > 1e-6:
            return False, missing_duration
        return True, missing_duration

    def recover_partial_tail_segments_no_target(
        self,
        missing_indices: List[int],
        tasks: List[SegmentTask],
        confirmed_by_index: Dict[int, dict],
    ) -> Tuple[int, float]:
        """
        尾部短段恢复（禁兜底）：
        - 当缺失段全部集中在尾部时，优先尝试把“前一已确认段所属源视频”剩余尾巴补成一个短段；
        - 不强行要求补满整段，只要画面/音频对得上，就允许以短段形式恢复；
        - 剩余补不满的部分继续走尾部短缺容忍。
        """
        if self.enable_target_video_fallback:
            return 0, 0.0
        if not missing_indices:
            return 0, 0.0
        expected_tail = list(range(missing_indices[0], len(tasks)))
        if missing_indices != expected_tail:
            return 0, 0.0

        recovered = 0
        shortfall_sec = 0.0
        pending = list(missing_indices)

        while pending:
            idx = int(pending[0])
            task = tasks[idx]
            prev_seg = confirmed_by_index.get(idx - 1)
            if prev_seg is None:
                break

            source = Path(prev_seg.get("source", ""))
            if (not source) or source == self.target_video:
                break

            source_duration = self.get_video_duration(source)
            if source_duration <= 0.0:
                break

            prev_end = float(prev_seg["start"]) + float(prev_seg["duration"])
            available = max(0.0, float(source_duration - prev_end))
            min_piece = max(0.5, min(1.0, float(task.duration) * 0.25))
            if available + 1e-6 < min_piece:
                break

            partial_duration = min(float(task.duration), float(available))
            if partial_duration <= 1e-6:
                break

            source_start = max(0.0, float(prev_end))
            verify_passed, verify_avg = self.quick_verify(
                source=source,
                source_start=source_start,
                target_start=float(task.target_start),
                duration=float(partial_duration),
            )
            if not verify_passed:
                break

            audio_passed, audio_meta = self.quick_verify_audio(
                source=source,
                source_start=source_start,
                target_start=float(task.target_start),
                duration=float(partial_duration),
                combined_score=float(verify_avg),
            )
            if not audio_passed:
                break

            confirmed_by_index[idx] = {
                "index": int(task.index),
                "source": source,
                "start": float(source_start),
                "duration": float(partial_duration),
                "target_start": float(task.target_start),
                "quality": {
                    "combined": float(verify_avg),
                    "recovered_from_neighbors": True,
                    "recover_mode": "partial_source_tail_no_target",
                    "recover_partial_tail": True,
                    "recover_available_sec": float(available),
                    "recover_shortfall_sec": float(max(0.0, float(task.duration) - float(partial_duration))),
                    "audio_guard": audio_meta,
                },
            }
            recovered += 1
            shortfall_sec += float(max(0.0, float(task.duration) - float(partial_duration)))
            pending.pop(0)

        return recovered, shortfall_sec

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
            # 没有任何音轨流时直接判 0，不能回退到 format duration。
            duration = 0.0
            with self.meta_cache_lock:
                self.audio_duration_cache[key] = float(duration)
            return float(duration)
        raw = value[0].strip() if value else ""
        if raw and raw.upper() != "N/A":
            try:
                duration = max(0.0, float(raw))
            except Exception:
                duration = 0.0
        else:
            duration = 0.0

        # 某些组合（如 mkv + pcm）不会填充 stream.duration，回退到 format.duration。
        if duration <= 0.0:
            fmt_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            fmt_result = subprocess.run(fmt_cmd, capture_output=True, text=True)
            try:
                duration = max(0.0, float((fmt_result.stdout or "").strip()))
            except Exception:
                duration = 0.0

        with self.meta_cache_lock:
            self.audio_duration_cache[key] = float(duration)
        return float(duration)

    def _invalidate_media_duration_cache(self, video_path: Path) -> None:
        """文件被覆盖后，清理本地 duration 缓存，避免读取到旧值。"""
        key = str(Path(video_path).resolve())
        with self.meta_cache_lock:
            self.video_duration_cache.pop(key, None)
            self.audio_duration_cache.pop(key, None)

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
        - 先做半秒级偏移探测；对齐点很差时再扩展到配置偏移探测
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

        # 可疑段先做半秒级探测；对齐极差再扩展到配置偏移，兼顾性能和错位拦截。
        probe_shifts: List[float] = []
        if need_check:
            probe_shifts.extend([-0.5, 0.5])
        if aligned_sim < float(self.audio_guard_min_similarity):
            probe_shifts.extend([float(x) for x in self.audio_guard_shift_candidates])

        seen_probe_shifts: Set[float] = set()
        for raw_shift in probe_shifts:
            shift = float(raw_shift)
            if abs(shift) <= 1e-6:
                continue
            shift_key = round(shift, 3)
            if shift_key in seen_probe_shifts:
                continue
            seen_probe_shifts.add(shift_key)
            shifted_start = src_clip_start + shift
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
            "shift_gain": float(best_shift_sim - aligned_sim),
        })

        fail_reason = ""
        shift_gain = float(best_shift_sim - aligned_sim)
        long_shift_margin = max(0.10, float(self.audio_guard_shift_margin) * 0.70)
        half_shift_margin = max(0.08, float(self.audio_guard_shift_margin) * 0.65)
        shift_bias_aligned_cap = max(float(self.audio_guard_min_similarity) + 0.28, 0.78)
        half_shift_bias_aligned_cap = max(float(self.audio_guard_min_similarity) + 0.36, 0.82)
        if aligned_sim < float(self.audio_guard_hard_floor):
            fail_reason = "audio_guard_hard_floor"
        elif (
            shift_gain > 1e-6
            and abs(best_shift) >= 1.0
            and shift_gain >= long_shift_margin
            and aligned_sim <= shift_bias_aligned_cap
        ):
            fail_reason = "audio_guard_shift_bias"
        elif (
            shift_gain > 1e-6
            and abs(best_shift) >= 0.45
            and abs(best_shift) < 1.0
            and shift_gain >= half_shift_margin
            and aligned_sim <= half_shift_bias_aligned_cap
        ):
            fail_reason = "audio_guard_shift_bias_halfsec"
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

    def try_audio_guard_shift_realign(
        self,
        source: Path,
        source_start: float,
        target_start: float,
        duration: float,
        combined_score: float,
        audio_meta: Optional[Dict[str, object]],
    ) -> Tuple[float, Dict[str, object], Optional[Dict[str, object]]]:
        """
        音频守卫自动纠偏（轻量）：
        - 仅当音频守卫已检测并出现稳定偏移收益时触发
        - 应用偏移后再次进行画面核验 + 音频守卫，双通过才落地
        """
        meta: Dict[str, object] = {"applied": False, "checked": False}
        if source is None or source == self.target_video:
            meta["reason"] = "target_segment"
            return float(source_start), meta, None
        if not bool(getattr(self, "audio_guard_auto_shift_enabled", True)):
            meta["reason"] = "disabled"
            return float(source_start), meta, None
        if not isinstance(audio_meta, dict) or not bool(audio_meta.get("checked", False)):
            meta["reason"] = "audio_meta_not_checked"
            return float(source_start), meta, None

        best_shift = float(audio_meta.get("best_shift_sec", 0.0) or 0.0)
        shift_gain = float(audio_meta.get("shift_gain", 0.0) or 0.0)
        min_abs = max(0.0, float(getattr(self, "audio_guard_auto_shift_min_abs_sec", 0.45)))
        min_gain = max(0.0, float(getattr(self, "audio_guard_auto_shift_min_gain", 0.045)))
        near_source_head = bool(float(source_start) <= max(0.72, float(duration) * 0.16))
        if near_source_head and float(best_shift) < -0.45:
            min_gain = min(float(min_gain), 0.02)
        if abs(best_shift) < min_abs:
            meta["reason"] = "shift_too_small"
            meta["best_shift_sec"] = float(best_shift)
            return float(source_start), meta, None
        if shift_gain < min_gain:
            meta["reason"] = "gain_too_small"
            meta["best_shift_sec"] = float(best_shift)
            meta["shift_gain"] = float(shift_gain)
            return float(source_start), meta, None

        src_duration = self.get_video_duration(source)
        if src_duration <= 0.0:
            meta["reason"] = "invalid_source_duration"
            return float(source_start), meta, None
        max_start = max(0.0, float(src_duration - duration))
        cand_start = max(0.0, min(float(source_start) + float(best_shift), max_start))
        if near_source_head and float(best_shift) > 0.45 and float(cand_start) > float(source_start) + 1e-6:
            meta["checked"] = True
            meta["reason"] = "head_anchored_positive_shift_blocked"
            meta["best_shift_sec"] = float(best_shift)
            meta["shift_gain"] = float(shift_gain)
            meta["candidate_start"] = float(cand_start)
            return float(source_start), meta, None
        if abs(cand_start - float(source_start)) <= 1e-6:
            meta["reason"] = "clamped_no_change"
            return float(source_start), meta, None

        meta["checked"] = True
        base_passed, base_avg = self.quick_verify(
            source=source,
            source_start=float(source_start),
            target_start=float(target_start),
            duration=float(duration),
        )
        cand_passed, cand_avg = self.quick_verify(
            source=source,
            source_start=float(cand_start),
            target_start=float(target_start),
            duration=float(duration),
        )
        if not cand_passed:
            meta["reason"] = "candidate_visual_failed"
            meta["candidate_start"] = float(cand_start)
            meta["candidate_verify_avg"] = float(cand_avg)
            return float(source_start), meta, None
        max_drop = max(0.0, float(getattr(self, "audio_guard_auto_shift_max_verify_drop", 0.02)))
        if base_passed and (float(cand_avg) + max_drop) < float(base_avg):
            meta["reason"] = "candidate_visual_drop_too_large"
            meta["base_verify_avg"] = float(base_avg)
            meta["candidate_verify_avg"] = float(cand_avg)
            return float(source_start), meta, None

        cand_audio_passed, cand_audio_meta = self.quick_verify_audio(
            source=source,
            source_start=float(cand_start),
            target_start=float(target_start),
            duration=float(duration),
            combined_score=max(float(combined_score), float(cand_avg)),
        )
        if not cand_audio_passed:
            meta["reason"] = "candidate_audio_guard_failed"
            meta["candidate_audio_reason"] = str((cand_audio_meta or {}).get("reason", ""))
            return float(source_start), meta, cand_audio_meta

        base_aligned = float(audio_meta.get("aligned_similarity", 0.0) or 0.0)
        cand_aligned = float((cand_audio_meta or {}).get("aligned_similarity", 0.0) or 0.0)
        if bool((cand_audio_meta or {}).get("checked", False)) and bool(audio_meta.get("checked", False)):
            if cand_aligned + 0.03 < base_aligned:
                meta["reason"] = "candidate_aligned_similarity_regressed"
                meta["base_aligned_similarity"] = float(base_aligned)
                meta["candidate_aligned_similarity"] = float(cand_aligned)
                return float(source_start), meta, cand_audio_meta

        meta.update(
            {
                "applied": True,
                "reason": "audio_guard_shift_realign_applied",
                "from_start": float(source_start),
                "to_start": float(cand_start),
                "applied_shift_sec": float(cand_start - float(source_start)),
                "best_shift_sec": float(best_shift),
                "shift_gain": float(shift_gain),
                "base_verify_avg": float(base_avg),
                "candidate_verify_avg": float(cand_avg),
                "base_aligned_similarity": float(base_aligned),
                "candidate_aligned_similarity": float(cand_aligned),
            }
        )
        return float(cand_start), meta, cand_audio_meta

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
        processed = self.preprocess_frame_for_phash(img)
        return imagehash.phash(processed, hash_size=8)

    def _phash_cache_key(self, video_path: Path, time_sec: float) -> Tuple[str, float, int]:
        return (
            str(Path(video_path).resolve()),
            self._normalize_frame_time(time_sec),
            int(getattr(self, "phash_preprocess_version", 1)),
        )

    def _remember_cached_phash(self, key: Tuple[str, float, int], phash_value: imagehash.ImageHash) -> None:
        with self.phash_cache_lock:
            self.phash_cache[key] = phash_value
            self.phash_cache.move_to_end(key)
            while len(self.phash_cache) > self.phash_cache_max_items:
                self.phash_cache.popitem(last=False)

    def _lookup_cached_phash(self, key: Tuple[str, float, int]) -> Optional[imagehash.ImageHash]:
        with self.phash_cache_lock:
            cached = self.phash_cache.get(key)
            if cached is not None:
                self.phash_cache.move_to_end(key)
                return cached
        return None

    def get_frame_phash(self, video_path: Path, time_sec: float) -> Optional[imagehash.ImageHash]:
        key = self._phash_cache_key(video_path, time_sec)
        cached = self._lookup_cached_phash(key)
        if cached is not None:
            return cached
        img = self.extract_frame_to_pil(video_path, time_sec)
        if img is None:
            return None
        phash_value = self.compute_phash(img)
        self._remember_cached_phash(key, phash_value)
        return phash_value

    def _target_phash_offsets(self, duration: float) -> List[float]:
        interval = max(0.1, float(getattr(self, "frame_index_sample_interval", 0.2)))
        frame_count = max(1, int(getattr(self, "phash_match_frame_count", 20)))
        max_offset = max(0.0, float(duration) - 0.1)
        offsets: List[float] = []
        cursor = 0.0
        while len(offsets) < frame_count and cursor <= max_offset + 1e-6:
            offsets.append(round(float(cursor), 3))
            cursor += interval
        if not offsets:
            offsets.append(0.0)
        return offsets

    def _score_phash_window(
        self,
        target_hashes: List[imagehash.ImageHash],
        target_offsets: List[float],
        source_frames: List[Tuple[float, imagehash.ImageHash]],
        start_idx: int,
    ) -> Optional[Dict[str, float]]:
        frame_count = len(target_hashes)
        if frame_count <= 0 or (start_idx + frame_count) > len(source_frames):
            return None

        start_time = float(source_frames[start_idx][0])
        interval = max(0.1, float(getattr(self, "frame_index_sample_interval", 0.2)))
        jitter_tol = max(0.12, interval * 0.65)
        max_distance = max(1, int(getattr(self, "phash_match_max_distance", 24)))
        strong_distance = 28

        sims: List[float] = []
        strong_hits = 0
        max_jitter = 0.0

        for rel_idx, target_hash in enumerate(target_hashes):
            source_time, source_hash = source_frames[start_idx + rel_idx]
            expected_offset = float(target_offsets[rel_idx])
            actual_offset = float(source_time) - start_time
            jitter = abs(actual_offset - expected_offset)
            max_jitter = max(max_jitter, jitter)
            if jitter > jitter_tol:
                return None

            distance = int(target_hash - source_hash)
            if distance > max_distance:
                return None
            if distance <= strong_distance:
                strong_hits += 1
            sims.append(1.0 - (float(distance) / 64.0))

        if not sims:
            return None

        avg_sim = float(np.mean(sims))
        min_sim = float(np.min(sims))
        strong_ratio = float(strong_hits) / float(len(sims))
        prefix_count = min(6, len(sims))
        suffix_count = min(6, len(sims))
        prefix_avg = float(np.mean(sims[:prefix_count]))
        suffix_avg = float(np.mean(sims[-suffix_count:]))
        jitter_penalty = float(max_jitter / max(interval, 1e-6))
        score = (
            avg_sim * 0.78
            + prefix_avg * 0.14
            + suffix_avg * 0.08
            - jitter_penalty * 0.03
        )

        min_window_score = float(getattr(self, "phash_match_min_window_score", 0.72))
        min_frame_score = float(getattr(self, "phash_match_min_frame_score", 0.56))
        min_strong_ratio = float(getattr(self, "phash_match_min_strong_ratio", 0.55))
        if avg_sim < min_window_score or min_sim < min_frame_score or strong_ratio < min_strong_ratio:
            return None

        return {
            "score": float(score),
            "avg_similarity": float(avg_sim),
            "min_similarity": float(min_sim),
            "strong_ratio": float(strong_ratio),
            "start_time": float(start_time),
        }

    def _find_primary_activity_span(self, scores: np.ndarray, min_len: int) -> Optional[Tuple[int, int]]:
        """从一维活动分数中找主要内容区间。"""
        values = np.asarray(scores, dtype=np.float32)
        total = int(values.shape[0])
        if total <= 0:
            return None

        p55 = float(np.percentile(values, 55))
        p90 = float(np.percentile(values, 90))
        mean_v = float(np.mean(values))
        threshold = max(p55 + (p90 - p55) * 0.35, mean_v * 0.90, 0.8)

        mask = values >= threshold
        if not np.any(mask):
            return None

        segments: List[Tuple[int, int]] = []
        start = -1
        for idx, flag in enumerate(mask):
            if flag and start < 0:
                start = idx
            elif (not flag) and start >= 0:
                segments.append((start, idx))
                start = -1
        if start >= 0:
            segments.append((start, total))

        if not segments:
            return None

        gap_allow = max(1, int(round(total * 0.02)))
        merged: List[Tuple[int, int]] = []
        for seg_start, seg_end in segments:
            if merged and seg_start - merged[-1][1] <= gap_allow:
                merged[-1] = (merged[-1][0], seg_end)
            else:
                merged.append((seg_start, seg_end))

        center = total * 0.5
        best = None
        best_score = -1.0
        longest = None
        longest_len = -1
        for seg_start, seg_end in merged:
            seg_len = seg_end - seg_start
            if seg_len > longest_len:
                longest_len = seg_len
                longest = (seg_start, seg_end)
            if seg_len < min_len:
                continue
            seg_energy = float(np.sum(values[seg_start:seg_end]))
            seg_center = (seg_start + seg_end) * 0.5
            center_penalty = abs(seg_center - center) / max(1.0, center)
            seg_score = seg_energy * (1.0 - 0.35 * center_penalty)
            if seg_score > best_score:
                best_score = seg_score
                best = (seg_start, seg_end)

        if best is None:
            if longest is None or longest_len < max(12, int(min_len * 0.70)):
                return None
            best = longest

        margin = max(1, int(round(total * 0.02)))
        left = max(0, best[0] - margin)
        right = min(total, best[1] + margin)
        if right - left < min_len:
            need = min_len - (right - left)
            left = max(0, left - need // 2)
            right = min(total, right + need - need // 2)
        return left, right

    def preprocess_frame_for_phash(self, img: Image.Image) -> Image.Image:
        """
        对帧做内容区裁剪，降低黑边/角标/底部文案对 pHash 的干扰。
        """
        try:
            rgb = img.convert("RGB")
            arr = np.array(rgb)
            if arr.ndim != 3 or arr.shape[0] < 32 or arr.shape[1] < 32:
                return rgb

            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge = cv2.magnitude(grad_x, grad_y)

            row_scores = edge.mean(axis=1) + 0.20 * gray.std(axis=1).astype(np.float32)
            col_scores = edge.mean(axis=0) + 0.20 * gray.std(axis=0).astype(np.float32)

            h, w = gray.shape
            row_span = self._find_primary_activity_span(row_scores, min_len=max(24, int(round(h * 0.35))))
            col_span = self._find_primary_activity_span(col_scores, min_len=max(24, int(round(w * 0.35))))

            top, bottom = (row_span if row_span is not None else (0, h))
            left, right = (col_span if col_span is not None else (0, w))
            if bottom - top <= 0 or right - left <= 0:
                return rgb

            probe_h = max(8, int(round(h * 0.05)))
            probe_w = max(8, int(round(w * 0.05)))
            row_core_start = int(round(h * 0.35))
            row_core_end = int(round(h * 0.65))
            col_core_start = int(round(w * 0.35))
            col_core_end = int(round(w * 0.65))

            row_core_slice = row_scores[row_core_start:row_core_end]
            col_core_slice = col_scores[col_core_start:col_core_end]
            row_core = float(np.mean(row_core_slice)) if row_core_slice.size >= 8 else float(np.mean(row_scores))
            col_core = float(np.mean(col_core_slice)) if col_core_slice.size >= 8 else float(np.mean(col_scores))
            row_core = max(row_core, 1e-6)
            col_core = max(col_core, 1e-6)

            row_top_edge = float(np.mean(row_scores[:probe_h]))
            row_bottom_edge = float(np.mean(row_scores[-probe_h:]))
            col_left_edge = float(np.mean(col_scores[:probe_w]))
            col_right_edge = float(np.mean(col_scores[-probe_w:]))

            # 仅当边缘活动显著低于画面主体时，才允许对应方向裁剪，避免误裁主体。
            if top > 0 and row_top_edge > row_core * 0.68:
                top = 0
            if bottom < h and row_bottom_edge > row_core * 0.68:
                bottom = h

            row_trim_ratio = (top + (h - bottom)) / float(max(1, h))
            if row_trim_ratio >= 0.18:
                # 已经识别出明显上下黑边时，不再做左右裁剪，避免中心人像被误裁。
                left, right = 0, w
            else:
                if left > 0 and col_left_edge > col_core * 0.62:
                    left = 0
                if right < w and col_right_edge > col_core * 0.62:
                    right = w

            min_h = max(24, int(round(h * 0.30)))
            min_w = max(24, int(round(w * 0.30)))
            if bottom - top < min_h:
                top, bottom = 0, h
            if right - left < min_w:
                left, right = 0, w

            if (bottom - top) >= int(h * 0.98) and (right - left) >= int(w * 0.98):
                return rgb

            cropped = arr[top:bottom, left:right]
            if cropped.size == 0:
                return rgb
            return Image.fromarray(cropped)
        except Exception:
            return img.convert("RGB")

    def build_frame_index(self, sample_interval: float = 1.0):
        """预建帧索引：提取所有源视频的帧 pHash"""
        interval = max(0.1, float(sample_interval))
        interval_tag = f"{interval:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        preprocess_tag = f"ppv{int(getattr(self, 'phash_preprocess_version', 1))}"
        source_sig = self._frame_index_source_signature()
        index_file = self.frame_index_cache_dir / (
            f"frame_index_v7_src{source_sig}_si{interval_tag}_{preprocess_tag}.pkl"
        )
        lock_file = index_file.with_suffix(index_file.suffix + ".lock")
        legacy_candidates = [
            self.frame_index_cache_dir / f"frame_index_v6_si{interval_tag}_{preprocess_tag}.pkl",
            self.frame_index_cache_dir / "frame_index_v6.pkl",
        ]
        if self.cache_dir != self.frame_index_cache_dir:
            legacy_candidates.extend(
                [
                    self.cache_dir / f"frame_index_v6_si{interval_tag}_{preprocess_tag}.pkl",
                    self.cache_dir / "frame_index_v6.pkl",
                ]
            )

        if index_file.exists():
            print(f"\n📂 加载已有帧索引: {index_file}")
            with open(index_file, 'rb') as f:
                self.frame_index = pickle.load(f)
            total_frames = sum(len(v) for v in self.frame_index.values())
            print(f"   ✅ 已索引 {len(self.frame_index)} 个视频，共 {total_frames} 帧")
            return
        for legacy_index_file in legacy_candidates:
            if not legacy_index_file.exists():
                continue
            if (
                legacy_index_file.name == "frame_index_v6.pkl"
                and (
                    abs(interval - 1.0) >= 1e-9
                    or int(getattr(self, "phash_preprocess_version", 1)) > 1
                )
            ):
                continue
            suffix = " (legacy)" if legacy_index_file.name == "frame_index_v6.pkl" else ""
            print(f"\n📂 加载已有帧索引: {legacy_index_file}{suffix}")
            with open(legacy_index_file, 'rb') as f:
                self.frame_index = pickle.load(f)
            total_frames = sum(len(v) for v in self.frame_index.values())
            print(f"   ✅ 已索引 {len(self.frame_index)} 个视频，共 {total_frames} 帧")
            if legacy_index_file != index_file:
                self._promote_frame_index_cache(index_file)
            return

        lock_fd = None
        wait_logged = False
        while True:
            try:
                lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(lock_fd, f"{os.getpid()}\n".encode("utf-8"))
                break
            except FileExistsError:
                if index_file.exists():
                    print(f"\n📂 加载已有帧索引: {index_file}")
                    with open(index_file, 'rb') as f:
                        self.frame_index = pickle.load(f)
                    total_frames = sum(len(v) for v in self.frame_index.values())
                    print(f"   ✅ 已索引 {len(self.frame_index)} 个视频，共 {total_frames} 帧")
                    return
                if not wait_logged:
                    print(f"\n⏳ 等待其他进程构建共享帧索引: {index_file.name}")
                    wait_logged = True
                import time
                time.sleep(1.0)

        print(f"\n🔨 构建帧索引 (采样间隔: {interval}s)...")
        try:
            if index_file.exists():
                print(f"\n📂 加载已有帧索引: {index_file}")
                with open(index_file, 'rb') as f:
                    self.frame_index = pickle.load(f)
                total_frames = sum(len(v) for v in self.frame_index.values())
                print(f"   ✅ 已索引 {len(self.frame_index)} 个视频，共 {total_frames} 帧")
                return

            for i, video_path in enumerate(self.source_videos):
                print(f"   [{i+1}/{len(self.source_videos)}] {video_path.name}")
                duration = self.get_video_duration(video_path)
                frames = []
                for t in np.arange(0, duration, interval):
                    img = self.extract_frame_to_pil(video_path, t)
                    if img:
                        frames.append((t, self.compute_phash(img)))
                self.frame_index[video_path] = frames
                print(f"      提取了 {len(frames)} 帧")

            tmp_index_file = index_file.with_suffix(index_file.suffix + f".tmp.{os.getpid()}")
            with open(tmp_index_file, 'wb') as f:
                pickle.dump(self.frame_index, f)
            os.replace(tmp_index_file, index_file)
            total_frames = sum(len(v) for v in self.frame_index.values())
            print(f"\n✅ 索引构建完成: {len(self.frame_index)} 个视频，共 {total_frames} 帧")
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            try:
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass

    def _frame_index_source_signature(self) -> str:
        """基于源池内容生成稳定签名，避免项目级共享索引误复用到错误源集。"""
        parts = []
        for video_path in sorted(self.source_videos, key=lambda p: str(p.resolve())):
            try:
                stat = video_path.stat()
                parts.append(
                    f"{video_path.resolve()}|{stat.st_size}|{getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))}"
                )
            except OSError:
                parts.append(f"{video_path.resolve()}|missing")
        digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
        return digest[:12]

    def _promote_frame_index_cache(self, index_file: Path) -> None:
        """将已命中的旧索引提升到项目共享路径，避免后续目标视频重复复用旧私有缓存。"""
        if index_file.exists():
            return
        try:
            index_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_index_file = index_file.with_suffix(index_file.suffix + f".tmp.{os.getpid()}")
            with open(tmp_index_file, "wb") as f:
                pickle.dump(self.frame_index, f)
            os.replace(tmp_index_file, index_file)
            print(f"   ♻️ 已迁移共享帧索引: {index_file}")
        except Exception as exc:
            print(f"   ⚠️ 共享帧索引迁移失败: {exc}")

    def find_match_by_phash(self, target_start: float, duration: float,
                            seg_index: int = 0, top_k: int = 10) -> List[Tuple[Path, float, float]]:
        """使用多帧 pHash 顺序检索候选，返回 [(source, start_time, similarity), ...]。"""
        target_offsets = self._target_phash_offsets(duration)
        target_hashes: List[imagehash.ImageHash] = []
        valid_offsets: List[float] = []
        for offset in target_offsets:
            phash_value = self.get_frame_phash(self.target_video, float(target_start) + float(offset))
            if phash_value is None:
                continue
            target_hashes.append(phash_value)
            valid_offsets.append(float(offset))

        if len(target_hashes) < max(6, min(10, len(target_offsets))):
            return []

        dedupe_sec = max(
            float(getattr(self, "phash_match_dedupe_sec", 0.6)),
            float(getattr(self, "frame_index_sample_interval", 0.2)) * 1.5,
        )

        def collect(include_secondary: bool) -> List[Tuple[Path, float, float]]:
            allowed = {self._source_key(p) for p in self._candidate_source_videos(include_secondary=include_secondary)}
            found: List[Tuple[Path, float, float, float, float]] = []
            for video_path, frames in self.frame_index.items():
                video_path = Path(video_path)
                if self._source_key(video_path) not in allowed:
                    continue
                if len(frames) < len(target_hashes):
                    continue

                best_by_bucket: Dict[int, Tuple[Path, float, float, float, float]] = {}
                for start_idx in range(0, len(frames) - len(target_hashes) + 1):
                    scored = self._score_phash_window(
                        target_hashes=target_hashes,
                        target_offsets=valid_offsets,
                        source_frames=frames,
                        start_idx=start_idx,
                    )
                    if scored is None:
                        continue
                    start_time = float(scored["start_time"])
                    bucket = int(round(start_time / max(dedupe_sec, 1e-6)))
                    item = (
                        video_path,
                        start_time,
                        float(scored["score"]),
                        float(scored["avg_similarity"]),
                        float(scored["strong_ratio"]),
                    )
                    existing = best_by_bucket.get(bucket)
                    if existing is None or item[2] > existing[2]:
                        best_by_bucket[bucket] = item
                found.extend(best_by_bucket.values())

            found.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
            return [(src, start, score) for src, start, score, _, _ in found[:top_k]]

        candidates = collect(include_secondary=False)
        if (not candidates) and self.secondary_source_videos:
            candidates = collect(include_secondary=True)
        return candidates

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
    
    def find_match_combined(self, target_start: float, duration: float, seg_index: int = 0) -> Tuple[Optional[Path], float, float, Dict[str, object]]:
        """三阶段匹配：多帧 pHash 预筛选 → (可选音频) → 画面精细定位"""
        best_source = None
        best_start = 0
        best_score = 0.0
        phash_unstable = False
        match_meta: Dict[str, object] = {
            "phash_mode": "multiframe_sequential",
            "phash_candidates": [],
            "candidate_count": 0,
            "ambiguous": False,
        }

        # 第一步：多帧 pHash 顺序预筛选候选
        phash_candidates = self.find_match_by_phash(target_start, duration, seg_index, top_k=24)
        match_meta["phash_candidates"] = [
            {
                "source": str(source),
                "start": float(start),
                "score": float(score),
            }
            for source, start, score in phash_candidates[:5]
        ]

        if phash_candidates:
            # 第二步：对 pHash 候选做精细验证（默认纯画面，可选音频）
            target_fp = extract_chromaprint(self.target_video, target_start, duration) if self.use_audio_matching else []
            target_frames = []
            check_offsets = [0, duration * 0.3, duration * 0.7]
            for offset in check_offsets:
                tf = self.get_cached_frame_path(self.target_video, target_start + offset)
                if tf and tf.exists():
                    target_frames.append((offset, tf))

            ranked_candidates: Dict[Tuple[str, float], Dict[str, object]] = {}
            for source, phash_time, phash_sim in phash_candidates:
                # 在 pHash 匹配时间点前后 ±2 秒精细搜索
                source_duration = self.get_video_duration(source)
                search_start = max(0, int(phash_time) - 2)
                search_end = min(int(source_duration - duration), int(phash_time) + 2)

                for start_sec in range(search_start, search_end + 1, 1):
                    verify_passed, verify_avg = self.quick_verify(
                        source,
                        float(start_sec),
                        float(target_start),
                        float(duration),
                    )
                    if not verify_passed:
                        continue

                    # 音频验证
                    audio_sim = 0.0
                    if self.use_audio_matching and target_fp and len(target_fp) >= 10:
                        source_fp = extract_chromaprint(source, start_sec, duration)
                        if source_fp and len(source_fp) >= 10:
                            audio_sim = compare_chromaprint(target_fp, source_fp)

                    if self.use_audio_matching:
                        # 综合评分：pHash 15% + 音频 35% + 五点画面核验 50%
                        combined_score = 0.15 * phash_sim + 0.35 * audio_sim + 0.50 * verify_avg
                    else:
                        # 默认纯画面：pHash 22% + 五点画面核验 78%
                        combined_score = 0.22 * phash_sim + 0.78 * verify_avg
                    key = (str(source), round(float(start_sec)))
                    existing = ranked_candidates.get(key)
                    candidate_item = {
                        "source": source,
                        "start": float(start_sec),
                        "score": float(combined_score),
                        "phash": float(phash_sim),
                        "visual": float(verify_avg),
                        "audio": float(audio_sim),
                    }
                    if existing is None or float(candidate_item["score"]) > float(existing["score"]):
                        ranked_candidates[key] = candidate_item

            ranked = sorted(ranked_candidates.values(), key=lambda item: float(item["score"]), reverse=True)
            match_meta["candidate_count"] = len(ranked)
            if ranked:
                best = ranked[0]
                runner_up = ranked[1] if len(ranked) > 1 else None
                best_source = Path(best["source"])
                best_start = float(best["start"])
                best_score = float(best["score"])
                match_meta["best_candidate"] = {
                    "source": str(best_source),
                    "start": float(best_start),
                    "score": float(best_score),
                    "phash": float(best.get("phash", 0.0)),
                    "visual": float(best.get("visual", 0.0)),
                    "audio": float(best.get("audio", 0.0)),
                }
                if runner_up is not None:
                    margin = float(best_score) - float(runner_up["score"])
                    match_meta["runner_up"] = {
                        "source": str(runner_up["source"]),
                        "start": float(runner_up["start"]),
                        "score": float(runner_up["score"]),
                    }
                    match_meta["best_margin"] = float(margin)
                    clear_margin = float(getattr(self, "phash_match_candidate_margin", 0.03))
                    if best_score < 0.90 and margin < clear_margin:
                        match_meta["ambiguous"] = True
                        match_meta["ambiguous_reason"] = "best_margin_too_small"
                        phash_unstable = True
                if best_score >= 0.70 and (not phash_unstable):
                    return best_source, best_start, best_score, match_meta

        # 多帧 pHash 未提供稳定候选时，补一层全源顺序视觉复核。
        if not self.use_audio_matching:
            visual_source, visual_start, visual_score = self.find_best_match_by_visual(
                target_start,
                duration,
                seg_index,
            )
            if visual_source is not None:
                accept_visual = bool(
                    (best_source is None and visual_score >= 0.80)
                    or (visual_score >= max(0.86, float(best_score) + 0.04))
                )
                match_meta["visual_fallback"] = {
                    "source": str(visual_source),
                    "start": float(visual_start),
                    "score": float(visual_score),
                    "accepted": bool(accept_visual),
                }
                if accept_visual:
                    return visual_source, float(visual_start), float(visual_score), match_meta
            match_meta["failed_reason"] = "no_stable_multiframe_phash_candidate"
            return None, 0.0, float(best_score), match_meta

        # 启用音频匹配时，回退到音频+画面搜索
        target_fp = extract_chromaprint(self.target_video, target_start, duration)
        if not target_fp or len(target_fp) < 10:
            match_meta["failed_reason"] = "no_target_audio_fp"
            return None, 0.0, float(best_score), match_meta

        audio_candidates = []

        def collect_audio_candidates(sources: List[Path]) -> None:
            for source in sources:
                source_duration = self.get_video_duration(source)
                for start_sec in range(0, int(source_duration - duration), 1):
                    source_fp = extract_chromaprint(source, start_sec, duration)
                    if not source_fp or len(source_fp) < 10:
                        continue
                    score = compare_chromaprint(target_fp, source_fp)
                    if score > 0.40:
                        audio_candidates.append((source, start_sec, score))

        collect_audio_candidates(self._candidate_source_videos(include_secondary=False))
        if (not audio_candidates) and self.secondary_source_videos:
            collect_audio_candidates(self._candidate_source_videos(include_secondary=True))

        if not audio_candidates:
            match_meta["failed_reason"] = "no_audio_candidates"
            return None, 0.0, float(best_score), match_meta

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
            match_meta["best_candidate"] = {
                "source": str(best[0]),
                "start": float(best[1]),
                "score": float(best[2]),
            }
            return best[0], best[1], best[2], match_meta

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

        if best_source is not None:
            match_meta["best_candidate"] = {
                "source": str(best_source),
                "start": float(best_start),
                "score": float(best_score),
            }
        else:
            match_meta["failed_reason"] = "audio_visual_fallback_failed"
        return best_source, best_start, best_score, match_meta

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
        
        source_passes: List[List[Path]] = [self._candidate_source_videos(include_secondary=False)]
        if self.secondary_source_videos:
            source_passes.append(list(self.secondary_source_videos))

        # 遍历所有源视频（当前不再做按文件名主次分级）
        for source_group in source_passes:
            for source in source_group:
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
            if best_source is not None and best_score >= 0.75:
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

    def estimate_segment_motion(
        self,
        video: Path,
        start: float,
        duration: float,
        sample_count: int = 7,
    ) -> Optional[float]:
        """
        估算片段动态量（0~1）：抽样帧之间的平均归一化差异。
        值越低，画面越“静止”。
        """
        if duration <= 0.3:
            return None
        count = max(3, int(sample_count))
        max_offset = max(0.0, float(duration) - 0.05)
        if max_offset <= 0:
            return None

        offsets = np.linspace(0.0, max_offset, num=count)
        last_small = None
        diffs: List[float] = []
        for off in offsets:
            t = max(0.0, float(start) + float(off))
            frame_path = self.get_cached_frame_path(video, t)
            if not frame_path or not frame_path.exists():
                continue
            gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            if last_small is not None:
                mad = np.mean(np.abs(small.astype(np.int16) - last_small.astype(np.int16))) / 255.0
                diffs.append(float(mad))
            last_small = small

        if not diffs:
            return None
        return float(np.mean(diffs))

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
        """处理单个段 - 音频+画面结合匹配"""

        def source_tail_risky(src: Path, src_start: float) -> Tuple[bool, float]:
            """
            判定候选是否命中源片尾高风险区间。
            返回 (是否高风险, 片尾剩余秒数)。
            """
            if not bool(getattr(self, "source_tail_safety_enabled", True)):
                return False, 999.0
            if src == self.target_video:
                return False, 999.0
            margin = max(0.0, float(getattr(self, "source_tail_safety_margin", 0.0)))
            if margin <= 1e-6:
                return False, 999.0
            if self.target_duration > 0:
                ignore_tail = max(0.0, float(getattr(self, "source_tail_safety_target_tail_ignore_sec", 0.0)))
                if float(task.target_start) >= max(0.0, float(self.target_duration - ignore_tail)):
                    return False, 999.0
            src_duration = self.get_video_duration(src)
            if src_duration <= 0.0:
                return False, 999.0
            tail_left = float(src_duration - (float(src_start) + float(task.duration)))
            return bool(tail_left < margin), float(tail_left)

        def try_rescue(
            reason: str,
            combined_hint: float,
            avoid_tail_risky: bool = False,
        ) -> Optional[Tuple[Path, float, Dict[str, object]]]:
            """
            禁用目标兜底时的候选重救：
            从 pHash 候选里挑“画面核验 + 音频守卫 +（尾段时）尾段守卫”均通过的最佳段。
            """
            candidates = self.find_match_by_phash(
                task.target_start,
                task.duration,
                seg_index=task.index,
                top_k=40,
            )
            if not candidates:
                return None

            best_item: Optional[Tuple[Path, float, Dict[str, object]]] = None
            best_rank = -1.0
            seen: Set[Tuple[str, float]] = set()
            for cand_source, cand_start, phash_sim in candidates:
                if cand_source == self.target_video:
                    continue
                sig = (str(cand_source), round(float(cand_start), 3))
                if sig in seen:
                    continue
                seen.add(sig)

                refined_start, refined_score = self.refine_start_by_visual(
                    source=cand_source,
                    initial_start=float(cand_start),
                    target_start=float(task.target_start),
                    duration=float(task.duration),
                )
                verify_passed, verify_avg = self.quick_verify(
                    cand_source,
                    float(refined_start),
                    float(task.target_start),
                    float(task.duration),
                )
                if not verify_passed:
                    continue

                tail_risky, tail_left = source_tail_risky(cand_source, float(refined_start))
                if avoid_tail_risky and tail_risky:
                    continue

                tail_avg = verify_avg
                if self.target_duration > 0 and task.target_start >= (self.target_duration - self.tail_guard_seconds):
                    offsets = [
                        0.0,
                        task.duration * 0.25,
                        task.duration * 0.5,
                        task.duration * 0.75,
                        max(0.0, task.duration - 0.1),
                    ]
                    tail_passed, tail_avg = self.verify_segment_visual(
                        cand_source,
                        float(refined_start),
                        float(task.target_start),
                        float(task.duration),
                        offsets=offsets,
                        min_avg=self.tail_verify_min_avg,
                        min_floor=self.tail_verify_min_floor,
                    )
                    if not tail_passed:
                        continue

                audio_passed = True
                audio_meta: Dict[str, object] = {"checked": False}
                if cand_source != self.target_video:
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=cand_source,
                        source_start=float(refined_start),
                        target_start=float(task.target_start),
                        duration=float(task.duration),
                        combined_score=max(float(combined_hint), float(verify_avg)),
                    )
                if not audio_passed:
                    continue
                shift_meta: Dict[str, object] = {"applied": False, "checked": False}
                if cand_source != self.target_video:
                    realigned_start, shift_meta, shifted_audio_meta = self.try_audio_guard_shift_realign(
                        source=cand_source,
                        source_start=float(refined_start),
                        target_start=float(task.target_start),
                        duration=float(task.duration),
                        combined_score=max(float(combined_hint), float(verify_avg)),
                        audio_meta=audio_meta,
                    )
                    if bool(shift_meta.get("applied", False)):
                        refined_start = float(realigned_start)
                        verify_avg = max(float(verify_avg), float(shift_meta.get("candidate_verify_avg", verify_avg)))
                        if isinstance(shifted_audio_meta, dict):
                            audio_meta = shifted_audio_meta
                    else:
                        shift_meta = dict(shift_meta or {})

                if (
                    reason in {"audio_guard_shift_bias", "audio_guard_shift_bias_halfsec"}
                    and cand_source != self.target_video
                    and float(refined_start) <= 1.6
                ):
                    src_duration = self.get_video_duration(cand_source)
                    max_anchor_start = max(0.0, min(1.6, float(src_duration - float(task.duration))))
                    best_head: Optional[Dict[str, object]] = None
                    for step_idx in range(int(max_anchor_start / 0.2) + 1):
                        anchor_start = round(float(step_idx) * 0.2, 3)
                        head_passed, head_avg = self.quick_verify(
                            cand_source,
                            float(anchor_start),
                            float(task.target_start),
                            float(task.duration),
                        )
                        if not head_passed:
                            continue
                        head_audio_passed, head_audio_meta = self.quick_verify_audio(
                            source=cand_source,
                            source_start=float(anchor_start),
                            target_start=float(task.target_start),
                            duration=float(task.duration),
                            combined_score=max(float(combined_hint), float(head_avg)),
                        )
                        if not head_audio_passed:
                            continue
                        head_realigned, head_shift_meta, head_shifted_audio_meta = self.try_audio_guard_shift_realign(
                            source=cand_source,
                            source_start=float(anchor_start),
                            target_start=float(task.target_start),
                            duration=float(task.duration),
                            combined_score=max(float(combined_hint), float(head_avg)),
                            audio_meta=head_audio_meta,
                        )
                        if bool(head_shift_meta.get("applied", False)):
                            anchor_start = float(head_realigned)
                            head_avg = max(float(head_avg), float(head_shift_meta.get("candidate_verify_avg", head_avg)))
                            if isinstance(head_shifted_audio_meta, dict):
                                head_audio_meta = head_shifted_audio_meta
                        head_score = float(head_avg) - float(anchor_start) * 0.18
                        if best_head is None:
                            best_head = {
                                "start": float(anchor_start),
                                "avg": float(head_avg),
                                "audio_meta": dict(head_audio_meta or {}),
                                "shift_meta": dict(head_shift_meta or {}),
                                "score": float(head_score),
                            }
                            continue
                        if float(head_avg) > float(best_head["avg"]) + 0.015:
                            best_head = {
                                "start": float(anchor_start),
                                "avg": float(head_avg),
                                "audio_meta": dict(head_audio_meta or {}),
                                "shift_meta": dict(head_shift_meta or {}),
                                "score": float(head_score),
                            }
                            continue
                        if abs(float(head_avg) - float(best_head["avg"])) <= 0.015 and float(anchor_start) + 1e-6 < float(best_head["start"]):
                            best_head = {
                                "start": float(anchor_start),
                                "avg": float(head_avg),
                                "audio_meta": dict(head_audio_meta or {}),
                                "shift_meta": dict(head_shift_meta or {}),
                                "score": float(head_score),
                            }

                    if best_head is not None and float(best_head["avg"]) + 0.02 >= float(verify_avg):
                        refined_start = float(best_head["start"])
                        verify_avg = max(float(verify_avg), float(best_head["avg"]))
                        audio_meta = dict(best_head["audio_meta"] or {})
                        shift_meta = dict(best_head["shift_meta"] or {})

                head_late_penalty = 0.0
                if (
                    reason in {"audio_guard_shift_bias", "audio_guard_shift_bias_halfsec"}
                    and float(refined_start) > 0.12
                    and float(refined_start) <= 1.6
                ):
                    head_late_penalty = min(0.28, float(refined_start) * 0.18)

                rank = (
                    float(verify_avg) * 0.75
                    + float(phash_sim) * 0.20
                    + float(refined_score) * 0.05
                    - float(head_late_penalty)
                )
                if rank <= best_rank:
                    continue

                best_rank = rank
                best_item = (
                    cand_source,
                    float(refined_start),
                    {
                        "triggered": True,
                        "reason": reason,
                        "verify_avg": float(verify_avg),
                        "tail_avg": float(tail_avg),
                        "phash_similarity": float(phash_sim),
                        "start_refine_score": float(refined_score),
                        "audio_guard": audio_meta,
                        "audio_shift_fix": shift_meta,
                        "head_late_penalty": float(head_late_penalty),
                        "tail_left_sec": float(tail_left),
                    },
                )
            return best_item

        # 音频+画面结合匹配
        source, source_start, combined_score, match_meta = self.find_match_combined(
            task.target_start, task.duration, task.index
        )

        quality = {
            "combined": combined_score,
            "low_confidence": combined_score < self.low_score_threshold,
            "rematch_triggered": False,
            "rematch_improved": False,
            "match_meta": match_meta,
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

        if source and source != self.target_video:
            tail_risky, tail_left = source_tail_risky(source, float(source_start))
            quality["source_tail_safety"] = {
                "checked": True,
                "risky": bool(tail_risky),
                "tail_left_sec": float(tail_left),
            }
            if tail_risky:
                rescue = try_rescue("source_tail_risky", float(combined_score), avoid_tail_risky=True)
                if rescue is not None:
                    rescue_verify = float((rescue[2] or {}).get("verify_avg", combined_score))
                    min_gain = max(0.0, float(getattr(self, "source_tail_safety_switch_min_gain", 0.0)))
                    # 仅当“确实抽不满（tail_left<0）”时，才允许无条件切换；
                    # 对仍可抽满但靠近片尾的情况，必须满足明显收益，避免提前跳源导致错序。
                    severe_tail_shortfall = float(tail_left) < -1e-3
                    allow_switch = severe_tail_shortfall or (rescue_verify >= float(combined_score) + min_gain)
                    tail_locked = bool(float(tail_left) <= max(0.20, float(task.duration) * 0.10))
                    hard_escape_floor = max(0.78, float(combined_score) - 0.16)
                    # 硬切仅用于“当前候选已无法抽满（tail_left<0）”的真实短缺场景。
                    # 对仅“贴近片尾但仍可抽满”的候选，不再用硬切强行换源，避免引入内容缺失。
                    hard_escape_tail = bool(
                        severe_tail_shortfall
                        and tail_locked
                        and (rescue_verify >= hard_escape_floor)
                    )
                    if (not allow_switch) and hard_escape_tail:
                        allow_switch = True
                        quality["source_tail_safety"]["hard_escape_tail_locked"] = True
                        quality["source_tail_safety"]["hard_escape_floor"] = float(hard_escape_floor)
                    else:
                        quality["source_tail_safety"]["hard_escape_tail_locked"] = False
                    soft_force_margin = max(
                        0.0,
                        float(
                            getattr(
                                self,
                                "source_tail_safety_soft_force_margin",
                                max(0.12, float(getattr(self, "source_tail_safety_margin", 0.0)) * 0.8),
                            )
                        ),
                    )
                    soft_force_max_drop = max(
                        0.0,
                        float(getattr(self, "source_tail_safety_soft_force_max_drop", 0.03)),
                    )
                    soft_force_tail = bool(float(tail_left) <= soft_force_margin)
                    # 软切换：近尾风险很高时，允许小幅收益不足但不显著变差的候选切换。
                    if (
                        (not allow_switch)
                        and soft_force_tail
                        and (rescue_verify >= float(combined_score) - soft_force_max_drop)
                    ):
                        allow_switch = True
                        quality["source_tail_safety"]["soft_forced"] = True
                        quality["source_tail_safety"]["soft_force_margin_sec"] = float(soft_force_margin)
                        quality["source_tail_safety"]["soft_force_max_drop"] = float(soft_force_max_drop)
                    else:
                        quality["source_tail_safety"]["soft_forced"] = False

                    # 跨源大跳变额外约束：防止片尾安全切换把段错误切到另一个时间带。
                    # 这里用“起点跳变量”做近似（同一 target_start 下等价于 mapping jump）。
                    if allow_switch and (rescue[0] != source):
                        rescue_start = float(rescue[1])
                        jump_sec = abs(float(rescue_start) - float(source_start))
                        jump_limit = max(1.2, float(task.duration) * 0.35)
                        if jump_sec > jump_limit and (not severe_tail_shortfall) and (not hard_escape_tail):
                            # 无明确短缺时，跨源大跳变必须有更高收益才允许。
                            extra_gain = max(0.08, min_gain * 2.0)
                            if rescue_verify < float(combined_score) + extra_gain:
                                allow_switch = False
                                quality["source_tail_safety"]["switched"] = False
                                quality["source_tail_safety"]["switch_reason"] = "nonlocal_jump_gain_insufficient"
                                quality["source_tail_safety"]["jump_sec"] = float(jump_sec)
                                quality["source_tail_safety"]["jump_limit_sec"] = float(jump_limit)
                        elif jump_sec > jump_limit and hard_escape_tail:
                            quality["source_tail_safety"]["nonlocal_jump_overridden_by_hard_escape"] = True
                    if allow_switch:
                        source, source_start, rescue_meta = rescue
                        quality["rescue"] = rescue_meta
                        quality["source_tail_safety"]["switched"] = True
                        if hard_escape_tail:
                            quality["source_tail_safety"]["switch_reason"] = "avoid_source_tail_risky_hard_escape"
                        elif bool(quality["source_tail_safety"].get("soft_forced", False)):
                            quality["source_tail_safety"]["switch_reason"] = "avoid_source_tail_risky_soft_force"
                        else:
                            quality["source_tail_safety"]["switch_reason"] = "avoid_source_tail_risky"
                        quality["source_tail_safety"]["switched_tail_left_sec"] = float(rescue_meta.get("tail_left_sec", 999.0))
                        quality["combined"] = max(float(quality.get("combined", 0.0)), float(rescue_meta.get("verify_avg", 0.0)))
                        combined_score = max(float(combined_score), float(rescue_meta.get("verify_avg", 0.0)))
                    else:
                        quality["source_tail_safety"]["switched"] = False
                        quality["source_tail_safety"]["switch_reason"] = "candidate_gain_insufficient"
                else:
                    quality["source_tail_safety"]["switched"] = False

        if not source or combined_score < 0.70:
            if self.enable_target_video_fallback:
                # 兜底：使用目标视频本身对应时间段
                print(f"   段 {task.index + 1}/{self.total_segments} ⚠️ 匹配失败 (score={combined_score:.2f})，使用目标视频兜底")
                return SegmentResult(
                    index=task.index,
                    success=True,
                    source=self.target_video,
                    source_start=task.target_start,
                    quality={
                        **quality,
                        "combined": 0.0,
                        "fallback": True,
                        "fallback_reason": "match_failed_low_score",
                    }
                )
            rescue = try_rescue("match_failed_low_score", float(combined_score))
            if rescue is not None:
                source, source_start, rescue_meta = rescue
                quality["rescue"] = rescue_meta
                quality["combined"] = max(float(quality.get("combined", 0.0)), float(rescue_meta.get("verify_avg", 0.0)))
                combined_score = max(float(combined_score), float(rescue_meta.get("verify_avg", 0.0)))
                quality["start_refine"] = {
                    "before": float(source_start),
                    "after": float(source_start),
                    "score": float(rescue_meta.get("start_refine_score", 0.0)),
                }
                quality["audio_guard"] = rescue_meta.get("audio_guard", {"checked": False})
                quality["strict_verify"] = {"passed": True, "avg": float(rescue_meta.get("verify_avg", 0.0))}
                print(
                    f"   段 {task.index + 1}/{self.total_segments} 🔄 候选重救成功 "
                    f"({source.name} @ {source_start:.3f}s, avg={float(rescue_meta.get('verify_avg', 0.0)):.2f})"
                )
            else:
                print(
                    f"   段 {task.index + 1}/{self.total_segments} ❌ 匹配失败 "
                    f"(score={combined_score:.2f})，且已禁用目标素材兜底"
                )
                return SegmentResult(
                    index=task.index,
                    success=False,
                    quality={
                        **quality,
                        "combined": float(combined_score),
                        "fallback_blocked": True,
                        "fallback_reason": "match_failed_no_target_fallback",
                    },
                )

        # 通用防误匹配：段级画面快速核验，失败则回退目标片段
        if self.strict_visual_verify:
            passed, verify_avg = self.quick_verify(source, source_start, task.target_start, task.duration)
            quality["strict_verify"] = {"passed": bool(passed), "avg": float(verify_avg)}
            if not passed:
                if self.enable_target_video_fallback:
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
                rescue = try_rescue("strict_visual_verify_failed", float(combined_score))
                if rescue is not None:
                    source, source_start, rescue_meta = rescue
                    quality["rescue"] = rescue_meta
                    quality["strict_verify"] = {"passed": True, "avg": float(rescue_meta.get("verify_avg", verify_avg))}
                    quality["combined"] = max(float(quality.get("combined", 0.0)), float(rescue_meta.get("verify_avg", 0.0)))
                    quality["audio_guard"] = rescue_meta.get("audio_guard", {"checked": False})
                    print(
                        f"   段 {task.index + 1}/{self.total_segments} 🔄 画面失败后重救成功 "
                        f"({source.name} @ {source_start:.3f}s, avg={float(rescue_meta.get('verify_avg', 0.0)):.2f})"
                    )
                else:
                    print(
                        f"   段 {task.index + 1}/{self.total_segments} ❌ 画面核验失败 "
                        f"(avg={verify_avg:.2f})，且已禁用目标素材兜底"
                    )
                    return SegmentResult(
                        index=task.index,
                        success=False,
                        quality={
                            **quality,
                            "fallback_blocked": True,
                            "fallback_reason": "strict_visual_verify_failed_no_target_fallback",
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
                strict_verify_avg = float((quality.get("strict_verify", {}) or {}).get("avg", 0.0))
                is_final_partial_tail = bool(
                    self.target_duration > 0.0
                    and (float(task.target_start) + float(task.duration) >= float(self.target_duration) - 1e-3)
                    and float(task.duration) < max(0.5, float(self.segment_duration) - 0.1)
                )
                if (
                    (reason in {"audio_guard_shift_bias", "audio_guard_shift_bias_halfsec"})
                    and is_final_partial_tail
                    and strict_verify_avg >= 0.90
                ):
                    audio_meta["passed"] = True
                    audio_meta["bypassed"] = True
                    audio_meta["bypass_reason"] = "tail_partial_visual_strong_keep_current"
                    quality["audio_guard"] = audio_meta
                    quality["audio_guard_tail_partial_bypassed_no_target"] = True
                    quality["audio_guard_tail_partial_bypass_reason"] = str(reason)
                elif self.enable_target_video_fallback:
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
                else:
                    rescue = try_rescue(reason, float(combined_score))
                    if rescue is not None:
                        source, source_start, rescue_meta = rescue
                        quality["rescue"] = rescue_meta
                        quality["audio_guard"] = rescue_meta.get("audio_guard", {"checked": False})
                        quality["combined"] = max(float(quality.get("combined", 0.0)), float(rescue_meta.get("verify_avg", 0.0)))
                        print(
                            f"   段 {task.index + 1}/{self.total_segments} 🔄 音频失败后重救成功 "
                            f"({source.name} @ {source_start:.3f}s)"
                        )
                    else:
                        print(
                            f"   段 {task.index + 1}/{self.total_segments} ❌ 音频守卫失败 "
                            f"(sim={aligned_sim:.2f}, shift={best_shift:+.1f}s, reason={reason})，且已禁用目标素材兜底"
                        )
                        return SegmentResult(
                            index=task.index,
                            success=False,
                            quality={
                                **quality,
                                "fallback_blocked": True,
                                "fallback_reason": f"{reason}_no_target_fallback",
                            },
                        )
            else:
                shifted_start, shift_meta, shifted_audio_meta = self.try_audio_guard_shift_realign(
                    source=source,
                    source_start=float(source_start),
                    target_start=float(task.target_start),
                    duration=float(task.duration),
                    combined_score=float(quality.get("combined", combined_score)),
                    audio_meta=audio_meta,
                )
                quality["audio_guard_shift_fix"] = shift_meta
                if bool(shift_meta.get("applied", False)):
                    source_start = float(shifted_start)
                    if isinstance(shifted_audio_meta, dict):
                        quality["audio_guard"] = shifted_audio_meta
                    combined_score = max(float(combined_score), float(shift_meta.get("candidate_verify_avg", 0.0)))
                    quality["combined"] = max(float(quality.get("combined", 0.0)), float(combined_score))

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
                if self.enable_target_video_fallback:
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
                rescue = try_rescue("tail_guard_failed", float(combined_score))
                if rescue is not None:
                    source, source_start, rescue_meta = rescue
                    quality["rescue"] = rescue_meta
                    quality["tail_verify"] = {"passed": True, "avg": float(rescue_meta.get("tail_avg", tail_avg))}
                    quality["audio_guard"] = rescue_meta.get("audio_guard", {"checked": False})
                    quality["combined"] = max(float(quality.get("combined", 0.0)), float(rescue_meta.get("verify_avg", 0.0)))
                    print(
                        f"   段 {task.index + 1}/{self.total_segments} 🔄 尾段失败后重救成功 "
                        f"({source.name} @ {source_start:.3f}s)"
                    )
                else:
                    print(
                        f"   段 {task.index + 1}/{self.total_segments} ❌ 尾段守卫失败 "
                        f"(avg={tail_avg:.2f})，且已禁用目标素材兜底"
                    )
                    return SegmentResult(
                        index=task.index,
                        success=False,
                        quality={
                            **quality,
                            "fallback_blocked": True,
                            "fallback_reason": "tail_guard_failed_no_target_fallback",
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

    def recover_segment_from_neighbors(
        self,
        index: int,
        tasks: List[SegmentTask],
        confirmed_by_index: Dict[int, dict],
    ) -> Optional[dict]:
        """
        在禁用目标兜底时，尝试用邻段时间轴插值恢复缺失段，避免直接输出错序。
        """
        if index < 0 or index >= len(tasks):
            return None
        task = tasks[index]

        def find_prev() -> Tuple[Optional[dict], Optional[int]]:
            j = index - 1
            while j >= 0:
                seg = confirmed_by_index.get(j)
                if seg is not None and seg.get("source") is not None and seg.get("source") != self.target_video:
                    return seg, j
                j -= 1
            return None, None

        def find_next() -> Tuple[Optional[dict], Optional[int]]:
            j = index + 1
            while j < len(tasks):
                seg = confirmed_by_index.get(j)
                if seg is not None and seg.get("source") is not None and seg.get("source") != self.target_video:
                    return seg, j
                j += 1
            return None, None

        prev_seg, _ = find_prev()
        next_seg, _ = find_next()
        candidates: List[Tuple[Path, float, str]] = []

        def clamp_start_by_neighbor_bounds(src: Path, raw_start: float) -> Tuple[float, Dict[str, object]]:
            """
            将候选起点限制在“同源前后邻段”允许窗口内。
            若窗口出现反转（说明当前段时窗被挤压，不可完全无重叠放入），
            则落到窗口中点，把不可避免的重叠尽量均摊到两侧，降低单侧重读感。
            """
            src_duration = self.get_video_duration(src)
            max_start = max(0.0, float(src_duration - float(task.duration)))
            start = max(0.0, min(float(raw_start), max_start))

            overlap_allow = max(0.35, float(task.duration) * 0.07)
            lower_bound: Optional[float] = None
            upper_bound: Optional[float] = None
            if prev_seg is not None and prev_seg.get("source") == src:
                lower_bound = float(prev_seg["start"]) + float(prev_seg["duration"]) - overlap_allow
            if next_seg is not None and next_seg.get("source") == src:
                upper_bound = float(next_seg["start"]) - float(task.duration) + overlap_allow

            compressed_window = False
            if lower_bound is not None and upper_bound is not None and lower_bound > (upper_bound + 1e-6):
                compressed_window = True
                start = 0.5 * (float(lower_bound) + float(upper_bound))
            else:
                if lower_bound is not None:
                    start = max(float(start), float(lower_bound))
                if upper_bound is not None:
                    start = min(float(start), float(upper_bound))

            start = max(0.0, min(float(start), max_start))
            return float(start), {
                "lower_bound": float(lower_bound) if lower_bound is not None else None,
                "upper_bound": float(upper_bound) if upper_bound is not None else None,
                "compressed_window": bool(compressed_window),
            }

        def boundary_excess_penalty(src: Path, start: float) -> float:
            """
            计算候选在同源邻段边界上的“超限量”。
            用于候选排序，优先避免把重叠/断缝集中到单侧。
            """
            penalty = 0.0

            if prev_seg is not None and prev_seg.get("source") == src:
                prev_end = float(prev_seg["start"]) + float(prev_seg["duration"])
                gap_prev = float(start) - prev_end
                neg_trigger = max(0.12, float(task.duration) * 0.02)
                pos_trigger = max(0.20, float(task.duration) * 0.04)
                if gap_prev < -neg_trigger:
                    penalty += float((-gap_prev) - neg_trigger)
                elif gap_prev > pos_trigger:
                    penalty += float(gap_prev - pos_trigger) * 0.65

            if next_seg is not None and next_seg.get("source") == src:
                curr_end = float(start) + float(task.duration)
                gap_next = float(next_seg["start"]) - curr_end
                next_duration = float(next_seg.get("duration", task.duration))
                neg_trigger = max(0.12, next_duration * 0.02)
                pos_trigger = max(0.20, next_duration * 0.04)
                if gap_next < -neg_trigger:
                    penalty += float((-gap_next) - neg_trigger)
                elif gap_next > pos_trigger:
                    penalty += float(gap_next - pos_trigger) * 0.65

            return float(max(0.0, penalty))

        def verify_for_missing_recovery(src: Path, src_start: float) -> Tuple[bool, float, bool]:
            """
            缺失段恢复核验：
            先走严格 quick_verify；若失败，再走仅用于缺失恢复的宽松多点核验。
            返回: (是否通过, 分数, 是否走了宽松核验)。
            """
            strict_passed, strict_avg = self.quick_verify(
                src,
                float(src_start),
                float(task.target_start),
                float(task.duration),
            )
            if strict_passed:
                return True, float(strict_avg), False

            offsets = [
                0.0,
                task.duration * 0.25,
                task.duration * 0.5,
                task.duration * 0.75,
                max(0.0, task.duration - 0.1),
            ]
            relaxed_min_avg = max(0.76, float(self.strict_verify_min_sim) - 0.02)
            relaxed_min_floor = max(0.58, relaxed_min_avg - 0.20)
            relaxed_passed, relaxed_avg = self.verify_segment_visual(
                src,
                float(src_start),
                float(task.target_start),
                float(task.duration),
                offsets=offsets,
                min_avg=relaxed_min_avg,
                min_floor=relaxed_min_floor,
            )
            return bool(relaxed_passed), float(relaxed_avg), True

        if prev_seg is not None and next_seg is not None and prev_seg["source"] == next_seg["source"]:
            prev_t = float(prev_seg["target_start"])
            next_t = float(next_seg["target_start"])
            if next_t > prev_t + 1e-6:
                ratio = (float(task.target_start) - prev_t) / (next_t - prev_t)
                ratio = max(0.0, min(1.0, ratio))
                interp_start = float(prev_seg["start"]) + (float(next_seg["start"]) - float(prev_seg["start"])) * ratio
                candidates.append((prev_seg["source"], interp_start, "neighbors_interp"))

        if prev_seg is not None:
            extrap_prev = float(prev_seg["start"]) + (float(task.target_start) - float(prev_seg["target_start"]))
            candidates.append((prev_seg["source"], extrap_prev, "prev_extrap"))
        if next_seg is not None:
            extrap_next = float(next_seg["start"]) - (float(next_seg["target_start"]) - float(task.target_start))
            candidates.append((next_seg["source"], extrap_next, "next_extrap"))

        checked: Set[Tuple[str, float]] = set()
        for source, raw_start, mode in candidates:
            if source is None or source == self.target_video or raw_start < 0:
                continue
            if self._is_secondary_source(Path(source)):
                continue
            sig = (str(source), round(float(raw_start), 3))
            if sig in checked:
                continue
            checked.add(sig)

            bounded_start, bound_meta = clamp_start_by_neighbor_bounds(source, float(raw_start))

            refined_start, refine_score = self.refine_start_by_visual(
                source=source,
                initial_start=float(bounded_start),
                target_start=float(task.target_start),
                duration=float(task.duration),
            )
            refined_start, bound_meta = clamp_start_by_neighbor_bounds(source, float(refined_start))
            verify_passed, verify_avg, relaxed_verify = verify_for_missing_recovery(
                source,
                float(refined_start),
            )
            if not verify_passed:
                continue

            audio_passed = True
            audio_meta: Dict[str, object] = {"checked": False}
            if source != self.target_video:
                audio_passed, audio_meta = self.quick_verify_audio(
                    source=source,
                    source_start=float(refined_start),
                    target_start=float(task.target_start),
                    duration=float(task.duration),
                    combined_score=float(verify_avg),
                )
            if not audio_passed:
                continue

            tail_meta: Dict[str, object] = {"checked": False}
            if self.target_duration > 0 and task.target_start >= (self.target_duration - self.tail_guard_seconds):
                offsets = [
                    0.0,
                    task.duration * 0.25,
                    task.duration * 0.5,
                    task.duration * 0.75,
                    max(0.0, task.duration - 0.1),
                ]
                tail_passed, tail_avg = self.verify_segment_visual(
                    source,
                    float(refined_start),
                    float(task.target_start),
                    float(task.duration),
                    offsets=offsets,
                    min_avg=self.tail_verify_min_avg,
                    min_floor=self.tail_verify_min_floor,
                )
                tail_meta = {"checked": True, "passed": bool(tail_passed), "avg": float(tail_avg)}
                if not tail_passed:
                    continue

            return {
                "index": int(task.index),
                "source": source,
                "start": float(refined_start),
                "duration": float(task.duration),
                "target_start": float(task.target_start),
                "quality": {
                    "combined": float(verify_avg),
                    "recovered_from_neighbors": True,
                    "recover_mode": mode,
                    "recover_verify_avg": float(verify_avg),
                    "recover_refine_score": float(refine_score),
                    "recover_relaxed_verify": bool(relaxed_verify),
                    "recover_boundary_excess": float(boundary_excess_penalty(source, float(refined_start))),
                    "recover_neighbor_bounds": bound_meta,
                    "audio_guard": audio_meta,
                    "tail_verify": tail_meta,
                },
            }

        # 邻段推导未命中时，使用 pHash 全库候选做二次救援，避免直接进入“未核验强制插值”。
        phash_candidates = self.find_match_by_phash(
            task.target_start,
            task.duration,
            seg_index=task.index,
            top_k=48,
        )
        best_rescue = None
        seen_phash: Set[Tuple[str, float]] = set()
        for source, raw_start, phash_sim in phash_candidates:
            if source is None or source == self.target_video:
                continue
            if self._is_secondary_source(Path(source)):
                continue

            sig = (str(source), round(float(raw_start), 3))
            if sig in seen_phash:
                continue
            seen_phash.add(sig)

            bounded_start, bound_meta = clamp_start_by_neighbor_bounds(source, float(raw_start))

            refined_start, refine_score = self.refine_start_by_visual(
                source=source,
                initial_start=float(bounded_start),
                target_start=float(task.target_start),
                duration=float(task.duration),
            )
            refined_start, bound_meta = clamp_start_by_neighbor_bounds(source, float(refined_start))
            verify_passed, verify_avg, relaxed_verify = verify_for_missing_recovery(
                source,
                float(refined_start),
            )
            if not verify_passed:
                continue

            audio_passed = True
            audio_meta: Dict[str, object] = {"checked": False}
            if source != self.target_video:
                audio_passed, audio_meta = self.quick_verify_audio(
                    source=source,
                    source_start=float(refined_start),
                    target_start=float(task.target_start),
                    duration=float(task.duration),
                    combined_score=float(verify_avg),
                )
            if not audio_passed:
                continue

            tail_meta: Dict[str, object] = {"checked": False}
            if self.target_duration > 0 and task.target_start >= (self.target_duration - self.tail_guard_seconds):
                offsets = [
                    0.0,
                    task.duration * 0.25,
                    task.duration * 0.5,
                    task.duration * 0.75,
                    max(0.0, task.duration - 0.1),
                ]
                tail_passed, tail_avg = self.verify_segment_visual(
                    source,
                    float(refined_start),
                    float(task.target_start),
                    float(task.duration),
                    offsets=offsets,
                    min_avg=self.tail_verify_min_avg,
                    min_floor=self.tail_verify_min_floor,
                )
                tail_meta = {"checked": True, "passed": bool(tail_passed), "avg": float(tail_avg)}
                if not tail_passed:
                    continue

            neighbor_bonus = 0.0
            if prev_seg is not None and source == prev_seg.get("source"):
                neighbor_bonus += 0.01
            if next_seg is not None and source == next_seg.get("source"):
                neighbor_bonus += 0.01
            boundary_excess = boundary_excess_penalty(source, float(refined_start))
            rank = (
                float(verify_avg) * 0.78
                + float(phash_sim) * 0.18
                + float(refine_score) * 0.04
                + neighbor_bonus
                - float(boundary_excess) * 0.28
            )
            if best_rescue is None or rank > float(best_rescue["rank"]):
                best_rescue = {
                    "source": source,
                    "start": float(refined_start),
                    "verify_avg": float(verify_avg),
                    "refine_score": float(refine_score),
                    "phash_similarity": float(phash_sim),
                    "relaxed_verify": bool(relaxed_verify),
                    "audio_meta": audio_meta,
                    "tail_meta": tail_meta,
                    "boundary_excess": float(boundary_excess),
                    "neighbor_bounds": bound_meta,
                    "rank": float(rank),
                }

        if best_rescue is not None:
            return {
                "index": int(task.index),
                "source": best_rescue["source"],
                "start": float(best_rescue["start"]),
                "duration": float(task.duration),
                "target_start": float(task.target_start),
                "quality": {
                    "combined": float(best_rescue["verify_avg"]),
                    "recovered_from_neighbors": True,
                    "recover_mode": "phash_rescue_no_target_missing",
                    "recover_verify_avg": float(best_rescue["verify_avg"]),
                    "recover_refine_score": float(best_rescue["refine_score"]),
                    "recover_phash_similarity": float(best_rescue["phash_similarity"]),
                    "recover_relaxed_verify": bool(best_rescue.get("relaxed_verify", False)),
                    "recover_boundary_excess": float(best_rescue.get("boundary_excess", 0.0)),
                    "recover_neighbor_bounds": best_rescue.get("neighbor_bounds", {}),
                    "audio_guard": best_rescue["audio_meta"],
                    "tail_verify": best_rescue["tail_meta"],
                },
            }

        # 强制邻段插值（仅同源双邻段）：
        # 当严格核验无法通过但时间轴强连续时，优先保顺序，避免整条任务失败。
        if prev_seg is not None and next_seg is not None and prev_seg["source"] == next_seg["source"]:
            if self._is_secondary_source(Path(prev_seg["source"])):
                prev_seg = None
                next_seg = None
        if prev_seg is not None and next_seg is not None and prev_seg["source"] == next_seg["source"]:
            prev_t = float(prev_seg["target_start"])
            next_t = float(next_seg["target_start"])
            if next_t > prev_t + 1e-6:
                # 尾段对“未核验强制插值”更敏感，容易引入重复/回放感，优先拒绝。
                in_tail_zone = (
                    self.target_duration > 0.0
                    and float(task.target_start)
                    >= max(0.0, float(self.target_duration - max(float(self.tail_guard_seconds), 20.0)))
                )
                ratio = (float(task.target_start) - prev_t) / (next_t - prev_t)
                ratio = max(0.0, min(1.0, ratio))
                forced_start = float(prev_seg["start"]) + (float(next_seg["start"]) - float(prev_seg["start"])) * ratio
                from_prev = float(prev_seg["start"]) + (float(task.target_start) - prev_t)
                from_next = float(next_seg["start"]) - (next_t - float(task.target_start))
                bridge_gap = abs(from_prev - from_next)
                bridge_tol = max(0.35, float(task.duration) * 0.08)
                prev_q = prev_seg.get("quality", {}) or {}
                next_q = next_seg.get("quality", {}) or {}
                tail_force_strict_ok = (
                    bridge_gap <= max(0.25, float(task.duration) * 0.06)
                    and float(prev_q.get("combined", 0.0)) >= 0.88
                    and float(next_q.get("combined", 0.0)) >= 0.88
                )
                if forced_start >= 0 and bridge_gap <= bridge_tol and (not in_tail_zone or tail_force_strict_ok):
                    source_duration = self.get_video_duration(prev_seg["source"])
                    max_start = max(0.0, float(source_duration - float(task.duration)))
                    forced_start = max(0.0, min(float(forced_start), max_start))
                    return {
                        "index": int(task.index),
                        "source": prev_seg["source"],
                        "start": float(forced_start),
                        "duration": float(task.duration),
                        "target_start": float(task.target_start),
                        "quality": {
                            "combined": 0.0,
                            "recovered_from_neighbors": True,
                            "recover_mode": "neighbors_forced_no_verify",
                            "recover_forced": True,
                            "recover_tail_relaxed_force": bool(in_tail_zone),
                            "recover_bridge_gap_sec": float(bridge_gap),
                        },
                    }

        # 最后兜底（仍不使用目标素材）：
        # 当前后同源强制插值也不可用时，允许单侧外推并做宽松核验，避免整条任务直接失败。
        last_resort: List[Tuple[Path, float, str]] = []
        if prev_seg is not None and prev_seg.get("source") not in (None, self.target_video):
            if self._is_secondary_source(Path(prev_seg["source"])):
                prev_seg = None
        if next_seg is not None and next_seg.get("source") not in (None, self.target_video):
            if self._is_secondary_source(Path(next_seg["source"])):
                next_seg = None
        if prev_seg is not None and prev_seg.get("source") not in (None, self.target_video):
            est = float(prev_seg["start"]) + (float(task.target_start) - float(prev_seg["target_start"]))
            last_resort.append((prev_seg["source"], est, "prev_last_resort"))
        if next_seg is not None and next_seg.get("source") not in (None, self.target_video):
            est = float(next_seg["start"]) - (float(next_seg["target_start"]) - float(task.target_start))
            last_resort.append((next_seg["source"], est, "next_last_resort"))
        if (
            prev_seg is not None
            and next_seg is not None
            and prev_seg.get("source") not in (None, self.target_video)
            and prev_seg.get("source") == next_seg.get("source")
        ):
            prev_map = float(prev_seg["start"]) + (float(task.target_start) - float(prev_seg["target_start"]))
            next_map = float(next_seg["start"]) - (float(next_seg["target_start"]) - float(task.target_start))
            balanced = 0.5 * (float(prev_map) + float(next_map))
            last_resort.append((prev_seg["source"], balanced, "balanced_last_resort"))

        best_force = None
        seen_force: Set[Tuple[str, float]] = set()
        for source, est_start, mode in last_resort:
            src_duration = self.get_video_duration(source)
            if src_duration <= 0.0:
                continue
            max_start = max(0.0, float(src_duration - float(task.duration)))
            cand_start = max(0.0, min(float(est_start), max_start))
            cand_start, bound_meta = clamp_start_by_neighbor_bounds(source, float(cand_start))
            sig = (str(source), round(float(cand_start), 3))
            if sig in seen_force:
                continue
            seen_force.add(sig)

            refined_start, refine_score = self.refine_start_by_visual(
                source=source,
                initial_start=float(cand_start),
                target_start=float(task.target_start),
                duration=float(task.duration),
            )
            refined_start = max(0.0, min(float(refined_start), max_start))
            refined_start, bound_meta = clamp_start_by_neighbor_bounds(source, float(refined_start))
            offsets = [
                0.0,
                task.duration * 0.25,
                task.duration * 0.5,
                task.duration * 0.75,
                max(0.0, task.duration - 0.1),
            ]
            soft_passed, soft_avg = self.verify_segment_visual(
                source=source,
                source_start=float(refined_start),
                target_start=float(task.target_start),
                duration=float(task.duration),
                offsets=offsets,
                min_avg=0.70,
                min_floor=0.45,
            )
            if not soft_passed:
                continue
            boundary_excess = boundary_excess_penalty(source, float(refined_start))
            rank = float(soft_avg) + float(refine_score) * 0.02 - float(boundary_excess) * 0.35
            if best_force is None or rank > float(best_force["rank"]):
                best_force = {
                    "source": source,
                    "start": float(refined_start),
                    "mode": mode,
                    "soft_avg": float(soft_avg),
                    "boundary_excess": float(boundary_excess),
                    "neighbor_bounds": bound_meta,
                    "rank": float(rank),
                }

        if best_force is not None:
            return {
                "index": int(task.index),
                "source": best_force["source"],
                "start": float(best_force["start"]),
                "duration": float(task.duration),
                "target_start": float(task.target_start),
                "quality": {
                    "combined": float(best_force["soft_avg"]),
                    "recovered_from_neighbors": True,
                    "recover_mode": "neighbors_last_resort_no_target",
                    "recover_forced": True,
                    "recover_last_resort_side": str(best_force["mode"]),
                    "recover_soft_verify_avg": float(best_force["soft_avg"]),
                    "recover_boundary_excess": float(best_force.get("boundary_excess", 0.0)),
                    "recover_neighbor_bounds": best_force.get("neighbor_bounds", {}),
                },
            }
        return None
    
    def reconstruct_fast(self, output_path: str) -> bool:
        """极速重构"""
        import time

        print(f"\n{'='*70}")
        print(f"🚀 极速高精度重构 V3 + pHash")
        print(f"{'='*70}")

        start_wall = time.time()
        overall_perf = time.perf_counter()

        # 预建 pHash 帧索引（首次运行后缓存，后续秒速加载）
        self.build_frame_index(sample_interval=float(self.frame_index_sample_interval))

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
        
        # 整理结果：默认保全所有段；若禁用目标兜底则缺失段直接记失败
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
        recovered_missing = 0
        partial_tail_shortfall_sec = 0.0
        unresolved_missing_indices: List[int] = []
        confirmed_segments = []
        for task in tasks:
            seg = confirmed_by_index.get(task.index)
            if seg is None:
                missing_count += 1
                if self.enable_target_video_fallback:
                    seg = {
                        'index': task.index,
                        'source': self.target_video,
                        'start': task.target_start,
                        'duration': task.duration,
                        'target_start': task.target_start,
                        'quality': {'combined': 0.0, 'fallback': True, 'reason': 'missing_result'}
                    }
                else:
                    recovered = self.recover_segment_from_neighbors(task.index, tasks, confirmed_by_index)
                    if recovered is None:
                        unresolved_missing_indices.append(int(task.index))
                        continue
                    seg = recovered
                    confirmed_by_index[task.index] = recovered
                    recovered_missing += 1
            confirmed_segments.append(seg)

        if missing_count > 0:
            if self.enable_target_video_fallback:
                print(f"   ⚠️ 自动补齐缺失段: {missing_count} 段")
            else:
                neighbor_recovered_missing = int(recovered_missing)
                if neighbor_recovered_missing > 0:
                    unresolved = max(0, missing_count - neighbor_recovered_missing)
                    print(f"   🔧 邻段恢复成功: {neighbor_recovered_missing} 段，未恢复: {unresolved} 段")
                    missing_count = unresolved
                partial_tail_recovered, partial_tail_shortfall_sec = self.recover_partial_tail_segments_no_target(
                    unresolved_missing_indices,
                    tasks,
                    confirmed_by_index,
                )
                if partial_tail_recovered > 0:
                    print(
                        f"   🧩 尾部短段恢复: 补回 {partial_tail_recovered} 段，"
                        f"仍短缺 {partial_tail_shortfall_sec:.2f}s"
                    )
                    missing_count = 0
                    unresolved_missing_indices = []
                    confirmed_segments = []
                    for task in tasks:
                        seg = confirmed_by_index.get(task.index)
                        if seg is None:
                            unresolved_missing_indices.append(int(task.index))
                            missing_count += 1
                            continue
                        confirmed_segments.append(seg)
                if missing_count <= 0:
                    pass
                else:
                    allow_tail_shortfall, tail_shortfall_sec = self._resolve_missing_tail_shortfall_no_target(
                        unresolved_missing_indices,
                        tasks,
                        extra_shortfall_sec=float(partial_tail_shortfall_sec),
                    )
                    if allow_tail_shortfall:
                        print(
                            f"   ⚠️ 尾部缺段容忍生效: 缺失 {missing_count} 段 "
                            f"({tail_shortfall_sec:.2f}s)，直接输出略短成片"
                        )
                        missing_count = 0
                    else:
                        print(f"❌ 存在缺失段 {missing_count} 段，且已禁用目标素材兜底，终止输出")
                        return False

        if (not self.enable_target_video_fallback) and recovered_missing > 0:
            print(f"   🧩 邻段插值恢复: {recovered_missing} 段")

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
        bridge_recovered = self.recover_isolated_fallback_bridges(confirmed_segments)
        if bridge_recovered > 0:
            print(f"   🧩 孤立兜底桥接恢复: 恢复 {bridge_recovered} 个段到同源时间轴")
        no_target_isolated_repaired = 0
        no_target_micro_gap_snapped = 0
        no_target_boundary_unresolved = 0
        no_target_boundary_hard_clamped = 0
        no_target_boundary_audio_detected = 0
        no_target_boundary_audio_repaired = 0
        if not self.enable_target_video_fallback:
            self.last_boundary_repair_stats = {}
            no_target_isolated_repaired = self.repair_isolated_outliers_without_target_fallback(confirmed_segments)
            if no_target_isolated_repaired > 0:
                print(f"   🧪 禁兜底孤立段修复: 恢复 {no_target_isolated_repaired} 个异常跳点")
            no_target_micro_gap_snapped, no_target_boundary_unresolved = self.snap_small_adjacent_gaps_without_target_fallback(confirmed_segments)
            boundary_stats = self.last_boundary_repair_stats if isinstance(self.last_boundary_repair_stats, dict) else {}
            no_target_boundary_hard_clamped = int(boundary_stats.get("hard_clamped", 0))
            no_target_boundary_audio_detected = int(boundary_stats.get("audio_issue_detected", 0))
            no_target_boundary_audio_repaired = int(boundary_stats.get("audio_repaired", 0))
            if no_target_micro_gap_snapped > 0:
                print(f"   🎚️ 禁兜底边界微调: 对齐 {no_target_micro_gap_snapped} 个相邻段边界")
            if no_target_boundary_hard_clamped > 0:
                print(f"   📏 边界硬约束回调: {no_target_boundary_hard_clamped} 处")
            if no_target_boundary_audio_detected > 0:
                print(
                    f"   🔎 边界音频探针: 发现 {no_target_boundary_audio_detected} 处, "
                    f"局部修复 {no_target_boundary_audio_repaired} 处"
                )
            if no_target_boundary_unresolved > 0:
                print(f"   ⚠️ 禁兜底边界未完全收敛: {no_target_boundary_unresolved} 处")

            # 后置再做一轮孤立跨源段清理：
            # 前面的边界/回推修复可能改变邻段来源，导致新出现“单段错源”。
            # 二次修复后再快速收敛一次边界，可显著降低局部内容错位与句尾断裂。
            post_no_target_isolated_repaired = self.repair_isolated_outliers_without_target_fallback(confirmed_segments)
            if post_no_target_isolated_repaired > 0:
                no_target_isolated_repaired += int(post_no_target_isolated_repaired)
                print(
                    f"   🧪 禁兜底孤立段二次修复: 恢复 {post_no_target_isolated_repaired} 个后置异常跳点"
                )
                post_micro_gap_snapped, post_boundary_unresolved = self.snap_small_adjacent_gaps_without_target_fallback(
                    confirmed_segments
                )
                if post_micro_gap_snapped > 0:
                    no_target_micro_gap_snapped += int(post_micro_gap_snapped)
                    print(f"   🎚️ 禁兜底边界二次微调: 对齐 {post_micro_gap_snapped} 个相邻段边界")
                no_target_boundary_unresolved = int(post_boundary_unresolved)
                post_boundary_stats = self.last_boundary_repair_stats if isinstance(self.last_boundary_repair_stats, dict) else {}
                no_target_boundary_hard_clamped += int(post_boundary_stats.get("hard_clamped", 0))
                no_target_boundary_audio_detected += int(post_boundary_stats.get("audio_issue_detected", 0))
                no_target_boundary_audio_repaired += int(post_boundary_stats.get("audio_repaired", 0))

            # 终态兜底：避免“修完又被后续步骤改回”的孤立错源段残留到最终输出。
            final_no_target_isolated_repaired = self.repair_isolated_outliers_without_target_fallback(confirmed_segments)
            if final_no_target_isolated_repaired > 0:
                no_target_isolated_repaired += int(final_no_target_isolated_repaired)
                print(
                    f"   🧪 禁兜底孤立段终态修复: 恢复 {final_no_target_isolated_repaired} 个终态异常跳点"
                )
                final_micro_gap_snapped, final_boundary_unresolved = self.snap_small_adjacent_gaps_without_target_fallback(
                    confirmed_segments
                )
                if final_micro_gap_snapped > 0:
                    no_target_micro_gap_snapped += int(final_micro_gap_snapped)
                    print(f"   🎚️ 禁兜底边界终态微调: 对齐 {final_micro_gap_snapped} 个相邻段边界")
                no_target_boundary_unresolved = int(final_boundary_unresolved)
                final_boundary_stats = self.last_boundary_repair_stats if isinstance(self.last_boundary_repair_stats, dict) else {}
                no_target_boundary_hard_clamped += int(final_boundary_stats.get("hard_clamped", 0))
                no_target_boundary_audio_detected += int(final_boundary_stats.get("audio_issue_detected", 0))
                no_target_boundary_audio_repaired += int(final_boundary_stats.get("audio_repaired", 0))

            neighbor_source_repaired = self._repair_isolated_source_switches_with_neighbor_source_no_target(confirmed_segments)
            if neighbor_source_repaired > 0:
                no_target_isolated_repaired += int(neighbor_source_repaired)
                print(
                    f"   🧪 邻居同源优先修复: 收敛 {neighbor_source_repaired} 个孤立错源段"
                )
                # 邻居同源修复后只做“轻量边界收口”，避免再次触发跨源重匹配把错源段拉回去。
                post_neighbor_small_overlap = self._suppress_small_negative_overlaps_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                post_neighbor_hard = self._enforce_boundary_hard_constraints_no_target(
                    confirmed_segments,
                    max_passes=2,
                )
                post_neighbor_backprop = self._backprop_resolve_locked_tail_overlaps_no_target(confirmed_segments)
                post_neighbor_audio_repaired, post_neighbor_audio_issues = self._repair_boundary_audio_locally_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                post_neighbor_hard_after_audio = 0
                post_neighbor_backprop_after_audio = 0
                if post_neighbor_audio_issues > 0:
                    post_neighbor_hard_after_audio = self._enforce_boundary_hard_constraints_no_target(
                        confirmed_segments,
                        max_passes=1,
                    )
                    post_neighbor_backprop_after_audio = self._backprop_resolve_locked_tail_overlaps_no_target(
                        confirmed_segments
                    )

                post_neighbor_adjusted = int(
                    post_neighbor_small_overlap
                    + post_neighbor_hard
                    + post_neighbor_backprop
                    + post_neighbor_audio_repaired
                    + post_neighbor_hard_after_audio
                    + post_neighbor_backprop_after_audio
                )
                if post_neighbor_adjusted > 0:
                    no_target_micro_gap_snapped += int(post_neighbor_adjusted)
                    print(
                        f"   🎚️ 邻居同源修复后边界收口: 对齐 {post_neighbor_adjusted} 个相邻段边界(轻量)"
                    )
                no_target_boundary_hard_clamped += int(post_neighbor_hard + post_neighbor_hard_after_audio)
                no_target_boundary_audio_detected += int(post_neighbor_audio_issues)
                no_target_boundary_audio_repaired += int(post_neighbor_audio_repaired)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

            # 终态全局 rematch 收尾：
            # 邻居修复与边界收口可能在尾部重新引入未收敛边界，
            # 这里再执行一次定点重匹配，避免重复内容落入最终输出。
            final_global_rematch_attempts = max(
                0,
                int(getattr(self, "no_target_boundary_rematch_max_attempts", 0)),
            )
            if no_target_boundary_unresolved > 0 and final_global_rematch_attempts > 0:
                final_global_rematched = self._rematch_unresolved_boundaries_without_target_fallback(
                    confirmed_segments,
                    max_attempts=max(4, final_global_rematch_attempts * 2),
                )
                if final_global_rematched > 0:
                    no_target_micro_gap_snapped += int(final_global_rematched)
                    print(f"   🎯 终态全局重匹配: 收敛 {final_global_rematched} 个未收敛边界候选")
                    final_global_hard = self._enforce_boundary_hard_constraints_no_target(
                        confirmed_segments,
                        max_passes=2,
                    )
                    if final_global_hard > 0:
                        no_target_micro_gap_snapped += int(final_global_hard)
                        no_target_boundary_hard_clamped += int(final_global_hard)
                    final_global_audio_repaired, final_global_audio_issues = self._repair_boundary_audio_locally_no_target(
                        confirmed_segments,
                        max_passes=1,
                    )
                    if final_global_audio_repaired > 0:
                        no_target_micro_gap_snapped += int(final_global_audio_repaired)
                    no_target_boundary_audio_detected += int(final_global_audio_issues)
                    no_target_boundary_audio_repaired += int(final_global_audio_repaired)
                    no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

            secondary_replaced = self._replace_secondary_source_segments_no_target(confirmed_segments, max_passes=2)
            if secondary_replaced > 0:
                no_target_isolated_repaired += int(secondary_replaced)
                print(f"   🧹 次级源终态替换: 收回 {secondary_replaced} 个残留次级源段")
                secondary_hard = self._enforce_boundary_hard_constraints_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                if secondary_hard > 0:
                    no_target_micro_gap_snapped += int(secondary_hard)
                    no_target_boundary_hard_clamped += int(secondary_hard)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

            last_resort_stabilized = self._stabilize_last_resort_segments_no_target(
                confirmed_segments,
                max_passes=2,
            )
            if last_resort_stabilized > 0:
                no_target_micro_gap_snapped += int(last_resort_stabilized)
                print(f"   🎯 末段连续性收口: 调整 {last_resort_stabilized} 个高风险尾段")
                stabilize_hard = self._enforce_boundary_hard_constraints_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                stabilize_audio_repaired, stabilize_audio_issues = self._repair_boundary_audio_locally_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                if stabilize_hard > 0:
                    no_target_micro_gap_snapped += int(stabilize_hard)
                    no_target_boundary_hard_clamped += int(stabilize_hard)
                if stabilize_audio_repaired > 0:
                    no_target_micro_gap_snapped += int(stabilize_audio_repaired)
                no_target_boundary_audio_detected += int(stabilize_audio_issues)
                no_target_boundary_audio_repaired += int(stabilize_audio_repaired)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

            final_neighbor_source_repaired = self._repair_isolated_source_switches_with_neighbor_source_no_target(
                confirmed_segments,
                max_passes=2,
            )
            if final_neighbor_source_repaired > 0:
                no_target_isolated_repaired += int(final_neighbor_source_repaired)
                print(f"   🧩 终态孤立错源回收: 收回 {final_neighbor_source_repaired} 个错位段")
                final_neighbor_hard = self._enforce_boundary_hard_constraints_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                final_neighbor_audio_repaired, final_neighbor_audio_issues = self._repair_boundary_audio_locally_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                if final_neighbor_hard > 0:
                    no_target_micro_gap_snapped += int(final_neighbor_hard)
                    no_target_boundary_hard_clamped += int(final_neighbor_hard)
                if final_neighbor_audio_repaired > 0:
                    no_target_micro_gap_snapped += int(final_neighbor_audio_repaired)
                no_target_boundary_audio_detected += int(final_neighbor_audio_issues)
                no_target_boundary_audio_repaired += int(final_neighbor_audio_repaired)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

            bridge_run_realigned = self._realign_bridge_affected_same_source_runs_by_audio_no_target(
                confirmed_segments,
                max_passes=2,
            )
            if bridge_run_realigned > 0:
                no_target_micro_gap_snapped += int(bridge_run_realigned)
                print(f"   🎵 同源链前拉修复: 调整 {bridge_run_realigned} 个段，修复句首缺音/节奏拖晚")
                bridge_run_hard = self._enforce_boundary_hard_constraints_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                bridge_run_audio_repaired, bridge_run_audio_issues = self._repair_boundary_audio_locally_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                if bridge_run_hard > 0:
                    no_target_micro_gap_snapped += int(bridge_run_hard)
                    no_target_boundary_hard_clamped += int(bridge_run_hard)
                if bridge_run_audio_repaired > 0:
                    no_target_micro_gap_snapped += int(bridge_run_audio_repaired)
                no_target_boundary_audio_detected += int(bridge_run_audio_issues)
                no_target_boundary_audio_repaired += int(bridge_run_audio_repaired)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

            head_aligned_run_snapped = self._snap_head_anchored_same_source_runs_no_target(
                confirmed_segments,
                max_passes=1,
            )
            if head_aligned_run_snapped > 0:
                no_target_micro_gap_snapped += int(head_aligned_run_snapped)
                print(f"   🎯 同源片头链回零: 调整 {head_aligned_run_snapped} 个段，修复句首占位无声")
                head_snap_hard = self._enforce_boundary_hard_constraints_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                head_snap_audio_repaired, head_snap_audio_issues = self._repair_boundary_audio_locally_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                if head_snap_hard > 0:
                    no_target_micro_gap_snapped += int(head_snap_hard)
                    no_target_boundary_hard_clamped += int(head_snap_hard)
                if head_snap_audio_repaired > 0:
                    no_target_micro_gap_snapped += int(head_snap_audio_repaired)
                no_target_boundary_audio_detected += int(head_snap_audio_issues)
                no_target_boundary_audio_repaired += int(head_snap_audio_repaired)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

            cross_source_shortfall_bridged = self._prepare_cross_source_shortfall_bridges_no_target(
                confirmed_segments
            )
            if cross_source_shortfall_bridged > 0:
                no_target_micro_gap_snapped += int(cross_source_shortfall_bridged)
                print(f"   🌉 跨源短缺拼桥: 修复 {cross_source_shortfall_bridged} 处尾接片头断裂")

            prev_tail_carryover_repaired = self._prepare_cross_source_prev_tail_carryovers_no_target(
                confirmed_segments
            )
            if prev_tail_carryover_repaired > 0:
                no_target_micro_gap_snapped += int(prev_tail_carryover_repaired)
                print(f"   🧵 前源尾巴接入: 修复 {prev_tail_carryover_repaired} 处跨源丢内容边界")
                carryover_rebalanced = self._rebalance_neighbor_recovered_segments_no_target(
                    confirmed_segments
                )
                if carryover_rebalanced > 0:
                    no_target_micro_gap_snapped += int(carryover_rebalanced)
                    print(f"   🎯 前源接入后重平衡: 调整 {carryover_rebalanced} 个末段恢复边界")
                carryover_overlap_clamped = self._clamp_carryover_shifted_overlap_boundaries_no_target(
                    confirmed_segments
                )
                if carryover_overlap_clamped > 0:
                    no_target_micro_gap_snapped += int(carryover_overlap_clamped)
                    print(f"   📎 前源接入后重叠收口: 调整 {carryover_overlap_clamped} 个边界")
                carryover_prev_trimmed = self._trim_prev_for_carryover_shifted_overlaps_no_target(
                    confirmed_segments
                )
                if carryover_prev_trimmed > 0:
                    no_target_micro_gap_snapped += int(carryover_prev_trimmed)
                    print(f"   ✂️ 前源接入后裁前段尾巴: 调整 {carryover_prev_trimmed} 个边界")
                carryover_audio_repaired, carryover_audio_issues = self._repair_boundary_audio_locally_no_target(
                    confirmed_segments,
                    max_passes=1,
                )
                if carryover_audio_repaired > 0:
                    no_target_micro_gap_snapped += int(carryover_audio_repaired)
                no_target_boundary_audio_detected += int(carryover_audio_issues)
                no_target_boundary_audio_repaired += int(carryover_audio_repaired)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

                # 前源尾巴接入/拼桥后，可能重新制造“前后同源、中间单段异源”的孤立错源段。
                # 这里补一轮最终邻居同源回收，避免像 195/200/205s 这种边界在最后一步又插回错误源。
                post_carryover_neighbor_source_repaired = self._repair_isolated_source_switches_with_neighbor_source_no_target(
                    confirmed_segments,
                    max_passes=2,
                )
                post_carryover_hard = 0
                post_carryover_audio_repaired = 0
                post_carryover_audio_issues = 0
                if post_carryover_neighbor_source_repaired > 0:
                    no_target_isolated_repaired += int(post_carryover_neighbor_source_repaired)
                    print(
                        f"   🧩 前源接入后孤立错源回收: 收回 {post_carryover_neighbor_source_repaired} 个错位段"
                    )
                    post_carryover_hard = self._enforce_boundary_hard_constraints_no_target(
                        confirmed_segments,
                        max_passes=1,
                    )
                    post_carryover_audio_repaired, post_carryover_audio_issues = self._repair_boundary_audio_locally_no_target(
                        confirmed_segments,
                        max_passes=1,
                    )
                    if post_carryover_hard > 0:
                        no_target_micro_gap_snapped += int(post_carryover_hard)
                        no_target_boundary_hard_clamped += int(post_carryover_hard)
                if post_carryover_audio_repaired > 0:
                    no_target_micro_gap_snapped += int(post_carryover_audio_repaired)
                no_target_boundary_audio_detected += int(post_carryover_audio_issues)
                no_target_boundary_audio_repaired += int(post_carryover_audio_repaired)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)

            # 终态收尾：前面的音频修边/拼桥/同源回收可能再次引入同源深重叠，
            # 这里最后再压一次同源 run，避免 3:22 这类“内容已回正，但边界又开始重复”。
            final_same_source_overlap_fixed = self._resolve_severe_same_source_overlaps_no_target(
                confirmed_segments,
                max_passes=1,
            )
            final_head_snapped = self._snap_head_anchored_same_source_runs_no_target(
                confirmed_segments,
                max_passes=1,
            )
            final_run_right_shifted = self._shift_same_source_runs_right_to_clear_overlap_no_target(
                confirmed_segments,
                max_passes=1,
            )
            final_prev_trimmed = self._trim_prev_for_unresolved_same_source_overlap_no_target(
                confirmed_segments,
                max_passes=1,
            )
            final_small_overlap_suppressed = self._suppress_small_negative_overlaps_no_target(
                confirmed_segments,
                max_passes=1,
            )
            final_terminal_hard = self._enforce_boundary_hard_constraints_no_target(
                confirmed_segments,
                max_passes=1,
            )
            final_tail_shortfall_trimmed = self._trim_tail_overlaps_with_shortfall_tolerance_no_target(
                confirmed_segments,
            )
            final_terminal_adjusted = int(
                final_same_source_overlap_fixed
                + final_head_snapped
                + final_run_right_shifted
                + final_prev_trimmed
                + final_small_overlap_suppressed
                + final_terminal_hard
                + final_tail_shortfall_trimmed
            )
            if final_terminal_adjusted > 0:
                no_target_micro_gap_snapped += int(final_terminal_adjusted)
                no_target_boundary_hard_clamped += int(final_terminal_hard)
                no_target_boundary_unresolved = self._count_no_target_boundary_unresolved(confirmed_segments)
                print(f"   🧹 终态同源重叠收尾: 调整 {final_terminal_adjusted} 个边界")

        self.timeline_guard_elapsed_sec = time.perf_counter() - guard_perf
        self.guard_stats = {
            "missing_filled": int(missing_count),
            "overlap_adjusted": int(overlap_adjusted),
            "overlap_fallback": int(overlap_fallback),
            "global_repeat_fallback": int(global_repeat_fallback),
            "step_guard_fallback": int(step_guard_fallback),
            "alignment_bias_fallback": int(alignment_bias_fallback),
            "bridge_recovered": int(bridge_recovered),
            "no_target_isolated_repaired": int(no_target_isolated_repaired),
            "no_target_micro_gap_snapped": int(no_target_micro_gap_snapped),
            "no_target_boundary_unresolved": int(no_target_boundary_unresolved),
            "no_target_boundary_hard_clamped": int(no_target_boundary_hard_clamped),
            "no_target_boundary_audio_detected": int(no_target_boundary_audio_detected),
            "no_target_boundary_audio_repaired": int(no_target_boundary_audio_repaired),
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

        boundary_details = []
        same_source_boundaries = 0
        negative_overlap_boundaries = 0
        positive_gap_boundaries = 0
        boundary_hard_clamped_count = 0
        boundary_audio_issue_count = 0
        boundary_audio_repaired_count = 0
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            same_source = bool(prev.get("source") == curr.get("source"))
            gap_sec = None
            if same_source:
                same_source_boundaries += 1
                gap_sec = float(curr["start"]) - (float(prev["start"]) + float(prev["duration"]))
                if gap_sec < 0.0:
                    negative_overlap_boundaries += 1
                elif gap_sec > 0.0:
                    positive_gap_boundaries += 1

            q = curr.get("quality", {}) or {}
            hard_clamped = bool(q.get("boundary_hard_clamped_no_target", False))
            audio_issue = bool(q.get("boundary_audio_issue_detected_no_target", False))
            audio_repaired = bool(q.get("boundary_audio_repaired_no_target", False))
            if hard_clamped:
                boundary_hard_clamped_count += 1
            if audio_issue:
                boundary_audio_issue_count += 1
            if audio_repaired:
                boundary_audio_repaired_count += 1

            boundary_details.append(
                {
                    "boundary_index": int(i),
                    "target_second": float(curr["target_start"]),
                    "prev_segment_index": int(prev["index"]),
                    "curr_segment_index": int(curr["index"]),
                    "same_source": bool(same_source),
                    "source": str(curr["source"]) if same_source else "",
                    "gap_sec": gap_sec,
                    "hard_clamped": hard_clamped,
                    "audio_issue_detected": audio_issue,
                    "audio_repaired": audio_repaired,
                    "before_gap_sec": q.get(
                        "boundary_audio_repair_before_gap",
                        q.get("boundary_hard_before_gap", q.get("boundary_audio_issue_gap_before")),
                    ),
                    "after_gap_sec": q.get(
                        "boundary_audio_repair_after_gap",
                        q.get("boundary_hard_after_gap"),
                    ),
                    "repair_from_start": q.get(
                        "boundary_audio_repair_from",
                        q.get("boundary_hard_from"),
                    ),
                    "repair_to_start": q.get(
                        "boundary_audio_repair_to",
                        q.get("boundary_hard_to"),
                    ),
                    "audio_similarity_before": q.get("boundary_audio_similarity_before"),
                    "audio_similarity_anchor": q.get("boundary_audio_similarity_anchor"),
                    "audio_similarity_after": q.get("boundary_audio_similarity_after"),
                    "silence_mismatch_before": q.get("boundary_audio_silence_mismatch_before"),
                    "silence_mismatch_after": q.get("boundary_audio_silence_mismatch_after"),
                }
            )

        boundary_summary = {
            "total_boundaries": max(0, len(segments) - 1),
            "same_source_boundaries": int(same_source_boundaries),
            "negative_overlap_boundaries": int(negative_overlap_boundaries),
            "positive_gap_boundaries": int(positive_gap_boundaries),
            "hard_clamped_boundaries": int(boundary_hard_clamped_count),
            "audio_issue_boundaries": int(boundary_audio_issue_count),
            "audio_repaired_boundaries": int(boundary_audio_repaired_count),
        }

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
            "boundary_summary": to_jsonable(boundary_summary),
            "boundary_details": to_jsonable(boundary_details),
            "segments": segment_details,
        }

        report_path = Path(output_path).with_suffix(".quality_report.json")
        # 先删除旧报告，再写同名新文件，确保文件创建时间反映本次运行。
        report_path.unlink(missing_ok=True)
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
            if not self.enable_target_video_fallback:
                q = seg.get("quality", {}) or {}
                q["fallback_blocked"] = True
                q["fallback_reason"] = f"{reason}_no_target_fallback"
                q["timeline_guard_metric"] = float(metric)
                seg["quality"] = q
                return
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
            if not self.enable_target_video_fallback:
                q = seg.get("quality", {}) or {}
                q["fallback_blocked"] = True
                q["fallback_reason"] = f"{reason}_no_target_fallback"
                seg["quality"] = q
                return
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
            if not self.enable_target_video_fallback:
                q = seg.get("quality", {}) or {}
                q["fallback_blocked"] = True
                q["fallback_reason"] = f"{reason}_no_target_fallback"
                q["timeline_bias_metric"] = float(metric)
                seg["quality"] = q
                return
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
            if not self.enable_target_video_fallback:
                q = seg.get("quality", {}) or {}
                q["fallback_blocked"] = True
                q["fallback_reason"] = f"{reason}_no_target_fallback"
                q["timeline_step_metric"] = float(metric)
                seg["quality"] = q
                return
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

    def recover_isolated_fallback_bridges(self, segments: List[dict]) -> int:
        """
        孤立兜底桥接恢复：
        - 仅处理被“前后同源高置信段”夹住的单个 fallback 段
        - 优先按前后连续性推导候选起点，再做 quick_verify 复核
        - 通过后恢复为同源段，降低“中间 5 秒像卡住/跳源”的观感风险
        """
        recovered = 0
        if not self.enable_target_video_fallback:
            return recovered
        if len(segments) < 3:
            return recovered

        for i in range(1, len(segments) - 1):
            prev = segments[i - 1]
            curr = segments[i]
            nxt = segments[i + 1]

            curr_q = curr.get("quality", {}) or {}
            if curr["source"] != self.target_video:
                continue
            if not bool(curr_q.get("fallback", False)):
                continue
            if float(curr.get("target_start", 0.0)) < float(self.bridge_recover_min_target_start):
                continue

            # 仅恢复“原始匹配失败型”兜底，避免覆盖 timeline_guard 的主动保护决策。
            fallback_reason = str(curr_q.get("fallback_reason", "") or "")
            if fallback_reason not in {"", "match_failed_low_score", "missing_result"}:
                continue

            if prev["source"] == self.target_video or nxt["source"] == self.target_video:
                continue
            if prev["source"] != nxt["source"]:
                continue

            prev_q = prev.get("quality", {}) or {}
            next_q = nxt.get("quality", {}) or {}
            if bool(prev_q.get("fallback", False)) or bool(next_q.get("fallback", False)):
                continue

            quality_floor = max(0.82, float(self.strict_verify_min_sim))
            if float(prev_q.get("combined", 0.0)) < quality_floor:
                continue
            if float(next_q.get("combined", 0.0)) < quality_floor:
                continue

            dur = float(curr["duration"])
            cand_from_prev = float(prev["start"]) + float(prev["duration"])
            cand_from_next = float(nxt["start"]) - dur
            bridge_gap = abs(cand_from_prev - cand_from_next)
            bridge_tol = max(0.45, dur * 0.12)
            if bridge_gap > bridge_tol:
                continue

            candidate = 0.5 * (cand_from_prev + cand_from_next)
            if candidate < 0.0:
                continue

            step_tolerance = max(0.9, dur * 0.30)
            target_step = float(curr["target_start"]) - float(prev["target_start"])
            source_step = float(candidate) - float(prev["start"])
            if abs(source_step - target_step) > step_tolerance:
                continue

            passed, verify_avg = self.quick_verify(
                source=prev["source"],
                source_start=float(candidate),
                target_start=float(curr["target_start"]),
                duration=dur,
            )
            if not passed:
                continue

            if self.bridge_motion_guard_enabled:
                target_motion = self.estimate_segment_motion(
                    self.target_video,
                    float(curr["target_start"]),
                    dur,
                    sample_count=self.bridge_motion_samples,
                )
                source_motion = self.estimate_segment_motion(
                    prev["source"],
                    float(candidate),
                    dur,
                    sample_count=self.bridge_motion_samples,
                )
                curr_q["bridge_target_motion"] = None if target_motion is None else float(target_motion)
                curr_q["bridge_source_motion"] = None if source_motion is None else float(source_motion)
                if (
                    target_motion is not None
                    and source_motion is not None
                    and target_motion >= float(self.bridge_motion_min_target_motion)
                    and target_motion > 1e-6
                ):
                    ratio = float(source_motion / target_motion)
                    curr_q["bridge_motion_ratio"] = ratio
                    if ratio < float(self.bridge_motion_min_ratio):
                        curr_q["bridge_recover_blocked"] = "motion_guard_source_too_static"
                        curr["quality"] = curr_q
                        continue

            curr["source"] = prev["source"]
            curr["start"] = float(candidate)
            curr_q["fallback"] = False
            curr_q["fallback_reason"] = "bridge_recovered_from_neighbors"
            curr_q["bridge_recovered"] = True
            curr_q["bridge_verify_avg"] = float(verify_avg)
            curr_q["bridge_gap_sec"] = float(bridge_gap)
            curr_q["combined"] = max(float(curr_q.get("combined", 0.0)), float(verify_avg))
            curr["quality"] = curr_q
            recovered += 1

        return recovered

    def repair_isolated_outliers_without_target_fallback(self, segments: List[dict]) -> int:
        """
        禁用目标兜底时的孤立异常段修复：
        - 前后同源且非目标时，修复中间“跨源跳点”或“单段时间轴异常”。
        - 通过前后插值推导候选起点，并做 visual/audio 复核后再落盘。
        """
        repaired = 0
        if self.enable_target_video_fallback:
            return repaired
        if len(segments) < 3:
            return repaired

        for i in range(1, len(segments) - 1):
            prev = segments[i - 1]
            curr = segments[i]
            nxt = segments[i + 1]

            if prev["source"] == self.target_video or nxt["source"] == self.target_video:
                continue
            if prev["source"] != nxt["source"]:
                continue

            prev_q = prev.get("quality", {}) or {}
            next_q = nxt.get("quality", {}) or {}
            if bool(prev_q.get("fallback", False)) or bool(next_q.get("fallback", False)):
                continue

            prev_t = float(prev["target_start"])
            curr_t = float(curr["target_start"])
            next_t = float(nxt["target_start"])
            if next_t <= prev_t + 1e-6:
                continue

            bridge_source = prev["source"]
            if self._is_secondary_source(Path(bridge_source)):
                continue
            duration = float(curr["duration"])
            source_duration = self.get_video_duration(bridge_source)
            if source_duration <= 0.0:
                continue
            max_start = max(0.0, float(source_duration - duration))

            prev_start = float(prev["start"])
            next_start = float(nxt["start"])
            ratio = (curr_t - prev_t) / (next_t - prev_t)
            ratio = max(0.0, min(1.0, ratio))
            expected = prev_start + (next_start - prev_start) * ratio
            curr_start = float(curr["start"])

            outlier_tolerance = max(0.35, duration * 0.10)
            should_repair = (
                curr["source"] != bridge_source
                or abs(curr_start - expected) > outlier_tolerance
            )
            if not should_repair:
                continue

            quality_floor = max(0.72, float(self.strict_verify_min_sim) - 0.06)
            if float(prev_q.get("combined", 1.0)) < quality_floor:
                continue
            if float(next_q.get("combined", 1.0)) < quality_floor:
                continue

            candidates = [
                expected,
                prev_start + (curr_t - prev_t),
                next_start - (next_t - curr_t),
            ]
            if curr["source"] == bridge_source:
                candidates.append(curr_start)

            best_choice = None
            seen_starts: Set[float] = set()
            for raw_start in candidates:
                start = max(0.0, min(float(raw_start), max_start))
                key = round(start, 3)
                if key in seen_starts:
                    continue
                seen_starts.add(key)

                refined_start, refine_score = self.refine_start_by_visual(
                    source=bridge_source,
                    initial_start=start,
                    target_start=curr_t,
                    duration=duration,
                )
                refined_start = max(0.0, min(float(refined_start), max_start))

                # 尾部容量守卫（禁兜底）：
                # 若把当前段并入 bridge_source 后，后续同源链到源尾的可用时长明显不足，
                # 将不可避免引入连续压缩重叠（重复词/节奏错位），直接放弃该候选。
                run_end = i
                scan = i + 1
                while scan < len(segments) and segments[scan]["source"] == bridge_source:
                    run_end = scan
                    scan += 1
                if run_end > i:
                    run_target_end = float(segments[run_end]["target_start"]) + float(segments[run_end]["duration"])
                    required_span = max(0.0, float(run_target_end - curr_t))
                    available_span = max(0.0, float(source_duration - refined_start))
                    tail_margin = max(0.35, required_span * 0.03)
                    if available_span + tail_margin < required_span:
                        continue

                verify_passed, verify_avg = self.quick_verify(
                    source=bridge_source,
                    source_start=refined_start,
                    target_start=curr_t,
                    duration=duration,
                )
                if not verify_passed:
                    continue

                audio_passed, audio_meta = self.quick_verify_audio(
                    source=bridge_source,
                    source_start=refined_start,
                    target_start=curr_t,
                    duration=duration,
                    combined_score=float(verify_avg),
                )
                if not audio_passed:
                    continue

                rank_score = float(verify_avg) + 0.01 * float(refine_score)
                if best_choice is None or rank_score > best_choice["rank"]:
                    best_choice = {
                        "start": float(refined_start),
                        "verify_avg": float(verify_avg),
                        "refine_score": float(refine_score),
                        "audio_meta": audio_meta,
                        "rank": float(rank_score),
                    }

            if best_choice is None:
                continue

            old_source = str(curr["source"])
            curr["source"] = bridge_source
            curr["start"] = float(best_choice["start"])
            curr_q = curr.get("quality", {}) or {}
            curr_q["combined"] = max(float(curr_q.get("combined", 0.0)), float(best_choice["verify_avg"]))
            curr_q["repair_isolated_outlier_no_target"] = True
            curr_q["repair_mode"] = "isolated_bridge_no_target"
            curr_q["repair_from_source"] = old_source
            curr_q["repair_verify_avg"] = float(best_choice["verify_avg"])
            curr_q["repair_refine_score"] = float(best_choice["refine_score"])
            curr_q["audio_guard"] = best_choice["audio_meta"]
            curr_q.pop("fallback_blocked", None)
            if str(curr_q.get("fallback_reason", "")).endswith("_no_target_fallback"):
                curr_q["fallback_reason"] = "repaired_no_target_bridge"
            curr["quality"] = curr_q
            repaired += 1

        return repaired

    def _repair_isolated_source_switches_with_neighbor_source_no_target(
        self,
        segments: List[dict],
        max_passes: int = 2,
    ) -> int:
        """
        邻居同源优先修复（禁兜底）：
        - 仅处理 prev/next 同源、current 异源的孤立切源段；
        - 强制评估“邻居同源候选”，若 visual/audio 复核不劣于当前段则切回同源；
        - 用于修复“中间 5 秒错源”导致的内容错位/台词截断。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 3:
            return 0

        repaired = 0
        for _ in range(max(1, int(max_passes))):
            changed = False
            for i in range(1, len(segments) - 1):
                prev = segments[i - 1]
                curr = segments[i]
                nxt = segments[i + 1]

                if prev["source"] == self.target_video or nxt["source"] == self.target_video:
                    continue
                if prev["source"] != nxt["source"]:
                    continue
                bridge_source = prev["source"]
                if self._is_secondary_source(Path(bridge_source)):
                    continue
                if curr["source"] == bridge_source:
                    continue

                duration = float(curr["duration"])
                source_duration = self.get_video_duration(bridge_source)
                if source_duration <= 0.0:
                    continue
                max_start = max(0.0, float(source_duration - duration))

                prev_t = float(prev["target_start"])
                curr_t = float(curr["target_start"])
                next_t = float(nxt["target_start"])
                if next_t <= prev_t + 1e-6:
                    continue
                ratio = max(0.0, min(1.0, (curr_t - prev_t) / (next_t - prev_t)))

                expected = float(prev["start"]) + (float(nxt["start"]) - float(prev["start"])) * ratio
                raw_candidates = [
                    expected,
                    float(prev["start"]) + (curr_t - prev_t),
                    float(nxt["start"]) - (next_t - curr_t),
                    expected - 0.25,
                    expected + 0.25,
                ]
                candidates: List[float] = []
                seen: Set[float] = set()
                for raw in raw_candidates:
                    cand = max(0.0, min(float(raw), max_start))
                    key = round(cand, 3)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(float(cand))

                curr_q = curr.get("quality", {}) or {}
                baseline_avg = float(curr_q.get("combined", 0.0))
                best_choice: Optional[Dict[str, object]] = None
                best_rank_score: Optional[float] = None
                prev_duration = float(prev["duration"])
                min_prev_duration = max(1.5, min(3.0, prev_duration * 0.60))
                prev_trim_cap = max(0.0, float(prev_duration - min_prev_duration))

                for cand_start in candidates:
                    refined_start, refine_score = self.refine_start_by_visual(
                        source=bridge_source,
                        initial_start=float(cand_start),
                        target_start=float(curr_t),
                        duration=float(duration),
                    )
                    refined_start = max(0.0, min(float(refined_start), max_start))
                    # 邻居同源修复时，约束视觉细调偏移，避免被重复画面“拉飞”到错误时间带。
                    max_refine_dev = max(0.6, float(duration) * 0.18)
                    if abs(float(refined_start) - float(cand_start)) > max_refine_dev:
                        refined_start = float(cand_start)
                    probe_starts: List[Tuple[float, float]] = [(float(cand_start), 0.0)]
                    if abs(float(refined_start) - float(cand_start)) > 1e-6:
                        probe_starts.append((float(refined_start), float(refine_score)))

                    seen_probe: Set[float] = set()
                    for probe_start, probe_refine_score in probe_starts:
                        probe_key = round(float(probe_start), 3)
                        if probe_key in seen_probe:
                            continue
                        seen_probe.add(probe_key)

                        passed, avg = self.quick_verify(
                            source=bridge_source,
                            source_start=float(probe_start),
                            target_start=float(curr_t),
                            duration=float(duration),
                        )
                        if not passed:
                            continue

                        audio_passed, audio_meta = self.quick_verify_audio(
                            source=bridge_source,
                            source_start=float(probe_start),
                            target_start=float(curr_t),
                            duration=float(duration),
                            combined_score=float(avg),
                        )
                        if not audio_passed:
                            continue

                        distance_penalty = abs(float(probe_start) - float(expected))
                        prev_end = float(prev["start"]) + float(prev["duration"])
                        next_start = float(nxt["start"])
                        prev_gap = float(probe_start - prev_end)
                        next_gap = float(next_start - (float(probe_start) + float(duration)))
                        continuity_penalty = abs(float(prev_gap)) + abs(float(next_gap))
                        allow_prev_trim = False
                        prev_trim_sec = 0.0
                        # 这类修复的目标是“纠正孤立错源”，不是为了收边界把正确内容换坏。
                        # 若邻居同源候选本身连续性代价过大，直接拒绝，避免重演 100s 段被换坏的问题。
                        max_distance_penalty = max(0.35, float(duration) * 0.08)
                        max_continuity_penalty = max(0.55, float(duration) * 0.12)
                        if float(distance_penalty) > float(max_distance_penalty):
                            continue
                        if float(continuity_penalty) > float(max_continuity_penalty):
                            overlap = max(0.0, -float(prev_gap))
                            next_neg_tol = max(0.12, float(duration) * 0.02)
                            if (
                                overlap > 1e-6
                                and overlap <= (float(prev_trim_cap) + 1e-6)
                                and overlap <= max(0.45, float(duration) * 0.24)
                                and float(next_gap) >= (-float(next_neg_tol) - 1e-6)
                                and float(distance_penalty) <= max(float(max_distance_penalty), 0.65)
                            ):
                                allow_prev_trim = True
                                prev_trim_sec = float(overlap)
                                continuity_penalty = abs(float(next_gap))
                            else:
                                continue
                        rank_score = (
                            float(avg)
                            - float(distance_penalty) * 0.04
                            - float(continuity_penalty) * 0.12
                        )
                        if best_rank_score is None or rank_score > best_rank_score:
                            best_rank_score = float(rank_score)
                            best_choice = {
                                "start": float(probe_start),
                                "verify_avg": float(avg),
                                "refine_score": float(probe_refine_score),
                                "audio_meta": audio_meta,
                                "distance_penalty": float(distance_penalty),
                                "continuity_penalty": float(continuity_penalty),
                                "prev_gap": float(prev_gap),
                                "next_gap": float(next_gap),
                                "allow_prev_trim": bool(allow_prev_trim),
                                "prev_trim_sec": float(prev_trim_sec),
                            }

                if best_choice is None:
                    continue

                # 连续性优先：允许轻微分数下降换取同源连贯，但不能明显劣化。
                if float(best_choice["verify_avg"]) + 1e-6 < (baseline_avg - 0.03):
                    continue

                old_source = str(curr["source"])
                old_start = float(curr["start"])
                prev_trim_sec = float(best_choice.get("prev_trim_sec", 0.0) or 0.0)
                if prev_trim_sec > 1e-6:
                    prev["duration"] = max(0.0, float(prev["duration"]) - float(prev_trim_sec))
                    prev_q = prev.get("quality", {}) or {}
                    prev_q["neighbor_source_prev_trim_no_target"] = True
                    prev_q["neighbor_source_prev_trim_sec"] = float(prev_trim_sec)
                    prev_q["neighbor_source_prev_trim_boundary_index"] = int(i)
                    prev_q["neighbor_source_prev_trim_to_source"] = str(bridge_source)
                    prev["quality"] = prev_q
                curr["source"] = bridge_source
                curr["start"] = float(best_choice["start"])
                q = curr.get("quality", {}) or {}
                q["combined"] = max(float(q.get("combined", 0.0)), float(best_choice["verify_avg"]))
                q["repair_isolated_neighbor_source_no_target"] = True
                q["repair_mode"] = "isolated_neighbor_source_no_target"
                q["repair_from_source"] = old_source
                q["repair_from_start"] = float(old_start)
                q["repair_verify_avg"] = float(best_choice["verify_avg"])
                q["repair_refine_score"] = float(best_choice["refine_score"])
                q["repair_expected_distance"] = float(best_choice["distance_penalty"])
                q["repair_neighbor_continuity_penalty"] = float(best_choice.get("continuity_penalty", 0.0))
                q["repair_neighbor_prev_gap"] = float(best_choice.get("prev_gap", 0.0))
                q["repair_neighbor_next_gap"] = float(best_choice.get("next_gap", 0.0))
                q["repair_neighbor_prev_trim_sec"] = float(prev_trim_sec)
                q["audio_guard"] = best_choice["audio_meta"]
                curr["quality"] = q
                repaired += 1
                changed = True

            if not changed:
                break

        return int(repaired)

    def _replace_secondary_source_segments_no_target(
        self,
        segments: List[dict],
        max_passes: int = 2,
    ) -> int:
        """
        兼容旧分支保留的终态替换入口。
        当前源目录下视频一律视为同级原视频，因此这里通常直接短路返回。
        """
        if self.enable_target_video_fallback:
            return 0
        if not self.secondary_source_videos:
            return 0
        if len(segments) == 0:
            return 0

        replaced = 0
        for _ in range(max(1, int(max_passes))):
            changed = False
            for i, seg in enumerate(segments):
                source = seg.get("source")
                if source in (None, self.target_video):
                    continue
                if not self._is_secondary_source(Path(source)):
                    continue

                duration = float(seg["duration"])
                target_start = float(seg["target_start"])
                old_combined = float((seg.get("quality", {}) or {}).get("combined", 0.0))
                old_source = str(source)
                old_start = float(seg["start"])
                candidates: List[Tuple[Path, float, str]] = []

                prev = segments[i - 1] if i > 0 else None
                nxt = segments[i + 1] if (i + 1) < len(segments) else None
                if prev is not None and prev.get("source") not in (None, self.target_video):
                    prev_src = Path(prev["source"])
                    if not self._is_secondary_source(prev_src):
                        est = float(prev["start"]) + (target_start - float(prev["target_start"]))
                        candidates.append((prev_src, float(est), "neighbor_prev"))
                if nxt is not None and nxt.get("source") not in (None, self.target_video):
                    next_src = Path(nxt["source"])
                    if not self._is_secondary_source(next_src):
                        est = float(nxt["start"]) - (float(nxt["target_start"]) - target_start)
                        candidates.append((next_src, float(est), "neighbor_next"))
                if (
                    prev is not None
                    and nxt is not None
                    and prev.get("source") not in (None, self.target_video)
                    and prev.get("source") == nxt.get("source")
                ):
                    bridge_src = Path(prev["source"])
                    if not self._is_secondary_source(bridge_src):
                        prev_t = float(prev["target_start"])
                        next_t = float(nxt["target_start"])
                        if next_t > prev_t + 1e-6:
                            ratio = max(0.0, min(1.0, (target_start - prev_t) / (next_t - prev_t)))
                            interp = float(prev["start"]) + (float(nxt["start"]) - float(prev["start"])) * ratio
                            candidates.append((bridge_src, float(interp), "neighbor_interp"))

                phash_candidates = self.find_match_by_phash(
                    target_start=target_start,
                    duration=duration,
                    seg_index=int(seg["index"]),
                    top_k=64,
                )
                for cand_source, cand_start, _phash in phash_candidates:
                    cand_path = Path(cand_source)
                    if cand_path == self.target_video or self._is_secondary_source(cand_path):
                        continue
                    candidates.append((cand_path, float(cand_start), "primary_phash"))

                best_choice: Optional[Dict[str, object]] = None
                seen: Set[Tuple[str, float]] = set()
                for cand_source, raw_start, mode in candidates:
                    sig = (str(cand_source), round(float(raw_start), 3))
                    if sig in seen:
                        continue
                    seen.add(sig)

                    src_duration = self.get_video_duration(cand_source)
                    if src_duration <= 0.0:
                        continue
                    max_start = max(0.0, float(src_duration - duration))
                    cand_start = max(0.0, min(float(raw_start), max_start))
                    refined_start, refine_score = self.refine_start_by_visual(
                        source=cand_source,
                        initial_start=float(cand_start),
                        target_start=target_start,
                        duration=duration,
                    )
                    refined_start = max(0.0, min(float(refined_start), max_start))

                    verify_passed, verify_avg = self.quick_verify(
                        source=cand_source,
                        source_start=float(refined_start),
                        target_start=target_start,
                        duration=duration,
                    )
                    if not verify_passed:
                        continue
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=cand_source,
                        source_start=float(refined_start),
                        target_start=target_start,
                        duration=duration,
                        combined_score=float(verify_avg),
                    )
                    if not audio_passed:
                        continue

                    continuity_penalty = 0.0
                    same_source_bonus = 0.0
                    if prev is not None and prev.get("source") == cand_source:
                        prev_end = float(prev["start"]) + float(prev["duration"])
                        continuity_penalty += abs(float(refined_start) - prev_end)
                        same_source_bonus += 0.03
                    if nxt is not None and nxt.get("source") == cand_source:
                        next_gap = float(nxt["start"]) - (float(refined_start) + duration)
                        continuity_penalty += abs(float(next_gap))
                        same_source_bonus += 0.03

                    rank = float(verify_avg) + same_source_bonus - float(continuity_penalty) * 0.12
                    if best_choice is None or rank > float(best_choice["rank"]):
                        best_choice = {
                            "source": cand_source,
                            "start": float(refined_start),
                            "verify_avg": float(verify_avg),
                            "refine_score": float(refine_score),
                            "rank": float(rank),
                            "mode": str(mode),
                            "audio_meta": dict(audio_meta or {}),
                        }

                if best_choice is None:
                    continue
                if float(best_choice["verify_avg"]) + 1e-6 < max(0.86, old_combined - 0.05):
                    continue

                seg["source"] = best_choice["source"]
                seg["start"] = float(best_choice["start"])
                q = seg.get("quality", {}) or {}
                q["combined"] = max(float(q.get("combined", 0.0)), float(best_choice["verify_avg"]))
                q["secondary_source_replaced_no_target"] = True
                q["secondary_source_replaced_from"] = str(old_source)
                q["secondary_source_replaced_from_start"] = float(old_start)
                q["secondary_source_replaced_mode"] = str(best_choice["mode"])
                q["secondary_source_replaced_verify_avg"] = float(best_choice["verify_avg"])
                q["secondary_source_replaced_refine_score"] = float(best_choice["refine_score"])
                q["audio_guard"] = best_choice["audio_meta"]
                seg["quality"] = q
                replaced += 1
                changed = True

            if not changed:
                break

        return int(replaced)

    def _realign_bridge_affected_same_source_runs_by_audio_no_target(
        self,
        segments: List[dict],
        max_passes: int = 1,
    ) -> int:
        """
        桥接后同源链前拉（禁兜底）：
        - 针对“跨源桥接后，下一同源 run 整体拖晚”的场景；
        - 当当前段存在稳定的前拉收益时，允许整条同源 run 同步前拉；
        - 用裁掉前一段尾巴的方式消化重叠，优先恢复句首音频，而不是把后文继续压快。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 3:
            return 0

        adjusted = 0
        for _ in range(max(1, int(max_passes))):
            changed = False
            for i in range(1, len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    continue
                if prev["source"] != curr["source"]:
                    continue

                curr_q = curr.get("quality", {}) or {}
                if (i + 1) >= len(segments) or segments[i + 1]["source"] != curr["source"]:
                    continue

                duration = float(curr["duration"])
                base_start = float(curr["start"])
                target_start = float(curr["target_start"])
                base_avg = float(curr_q.get("combined", 0.0))
                base_audio_passed, base_audio_meta = self.quick_verify_audio(
                    source=curr["source"],
                    source_start=float(base_start),
                    target_start=float(target_start),
                    duration=float(duration),
                    combined_score=max(float(base_avg), 0.0),
                )
                if not isinstance(base_audio_meta, dict):
                    continue
                base_aligned = float(base_audio_meta.get("aligned_similarity", 0.0) or 0.0)
                base_shift = float(base_audio_meta.get("best_shift_sec", 0.0) or 0.0)
                base_gain = float(base_audio_meta.get("shift_gain", 0.0) or 0.0)
                if not (
                    (base_shift <= -0.45 and base_gain >= 0.03)
                    or (not base_audio_passed and str(base_audio_meta.get("reason", "")).startswith("audio_guard_shift_bias"))
                ):
                    continue

                prev_duration = float(prev["duration"])
                prev_q = prev.get("quality", {}) or {}
                min_prev_duration = max(1.5, min(3.0, prev_duration * 0.60))
                # 片头段/救援段本身就更容易丢词尾，避免为了前拉当前 run 把前一段裁得过深。
                prev_is_sensitive_head = bool(float(prev.get("start", 0.0)) <= 0.8)
                prev_is_sensitive_rescue = bool(
                    prev_q.get("rescue")
                    or prev_q.get("cross_source_shortfall_bridge_from_prev")
                    or prev_q.get("cross_source_prev_tail_carryover_no_target")
                )
                if prev_is_sensitive_head or prev_is_sensitive_rescue:
                    min_prev_duration = max(float(min_prev_duration), float(prev_duration) - 0.45)
                prev_audio_meta = prev_q.get("audio_guard", {}) or {}
                prev_aligned = float(prev_audio_meta.get("aligned_similarity", 0.0) or 0.0)
                prev_best_shift = float(prev_audio_meta.get("best_shift_sec", 0.0) or 0.0)
                prev_shift_gain = float(prev_audio_meta.get("shift_gain", 0.0) or 0.0)
                prev_verify_avg = float((prev_q.get("strict_verify", {}) or {}).get("avg", prev_q.get("combined", 0.0)) or 0.0)
                prev_is_sentence_tail_sensitive = bool(
                    prev_duration >= 4.6
                    and prev_aligned >= 0.74
                    and abs(prev_best_shift) <= 0.10
                    and prev_shift_gain <= 0.02
                    and prev_verify_avg >= 0.94
                    and not prev_is_sensitive_head
                    and not prev_is_sensitive_rescue
                )
                if prev_is_sentence_tail_sensitive:
                    # 这类前段已经稳定贴合当前台词，继续深裁通常会表现成“有口型/有占位，但中间缺字没出声”。
                    min_prev_duration = max(float(min_prev_duration), float(prev_duration) - 0.22)
                prev_trim_cap = max(0.0, float(prev_duration - min_prev_duration))
                if prev_trim_cap <= 1e-6:
                    continue

                run_end = i
                while (run_end + 1) < len(segments) and segments[run_end + 1]["source"] == curr["source"]:
                    run_end += 1

                raw_candidates: List[float] = [
                    float(base_start + base_shift),
                    float(base_start - 0.25),
                    float(base_start - 0.35),
                    float(base_start - 0.45),
                    float(base_start - 0.55),
                    float(base_start - 0.65),
                    float(base_start - 0.75),
                    float(base_start - 0.85),
                    float(base_start - 0.95),
                    float(base_start - 1.05),
                    float(base_start - 1.15),
                ]

                best_choice: Optional[Dict[str, object]] = None
                best_score: Optional[float] = None
                seen_candidates: Set[float] = set()
                for raw_start in raw_candidates:
                    cand_seed = max(0.0, min(float(raw_start), float(base_start - 0.12)))
                    key = round(cand_seed, 3)
                    if key in seen_candidates:
                        continue
                    seen_candidates.add(key)

                    refined_start, refine_score = self.refine_start_by_visual(
                        source=curr["source"],
                        initial_start=float(cand_seed),
                        target_start=float(target_start),
                        duration=float(duration),
                    )
                    cand_start = max(0.0, min(float(refined_start), float(base_start - 0.12)))
                    delta = float(base_start - cand_start)
                    if delta <= 0.12 or delta > 1.25:
                        continue

                    prev_end = float(prev["start"]) + float(prev_duration)
                    overlap = max(0.0, float(prev_end - cand_start))
                    if overlap > (float(prev_trim_cap) + 1e-6):
                        continue

                    probe_current_passed, probe_current_avg = self.quick_verify(
                        source=curr["source"],
                        source_start=float(cand_start),
                        target_start=float(target_start),
                        duration=float(duration),
                    )
                    if not probe_current_passed:
                        continue
                    probe_current_audio_passed, probe_current_audio_meta = self.quick_verify_audio(
                        source=curr["source"],
                        source_start=float(cand_start),
                        target_start=float(target_start),
                        duration=float(duration),
                        combined_score=float(probe_current_avg),
                    )
                    if not probe_current_audio_passed:
                        continue
                    curr_aligned = float((probe_current_audio_meta or {}).get("aligned_similarity", 0.0) or 0.0)
                    curr_best_shift = float((probe_current_audio_meta or {}).get("best_shift_sec", 0.0) or 0.0)
                    curr_shift_gain = float((probe_current_audio_meta or {}).get("shift_gain", 0.0) or 0.0)
                    if abs(curr_best_shift) >= 0.45 and curr_shift_gain >= 0.08:
                        continue
                    if curr_aligned + 0.06 < base_aligned:
                        continue

                    run_score = float(curr_aligned) + float(probe_current_avg) * 0.05
                    shifted_starts: Dict[int, float] = {i: float(cand_start)}
                    run_audio_meta: Dict[int, Dict[str, object]] = {i: probe_current_audio_meta}
                    run_avg_meta: Dict[int, float] = {i: float(probe_current_avg)}
                    run_ok = True
                    for k in range(i + 1, run_end + 1):
                        seg = segments[k]
                        shifted_start = float(seg["start"] - delta)
                        if shifted_start < 0.0:
                            run_ok = False
                            break
                        passed, avg = self.quick_verify(
                            source=seg["source"],
                            source_start=float(shifted_start),
                            target_start=float(seg["target_start"]),
                            duration=float(seg["duration"]),
                        )
                        audio_passed, audio_meta = self.quick_verify_audio(
                            source=seg["source"],
                            source_start=float(shifted_start),
                            target_start=float(seg["target_start"]),
                            duration=float(seg["duration"]),
                            combined_score=float(avg),
                        )
                        if not audio_passed:
                            run_ok = False
                            break
                        relaxed_tail_ok = bool(
                            (not passed)
                            and float(avg) >= 0.90
                            and float((audio_meta or {}).get("aligned_similarity", 0.0) or 0.0) >= 0.66
                        )
                        if (not passed) and (not relaxed_tail_ok):
                            run_ok = False
                            break
                        aligned = float((audio_meta or {}).get("aligned_similarity", 0.0) or 0.0)
                        best_shift = float((audio_meta or {}).get("best_shift_sec", 0.0) or 0.0)
                        shift_gain = float((audio_meta or {}).get("shift_gain", 0.0) or 0.0)
                        if abs(best_shift) >= 0.45 and shift_gain >= 0.10:
                            run_ok = False
                            break
                        shifted_starts[k] = float(shifted_start)
                        run_audio_meta[k] = audio_meta
                        run_avg_meta[k] = float(avg)
                        run_score += float(aligned) + float(avg) * 0.03

                    if not run_ok:
                        continue

                    score = (
                        float(run_score)
                        + float(curr_aligned - base_aligned) * 0.8
                        - float(overlap) * 0.05
                        - abs(float(curr_best_shift)) * 0.02
                        + float(refine_score) * 0.001
                    )
                    if best_score is None or score > best_score:
                        best_score = float(score)
                        best_choice = {
                            "delta": float(delta),
                            "prev_trim_sec": float(overlap),
                            "shifted_starts": shifted_starts,
                            "run_audio_meta": run_audio_meta,
                            "run_avg_meta": run_avg_meta,
                            "current_aligned": float(curr_aligned),
                            "base_aligned": float(base_aligned),
                            "refine_score": float(refine_score),
                        }

                if best_choice is None:
                    continue

                prev_trim_sec = float(best_choice.get("prev_trim_sec", 0.0) or 0.0)
                if prev_trim_sec > 1e-6:
                    prev["duration"] = max(0.0, float(prev["duration"]) - float(prev_trim_sec))
                    prev_q = prev.get("quality", {}) or {}
                    prev_q["bridge_run_prev_trim_no_target"] = True
                    prev_q["bridge_run_prev_trim_sec"] = float(prev_trim_sec)
                    prev_q["bridge_run_prev_trim_boundary_index"] = int(i)
                    prev["quality"] = prev_q

                for k in range(i, run_end + 1):
                    seg = segments[k]
                    old_start = float(seg["start"])
                    seg["start"] = float(best_choice["shifted_starts"][k])
                    q = seg.get("quality", {}) or {}
                    q["bridge_run_audio_realign_no_target"] = True
                    q["bridge_run_audio_realign_from"] = float(old_start)
                    q["bridge_run_audio_realign_to"] = float(seg["start"])
                    q["bridge_run_audio_realign_delta"] = float(best_choice["delta"])
                    q["bridge_run_audio_realign_anchor_index"] = int(i)
                    if k == i:
                        q["bridge_run_audio_realign_prev_trim_sec"] = float(prev_trim_sec)
                        q["bridge_run_audio_realign_base_aligned"] = float(best_choice["base_aligned"])
                        q["bridge_run_audio_realign_current_aligned"] = float(best_choice["current_aligned"])
                        q["bridge_run_audio_realign_refine_score"] = float(best_choice["refine_score"])
                    q["audio_guard"] = best_choice["run_audio_meta"].get(k, q.get("audio_guard", {}))
                    seg["quality"] = q
                    adjusted += 1

                changed = True

            if not changed:
                break

        return int(adjusted)

    def _stabilize_last_resort_segments_no_target(
        self,
        segments: List[dict],
        max_passes: int = 2,
    ) -> int:
        """
        终态连续性收口（禁兜底）：
        - 仅处理 recover_mode=neighbors_last_resort_no_target 且边界惩罚仍偏高的段；
        - 用同源邻段推导候选起点，优先消除词尾双读与尾段轻微倒放感。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        for _ in range(max(1, int(max_passes))):
            changed = False
            for i, seg in enumerate(segments):
                q = seg.get("quality", {}) or {}
                if str(q.get("recover_mode", "") or "") != "neighbors_last_resort_no_target":
                    continue
                source = seg.get("source")
                if source in (None, self.target_video):
                    continue

                unresolved_now = False
                if i > 0 and segments[i - 1].get("source") == source:
                    unresolved_now, _gap, _meta = self._no_target_boundary_unresolved_meta(segments[i - 1], seg)
                if (not unresolved_now) and (i + 1) < len(segments) and segments[i + 1].get("source") == source:
                    unresolved_now, _gap, _meta = self._no_target_boundary_unresolved_meta(seg, segments[i + 1])
                if not unresolved_now:
                    continue

                duration = float(seg["duration"])
                src_duration = self.get_video_duration(Path(source))
                if src_duration <= 0.0:
                    continue
                max_start = max(0.0, float(src_duration - duration))
                candidates: List[float] = [float(seg["start"])]

                prev = segments[i - 1] if i > 0 else None
                nxt = segments[i + 1] if (i + 1) < len(segments) else None
                if prev is not None and prev.get("source") == source:
                    prev_end = float(prev["start"]) + float(prev["duration"])
                    candidates.extend([prev_end, prev_end + 0.05, prev_end + 0.10])
                if nxt is not None and nxt.get("source") == source:
                    next_start = float(nxt["start"]) - duration
                    candidates.extend([next_start, next_start - 0.05, next_start - 0.10])
                if prev is not None and nxt is not None and prev.get("source") == source and nxt.get("source") == source:
                    candidates.append(0.5 * ((float(prev["start"]) + float(prev["duration"])) + (float(nxt["start"]) - duration)))

                best_choice: Optional[Dict[str, float]] = None
                seen_starts: Set[float] = set()
                base_combined = float(q.get("combined", 0.0))
                for raw_start in candidates:
                    start = max(0.0, min(float(raw_start), max_start))
                    key = round(float(start), 3)
                    if key in seen_starts:
                        continue
                    seen_starts.add(key)

                    passed, avg = self.verify_segment_visual(
                        source=Path(source),
                        source_start=float(start),
                        target_start=float(seg["target_start"]),
                        duration=duration,
                        offsets=[
                            0.0,
                            duration * 0.25,
                            duration * 0.5,
                            duration * 0.75,
                            max(0.0, duration - 0.1),
                        ],
                        min_avg=0.70,
                        min_floor=0.45,
                    )
                    if not passed:
                        continue
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=Path(source),
                        source_start=float(start),
                        target_start=float(seg["target_start"]),
                        duration=duration,
                        combined_score=float(avg),
                    )
                    if not audio_passed:
                        continue

                    penalty = 0.0
                    if prev is not None and prev.get("source") == source:
                        penalty += abs(float(start) - (float(prev["start"]) + float(prev["duration"])))
                    if nxt is not None and nxt.get("source") == source:
                        penalty += abs(float(nxt["start"]) - (float(start) + duration))
                    rank = float(avg) - float(penalty) * 0.16
                    if best_choice is None or rank > float(best_choice["rank"]):
                        best_choice = {
                            "start": float(start),
                            "avg": float(avg),
                            "rank": float(rank),
                            "penalty": float(penalty),
                            "audio_meta": dict(audio_meta or {}),
                        }

                if best_choice is None:
                    continue
                if float(best_choice["avg"]) + 1e-6 < max(0.80, base_combined - 0.06):
                    continue
                if abs(float(best_choice["start"]) - float(seg["start"])) < 0.04:
                    continue

                old_start = float(seg["start"])
                seg["start"] = float(best_choice["start"])
                q["last_resort_stabilized_no_target"] = True
                q["last_resort_stabilized_from"] = float(old_start)
                q["last_resort_stabilized_to"] = float(best_choice["start"])
                q["last_resort_stabilized_verify_avg"] = float(best_choice["avg"])
                q["last_resort_stabilized_penalty"] = float(best_choice["penalty"])
                q["audio_guard"] = best_choice["audio_meta"]
                seg["quality"] = q
                adjusted += 1
                changed = True

            if not changed:
                break

        return int(adjusted)

    def _count_no_target_boundary_unresolved(self, segments: List[dict]) -> int:
        """统计禁兜底模式下同源相邻段未收敛边界数量。"""
        unresolved = 0
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev["source"] == self.target_video or curr["source"] == self.target_video:
                continue
            if prev["source"] != curr["source"]:
                continue
            is_unresolved, _gap, _meta = self._no_target_boundary_unresolved_meta(prev, curr)
            if is_unresolved:
                unresolved += 1
        return int(unresolved)

    def _collect_no_target_unresolved_pairs(self, segments: List[dict]) -> List[Tuple[int, int, float]]:
        """
        收集禁兜底模式下未收敛同源边界。
        返回 (prev_index, curr_index, gap)；按 |gap| 从大到小排序。
        """
        pairs: List[Tuple[int, int, float]] = []
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev["source"] == self.target_video or curr["source"] == self.target_video:
                continue
            if prev["source"] != curr["source"]:
                continue
            is_unresolved, gap, _meta = self._no_target_boundary_unresolved_meta(prev, curr)
            if is_unresolved:
                pairs.append((i - 1, i, float(gap)))
        pairs.sort(key=lambda x: abs(float(x[2])), reverse=True)
        return pairs

    def _rematch_unresolved_boundaries_without_target_fallback(self, segments: List[dict], max_attempts: int) -> int:
        """
        对禁兜底模式未收敛边界做定点重匹配：
        - 支持对同一边界左右两侧分别尝试；
        - 对“右段贴源尾导致无法后移”的边界，优先重匹配右侧段；
        - 在 pHash 候选不足时追加一次宽范围视觉候选，减少“明明可换源却没被尝试”。
        - 仅做少量尝试，避免显著拉长整体耗时
        返回：成功替换段数量。
        """
        if self.enable_target_video_fallback:
            return 0
        if not bool(getattr(self, "no_target_boundary_rematch_enabled", True)):
            return 0

        unresolved_pairs = self._collect_no_target_unresolved_pairs(segments)
        if not unresolved_pairs:
            return 0

        def boundary_penalty(left_idx: int, right_idx: int) -> float:
            if left_idx < 0 or right_idx >= len(segments):
                return 0.0
            left = segments[left_idx]
            right = segments[right_idx]
            if left["source"] == self.target_video or right["source"] == self.target_video:
                return 0.0
            if left["source"] != right["source"]:
                return 0.0
            is_unresolved, gap, meta = self._no_target_boundary_unresolved_meta(left, right)
            neg_trigger = float(meta.get("neg_trigger", max(0.12, float(right["duration"]) * 0.02)))
            pos_trigger = float(meta.get("pos_trigger", max(0.20, float(right["duration"]) * 0.04)))
            if gap < -neg_trigger:
                return float((-gap) - neg_trigger)
            if gap > pos_trigger:
                return float(gap - pos_trigger)
            if is_unresolved:
                return max(
                    0.05,
                    float(meta.get("recover_boundary_excess", 0.0)),
                    float(meta.get("boundary_audio_next_penalty_after", 0.0)),
                )
            return 0.0

        def local_penalty(center_idx: int) -> float:
            return boundary_penalty(center_idx - 1, center_idx) + boundary_penalty(center_idx, center_idx + 1)

        def total_boundary_excess() -> float:
            total = 0.0
            for left_i, right_i, _gap in self._collect_no_target_unresolved_pairs(segments):
                total += boundary_penalty(left_i, right_i)
            return float(total)

        def is_tail_locked(seg_idx: int) -> bool:
            if seg_idx < 0 or seg_idx >= len(segments):
                return False
            seg = segments[seg_idx]
            src = seg.get("source")
            if src in (None, self.target_video):
                return False
            src_duration = self.get_video_duration(src)
            if src_duration <= 0.0:
                return False
            seg_duration = max(0.0, float(seg.get("duration", 0.0)))
            max_start = max(0.0, float(src_duration - seg_duration))
            lock_tol = max(0.06, seg_duration * 0.015)
            return float(seg.get("start", 0.0)) >= (max_start - lock_tol)

        def is_run_terminal(seg_idx: int) -> bool:
            if seg_idx < 0 or seg_idx >= len(segments):
                return False
            src = segments[seg_idx].get("source")
            if src in (None, self.target_video):
                return True
            next_idx = seg_idx + 1
            if next_idx >= len(segments):
                return True
            return segments[next_idx].get("source") != src

        applied = 0
        attempts = 0
        attempted_signatures: Set[Tuple[int, int, int]] = set()
        for prev_i, curr_i, pair_gap in unresolved_pairs:
            if attempts >= max_attempts:
                break

            prev = segments[prev_i]
            curr = segments[curr_i]
            prev_q = prev.get("quality", {}) or {}
            curr_q = curr.get("quality", {}) or {}
            prev_combined = float(prev_q.get("combined", 0.0))
            curr_combined = float(curr_q.get("combined", 0.0))
            pair_severity = abs(float(pair_gap))
            pair_overlap = float(pair_gap) < 0.0
            pair_tail_locked_curr = is_tail_locked(curr_i)

            # 同一边界支持左右两侧尝试；负重叠场景优先尝试右段（更符合“去重复”目标）。
            candidate_indices: List[int] = []
            if pair_overlap:
                candidate_indices.append(curr_i)
            candidate_indices.append(curr_i if curr_combined <= prev_combined else prev_i)
            candidate_indices.append(prev_i)
            candidate_indices.append(curr_i)
            if pair_tail_locked_curr:
                candidate_indices.insert(0, curr_i)
            ordered_indices: List[int] = []
            seen_idx: Set[int] = set()
            for idx in candidate_indices:
                if idx in seen_idx:
                    continue
                seen_idx.add(idx)
                ordered_indices.append(idx)

            pair_applied = False
            for cand_i in ordered_indices:
                if attempts >= max_attempts:
                    break
                attempt_sig = (int(prev_i), int(curr_i), int(cand_i))
                if attempt_sig in attempted_signatures:
                    continue
                attempted_signatures.add(attempt_sig)
                attempts += 1

                seg = segments[cand_i]
                old_q = dict(seg.get("quality", {}) or {})
                old_combined = float(old_q.get("combined", 0.0))
                old_source = seg["source"]
                old_start = float(seg["start"])
                before_unresolved = self._count_no_target_boundary_unresolved(segments)
                before_excess = total_boundary_excess()
                before_pen = local_penalty(cand_i)

                candidates_to_try: List[Dict[str, object]] = []
                neighbor_sources: Set[str] = set()
                for nei_i in (cand_i - 1, cand_i + 1):
                    if nei_i < 0 or nei_i >= len(segments):
                        continue
                    nei_src = segments[nei_i]["source"]
                    if nei_src == self.target_video:
                        continue
                    neighbor_sources.add(str(nei_src))

                # 方案A：沿用现有 process_segment 结果（原策略）
                task = SegmentTask(
                    index=int(seg["index"]),
                    target_start=float(seg["target_start"]),
                    duration=float(seg["duration"]),
                )
                try:
                    result = self.process_segment(task)
                except Exception:
                    result = None
                if (
                    result is not None
                    and result.success
                    and not self._is_secondary_source(Path(result.source))
                ):
                    candidates_to_try.append(
                        {
                            "source": result.source,
                            "start": float(result.source_start),
                            "quality": dict(result.quality or {}),
                            "mode": "process_segment",
                        }
                    )

                # 方案B：边界感知候选重搜（优先减少局部重叠/断缝）
                phash_cands = self.find_match_by_phash(
                    target_start=float(seg["target_start"]),
                    duration=float(seg["duration"]),
                    seg_index=int(seg["index"]),
                    top_k=120,
                )
                # 附加邻段外推候选（帮助跨源切换时快速贴合）
                for nei_i in (cand_i - 1, cand_i + 1):
                    if nei_i < 0 or nei_i >= len(segments):
                        continue
                    nei = segments[nei_i]
                    est = float(nei["start"]) + (float(seg["target_start"]) - float(nei["target_start"]))
                    phash_cands.append((nei["source"], float(est), 0.0))

                seen_boundary_cands: Set[Tuple[str, float]] = set()
                for src, raw_start, phash_sim in phash_cands:
                    src = Path(src)
                    if src == self.target_video:
                        continue
                    if self._is_secondary_source(src):
                        continue
                    sig = (str(src), round(float(raw_start), 3))
                    if sig in seen_boundary_cands:
                        continue
                    seen_boundary_cands.add(sig)

                    src_dur = self.get_video_duration(src)
                    if src_dur <= 0.0:
                        continue
                    max_start = max(0.0, float(src_dur - float(seg["duration"])))
                    cand_start = max(0.0, min(float(raw_start), max_start))

                    refined_start, refine_score = self.refine_start_by_visual(
                        source=src,
                        initial_start=float(cand_start),
                        target_start=float(seg["target_start"]),
                        duration=float(seg["duration"]),
                    )
                    refined_start = max(0.0, min(float(refined_start), max_start))

                    verify_passed, verify_avg = self.quick_verify(
                        source=src,
                        source_start=float(refined_start),
                        target_start=float(seg["target_start"]),
                        duration=float(seg["duration"]),
                    )
                    allow_desperation = bool(
                        pair_severity >= 1.60
                        and (cand_i == curr_i or pair_tail_locked_curr)
                    )
                    relaxed_verify = False
                    desperation_nonlocal = False
                    if not verify_passed:
                        # 仅用于“边界未收敛重匹配”的受控宽松核验，避免因严格阈值导致长时间重叠无法消除。
                        relaxed_min_avg = max(0.76, float(self.strict_verify_min_sim) - 0.02)
                        relaxed_min_floor = max(0.58, float(self.strict_verify_min_sim) - 0.20)
                        if pair_severity >= 1.20 and (cand_i == curr_i or pair_tail_locked_curr):
                            relaxed_min_avg = min(relaxed_min_avg, 0.70)
                            relaxed_min_floor = min(relaxed_min_floor, 0.45)
                        if pair_severity >= 2.50 and (cand_i == curr_i or pair_tail_locked_curr):
                            relaxed_min_avg = min(relaxed_min_avg, 0.64)
                            relaxed_min_floor = min(relaxed_min_floor, 0.36)
                        offsets = [
                            0.0,
                            float(seg["duration"]) * 0.25,
                            float(seg["duration"]) * 0.5,
                            float(seg["duration"]) * 0.75,
                            max(0.0, float(seg["duration"]) - 0.1),
                        ]
                        relaxed_passed, relaxed_avg = self.verify_segment_visual(
                            source=src,
                            source_start=float(refined_start),
                            target_start=float(seg["target_start"]),
                            duration=float(seg["duration"]),
                            offsets=offsets,
                            min_avg=float(relaxed_min_avg),
                            min_floor=float(relaxed_min_floor),
                        )
                        if not relaxed_passed:
                            if not allow_desperation:
                                continue
                            # 极端重叠兜底：允许次优画面但音频可用的跨源候选先去重，
                            # 防止末段长期保留 2~5 秒重复内容。
                            _, loose_avg = self.verify_segment_visual(
                                source=src,
                                source_start=float(refined_start),
                                target_start=float(seg["target_start"]),
                                duration=float(seg["duration"]),
                                offsets=offsets,
                                min_avg=0.0,
                                min_floor=0.0,
                            )
                            if float(loose_avg) < 0.82:
                                continue
                            verify_avg = float(loose_avg)
                            relaxed_verify = True
                            desperation_nonlocal = True
                        else:
                            verify_avg = float(relaxed_avg)
                            relaxed_verify = True

                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=src,
                        source_start=float(refined_start),
                        target_start=float(seg["target_start"]),
                        duration=float(seg["duration"]),
                        combined_score=float(verify_avg),
                    )
                    if not audio_passed:
                        continue

                    candidates_to_try.append(
                        {
                            "source": src,
                            "start": float(refined_start),
                            "quality": {
                                "combined": float(verify_avg),
                                "boundary_unresolved_rematch_no_target": True,
                                "boundary_unresolved_rematch_mode": (
                                    "boundary_aware_candidate_desperate"
                                    if desperation_nonlocal
                                    else "boundary_aware_candidate"
                                ),
                                "boundary_unresolved_rematch_phash_similarity": float(phash_sim),
                                "boundary_unresolved_rematch_refine_score": float(refine_score),
                                "boundary_unresolved_rematch_relaxed_verify": bool(relaxed_verify),
                                "boundary_unresolved_rematch_desperate_nonlocal": bool(desperation_nonlocal),
                                "audio_guard": audio_meta,
                            },
                            "mode": (
                                "boundary_aware_candidate_desperate"
                                if desperation_nonlocal
                                else "boundary_aware_candidate"
                            ),
                        }
                    )

                # 方案C：当边界重叠较大或右段贴尾时，补一次宽范围视觉候选。
                need_broad_visual = (
                    pair_severity >= 0.55
                    or pair_tail_locked_curr
                    or len(phash_cands) < 6
                )
                if need_broad_visual:
                    fb_source, fb_start, fb_score = self.find_best_match_by_visual(
                        target_start=float(seg["target_start"]),
                        duration=float(seg["duration"]),
                        seg_index=int(seg["index"]),
                    )
                    if (
                        fb_source is not None
                        and fb_source != self.target_video
                        and (not self._is_secondary_source(Path(fb_source)))
                    ):
                        fb_source = Path(fb_source)
                        fb_dur = self.get_video_duration(fb_source)
                        if fb_dur > 0.0:
                            fb_max_start = max(0.0, float(fb_dur - float(seg["duration"])))
                            fb_start = max(0.0, min(float(fb_start), fb_max_start))
                            refined_start, refine_score = self.refine_start_by_visual(
                                source=fb_source,
                                initial_start=float(fb_start),
                                target_start=float(seg["target_start"]),
                                duration=float(seg["duration"]),
                            )
                            refined_start = max(0.0, min(float(refined_start), fb_max_start))

                            verify_passed, verify_avg = self.quick_verify(
                                source=fb_source,
                                source_start=float(refined_start),
                                target_start=float(seg["target_start"]),
                                duration=float(seg["duration"]),
                            )
                            if verify_passed:
                                audio_passed, audio_meta = self.quick_verify_audio(
                                    source=fb_source,
                                    source_start=float(refined_start),
                                    target_start=float(seg["target_start"]),
                                    duration=float(seg["duration"]),
                                    combined_score=float(verify_avg),
                                )
                                if audio_passed:
                                    candidates_to_try.append(
                                        {
                                            "source": fb_source,
                                            "start": float(refined_start),
                                            "quality": {
                                                "combined": float(verify_avg),
                                                "boundary_unresolved_rematch_no_target": True,
                                                "boundary_unresolved_rematch_mode": "boundary_visual_fallback",
                                                "boundary_unresolved_rematch_visual_search_score": float(fb_score),
                                                "boundary_unresolved_rematch_refine_score": float(refine_score),
                                                "audio_guard": audio_meta,
                                            },
                                            "mode": "boundary_visual_fallback",
                                        }
                                    )

                # 评估候选，选择“局部边界惩罚最小、总体惩罚下降、且分数可接受”的最佳方案
                best_choice: Optional[Dict[str, object]] = None
                best_key: Optional[Tuple[float, float, float]] = None
                for cand in candidates_to_try:
                    new_source = cand.get("source")
                    new_start = float(cand.get("start", 0.0))
                    if new_source is None:
                        continue
                    if str(new_source) == str(old_source) and abs(new_start - old_start) <= 0.06:
                        continue

                    seg["source"] = new_source
                    seg["start"] = new_start
                    after_pen = local_penalty(cand_i)
                    after_excess = total_boundary_excess()
                    after_unresolved = self._count_no_target_boundary_unresolved(segments)
                    seg["source"] = old_source
                    seg["start"] = old_start

                    improved_local = (after_pen + 1e-6) < before_pen
                    improved_total = (after_excess + 1e-6) < before_excess
                    if not (improved_local or improved_total):
                        continue
                    allow_unresolved_increase = 0
                    if pair_severity >= 1.20 and (cand_i == curr_i or pair_tail_locked_curr):
                        # 严重重叠边界允许“临时+1 未收敛”以换取本边界先消除重复，
                        # 后续硬约束/音频修复会再做收口。
                        allow_unresolved_increase = 1
                    if after_unresolved > (before_unresolved + allow_unresolved_increase):
                        continue

                    new_q = cand.get("quality", {}) or {}
                    new_combined = float(new_q.get("combined", old_combined))
                    matches_neighbor_source = (str(new_source) in neighbor_sources) if neighbor_sources else True
                    old_matches_neighbor_source = (str(old_source) in neighbor_sources) if neighbor_sources else True
                    if (
                        old_matches_neighbor_source
                        and (not matches_neighbor_source)
                        and len(neighbor_sources) == 1
                    ):
                        # 当前段已经处在“唯一同源邻居链”上时，不允许终态 rematch 再把它抢到异源。
                        # 这类跨源回跳通常会重新制造 3:23 这类尾部重复/错内容。
                        severe_escape = bool(
                            pair_severity >= 2.50
                            and new_combined >= max(0.92, old_combined + 0.04)
                        )
                        if not severe_escape:
                            continue
                    old_locked_to_neighbor_source = bool(
                        old_matches_neighbor_source
                        and (
                            old_q.get("repair_isolated_neighbor_source_no_target")
                            or str(old_q.get("repair_mode", "") or "") == "isolated_neighbor_source_no_target"
                            or old_q.get("recover_rebalanced_no_target")
                        )
                    )
                    if (
                        old_locked_to_neighbor_source
                        and (not matches_neighbor_source)
                        and len(neighbor_sources) == 1
                    ):
                        # 已经被“邻居同源修复”收回来的段，不允许再被终态 rematch 轻易拉到异源。
                        # 这类回跳会把 3:23 这类重复/错内容问题重新带回最终成片。
                        severe_escape = bool(
                            pair_severity >= 1.90
                            and new_combined >= max(0.90, old_combined - 0.03)
                        )
                        if not severe_escape:
                            continue
                    if not matches_neighbor_source:
                        # 边界修复可跨源，但对跨源候选分段设置动态门限：
                        # 仅在“run 末端/贴尾且重叠严重”场景放宽，避免中段跨源误跳。
                        nonlocal_min = 0.88
                        curr_terminal = is_run_terminal(curr_i)
                        if pair_severity >= 0.55 and curr_terminal:
                            nonlocal_min = 0.86
                        if (
                            pair_severity >= 1.10
                            and curr_terminal
                            and (cand_i == curr_i or pair_tail_locked_curr)
                        ):
                            nonlocal_min = 0.80
                        if (
                            pair_severity >= 1.90
                            and curr_terminal
                            and (cand_i == curr_i or pair_tail_locked_curr)
                        ):
                            nonlocal_min = 0.76
                        nonlocal_min = max(0.76, min(0.92, nonlocal_min))
                        if new_combined + 1e-6 < nonlocal_min:
                            continue
                        new_q["boundary_unresolved_rematch_nonlocal_source"] = True
                        new_q["boundary_unresolved_rematch_nonlocal_min_required"] = float(nonlocal_min)

                    mode_name = str(cand.get("mode", ""))
                    max_drop = 0.12 if improved_local else 0.08
                    if pair_severity >= 1.60 and (cand_i == curr_i or pair_tail_locked_curr):
                        max_drop = max(max_drop, 0.18)
                    if pair_severity >= 2.50 and (cand_i == curr_i or pair_tail_locked_curr):
                        max_drop = max(max_drop, 0.24)
                    if mode_name == "boundary_aware_candidate_desperate":
                        max_drop = max(max_drop, 0.28 if pair_severity >= 2.50 else 0.22)
                    if new_combined + max_drop < old_combined:
                        continue

                    # 排序优先级：局部惩罚更小 > 全局惩罚更小 > 分数更高
                    nonlocal_penalty = 0.0
                    if not matches_neighbor_source:
                        nonlocal_penalty = 0.35
                        if pair_severity >= 1.20:
                            nonlocal_penalty = 0.12
                        if pair_severity >= 2.50:
                            nonlocal_penalty = 0.0
                    mode_penalty = 0.03 if mode_name == "boundary_visual_fallback" else 0.0
                    key = (float(after_pen) + nonlocal_penalty + mode_penalty, float(after_excess), -float(new_combined))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_choice = {
                            "source": new_source,
                            "start": float(new_start),
                            "quality": dict(new_q),
                            "after_pen": float(after_pen),
                            "after_excess": float(after_excess),
                            "after_unresolved": int(after_unresolved),
                            "combined": float(new_combined),
                            "mode": mode_name,
                        }

                if best_choice is None:
                    continue

                seg["source"] = best_choice["source"]
                seg["start"] = float(best_choice["start"])
                q = dict(old_q)
                q.update(dict(best_choice.get("quality", {}) or {}))
                q["boundary_unresolved_rematch_no_target"] = True
                q["boundary_unresolved_rematch_pair_prev_index"] = int(prev_i)
                q["boundary_unresolved_rematch_pair_curr_index"] = int(curr_i)
                q["boundary_unresolved_rematch_pair_gap"] = float(pair_gap)
                q["boundary_unresolved_rematch_candidate_index"] = int(cand_i)
                q["boundary_unresolved_rematch_tail_locked_curr"] = bool(pair_tail_locked_curr)
                q["boundary_unresolved_rematch_from_source"] = str(old_source)
                q["boundary_unresolved_rematch_from_start"] = float(old_start)
                q["boundary_unresolved_rematch_before_pen"] = float(before_pen)
                q["boundary_unresolved_rematch_after_pen"] = float(best_choice["after_pen"])
                q["boundary_unresolved_rematch_before_excess"] = float(before_excess)
                q["boundary_unresolved_rematch_after_excess"] = float(best_choice["after_excess"])
                q["boundary_unresolved_rematch_mode"] = str(best_choice.get("mode", ""))
                seg["quality"] = q
                applied += 1
                pair_applied = True
                break

            if pair_applied:
                continue

        return int(applied)

    def _backprop_resolve_locked_tail_overlaps_no_target(self, segments: List[dict]) -> int:
        """
        反向回推修复（禁兜底）：
        - 处理同源相邻段未收敛的负重叠边界；
        - 将重叠量向前分摊到同源 run，优先消除重复词/局部倒放感；
        - 仅对置信度可接受或强制邻段恢复段生效，避免低置信大幅漂移。
        返回：成功回推修复的边界数。
        """
        if self.enable_target_video_fallback:
            return 0
        if not bool(getattr(self, "no_target_backprop_overlap_fix", True)):
            return 0
        if len(segments) < 2:
            return 0

        fixed_boundaries = 0
        max_shift_cap = max(0.0, float(getattr(self, "no_target_backprop_max_shift", 0.0)))
        min_quality = max(0.0, float(getattr(self, "no_target_backprop_min_quality", 0.0)))
        neg_floor = max(0.0, float(getattr(self, "no_target_backprop_neg_trigger_floor", 0.08)))

        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev["source"] == self.target_video or curr["source"] == self.target_video:
                continue
            if prev["source"] != curr["source"]:
                continue

            prev_end = float(prev["start"]) + float(prev["duration"])
            curr_start = float(curr["start"])
            gap = curr_start - prev_end
            neg_trigger = max(float(neg_floor), float(curr["duration"]) * 0.02)
            if gap >= -neg_trigger:
                continue

            source_duration = self.get_video_duration(curr["source"])
            if source_duration <= 0.0:
                continue
            overlap = -gap
            if overlap <= 1e-6:
                continue
            rollback: Dict[int, float] = {}

            def save_old(idx: int) -> None:
                if idx not in rollback:
                    rollback[idx] = float(segments[idx]["start"])

            def shift_left_with_chain(idx: int, amount: float) -> float:
                if amount <= 1e-6:
                    return 0.0
                if idx < 0:
                    return 0.0
                seg = segments[idx]
                if seg["source"] != curr["source"] or seg["source"] == self.target_video:
                    return 0.0

                seg_start_now = float(seg["start"])
                min_start_now = 0.0
                has_prev_same = (
                    idx > 0
                    and segments[idx - 1]["source"] == seg["source"]
                    and segments[idx - 1]["source"] != self.target_video
                )
                if has_prev_same:
                    min_start_now = float(segments[idx - 1]["start"]) + float(segments[idx - 1]["duration"])
                movable_now = max(0.0, seg_start_now - min_start_now)

                need_extra = max(0.0, float(amount) - movable_now)
                if need_extra > 1e-6 and has_prev_same:
                    shift_left_with_chain(idx - 1, need_extra)
                    seg_start_now = float(seg["start"])
                    min_start_now = float(segments[idx - 1]["start"]) + float(segments[idx - 1]["duration"])
                    movable_now = max(0.0, seg_start_now - min_start_now)

                applied = min(float(amount), movable_now)
                if applied > 1e-6:
                    save_old(idx)
                    seg["start"] = seg_start_now - applied
                return float(applied)

            curr_max_start = max(0.0, float(source_duration - float(curr["duration"])))
            if (i + 1) < len(segments):
                nxt = segments[i + 1]
                if nxt["source"] == curr["source"] and nxt["source"] != self.target_video:
                    # 关键保护：若后续仍是同源段，不要把当前段前推到“挤压下一段”的位置，
                    # 否则会把重复集中到 run 尾部，形成用户可感知的末尾回放感。
                    next_gap_allow = max(float(neg_floor), float(nxt["duration"]) * 0.02)
                    next_bound = float(nxt["start"]) - float(curr["duration"]) + float(next_gap_allow)
                    curr_max_start = min(curr_max_start, max(0.0, float(next_bound)))
            moved_curr = 0.0
            curr_headroom = max(0.0, curr_max_start - float(curr["start"]))
            if curr_headroom > 1e-6:
                moved_curr = min(float(overlap), curr_headroom)
                if moved_curr > 1e-6:
                    save_old(i)
                    curr["start"] = float(curr["start"]) + moved_curr

            remaining = max(0.0, float(overlap) - moved_curr)
            if max_shift_cap > 1e-6:
                remaining = min(remaining, max_shift_cap)
            shifted_prev = 0.0
            if remaining > 1e-6:
                shifted_prev = shift_left_with_chain(i - 1, remaining)

            total_shift = moved_curr + shifted_prev
            if total_shift <= 1e-6:
                for idx, old_start in rollback.items():
                    segments[idx]["start"] = old_start
                continue

            # 质量守卫：仅允许高置信段或“邻段强制恢复”段参与大幅回推。
            quality_ok = True
            for idx, old_start in rollback.items():
                seg = segments[idx]
                q = seg.get("quality", {}) or {}
                combined = float(q.get("combined", 0.0))
                recover_mode = str(q.get("recover_mode", "") or "")
                if (combined + 1e-6) < min_quality and recover_mode != "neighbors_forced_no_verify":
                    quality_ok = False
                    break

                large_shift = abs(float(seg["start"]) - float(old_start)) > max(0.9, float(seg["duration"]) * 0.22)
                verify_needed = large_shift and combined < max(0.86, min_quality + 0.08)
                if verify_needed:
                    passed, verify_avg = self.quick_verify(
                        source=seg["source"],
                        source_start=float(seg["start"]),
                        target_start=float(seg["target_start"]),
                        duration=float(seg["duration"]),
                    )
                    if not passed and recover_mode != "neighbors_forced_no_verify":
                        quality_ok = False
                        break
                    q["micro_gap_backprop_verify_passed"] = bool(passed)
                    q["micro_gap_backprop_verify_avg"] = float(verify_avg)
                    seg["quality"] = q

            if not quality_ok:
                for idx, old_start in rollback.items():
                    segments[idx]["start"] = old_start
                continue

            # 安全阀：禁止回推修复引入明显同源负重叠（会直接表现为重读/回放感）。
            overlap_safe = True
            for idx in sorted(rollback.keys()):
                if idx <= 0:
                    continue
                left = segments[idx - 1]
                right = segments[idx]
                if right["source"] == self.target_video or left["source"] != right["source"]:
                    continue
                local_gap = float(right["start"]) - (float(left["start"]) + float(left["duration"]))
                local_neg_trigger = max(float(neg_floor), float(right["duration"]) * 0.02)
                if local_gap < -max(0.35, local_neg_trigger * 2.0):
                    overlap_safe = False
                    break
            if not overlap_safe:
                for idx, old_start in rollback.items():
                    segments[idx]["start"] = old_start
                continue

            new_prev_end = float(segments[i - 1]["start"]) + float(segments[i - 1]["duration"])
            new_gap = float(segments[i]["start"]) - new_prev_end
            before_excess = max(0.0, (-float(gap)) - float(neg_trigger))
            after_excess = max(0.0, (-float(new_gap)) - float(neg_trigger))
            if (after_excess + 1e-5) >= before_excess:
                for idx, old_start in rollback.items():
                    segments[idx]["start"] = old_start
                continue

            for idx, old_start in rollback.items():
                seg = segments[idx]
                q = seg.get("quality", {}) or {}
                shift_applied = float(old_start - float(seg["start"]))
                q["micro_gap_backprop_shift_no_target"] = True
                q["micro_gap_backprop_shift_sec"] = float(q.get("micro_gap_backprop_shift_sec", 0.0)) + shift_applied
                q["micro_gap_backprop_anchor_index"] = int(i)
                if idx == i:
                    q["micro_gap_backprop_forward_to_max"] = True
                seg["quality"] = q

            fixed_boundaries += 1

        return int(fixed_boundaries)

    def _nudge_cross_source_head_boundaries_no_target(self, segments: List[dict]) -> int:
        """
        跨源边界头部微调（禁兜底）：
        - 命中条件：前段接近源片尾，后段接近源片头，且两段不同源
        - 通过小步前移后段起点，降低跨源接缝处“重复词/黑一下”的主观感受
        """
        if self.enable_target_video_fallback:
            return 0
        if not bool(getattr(self, "cross_source_head_nudge_enabled", True)):
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        prev_tail_window = max(0.0, float(getattr(self, "cross_source_head_nudge_prev_tail_window", 0.0)))
        curr_head_window = max(0.0, float(getattr(self, "cross_source_head_nudge_curr_head_window", 0.0)))
        max_offset = max(0.0, float(getattr(self, "cross_source_head_nudge_max_offset", 0.0)))
        score_bias = float(getattr(self, "cross_source_head_nudge_score_bias", 0.0))
        max_verify_drop = max(0.0, float(getattr(self, "cross_source_head_nudge_max_verify_drop", 0.0)))
        forward_cap = max(0.0, float(getattr(self, "cross_source_head_nudge_forward_cap", 0.35)))
        forward_gain_trigger = max(
            0.0,
            float(getattr(self, "cross_source_head_nudge_forward_gain_trigger", 0.06)),
        )
        allow_backward = bool(getattr(self, "cross_source_head_nudge_allow_backward", True))
        backward_cap = max(0.0, float(getattr(self, "cross_source_head_nudge_backward_cap", 0.8)))
        backward_gain_trigger = max(
            0.0,
            float(getattr(self, "cross_source_head_nudge_backward_gain_trigger", 0.015)),
        )
        boundary_audio_weight = max(
            0.0,
            float(getattr(self, "cross_source_head_nudge_boundary_audio_weight", 0.06)),
        )
        if max_offset <= 1e-6:
            return 0

        max_forward_offset = min(float(max_offset), float(forward_cap)) if forward_cap > 1e-6 else float(max_offset)
        base_positive_offsets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
        positive_offsets = [x for x in base_positive_offsets if x <= (max_forward_offset + 1e-6)]
        if max_forward_offset > 1e-6:
            rounded_cap = round(float(max_forward_offset), 3)
            if all(abs(float(x) - float(rounded_cap)) > 1e-6 for x in positive_offsets):
                positive_offsets.append(float(rounded_cap))
        positive_offsets = sorted(set(float(x) for x in positive_offsets))

        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev["source"] == self.target_video or curr["source"] == self.target_video:
                continue
            if prev["source"] == curr["source"]:
                continue

            curr_start = float(curr["start"])
            if curr_start > curr_head_window + 1e-6:
                continue

            prev_source_duration = self.get_video_duration(prev["source"])
            if prev_source_duration <= 0.0:
                continue
            prev_tail_left = float(prev_source_duration - (float(prev["start"]) + float(prev["duration"])))
            if prev_tail_left > prev_tail_window + 1e-6:
                continue

            curr_source_duration = self.get_video_duration(curr["source"])
            if curr_source_duration <= 0.0:
                continue
            max_start = max(0.0, float(curr_source_duration - float(curr["duration"])))
            curr_q = curr.get("quality", {}) or {}
            rescue_meta = curr_q.get("rescue", {}) if isinstance(curr_q.get("rescue", {}), dict) else {}
            rescue_triggered = bool(rescue_meta.get("triggered", False))
            max_backward_offset = 0.0
            if allow_backward:
                max_backward_offset = min(float(curr_start), float(backward_cap), float(max_offset))
                if rescue_triggered:
                    # 片头已被音频守卫/重救回拉到句首附近时，不再允许跨源微调把它反向拽回更早位置。
                    max_backward_offset = 0.0

            offset_candidates = [float(x) for x in positive_offsets]
            if allow_backward and max_backward_offset > 1e-6:
                negative_base = [-0.8, -0.6, -0.4, -0.3, -0.2, -0.1, -0.05]
                for off in negative_base:
                    if abs(float(off)) <= (max_backward_offset + 1e-6):
                        offset_candidates.append(float(off))
                rounded_back_cap = round(float(max_backward_offset), 3)
                if all(abs(float(x) + float(rounded_back_cap)) > 1e-6 for x in offset_candidates):
                    offset_candidates.append(-float(rounded_back_cap))
            offset_candidates = sorted(set(float(x) for x in offset_candidates))
            if 0.0 not in offset_candidates:
                offset_candidates.insert(0, 0.0)

            base_passed, base_avg = self.quick_verify(
                source=curr["source"],
                source_start=float(curr_start),
                target_start=float(curr["target_start"]),
                duration=float(curr["duration"]),
            )
            if not base_passed:
                continue
            base_audio_passed, base_audio_meta = self.quick_verify_audio(
                source=curr["source"],
                source_start=float(curr_start),
                target_start=float(curr["target_start"]),
                duration=float(curr["duration"]),
                combined_score=float(base_avg),
            )
            base_aligned = float(base_audio_meta.get("aligned_similarity", 0.0) or 0.0)
            base_shift_gain = float(base_audio_meta.get("shift_gain", 0.0) or 0.0)
            boundary_audio_window = max(
                0.30,
                float(getattr(self, "boundary_audio_probe_window_sec", 0.45)),
            )
            base_boundary_sim = self._boundary_audio_similarity(
                source=curr["source"],
                target_boundary_sec=float(curr["target_start"]),
                source_boundary_sec=float(curr_start),
                window_sec=float(boundary_audio_window),
            )
            base_score = (
                float(base_avg)
                + float(base_aligned) * 0.03
                + float(base_boundary_sim) * float(boundary_audio_weight)
                - float(base_shift_gain) * 0.06
            )
            if not base_audio_passed:
                # 当前片头候选已被音频守卫判失败时，仍允许继续搜索偏移候选，
                # 避免“卡在错误片头起点”而无法回拉到正确句首。
                base_score -= 0.25

            best = {
                "start": float(curr_start),
                "avg": float(base_avg),
                "offset": 0.0,
                "score": float(base_score),
                "audio_meta": base_audio_meta,
                "boundary_sim": float(base_boundary_sim),
                "base_audio_failed": bool(not base_audio_passed),
            }

            for off in offset_candidates:
                if off <= 1e-6:
                    if off >= -1e-6:
                        continue
                cand = max(0.0, min(float(curr_start + off), max_start))
                if abs(cand - curr_start) <= 1e-6:
                    continue
                # 约束：不能把当前段前移到与“下一同源段”形成明显负重叠
                if (i + 1) < len(segments):
                    nxt = segments[i + 1]
                    if nxt["source"] == curr["source"] and nxt["source"] != self.target_video:
                        next_gap = float(nxt["start"]) - (float(cand) + float(curr["duration"]))
                        neg_trigger = max(0.12, float(nxt["duration"]) * 0.02)
                        if next_gap < -neg_trigger:
                            continue
                passed, avg = self.quick_verify(
                    source=curr["source"],
                    source_start=float(cand),
                    target_start=float(curr["target_start"]),
                    duration=float(curr["duration"]),
                )
                if not passed:
                    continue
                if float(avg) + max_verify_drop < float(base_avg):
                    continue
                audio_passed, audio_meta = self.quick_verify_audio(
                    source=curr["source"],
                    source_start=float(cand),
                    target_start=float(curr["target_start"]),
                    duration=float(curr["duration"]),
                    combined_score=float(avg),
                )
                if not audio_passed:
                    continue
                realigned_start, shift_meta, shifted_audio_meta = self.try_audio_guard_shift_realign(
                    source=curr["source"],
                    source_start=float(cand),
                    target_start=float(curr["target_start"]),
                    duration=float(curr["duration"]),
                    combined_score=float(avg),
                    audio_meta=audio_meta,
                )
                if bool(shift_meta.get("applied", False)):
                    cand = float(realigned_start)
                    avg = max(float(avg), float(shift_meta.get("candidate_verify_avg", avg)))
                    if isinstance(shifted_audio_meta, dict):
                        audio_meta = shifted_audio_meta
                else:
                    shift_meta = dict(shift_meta or {})
                aligned = float(audio_meta.get("aligned_similarity", 0.0) or 0.0)
                shift_gain = float(audio_meta.get("shift_gain", 0.0) or 0.0)
                boundary_sim = self._boundary_audio_similarity(
                    source=curr["source"],
                    target_boundary_sec=float(curr["target_start"]),
                    source_boundary_sec=float(cand),
                    window_sec=float(boundary_audio_window),
                )
                forward_bias = float(max(0.0, float(off))) * score_bias
                score = (
                    float(avg)
                    + float(forward_bias)
                    + float(aligned) * 0.03
                    + float(boundary_sim) * float(boundary_audio_weight)
                    - float(shift_gain) * 0.06
                )
                if float(off) > 1e-6:
                    required_gain = float(forward_gain_trigger)
                elif float(off) < -1e-6:
                    required_gain = float(backward_gain_trigger)
                else:
                    required_gain = 0.0
                if bool(best.get("base_audio_failed", False)):
                    # 基线音频失败时，放宽收益门槛，只要有净提升就允许替换。
                    required_gain = 0.0
                if score > float(best["score"]) + float(required_gain) + 1e-6:
                    best = {
                        "start": float(cand),
                        "avg": float(avg),
                        "offset": float(off),
                        "score": float(score),
                        "audio_meta": audio_meta,
                        "audio_shift_meta": shift_meta,
                        "boundary_sim": float(boundary_sim),
                        "base_audio_failed": bool(best.get("base_audio_failed", False)),
                    }

            if abs(float(best["start"]) - curr_start) <= 1e-6:
                continue

            curr["start"] = float(best["start"])
            q = curr.get("quality", {}) or {}
            q["cross_source_head_nudged_no_target"] = True
            q["cross_source_head_nudge_offset_sec"] = float(best["offset"])
            q["cross_source_head_nudge_verify_avg"] = float(best["avg"])
            q["cross_source_head_nudge_prev_tail_left"] = float(prev_tail_left)
            q["cross_source_head_nudge_direction"] = (
                "backward" if float(best["offset"]) < -1e-6 else "forward"
            )
            q["cross_source_head_nudge_boundary_audio_sim"] = float(best.get("boundary_sim", 0.0))
            q["cross_source_head_nudge_audio_guard"] = best["audio_meta"]
            q["cross_source_head_nudge_audio_shift_fix"] = best.get("audio_shift_meta", {})
            curr["quality"] = q
            adjusted += 1

        return int(adjusted)

    def _rebalance_neighbor_recovered_segments_no_target(self, segments: List[dict]) -> int:
        """
        邻段恢复后置重平衡（禁兜底）：
        - 仅处理 recover_mode=neighbors_last_resort_no_target 的段；
        - 以最终邻段为准重新计算边界窗口，避免“上游段后续改源后”把重叠集中压到单侧；
        - 若窗口反转（不可避免重叠），取中点均摊，降低局部重读感。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 3:
            return 0

        adjusted = 0
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            q = seg.get("quality", {}) or {}
            recover_mode = str(q.get("recover_mode", "") or "")
            if recover_mode != "neighbors_last_resort_no_target":
                continue
            source = seg.get("source")
            if source in (None, self.target_video):
                continue

            prev = segments[i - 1]
            nxt = segments[i + 1]
            if prev.get("source") != source or nxt.get("source") != source:
                continue
            if prev.get("source") == self.target_video or nxt.get("source") == self.target_video:
                continue

            prev_end = float(prev["start"]) + float(prev["duration"])
            next_start = float(nxt["start"])
            seg_duration = float(seg["duration"])
            overlap_allow = max(0.35, seg_duration * 0.07)
            # 前源尾巴接入后，neighbors_last_resort 段已经被整体前移过一次；
            # 此时再保留 0.35s 级别的重叠窗口，很容易把重复内容重新带回成片。
            if bool(q.get("cross_source_prev_tail_carryover_shifted_no_target", False)):
                overlap_allow = min(overlap_allow, max(0.08, seg_duration * 0.02))
            lower_bound = prev_end - overlap_allow
            upper_bound = next_start - seg_duration + overlap_allow
            compressed_window = bool(lower_bound > (upper_bound + 1e-6))

            if compressed_window:
                cand_start = 0.5 * (lower_bound + upper_bound)
            else:
                cand_start = min(max(float(seg["start"]), lower_bound), upper_bound)

            src_duration = self.get_video_duration(source)
            if src_duration <= 0.0:
                continue
            max_start = max(0.0, float(src_duration - seg_duration))
            cand_start = max(0.0, min(float(cand_start), max_start))
            old_start = float(seg["start"])
            if abs(cand_start - old_start) < 0.04:
                continue
            prev_gap_before = float(old_start - prev_end)
            next_gap_before = float(next_start - (old_start + seg_duration))
            severe_overlap = bool((prev_gap_before < -1.0) or (next_gap_before < -1.0))
            low_confidence_recover = bool(float(q.get("combined", 0.0)) <= 0.90)
            force_continuity = bool(low_confidence_recover and (compressed_window or severe_overlap))

            offsets = [
                0.0,
                seg_duration * 0.25,
                seg_duration * 0.5,
                seg_duration * 0.75,
                max(0.0, seg_duration - 0.1),
            ]
            passed, avg = self.verify_segment_visual(
                source=source,
                source_start=float(cand_start),
                target_start=float(seg["target_start"]),
                duration=seg_duration,
                offsets=offsets,
                min_avg=0.70,
                min_floor=0.45,
            )
            forced_no_verify = False
            audio_meta: Dict[str, object] = {"checked": False}
            shift_meta: Dict[str, object] = {"applied": False, "checked": False}
            if not passed:
                if not force_continuity:
                    continue
                forced_no_verify = True
                avg = float(q.get("combined", 0.0))
                audio_meta = {
                    "checked": False,
                    "forced_no_verify": True,
                    "reason": "recover_rebalance_visual_failed_force_continuity",
                }
                shift_meta = {"applied": False, "checked": False, "reason": "forced_no_verify"}
            else:
                audio_passed, audio_meta = self.quick_verify_audio(
                    source=source,
                    source_start=float(cand_start),
                    target_start=float(seg["target_start"]),
                    duration=seg_duration,
                    combined_score=float(avg),
                )
                if not audio_passed:
                    if not force_continuity:
                        continue
                    forced_no_verify = True
                    audio_meta = {
                        "checked": False,
                        "forced_no_verify": True,
                        "reason": "recover_rebalance_audio_failed_force_continuity",
                    }
                    shift_meta = {"applied": False, "checked": False, "reason": "forced_no_verify"}
                else:
                    realigned_start, shift_meta, shifted_audio_meta = self.try_audio_guard_shift_realign(
                        source=source,
                        source_start=float(cand_start),
                        target_start=float(seg["target_start"]),
                        duration=seg_duration,
                        combined_score=float(avg),
                        audio_meta=audio_meta,
                    )
                    if bool(shift_meta.get("applied", False)):
                        cand_start = float(realigned_start)
                        avg = max(float(avg), float(shift_meta.get("candidate_verify_avg", avg)))
                        if isinstance(shifted_audio_meta, dict):
                            audio_meta = shifted_audio_meta
                    else:
                        shift_meta = dict(shift_meta or {})

            seg["start"] = float(cand_start)
            q["recover_rebalanced_no_target"] = True
            q["recover_rebalanced_from"] = float(old_start)
            q["recover_rebalanced_to"] = float(cand_start)
            q["recover_rebalanced_verify_avg"] = float(avg)
            q["recover_rebalanced_bounds"] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "compressed_window": bool(compressed_window),
            }
            q["recover_rebalanced_prev_gap_before"] = float(prev_gap_before)
            q["recover_rebalanced_next_gap_before"] = float(next_gap_before)
            q["recover_rebalanced_forced_no_verify"] = bool(forced_no_verify)
            q["audio_guard"] = audio_meta
            q["recover_rebalanced_audio_shift_fix"] = shift_meta
            seg["quality"] = q
            adjusted += 1

        return int(adjusted)

    def _repair_same_source_step_lag_runs_no_target(self, segments: List[dict]) -> int:
        """
        禁兜底同源步长滞后链修复：
        - 识别同源 run 中连续“source 步长明显小于 target 步长”的链；
        - 逐段向前校正起点，优先消除重复词/局部倒放观感；
        - 以“逼近期望步长”为主目标，并通过 visual/audio 复核。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 3:
            return 0

        fixed = 0
        i = 0
        while i < len(segments):
            src = segments[i]["source"]
            if src == self.target_video:
                i += 1
                continue
            j = i + 1
            while j < len(segments) and segments[j]["source"] == src:
                j += 1
            if (j - i) < 3:
                i = j
                continue
            if any(self._is_tail_sensitive_last_resort_segment(segments[k]) for k in range(i, j)):
                i = j
                continue

            durations = [float(segments[k]["duration"]) for k in range(i, j)]
            base_duration = float(np.median(durations)) if durations else 5.0
            lag_trigger = max(0.55, base_duration * 0.12)
            lag_boundary_count = 0
            for k in range(i + 1, j):
                prev = segments[k - 1]
                curr = segments[k]
                target_step = float(curr["target_start"]) - float(prev["target_start"])
                if target_step <= 0.0:
                    continue
                source_step = float(curr["start"]) - float(prev["start"])
                if (target_step - source_step) > lag_trigger:
                    lag_boundary_count += 1
            if lag_boundary_count < 2:
                i = j
                continue

            for k in range(i + 1, j):
                prev = segments[k - 1]
                curr = segments[k]
                target_step = float(curr["target_start"]) - float(prev["target_start"])
                if target_step <= 1e-6:
                    continue

                expected_start = float(prev["start"]) + float(target_step)
                curr_start = float(curr["start"])
                needed_shift = expected_start - curr_start
                if needed_shift <= 0.06:
                    continue

                src_duration = self.get_video_duration(curr["source"])
                if src_duration <= 0.0:
                    continue
                max_start = max(0.0, float(src_duration - float(curr["duration"])))
                max_single_shift = max(0.8, float(curr["duration"]) * 0.56)
                needed_shift = min(float(needed_shift), float(max_single_shift))
                if needed_shift <= 0.06:
                    continue

                candidate_starts_raw = [
                    curr_start + needed_shift,
                    curr_start + min(needed_shift * 0.80, max_single_shift),
                    curr_start + min(needed_shift * 0.60, max_single_shift),
                ]
                candidate_starts: List[float] = []
                seen: Set[float] = set()
                for c in candidate_starts_raw:
                    cand = max(0.0, min(float(c), max_start))
                    key = round(cand, 3)
                    if key in seen:
                        continue
                    seen.add(key)
                    if cand <= curr_start + 0.04:
                        continue
                    candidate_starts.append(float(cand))
                if not candidate_starts:
                    continue

                best_choice: Optional[Dict[str, object]] = None
                best_key: Optional[Tuple[float, float]] = None
                original_start = float(curr["start"])
                for cand_start in candidate_starts:
                    offsets = [
                        0.0,
                        float(curr["duration"]) * 0.25,
                        float(curr["duration"]) * 0.5,
                        float(curr["duration"]) * 0.75,
                        max(0.0, float(curr["duration"]) - 0.1),
                    ]
                    passed, avg = self.verify_segment_visual(
                        source=curr["source"],
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        offsets=offsets,
                        min_avg=max(0.66, float(self.strict_verify_min_sim) - 0.12),
                        min_floor=max(0.42, float(self.strict_verify_min_sim) - 0.28),
                    )
                    if not passed:
                        continue
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=curr["source"],
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=float(avg),
                    )
                    if not audio_passed:
                        continue
                    realigned_start, shift_meta, shifted_audio_meta = self.try_audio_guard_shift_realign(
                        source=curr["source"],
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=float(avg),
                        audio_meta=audio_meta,
                    )
                    if bool(shift_meta.get("applied", False)):
                        cand_start = float(realigned_start)
                        avg = max(float(avg), float(shift_meta.get("candidate_verify_avg", avg)))
                        if isinstance(shifted_audio_meta, dict):
                            audio_meta = shifted_audio_meta
                    else:
                        shift_meta = dict(shift_meta or {})

                    residual = abs(float(expected_start) - float(cand_start))
                    key = (float(residual), -float(avg))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_choice = {
                            "start": float(cand_start),
                            "avg": float(avg),
                            "residual": float(residual),
                            "audio_meta": audio_meta,
                            "audio_shift_meta": shift_meta,
                        }

                if best_choice is None:
                    continue

                curr["start"] = float(best_choice["start"])
                q = curr.get("quality", {}) or {}
                q["step_lag_repaired_no_target"] = True
                q["step_lag_repair_from"] = float(original_start)
                q["step_lag_repair_to"] = float(best_choice["start"])
                q["step_lag_repair_needed_shift"] = float(needed_shift)
                q["step_lag_repair_expected_start"] = float(expected_start)
                q["step_lag_repair_expected_residual"] = float(best_choice["residual"])
                q["step_lag_repair_verify_avg"] = float(best_choice["avg"])
                q["audio_guard"] = best_choice["audio_meta"]
                q["step_lag_repair_audio_shift_fix"] = best_choice.get("audio_shift_meta", {})
                curr["quality"] = q
                fixed += 1

            i = j

        return int(fixed)

    def _cleanup_unresolved_boundaries_post_lag_no_target(self, segments: List[dict], max_passes: int = 2) -> int:
        """
        滞后回正后的边界清理（禁兜底）：
        - 对仍未收敛的同源边界，优先将右段贴近 prev_end；
        - 仅接受 visual/audio 复核通过且边界超限不变差的候选。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        for _ in range(max(1, int(max_passes))):
            changed = False
            for i in range(1, len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                nxt = segments[i + 1] if (i + 1) < len(segments) else None
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    continue
                if prev["source"] != curr["source"]:
                    continue
                curr_q = curr.get("quality", {}) or {}
                # 前源尾巴接入后的同源 run 已经被专门前移过；
                # 再走 post-lag cleanup 会把整条 run 按 prev_end 拉回去，
                # 直接抵消 carryover 修复，重新制造 2:49 这类“有占位但缺字”的问题。
                if (
                    bool(curr_q.get("cross_source_prev_tail_carryover_shifted_no_target", False))
                    or bool(curr_q.get("cross_source_head_only_shifted_no_target", False))
                ):
                    continue
                if self._is_tail_sensitive_last_resort_segment(prev) or self._is_tail_sensitive_last_resort_segment(curr):
                    continue
                if (
                    nxt is not None
                    and nxt.get("source") == curr["source"]
                    and self._is_tail_sensitive_last_resort_segment(nxt)
                ):
                    continue

                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = float(curr_start - prev_end)
                neg_trigger = max(0.12, float(curr["duration"]) * 0.02)
                pos_trigger = max(0.20, float(curr["duration"]) * 0.04)
                if (gap >= -neg_trigger) and (gap <= pos_trigger):
                    continue

                current_excess = 0.0
                if gap < -neg_trigger:
                    current_excess = float((-gap) - neg_trigger)
                elif gap > pos_trigger:
                    current_excess = float(gap - pos_trigger)

                source_duration = self.get_video_duration(curr["source"])
                if source_duration <= 0.0:
                    continue
                max_start = max(0.0, float(source_duration - float(curr["duration"])))
                anchor = max(0.0, min(float(prev_end), max_start))

                raw_candidates = [
                    anchor,
                    anchor - 0.20,
                    anchor + 0.20,
                    anchor - 0.50,
                    anchor + 0.50,
                ]
                candidates: List[float] = []
                seen: Set[float] = set()
                for raw in raw_candidates:
                    cand = max(0.0, min(float(raw), max_start))
                    key = round(cand, 3)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(float(cand))

                best_choice: Optional[Dict[str, object]] = None
                best_key: Optional[Tuple[float, float, float]] = None
                for cand_start in candidates:
                    after_gap = float(cand_start - prev_end)
                    after_excess = 0.0
                    if after_gap < -neg_trigger:
                        after_excess = float((-after_gap) - neg_trigger)
                    elif after_gap > pos_trigger:
                        after_excess = float(after_gap - pos_trigger)
                    if after_excess > (current_excess + 1e-6):
                        continue

                    offsets = [
                        0.0,
                        float(curr["duration"]) * 0.25,
                        float(curr["duration"]) * 0.5,
                        float(curr["duration"]) * 0.75,
                        max(0.0, float(curr["duration"]) - 0.1),
                    ]
                    passed, avg = self.verify_segment_visual(
                        source=curr["source"],
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        offsets=offsets,
                        min_avg=max(0.68, float(self.strict_verify_min_sim) - 0.10),
                        min_floor=max(0.45, float(self.strict_verify_min_sim) - 0.25),
                    )
                    if not passed:
                        continue
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=curr["source"],
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=float(avg),
                    )
                    if not audio_passed:
                        continue
                    realigned_start, shift_meta, shifted_audio_meta = self.try_audio_guard_shift_realign(
                        source=curr["source"],
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=float(avg),
                        audio_meta=audio_meta,
                    )
                    if bool(shift_meta.get("applied", False)):
                        cand_start = float(realigned_start)
                        avg = max(float(avg), float(shift_meta.get("candidate_verify_avg", avg)))
                        if isinstance(shifted_audio_meta, dict):
                            audio_meta = shifted_audio_meta
                    else:
                        shift_meta = dict(shift_meta or {})

                    after_gap = float(cand_start - prev_end)
                    after_excess = 0.0
                    if after_gap < -neg_trigger:
                        after_excess = float((-after_gap) - neg_trigger)
                    elif after_gap > pos_trigger:
                        after_excess = float(after_gap - pos_trigger)
                    if after_excess > (current_excess + 1e-6):
                        continue

                    key = (float(after_excess), abs(float(cand_start - anchor)), -float(avg))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_choice = {
                            "start": float(cand_start),
                            "avg": float(avg),
                            "after_gap": float(after_gap),
                            "after_excess": float(after_excess),
                            "audio_meta": audio_meta,
                            "audio_shift_meta": shift_meta,
                        }

                if best_choice is None:
                    continue
                if abs(float(best_choice["start"]) - curr_start) <= 1e-6:
                    continue

                curr["start"] = float(best_choice["start"])
                q = curr.get("quality", {}) or {}
                q["post_lag_boundary_cleanup_no_target"] = True
                q["post_lag_boundary_cleanup_from"] = float(curr_start)
                q["post_lag_boundary_cleanup_to"] = float(best_choice["start"])
                q["post_lag_boundary_cleanup_before_gap"] = float(gap)
                q["post_lag_boundary_cleanup_after_gap"] = float(best_choice["after_gap"])
                q["post_lag_boundary_cleanup_after_excess"] = float(best_choice["after_excess"])
                q["post_lag_boundary_cleanup_verify_avg"] = float(best_choice["avg"])
                q["audio_guard"] = best_choice["audio_meta"]
                q["post_lag_boundary_cleanup_audio_shift_fix"] = best_choice.get("audio_shift_meta", {})
                curr["quality"] = q
                adjusted += 1
                changed = True

            if not changed:
                break

        return int(adjusted)

    def _resolve_severe_same_source_overlaps_no_target(self, segments: List[dict], max_passes: int = 2) -> int:
        """
        禁兜底同源严重负重叠 run 前推修复：
        - 当同源边界出现明显负重叠（例如 1~3 秒）时，将该边界右侧整段 run 同步前推；
        - 优先消除“重复词/倒放感”，避免把重叠反复在局部边界间转移；
        - 对低置信或邻段强制恢复段允许连续性优先强制落地。
        """
        if self.enable_target_video_fallback:
            return 0
        if not bool(getattr(self, "no_target_severe_overlap_run_fix", True)):
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        severe_floor = max(0.35, float(getattr(self, "no_target_severe_overlap_trigger", 0.85)))
        safe_tol = max(0.0, float(getattr(self, "no_target_severe_overlap_safe_tol", 0.02)))
        force_combined_max = min(
            1.0,
            max(0.0, float(getattr(self, "no_target_severe_overlap_force_combined_max", 0.92))),
        )

        for _ in range(max(1, int(max_passes))):
            changed = False
            i = 1
            while i < len(segments):
                prev = segments[i - 1]
                curr = segments[i]
                nxt = segments[i + 1] if (i + 1) < len(segments) else None
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    i += 1
                    continue
                if prev["source"] != curr["source"]:
                    i += 1
                    continue
                if self._is_tail_sensitive_last_resort_segment(prev) or self._is_tail_sensitive_last_resort_segment(curr):
                    i += 1
                    continue
                if (
                    nxt is not None
                    and nxt.get("source") == curr["source"]
                    and self._is_tail_sensitive_last_resort_segment(nxt)
                ):
                    i += 1
                    continue

                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = float(curr_start - prev_end)
                severe_trigger = max(float(severe_floor), float(curr["duration"]) * 0.16)
                if gap >= -severe_trigger:
                    i += 1
                    continue

                src = curr["source"]
                src_duration = self.get_video_duration(src)
                if src_duration <= 0.0:
                    i += 1
                    continue

                run_end = i
                while (
                    (run_end + 1) < len(segments)
                    and segments[run_end + 1]["source"] == src
                    and segments[run_end + 1]["source"] != self.target_video
                ):
                    run_end += 1

                required_shift = max(0.0, (float(prev_end) - float(curr_start)) + float(safe_tol))
                tail_end = float(segments[run_end]["start"]) + float(segments[run_end]["duration"])
                run_headroom = max(0.0, float(src_duration) - float(tail_end))
                applied_shift = min(float(required_shift), float(run_headroom))
                if applied_shift <= 1e-6:
                    i = run_end + 1
                    continue

                curr_q = curr.get("quality", {}) or {}
                curr_combined = float(curr_q.get("combined", 0.0))
                curr_recover_mode = str(curr_q.get("recover_mode", "") or "")
                force_continuity = bool(
                    (abs(float(gap)) >= max(1.20, float(severe_trigger) * 1.5))
                    or (curr_combined <= force_combined_max)
                    or (curr_recover_mode in {"neighbors_last_resort_no_target", "neighbors_forced_no_verify"})
                )

                rollback: Dict[int, float] = {}
                for k in range(i, run_end + 1):
                    rollback[k] = float(segments[k]["start"])
                    segments[k]["start"] = float(segments[k]["start"]) + float(applied_shift)

                new_gap = float(segments[i]["start"]) - float(prev_end)
                improved = bool((abs(float(new_gap)) + 1e-6) < abs(float(gap)))
                safe_after = bool(float(new_gap) >= (-float(safe_tol) - 1e-6))
                if (not improved) or (not safe_after):
                    for k, old_start in rollback.items():
                        segments[k]["start"] = old_start
                    i = run_end + 1
                    continue

                verify_avg = float(curr_combined)
                audio_meta: Dict[str, object] = {"checked": False}
                if not force_continuity:
                    passed, verify_avg = self.quick_verify(
                        source=src,
                        source_start=float(segments[i]["start"]),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                    )
                    if not passed:
                        for k, old_start in rollback.items():
                            segments[k]["start"] = old_start
                        i = run_end + 1
                        continue
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=src,
                        source_start=float(segments[i]["start"]),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=max(float(curr_combined), float(verify_avg)),
                    )
                    if not audio_passed:
                        for k, old_start in rollback.items():
                            segments[k]["start"] = old_start
                        i = run_end + 1
                        continue
                else:
                    audio_meta = {
                        "checked": False,
                        "forced_no_verify": True,
                        "reason": "severe_overlap_run_shift_force_continuity",
                    }

                for k in range(i, run_end + 1):
                    seg = segments[k]
                    q = seg.get("quality", {}) or {}
                    old_start = float(rollback[k])
                    q["severe_overlap_run_shift_no_target"] = True
                    q["severe_overlap_run_shift_from"] = float(old_start)
                    q["severe_overlap_run_shift_to"] = float(seg["start"])
                    q["severe_overlap_run_shift_delta"] = float(applied_shift)
                    q["severe_overlap_run_shift_anchor_index"] = int(i)
                    if k == i:
                        q["severe_overlap_run_shift_before_gap"] = float(gap)
                        q["severe_overlap_run_shift_after_gap"] = float(new_gap)
                        q["severe_overlap_run_shift_force_continuity"] = bool(force_continuity)
                        q["severe_overlap_run_shift_verify_avg"] = float(verify_avg)
                        q["audio_guard"] = audio_meta
                    seg["quality"] = q

                adjusted += int(run_end - i + 1)
                changed = True
                i = run_end + 1

            if not changed:
                break

        return int(adjusted)

    def _redistribute_mild_run_overflow_no_target(self, segments: List[dict]) -> int:
        """
        禁兜底同源 run 轻量溢出分摊：
        - 当 run 总时长略大于源可用容量时，把溢出量按边界均匀分摊；
        - 仅处理“每边界分摊量较小”的 run，避免引入大范围错位；
        - 每段都需通过 visual/audio 复核后才落地。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 3:
            return 0

        max_per_boundary = max(
            0.04,
            float(getattr(self, "no_target_run_overflow_max_per_boundary", 0.16)),
        )
        tiny_forward_freeze = max(
            0.0,
            float(getattr(self, "no_target_run_overflow_tiny_forward_freeze", 0.03)),
        )
        adjusted = 0
        i = 0
        while i < len(segments):
            src = segments[i]["source"]
            if src == self.target_video:
                i += 1
                continue

            j = i + 1
            while j < len(segments) and segments[j]["source"] == src:
                j += 1
            run_len = j - i
            if run_len < 3:
                i = j
                continue
            if any(self._is_tail_sensitive_last_resort_segment(segments[k]) for k in range(i, j)):
                i = j
                continue

            src_duration = self.get_video_duration(src)
            if src_duration <= 0.0:
                i = j
                continue

            first_start = float(segments[i]["start"])
            total_duration = sum(float(segments[k]["duration"]) for k in range(i, j))
            overflow = max(0.0, float(first_start + total_duration - src_duration))
            boundaries = run_len - 1
            if overflow <= 1e-6 or boundaries <= 0:
                i = j
                continue
            per_boundary = float(overflow / float(boundaries))
            if per_boundary > max_per_boundary:
                i = j
                continue

            proposed: List[float] = [float(first_start)]
            for k in range(i + 1, j):
                prev_seg = segments[k - 1]
                proposed.append(float(proposed[-1] + float(prev_seg["duration"]) - per_boundary))

            updates: Dict[int, Dict[str, object]] = {}
            run_ok = True
            fail_count = 0
            for k in range(i + 1, j):
                seg = segments[k]
                old_start = float(seg["start"])
                cand_start = float(proposed[k - i])
                max_start = max(0.0, float(src_duration - float(seg["duration"])))
                cand_start = max(0.0, min(float(cand_start), max_start))
                prev_after_start = float(updates.get(k - 1, {}).get("start", float(segments[k - 1]["start"])))
                prev_after_end = float(prev_after_start + float(segments[k - 1]["duration"]))
                prev_gap_after = float(cand_start - prev_after_end)
                prev_neg_trigger = max(0.12, float(seg["duration"]) * 0.02)
                if prev_gap_after < -prev_neg_trigger:
                    run_ok = False
                    fail_count += 1
                    continue
                # 末端小幅“向后推”更容易触发半秒级音频偏置；在极小差值下保守保留原起点。
                if (cand_start > old_start) and ((cand_start - old_start) <= tiny_forward_freeze):
                    cand_start = float(old_start)
                if abs(cand_start - old_start) <= 0.02:
                    continue

                offsets = [
                    0.0,
                    float(seg["duration"]) * 0.25,
                    float(seg["duration"]) * 0.5,
                    float(seg["duration"]) * 0.75,
                    max(0.0, float(seg["duration"]) - 0.1),
                ]
                passed, avg = self.verify_segment_visual(
                    source=src,
                    source_start=float(cand_start),
                    target_start=float(seg["target_start"]),
                    duration=float(seg["duration"]),
                    offsets=offsets,
                    min_avg=max(0.70, float(self.strict_verify_min_sim) - 0.08),
                    min_floor=max(0.45, float(self.strict_verify_min_sim) - 0.24),
                )
                if not passed:
                    run_ok = False
                    fail_count += 1
                    continue

                audio_passed, audio_meta = self.quick_verify_audio(
                    source=src,
                    source_start=float(cand_start),
                    target_start=float(seg["target_start"]),
                    duration=float(seg["duration"]),
                    combined_score=float(avg),
                )
                if not audio_passed:
                    run_ok = False
                    fail_count += 1
                    continue
                realigned_start, shift_meta, shifted_audio_meta = self.try_audio_guard_shift_realign(
                    source=src,
                    source_start=float(cand_start),
                    target_start=float(seg["target_start"]),
                    duration=float(seg["duration"]),
                    combined_score=float(avg),
                    audio_meta=audio_meta,
                )
                if bool(shift_meta.get("applied", False)):
                    cand_start = float(realigned_start)
                    avg = max(float(avg), float(shift_meta.get("candidate_verify_avg", avg)))
                    if isinstance(shifted_audio_meta, dict):
                        audio_meta = shifted_audio_meta
                else:
                    shift_meta = dict(shift_meta or {})

                updates[k] = {
                    "start": float(cand_start),
                    "old_start": float(old_start),
                    "avg": float(avg),
                    "audio_meta": audio_meta,
                    "audio_shift_meta": shift_meta,
                    "prev_gap_after": float(prev_gap_after),
                    "prev_neg_trigger": float(prev_neg_trigger),
                }

            if run_ok and updates:
                for k, meta in updates.items():
                    seg = segments[k]
                    seg["start"] = float(meta["start"])
                    q = seg.get("quality", {}) or {}
                    q["run_overflow_redistributed_no_target"] = True
                    q["run_overflow_redistributed_from"] = float(meta["old_start"])
                    q["run_overflow_redistributed_to"] = float(meta["start"])
                    q["run_overflow_redistributed_per_boundary"] = float(per_boundary)
                    q["run_overflow_redistributed_overflow"] = float(overflow)
                    q["run_overflow_redistributed_verify_avg"] = float(meta["avg"])
                    q["audio_guard"] = meta["audio_meta"]
                    q["run_overflow_redistributed_audio_shift_fix"] = meta.get("audio_shift_meta", {})
                    seg["quality"] = q
                    adjusted += 1
            elif updates and fail_count > 0 and overflow > 0.35:
                # 部分段复核失败时，对“低置信/强制恢复”段启用连续性优先的保守落地，
                # 目的是避免尾部把重叠集中到单一边界（常见表现：末段重读/倒放感）。
                forced_updates = 0
                for k, meta in updates.items():
                    seg = segments[k]
                    q = seg.get("quality", {}) or {}
                    recover_mode = str(q.get("recover_mode", "") or "")
                    combined = float(q.get("combined", 0.0))
                    tail_zone = bool(k >= (j - 3))
                    forceable = bool(
                        (combined <= 0.90)
                        or (recover_mode in {"neighbors_last_resort_no_target", "neighbors_forced_no_verify"})
                        or tail_zone
                    )
                    if not forceable:
                        continue
                    prev_gap_after = float(meta.get("prev_gap_after", 0.0))
                    prev_neg_trigger = float(meta.get("prev_neg_trigger", max(0.12, float(seg["duration"]) * 0.02)))
                    if prev_gap_after < -prev_neg_trigger:
                        continue
                    seg["start"] = float(meta["start"])
                    q["run_overflow_redistributed_no_target"] = True
                    q["run_overflow_redistributed_from"] = float(meta["old_start"])
                    q["run_overflow_redistributed_to"] = float(meta["start"])
                    q["run_overflow_redistributed_per_boundary"] = float(per_boundary)
                    q["run_overflow_redistributed_overflow"] = float(overflow)
                    q["run_overflow_redistributed_verify_avg"] = float(meta["avg"])
                    q["audio_guard"] = meta["audio_meta"]
                    q["run_overflow_redistributed_audio_shift_fix"] = meta.get("audio_shift_meta", {})
                    q["run_overflow_redistributed_force_continuity"] = True
                    q["run_overflow_redistributed_fail_count"] = int(fail_count)
                    seg["quality"] = q
                    adjusted += 1
                    forced_updates += 1

            i = j

        return int(adjusted)

    def _boundary_audio_similarity(
        self,
        source: Path,
        target_boundary_sec: float,
        source_boundary_sec: float,
        window_sec: float,
    ) -> float:
        """边界音频相似度探针（短窗）。"""
        clip_dur = max(0.25, float(window_sec))
        tgt_start = max(0.0, float(target_boundary_sec) - clip_dur * 0.5)
        src_start = max(0.0, float(source_boundary_sec) - clip_dur * 0.5)
        target_fp = self._get_audio_fp_cached(self.target_video, tgt_start, clip_dur)
        source_fp = self._get_audio_fp_cached(source, src_start, clip_dur)
        if not target_fp or not source_fp:
            return 0.0
        return float(compare_chromaprint(target_fp, source_fp))

    def _boundary_audio_silence_mismatch(
        self,
        source: Path,
        target_boundary_sec: float,
        source_boundary_sec: float,
        window_sec: float,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        边界静音错配探针：
        - 目标边界有明显语音，但候选边界接近静音时判为 mismatch。
        """
        clip_dur = max(0.25, float(window_sec))
        tgt_start = max(0.0, float(target_boundary_sec) - clip_dur * 0.5)
        src_start = max(0.0, float(source_boundary_sec) - clip_dur * 0.5)
        tgt_db = self._audio_dbfs_sample(self.target_video, tgt_start, clip_dur)
        src_db = self._audio_dbfs_sample(source, src_start, clip_dur)
        mismatch = bool(
            (src_db is not None)
            and (float(src_db) <= -45.0)
            and ((tgt_db is None) or (float(tgt_db) > -40.0))
        )
        return mismatch, src_db, tgt_db

    def _enforce_boundary_hard_constraints_no_target(self, segments: List[dict], max_passes: int = 2) -> int:
        """
        边界硬约束（禁兜底）：
        - 同源相邻段禁止负重叠（仅允许极小容差）；
        - 同时抑制明显正缝隙，减少丢词风险。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        neg_tol = max(0.0, float(getattr(self, "boundary_hard_max_negative_overlap", 0.01)))
        pos_tol = max(0.0, float(getattr(self, "boundary_hard_max_positive_gap", 0.06)))
        max_shift_sec = max(0.0, float(getattr(self, "boundary_hard_max_shift_sec", 1.2)))
        for _ in range(max(1, int(max_passes))):
            changed = False
            for i in range(1, len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    continue
                if prev["source"] != curr["source"]:
                    continue

                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = float(curr_start - prev_end)
                if gap >= -neg_tol and gap <= pos_tol:
                    continue

                src = curr["source"]
                src_duration = self.get_video_duration(src)
                if src_duration <= 0.0:
                    continue
                max_start = max(0.0, float(src_duration - float(curr["duration"])))
                cand_start = max(0.0, min(float(prev_end), max_start))
                if abs(cand_start - curr_start) <= 1e-6:
                    continue
                shift_sec = abs(float(cand_start) - float(curr_start))
                if max_shift_sec > 1e-6 and shift_sec > (max_shift_sec + 1e-6):
                    q = curr.get("quality", {}) or {}
                    q["boundary_hard_skipped_large_shift_no_target"] = True
                    q["boundary_hard_skip_shift_sec"] = float(shift_sec)
                    q["boundary_hard_skip_max_shift_sec"] = float(max_shift_sec)
                    q["boundary_hard_skip_before_gap"] = float(gap)
                    curr["quality"] = q
                    continue

                passed, avg = self.quick_verify(
                    source=src,
                    source_start=float(cand_start),
                    target_start=float(curr["target_start"]),
                    duration=float(curr["duration"]),
                )
                if not passed:
                    continue
                audio_passed, audio_meta = self.quick_verify_audio(
                    source=src,
                    source_start=float(cand_start),
                    target_start=float(curr["target_start"]),
                    duration=float(curr["duration"]),
                    combined_score=float(avg),
                )
                if not audio_passed:
                    continue

                curr["start"] = float(cand_start)
                after_gap = float(cand_start - prev_end)
                q = curr.get("quality", {}) or {}
                q["boundary_hard_clamped_no_target"] = True
                q["boundary_hard_from"] = float(curr_start)
                q["boundary_hard_to"] = float(cand_start)
                q["boundary_hard_before_gap"] = float(gap)
                q["boundary_hard_after_gap"] = float(after_gap)
                q["boundary_hard_verify_avg"] = float(avg)
                q["audio_guard"] = audio_meta
                curr["quality"] = q
                adjusted += 1
                changed = True

            if not changed:
                break

        return int(adjusted)

    def _repair_boundary_audio_locally_no_target(self, segments: List[dict], max_passes: int = 2) -> Tuple[int, int]:
        """
        边界音频局部迭代修复（禁兜底）：
        - 每个同源边界做短窗音频探针（±0.3~0.5s）；
        - 命中重读/丢词/静音风险时，只局部重算当前边界起点；
        - 不触发整片重跑。
        返回: (修复数, 发现问题边界数)
        """
        if self.enable_target_video_fallback:
            return 0, 0
        if len(segments) < 2:
            return 0, 0

        repaired = 0
        issues = 0
        window_sec = max(0.3, float(getattr(self, "boundary_audio_probe_window_sec", 0.45)))
        gain_trigger = max(0.02, float(getattr(self, "boundary_audio_expected_gain_trigger", 0.08)))
        max_offset = max(0.1, float(getattr(self, "boundary_audio_repair_max_offset", 0.5)))
        neg_tol = max(0.0, float(getattr(self, "boundary_hard_max_negative_overlap", 0.01)))
        pos_tol = max(0.0, float(getattr(self, "boundary_hard_max_positive_gap", 0.06)))
        repeat_safe_tol = max(
            0.0,
            float(
                getattr(
                    self,
                    "boundary_audio_repeat_max_negative_overlap",
                    max(0.02, float(neg_tol)),
                )
            ),
        )

        offset_candidates = [0.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5]
        offset_candidates = [x for x in offset_candidates if abs(float(x)) <= (max_offset + 1e-6)]
        if 0.0 not in offset_candidates:
            offset_candidates.insert(0, 0.0)

        for _ in range(max(1, int(max_passes))):
            changed = False
            for i in range(1, len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    continue
                if prev["source"] != curr["source"]:
                    continue

                src = curr["source"]
                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = float(curr_start - prev_end)
                target_boundary = float(curr["target_start"])

                sim_curr = self._boundary_audio_similarity(
                    source=src,
                    target_boundary_sec=target_boundary,
                    source_boundary_sec=float(curr_start),
                    window_sec=window_sec,
                )
                sim_anchor = self._boundary_audio_similarity(
                    source=src,
                    target_boundary_sec=target_boundary,
                    source_boundary_sec=float(prev_end),
                    window_sec=window_sec,
                )
                silence_mismatch_before, src_db_before, tgt_db_before = self._boundary_audio_silence_mismatch(
                    source=src,
                    target_boundary_sec=target_boundary,
                    source_boundary_sec=float(curr_start),
                    window_sec=window_sec,
                )

                repeat_risk = bool(gap < -neg_tol)
                drop_risk = bool(gap > pos_tol)
                expected_better = bool(
                    (sim_anchor >= (sim_curr + gain_trigger))
                    and (abs(gap) >= 0.02)
                )
                if not (repeat_risk or drop_risk or expected_better or silence_mismatch_before):
                    continue

                issues += 1
                src_duration = self.get_video_duration(src)
                if src_duration <= 0.0:
                    continue
                max_start = max(0.0, float(src_duration - float(curr["duration"])))
                anchor = max(0.0, min(float(prev_end), max_start))
                nxt: Optional[dict] = None
                next_gap_before = 0.0
                next_penalty_before = 0.0
                next_neg_trigger = 0.0
                next_pos_trigger = 0.0
                if (i + 1) < len(segments):
                    maybe_next = segments[i + 1]
                    if maybe_next["source"] == curr["source"] and maybe_next["source"] != self.target_video:
                        nxt = maybe_next
                        next_gap_before = float(maybe_next["start"]) - (float(curr_start) + float(curr["duration"]))
                        next_neg_trigger = max(0.12, float(maybe_next["duration"]) * 0.02)
                        next_pos_trigger = max(0.20, float(maybe_next["duration"]) * 0.04)
                        if next_gap_before < -next_neg_trigger:
                            next_penalty_before = float((-next_gap_before) - next_neg_trigger)
                        elif next_gap_before > next_pos_trigger:
                            next_penalty_before = float(next_gap_before - next_pos_trigger)
                next_q = (nxt.get("quality", {}) or {}) if nxt is not None else {}
                next_is_carryover_last_resort = bool(
                    nxt is not None
                    and str(next_q.get("recover_mode", "") or "") == "neighbors_last_resort_no_target"
                    and bool(next_q.get("cross_source_prev_tail_carryover_shifted_no_target", False))
                )
                severe_next_penalty_limit = max(0.35, float(next_penalty_before) + 0.18)
                if next_is_carryover_last_resort:
                    severe_next_penalty_limit = min(
                        severe_next_penalty_limit,
                        float(next_penalty_before) + 0.02,
                    )

                curr_q = curr.get("quality", {}) or {}
                curr_combined = float(curr_q.get("combined", 0.0))
                curr_is_carryover_shifted = bool(curr_q.get("cross_source_prev_tail_carryover_shifted_no_target", False))
                def build_anchor_force_candidate(min_avg: float) -> Optional[Dict[str, object]]:
                    force_start = float(anchor)
                    if abs(force_start - float(curr_start)) <= 1e-6:
                        return None
                    passed, avg = self.quick_verify(
                        source=src,
                        source_start=float(force_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                    )
                    if not passed or float(avg) + 1e-6 < float(min_avg):
                        return None
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=src,
                        source_start=float(force_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=max(float(curr_combined), float(avg)),
                    )
                    if not audio_passed:
                        return None
                    sim_force = self._boundary_audio_similarity(
                        source=src,
                        target_boundary_sec=target_boundary,
                        source_boundary_sec=float(force_start),
                        window_sec=window_sec,
                    )
                    gap_force = float(force_start - prev_end)
                    if curr_is_carryover_shifted and gap_force < -0.12:
                        return None
                    if gap_force < (-repeat_safe_tol):
                        return None
                    next_gap_force = 0.0
                    next_penalty_force = 0.0
                    if nxt is not None:
                        next_gap_force = float(nxt["start"]) - (float(force_start) + float(curr["duration"]))
                        if next_gap_force < -next_neg_trigger:
                            next_penalty_force = float((-next_gap_force) - next_neg_trigger)
                        elif next_gap_force > next_pos_trigger:
                            next_penalty_force = float(next_gap_force - next_pos_trigger)
                        if next_penalty_force > severe_next_penalty_limit:
                            return None
                    return {
                        "start": float(force_start),
                        "avg": float(avg),
                        "sim": float(sim_force),
                        "gap_after": float(gap_force),
                        "next_gap_after": float(next_gap_force),
                        "next_boundary_penalty_after": float(next_penalty_force),
                        "audio_meta": audio_meta,
                        "forced_repeat_safe_anchor": True,
                    }
                best_any: Optional[Dict[str, object]] = None
                best_any_score = None
                best_safe: Optional[Dict[str, object]] = None
                best_safe_score = None
                for off in offset_candidates:
                    cand_start = max(0.0, min(float(anchor + float(off)), max_start))
                    next_gap = 0.0
                    next_boundary_penalty = 0.0
                    if nxt is not None:
                        next_gap = float(nxt["start"]) - (float(cand_start) + float(curr["duration"]))
                        if next_gap < -next_neg_trigger:
                            next_boundary_penalty = float((-next_gap) - next_neg_trigger)
                        elif next_gap > next_pos_trigger:
                            next_boundary_penalty = float(next_gap - next_pos_trigger)
                        if next_is_carryover_last_resort and next_gap < -0.12:
                            continue
                        if next_boundary_penalty > severe_next_penalty_limit:
                            continue
                        if (not repeat_risk) and (not drop_risk) and next_boundary_penalty > (float(next_penalty_before) + 0.02):
                            continue

                    passed, avg = self.quick_verify(
                        source=src,
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                    )
                    if not passed and abs(float(cand_start) - float(curr_start)) > 0.03:
                        continue

                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=src,
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=max(float(curr_combined), float(avg)),
                    )
                    if not audio_passed:
                        continue

                    sim_cand = self._boundary_audio_similarity(
                        source=src,
                        target_boundary_sec=target_boundary,
                        source_boundary_sec=float(cand_start),
                        window_sec=window_sec,
                    )
                    gap_after = float(cand_start - prev_end)
                    if curr_is_carryover_shifted and gap_after < -0.12:
                        continue
                    if repeat_risk and gap_after < (float(gap) - 0.03):
                        continue
                    if drop_risk and gap_after > (float(gap) + 0.03):
                        continue
                    overlap_penalty = max(0.0, -gap_after)
                    drop_penalty = max(0.0, gap_after)
                    overlap_weight = 3.4 if repeat_risk else 2.2
                    drop_weight = 1.8 if drop_risk else 1.5
                    score = (
                        float(sim_cand)
                        - float(overlap_penalty) * overlap_weight
                        - float(drop_penalty) * drop_weight
                        - float(next_boundary_penalty) * 0.80
                        - abs(float(cand_start) - float(curr_start)) * 0.01
                    )
                    candidate = {
                        "start": float(cand_start),
                        "avg": float(avg),
                        "sim": float(sim_cand),
                        "gap_after": float(gap_after),
                        "next_gap_after": float(next_gap),
                        "next_boundary_penalty_after": float(next_boundary_penalty),
                        "audio_meta": audio_meta,
                    }
                    if best_any_score is None or score > best_any_score:
                        best_any_score = float(score)
                        best_any = candidate

                    repeat_safe = bool((not repeat_risk) or (gap_after >= (-repeat_safe_tol)))
                    if repeat_safe and (best_safe_score is None or score > best_safe_score):
                        best_safe_score = float(score)
                        best_safe = candidate

                best = best_safe if best_safe is not None else best_any
                if repeat_risk and best_safe is None:
                    # 对“重读风险但无安全候选”的边界，强制尝试 anchor(=prev_end) 收口，避免词尾双读。
                    force_min_avg = max(0.80, float(curr_combined) - 0.08)
                    forced_best = build_anchor_force_candidate(min_avg=float(force_min_avg))
                    if forced_best is not None:
                        best = forced_best
                if best is None:
                    q = curr.get("quality", {}) or {}
                    q["boundary_audio_issue_detected_no_target"] = True
                    q["boundary_audio_issue_gap_before"] = float(gap)
                    q["boundary_audio_similarity_before"] = float(sim_curr)
                    q["boundary_audio_similarity_anchor"] = float(sim_anchor)
                    q["boundary_audio_silence_mismatch_before"] = bool(silence_mismatch_before)
                    q["boundary_audio_source_dbfs_before"] = src_db_before
                    q["boundary_audio_target_dbfs_before"] = tgt_db_before
                    q["boundary_audio_next_gap_before"] = float(next_gap_before)
                    q["boundary_audio_next_penalty_before"] = float(next_penalty_before)
                    curr["quality"] = q
                    continue

                silence_mismatch_after, src_db_after, _ = self._boundary_audio_silence_mismatch(
                    source=src,
                    target_boundary_sec=target_boundary,
                    source_boundary_sec=float(best["start"]),
                    window_sec=window_sec,
                )

                improved = bool(
                    (abs(float(best["gap_after"])) + 0.01) < abs(float(gap))
                    or (float(best["sim"]) >= float(sim_curr) + 0.03)
                    or (bool(silence_mismatch_before) and not bool(silence_mismatch_after))
                )
                unresolved_repeat_risk_after = bool(
                    repeat_risk and (float(best["gap_after"]) < (-repeat_safe_tol))
                )
                if unresolved_repeat_risk_after and (not bool(best.get("forced_repeat_safe_anchor", False))):
                    # 即使局部评分更高，只要重读风险仍未消除，就再尝试一次 anchor 强收口。
                    force_min_avg = max(0.78, float(best["avg"]) - 0.05)
                    forced_best = build_anchor_force_candidate(min_avg=float(force_min_avg))
                    if forced_best is not None:
                        best = forced_best
                        unresolved_repeat_risk_after = False
                silence_mismatch_after, src_db_after, _ = self._boundary_audio_silence_mismatch(
                    source=src,
                    target_boundary_sec=target_boundary,
                    source_boundary_sec=float(best["start"]),
                    window_sec=window_sec,
                )
                improved = bool(
                    (abs(float(best["gap_after"])) + 0.01) < abs(float(gap))
                    or (float(best["sim"]) >= float(sim_curr) + 0.03)
                    or (bool(silence_mismatch_before) and not bool(silence_mismatch_after))
                    or bool(best.get("forced_repeat_safe_anchor", False))
                )
                if (not improved) or (abs(float(best["start"]) - float(curr_start)) <= 1e-6):
                    q = curr.get("quality", {}) or {}
                    q["boundary_audio_issue_detected_no_target"] = True
                    q["boundary_audio_issue_gap_before"] = float(gap)
                    q["boundary_audio_similarity_before"] = float(sim_curr)
                    q["boundary_audio_similarity_anchor"] = float(sim_anchor)
                    q["boundary_audio_similarity_after"] = float(best["sim"])
                    q["boundary_audio_silence_mismatch_before"] = bool(silence_mismatch_before)
                    q["boundary_audio_silence_mismatch_after"] = bool(silence_mismatch_after)
                    q["boundary_audio_source_dbfs_before"] = src_db_before
                    q["boundary_audio_source_dbfs_after"] = src_db_after
                    q["boundary_audio_target_dbfs_before"] = tgt_db_before
                    q["boundary_audio_repeat_risk_unresolved_after"] = bool(unresolved_repeat_risk_after)
                    q["boundary_audio_next_gap_before"] = float(next_gap_before)
                    q["boundary_audio_next_gap_after"] = float(best.get("next_gap_after", 0.0))
                    q["boundary_audio_next_penalty_before"] = float(next_penalty_before)
                    q["boundary_audio_next_penalty_after"] = float(best.get("next_boundary_penalty_after", 0.0))
                    curr["quality"] = q
                    continue

                curr["start"] = float(best["start"])
                q = curr.get("quality", {}) or {}
                q["boundary_audio_issue_detected_no_target"] = True
                q["boundary_audio_repaired_no_target"] = True
                q["boundary_audio_repair_from"] = float(curr_start)
                q["boundary_audio_repair_to"] = float(best["start"])
                q["boundary_audio_repair_before_gap"] = float(gap)
                q["boundary_audio_repair_after_gap"] = float(best["gap_after"])
                q["boundary_audio_similarity_before"] = float(sim_curr)
                q["boundary_audio_similarity_anchor"] = float(sim_anchor)
                q["boundary_audio_similarity_after"] = float(best["sim"])
                q["boundary_audio_silence_mismatch_before"] = bool(silence_mismatch_before)
                q["boundary_audio_silence_mismatch_after"] = bool(silence_mismatch_after)
                q["boundary_audio_source_dbfs_before"] = src_db_before
                q["boundary_audio_source_dbfs_after"] = src_db_after
                q["boundary_audio_target_dbfs_before"] = tgt_db_before
                q["boundary_audio_repair_verify_avg"] = float(best["avg"])
                q["boundary_audio_repeat_safe_tol"] = float(repeat_safe_tol)
                q["boundary_audio_repeat_risk_unresolved_after"] = bool(unresolved_repeat_risk_after)
                q["boundary_audio_repeat_forced_anchor"] = bool(best.get("forced_repeat_safe_anchor", False))
                q["boundary_audio_next_gap_before"] = float(next_gap_before)
                q["boundary_audio_next_gap_after"] = float(best.get("next_gap_after", 0.0))
                q["boundary_audio_next_penalty_before"] = float(next_penalty_before)
                q["boundary_audio_next_penalty_after"] = float(best.get("next_boundary_penalty_after", 0.0))
                q["audio_guard"] = best["audio_meta"]
                curr["quality"] = q
                repaired += 1
                changed = True

            if not changed:
                break

        return int(repaired), int(issues)

    def _trim_prev_for_locked_tail_severe_overlaps_no_target(self, segments: List[dict], max_passes: int = 1) -> int:
        """
        尾部锁死重叠裁前段（禁兜底）：
        - 处理尾部同源 run 已经没有向后腾挪空间、但当前边界仍存在明显负重叠的情况；
        - 这类场景继续强推后段只会把重复在尾部 run 里来回转移；
        - 改为优先裁掉前一段尾巴，宁可尾部略短，也不要保留可感知重复。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        severe_floor = max(0.35, float(getattr(self, "no_target_severe_overlap_trigger", 0.85)))
        tail_boundary_window = max(2, int(getattr(self, "no_target_tail_overlap_trim_boundary_window", 4)))
        min_prev_duration_floor = max(2.8, float(getattr(self, "no_target_tail_overlap_trim_min_prev_duration", 3.6)))

        for _ in range(max(1, int(max_passes))):
            changed = False
            for i in range(max(1, len(segments) - tail_boundary_window), len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    continue
                if prev["source"] != curr["source"]:
                    continue

                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = float(curr_start - prev_end)
                severe_trigger = max(float(severe_floor), float(curr["duration"]) * 0.16)
                if gap >= -severe_trigger:
                    continue

                run_end = i
                while (
                    (run_end + 1) < len(segments)
                    and segments[run_end + 1]["source"] == curr["source"]
                    and segments[run_end + 1]["source"] != self.target_video
                ):
                    run_end += 1
                src_duration = self.get_video_duration(curr["source"])
                if src_duration <= 0.0:
                    continue
                tail_end = float(segments[run_end]["start"]) + float(segments[run_end]["duration"])
                tail_headroom = max(0.0, float(src_duration - tail_end))
                required_shift = max(0.0, -float(gap))
                if tail_headroom + 0.05 >= required_shift:
                    continue

                curr_q = curr.get("quality", {}) or {}
                next_seg = segments[i + 1] if (i + 1) < len(segments) else None
                next_tail_sensitive = bool(
                    next_seg is not None
                    and next_seg.get("source") == curr["source"]
                    and self._is_tail_sensitive_last_resort_segment(next_seg)
                )
                if not (
                    str(curr_q.get("recover_mode", "") or "") in {"neighbors_last_resort_no_target", "neighbors_forced_no_verify"}
                    or bool(curr_q.get("boundary_audio_repeat_risk_unresolved_after", False))
                    or next_tail_sensitive
                    or run_end >= (len(segments) - 2)
                ):
                    continue

                prev_duration = float(prev["duration"])
                min_prev_duration = max(float(min_prev_duration_floor), float(prev_duration) * 0.72)
                trim_cap = max(0.0, float(prev_duration - min_prev_duration))
                trim_sec = min(required_shift, trim_cap)
                if trim_sec <= 1e-6:
                    continue

                prev["duration"] = max(0.0, float(prev["duration"]) - float(trim_sec))
                prev_q = prev.get("quality", {}) or {}
                prev_q["tail_locked_overlap_trim_prev_no_target"] = True
                prev_q["tail_locked_overlap_trim_boundary_index"] = int(i)
                prev_q["tail_locked_overlap_trim_prev_sec"] = float(trim_sec)
                prev_q["tail_locked_overlap_trim_before_gap"] = float(gap)
                prev_q["tail_locked_overlap_trim_after_gap"] = float(curr_start - (float(prev["start"]) + float(prev["duration"])))
                prev_q["tail_locked_overlap_trim_tail_headroom"] = float(tail_headroom)
                prev_q["tail_locked_overlap_trim_required_shift"] = float(required_shift)
                prev["quality"] = prev_q
                adjusted += 1
                changed = True

            if not changed:
                break

        return int(adjusted)

    def _snap_head_anchored_same_source_runs_no_target(self, segments: List[dict], max_passes: int = 1) -> int:
        """
        同源片头链回零（禁兜底）：
        - 某些 run 会整体晚 0.3~0.5s 落到 0.4/5.4/10.4 这种节奏；
        - 画面连续但句首缺字时，主观感受就是“有占位、没声音”；
        - 对这类从源片头起步的同源 run，优先尝试整体前拉回 0.0 相位。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        start_min = 0.08
        start_max = 0.82
        max_probe_segments = 3

        for _ in range(max(1, int(max_passes))):
            changed = False
            i = 0
            while i < len(segments):
                src = segments[i]["source"]
                if src == self.target_video:
                    i += 1
                    continue
                j = i + 1
                while j < len(segments) and segments[j]["source"] == src:
                    j += 1
                run_len = j - i
                if run_len < 2:
                    i = j
                    continue

                first = segments[i]
                first_start = float(first["start"])
                if not (start_min <= first_start <= start_max):
                    i = j
                    continue

                # 只处理“按固定步长整体晚半拍”的 run，避免误伤正常局部修复段。
                run_ok = True
                for k in range(i + 1, min(j, i + max_probe_segments)):
                    prev = segments[k - 1]
                    curr = segments[k]
                    target_step = float(curr["target_start"]) - float(prev["target_start"])
                    source_step = float(curr["start"]) - float(prev["start"])
                    if target_step <= 0.0 or abs(source_step - target_step) > 0.18:
                        run_ok = False
                        break
                if not run_ok:
                    i = j
                    continue

                shift = float(first_start)
                probe_count = min(run_len, max_probe_segments)
                probe_result: List[Tuple[int, float, Dict[str, object], float, float]] = []
                for k in range(i, i + probe_count):
                    seg = segments[k]
                    cand_start = float(seg["start"]) - shift
                    if cand_start < -1e-6:
                        break
                    cand_start = max(0.0, float(cand_start))
                    passed, avg = self.quick_verify(
                        source=seg["source"],
                        source_start=float(cand_start),
                        target_start=float(seg["target_start"]),
                        duration=float(seg["duration"]),
                    )
                    if not passed:
                        break
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=seg["source"],
                        source_start=float(cand_start),
                        target_start=float(seg["target_start"]),
                        duration=float(seg["duration"]),
                        combined_score=float(avg),
                    )
                    if not audio_passed:
                        audio_reason = str((audio_meta or {}).get("reason", "") or "")
                        allow_visual_only = bool(
                            float(cand_start) <= 0.12
                            and audio_reason in {"audio_guard_shift_bias", "audio_guard_shift_bias_halfsec"}
                        )
                        if not allow_visual_only:
                            break
                    q = seg.get("quality", {}) or {}
                    base_avg = float((q.get("strict_verify", {}) or {}).get("avg", q.get("combined", 0.0)) or 0.0)
                    base_aligned = float((q.get("audio_guard", {}) or {}).get("aligned_similarity", 0.0) or 0.0)
                    cand_aligned = float((audio_meta or {}).get("aligned_similarity", 0.0) or 0.0)
                    if avg + 0.03 < base_avg:
                        break
                    if audio_passed and (cand_aligned + 0.04 < base_aligned):
                        break
                    probe_result.append((k, float(cand_start), dict(audio_meta or {}), float(avg), float(cand_aligned)))
                apply_len = len(probe_result)
                if (not run_ok) or apply_len < 2:
                    i = j
                    continue

                apply_end = min(j, i + apply_len)
                for k in range(i, apply_end):
                    seg = segments[k]
                    old_start = float(seg["start"])
                    seg["start"] = max(0.0, float(old_start - shift))
                    q = seg.get("quality", {}) or {}
                    q["head_aligned_run_snap_no_target"] = True
                    q["head_aligned_run_snap_from"] = float(old_start)
                    q["head_aligned_run_snap_to"] = float(seg["start"])
                    q["head_aligned_run_snap_shift_sec"] = float(shift)
                    q["head_aligned_run_snap_anchor_index"] = int(i)
                    q["head_aligned_run_snap_prefix_len"] = int(apply_len)
                    for probe_idx, cand_start, audio_meta, avg, cand_aligned in probe_result:
                        if probe_idx == k:
                            q["audio_guard"] = audio_meta
                            q["head_aligned_run_snap_verify_avg"] = float(avg)
                            q["head_aligned_run_snap_aligned_similarity"] = float(cand_aligned)
                            break
                    seg["quality"] = q
                    adjusted += 1

                changed = True
                i = j

            if not changed:
                break

        return int(adjusted)

    def _trim_tail_overlaps_with_shortfall_tolerance_no_target(self, segments: List[dict]) -> int:
        """
        尾段短缺容忍裁尾（禁兜底）：
        - 对尾部同源负重叠边界，优先裁掉前一段尾巴；
        - 只在尾段允许短缺预算内生效，避免为了凑满总时长把重复留到成片结尾；
        - 不改后段起点，尽量保持后续句子连续。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        budget = max(0.0, float(getattr(self, "no_target_tail_shortfall_tolerance_sec", 0.0)))
        if budget <= 1e-6:
            return 0

        adjusted = 0
        min_prev_duration = max(0.8, min(2.0, float(self.segment_duration) * 0.18))
        total_segments = len(segments)

        for i in range(total_segments - 1, 0, -1):
            if budget <= 1e-6:
                break
            prev = segments[i - 1]
            curr = segments[i]
            if prev["source"] == self.target_video or curr["source"] == self.target_video:
                continue
            if prev["source"] != curr["source"]:
                continue
            if not self._is_tail_shortfall_relax_boundary(i, curr, total_segments):
                continue

            prev_start = float(prev["start"])
            prev_duration = float(prev["duration"])
            curr_start = float(curr["start"])
            gap = float(curr_start - (prev_start + prev_duration))
            overlap = max(0.0, -gap)
            if overlap <= 0.015:
                continue

            trimmable = max(0.0, float(prev_duration - min_prev_duration))
            if trimmable <= 1e-6:
                continue
            trim = min(float(overlap), float(trimmable), float(budget))
            if trim + 0.015 < overlap:
                continue

            new_prev_duration = float(prev_duration - trim)
            new_gap = float(curr_start - (prev_start + new_prev_duration))
            if new_gap < -0.02:
                continue

            prev["duration"] = float(new_prev_duration)
            prev_q = prev.get("quality", {}) or {}
            prev_q["tail_shortfall_overlap_trim_no_target"] = True
            prev_q["tail_shortfall_overlap_trim_sec"] = float(trim)
            prev_q["tail_shortfall_overlap_trim_boundary_index"] = int(i)
            prev_q["tail_shortfall_overlap_trim_before_gap"] = float(gap)
            prev_q["tail_shortfall_overlap_trim_after_gap"] = float(new_gap)
            prev_q["tail_shortfall_overlap_trim_budget_left"] = float(max(0.0, budget - trim))
            prev["quality"] = prev_q

            curr_q = curr.get("quality", {}) or {}
            curr_q["tail_shortfall_overlap_trimmed_prev_no_target"] = True
            curr_q["tail_shortfall_overlap_trim_boundary_index"] = int(i)
            curr_q["tail_shortfall_overlap_trim_prev_sec"] = float(trim)
            curr_q["tail_shortfall_overlap_trim_before_gap"] = float(gap)
            curr_q["tail_shortfall_overlap_trim_after_gap"] = float(new_gap)
            curr["quality"] = curr_q

            budget -= float(trim)
            adjusted += 1

        return int(adjusted)

    def _clamp_carryover_shifted_overlap_boundaries_no_target(self, segments: List[dict]) -> int:
        """
        前源尾巴接入后的重叠收口（禁兜底）：
        - 仅处理已被 cross_source_prev_tail_carryover 前移过的同源段；
        - 若前边界仍残留 >0.12s 的负重叠，则优先将当前段轻微后推贴近边界；
        - 目的不是追求完全 gap=0，而是避免 0.3s 级别的可听重读反复出现。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev["source"] == self.target_video or curr["source"] == self.target_video:
                continue
            if prev["source"] != curr["source"]:
                continue

            curr_q = curr.get("quality", {}) or {}
            if not bool(curr_q.get("cross_source_prev_tail_carryover_shifted_no_target", False)):
                continue

            prev_end = float(prev["start"]) + float(prev["duration"])
            curr_start = float(curr["start"])
            gap = float(curr_start - prev_end)
            if gap >= -0.12:
                continue

            src = curr["source"]
            src_duration = self.get_video_duration(src)
            if src_duration <= 0.0:
                continue
            duration = float(curr["duration"])
            max_start = max(0.0, float(src_duration - duration))

            next_seg: Optional[dict] = None
            if (i + 1) < len(segments) and segments[i + 1]["source"] == curr["source"]:
                next_seg = segments[i + 1]

            candidate_starts: List[float] = []
            for offset in (-0.12, -0.08, -0.04, 0.0):
                candidate_starts.append(max(0.0, min(float(prev_end + offset), max_start)))
            candidate_starts.append(float(curr_start))

            best: Optional[Dict[str, object]] = None
            best_score: Optional[float] = None
            seen: Set[float] = set()
            for raw_start in candidate_starts:
                cand_start = float(raw_start)
                key = round(cand_start, 3)
                if key in seen:
                    continue
                seen.add(key)

                cand_gap = float(cand_start - prev_end)
                if cand_gap < -0.12:
                    continue

                next_gap = 0.0
                if next_seg is not None:
                    next_gap = float(next_seg["start"]) - (cand_start + duration)
                    if next_gap < -0.20:
                        continue

                passed, avg = self.quick_verify(
                    source=src,
                    source_start=float(cand_start),
                    target_start=float(curr["target_start"]),
                    duration=duration,
                )
                if not passed:
                    continue
                audio_passed, audio_meta = self.quick_verify_audio(
                    source=src,
                    source_start=float(cand_start),
                    target_start=float(curr["target_start"]),
                    duration=duration,
                    combined_score=float(avg),
                )
                if not audio_passed:
                    continue

                aligned = float((audio_meta or {}).get("aligned_similarity", 0.0) or 0.0)
                score = float(avg) + aligned * 0.04 - abs(cand_gap) * 0.9 - max(0.0, -next_gap) * 0.4
                if best_score is None or score > best_score:
                    best_score = float(score)
                    best = {
                        "start": float(cand_start),
                        "avg": float(avg),
                        "gap": float(cand_gap),
                        "next_gap": float(next_gap),
                        "audio_meta": dict(audio_meta or {}),
                    }

            if best is None:
                continue
            if abs(float(best["start"]) - float(curr_start)) <= 1e-6:
                continue

            curr["start"] = float(best["start"])
            curr_q["carryover_overlap_clamped_no_target"] = True
            curr_q["carryover_overlap_clamped_from"] = float(curr_start)
            curr_q["carryover_overlap_clamped_to"] = float(best["start"])
            curr_q["carryover_overlap_clamped_gap_before"] = float(gap)
            curr_q["carryover_overlap_clamped_gap_after"] = float(best["gap"])
            curr_q["carryover_overlap_clamped_next_gap_after"] = float(best["next_gap"])
            curr_q["carryover_overlap_clamped_verify_avg"] = float(best["avg"])
            curr_q["audio_guard"] = best["audio_meta"]
            curr["quality"] = curr_q
            adjusted += 1

        return int(adjusted)

    def _trim_prev_for_carryover_shifted_overlaps_no_target(self, segments: List[dict]) -> int:
        """
        前源尾巴接入后的前段裁尾（禁兜底）：
        - 对 carryover 后仍残留明显负重叠的边界，优先裁掉前一段尾巴；
        - 避免当前段继续前后游走导致重复问题反复出现。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        min_prev_duration = max(0.8, min(2.0, float(self.segment_duration) * 0.18))
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev["source"] == self.target_video or curr["source"] == self.target_video:
                continue
            if prev["source"] != curr["source"]:
                continue

            curr_q = curr.get("quality", {}) or {}
            if not bool(curr_q.get("cross_source_prev_tail_carryover_shifted_no_target", False)):
                continue
            prev_q = prev.get("quality", {}) or {}
            # 前段本身若是“前源尾巴 + 当前源片头”的跨源拼桥段，
            # 再裁它的尾巴会直接挖掉刚刚借来的连续内容，造成句尾缺失。
            if bool(prev_q.get("cross_source_prev_tail_carryover_no_target", False)):
                continue

            prev_start = float(prev["start"])
            prev_duration = float(prev["duration"])
            curr_start = float(curr["start"])
            gap = float(curr_start - (prev_start + prev_duration))
            if gap >= -0.12:
                continue

            trimmable = max(0.0, float(prev_duration - min_prev_duration))
            if trimmable <= 1e-6:
                continue

            trim = min(float((-gap) - 0.12), float(trimmable))
            if trim <= 1e-6:
                continue

            prev["duration"] = float(prev_duration - trim)
            new_gap = float(curr_start - (prev_start + float(prev["duration"])))

            prev_q["carryover_prev_trim_no_target"] = True
            prev_q["carryover_prev_trim_boundary_index"] = int(i)
            prev_q["carryover_prev_trim_sec"] = float(trim)
            prev_q["carryover_prev_trim_gap_before"] = float(gap)
            prev_q["carryover_prev_trim_gap_after"] = float(new_gap)
            prev["quality"] = prev_q

            curr_q["carryover_prev_trimmed_no_target"] = True
            curr_q["carryover_prev_trim_boundary_index"] = int(i)
            curr_q["carryover_prev_trim_sec"] = float(trim)
            curr_q["carryover_prev_trim_gap_before"] = float(gap)
            curr_q["carryover_prev_trim_gap_after"] = float(new_gap)
            curr["quality"] = curr_q
            adjusted += 1

        return int(adjusted)

    def _suppress_small_negative_overlaps_no_target(self, segments: List[dict], max_passes: int = 2) -> int:
        """
        语音敏感小重叠抑制（禁兜底）：
        - 处理同源相邻段的 -0.04~-0.20s 小负重叠（最容易触发单字重读）；
        - 仅对高置信边界生效，并要求 visual/audio 复核通过；
        - 优先把边界贴到 prev_end（gap≈0），避免词头重复。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        for _ in range(max(1, int(max_passes))):
            changed = False
            for i in range(1, len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    continue
                if prev["source"] != curr["source"]:
                    continue

                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = float(curr_start - prev_end)
                # 只处理中小幅负重叠：过大重叠由前序强修复逻辑处理。
                if gap >= -0.04 or gap < -0.20:
                    continue

                prev_q = prev.get("quality", {}) or {}
                curr_q = curr.get("quality", {}) or {}
                prev_comb = float(prev_q.get("combined", 0.0))
                curr_comb = float(curr_q.get("combined", 0.0))
                if min(prev_comb, curr_comb) < 0.84:
                    continue

                src = curr["source"]
                src_duration = self.get_video_duration(src)
                if src_duration <= 0.0:
                    continue
                max_start = max(0.0, float(src_duration - float(curr["duration"])))
                anchor = max(0.0, min(float(prev_end), max_start))

                raw_candidates = [
                    anchor,
                    anchor + 0.02,
                    anchor - 0.02,
                    anchor + 0.05,
                    anchor - 0.05,
                ]
                candidates: List[float] = []
                seen: Set[float] = set()
                for raw in raw_candidates:
                    cand = max(0.0, min(float(raw), max_start))
                    key = round(cand, 3)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(float(cand))

                best: Optional[Dict[str, object]] = None
                best_score: Optional[float] = None
                for cand_start in candidates:
                    new_gap = float(cand_start - prev_end)
                    overlap_after = max(0.0, -new_gap)
                    # 至少要把重叠显著压缩，目标接近 0。
                    if overlap_after > 0.025:
                        continue

                    if (i + 1) < len(segments):
                        nxt = segments[i + 1]
                        if nxt["source"] == curr["source"] and nxt["source"] != self.target_video:
                            next_gap = float(nxt["start"]) - (float(cand_start) + float(curr["duration"]))
                            next_neg_trigger = max(0.12, float(nxt["duration"]) * 0.02)
                            if next_gap < -next_neg_trigger:
                                continue

                    passed, avg = self.quick_verify(
                        source=src,
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                    )
                    if not passed:
                        continue
                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=src,
                        source_start=float(cand_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=max(float(avg), float(curr_comb)),
                    )
                    if not audio_passed:
                        continue

                    score = float(avg) - float(overlap_after) * 0.35 - abs(float(cand_start) - float(curr_start)) * 0.01
                    if best_score is None or score > best_score:
                        best_score = float(score)
                        best = {
                            "start": float(cand_start),
                            "avg": float(avg),
                            "new_gap": float(new_gap),
                            "audio_meta": audio_meta,
                        }

                if best is None:
                    continue
                if abs(float(best["start"]) - float(curr_start)) <= 1e-6:
                    continue

                curr["start"] = float(best["start"])
                q = curr.get("quality", {}) or {}
                q["small_overlap_suppressed_no_target"] = True
                q["small_overlap_suppressed_from"] = float(curr_start)
                q["small_overlap_suppressed_to"] = float(best["start"])
                q["small_overlap_suppressed_before_gap"] = float(gap)
                q["small_overlap_suppressed_after_gap"] = float(best["new_gap"])
                q["small_overlap_suppressed_verify_avg"] = float(best["avg"])
                q["audio_guard"] = best["audio_meta"]
                curr["quality"] = q
                adjusted += 1
                changed = True

            if not changed:
                break

        return int(adjusted)

    def _shift_same_source_runs_right_to_clear_overlap_no_target(self, segments: List[dict], max_passes: int = 2) -> int:
        """
        禁兜底同源 run 整体右移修复：
        - 处理 -0.20~-0.80s 的同源负重叠，这类问题往往不是单段错，而是整条 run 整体早了半拍；
        - 将当前段及后续同源 run 一起右移，避免只修当前段反而把下一边界再次挤坏；
        - 优先用于 1:28~1:30、尾部轻微回放这类“听感是回放/重读，但整体源是对的”的场景。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        min_overlap = max(0.20, float(getattr(self, "no_target_run_right_shift_min_overlap", 0.20)))
        max_overlap = max(min_overlap, float(getattr(self, "no_target_run_right_shift_max_overlap", 0.80)))
        safe_tol = max(0.0, float(getattr(self, "no_target_run_right_shift_safe_tol", 0.0)))

        for _ in range(max(1, int(max_passes))):
            changed = False
            i = 1
            while i < len(segments):
                prev = segments[i - 1]
                curr = segments[i]
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    i += 1
                    continue
                if prev["source"] != curr["source"]:
                    i += 1
                    continue

                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = float(curr_start - prev_end)
                overlap = max(0.0, -float(gap))
                if overlap < (float(min_overlap) - 1e-6) or overlap > (float(max_overlap) + 1e-6):
                    i += 1
                    continue

                src = curr["source"]
                src_duration = self.get_video_duration(src)
                if src_duration <= 0.0:
                    i += 1
                    continue

                run_end = i
                while (
                    (run_end + 1) < len(segments)
                    and segments[run_end + 1]["source"] == src
                    and segments[run_end + 1]["source"] != self.target_video
                ):
                    run_end += 1

                curr_q = curr.get("quality", {}) or {}
                curr_combined = float(curr_q.get("combined", 0.0))
                required_shift = max(0.0, float(overlap) + float(safe_tol))
                tail_end = float(segments[run_end]["start"]) + float(segments[run_end]["duration"])
                run_headroom = max(0.0, float(src_duration) - float(tail_end))
                max_shift = min(float(required_shift), float(run_headroom))
                if max_shift <= 0.04:
                    i = run_end + 1
                    continue

                candidate_shifts: List[float] = []
                for ratio in (1.0, 0.9, 0.8, 0.7, 0.6):
                    cand_shift = float(max_shift) * float(ratio)
                    if cand_shift <= 0.04:
                        continue
                    if all(abs(float(cand_shift) - float(x)) > 1e-6 for x in candidate_shifts):
                        candidate_shifts.append(float(cand_shift))

                best_choice: Optional[Dict[str, object]] = None
                best_key: Optional[Tuple[float, float]] = None
                for cand_shift in candidate_shifts:
                    candidate_start = float(curr_start) + float(cand_shift)
                    passed, verify_avg = self.quick_verify(
                        source=src,
                        source_start=float(candidate_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                    )
                    relaxed_visual_keep = bool(
                        (not passed)
                        and (float(verify_avg) >= max(0.92, float(curr_combined) - 0.015))
                    )
                    if (not passed) and (not relaxed_visual_keep):
                        continue

                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=src,
                        source_start=float(candidate_start),
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                        combined_score=float(max(verify_avg, curr_combined)),
                    )
                    if not audio_passed:
                        continue

                    after_gap = float(candidate_start - prev_end)
                    overlap_after = max(0.0, -float(after_gap))
                    key = (float(overlap_after), -float(verify_avg))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_choice = {
                            "shift": float(cand_shift),
                            "start": float(candidate_start),
                            "verify_avg": float(verify_avg),
                            "audio_meta": audio_meta,
                            "relaxed_visual_keep": bool(relaxed_visual_keep),
                        }

                if best_choice is None:
                    i = run_end + 1
                    continue

                applied_shift = float(best_choice["shift"])

                for k in range(i, run_end + 1):
                    segments[k]["start"] = float(segments[k]["start"]) + float(applied_shift)
                    q = segments[k].get("quality", {}) or {}
                    q["run_right_shift_no_target"] = True
                    q["run_right_shift_delta_sec"] = float(applied_shift)
                    q["run_right_shift_anchor_index"] = int(i)
                    if k == i:
                        q["run_right_shift_before_gap"] = float(gap)
                        q["run_right_shift_after_gap"] = float(
                            float(segments[k]["start"]) - float(prev_end)
                        )
                        q["run_right_shift_verify_avg"] = float(best_choice["verify_avg"])
                        q["run_right_shift_relaxed_visual_keep"] = bool(best_choice["relaxed_visual_keep"])
                        q["audio_guard"] = best_choice["audio_meta"]
                    segments[k]["quality"] = q

                adjusted += int(run_end - i + 1)
                changed = True
                i = run_end + 1

            if not changed:
                break

        return int(adjusted)

    def _trim_prev_for_unresolved_same_source_overlap_no_target(self, segments: List[dict], max_passes: int = 2) -> int:
        """
        禁兜底同源重叠裁前段尾巴：
        - 当前段推不动、但同源边界仍保留 0.25~0.75s 重叠时，优先裁掉前一段重复出来的尾巴；
        - 只对高置信段生效，并用 relaxed visual 验证守住“裁掉的是重复，不是新内容”。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        adjusted = 0
        min_overlap = 0.25
        max_overlap = 0.75
        min_prev_duration = 3.8

        for _ in range(max(1, int(max_passes))):
            changed = False
            for i in range(1, len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    continue
                if prev["source"] != curr["source"]:
                    continue

                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = float(curr_start - prev_end)
                overlap = max(0.0, -float(gap))
                if overlap < (min_overlap - 1e-6) or overlap > (max_overlap + 1e-6):
                    continue

                prev_q = prev.get("quality", {}) or {}
                curr_q = curr.get("quality", {}) or {}
                prev_combined = float(prev_q.get("combined", 0.0))
                curr_combined = float(curr_q.get("combined", 0.0))
                if min(prev_combined, curr_combined) < 0.90:
                    continue

                new_duration = float(prev["duration"]) - float(overlap)
                if new_duration < float(min_prev_duration):
                    continue

                passed, verify_avg = self.quick_verify(
                    source=prev["source"],
                    source_start=float(prev["start"]),
                    target_start=float(prev["target_start"]),
                    duration=float(new_duration),
                )
                relaxed_visual_keep = bool(
                    (not passed)
                    and (float(verify_avg) >= max(0.92, float(prev_combined) - 0.02))
                )
                if (not passed) and (not relaxed_visual_keep):
                    continue

                prev["duration"] = float(new_duration)
                q = prev_q
                q["trim_prev_same_source_overlap_no_target"] = True
                q["trim_prev_same_source_overlap_sec"] = float(overlap)
                q["trim_prev_same_source_overlap_boundary_index"] = int(i)
                q["trim_prev_same_source_overlap_before_gap"] = float(gap)
                q["trim_prev_same_source_overlap_after_gap"] = 0.0
                q["trim_prev_same_source_overlap_new_duration"] = float(new_duration)
                q["trim_prev_same_source_overlap_verify_avg"] = float(verify_avg)
                q["trim_prev_same_source_overlap_relaxed_visual_keep"] = bool(relaxed_visual_keep)
                prev["quality"] = q

                curr_quality = curr.get("quality", {}) or {}
                curr_quality["trim_prev_same_source_overlap_boundary_index"] = int(i)
                curr["quality"] = curr_quality

                adjusted += 1
                changed = True

            if not changed:
                break

        return int(adjusted)

    def snap_small_adjacent_gaps_without_target_fallback(self, segments: List[dict]) -> Tuple[int, int]:
        """
        禁用目标兜底时的同源边界校正：
        - 处理同源相邻段的重叠/断缝，减少重复词、回放感、局部节奏错位。
        - 优先使用 quick_verify 通过的候选；必要时仅对高置信段做受控强制贴齐。
        返回：(已修复边界数, 未收敛边界数)
        """
        if self.enable_target_video_fallback:
            return 0, 0
        if len(segments) < 2:
            return 0, 0

        adjusted = 0
        max_passes = 3
        unresolved = 0
        boundary_hard_clamped = 0
        boundary_audio_issues = 0
        boundary_audio_repaired = 0

        def verify_score(source: Path, source_start: float, target_start: float, duration: float) -> Tuple[bool, float]:
            passed, avg = self.quick_verify(
                source=source,
                source_start=float(source_start),
                target_start=float(target_start),
                duration=float(duration),
            )
            return bool(passed), float(avg)

        for _ in range(max_passes):
            changed = False
            unresolved = 0
            for i in range(1, len(segments)):
                prev = segments[i - 1]
                curr = segments[i]
                if prev["source"] == self.target_video or curr["source"] == self.target_video:
                    continue
                if prev["source"] != curr["source"]:
                    continue

                prev_end = float(prev["start"]) + float(prev["duration"])
                curr_start = float(curr["start"])
                gap = curr_start - prev_end

                neg_trigger = max(0.12, float(curr["duration"]) * 0.02)
                pos_trigger = max(0.20, float(curr["duration"]) * 0.04)
                if (gap >= -neg_trigger) and (gap <= pos_trigger):
                    continue
                current_excess = 0.0
                if gap < -neg_trigger:
                    current_excess = float((-gap) - neg_trigger)
                elif gap > pos_trigger:
                    current_excess = float(gap - pos_trigger)

                source_duration = self.get_video_duration(curr["source"])
                max_start = max(0.0, float(source_duration - float(curr["duration"])))
                anchor = max(0.0, min(float(prev_end), max_start))
                if abs(anchor - curr_start) <= 1e-6:
                    continue

                next_same_source = False
                next_seg: Optional[dict] = None
                if i + 1 < len(segments):
                    nxt = segments[i + 1]
                    if nxt["source"] == curr["source"] and nxt["source"] != self.target_video:
                        next_same_source = True
                        next_seg = nxt

                # 从当前段到同源 run 尾的最小占用时长，用于评估“尾部容量是否被挤爆”。
                run_end = i
                while (run_end + 1) < len(segments) and segments[run_end + 1]["source"] == curr["source"]:
                    run_end += 1
                remaining_run_duration = sum(float(segments[k]["duration"]) for k in range(i, run_end + 1))
                tail_overflow_at_anchor = max(
                    0.0,
                    float(anchor) + float(remaining_run_duration) - float(source_duration),
                )

                # 在锚点附近做小范围搜索：
                # - 负重叠：仅允许向后（>= anchor），避免继续重叠导致重复词
                # - 正缝隙：仅允许向前（<= anchor），避免继续跳过内容
                if gap < -neg_trigger:
                    max_forward = max(0.5, min(float(curr["duration"]) * 0.60, max_start - anchor))
                    max_backward_relief = 0.0
                    if tail_overflow_at_anchor > 0.04:
                        # run 尾部容量不足时，允许少量向前回退分摊重叠，避免把重叠全部挤到后续边界。
                        max_backward_relief = max(
                            0.35,
                            min(
                                float(anchor),
                                max(float(curr["duration"]) * 0.60, float(tail_overflow_at_anchor) + 0.20),
                            ),
                        )
                    needed = max(0.0, min(-float(gap), max_forward))
                    raw_offsets = [
                        0.0, 0.04, 0.08, 0.12, 0.2, 0.35,
                        0.5, 0.8, 1.0, 1.2, 1.5, 2.0,
                        needed * 0.5,
                        needed * 0.75,
                        needed,
                        min(max_forward, needed + 0.2),
                    ]
                    if max_backward_relief > 0.0:
                        raw_offsets.extend(
                            [
                                -min(max_backward_relief, float(tail_overflow_at_anchor) * 0.50),
                                -min(max_backward_relief, float(tail_overflow_at_anchor) * 0.80),
                                -min(max_backward_relief, float(tail_overflow_at_anchor)),
                                -float(max_backward_relief),
                            ]
                        )
                    offsets = sorted(
                        {
                            round(max(-max_backward_relief, min(max_forward, float(o))), 3)
                            for o in raw_offsets
                        }
                    )
                elif gap > pos_trigger:
                    max_backward = max(0.5, min(float(curr["duration"]) * 0.60, anchor))
                    needed = max(0.0, min(float(gap), max_backward))
                    raw_offsets = [
                        0.0, -0.04, -0.08, -0.12, -0.2, -0.35,
                        -0.5, -0.8, -1.0, -1.2, -1.5, -2.0,
                        -(needed * 0.5),
                        -(needed * 0.75),
                        -needed,
                        -min(max_backward, needed + 0.2),
                    ]
                    offsets = sorted(
                        {
                            round(max(-max_backward, min(0.0, float(o))), 3)
                            for o in raw_offsets
                        }
                    )
                else:
                    offsets = [0.0]
                best = None
                for off in offsets:
                    cand = max(0.0, min(anchor + off, max_start))
                    passed, avg = verify_score(
                        source=curr["source"],
                        source_start=cand,
                        target_start=float(curr["target_start"]),
                        duration=float(curr["duration"]),
                    )
                    audio_passed = False
                    audio_meta: Dict[str, object] = {"checked": False}
                    shift_meta: Dict[str, object] = {"applied": False, "checked": False}
                    if passed:
                        audio_passed, audio_meta = self.quick_verify_audio(
                            source=curr["source"],
                            source_start=float(cand),
                            target_start=float(curr["target_start"]),
                            duration=float(curr["duration"]),
                            combined_score=float(avg),
                        )
                        if not audio_passed:
                            passed = False
                        else:
                            realigned_start, shift_meta, shifted_audio_meta = self.try_audio_guard_shift_realign(
                                source=curr["source"],
                                source_start=float(cand),
                                target_start=float(curr["target_start"]),
                                duration=float(curr["duration"]),
                                combined_score=float(avg),
                                audio_meta=audio_meta,
                            )
                            if bool(shift_meta.get("applied", False)):
                                cand = float(realigned_start)
                                avg = max(float(avg), float(shift_meta.get("candidate_verify_avg", avg)))
                                if isinstance(shifted_audio_meta, dict):
                                    audio_meta = shifted_audio_meta
                            else:
                                shift_meta = dict(shift_meta or {})

                    after_gap = float(cand - prev_end)
                    residual_penalty = 0.0
                    if after_gap < -neg_trigger:
                        residual_penalty = float((-after_gap) - neg_trigger)
                    elif after_gap > pos_trigger:
                        residual_penalty = float(after_gap - pos_trigger)
                    allow_after_excess_increase = 0.0
                    if gap < -neg_trigger and tail_overflow_at_anchor > 0.04:
                        allow_after_excess_increase = min(1.8, float(tail_overflow_at_anchor) * 0.95)
                    if residual_penalty > (current_excess + allow_after_excess_increase + 1e-6):
                        continue

                    next_gap = 0.0
                    next_boundary_penalty = 0.0
                    if next_same_source and next_seg is not None:
                        next_gap = float(next_seg["start"]) - (float(cand) + float(curr["duration"]))
                        next_neg_trigger = max(0.12, float(next_seg["duration"]) * 0.02)
                        next_pos_trigger = max(0.20, float(next_seg["duration"]) * 0.04)
                        if next_gap < -next_neg_trigger:
                            next_boundary_penalty = float((-next_gap) - next_neg_trigger)
                        elif next_gap > next_pos_trigger:
                            next_boundary_penalty = float(next_gap - next_pos_trigger)

                    run_tail_overflow = max(
                        0.0,
                        float(cand) + float(remaining_run_duration) - float(source_duration),
                    )
                    offset_penalty = abs(float(cand) - float(anchor))

                    score = (
                        float(avg)
                        - float(offset_penalty) * 0.02
                        - float(residual_penalty) * 0.30
                        - float(next_boundary_penalty) * 0.65
                        - float(run_tail_overflow) * 0.55
                    )
                    if best is None or score > best["score"]:
                        best = {
                            "start": float(cand),
                            "avg": float(avg),
                            "passed": bool(passed),
                            "audio_passed": bool(audio_passed),
                            "audio_checked": bool((audio_meta or {}).get("checked", False)),
                            "audio_meta": dict(audio_meta or {}),
                            "audio_shift_meta": dict(shift_meta or {}),
                            "after_gap": float(after_gap),
                            "residual_penalty": float(residual_penalty),
                            "next_gap": float(next_gap),
                            "next_boundary_penalty": float(next_boundary_penalty),
                            "run_tail_overflow": float(run_tail_overflow),
                            "offset": float(off),
                            "score": float(score),
                        }

                if best is None:
                    unresolved += 1
                    continue

                forced_by_confidence = False
                selected = best
                if not bool(selected["passed"]):
                    if bool(selected.get("audio_checked", False)) and (not bool(selected.get("audio_passed", False))):
                        unresolved += 1
                        continue
                    curr_q = curr.get("quality", {}) or {}
                    combined = float(curr_q.get("combined", 0.0))
                    recover_mode = str(curr_q.get("recover_mode", "") or "")
                    force_overlap_limit = max(0.9, float(curr["duration"]) * 0.22)
                    # 仅允许“负重叠 + 高置信 + 核验分不差”的受控强制贴齐
                    if (
                        gap < -neg_trigger
                        and abs(gap) <= force_overlap_limit
                        and combined >= 0.92
                        and float(selected["avg"]) >= 0.88
                    ):
                        forced_by_confidence = True
                    elif (
                        gap < -neg_trigger
                        and abs(gap) <= max(0.35, float(curr["duration"]) * 0.07)
                        and combined >= 0.95
                    ):
                        # 小幅负重叠在高置信段上优先消除重复，避免反复出现“词尾双读”。
                        forced_by_confidence = True
                    elif (
                        gap < -neg_trigger
                        and recover_mode == "neighbors_forced_no_verify"
                        and abs(gap) <= max(2.5, float(curr["duration"]) * 0.60)
                    ):
                        # 低置信强制邻段恢复本身风险更高，优先保证时间轴不重叠。
                        forced_by_confidence = True
                    else:
                        unresolved += 1
                        continue

                new_start = float(selected["start"])
                if abs(new_start - curr_start) <= 1e-6:
                    continue

                # 防止“修复当前边界却把下一边界推到明显重叠/断缝”。
                next_penalty = float(selected.get("next_boundary_penalty", 0.0))
                if next_penalty > max(0.55, current_excess + 0.28):
                    unresolved += 1
                    continue

                curr["start"] = new_start
                q = curr.get("quality", {}) or {}
                q["micro_gap_snapped_no_target"] = True
                q["micro_gap_before"] = float(gap)
                q["micro_gap_after"] = float(selected.get("after_gap", curr["start"] - prev_end))
                q["micro_gap_residual_penalty"] = float(selected.get("residual_penalty", 0.0))
                q["micro_gap_next_gap"] = float(selected.get("next_gap", 0.0))
                q["micro_gap_next_boundary_penalty"] = float(selected.get("next_boundary_penalty", 0.0))
                q["micro_gap_run_tail_overflow"] = float(selected.get("run_tail_overflow", 0.0))
                q["micro_gap_verify_passed"] = bool(selected["passed"])
                q["micro_gap_verify_avg"] = float(selected["avg"])
                q["micro_gap_search_offset"] = float(selected["offset"])
                q["micro_gap_forced_by_confidence"] = bool(forced_by_confidence)
                q["audio_guard"] = selected.get("audio_meta", {"checked": False})
                q["micro_gap_audio_shift_fix"] = selected.get("audio_shift_meta", {})
                curr["quality"] = q
                adjusted += 1
                changed = True

            if not changed:
                break

        unresolved = self._count_no_target_boundary_unresolved(segments)
        if unresolved > 0:
            backprop_fixed = self._backprop_resolve_locked_tail_overlaps_no_target(segments)
            if backprop_fixed > 0:
                adjusted += int(backprop_fixed)
                unresolved = self._count_no_target_boundary_unresolved(segments)
        if unresolved > 0:
            rematch_max_attempts = max(0, int(getattr(self, "no_target_boundary_rematch_max_attempts", 0)))
            if rematch_max_attempts > 0:
                rematched = self._rematch_unresolved_boundaries_without_target_fallback(
                    segments,
                    max_attempts=rematch_max_attempts,
                )
                if rematched > 0:
                    adjusted += int(rematched)
                    unresolved = self._count_no_target_boundary_unresolved(segments)
        cross_nudged = self._nudge_cross_source_head_boundaries_no_target(segments)
        if cross_nudged > 0:
            adjusted += int(cross_nudged)
            unresolved = self._count_no_target_boundary_unresolved(segments)
        rebalanced = self._rebalance_neighbor_recovered_segments_no_target(segments)
        if rebalanced > 0:
            adjusted += int(rebalanced)
            unresolved = self._count_no_target_boundary_unresolved(segments)
        lag_fixed = self._repair_same_source_step_lag_runs_no_target(segments)
        if lag_fixed > 0:
            adjusted += int(lag_fixed)
            unresolved = self._count_no_target_boundary_unresolved(segments)
        post_lag_cleanup = self._cleanup_unresolved_boundaries_post_lag_no_target(segments, max_passes=2)
        if post_lag_cleanup > 0:
            adjusted += int(post_lag_cleanup)
            unresolved = self._count_no_target_boundary_unresolved(segments)
        run_redistributed = self._redistribute_mild_run_overflow_no_target(segments)
        if run_redistributed > 0:
            adjusted += int(run_redistributed)
            unresolved = self._count_no_target_boundary_unresolved(segments)
        severe_overlap_fixed = self._resolve_severe_same_source_overlaps_no_target(segments, max_passes=2)
        if severe_overlap_fixed > 0:
            adjusted += int(severe_overlap_fixed)
            unresolved = self._count_no_target_boundary_unresolved(segments)
        small_overlap_suppressed = self._suppress_small_negative_overlaps_no_target(segments, max_passes=2)
        if small_overlap_suppressed > 0:
            adjusted += int(small_overlap_suppressed)
            unresolved = self._count_no_target_boundary_unresolved(segments)
        hard_clamped = self._enforce_boundary_hard_constraints_no_target(segments, max_passes=2)
        if hard_clamped > 0:
            adjusted += int(hard_clamped)
            boundary_hard_clamped += int(hard_clamped)
            unresolved = self._count_no_target_boundary_unresolved(segments)
        audio_repaired, audio_issues = self._repair_boundary_audio_locally_no_target(
            segments,
            max_passes=max(1, int(getattr(self, "boundary_audio_repair_max_passes", 2))),
        )
        boundary_audio_repaired += int(audio_repaired)
        boundary_audio_issues += int(audio_issues)
        if audio_repaired > 0:
            adjusted += int(audio_repaired)
        if audio_issues > 0:
            unresolved = self._count_no_target_boundary_unresolved(segments)
            # 音频边界修复后执行一次“硬约束 + 回推收尾”：
            # 有些边界贴近源尾时，单点修复无法完全消除负重叠，需要把重叠向前分摊。
            hard_clamped_after = self._enforce_boundary_hard_constraints_no_target(segments, max_passes=1)
            if hard_clamped_after > 0:
                adjusted += int(hard_clamped_after)
                boundary_hard_clamped += int(hard_clamped_after)
            backprop_after_audio = self._backprop_resolve_locked_tail_overlaps_no_target(segments)
            if backprop_after_audio > 0:
                adjusted += int(backprop_after_audio)
            severe_after_audio = self._resolve_severe_same_source_overlaps_no_target(segments, max_passes=1)
            if severe_after_audio > 0:
                adjusted += int(severe_after_audio)
            if hard_clamped_after > 0 or backprop_after_audio > 0:
                unresolved = self._count_no_target_boundary_unresolved(segments)
            if severe_after_audio > 0:
                unresolved = self._count_no_target_boundary_unresolved(segments)

        # 终态重匹配收尾：
        # 前序 lag/overflow/边界音频修复可能在尾部再引入新的未收敛边界，
        # 这里再执行一次定点重匹配，避免“最后几段重复词/断词”残留到最终输出。
        if unresolved > 0:
            final_rematch_attempts = max(0, int(getattr(self, "no_target_boundary_rematch_max_attempts", 0)))
            if final_rematch_attempts > 0:
                final_rematched = self._rematch_unresolved_boundaries_without_target_fallback(
                    segments,
                    max_attempts=final_rematch_attempts,
                )
                if final_rematched > 0:
                    adjusted += int(final_rematched)
                    unresolved = self._count_no_target_boundary_unresolved(segments)
                    hard_after_final_rematch = self._enforce_boundary_hard_constraints_no_target(segments, max_passes=1)
                    if hard_after_final_rematch > 0:
                        adjusted += int(hard_after_final_rematch)
                        boundary_hard_clamped += int(hard_after_final_rematch)
                        unresolved = self._count_no_target_boundary_unresolved(segments)
                    audio_repaired_final, audio_issues_final = self._repair_boundary_audio_locally_no_target(
                        segments,
                        max_passes=1,
                    )
                    if audio_repaired_final > 0:
                        adjusted += int(audio_repaired_final)
                    boundary_audio_issues += int(audio_issues_final)
                    boundary_audio_repaired += int(audio_repaired_final)
                    unresolved = self._count_no_target_boundary_unresolved(segments)

        tail_locked_prev_trimmed = self._trim_prev_for_locked_tail_severe_overlaps_no_target(segments, max_passes=1)
        if tail_locked_prev_trimmed > 0:
            adjusted += int(tail_locked_prev_trimmed)
            unresolved = self._count_no_target_boundary_unresolved(segments)

        tail_shortfall_trimmed = self._trim_tail_overlaps_with_shortfall_tolerance_no_target(segments)
        if tail_shortfall_trimmed > 0:
            adjusted += int(tail_shortfall_trimmed)
            unresolved = self._count_no_target_boundary_unresolved(segments)

        self.last_boundary_repair_stats = {
            "hard_clamped": int(boundary_hard_clamped),
            "audio_issue_detected": int(boundary_audio_issues),
            "audio_repaired": int(boundary_audio_repaired),
            "tail_locked_prev_trimmed": int(tail_locked_prev_trimmed),
            "tail_shortfall_trimmed": int(tail_shortfall_trimmed),
        }

        return adjusted, unresolved

    def _prepare_cross_source_shortfall_bridges_no_target(self, segments: List[dict]) -> int:
        """
        跨源短缺拼桥（禁兜底）：
        - 当前段命中源尾、与上一同源段存在可感知重复时，
          若下一段来自新源且接近片头，则允许把下一源的开头借到当前段尾部；
        - 同步顺延下一段起点，避免“前段重复 + 后段缺字/断音”反复出现。
        """
        if self.enable_target_video_fallback:
            return 0
        if not bool(getattr(self, "cross_source_shortfall_bridge_enabled", True)):
            return 0
        if len(segments) < 3:
            return 0

        min_shortfall = max(0.05, float(getattr(self, "cross_source_shortfall_bridge_min_sec", 0.12)))
        max_shortfall = max(min_shortfall, float(getattr(self, "cross_source_shortfall_bridge_max_sec", 1.20)))
        next_head_window = max(0.2, float(getattr(self, "cross_source_shortfall_bridge_next_head_window", 0.95)))
        base_search_cap = max(0.0, float(getattr(self, "cross_source_shortfall_bridge_base_search_cap", 0.40)))
        extra_trim_cap = max(0.0, float(getattr(self, "cross_source_shortfall_bridge_extra_trim_cap", 0.90)))
        repaired = 0

        for i in range(1, len(segments) - 1):
            prev = segments[i - 1]
            curr = segments[i]
            nxt = segments[i + 1]

            if prev["source"] == self.target_video or curr["source"] == self.target_video or nxt["source"] == self.target_video:
                continue
            if prev["source"] != curr["source"]:
                continue
            if curr["source"] == nxt["source"]:
                continue
            if self._is_secondary_source(Path(curr["source"])) or self._is_secondary_source(Path(nxt["source"])):
                continue

            curr_q = curr.get("quality", {}) or {}
            if bool(curr_q.get("cross_source_shortfall_bridge_no_target", False)):
                continue

            curr_duration = float(curr["duration"])
            next_duration = float(nxt["duration"])
            curr_source_duration = self.get_video_duration(curr["source"])
            next_source_duration = self.get_video_duration(nxt["source"])
            if curr_source_duration <= 0.0 or next_source_duration <= 0.0:
                continue

            prev_end = float(prev["start"]) + float(prev["duration"])
            curr_start = float(curr["start"])
            overlap_trim = max(0.0, float(prev_end - curr_start))
            if overlap_trim < max(0.08, curr_duration * 0.015):
                continue

            effective_curr_start = max(float(curr_start), float(prev_end))
            curr_available = max(0.0, float(curr_source_duration - effective_curr_start))
            if curr_available >= (curr_duration - 0.06):
                continue
            shortfall = float(curr_duration - curr_available)
            if shortfall < min_shortfall or shortfall > max_shortfall:
                continue
            if curr_available < max(1.0, curr_duration * 0.55):
                continue

            next_start_now = float(nxt["start"])
            if next_start_now > (next_head_window + shortfall + 0.12):
                continue

            max_next_start = max(0.0, float(next_source_duration - next_duration))
            if max_next_start <= 0.0:
                continue

            curr_target_start = float(curr["target_start"])
            next_target_start = float(nxt["target_start"])
            base_passed, base_avg = self.quick_verify(
                source=nxt["source"],
                source_start=float(next_start_now),
                target_start=next_target_start,
                duration=next_duration,
            )
            if not base_passed:
                continue
            base_audio_passed, base_audio_meta = self.quick_verify_audio(
                source=nxt["source"],
                source_start=float(next_start_now),
                target_start=next_target_start,
                duration=next_duration,
                combined_score=float(base_avg),
            )
            base_score = (
                float(base_avg)
                + float(base_audio_meta.get("aligned_similarity", 0.0) or 0.0) * 0.03
                - float(base_audio_meta.get("shift_gain", 0.0) or 0.0) * 0.06
            )
            if not base_audio_passed:
                base_score -= 0.25

            baseline_curr_start = float(curr["start"])
            baseline_curr_passed, baseline_curr_avg = self.quick_verify(
                source=curr["source"],
                source_start=baseline_curr_start,
                target_start=curr_target_start,
                duration=curr_duration,
            )
            if baseline_curr_passed:
                baseline_curr_audio_passed, baseline_curr_audio_meta = self.quick_verify_audio(
                    source=curr["source"],
                    source_start=baseline_curr_start,
                    target_start=curr_target_start,
                    duration=curr_duration,
                    combined_score=float(baseline_curr_avg),
                )
                baseline_curr_score = (
                    float(baseline_curr_avg)
                    + float(baseline_curr_audio_meta.get("aligned_similarity", 0.0) or 0.0) * 0.03
                    - float(baseline_curr_audio_meta.get("shift_gain", 0.0) or 0.0) * 0.06
                )
                if not baseline_curr_audio_passed:
                    baseline_curr_score -= 0.25
            else:
                baseline_curr_audio_meta = {"checked": False, "reason": "baseline_curr_single_source_failed"}
                baseline_curr_score = float(curr_q.get("combined", 0.0) or 0.0)
            base_total_score = float(base_score) + float(baseline_curr_score)

            base_candidates = [0.0, 0.1, 0.2, 0.3, 0.4]
            dynamic_candidates = [
                max(0.0, float(next_start_now - shortfall)),
                min(float(next_start_now), float(base_search_cap)),
                min(float(base_search_cap), float(shortfall)),
            ]
            candidate_bases: List[float] = []
            seen_base: Set[float] = set()
            for raw_base in [*base_candidates, *dynamic_candidates]:
                base = max(0.0, min(float(raw_base), float(base_search_cap), float(max_next_start)))
                key = round(float(base), 3)
                if key in seen_base:
                    continue
                seen_base.add(key)
                candidate_bases.append(float(base))

            max_extra_trim = min(
                float(extra_trim_cap),
                max(0.0, float(curr_available - max(0.35, curr_duration * 0.58))),
                max(0.0, float(max_shortfall - shortfall)),
            )
            extra_trim_candidates_raw = [0.0, 0.06, 0.12, 0.18, 0.24, 0.32, 0.48, 0.64, 0.80]
            extra_trim_candidates: List[float] = []
            seen_trim: Set[float] = set()
            for raw_trim in extra_trim_candidates_raw:
                trim = max(0.0, min(float(raw_trim), float(max_extra_trim)))
                key = round(float(trim), 3)
                if key in seen_trim:
                    continue
                seen_trim.add(key)
                extra_trim_candidates.append(float(trim))

            best_choice: Optional[Dict[str, object]] = None
            best_score: Optional[float] = None

            for extra_trim in extra_trim_candidates:
                curr_first_start = float(effective_curr_start + extra_trim)
                first_part_duration = max(0.0, float(curr_source_duration - curr_first_start))
                if first_part_duration < max(0.35, curr_duration * 0.55):
                    continue
                borrow_duration = float(curr_duration - first_part_duration)
                if borrow_duration < min_shortfall or borrow_duration > max_shortfall:
                    continue

                for base in candidate_bases:
                    new_next_start = float(base + borrow_duration)
                    if new_next_start > (max_next_start + 1e-6):
                        continue

                    if (i + 2) < len(segments):
                        nxt2 = segments[i + 2]
                        if nxt2["source"] == nxt["source"] and nxt2["source"] != self.target_video:
                            next_gap = float(nxt2["start"]) - (float(new_next_start) + next_duration)
                            if next_gap < -max(0.30, next_duration * 0.04):
                                continue
                        else:
                            next_gap = None
                    else:
                        next_gap = None

                    composite_parts = [
                        {
                            "source": str(curr["source"]),
                            "start": float(curr_first_start),
                            "duration": float(first_part_duration),
                        },
                        {
                            "source": str(nxt["source"]),
                            "start": float(base),
                            "duration": float(borrow_duration),
                        },
                    ]
                    curr_probe = self._probe_composite_parts_against_target(
                        composite_parts,
                        target_start=curr_target_start,
                        duration=curr_duration,
                    )
                    if curr_probe is None:
                        continue

                    passed, avg = self.quick_verify(
                        source=nxt["source"],
                        source_start=float(new_next_start),
                        target_start=next_target_start,
                        duration=next_duration,
                    )
                    if not passed:
                        continue

                    audio_passed, audio_meta = self.quick_verify_audio(
                        source=nxt["source"],
                        source_start=float(new_next_start),
                        target_start=next_target_start,
                        duration=next_duration,
                        combined_score=float(avg),
                    )
                    if not audio_passed:
                        continue

                    next_score = (
                        float(avg)
                        + float(audio_meta.get("aligned_similarity", 0.0) or 0.0) * 0.03
                        - float(audio_meta.get("shift_gain", 0.0) or 0.0) * 0.06
                        - abs(float(new_next_start) - float(next_start_now)) * 0.01
                        - float(base) * 0.01
                    )
                    if next_gap is not None:
                        next_score -= abs(float(next_gap)) * 0.08

                    total_score = float(curr_probe["score"]) * 1.35 + float(next_score)
                    total_score -= float(extra_trim) * 0.02
                    if not baseline_curr_passed:
                        # 单源当前段本身已不足以覆盖目标内容时，优先给更长的借段机会，
                        # 避免只补到“刚够时长”却仍吞掉句尾关键词。
                        total_score += float(borrow_duration) * 0.12
                    if best_score is None or total_score > best_score + 1e-6:
                        best_score = float(total_score)
                        best_choice = {
                            "curr_start": float(curr_first_start),
                            "first_part_duration": float(first_part_duration),
                            "borrow_duration": float(borrow_duration),
                            "base_start": float(base),
                            "next_start": float(new_next_start),
                            "verify_avg": float(avg),
                            "audio_meta": audio_meta,
                            "next_gap": None if next_gap is None else float(next_gap),
                            "curr_probe": curr_probe,
                            "curr_parts": composite_parts,
                            "extra_trim": float(extra_trim),
                        }

            if best_choice is None:
                continue
            if float(best_score) + 0.015 < float(base_total_score):
                continue

            old_curr_start = float(curr["start"])
            old_next_start = float(nxt["start"])
            curr["start"] = float(best_choice["curr_start"])
            curr["composite_parts"] = list(best_choice["curr_parts"])
            nxt["start"] = float(best_choice["next_start"])

            curr_q["cross_source_shortfall_bridge_no_target"] = True
            curr_q["cross_source_shortfall_bridge_curr_start_from"] = float(old_curr_start)
            curr_q["cross_source_shortfall_bridge_curr_start_to"] = float(best_choice["curr_start"])
            curr_q["cross_source_shortfall_bridge_overlap_trim_sec"] = float(overlap_trim)
            curr_q["cross_source_shortfall_bridge_shortfall_sec"] = float(best_choice["borrow_duration"])
            curr_q["cross_source_shortfall_bridge_first_part_duration"] = float(best_choice["first_part_duration"])
            curr_q["cross_source_shortfall_bridge_extra_trim_sec"] = float(best_choice["extra_trim"])
            curr_q["cross_source_shortfall_bridge_borrow_source"] = str(nxt["source"])
            curr_q["cross_source_shortfall_bridge_borrow_start"] = float(best_choice["base_start"])
            curr_q["cross_source_shortfall_bridge_borrow_duration"] = float(best_choice["borrow_duration"])
            curr_q["cross_source_shortfall_bridge_next_start_from"] = float(old_next_start)
            curr_q["cross_source_shortfall_bridge_next_start_to"] = float(best_choice["next_start"])
            curr_q["cross_source_shortfall_bridge_next_verify_avg"] = float(best_choice["verify_avg"])
            curr_q["cross_source_shortfall_bridge_next_audio_guard"] = best_choice["audio_meta"]
            curr_q["cross_source_shortfall_bridge_parts"] = list(curr["composite_parts"])
            curr_q["cross_source_shortfall_bridge_curr_verify_avg"] = float(best_choice["curr_probe"]["verify_avg"])
            curr_q["cross_source_shortfall_bridge_curr_audio_guard"] = best_choice["curr_probe"]["audio_meta"]
            curr_q["cross_source_shortfall_bridge_total_score"] = float(best_score)
            curr_q["cross_source_shortfall_bridge_baseline_curr_single_source_passed"] = bool(baseline_curr_passed)
            curr_q["cross_source_shortfall_bridge_baseline_curr_score"] = float(baseline_curr_score)
            curr_q["cross_source_shortfall_bridge_baseline_curr_audio_guard"] = baseline_curr_audio_meta
            curr["quality"] = curr_q

            next_q = nxt.get("quality", {}) or {}
            next_q["cross_source_shortfall_bridge_from_prev"] = True
            next_q["cross_source_shortfall_bridge_prev_index"] = int(curr["index"])
            next_q["cross_source_shortfall_bridge_prev_source"] = str(curr["source"])
            next_q["cross_source_shortfall_bridge_prev_borrow_start"] = float(best_choice["base_start"])
            next_q["cross_source_shortfall_bridge_prev_borrow_duration"] = float(best_choice["borrow_duration"])
            next_q["cross_source_shortfall_bridge_start_from"] = float(old_next_start)
            next_q["cross_source_shortfall_bridge_start_to"] = float(best_choice["next_start"])
            next_q["cross_source_shortfall_bridge_verify_avg"] = float(best_choice["verify_avg"])
            next_q["cross_source_shortfall_bridge_audio_guard"] = best_choice["audio_meta"]
            if best_choice.get("next_gap") is not None:
                next_q["cross_source_shortfall_bridge_next_gap"] = float(best_choice["next_gap"])
            nxt["quality"] = next_q
            repaired += 1

        return int(repaired)

    def _prepare_cross_source_prev_tail_carryovers_no_target(self, segments: List[dict]) -> int:
        """
        跨源前源尾巴接入（禁兜底）：
        - 上一源仍剩一小段尾巴、下一段却直接从新源片头开始时，
          允许把上一源尾巴接到当前段开头；
        - 同步把当前源后续同源 run 向前平移，避免切源处丢掉 0.5~1.5 秒内容。
        """
        if self.enable_target_video_fallback:
            return 0
        if len(segments) < 2:
            return 0

        min_borrow = max(0.10, float(getattr(self, "cross_source_shortfall_bridge_min_sec", 0.12)))
        max_borrow = max(
            min_borrow,
            min(1.60, float(getattr(self, "cross_source_shortfall_bridge_max_sec", 1.20)) + 0.25),
        )
        head_window = max(
            0.25,
            float(getattr(self, "cross_source_shortfall_bridge_next_head_window", 0.95)) + 0.10,
        )
        verify_window_sec = max(10.0, float(self.segment_duration) * 5.0)
        repaired = 0

        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev["source"] == self.target_video or curr["source"] == self.target_video:
                continue
            if prev["source"] == curr["source"]:
                continue
            if self._is_secondary_source(Path(prev["source"])) or self._is_secondary_source(Path(curr["source"])):
                continue

            curr_q = curr.get("quality", {}) or {}
            if bool(curr_q.get("cross_source_prev_tail_carryover_no_target", False)):
                continue

            prev_src_dur = self.get_video_duration(prev["source"])
            if prev_src_dur <= 0.0:
                continue
            prev_end = float(prev["start"]) + float(prev["duration"])
            prev_tail_left = max(0.0, float(prev_src_dur - prev_end))
            if prev_tail_left < min_borrow:
                continue

            curr_start = float(curr["start"])
            if curr_start > head_window:
                continue

            run_end = i
            while (run_end + 1) < len(segments) and segments[run_end + 1]["source"] == curr["source"]:
                run_end += 1
            if run_end == i:
                continue

            raw_borrows = [
                min(float(prev_tail_left), float(max_borrow)),
                min(float(prev_tail_left), float(max_borrow), 1.20),
                min(float(prev_tail_left), float(max_borrow), 1.00),
                min(float(prev_tail_left), float(max_borrow), 0.85),
                min(float(prev_tail_left), float(max_borrow), 0.65),
            ]
            borrow_candidates: List[float] = []
            seen_borrow: Set[float] = set()
            for raw in raw_borrows:
                borrow = max(0.0, min(float(raw), float(curr["duration"]) - 0.6))
                key = round(float(borrow), 3)
                if borrow < min_borrow or key in seen_borrow:
                    continue
                seen_borrow.add(key)
                borrow_candidates.append(float(borrow))
            if not borrow_candidates:
                continue

            best_choice: Optional[Dict[str, object]] = None
            best_score: Optional[float] = None
            best_carryover_choice: Optional[Dict[str, object]] = None
            best_carryover_score: Optional[float] = None
            best_strong_head_only_choice: Optional[Dict[str, object]] = None
            best_strong_head_only_score: Optional[float] = None
            def consider_choice(choice: Dict[str, object], score: float) -> None:
                nonlocal best_choice, best_score, best_carryover_choice, best_carryover_score
                if best_score is None or float(score) > float(best_score):
                    best_score = float(score)
                    best_choice = dict(choice)
                if str(choice.get("mode")) == "carryover":
                    if best_carryover_score is None or float(score) > float(best_carryover_score):
                        best_carryover_score = float(score)
                        best_carryover_choice = dict(choice)

            for borrow in borrow_candidates:
                composite_parts = [
                    {
                        "source": str(prev["source"]),
                        "start": float(prev_end),
                        "duration": float(borrow),
                    },
                    {
                        "source": str(curr["source"]),
                        "start": float(curr_start),
                        "duration": float(curr["duration"] - borrow),
                    },
                ]
                curr_probe = self._probe_composite_parts_against_target(
                    composite_parts,
                    target_start=float(curr["target_start"]),
                    duration=float(curr["duration"]),
                )
                if not isinstance(curr_probe, dict):
                    continue
                if not bool(curr_probe.get("passed", False)):
                    continue

                shifted_starts: Dict[int, float] = {}
                shifted_audio_meta: Dict[int, Dict[str, object]] = {}
                shifted_verify_avg: Dict[int, float] = {}
                run_ok = True
                total_score = float(curr_probe.get("verify_avg", 0.0)) + float(curr_probe.get("aligned_similarity", 0.0)) * 0.08
                verified_follow_duration = 0.0
                verified_follow_segments = 0
                for k in range(i + 1, run_end + 1):
                    seg = segments[k]
                    new_start = float(seg["start"]) - float(borrow)
                    if new_start < 0.0:
                        run_ok = False
                        break
                    shifted_starts[k] = float(new_start)
                    if verified_follow_duration < verify_window_sec:
                        passed, avg = self.quick_verify(
                            source=seg["source"],
                            source_start=float(new_start),
                            target_start=float(seg["target_start"]),
                            duration=float(seg["duration"]),
                        )
                        audio_passed, audio_meta = self.quick_verify_audio(
                            source=seg["source"],
                            source_start=float(new_start),
                            target_start=float(seg["target_start"]),
                            duration=float(seg["duration"]),
                            combined_score=float(avg),
                        )
                        if not audio_passed:
                            run_ok = False
                            break
                        aligned = float((audio_meta or {}).get("aligned_similarity", 0.0) or 0.0)
                        relaxed_ok = bool((not passed) and float(avg) >= 0.90 and aligned >= 0.66)
                        if (not passed) and (not relaxed_ok):
                            run_ok = False
                            break
                        best_shift = float((audio_meta or {}).get("best_shift_sec", 0.0) or 0.0)
                        shift_gain = float((audio_meta or {}).get("shift_gain", 0.0) or 0.0)
                        if abs(best_shift) >= 0.45 and shift_gain >= 0.10:
                            run_ok = False
                            break
                        shifted_audio_meta[k] = dict(audio_meta or {})
                        shifted_verify_avg[k] = float(avg)
                        total_score += float(avg) + aligned * 0.05
                        verified_follow_duration += float(seg["duration"])
                        verified_follow_segments += 1

                if not run_ok:
                    continue
                if verified_follow_segments <= 0:
                    continue

                score = float(total_score) - float(borrow) * 0.03
                consider_choice(
                    {
                        "mode": "carryover",
                        "borrow": float(borrow),
                        "curr_parts": composite_parts,
                        "curr_probe": curr_probe,
                        "shifted_starts": shifted_starts,
                        "shifted_audio_meta": shifted_audio_meta,
                        "shifted_verify_avg": shifted_verify_avg,
                    },
                    score,
                )

            # 有些边界表面看像“前源尾巴 + 当前源片头”，
            # 但实际台词主体已经完整落在当前源片头里。
            # 这时继续借前源尾巴，反而会把无关内容混进来。
            head_start_candidates: List[float] = []
            seen_head: Set[float] = set()
            for raw in (0.0, min(curr_start, 0.12), 0.30, 0.50):
                cand = max(0.0, min(float(raw), float(head_window)))
                key = round(cand, 3)
                if key in seen_head:
                    continue
                seen_head.add(key)
                head_start_candidates.append(float(cand))

            for head_start in head_start_candidates:
                passed, avg = self.quick_verify(
                    source=curr["source"],
                    source_start=float(head_start),
                    target_start=float(curr["target_start"]),
                    duration=float(curr["duration"]),
                )
                if not passed:
                    continue
                audio_passed, audio_meta = self.quick_verify_audio(
                    source=curr["source"],
                    source_start=float(head_start),
                    target_start=float(curr["target_start"]),
                    duration=float(curr["duration"]),
                    combined_score=float(avg),
                )
                if not audio_passed:
                    continue
                aligned = float((audio_meta or {}).get("aligned_similarity", 0.0) or 0.0)
                delta = float(curr_start - head_start)
                shifted_starts: Dict[int, float] = {}
                shifted_audio_meta: Dict[int, Dict[str, object]] = {}
                shifted_verify_avg: Dict[int, float] = {}
                run_ok = True
                total_score = float(avg) + aligned * 0.08
                verified_follow_duration = 0.0
                verified_follow_segments = 0
                for k in range(i + 1, run_end + 1):
                    seg = segments[k]
                    new_start = float(seg["start"]) - float(delta)
                    if new_start < 0.0:
                        run_ok = False
                        break
                    shifted_starts[k] = float(new_start)
                    if verified_follow_duration < verify_window_sec:
                        passed_follow, avg_follow = self.quick_verify(
                            source=seg["source"],
                            source_start=float(new_start),
                            target_start=float(seg["target_start"]),
                            duration=float(seg["duration"]),
                        )
                        audio_passed_follow, audio_meta_follow = self.quick_verify_audio(
                            source=seg["source"],
                            source_start=float(new_start),
                            target_start=float(seg["target_start"]),
                            duration=float(seg["duration"]),
                            combined_score=float(avg_follow),
                        )
                        if not audio_passed_follow:
                            run_ok = False
                            break
                        aligned_follow = float((audio_meta_follow or {}).get("aligned_similarity", 0.0) or 0.0)
                        relaxed_ok = bool((not passed_follow) and float(avg_follow) >= 0.90 and aligned_follow >= 0.66)
                        if (not passed_follow) and (not relaxed_ok):
                            run_ok = False
                            break
                        best_shift_follow = float((audio_meta_follow or {}).get("best_shift_sec", 0.0) or 0.0)
                        shift_gain_follow = float((audio_meta_follow or {}).get("shift_gain", 0.0) or 0.0)
                        if abs(best_shift_follow) >= 0.45 and shift_gain_follow >= 0.10:
                            run_ok = False
                            break
                        shifted_audio_meta[k] = dict(audio_meta_follow or {})
                        shifted_verify_avg[k] = float(avg_follow)
                        total_score += float(avg_follow) + aligned_follow * 0.05
                        verified_follow_duration += float(seg["duration"])
                        verified_follow_segments += 1

                if not run_ok:
                    continue
                if verified_follow_segments <= 0:
                    continue

                # 头部直接命中当前源的情况下，适当偏向“单源片头重锚定”而不是跨源拼桥，
                # 避免把无关尾巴借进来。
                score = float(total_score) + 0.04 - float(head_start) * 0.02
                choice = {
                    "mode": "head_only",
                    "head_start": float(head_start),
                    "delta": float(delta),
                    "verify_avg": float(avg),
                    "audio_meta": dict(audio_meta or {}),
                    "shifted_starts": shifted_starts,
                    "shifted_audio_meta": shifted_audio_meta,
                    "shifted_verify_avg": shifted_verify_avg,
                }
                consider_choice(choice, score)
                if head_start <= 0.12 and float(avg) >= 0.92 and float(aligned) >= 0.60:
                    if best_strong_head_only_score is None or float(score) > float(best_strong_head_only_score):
                        best_strong_head_only_score = float(score)
                        best_strong_head_only_choice = dict(choice)

            if best_choice is None:
                continue
            if best_strong_head_only_choice is not None:
                allow_strong_head_only = bool(prev_tail_left <= max(min_borrow * 1.5, 0.45))
                if (not allow_strong_head_only) and best_carryover_score is not None and best_strong_head_only_score is not None:
                    allow_strong_head_only = bool(
                        float(best_strong_head_only_score) >= float(best_carryover_score) + 0.045
                    )
                if allow_strong_head_only:
                    best_choice = dict(best_strong_head_only_choice)

            if str(best_choice.get("mode")) == "head_only":
                curr.pop("composite_parts", None)
                old_curr_start = float(curr["start"])
                curr["start"] = float(best_choice["head_start"])
                curr_q["cross_source_head_only_reanchor_no_target"] = True
                curr_q["cross_source_head_only_reanchor_from"] = float(old_curr_start)
                curr_q["cross_source_head_only_reanchor_to"] = float(best_choice["head_start"])
                curr_q["cross_source_head_only_reanchor_shift_sec"] = float(best_choice["delta"])
                curr_q["cross_source_head_only_reanchor_verify_avg"] = float(best_choice["verify_avg"])
                curr_q["audio_guard"] = best_choice["audio_meta"]
                curr["quality"] = curr_q
            else:
                curr["composite_parts"] = list(best_choice["curr_parts"])
                curr_q["cross_source_prev_tail_carryover_no_target"] = True
                curr_q["cross_source_prev_tail_carryover_prev_source"] = str(prev["source"])
                curr_q["cross_source_prev_tail_carryover_prev_tail_start"] = float(prev_end)
                curr_q["cross_source_prev_tail_carryover_prev_tail_duration"] = float(best_choice["borrow"])
                curr_q["cross_source_prev_tail_carryover_curr_source_start"] = float(curr_start)
                curr_q["cross_source_prev_tail_carryover_curr_source_duration"] = float(curr["duration"] - best_choice["borrow"])
                curr_q["cross_source_prev_tail_carryover_verify_avg"] = float(best_choice["curr_probe"].get("verify_avg", 0.0))
                curr_q["cross_source_prev_tail_carryover_audio_guard"] = best_choice["curr_probe"].get("audio_meta", {})
                curr["quality"] = curr_q

            for k, new_start in best_choice["shifted_starts"].items():
                seg = segments[k]
                old_start = float(seg["start"])
                seg["start"] = float(new_start)
                q = seg.get("quality", {}) or {}
                if str(best_choice.get("mode")) == "head_only":
                    q["cross_source_head_only_shifted_no_target"] = True
                    q["cross_source_head_only_shift_from"] = float(old_start)
                    q["cross_source_head_only_shift_to"] = float(new_start)
                    q["cross_source_head_only_shift_sec"] = float(best_choice["delta"])
                    q["cross_source_head_only_anchor_index"] = int(i)
                else:
                    q["cross_source_prev_tail_carryover_shifted_no_target"] = True
                    q["cross_source_prev_tail_carryover_shift_from"] = float(old_start)
                    q["cross_source_prev_tail_carryover_shift_to"] = float(new_start)
                    q["cross_source_prev_tail_carryover_shift_sec"] = float(best_choice["borrow"])
                    q["cross_source_prev_tail_carryover_anchor_index"] = int(i)
                q["audio_guard"] = best_choice["shifted_audio_meta"].get(k, q.get("audio_guard", {}))
                seg["quality"] = q

            repaired += 1

        return int(repaired)

    def _probe_composite_parts_against_target(
        self,
        parts: List[dict],
        target_start: float,
        duration: float,
    ) -> Optional[Dict[str, object]]:
        """临时拼接 composite 段并复用现有 visual/audio 守卫评分。"""
        probe_clip = self.temp_dir / f"bridge_probe_{uuid.uuid4().hex}.mp4"
        output_fps = max(12.0, float(self.output_fps or 24.0))
        fps_expr = f"fps={output_fps:.6f},format=yuv420p"
        ok, _ = self._extract_composite_av_clip(
            parts,
            probe_clip,
            fps_expr,
            total_duration=float(duration),
            expected_frames=max(1, int(round(float(duration) * output_fps))),
            include_audio=True,
        )
        if not ok:
            probe_clip.unlink(missing_ok=True)
            return None
        try:
            passed, avg = self.quick_verify(
                source=probe_clip,
                source_start=0.0,
                target_start=float(target_start),
                duration=float(duration),
            )
            if not passed:
                return None
            audio_passed, audio_meta = self.quick_verify_audio(
                source=probe_clip,
                source_start=0.0,
                target_start=float(target_start),
                duration=float(duration),
                combined_score=float(avg),
            )
            if not audio_passed:
                return None
            score = (
                float(avg)
                + float(audio_meta.get("aligned_similarity", 0.0) or 0.0) * 0.03
                - float(audio_meta.get("shift_gain", 0.0) or 0.0) * 0.06
            )
            return {
                "passed": True,
                "verify_avg": float(avg),
                "audio_meta": audio_meta,
                "score": float(score),
            }
        finally:
            probe_clip.unlink(missing_ok=True)

    def _extract_composite_av_clip(
        self,
        parts: List[dict],
        output_clip: Path,
        fps_expr: str,
        total_duration: float,
        expected_frames: int = 0,
        include_audio: bool = True,
    ) -> Tuple[bool, str]:
        """提取并拼接 composite 段（多源桥接段）。"""
        usable_parts = [dict(part) for part in parts if float(part.get("duration", 0.0) or 0.0) > 0.03]
        if not usable_parts:
            return False, "empty_composite_parts"

        output_clip.unlink(missing_ok=True)
        seg_ext = output_clip.suffix or ".mp4"
        output_fps = max(12.0, float(self.output_fps or 24.0))
        part_clips: List[Path] = []
        errors: List[str] = []

        for part_idx, part in enumerate(usable_parts):
            part_duration = float(part.get("duration", 0.0) or 0.0)
            part_clip = output_clip.with_name(f"{output_clip.stem}.part{part_idx}{seg_ext}")
            part_expected_frames = max(1, int(round(part_duration * output_fps))) if expected_frames else 0
            ok, err = self._extract_av_clip(
                Path(str(part["source"])),
                float(part["start"]),
                float(part_duration),
                part_clip,
                fps_expr,
                expected_frames=part_expected_frames,
                include_audio=include_audio,
            )
            if not ok:
                errors.append(f"part{part_idx}:{err}")
                for clip in part_clips:
                    clip.unlink(missing_ok=True)
                part_clip.unlink(missing_ok=True)
                return False, " | ".join(errors[:3])
            part_clips.append(part_clip)

        if len(part_clips) == 1:
            os.replace(part_clips[0], output_clip)
            return True, ""

        gop = max(12, int(round(float(self.output_fps or 25.0))))
        concat_cmd: List[str] = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
        for clip in part_clips:
            concat_cmd.extend(['-i', str(clip)])

        if include_audio:
            concat_inputs = "".join(f"[{idx}:v:0][{idx}:a:0]" for idx in range(len(part_clips)))
            filter_expr = (
                f"{concat_inputs}concat=n={len(part_clips)}:v=1:a=1[v][a_raw];"
                f"[a_raw]aresample=async=1:first_pts=0,atrim=0:{float(total_duration):.6f},"
                f"asetpts=PTS-STARTPTS[a]"
            )
        else:
            concat_inputs = "".join(f"[{idx}:v:0]" for idx in range(len(part_clips)))
            filter_expr = f"{concat_inputs}concat=n={len(part_clips)}:v=1:a=0[v]"

        concat_cmd.extend([
            '-filter_complex', filter_expr,
            '-map', '[v]',
            '-t', f"{float(total_duration):.6f}",
            '-reset_timestamps', '1',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-bf', '0',
            '-g', str(gop),
            '-keyint_min', str(gop),
            '-sc_threshold', '0',
        ])
        if include_audio:
            concat_cmd.extend([
                '-map', '[a]',
                '-ar', '48000',
                '-ac', '2',
            ])
            if bool(getattr(self, "segment_intermediate_pcm_audio", False)):
                concat_cmd.extend(['-c:a', 'pcm_s16le'])
            else:
                concat_cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        else:
            concat_cmd.extend(['-an'])
            if expected_frames > 0:
                concat_cmd.extend(['-frames:v', str(max(1, int(expected_frames)))])
        concat_cmd.append(str(output_clip))

        concat_proc = subprocess.run(concat_cmd, capture_output=True, text=True)
        for clip in part_clips:
            clip.unlink(missing_ok=True)
        if concat_proc.returncode != 0 or (not output_clip.exists()) or output_clip.stat().st_size <= 0:
            output_clip.unlink(missing_ok=True)
            return False, f"composite_concat_failed:{(concat_proc.stderr or '').strip()}"

        return True, ""

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
        start_sec = 0.0 if abs(float(start)) < 5e-4 else max(0.0, float(start))
        duration_sec = max(1e-3, float(duration))
        start_arg = f"{start_sec:.6f}"
        duration_arg = f"{duration_sec:.6f}"
        use_pcm_audio = bool(include_audio and getattr(self, "segment_intermediate_pcm_audio", False))
        # 播放兼容优先：固定 GOP + 关闭 B 帧，降低个别播放器“音频前进/画面卡住”观感概率。
        gop = max(12, int(round(float(self.output_fps or 25.0))))
        prefer_accurate_seek = bool(include_audio and getattr(self, "audio_segment_accurate_seek", True))
        seek_plan = [prefer_accurate_seek]
        if include_audio and (not prefer_accurate_seek):
            seek_plan.append(True)
        elif include_audio and prefer_accurate_seek:
            seek_plan.append(False)

        errors: List[str] = []

        def build_extract_cmd(use_accurate_seek: bool) -> List[str]:
            if use_accurate_seek:
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(source),
                    '-ss', start_arg,
                    '-t', duration_arg,
                    '-vf', fps_expr,
                    '-reset_timestamps', '1',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-bf', '0',
                    '-g', str(gop),
                    '-keyint_min', str(gop),
                    '-sc_threshold', '0',
                ]
            else:
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-ss', start_arg,
                    '-t', duration_arg,
                    '-i', str(source),
                    '-vf', fps_expr,
                    '-reset_timestamps', '1',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-bf', '0',
                    '-g', str(gop),
                    '-keyint_min', str(gop),
                    '-sc_threshold', '0',
                ]
            if include_audio:
                audio_chain = [
                    '-af', f'aresample=async=1:first_pts=0,atrim=0:{duration_sec:.6f},asetpts=PTS-STARTPTS',
                    '-ar', '48000',
                    '-ac', '2',
                ]
                if use_pcm_audio:
                    audio_chain.extend(['-c:a', 'pcm_s16le'])
                else:
                    audio_chain.extend(['-c:a', 'aac', '-b:a', '128k'])
                cmd.extend(audio_chain)
            else:
                cmd.extend(['-an'])
                if frame_count > 0:
                    cmd.extend(['-frames:v', str(frame_count)])
            cmd.append(str(output_clip))
            return cmd

        def pad_shortfall_if_needed(attempt_tag: str, shortfall: float) -> Tuple[bool, str]:
            if shortfall <= 1e-6:
                return True, ""

            padded_clip = output_clip.with_name(f"{output_clip.stem}.{attempt_tag}.pad.mp4")
            pad_vf = f"{fps_expr},tpad=stop_mode=clone:stop_duration={shortfall:.6f}"
            pad_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(output_clip),
                '-vf', pad_vf,
                '-t', f"{duration_sec:.6f}",
                '-reset_timestamps', '1',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-bf', '0',
                '-g', str(gop),
                '-keyint_min', str(gop),
                '-sc_threshold', '0',
            ]
            if include_audio:
                audio_chain = [
                    '-af', f'aresample=async=1:first_pts=0,apad,atrim=0:{duration_sec:.6f},asetpts=PTS-STARTPTS',
                    '-ar', '48000',
                    '-ac', '2',
                ]
                if use_pcm_audio:
                    audio_chain.extend(['-c:a', 'pcm_s16le'])
                else:
                    audio_chain.extend(['-c:a', 'aac', '-b:a', '128k'])
                pad_cmd.extend(audio_chain)
            else:
                pad_cmd.extend(['-an'])
            pad_cmd.append(str(padded_clip))

            pad_proc = subprocess.run(pad_cmd, capture_output=True, text=True)
            if pad_proc.returncode != 0 or (not padded_clip.exists()) or padded_clip.stat().st_size <= 0:
                padded_clip.unlink(missing_ok=True)
                return False, (pad_proc.stderr or "").strip()

            os.replace(padded_clip, output_clip)
            return True, ""

        for attempt_idx, use_accurate_seek in enumerate(seek_plan):
            attempt_tag = f"attempt{attempt_idx + 1}"
            cmd = build_extract_cmd(use_accurate_seek=bool(use_accurate_seek))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                mode = "accurate" if use_accurate_seek else "fast"
                errors.append(f"{mode}_seek_extract_failed:{(result.stderr or '').strip()}")
                continue
            if (not output_clip.exists()) or output_clip.stat().st_size <= 0:
                errors.append("empty_output")
                continue

            self._invalidate_media_duration_cache(output_clip)
            clip_video_dur = float(self.get_video_duration(output_clip))
            clip_audio_dur = float(self.get_audio_duration(output_clip)) if include_audio else clip_video_dur
            if clip_video_dur <= 0.02:
                errors.append("zero_video_duration")
                continue
            if include_audio and clip_audio_dur <= 0.02:
                mode = "accurate" if use_accurate_seek else "fast"
                errors.append(f"{mode}_seek_no_audio_stream")
                continue

            if bool(getattr(self, "segment_shortfall_pad", True)):
                base_dur = min(clip_video_dur, clip_audio_dur) if include_audio else clip_video_dur
                shortfall = max(0.0, float(duration_sec) - float(base_dur))
                if shortfall > max(0.06, float(duration_sec) * 0.015):
                    hard_fail_sec = max(0.0, float(getattr(self, "segment_shortfall_pad_hard_fail_sec", 0.90)))
                    hard_fail_ratio = max(0.0, float(getattr(self, "segment_shortfall_pad_hard_fail_ratio", 0.22)))
                    if shortfall > max(hard_fail_sec, float(duration_sec) * hard_fail_ratio):
                        errors.append(f"shortfall_too_large:{shortfall:.3f}s")
                        continue
                    ok_pad, pad_err = pad_shortfall_if_needed(attempt_tag=attempt_tag, shortfall=float(shortfall))
                    if not ok_pad:
                        # 轻微短缺允许继续；较大短缺强制进入下一抽取策略。
                        errors.append(f"pad_shortfall_failed:{pad_err[:200]}")
                        if shortfall > max(0.35, float(duration_sec) * 0.08):
                            continue

                    self._invalidate_media_duration_cache(output_clip)
                    clip_video_dur = float(self.get_video_duration(output_clip))
                    if include_audio:
                        clip_audio_dur = float(self.get_audio_duration(output_clip))
                        if clip_audio_dur <= 0.02:
                            errors.append("shortfall_pad_no_audio_stream")
                            continue

            return True, ""

        if not errors:
            return False, "extract_failed_unknown"
        return False, " | ".join(errors[:3])

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
        gop = max(12, int(round(float(source_fps))))
        encoder_args = [
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-bf', '0',
            '-g', str(gop),
            '-keyint_min', str(gop),
            '-sc_threshold', '0',
        ]
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
        pipe_broken = False
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
                try:
                    proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError:
                    pipe_broken = True
                    break
                prev_full_frame = frame
                frame_idx += 1
            if proc.stdin is not None:
                proc.stdin.close()
            ret = proc.wait()
        finally:
            cap.release()
            if proc.poll() is None:
                proc.kill()

        if pipe_broken or ret != 0 or not repaired_video_only.exists():
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

        # carryover run 在上游已经完成“整体前移”验证时，优先尊重那次通过的 shift_to；
        # 若后置 cleanup 又把起点拉回去，会在输出阶段造成片头缺字，但报告里仍留着原始 shift_to 证据。
        for seg in segments:
            q = seg.get("quality", {}) or {}
            if not bool(q.get("cross_source_prev_tail_carryover_shifted_no_target", False)):
                continue
            shifted_to = q.get("cross_source_prev_tail_carryover_shift_to")
            if shifted_to is None:
                continue
            shifted_to = float(shifted_to)
            curr_start = float(seg.get("start", 0.0) or 0.0)
            if abs(shifted_to - curr_start) <= 0.18:
                continue
            if not bool(q.get("post_lag_boundary_cleanup_no_target", False)):
                continue
            q["cross_source_prev_tail_carryover_shift_restored_no_target"] = True
            q["cross_source_prev_tail_carryover_shift_restored_from"] = float(curr_start)
            q["cross_source_prev_tail_carryover_shift_restored_to"] = float(shifted_to)
            seg["start"] = float(shifted_to)
            seg["quality"] = q

        # 若前一段是“前源尾巴 + 当前源片头”的拼桥段，且后一段又被进一步校正到了新的片头相位，
        # 则需要同步更新拼桥段里“当前源片头部分”的起点，否则两段之间会留下几百毫秒的真实内容空洞。
        for idx in range(len(segments) - 1):
            seg = segments[idx]
            nxt = segments[idx + 1]
            q = seg.get("quality", {}) or {}
            if not bool(q.get("cross_source_prev_tail_carryover_no_target", False)):
                continue
            curr_source = seg.get("source")
            if curr_source is None or curr_source != nxt.get("source"):
                continue
            curr_part_start = q.get("cross_source_prev_tail_carryover_curr_source_start")
            curr_part_dur = q.get("cross_source_prev_tail_carryover_curr_source_duration")
            if curr_part_start is None or curr_part_dur is None:
                continue
            curr_part_start = float(curr_part_start)
            curr_part_dur = float(curr_part_dur)
            next_start = float(nxt.get("start", 0.0))
            curr_part_end = float(curr_part_start + curr_part_dur)
            delta = float(next_start - curr_part_end)
            if abs(delta) <= 0.18:
                continue
            desired_curr_start = max(0.0, float(next_start - curr_part_dur))
            if abs(desired_curr_start - curr_part_start) <= 1e-6:
                continue
            q["cross_source_prev_tail_carryover_curr_source_start_synced_no_target"] = True
            q["cross_source_prev_tail_carryover_curr_source_start_from"] = float(curr_part_start)
            q["cross_source_prev_tail_carryover_curr_source_start_to"] = float(desired_curr_start)
            q["cross_source_prev_tail_carryover_curr_source_sync_delta"] = float(delta)
            q["cross_source_prev_tail_carryover_curr_source_sync_anchor_index"] = int(idx + 1)
            q["cross_source_prev_tail_carryover_curr_source_start"] = float(desired_curr_start)
            seg["quality"] = q

        # 若前一段在输出阶段被恢复到更早的 carryover shift_to，而后一段仍保留着旧的硬约束起点，
        # 最终会在两个同源段之间留下一个正向空洞，表现为“字幕还在，但后面几个字没读出来”。
        # 这里在正式抽段前，再把后一段同步回拉到与前段收口一致的位置。
        for idx in range(1, len(segments)):
            prev = segments[idx - 1]
            curr = segments[idx]
            if prev.get("source") != curr.get("source") or prev.get("source") == self.target_video:
                continue

            prev_q = prev.get("quality", {}) or {}
            curr_q = curr.get("quality", {}) or {}
            prev_from = prev_q.get("cross_source_prev_tail_carryover_shift_restored_from")
            prev_to = prev_q.get("cross_source_prev_tail_carryover_shift_restored_to")
            curr_restore_to = curr_q.get("cross_source_prev_tail_carryover_shift_restored_to")

            desired_candidates: List[float] = []
            if prev_from is not None and prev_to is not None:
                delta = float(prev_from) - float(prev_to)
                if delta > 0.18:
                    desired_candidates.append(float(curr.get("start", 0.0) or 0.0) - float(delta))
            if curr_restore_to is not None:
                desired_candidates.append(float(curr_restore_to))
            if not desired_candidates:
                continue

            curr_start = float(curr.get("start", 0.0) or 0.0)
            desired_start = max(0.0, min(float(x) for x in desired_candidates))
            if abs(desired_start - curr_start) <= 0.18:
                continue

            prev_end = float(prev.get("start", 0.0) or 0.0) + float(prev.get("duration", 0.0) or 0.0)
            before_gap = float(curr_start - prev_end)
            after_gap = float(desired_start - prev_end)
            if after_gap < -0.02:
                continue

            curr["start"] = float(desired_start)
            curr_q["carryover_restored_prev_sync_no_target"] = True
            curr_q["carryover_restored_prev_sync_from"] = float(curr_start)
            curr_q["carryover_restored_prev_sync_to"] = float(desired_start)
            curr_q["carryover_restored_prev_sync_before_gap"] = float(before_gap)
            curr_q["carryover_restored_prev_sync_after_gap"] = float(after_gap)
            curr["quality"] = curr_q

        # 某些尾段在上游已经写入了 carryover shift_to，但顶层 start 仍停留在旧值。
        # 这种情况最终不会表现为“重叠”，而是直接跳过一小段内容，常见听感就是句尾两三个字丢失。
        # 这里仅在“前一段同源收口正好落在 shift_to 附近”时，才把当前段同步到该 shift_to，
        # 避免把普通的中间修复误当成输出阶段同步问题。
        for idx in range(1, len(segments)):
            prev = segments[idx - 1]
            curr = segments[idx]
            if prev.get("source") != curr.get("source") or prev.get("source") == self.target_video:
                continue

            curr_q = curr.get("quality", {}) or {}
            if not bool(curr_q.get("cross_source_prev_tail_carryover_shifted_no_target", False)):
                continue

            shifted_to = curr_q.get("cross_source_prev_tail_carryover_shift_to")
            if shifted_to is None:
                continue

            curr_start = float(curr.get("start", 0.0) or 0.0)
            desired_start = max(0.0, float(shifted_to))
            if abs(desired_start - curr_start) <= 0.18:
                continue

            prev_end = float(prev.get("start", 0.0) or 0.0) + float(prev.get("duration", 0.0) or 0.0)
            before_gap = float(curr_start - prev_end)
            after_gap = float(desired_start - prev_end)

            align_to_prev_end = abs(desired_start - prev_end) <= 0.05
            positive_hole = before_gap > 0.18
            carryover_shifted = bool(curr_q.get("cross_source_prev_tail_carryover_shifted_no_target", False))
            recovered_tail = bool(
                curr_q.get("recovered_from_neighbors", False)
                or curr_q.get("tail_shortfall_overlap_trimmed_prev_no_target", False)
                or curr_q.get("boundary_hard_skipped_large_shift_no_target", False)
            )
            # 这里不再要求一定是“邻段恢复段”。
            # 只要当前段本身已经记录了 carryover shift_to，且该 shift_to 正好与前一段收口对齐，
            # 顶层 start 却还滞后一个明显正向空洞，就说明输出参数还停在旧值。
            if not (align_to_prev_end and positive_hole and (recovered_tail or carryover_shifted)):
                continue
            if after_gap < -0.02:
                continue

            curr["start"] = float(desired_start)
            curr_q["carryover_shift_to_output_synced_no_target"] = True
            curr_q["carryover_shift_to_output_synced_from"] = float(curr_start)
            curr_q["carryover_shift_to_output_synced_to"] = float(desired_start)
            curr_q["carryover_shift_to_output_synced_before_gap"] = float(before_gap)
            curr_q["carryover_shift_to_output_synced_after_gap"] = float(after_gap)
            curr["quality"] = curr_q

        def infer_composite_parts(seg: dict) -> Optional[List[Dict[str, object]]]:
            parts = seg.get("composite_parts")
            if isinstance(parts, list) and parts:
                return parts

            q = seg.get("quality", {}) or {}
            if not bool(q.get("cross_source_prev_tail_carryover_no_target", False)):
                return None

            prev_source = q.get("cross_source_prev_tail_carryover_prev_source")
            prev_start = q.get("cross_source_prev_tail_carryover_prev_tail_start")
            prev_dur = q.get("cross_source_prev_tail_carryover_prev_tail_duration")
            curr_source = seg.get("source")
            curr_start = q.get("cross_source_prev_tail_carryover_curr_source_start", seg.get("start"))
            curr_dur = q.get("cross_source_prev_tail_carryover_curr_source_duration")
            if not prev_source or prev_start is None or prev_dur is None or curr_source is None or curr_start is None or curr_dur is None:
                return None

            carry_prev = max(0.0, float(prev_dur))
            carry_curr = max(0.0, float(curr_dur))
            if carry_prev <= 1e-6 or carry_curr <= 1e-6:
                return None

            total_duration = float(seg.get("duration", 0.0) or 0.0)
            if total_duration > 1e-6:
                # 避免报告回放时因浮点误差把拼桥总时长撑大/缩小。
                scale = float(total_duration) / float(carry_prev + carry_curr)
                if abs(scale - 1.0) > 1e-3:
                    carry_prev *= scale
                    carry_curr *= scale

            return [
                {
                    "source": Path(str(prev_source)),
                    "start": float(prev_start),
                    "duration": float(carry_prev),
                },
                {
                    "source": Path(str(curr_source)),
                    "start": float(curr_start),
                    "duration": float(carry_curr),
                },
            ]

        def extract_one_segment(seg: dict) -> Dict[str, object]:
            seg_source = seg['source']
            seg_start = seg['start']
            seg_expected_frames = max(1, int(round(float(seg['duration']) * output_fps)))
            seg_ext = "mkv" if ((not render_target_audio_only) and bool(getattr(self, "segment_intermediate_pcm_audio", False))) else "mp4"
            av_clip = self.temp_dir / f"seg_{seg['index']:03d}_av.{seg_ext}"
            seg_extract_perf = time.perf_counter()

            composite_parts = infer_composite_parts(seg)
            if isinstance(composite_parts, list) and composite_parts:
                ok, err = self._extract_composite_av_clip(
                    composite_parts,
                    av_clip,
                    fps_expr,
                    total_duration=float(seg['duration']),
                    expected_frames=seg_expected_frames,
                    include_audio=(not render_target_audio_only),
                )
            else:
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
            if not ok and seg_source != self.target_video and self.enable_target_video_fallback:
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
        # 经验阈值：当大量段已兜底到目标素材时，copy 拼接更容易在个别播放器出现边界卡顿观感。
        concat_reencode_fallback_ratio = 0.85
        fallback_count = sum(
            1
            for seg in segments
            if bool((seg.get("quality", {}) or {}).get("fallback", False))
        )
        fallback_ratio = (float(fallback_count) / float(len(segments))) if segments else 0.0

        def _run_concat_reencode(out_path: Path) -> Tuple[bool, str]:
            reencode_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'concat', '-safe', '0',
                '-i', str(av_concat),
                '-t', f"{target_duration:.3f}",
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-bf', '0',
                '-g', str(max(12, int(round(output_fps)))),
                '-keyint_min', str(max(12, int(round(output_fps)))),
                '-sc_threshold', '0',
                '-r', fps_out,
                '-vsync', 'cfr',
                '-movflags', '+faststart',
            ]
            if render_target_audio_only:
                reencode_cmd.extend(['-an'])
            else:
                audio_advance = max(0.0, float(getattr(self, "audio_sync_advance_sec", 0.0)))
                if audio_advance > 1e-6:
                    af_expr = f"asetpts=PTS-{audio_advance:.6f}/TB,aresample=async=1:first_pts=0"
                else:
                    af_expr = 'aresample=async=1:first_pts=0'
                reencode_cmd.extend([
                    '-af', af_expr,
                    '-ar', '48000',
                    '-ac', '2',
                    '-c:a', 'aac', '-b:a', '128k',
                ])
            reencode_cmd.append(str(out_path))
            reencode_proc = subprocess.run(reencode_cmd, capture_output=True, text=True)
            if reencode_proc.returncode == 0 and out_path.exists():
                return True, ""
            return False, (reencode_proc.stderr or "").strip()

        concat_error = ""
        if not self.enable_target_video_fallback:
            # 严格禁兜底模式下优先稳定性：始终重编码拼接，避免 copy 拼接时间戳抖动导致掉音/循环。
            ok, concat_error = _run_concat_reencode(temp_output)
            if ok:
                concat_mode = "reencode_no_target_fallback"
            else:
                concat_mode = ""
        elif fallback_ratio >= concat_reencode_fallback_ratio:
            # 高兜底场景直接使用稳定重编码拼接，优先播放兼容性。
            ok, concat_error = _run_concat_reencode(temp_output)
            if ok:
                concat_mode = "reencode_high_fallback"
            else:
                concat_mode = ""
        else:
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
                # copy 虽成功，但若音轨长度异常偏短，通常意味着时间戳/流参数不可靠，
                # 这时切到重编码拼接可显著降低播放器边界卡顿观感。
                quick_audio = self.get_audio_duration(temp_output)
                min_audio_required_quick = max(5.0, float(target_duration) * 0.90)
                if quick_audio <= 0.0 or (quick_audio + 0.5) < min_audio_required_quick:
                    stable_output = self.temp_dir / "temp_output_reencode_stable.mp4"
                    ok, concat_error = _run_concat_reencode(stable_output)
                    if ok:
                        temp_output = stable_output
                        concat_mode = "reencode_after_copy_short_audio"
                    else:
                        concat_mode = "copy"
                else:
                    concat_mode = "copy"
            else:
                ok, concat_error = _run_concat_reencode(temp_output)
                if ok:
                    concat_mode = "reencode_after_copy_fail"
                else:
                    concat_mode = ""
        concat_elapsed = time.perf_counter() - concat_perf

        if not concat_mode:
            self.last_render_metrics = {
                "status": "failed",
                "error": "concat_failed",
                "error_detail": concat_error[:240],
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
        auto_target_audio_fallback = bool(
            self.enable_target_video_fallback
            and (
                stitched_audio_duration <= 0.0
                or (stitched_audio_duration + 0.5) < min_audio_required
            )
        )
        if auto_target_audio_fallback and not self.force_target_audio:
            print(
                f"   ⚠️ 拼接音轨偏短: {stitched_audio_duration:.2f}s/{target_duration:.2f}s，"
                "自动切换目标音轨封装"
            )

        if (not self.enable_target_video_fallback) and (
            stitched_audio_duration <= 0.0
            or (stitched_audio_duration + 0.5) < min_audio_required
        ):
            self.last_render_metrics = {
                "status": "failed",
                "error": "stitched_audio_too_short_without_target_fallback",
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
            print(
                f"❌ 拼接音轨偏短: {stitched_audio_duration:.2f}s/{target_duration:.2f}s，"
                "且已禁用目标素材兜底，终止输出"
            )
            return False

        use_target_audio_mux = bool(
            self.enable_target_video_fallback
            and (self.force_target_audio or auto_target_audio_fallback)
        )
        output_path_obj = Path(output_path)
        # 先删除旧输出，再写同名新文件，避免 Finder 继续显示旧创建时间。
        output_path_obj.unlink(missing_ok=True)

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
                shutil.copy(muxed_output, output_path_obj)
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
                shutil.copy(temp_output, output_path_obj)
                final_audio_duration = stitched_audio_duration
                print("   ⚠️ 目标音轨封装失败，回退为原拼接音轨")
        else:
            shutil.copy(temp_output, output_path_obj)
            final_audio_duration = stitched_audio_duration
            print("   🔊 保留拼接片段原音轨（未使用目标音轨封装）")

        if output_path_obj.exists():
            final_duration = self.get_video_duration(output_path_obj)
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

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", help="配置文件路径（JSON）")
    pre_args, _ = pre_parser.parse_known_args()

    try:
        cfg, cfg_path = load_section_config(root, "v6_fast", explicit_path=pre_args.config)
        cache_default = cfg_req_str(cfg, "cache")
        match_threshold_default = cfg_req_float(cfg, "match_threshold")
        segment_duration_default = cfg_req_float(cfg, "segment_duration")
        frame_index_sample_interval_default = cfg_req_float(cfg, "frame_index_sample_interval")
        workers_default = cfg_req_int(cfg, "workers")
        render_workers_default = cfg_req_int(cfg, "render_workers")
        low_score_threshold_default = cfg_req_float(cfg, "low_score_threshold")
        rematch_window_default = cfg_req_int(cfg, "rematch_window")
        rematch_max_window_default = cfg_req_int(cfg, "rematch_max_window")
        continuity_weight_default = cfg_req_float(cfg, "continuity_weight")
        strict_visual_verify_default = cfg_req_bool(cfg, "strict_visual_verify")
        strict_verify_min_sim_default = cfg_req_float(cfg, "strict_verify_min_sim")
        tail_guard_seconds_default = cfg_req_float(cfg, "tail_guard_seconds")
        tail_verify_min_avg_default = cfg_req_float(cfg, "tail_verify_min_avg")
        tail_verify_min_floor_default = cfg_req_float(cfg, "tail_verify_min_floor")
        adjacent_overlap_trigger_default = cfg_req_float(cfg, "adjacent_overlap_trigger")
        adjacent_lag_trigger_default = cfg_req_float(cfg, "adjacent_lag_trigger")
        isolated_drift_trigger_default = cfg_req_float(cfg, "isolated_drift_trigger")
        cross_source_mapping_jump_trigger_default = cfg_req_float(cfg, "cross_source_mapping_jump_trigger")
        max_mapping_drift_trigger_default = cfg_req_float(cfg, "max_mapping_drift_trigger")
        max_mapping_drift_combined_floor_default = cfg_req_float(cfg, "max_mapping_drift_combined_floor")
        use_audio_matching_default = cfg_req_bool(cfg, "use_audio_matching")
        force_target_audio_default = cfg_req_bool(cfg, "force_target_audio")
        verify_interval_default = cfg_req_float(cfg, "verify_interval")
        verify_clip_duration_default = cfg_req_float(cfg, "verify_clip_duration")
        verify_max_points_default = cfg_req_int(cfg, "verify_max_points")
        verify_asr_mode_default = cfg_req_str(cfg, "verify_asr_mode")
        verify_target_sub_default = cfg_req_str(cfg, "verify_target_sub")
        verify_output_sub_default = cfg_req_str(cfg, "verify_output_sub")
        verify_asr_cmd_default = cfg_req_str(cfg, "verify_asr_cmd")
        verify_asr_python_default = cfg_req_str(cfg, "verify_asr_python")
        verify_asr_model_default = cfg_req_str(cfg, "verify_asr_model")
        verify_language_default = cfg_req_str(cfg, "verify_language")
        verify_whisper_candidates_default = cfg_req_str_list(cfg, "verify_whisper_python_candidates")
        verify_output_root_default = cfg_req_str(cfg, "verify_output_root")
        run_evidence_validation_default = cfg_req_bool(cfg, "run_evidence_validation")
        run_ai_verify_snapshots_default = cfg_req_bool(cfg, "run_ai_verify_snapshots")
        boundary_glitch_fix_default = cfg_req_bool(cfg, "boundary_glitch_fix")
        boundary_glitch_hi_threshold_default = cfg_req_float(cfg, "boundary_glitch_hi_threshold")
        boundary_glitch_lo_threshold_default = cfg_req_float(cfg, "boundary_glitch_lo_threshold")
        boundary_glitch_gap_threshold_default = cfg_req_float(cfg, "boundary_glitch_gap_threshold")
        audio_guard_enabled_default = cfg_req_bool(cfg, "audio_guard_enabled")
        audio_guard_score_trigger_default = cfg_req_float(cfg, "audio_guard_score_trigger")
        audio_guard_sample_duration_default = cfg_req_float(cfg, "audio_guard_sample_duration")
        audio_guard_min_similarity_default = cfg_req_float(cfg, "audio_guard_min_similarity")
        audio_guard_hard_floor_default = cfg_req_float(cfg, "audio_guard_hard_floor")
        audio_guard_shift_margin_default = cfg_req_float(cfg, "audio_guard_shift_margin")
        audio_segment_accurate_seek_default = cfg_req_bool(cfg, "audio_segment_accurate_seek")
        segment_shortfall_pad_default = cfg_req_bool(cfg, "segment_shortfall_pad")
        segment_intermediate_pcm_audio_default = cfg_req_bool(cfg, "segment_intermediate_pcm_audio")
        audio_guard_shift_candidates_raw = cfg_req_str_list(
            cfg,
            "audio_guard_shift_candidates",
        )
        try:
            audio_guard_shift_candidates_default = [float(x) for x in audio_guard_shift_candidates_raw if str(x).strip()]
        except Exception as exc:
            raise RuntimeError(
                f"配置项 audio_guard_shift_candidates 需要 float 列表，当前值: {audio_guard_shift_candidates_raw!r}"
            ) from exc
        if not audio_guard_shift_candidates_default:
            raise RuntimeError("配置项 audio_guard_shift_candidates 不能为空")
        boundary_glitch_use_videotoolbox_default = cfg_req_bool(cfg, "boundary_glitch_use_videotoolbox")
        bridge_motion_guard_enabled_default = cfg_req_bool(cfg, "bridge_motion_guard_enabled")
        bridge_motion_min_target_motion_default = cfg_req_float(cfg, "bridge_motion_min_target_motion")
        bridge_motion_min_ratio_default = cfg_req_float(cfg, "bridge_motion_min_ratio")
        bridge_motion_samples_default = cfg_req_int(cfg, "bridge_motion_samples")
        bridge_recover_min_target_start_default = cfg_req_float(cfg, "bridge_recover_min_target_start")
        source_tail_safety_enabled_default = cfg_req_bool(cfg, "source_tail_safety_enabled")
        source_tail_safety_margin_default = cfg_req_float(cfg, "source_tail_safety_margin")
        source_tail_safety_target_tail_ignore_sec_default = cfg_req_float(cfg, "source_tail_safety_target_tail_ignore_sec")
        source_tail_safety_switch_min_gain_default = cfg_req_float(cfg, "source_tail_safety_switch_min_gain")
        cross_source_head_nudge_enabled_default = cfg_req_bool(cfg, "cross_source_head_nudge_enabled")
        cross_source_head_nudge_prev_tail_window_default = cfg_req_float(cfg, "cross_source_head_nudge_prev_tail_window")
        cross_source_head_nudge_curr_head_window_default = cfg_req_float(cfg, "cross_source_head_nudge_curr_head_window")
        cross_source_head_nudge_max_offset_default = cfg_req_float(cfg, "cross_source_head_nudge_max_offset")
        cross_source_head_nudge_score_bias_default = cfg_req_float(cfg, "cross_source_head_nudge_score_bias")
        cross_source_head_nudge_max_verify_drop_default = cfg_req_float(cfg, "cross_source_head_nudge_max_verify_drop")
        cross_source_head_nudge_allow_backward_default = cfg_req_bool(cfg, "cross_source_head_nudge_allow_backward")
        cross_source_head_nudge_backward_cap_default = cfg_req_float(cfg, "cross_source_head_nudge_backward_cap")
        cross_source_head_nudge_backward_gain_trigger_default = cfg_req_float(cfg, "cross_source_head_nudge_backward_gain_trigger")
        cross_source_head_nudge_boundary_audio_weight_default = cfg_req_float(cfg, "cross_source_head_nudge_boundary_audio_weight")
        no_target_backprop_overlap_fix_default = cfg_req_bool(cfg, "no_target_backprop_overlap_fix")
        no_target_backprop_max_shift_default = cfg_req_float(cfg, "no_target_backprop_max_shift")
        no_target_backprop_min_quality_default = cfg_req_float(cfg, "no_target_backprop_min_quality")
        no_target_backprop_neg_trigger_floor_default = cfg_req_float(cfg, "no_target_backprop_neg_trigger_floor")
        no_target_tail_shortfall_tolerance_sec_default = cfg_req_float(cfg, "no_target_tail_shortfall_tolerance_sec")
        no_target_missing_tail_tolerance_sec_default = cfg_req_float(cfg, "no_target_missing_tail_tolerance_sec")
        no_target_boundary_rematch_enabled_default = cfg_req_bool(cfg, "no_target_boundary_rematch_enabled")
        no_target_boundary_rematch_max_attempts_default = cfg_req_int(cfg, "no_target_boundary_rematch_max_attempts")
        boundary_hard_max_shift_sec_default = cfg_req_float(cfg, "boundary_hard_max_shift_sec")
        audio_fp_cache_max_items_default = cfg_req_int(cfg, "audio_fp_cache_max_items")
        frame_cache_max_items_default = cfg_req_int(cfg, "frame_cache_max_items")
        frame_feature_cache_max_items_default = cfg_req_int(cfg, "frame_feature_cache_max_items")
        allow_numeric_fallback_default = cfg_req_bool(cfg, "allow_numeric_fallback")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    parser = argparse.ArgumentParser(description="通用极速高精度重构器 V3 + pHash")
    parser.add_argument("--config", default=str(cfg_path) if cfg_path else "", help="配置文件路径（JSON）")
    parser.add_argument("--target", required=True, help="目标视频路径")
    parser.add_argument("--source-dir", required=True, help="源视频目录")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--frame-index-cache-dir", help="项目级共享帧索引缓存目录（可选）")
    parser.add_argument("--cache", default=cache_default, help="缓存目录")
    parser.add_argument("--segment-duration", type=float, default=segment_duration_default, help="分段时长（秒）")
    parser.add_argument("--frame-index-sample-interval", type=float, default=frame_index_sample_interval_default, help="pHash 帧索引采样间隔（秒）")
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
    add_bool_arg(parser, "--audio-segment-accurate-seek", audio_segment_accurate_seek_default, "带音轨分段提取使用精确寻址（降低词尾重复/音画错位）")
    add_bool_arg(parser, "--segment-shortfall-pad", segment_shortfall_pad_default, "分段素材不足时补齐末帧与静音，避免局部掉音/黑帧")
    add_bool_arg(parser, "--segment-intermediate-pcm-audio", segment_intermediate_pcm_audio_default, "中间分段使用 PCM 音轨（减少边界瞬断/吞字）")
    add_bool_arg(parser, "--source-tail-safety-enabled", source_tail_safety_enabled_default, "启用源片尾安全边距（避开片尾高风险黑场/尾音异常）")
    parser.add_argument("--source-tail-safety-margin", type=float, default=source_tail_safety_margin_default, help="源片尾安全边距（秒）")
    parser.add_argument("--source-tail-safety-target-tail-ignore-sec", type=float, default=source_tail_safety_target_tail_ignore_sec_default, help="目标视频尾段忽略源片尾安全边距检查的范围（秒）")
    parser.add_argument("--source-tail-safety-switch-min-gain", type=float, default=source_tail_safety_switch_min_gain_default, help="源片尾安全切换最小收益阈值（verify_avg 提升量）")
    add_bool_arg(parser, "--cross-source-head-nudge-enabled", cross_source_head_nudge_enabled_default, "启用跨源边界头部微调（减少跨源接缝重复词/黑一下）")
    parser.add_argument("--cross-source-head-nudge-prev-tail-window", type=float, default=cross_source_head_nudge_prev_tail_window_default, help="跨源微调触发：前段接近源片尾阈值（秒）")
    parser.add_argument("--cross-source-head-nudge-curr-head-window", type=float, default=cross_source_head_nudge_curr_head_window_default, help="跨源微调触发：后段接近源片头阈值（秒）")
    parser.add_argument("--cross-source-head-nudge-max-offset", type=float, default=cross_source_head_nudge_max_offset_default, help="跨源边界后段最大前移量（秒）")
    parser.add_argument("--cross-source-head-nudge-score-bias", type=float, default=cross_source_head_nudge_score_bias_default, help="跨源微调评分偏置（越大越倾向更大前移）")
    parser.add_argument("--cross-source-head-nudge-max-verify-drop", type=float, default=cross_source_head_nudge_max_verify_drop_default, help="跨源微调允许的画面核验下降上限")
    add_bool_arg(parser, "--cross-source-head-nudge-allow-backward", cross_source_head_nudge_allow_backward_default, "跨源片头微调允许向前回拉（贴近句首）")
    parser.add_argument("--cross-source-head-nudge-backward-cap", type=float, default=cross_source_head_nudge_backward_cap_default, help="跨源片头回拉最大幅度（秒）")
    parser.add_argument("--cross-source-head-nudge-backward-gain-trigger", type=float, default=cross_source_head_nudge_backward_gain_trigger_default, help="跨源片头回拉最小收益阈值")
    parser.add_argument("--cross-source-head-nudge-boundary-audio-weight", type=float, default=cross_source_head_nudge_boundary_audio_weight_default, help="跨源片头微调时边界音频相似度权重")
    add_bool_arg(parser, "--backprop-overlap-fix-no-target", no_target_backprop_overlap_fix_default, "禁兜底模式下启用同源尾段重叠反向回推修复")
    parser.add_argument("--no-target-backprop-max-shift", type=float, default=no_target_backprop_max_shift_default, help="禁兜底尾段重叠回推最大允许修正量（秒）")
    parser.add_argument("--no-target-backprop-min-quality", type=float, default=no_target_backprop_min_quality_default, help="禁兜底尾段回推最小置信门限（combined）")
    parser.add_argument("--no-target-backprop-neg-trigger-floor", type=float, default=no_target_backprop_neg_trigger_floor_default, help="禁兜底尾段回推负重叠触发下限（秒）")
    parser.add_argument("--no-target-tail-shortfall-tolerance-sec", type=float, default=no_target_tail_shortfall_tolerance_sec_default, help="禁兜底尾段允许短缺时长（秒，优先裁掉尾部重复）")
    parser.add_argument("--no-target-missing-tail-tolerance-sec", type=float, default=no_target_missing_tail_tolerance_sec_default, help="禁兜底尾部缺段允许短缺时长（秒，尾部缺失时允许直接输出略短成片）")
    add_bool_arg(parser, "--boundary-rematch-no-target", no_target_boundary_rematch_enabled_default, "禁兜底模式下对未收敛边界启用定点重匹配")
    parser.add_argument("--no-target-boundary-rematch-max-attempts", type=int, default=no_target_boundary_rematch_max_attempts_default, help="禁兜底未收敛边界定点重匹配最大尝试次数")
    parser.add_argument("--boundary-hard-max-shift-sec", type=float, default=boundary_hard_max_shift_sec_default, help="边界硬约束单次最大位移（秒，超过则跳过硬夹紧）")
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

    source_videos = [str(f) for f in sorted(source_dir.iterdir()) if f.suffix.lower() == '.mp4']
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

    reconstructor = FastHighPrecisionReconstructor(
        str(target),
        source_videos,
        str(cache),
        args.frame_index_cache_dir,
    )
    if reconstructor.secondary_source_videos:
        secondary_names = ", ".join(src.name for src in reconstructor.secondary_source_videos)
        print(
            f"🧹 源池分级: 主源 {len(reconstructor.primary_source_videos)} 个, "
            f"次级救援源 {len(reconstructor.secondary_source_videos)} 个 ({secondary_names})"
        )
    reconstructor.match_threshold = min(1.0, max(0.0, float(match_threshold_default)))
    reconstructor.segment_duration = max(1.0, args.segment_duration)
    reconstructor.frame_index_sample_interval = max(0.1, float(args.frame_index_sample_interval))
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
    reconstructor.audio_segment_accurate_seek = bool(args.audio_segment_accurate_seek)
    reconstructor.segment_shortfall_pad = bool(args.segment_shortfall_pad)
    reconstructor.segment_intermediate_pcm_audio = bool(args.segment_intermediate_pcm_audio)
    reconstructor.source_tail_safety_enabled = bool(args.source_tail_safety_enabled)
    reconstructor.source_tail_safety_margin = max(0.0, float(args.source_tail_safety_margin))
    reconstructor.source_tail_safety_target_tail_ignore_sec = max(0.0, float(args.source_tail_safety_target_tail_ignore_sec))
    reconstructor.source_tail_safety_switch_min_gain = max(0.0, float(args.source_tail_safety_switch_min_gain))
    reconstructor.cross_source_head_nudge_enabled = bool(args.cross_source_head_nudge_enabled)
    reconstructor.cross_source_head_nudge_prev_tail_window = max(0.0, float(args.cross_source_head_nudge_prev_tail_window))
    reconstructor.cross_source_head_nudge_curr_head_window = max(0.0, float(args.cross_source_head_nudge_curr_head_window))
    reconstructor.cross_source_head_nudge_max_offset = max(0.0, float(args.cross_source_head_nudge_max_offset))
    reconstructor.cross_source_head_nudge_score_bias = float(args.cross_source_head_nudge_score_bias)
    reconstructor.cross_source_head_nudge_max_verify_drop = max(0.0, float(args.cross_source_head_nudge_max_verify_drop))
    reconstructor.cross_source_head_nudge_allow_backward = bool(args.cross_source_head_nudge_allow_backward)
    reconstructor.cross_source_head_nudge_backward_cap = max(0.0, float(args.cross_source_head_nudge_backward_cap))
    reconstructor.cross_source_head_nudge_backward_gain_trigger = max(0.0, float(args.cross_source_head_nudge_backward_gain_trigger))
    reconstructor.cross_source_head_nudge_boundary_audio_weight = max(0.0, float(args.cross_source_head_nudge_boundary_audio_weight))
    reconstructor.no_target_backprop_overlap_fix = bool(args.backprop_overlap_fix_no_target)
    reconstructor.no_target_backprop_max_shift = max(0.0, float(args.no_target_backprop_max_shift))
    reconstructor.no_target_backprop_min_quality = min(1.0, max(0.0, float(args.no_target_backprop_min_quality)))
    reconstructor.no_target_backprop_neg_trigger_floor = max(0.0, float(args.no_target_backprop_neg_trigger_floor))
    reconstructor.no_target_tail_shortfall_tolerance_sec = max(0.0, float(args.no_target_tail_shortfall_tolerance_sec))
    reconstructor.no_target_missing_tail_tolerance_sec = max(0.0, float(args.no_target_missing_tail_tolerance_sec))
    reconstructor.no_target_boundary_rematch_enabled = bool(args.boundary_rematch_no_target)
    reconstructor.no_target_boundary_rematch_max_attempts = max(0, int(args.no_target_boundary_rematch_max_attempts))
    reconstructor.boundary_hard_max_shift_sec = max(0.0, float(args.boundary_hard_max_shift_sec))
    reconstructor.audio_guard_shift_candidates = [float(x) for x in audio_guard_shift_candidates_default]
    reconstructor.boundary_glitch_use_videotoolbox = bool(boundary_glitch_use_videotoolbox_default)
    reconstructor.bridge_motion_guard_enabled = bool(bridge_motion_guard_enabled_default)
    reconstructor.bridge_motion_min_target_motion = max(0.0, float(bridge_motion_min_target_motion_default))
    reconstructor.bridge_motion_min_ratio = min(1.0, max(0.0, float(bridge_motion_min_ratio_default)))
    reconstructor.bridge_motion_samples = max(1, int(bridge_motion_samples_default))
    reconstructor.bridge_recover_min_target_start = max(0.0, float(bridge_recover_min_target_start_default))
    reconstructor.audio_fp_cache_max_items = max(128, int(audio_fp_cache_max_items_default))
    reconstructor.frame_cache_max_items = max(128, int(frame_cache_max_items_default))
    reconstructor.frame_feature_cache_max_items = max(128, int(frame_feature_cache_max_items_default))
    reconstructor.use_audio_matching = bool(args.use_audio_matching)
    reconstructor.force_target_audio = bool(args.force_target_audio)
    reconstructor.run_ai_verify_snapshots = bool(args.run_ai_verify_snapshots)
    verify_whisper_candidates = split_csv(args.verify_whisper_candidates)

    exit_code = 1
    try:
        success = reconstructor.reconstruct_fast(str(output))

        if success:
            exit_code = 0
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
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
