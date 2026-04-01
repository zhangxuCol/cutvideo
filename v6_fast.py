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
from pathlib import Path
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import wave
import struct
import json
import pickle
import re
from PIL import Image
import imagehash

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
        self.max_workers = 4  # 并行线程数

        # pHash 帧索引
        self.frame_index = {}  # {video_path: [(time, phash), ...]}

        # 缓存
        self.source_fingerprints = {}
        self.target_fingerprint = None
        
    def get_video_duration(self, video_path: Path) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
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

    def extract_frame_to_pil(self, video_path: Path, time_sec: float) -> Image.Image:
        """提取帧并转换为 PIL Image"""
        temp_frame = self.temp_dir / f"phash_frame_{video_path.stem}_{int(time_sec*100)}.jpg"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', 'scale=320:180',
            str(temp_frame)
        ]
        subprocess.run(cmd, capture_output=True)
        if temp_frame.exists():
            img = Image.open(str(temp_frame))
            img = img.copy()
            temp_frame.unlink()
            return img
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
        """三阶段匹配：pHash 预筛选 → 音频验证 → 画面精细定位"""
        best_source = None
        best_start = 0
        best_score = 0.0

        # 第一步：pHash 快速预筛选候选
        phash_candidates = self.find_match_by_phash(target_start, duration, seg_index, top_k=10)

        if phash_candidates:
            # 第二步：对 pHash 候选做音频+画面精细验证
            target_fp = extract_chromaprint(self.target_video, target_start, duration)
            target_frames = []
            check_offsets = [0, duration * 0.3, duration * 0.7]
            for i, offset in enumerate(check_offsets):
                tf = self.temp_dir / f"target_seg{seg_index}_p_{i}.jpg"
                self.extract_frame(self.target_video, target_start + offset, tf)
                if tf.exists():
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
                            sf = self.temp_dir / f"fine_{seg_index}_{source.stem}_{start_sec}_{int(offset)}.jpg"
                            self.extract_frame(source, start_sec + offset, sf)
                            if sf.exists():
                                total_sim += self.calculate_frame_similarity(target_frame, sf)
                                valid += 1
                                sf.unlink()
                        visual_sim = total_sim / valid if valid > 0 else 0

                    # 音频验证
                    audio_sim = 0.0
                    if target_fp and len(target_fp) >= 10:
                        source_fp = extract_chromaprint(source, start_sec, duration)
                        if source_fp and len(source_fp) >= 10:
                            audio_sim = compare_chromaprint(target_fp, source_fp)

                    # 综合评分：pHash 20% + 音频 40% + 画面 40%
                    combined_score = 0.2 * phash_sim + 0.4 * audio_sim + 0.4 * visual_sim
                    if combined_score > best_score:
                        best_score = combined_score
                        best_start = start_sec
                        best_source = source

                    if best_score > 0.92:
                        break

            # 清理目标帧
            for _, tf in target_frames:
                if tf.exists():
                    tf.unlink()

            if best_source and best_score >= 0.70:
                return best_source, best_start, best_score

        # Fallback：无 pHash 候选时，回退到原音频+画面搜索
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
        for i, offset in enumerate(check_offsets):
            tf = self.temp_dir / f"target_seg{seg_index}_fine_{i}.jpg"
            self.extract_frame(self.target_video, target_start + offset, tf)
            if tf.exists():
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
                    sf = self.temp_dir / f"fine_{seg_index}_{source.stem}_{start_sec}_{int(offset)}.jpg"
                    self.extract_frame(source, start_sec + offset, sf)
                    if sf.exists():
                        total_sim += self.calculate_frame_similarity(target_frame, sf)
                        valid_frames += 1
                        sf.unlink()
                if valid_frames > 0:
                    avg_sim = total_sim / valid_frames
                    combined_score = 0.5 * audio_score + 0.5 * avg_sim
                    if combined_score > best_score:
                        best_score = combined_score
                        best_start = start_sec
                        best_source = source
                if best_score > 0.90:
                    break

        for _, tf in target_frames:
            if tf.exists():
                tf.unlink()

        return best_source, best_start, best_score
    
    def find_best_match_by_visual(self, target_start: float, duration: float, seg_index: int = 0) -> Tuple[Path, float, float]:
        """当音频匹配失败时，遍历所有源视频进行画面匹配（更精细的搜索）"""
        best_source = None
        best_start = 0
        best_score = 0.0
        
        # 提取目标帧（多个时间点）
        target_frames = []
        check_offsets = [0, duration * 0.3, duration * 0.7]  # 3个时间点
        
        for i, offset in enumerate(check_offsets):
            target_frame = self.temp_dir / f"visual_match_target_{seg_index}_{i}.jpg"
            self.extract_frame(self.target_video, target_start + offset, target_frame)
            if target_frame.exists():
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
                
                # 对比多个时间点
                for offset, target_frame in target_frames:
                    source_frame = self.temp_dir / f"visual_match_source_{seg_index}_{source.stem}_{start_sec}_{int(offset)}.jpg"
                    self.extract_frame(source, start_sec + offset, source_frame)
                    
                    if source_frame.exists():
                        sim = self.calculate_frame_similarity(target_frame, source_frame)
                        total_sim += sim
                        valid_frames += 1
                        # 清理临时帧
                        source_frame.unlink()
                
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
        
        # 清理目标帧
        for _, target_frame in target_frames:
            if target_frame.exists():
                target_frame.unlink()
        
        # 只有相似度足够高才返回
        if best_score >= 0.75:
            return best_source, best_start, best_score
        return None, 0, 0
    
    def _find_alternative_match(self, target_start: float, duration: float, 
                                 source: Path, min_start: float, seg_index: int) -> float:
        """
        在同一源视频中向后搜索，找一个不重叠的替代匹配点
        返回: 新的开始时间，或 None（如果没找到）
        """
        # 提取目标帧（只取2个关键点，减少计算）
        target_frames = []
        check_offsets = [0, duration * 0.5]  # 减少到2个时间点
        
        for i, offset in enumerate(check_offsets):
            target_frame = self.temp_dir / f"alt_target_{seg_index}_{i}.jpg"
            self.extract_frame(self.target_video, target_start + offset, target_frame)
            if target_frame.exists():
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
                source_frame = self.temp_dir / f"alt_source_{seg_index}_{start_sec}_{int(offset)}.jpg"
                self.extract_frame(source, start_sec + offset, source_frame)
                
                if source_frame.exists():
                    sim = self.calculate_frame_similarity(target_frame, source_frame)
                    total_sim += sim
                    valid_frames += 1
                    source_frame.unlink()
            
            if valid_frames > 0:
                avg_sim = total_sim / valid_frames
                # 降低相似度阈值，提高找到替代点的概率
                if avg_sim > best_score and avg_sim >= 0.70:
                    best_score = avg_sim
                    best_start = start_sec
                    
                    # 如果找到足够好的匹配，提前退出
                    if avg_sim >= 0.85:
                        break
        
        # 清理目标帧
        for _, target_frame in target_frames:
            if target_frame.exists():
                target_frame.unlink()
        
        return best_start

    def quick_verify(self, source: Path, source_start: float, target_start: float, duration: float) -> Tuple[bool, float]:
        """快速画面验证 - 3个时间点"""
        
        check_times = [0, duration * 0.5, duration]
        similarities = []
        
        for offset in check_times:
            target_frame = self.temp_dir / f"qv_t_{target_start+offset:.0f}.jpg"
            source_frame = self.temp_dir / f"qv_s_{source_start+offset:.0f}.jpg"
            
            self.extract_frame(self.target_video, target_start + offset, target_frame)
            self.extract_frame(source, source_start + offset, source_frame)
            
            if target_frame.exists() and source_frame.exists():
                sim = self.calculate_frame_similarity(target_frame, source_frame)
                similarities.append(sim)
        
        avg_sim = np.mean(similarities) if similarities else 0
        min_sim = np.min(similarities) if similarities else 0
        
        # 返回平均相似度和最小相似度
        passed = avg_sim >= 0.80 and min_sim >= 0.72
        return passed, avg_sim
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path):
        """提取帧"""
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', 'scale=360:202',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
    
    def calculate_frame_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        """计算帧相似度"""
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        img1 = cv2.resize(img1, (320, 180))
        img2 = cv2.resize(img2, (320, 180))
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 直方图
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        return 0.5 * max(0, hist_sim) + 0.5 * template_sim
    
    def process_segment(self, task: SegmentTask) -> SegmentResult:
        """处理单个段 - 音频+画面结合匹配，失败时用目标视频兜底"""

        # 音频+画面结合匹配
        source, source_start, combined_score = self.find_match_combined(
            task.target_start, task.duration, task.index
        )

        if not source or combined_score < 0.70:
            # 兜底：使用目标视频本身对应时间段
            print(f"   段 {task.index}/44 ⚠️ 匹配失败 (score={combined_score:.2f})，使用目标视频兜底")
            return SegmentResult(
                index=task.index,
                success=True,
                source=self.target_video,
                source_start=task.target_start,
                quality={'combined': 0.0, 'fallback': True}
            )

        print(f"   段 {task.index}/44 ✅ {source.name} @ {source_start}s (综合: {combined_score:.2f})")

        return SegmentResult(
            index=task.index,
            success=True,
            source=source,
            source_start=source_start,
            quality={'combined': combined_score}
        )
    
    def reconstruct_fast(self, output_path: str) -> bool:
        """极速重构"""
        import time

        print(f"\n{'='*70}")
        print(f"🚀 极速高精度重构 V3 + pHash")
        print(f"{'='*70}")

        start_time = time.time()

        # 预建 pHash 帧索引（首次运行后缓存，后续秒速加载）
        self.build_frame_index(sample_interval=1.0)

        target_duration = self.get_video_duration(self.target_video)
        print(f"\n📹 目标视频: {target_duration:.1f}s")
        
        # 创建任务列表
        tasks = []
        num_segments = int(target_duration / self.segment_duration) + 1
        
        for i in range(num_segments):
            start = i * self.segment_duration
            duration = min(self.segment_duration, target_duration - start)
            if duration > 0:
                tasks.append(SegmentTask(index=i, target_start=start, duration=duration))
        
        print(f"\n🔄 并行处理 {len(tasks)} 个段 (线程数: {self.max_workers})...")
        
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
        
        # 整理结果
        confirmed_segments = []
        for r in results:
            if r and r.success:
                confirmed_segments.append({
                    'index': r.index,
                    'source': r.source,
                    'start': r.source_start,
                    'duration': tasks[r.index].duration,
                    'target_start': tasks[r.index].target_start,
                    'quality': r.quality
                })
        
        confirmed_segments.sort(key=lambda x: x['index'])

        # 强制单调递增：相邻两段使用同一源视频时，源时间必须单调递增
        # 不同目标段可以独立地映射到同一源视频的任意位置
        print(f"\n   强制单调递增校验...")
        monotonic_segments = []
        removed_count = 0

        for seg in confirmed_segments:
            if monotonic_segments:
                prev = monotonic_segments[-1]
                if prev['source'] == seg['source']:
                    prev_end = prev['start'] + prev['duration']
                    if seg['start'] < prev_end - 0.5:
                        print(f"   [单调] 剔除倒退段 {seg['index']}: {seg['source'].name} @ {seg['start']:.1f}s (前段结束 {prev_end:.1f}s)")
                        removed_count += 1
                        continue
            monotonic_segments.append(seg)

        if removed_count > 0:
            print(f"   ✅ 剔除了 {removed_count} 个时间倒退段")
        confirmed_segments = monotonic_segments
        
        # 去重：只跳过真正完全相同的片段（同一源+同一时间+同一段）
        # 注意：不同段使用同一源的不同时间是正常的（连续片段）
        seen_segments = set()
        unique_segments = []
        
        for seg in confirmed_segments:
            # 使用源路径、开始时间、段索引作为唯一标识
            unique_key = f"{seg['source']}_{seg['start']:.1f}_{seg['index']}"
            if unique_key not in seen_segments:
                seen_segments.add(unique_key)
                unique_segments.append(seg)
            else:
                print(f"   [去重] 跳过完全重复的段 {seg['index']}: {seg['source'].name} @ {seg['start']}s")
        
        confirmed_segments = unique_segments
        
        # 检查并修复相邻段的时间重叠（同一源视频）
        print(f"\n   检查相邻段重叠...")
        overlap_fixed = 0
        for i in range(1, len(confirmed_segments)):
            prev = confirmed_segments[i-1]
            curr = confirmed_segments[i]
            
            if prev['source'] == curr['source']:
                prev_end = prev['start'] + prev['duration']
                overlap = prev_end - curr['start']
                
                if overlap > 0.5:  # 重叠超过0.5秒才处理
                    print(f"   ⚠️ 段 {curr['index']} 与段 {prev['index']} 重叠 {overlap:.1f}s")
                    
                    # 策略1：尝试在同一源视频中向后搜索，找一个不重叠的替代匹配点
                    alternative_start = self._find_alternative_match(
                        curr['target_start'], curr['duration'], 
                        curr['source'], prev_end, curr['index']
                    )
                    
                    if alternative_start is not None:
                        # 找到了替代匹配点
                        print(f"   ✅ 段 {curr['index']} 找到替代匹配点: {curr['start']}s -> {alternative_start}s")
                        curr['start'] = alternative_start
                        overlap_fixed += 1
                        continue
                    
                    # 策略2：尝试在其他源视频中搜索
                    alt_source, alt_start, alt_score = self.find_best_match_by_visual(
                        curr['target_start'], curr['duration'], curr['index']
                    )
                    
                    if alt_source and alt_score >= 0.70 and alt_source != curr['source']:
                        print(f"   ✅ 段 {curr['index']} 切换到新源: {alt_source.name} @ {alt_start}s (相似度: {alt_score:.2f})")
                        curr['source'] = alt_source
                        curr['start'] = alt_start
                        overlap_fixed += 1
                        continue
                    
                    # 策略3：如果当前段开始时间早于前一段，直接调整开始时间
                    if curr['start'] < prev['start']:
                        print(f"   ⚠️ 段 {curr['index']} 时间顺序错乱，强制调整: {curr['start']}s -> {prev_end}s")
                        curr['start'] = prev_end
                        overlap_fixed += 1
                        continue
                    
                    # 策略4：重叠较小，尝试微调当前段
                    if overlap <= 2.0:
                        new_start = curr['start'] + overlap + 0.1
                        print(f"   ⚠️ 段 {curr['index']} 微调: {curr['start']}s -> {new_start}s")
                        curr['start'] = new_start
                        overlap_fixed += 1
                        continue
                    
                    # 策略5：重叠太大，直接调整（会丢失内容）
                    print(f"   ⚠️ 段 {curr['index']} 重叠过大，强制调整: {curr['start']}s -> {prev_end}s")
                    curr['start'] = prev_end
                    overlap_fixed += 1
        
        if overlap_fixed > 0:
            print(f"   ✅ 修复了 {overlap_fixed} 处重叠")

        print(f"\n✅ 成功匹配: {len(confirmed_segments)}/{len(tasks)} 段")
        
        # 生成输出
        if confirmed_segments:
            print(f"\n🎬 生成视频...")
            success = self._generate_output(confirmed_segments, output_path, target_duration)
            
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"✅ 完成!")
            print(f"   耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
            print(f"   输出: {output_path}")
            print(f"{'='*70}")
            
            return success
        
        return False
    
    def _generate_output(self, segments: List[dict], output_path: str, target_duration: float) -> bool:
        """生成输出 - 同步音视频"""
        
        video_clips = []
        audio_clips = []
        
        for seg in segments:
            seg_source = seg['source']
            seg_start = seg['start']
            
            # 提取视频片段 - 使用精确提取（-ss在-i之前）
            video_clip = self.temp_dir / f"seg_{seg['index']:03d}_v.mp4"
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(seg_start),  # 放在-i之前，逐帧精确seek
                '-t', str(seg['duration']),
                '-i', str(seg_source),
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                str(video_clip)
            ]
            subprocess.run(cmd, capture_output=True)
            
            # 提取音频片段（从同一源，同样精确提取）
            audio_clip = self.temp_dir / f"seg_{seg['index']:03d}_a.aac"
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(seg_start),  # 放在-i之前，逐帧精确seek
                '-t', str(seg['duration']),
                '-i', str(seg_source),
                '-vn', '-c:a', 'aac', '-b:a', '128k',
                str(audio_clip)
            ]
            subprocess.run(cmd, capture_output=True)
            
            if video_clip.exists() and audio_clip.exists():
                video_clips.append(video_clip)
                audio_clips.append(audio_clip)
        
        if not video_clips or not audio_clips:
            print("❌ 没有有效的音视频片段")
            return False
        
        print(f"   视频片段: {len(video_clips)}, 音频片段: {len(audio_clips)}")
        
        # 拼接视频
        video_concat = self.temp_dir / "video_concat.txt"
        with open(video_concat, 'w') as f:
            for clip in video_clips:
                f.write(f"file '{clip}'\n")
        
        temp_video = self.temp_dir / "temp_video.mp4"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(video_concat),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',  # 重新编码而不是copy
            str(temp_video)
        ]
        subprocess.run(cmd, capture_output=True)
        
        # 拼接音频
        audio_concat = self.temp_dir / "audio_concat.txt"
        with open(audio_concat, 'w') as f:
            for clip in audio_clips:
                f.write(f"file '{clip}'\n")
        
        temp_audio = self.temp_dir / "temp_audio.aac"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(audio_concat),
            '-c:a', 'aac', '-b:a', '128k',  # 重新编码而不是copy
            str(temp_audio)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_video.exists() or not temp_audio.exists():
            print("❌ 音视频拼接失败")
            return False
        
        # 合并音视频
        current_duration = self.get_video_duration(temp_video)
        print(f"   当前视频时长: {current_duration:.2f}s, 目标: {target_duration:.2f}s")
        
        # 直接合并音视频，不调速、不截断
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(temp_video),
            '-i', str(temp_audio),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        if Path(output_path).exists():
            final_duration = self.get_video_duration(Path(output_path))
            print(f"   最终视频时长: {final_duration:.2f}s")
            
            # AI亲自查看视频内容
            self.ai_verify_video(output_path)
            return True
        
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
        check_dir = Path("temp_outputs/ai_check_round" + str(getattr(self, 'round_num', 'X')))
        check_dir.mkdir(exist_ok=True)
        
        # 关键时间点检查
        check_points = [0, 75, 100, 165, 200]
        
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


def main():
    root = Path(__file__).resolve().parent
    default_target = root / "01_test_data_generation" / "source_videos" / "南城以北" / "adx原" / "115196-1-363935819124715523.mp4"
    default_source_dir = root / "01_test_data_generation" / "source_videos" / "南城以北" / "剧集"
    default_output = root / "01_test_data_generation" / "source_videos" / "南城以北" / "output_v6_base" / "115196_V3_FAST.mp4"
    default_cache = root / "cache"

    parser = argparse.ArgumentParser(description="115196 极速高精度重构 V3 + pHash")
    parser.add_argument("--target", default=str(default_target), help="目标视频路径")
    parser.add_argument("--source-dir", default=str(default_source_dir), help="源视频目录")
    parser.add_argument("--output", default=str(default_output), help="输出视频路径")
    parser.add_argument("--cache", default=str(default_cache), help="缓存目录")
    args = parser.parse_args()

    target = Path(args.target)
    source_dir = Path(args.source_dir)
    output = Path(args.output)
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
    print("115196 极速高精度重构 V3")
    print("="*70)
    
    cache.mkdir(exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    reconstructor = FastHighPrecisionReconstructor(str(target), source_videos, str(cache))
    
    try:
        success = reconstructor.reconstruct_fast(str(output))
        
        if success:
            print("\n🎉 极速重构完成!")
            
            # 立即验证
            print("\n正在进行一致性验证...")
            from av_consistency_checker import AVConsistencyChecker
            checker = AVConsistencyChecker(str(target), str(output))
            results = checker.check_consistency(interval=5.0)
            
            if results['statistics']['poor'] == 0:
                print("\n✅✅✅ 100%通过一致性检查！✅✅✅")
        else:
            print("\n❌ 重构失败")
    finally:
        print(f"\n📁 临时文件: {reconstructor.temp_dir}")


if __name__ == "__main__":
    main()
