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
        self.segment_duration = 10.0
        self.max_workers = 4  # 并行线程数
        
        # 缓存
        self.source_fingerprints = {}
        self.target_fingerprint = None
        
    def get_video_duration(self, video_path: Path) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def extract_audio_fingerprint(self, video_path: Path, start: float = 0, duration: float = None) -> np.ndarray:
        """提取音频指纹 - 带缓存"""
        cache_key = f"{video_path.stem}_{start:.0f}_{duration:.0f if duration else 'full'}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # 检查缓存
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 提取
        if duration is None:
            duration = self.get_video_duration(video_path) - start
        
        temp_wav = self.temp_dir / f"fp_{video_path.stem}_{start:.0f}.wav"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(start), '-t', str(duration),
            '-i', str(video_path), '-vn',
            '-acodec', 'pcm_s16le', '-ar', '4000', '-ac', '1',  # 降低采样率
            str(temp_wav)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_wav.exists():
            return np.array([])
        
        with wave.open(str(temp_wav), 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            samples = struct.unpack(f'{n_frames}h', audio_data)
        
        samples_per_sec = 4000
        n_blocks = len(samples) // samples_per_sec
        features = []
        
        # 降低精度：每秒一个特征，步长2秒
        for i in range(0, min(n_blocks, 200), 2):
            block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
            fft = np.fft.rfft(block)
            magnitude = np.abs(fft)
            bands = np.array([np.mean(magnitude[j:j+len(magnitude)//10]) 
                            for j in range(0, len(magnitude), len(magnitude)//10)])
            features.append(bands[:10])
        
        result = np.array(features)
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    
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
    
    def find_match_fast(self, target_start: float, duration: float) -> Tuple[Path, float, float]:
        """快速匹配 - 使用预计算指纹"""
        
        # 获取目标指纹片段
        target_idx_start = int(target_start / 2)  # 步长2秒
        target_idx_end = int((target_start + duration) / 2)
        
        if target_idx_start >= len(self.target_fingerprint):
            return None, 0, 0
        
        target_segment = self.target_fingerprint[target_idx_start:target_idx_end]
        
        if len(target_segment) == 0:
            return None, 0, 0
        
        best_source = None
        best_start = 0
        best_score = 0
        
        # 搜索所有源视频
        for source, source_fp in self.source_fingerprints.items():
            if len(source_fp) < len(target_segment):
                continue
            
            # 大步长搜索（步长5）
            for start in range(0, len(source_fp) - len(target_segment), 5):
                end = start + len(target_segment)
                source_segment = source_fp[start:end]
                
                correlations = []
                for t, s in zip(target_segment, source_segment):
                    if len(t) == len(s) and np.std(t) > 0 and np.std(s) > 0:
                        corr = np.corrcoef(t, s)[0, 1]
                        correlations.append(corr)
                    else:
                        correlations.append(0)
                
                score = np.mean(correlations) if correlations else 0
                if score > best_score:
                    best_score = score
                    best_start = start * 2  # 转换回秒
                    best_source = source
                
                # 提前退出
                if best_score > 0.98:
                    break
        
        return best_source, best_start, best_score
    
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
        
        passed = avg_sim >= 0.90 and min_sim >= 0.85
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
        """处理单个段"""
        
        # 快速匹配
        source, source_start, audio_score = self.find_match_fast(
            task.target_start, task.duration
        )
        
        if not source or audio_score < 0.80:
            return SegmentResult(index=task.index, success=False)
        
        # 快速画面验证
        passed, frame_score = self.quick_verify(
            source, source_start, task.target_start, task.duration
        )
        
        if not passed:
            return SegmentResult(index=task.index, success=False)
        
        return SegmentResult(
            index=task.index,
            success=True,
            source=source,
            source_start=source_start,
            quality={'audio': audio_score, 'frame': frame_score}
        )
    
    def reconstruct_fast(self, output_path: str) -> bool:
        """极速重构"""
        import time
        
        print(f"\n{'='*70}")
        print(f"🚀 极速高精度重构 V3")
        print(f"   目标: 3分钟/视频")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        target_duration = self.get_video_duration(self.target_video)
        print(f"\n📹 目标视频: {target_duration:.1f}s")
        
        # 预计算指纹
        self.precompute_fingerprints()
        
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
        """生成输出"""
        
        clip_files = []
        for seg in segments:
            clip_file = self.temp_dir / f"seg_{seg['index']:03d}.mp4"
            
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(seg['start']),
                '-t', str(seg['duration']),
                '-i', str(seg['source']),
                '-c', 'copy',
                str(clip_file)
            ]
            subprocess.run(cmd, capture_output=True)
            
            if clip_file.exists():
                clip_files.append(clip_file)
        
        if not clip_files:
            return False
        
        # 拼接
        concat_file = self.temp_dir / "concat.txt"
        with open(concat_file, 'w') as f:
            for clip in clip_files:
                f.write(f"file '{clip}'\n")
        
        temp_output = self.temp_dir / "temp.mp4"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(temp_output)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_output.exists():
            return False
        
        # 时长对齐
        current_duration = self.get_video_duration(temp_output)
        
        if abs(current_duration - target_duration) > 0.5:
            # 调整速度
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(temp_output),
                '-vf', f'setpts={target_duration/current_duration}*PTS',
                '-af', f'atempo={current_duration/target_duration}',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True)
        else:
            shutil.copy(temp_output, output_path)
        
        return Path(output_path).exists()


def main():
    target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"
    output = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_V3_FAST.mp4"
    cache = "/Users/zhangxu/work/项目/cutvideo/cache"
    
    source_videos = [str(f) for f in Path(source_dir).iterdir() if f.suffix.lower() == '.mp4']
    
    print("="*70)
    print("115196 极速高精度重构 V3")
    print("="*70)
    
    Path(cache).mkdir(exist_ok=True)
    
    reconstructor = FastHighPrecisionReconstructor(target, source_videos, cache)
    
    try:
        success = reconstructor.reconstruct_fast(output)
        
        if success:
            print("\n🎉 极速重构完成!")
            
            # 立即验证
            print("\n正在进行一致性验证...")
            from av_consistency_checker import AVConsistencyChecker
            checker = AVConsistencyChecker(target, output)
            results = checker.check_consistency(interval=5.0)
            
            if results['statistics']['poor'] == 0:
                print("\n✅✅✅ 100%通过一致性检查！✅✅✅")
        else:
            print("\n❌ 重构失败")
    finally:
        print(f"\n📁 临时文件: {reconstructor.temp_dir}")


if __name__ == "__main__":
    from pathlib import Path
    main()
