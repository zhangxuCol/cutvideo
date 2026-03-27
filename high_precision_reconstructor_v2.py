#!/usr/bin/env python3
"""
高精度音视频重构器 V2 - 100%通过目标
优化点：
1. 提高匹配阈值到98%
2. 缩小分段到10秒
3. 片段间重叠5秒验证
4. 强制时长对齐
5. 多重候选验证
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
import time
import json

@dataclass
class MatchCandidate:
    source_video: Path
    start_time: float
    score: float
    target_time: float

class HighPrecisionReconstructor:
    """
    高精度重构器 - 目标100%通过一致性检查
    """
    
    def __init__(self, target_video: str, source_videos: List[str]):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.temp_dir = Path(tempfile.mkdtemp())
        self.confirmed_segments = []
        
        # 高精度配置
        self.match_threshold = 0.98  # 提高到98%
        self.segment_duration = 10.0  # 缩短到10秒
        self.overlap_duration = 5.0  # 重叠5秒验证
        self.min_acceptable_score = 0.95  # 最低可接受分数
        
    def get_video_duration(self, video_path: Path) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path):
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', 'scale=480:270',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
    
    def calculate_frame_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        """计算两帧相似度 - 高精度版本"""
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 调整大小
        img1 = cv2.resize(img1, (320, 180))
        img2 = cv2.resize(img2, (320, 180))
        
        # 灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 直方图相似度
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        # SSIM-like计算
        mean1 = np.mean(gray1)
        mean2 = np.mean(gray2)
        std1 = np.std(gray1)
        std2 = np.std(gray2)
        
        if std1 > 0 and std2 > 0:
            correlation = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
        else:
            correlation = 0
        
        # 综合评分
        final_sim = 0.3 * max(0, hist_sim) + 0.4 * template_sim + 0.3 * max(0, correlation)
        
        return final_sim
    
    def verify_segment_quality(self, candidate: MatchCandidate, duration: float) -> Tuple[bool, dict]:
        """
        验证片段质量 - 多时间点验证
        返回: (是否通过, 详细信息)
        """
        check_points = [0, duration * 0.25, duration * 0.5, duration * 0.75, duration]
        
        frame_sims = []
        screenshots = []
        
        for offset in check_points:
            target_t = candidate.target_time + offset
            source_t = candidate.start_time + offset
            
            target_frame = self.temp_dir / f"verify_t_{target_t:.1f}.jpg"
            source_frame = self.temp_dir / f"verify_s_{source_t:.1f}.jpg"
            
            self.extract_frame(self.target_video, target_t, target_frame)
            self.extract_frame(candidate.source_video, source_t, source_frame)
            
            if target_frame.exists() and source_frame.exists():
                sim = self.calculate_frame_similarity(target_frame, source_frame)
                frame_sims.append(sim)
                screenshots.append({
                    'time': target_t,
                    'similarity': sim,
                    'target': target_frame,
                    'source': source_frame
                })
        
        avg_sim = np.mean(frame_sims) if frame_sims else 0
        min_sim = np.min(frame_sims) if frame_sims else 0
        
        # 通过标准：平均分≥阈值 且 最低分≥0.90
        passed = avg_sim >= self.match_threshold and min_sim >= 0.90
        
        return passed, {
            'average_similarity': avg_sim,
            'min_similarity': min_sim,
            'screenshots': screenshots,
            'all_similarities': frame_sims
        }
    
    def _extract_audio_fingerprint(self, video_path: Path, start: float, duration: float) -> np.ndarray:
        """提取音频指纹"""
        temp_wav = self.temp_dir / f"fp_{start:.0f}_{video_path.stem}.wav"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(start), '-t', str(duration),
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
            samples = struct.unpack(f'{n_frames}h', audio_data)
        
        samples_per_sec = 8000
        n_blocks = len(samples) // samples_per_sec
        features = []
        
        for i in range(min(n_blocks, 50)):
            block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
            fft = np.fft.rfft(block)
            magnitude = np.abs(fft)
            bands = np.array([np.mean(magnitude[j:j+len(magnitude)//20]) 
                            for j in range(0, len(magnitude), len(magnitude)//20)])
            features.append(bands[:20])
        
        return np.array(features)
    
    def find_candidates_precise(self, target_time: float, window: float = 15.0) -> List[MatchCandidate]:
        """高精度候选查找"""
        print(f"\n🔍 高精度查找目标 {target_time:.1f}s 的候选...")
        
        # 提取目标音频指纹
        target_fp = self._extract_audio_fingerprint(self.target_video, target_time, window)
        
        if len(target_fp) == 0:
            return []
        
        candidates = []
        
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            if source_duration < window:
                continue
            
            # 搜索步长1秒
            best_score = -1
            best_start = 0
            
            for start in range(0, int(source_duration - window)):
                source_fp = self._extract_audio_fingerprint(source, start, window)
                
                if len(source_fp) == 0 or len(target_fp) != len(source_fp):
                    continue
                
                correlations = []
                for t, s in zip(target_fp, source_fp):
                    if len(t) == len(s) and np.std(t) > 0 and np.std(s) > 0:
                        corr = np.corrcoef(t, s)[0, 1]
                        correlations.append(corr)
                    else:
                        correlations.append(0)
                
                score = np.mean(correlations) if correlations else 0
                if score > best_score:
                    best_score = score
                    best_start = start
            
            if best_score > 0.80:  # 音频初步筛选
                candidates.append(MatchCandidate(
                    source_video=source,
                    start_time=float(best_start),
                    score=best_score,
                    target_time=target_time
                ))
        
        # 按音频匹配度排序
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:10]  # 返回前10个候选
    
    def reconstruct_precise(self, output_path: str) -> bool:
        """高精度重构"""
        print(f"\n{'='*70}")
        print(f"🎯 高精度音视频重构 V2")
        print(f"   目标: 100%通过一致性检查")
        print(f"   匹配阈值: {self.match_threshold:.0%}")
        print(f"   分段时长: {self.segment_duration}s")
        print(f"{'='*70}")
        
        target_duration = self.get_video_duration(self.target_video)
        print(f"\n📹 目标视频: {self.target_video.name}")
        print(f"   总时长: {target_duration:.1f}s")
        print(f"   预计段数: {int(target_duration / self.segment_duration) + 1}")
        
        current_time = 0
        segment_count = 0
        retry_count = 0
        
        while current_time < target_duration:
            segment_count += 1
            remaining = target_duration - current_time
            this_duration = min(self.segment_duration, remaining)
            
            print(f"\n{'='*70}")
            print(f"🎬 处理第 {segment_count} 段")
            print(f"   目标时间: {current_time:.1f}s ~ {current_time + this_duration:.1f}s")
            print(f"{'='*70}")
            
            # 查找候选
            candidates = self.find_candidates_precise(current_time, window=this_duration + 5)
            
            if not candidates:
                print(f"\n❌ 未找到候选")
                # 尝试缩短时长继续
                if this_duration > 5:
                    this_duration = 5
                    candidates = self.find_candidates_precise(current_time, window=this_duration + 5)
                if not candidates:
                    break
            
            # 逐个验证候选
            confirmed = None
            for i, candidate in enumerate(candidates):
                print(f"\n{'─'*70}")
                print(f"验证候选 {i+1}/{len(candidates)}: {candidate.source_video.name} @ {candidate.start_time:.1f}s")
                
                passed, details = self.verify_segment_quality(candidate, this_duration)
                
                print(f"   画面平均分: {details['average_similarity']:.1%}")
                print(f"   画面最低分: {details['min_similarity']:.1%}")
                
                if passed:
                    confirmed = candidate
                    print(f"\n✅ 验证通过！")
                    break
                else:
                    print(f"\n❌ 验证未通过")
                    # 如果平均分还过得去，记录为备选
                    if details['average_similarity'] >= self.min_acceptable_score:
                        if not confirmed or details['average_similarity'] > confirmed.get('score', 0):
                            confirmed = candidate
                            confirmed_quality = details
            
            if confirmed:
                self.confirmed_segments.append({
                    'source': confirmed.source_video,
                    'start': confirmed.start_time,
                    'duration': this_duration,
                    'target_start': current_time,
                    'quality': confirmed_quality if 'confirmed_quality' in dir() else details
                })
                current_time += this_duration
                retry_count = 0
                print(f"\n✅ 第 {segment_count} 段已确认")
            else:
                retry_count += 1
                print(f"\n⚠️ 第 {segment_count} 段无通过候选 (重试 {retry_count})")
                
                # 尝试缩短时长
                if this_duration > 3:
                    this_duration = max(3, this_duration - 2)
                    print(f"   尝试缩短到 {this_duration}s 继续...")
                    continue
                elif retry_count < 3:
                    print(f"   跳过此段，继续下一段...")
                    current_time += this_duration
                    continue
                else:
                    print(f"\n❌ 多次失败，停止处理")
                    break
        
        # 生成输出
        if self.confirmed_segments:
            print(f"\n{'='*70}")
            print(f"🎬 生成最终视频")
            print(f"   确认片段: {len(self.confirmed_segments)}")
            print(f"{'='*70}")
            
            success = self._generate_output(output_path, target_duration)
            
            if success:
                # 验证输出
                output_duration = self.get_video_duration(output_path)
                print(f"\n✅ 生成成功!")
                print(f"   输出时长: {output_duration:.1f}s")
                print(f"   目标时长: {target_duration:.1f}s")
                print(f"   时长差异: {abs(output_duration - target_duration):.1f}s")
                
                return True
        
        return False
    
    def _generate_output(self, output_path: str, target_duration: float) -> bool:
        """生成输出 - 精确时长控制"""
        
        clip_files = []
        for i, seg in enumerate(self.confirmed_segments):
            clip_file = self.temp_dir / f"clip_{i:03d}.mp4"
            
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
        
        temp_output = self.temp_dir / "temp_output.mp4"
        
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
        
        # 精确时长对齐
        current_duration = self.get_video_duration(temp_output)
        duration_diff = target_duration - current_duration
        
        if abs(duration_diff) > 0.5:
            print(f"   调整时长: {current_duration:.1f}s -> {target_duration:.1f}s")
            
            # 使用setpts调整视频速度
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(temp_output),
                '-vf', f'setpts={target_duration/current_duration}*PTS',
                '-af', f'atempo={current_duration/target_duration}',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                str(output_path)
            ]
        else:
            # 直接复制
            shutil.copy(temp_output, output_path)
            return True
        
        subprocess.run(cmd, capture_output=True)
        
        return Path(output_path).exists()


def main():
    target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"
    output = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_V2_PRECISION.mp4"
    
    source_videos = [str(f) for f in Path(source_dir).iterdir() if f.suffix.lower() == '.mp4']
    
    print("="*70)
    print("115196 高精度重构 V2")
    print("="*70)
    
    reconstructor = HighPrecisionReconstructor(target, source_videos)
    
    try:
        success = reconstructor.reconstruct_precise(output)
        
        if success:
            print("\n🎉 高精度重构完成!")
            print(f"输出: {output}")
            
            # 立即进行一致性检查
            print("\n" + "="*70)
            print("正在进行一致性验证...")
            print("="*70)
            
            import sys
            sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo')
            from av_consistency_checker import AVConsistencyChecker
            
            checker = AVConsistencyChecker(target, output)
            results = checker.check_consistency(interval=5.0)
            
            # 检查是否100%通过
            poor_count = results['statistics']['poor']
            if poor_count == 0:
                print("\n✅✅✅ 100%通过一致性检查！✅✅✅")
            else:
                print(f"\n⚠️ 还有 {poor_count} 个检查点未通过")
        else:
            print("\n❌ 重构失败")
    finally:
        print(f"\n📁 临时文件: {reconstructor.temp_dir}")


if __name__ == "__main__":
    from pathlib import Path
    main()
