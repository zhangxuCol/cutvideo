#!/usr/bin/env python3
"""
115196视频修复策略方案 - V6 Fix
针对多源拼接导致的音画不同步和画面抖动问题
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from typing import List, Tuple
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

class VideoReconstructorV6Fix:
    """
    V6修复版 - 解决多源拼接的音画同步和画面抖动问题
    
    核心改进：
    1. 音频跟随视频片段同步裁剪（而非使用完整目标音频）
    2. 片段间添加过渡效果减少抖动
    3. 时长精确对齐
    """
    
    def __init__(self, target_video: str, source_videos: List[str], config: dict = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.config = config or {}
        self.temp_dir = None
        
        # 配置参数
        self.fps = config.get('fps', 5) if config else 5
        self.similarity_threshold = config.get('similarity_threshold', 0.85) if config else 0.85
        self.min_segment_duration = config.get('min_segment_duration', 0.5) if config else 0.5
        self.match_threshold = config.get('match_threshold', 0.6) if config else 0.6
        self.audio_weight = config.get('audio_weight', 0.4) if config else 0.4
        self.video_weight = config.get('video_weight', 0.6) if config else 0.6
        
        # 新增：过渡时长
        self.transition_duration = config.get('transition_duration', 0.3) if config else 0.3

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
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
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
        """提取音频指纹"""
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
        """音频指纹匹配"""
        source_fp = self._extract_audio_fingerprint(source_video)
        
        if len(target_fp) == 0 or len(source_fp) == 0 or len(target_fp) > len(source_fp):
            return 0, 0
        
        best_score = -1
        best_start = 0
        
        for start in range(0, len(source_fp) - len(target_fp) + 1, 1):
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
            
            audio_score, start_time = self._find_audio_match(target_audio, source)
            
            if audio_score > best_score:
                # 视频帧验证
                sample_times = [0, target_duration * 0.25, target_duration * 0.5]
                video_scores = []
                
                for t in sample_times:
                    if t < target_duration:
                        target_frame = self.temp_dir / f"target_{t:.0f}.jpg"
                        source_frame = self.temp_dir / f"source_{source.stem}_{t:.0f}.jpg"
                        self.extract_frame_at(self.target_video, t, target_frame)
                        self.extract_frame_at(source, start_time + t, source_frame)
                        
                        if target_frame.exists() and source_frame.exists():
                            sim = self.calculate_frame_similarity(target_frame, source_frame)
                            video_scores.append(sim)
                
                video_score = np.mean(video_scores) if video_scores else 0
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
        """多源片段匹配 - 从多个源视频中找最佳片段"""
        print(f"\n🔍 阶段2: 多源片段拼接...")
        
        # 提取目标视频的关键帧
        target_times = np.arange(0, target_duration, 1.0)  # 每秒一帧
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
                # 连续
                current_segment['end_time'] = match['source_time'] + 1
                current_segment['scores'].append(match['score'])
            else:
                # 保存当前片段
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
        
        # 保存最后一个片段
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
            print(f"      片段{i}: {seg.source_video.name} @{seg.start_time:.1f}s~{seg.end_time:.1f}s (目标{seg.target_start:.1f}s~{seg.target_end:.1f}s)")
        
        if len(segments) > 5:
            print(f"      ... 还有 {len(segments)-5} 个片段")
        
        return segments

    def generate_multi_source_output_fixed(self, segments: List[VideoSegment], output_path: str, target_duration: float):
        """
        修复版多源拼接输出 - 解决音画同步问题
        
        核心改进：
        1. 从每个源片段中提取对应的音频片段（而非使用完整目标音频）
        2. 精确控制每个片段的时长
        3. 添加淡入淡出过渡减少抖动
        """
        print(f"\n🎬 生成修复版多源拼接视频...")
        
        if not segments:
            print(f"   ❌ 没有可用片段")
            return False
        
        # 按目标时间排序
        segments = sorted(segments, key=lambda x: x.target_start)
        
        # 计算每个片段应该贡献的时长（基于目标时间对齐）
        segment_clips = []
        total_target_duration = 0
        
        for i, seg in enumerate(segments):
            # 计算这个片段应该覆盖的目标时长
            if i < len(segments) - 1:
                target_duration_seg = segments[i+1].target_start - seg.target_start
            else:
                target_duration_seg = target_duration - seg.target_start
            
            # 源片段实际可用时长
            source_available = seg.end_time - seg.start_time
            
            # 取最小值（不能超过源片段可用时长）
            actual_duration = min(target_duration_seg, source_available)
            
            segment_clips.append({
                'source': seg.source_video,
                'start': seg.start_time,
                'duration': actual_duration,
                'target_start': seg.target_start
            })
            
            total_target_duration += actual_duration
        
        print(f"   计划拼接 {len(segment_clips)} 个片段，总时长 {total_target_duration:.1f}s")
        
        # 第一步：提取每个片段的视频和音频（带淡入淡出）
        clip_files = []
        for i, clip in enumerate(segment_clips):
            # 视频片段（带淡入淡出）
            video_clip = self.temp_dir / f"clip_{i:03d}_v.mp4"
            
            fade_in = 0.1 if i > 0 else 0
            fade_out = 0.1 if i < len(segment_clips) - 1 else 0
            
            cmd_v = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(clip['start']),
                '-t', str(clip['duration']),
                '-i', str(clip['source']),
                '-vf', f'fade=t=in:st=0:d={fade_in},fade=t=out:st={clip["duration"]-fade_out}:d={fade_out}',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-an',  # 无音频
                str(video_clip)
            ]
            subprocess.run(cmd_v, capture_output=True)
            
            # 音频片段（从同一源提取，带淡入淡出）
            audio_clip = self.temp_dir / f"clip_{i:03d}_a.aac"
            cmd_a = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(clip['start']),
                '-t', str(clip['duration']),
                '-i', str(clip['source']),
                '-vn',  # 无视频
                '-af', f'afade=t=in:st=0:d={fade_in},afade=t=out:st={clip["duration"]-fade_out}:d={fade_out}',
                '-c:a', 'aac', '-b:a', '128k',
                str(audio_clip)
            ]
            subprocess.run(cmd_a, capture_output=True)
            
            if video_clip.exists() and audio_clip.exists():
                clip_files.append({
                    'video': video_clip,
                    'audio': audio_clip,
                    'duration': clip['duration']
                })
        
        if not clip_files:
            print(f"   ❌ 没有成功提取的片段")
            return False
        
        print(f"   成功提取 {len(clip_files)} 个片段")
        
        # 第二步：创建concat列表
        concat_video_list = self.temp_dir / "concat_video.txt"
        concat_audio_list = self.temp_dir / "concat_audio.txt"
        
        with open(concat_video_list, 'w') as f:
            for clip in clip_files:
                f.write(f"file '{clip['video']}'\n")
        
        with open(concat_audio_list, 'w') as f:
            for clip in clip_files:
                f.write(f"file '{clip['audio']}'\n")
        
        # 第三步：分别拼接视频和音频
        temp_video = self.temp_dir / "temp_video.mp4"
        temp_audio = self.temp_dir / "temp_audio.aac"
        
        cmd_concat_v = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_video_list),
            '-c', 'copy',
            str(temp_video)
        ]
        subprocess.run(cmd_concat_v, capture_output=True)
        
        cmd_concat_a = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_audio_list),
            '-c', 'copy',
            str(temp_audio)
        ]
        subprocess.run(cmd_concat_a, capture_output=True)
        
        # 第四步：合并视频和音频
        if temp_video.exists() and temp_audio.exists():
            cmd_merge = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(temp_video),
                '-i', str(temp_audio),
                '-c:v', 'copy', '-c:a', 'copy',
                '-shortest',
                str(output_path)
            ]
            subprocess.run(cmd_merge, capture_output=True)
        
        if Path(output_path).exists():
            output_duration = self.get_video_duration(output_path)
            print(f"   ✅ 生成成功: {output_duration:.1f}s")
            return True
        else:
            print(f"   ❌ 生成失败")
            return False

    def reconstruct_fixed(self, output_path: str) -> List[VideoSegment]:
        """修复版重构 - 解决音画同步问题"""
        print(f"\n{'='*60}")
        print(f"🎬 V6 Fix 修复版重构")
        print(f"   解决：音画同步 + 画面抖动")
        print(f"{'='*60}")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            target_duration = self.get_video_duration(self.target_video)
            print(f"   目标视频: {self.target_video.name} ({target_duration:.1f}s)")
            print(f"   源视频数: {len(self.source_videos)}")
            
            # 阶段1: 尝试单源完整匹配
            single_match = self.find_best_single_source_match(target_duration)
            
            if single_match and single_match['combined_score'] > 0.85:
                print(f"\n✅ 单源匹配成功!")
                print(f"   来源: {single_match['source'].name}")
                print(f"   位置: @{single_match['start_time']:.1f}s")
                print(f"   综合得分: {single_match['combined_score']:.1%}")
                
                segment = VideoSegment(
                    source_video=single_match['source'],
                    start_time=single_match['start_time'],
                    end_time=single_match['start_time'] + target_duration,
                    similarity_score=single_match['combined_score']
                )
                
                self._generate_single_output(segment, output_path)
                return [segment]
            else:
                print(f"\n⚠️ 单源匹配不足，切换到多源拼接模式")
                if single_match:
                    print(f"   最佳单源得分: {single_match['combined_score']:.1%} (需要>85%)")
                
                # 阶段2: 多源片段拼接（使用修复版）
                segments = self.find_multi_source_segments(target_duration)
                
                if segments:
                    total_covered = sum(seg.target_end - seg.target_start for seg in segments)
                    coverage = total_covered / target_duration
                    
                    print(f"\n   覆盖时长: {total_covered:.1f}s / {target_duration:.1f}s ({coverage:.1%})")
                    
                    if coverage > 0.5:  # 至少覆盖50%
                        self.generate_multi_source_output_fixed(segments, output_path, target_duration)
                        return segments
                    else:
                        print(f"   ❌ 覆盖不足，无法生成")
                        return []
                else:
                    print(f"   ❌ 未找到可用片段")
                    return []
        
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    def _generate_single_output(self, segment: VideoSegment, output_path: str):
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


def test_115196_fix():
    """测试115196修复"""
    import sys
    sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')
    
    target_video = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"
    output_path = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196-1-363935819124715523_reconstructed_FIXED.mp4"
    
    # 扫描源视频
    from pathlib import Path
    source_videos = [str(f) for f in Path(source_dir).iterdir() if f.suffix.lower() == '.mp4']
    
    print("="*60)
    print("115196 修复测试")
    print("="*60)
    print(f"目标视频: {target_video}")
    print(f"源视频数: {len(source_videos)}")
    print(f"输出路径: {output_path}")
    
    reconstructor = VideoReconstructorV6Fix(
        target_video=target_video,
        source_videos=source_videos,
        config={
            'fps': 5,
            'similarity_threshold': 0.85,
            'match_threshold': 0.6,
            'audio_weight': 0.4,
            'video_weight': 0.6,
            'transition_duration': 0.3
        }
    )
    
    segments = reconstructor.reconstruct_fixed(output_path)
    
    if segments:
        print(f"\n✅ 修复成功！生成 {len(segments)} 个片段")
        
        # 验证时长
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', output_path],
            capture_output=True, text=True
        )
        output_duration = float(result.stdout.strip())
        
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', target_video],
            capture_output=True, text=True
        )
        target_duration = float(result.stdout.strip())
        
        print(f"\n📊 时长对比:")
        print(f"   原始视频: {target_duration:.2f}s")
        print(f"   输出视频: {output_duration:.2f}s")
        print(f"   差异: {abs(output_duration - target_duration):.2f}s")
        
        if abs(output_duration - target_duration) < 1.0:
            print(f"   ✅ 时长对齐良好")
        else:
            print(f"   ⚠️ 时长仍有差异")
    else:
        print(f"\n❌ 修复失败")


if __name__ == "__main__":
    test_115196_fix()
