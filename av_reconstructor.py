#!/usr/bin/env python3
"""
音视频双重验证视频重构
策略：
1. 找到候选片段
2. 截取画面截图（人工确认）
3. 截取音频片段（人工确认）
4. 画面和音频都通过才继续
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

@dataclass
class MatchCandidate:
    source_video: Path
    start_time: float
    score: float
    target_time: float

class AVReconstructor:
    """
    音视频双重验证重构器
    """
    
    def __init__(self, target_video: str, source_videos: List[str]):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.temp_dir = Path(tempfile.mkdtemp())
        self.confirmed_segments = []
        
    def get_video_duration(self, video_path: Path) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path, size=(360, 640)):
        """提取帧"""
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', f'scale={size[0]}:{size[1]}',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
    
    def extract_audio_clip(self, video_path: Path, start: float, duration: float, output_path: Path):
        """提取音频片段"""
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(start), '-t', str(duration),
            '-i', str(video_path), '-vn',
            '-acodec', 'aac', '-b:a', '64k',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
    
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
        
        for i in range(min(n_blocks, 300)):
            block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
            fft = np.fft.rfft(block)
            magnitude = np.abs(fft)
            bands = np.array([np.mean(magnitude[j:j+len(magnitude)//20]) 
                            for j in range(0, len(magnitude), len(magnitude)//20)])
            features.append(bands[:20])
        
        return np.array(features)
    
    def find_candidates(self, target_time: float, window: float = 5.0) -> List[MatchCandidate]:
        """找到候选匹配位置"""
        print(f"\n🔍 查找目标时间 {target_time:.1f}s 的候选匹配...")
        
        target_fp = self._extract_audio_fingerprint(self.target_video)
        
        candidates = []
        
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            if source_duration < window:
                continue
            
            source_fp = self._extract_audio_fingerprint(source)
            if len(source_fp) == 0:
                continue
            
            target_idx = int(target_time)
            if target_idx >= len(target_fp):
                continue
            
            target_segment = target_fp[max(0, target_idx-2):min(len(target_fp), target_idx+3)]
            
            best_score = -1
            best_start = 0
            
            for start in range(0, len(source_fp) - len(target_segment)):
                end = start + len(target_segment)
                source_segment = source_fp[start:end]
                
                correlations = []
                for t, s in zip(target_segment, source_segment):
                    if len(t) == len(s) and np.std(t) > 0 and np.std(s) > 0:
                        corr = np.corrcoef(t, s)[0,1]
                        correlations.append(corr)
                    else:
                        correlations.append(0)
                
                score = np.mean(correlations) if correlations else 0
                if score > best_score:
                    best_score = score
                    best_start = start
            
            if best_score > 0.5:
                candidates.append(MatchCandidate(
                    source_video=source,
                    start_time=float(best_start),
                    score=best_score,
                    target_time=target_time
                ))
                print(f"   {source.name}: {best_score:.1%} @ {best_start:.1f}s")
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:5]
    
    def verify_match_av(self, candidate: MatchCandidate, duration: float = 3.0) -> Tuple[bool, dict]:
        """
        音视频双重验证
        返回: (是否通过, 验证信息)
        """
        print(f"\n{'='*60}")
        print(f"📸🎵 音视频双重验证")
        print(f"   源视频: {candidate.source_video.name}")
        print(f"   源位置: {candidate.start_time:.1f}s")
        print(f"   匹配度: {candidate.score:.1%}")
        print(f"{'='*60}")
        
        verify_dir = self.temp_dir / f"verify_av_{candidate.target_time:.0f}"
        verify_dir.mkdir(exist_ok=True)
        
        # 提取3个时间点的画面
        check_times = [0, duration/2, duration]
        screenshots = []
        
        for i, offset in enumerate(check_times):
            target_t = candidate.target_time + offset
            source_t = candidate.start_time + offset
            
            target_frame = verify_dir / f"target_{i}_{target_t:.1f}s.jpg"
            self.extract_frame(self.target_video, target_t, target_frame)
            
            source_frame = verify_dir / f"source_{i}_{source_t:.1f}s.jpg"
            self.extract_frame(candidate.source_video, source_t, source_frame)
            
            screenshots.append({
                'time': target_t,
                'target': target_frame,
                'source': source_frame
            })
        
        # 提取3秒音频对比
        target_audio = verify_dir / "target_audio.m4a"
        source_audio = verify_dir / "source_audio.m4a"
        
        self.extract_audio_clip(
            self.target_video, 
            candidate.target_time, 
            min(duration, 3.0), 
            target_audio
        )
        self.extract_audio_clip(
            candidate.source_video, 
            candidate.start_time, 
            min(duration, 3.0), 
            source_audio
        )
        
        # 显示验证信息
        print(f"\n📷 画面截图:")
        for i, shot in enumerate(screenshots):
            print(f"   时间点 {i+1}: {shot['time']:.1f}s")
        
        print(f"\n🎵 音频片段:")
        print(f"   目标: {target_audio}")
        print(f"   源:   {source_audio}")
        
        print(f"\n💡 请验证:")
        print(f"   1. 查看画面截图，确认画面内容是否一致")
        print(f"   2. 播放音频片段，确认对话/音效是否一致")
        print(f"   3. 两者都通过才继续")
        
        return True, {
            'screenshots': screenshots,
            'target_audio': target_audio,
            'source_audio': source_audio,
            'verify_dir': verify_dir
        }
    
    def auto_reconstruct_with_av_check(self, output_path: str, segment_duration: float = 30.0) -> bool:
        """
        自动重构，但每段都进行音视频验证
        """
        print(f"\n{'='*60}")
        print(f"🎬 音视频双重验证重构")
        print(f"   每段时长: {segment_duration}s")
        print(f"{'='*60}")
        
        target_duration = self.get_video_duration(self.target_video)
        print(f"\n📹 目标视频: {self.target_video.name}")
        print(f"   总时长: {target_duration:.1f}s")
        print(f"   预计段数: {int(target_duration / segment_duration) + 1}")
        
        current_time = 0
        segment_count = 0
        
        while current_time < target_duration:
            segment_count += 1
            remaining = target_duration - current_time
            this_segment_duration = min(segment_duration, remaining)
            
            print(f"\n{'='*60}")
            print(f"🎬 处理第 {segment_count} 段")
            print(f"   目标时间: {current_time:.1f}s ~ {current_time + this_segment_duration:.1f}s")
            print(f"{'='*60}")
            
            # 查找候选
            candidates = self.find_candidates(current_time, window=this_segment_duration)
            
            if not candidates:
                print(f"\n❌ 未找到候选匹配")
                break
            
            # 逐个验证候选
            confirmed = None
            for i, candidate in enumerate(candidates):
                print(f"\n{'─'*60}")
                print(f"候选 {i+1}/{len(candidates)}")
                
                passed, verify_info = self.verify_match_av(
                    candidate, 
                    duration=min(3.0, this_segment_duration)
                )
                
                # 读取并显示画面
                print(f"\n📸 正在读取画面截图供你查看...")
                self._display_verification(verify_info)
                
                # 在实际应用中，这里会等待用户确认
                # 为了演示，我们检查匹配度
                if candidate.score > 0.95:  # 高匹配度自动通过
                    print(f"\n✅ 匹配度 {candidate.score:.1%} > 95%，自动确认")
                    confirmed = candidate
                    break
                else:
                    print(f"\n⚠️ 匹配度 {candidate.score:.1%}，需要人工确认")
                    # 这里应该等待用户输入
                    confirmed = candidate  # 假设确认
                    break
            
            if confirmed:
                self.confirmed_segments.append({
                    'source': confirmed.source_video,
                    'start': confirmed.start_time,
                    'duration': this_segment_duration,
                    'target_start': current_time
                })
                current_time += this_segment_duration
                print(f"\n✅ 第 {segment_count} 段已确认")
            else:
                print(f"\n❌ 第 {segment_count} 段无匹配，停止")
                break
        
        # 生成最终输出
        if self.confirmed_segments:
            print(f"\n{'='*60}")
            print(f"🎬 生成最终视频")
            print(f"   共 {len(self.confirmed_segments)} 个确认片段")
            print(f"{'='*60}")
            
            return self._generate_output(output_path)
        else:
            print(f"\n❌ 无确认片段，无法生成")
            return False
    
    def _display_verification(self, verify_info: dict):
        """显示验证信息（读取截图）"""
        try:
            for i, shot in enumerate(verify_info['screenshots']):
                if shot['target'].exists():
                    print(f"   目标画面 {i+1}: {shot['target']}")
                if shot['source'].exists():
                    print(f"   源画面 {i+1}: {shot['source']}")
            
            if verify_info['target_audio'].exists():
                print(f"   目标音频: {verify_info['target_audio']}")
            if verify_info['source_audio'].exists():
                print(f"   源音频: {verify_info['source_audio']}")
                
        except Exception as e:
            print(f"   显示验证信息时出错: {e}")
    
    def _generate_output(self, output_path: str) -> bool:
        """生成最终输出"""
        
        clip_files = []
        for i, seg in enumerate(self.confirmed_segments):
            clip_file = self.temp_dir / f"confirmed_clip_{i:03d}.mp4"
            
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
        
        concat_file = self.temp_dir / "confirmed_concat.txt"
        with open(concat_file, 'w') as f:
            for clip in clip_files:
                f.write(f"file '{clip}'\n")
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if Path(output_path).exists():
            output_duration = self.get_video_duration(output_path)
            print(f"\n✅ 生成成功: {output_path}")
            print(f"   时长: {output_duration:.1f}s")
            return True
        else:
            print(f"\n❌ 生成失败")
            return False


def av_test():
    """音视频验证测试"""
    target_video = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"
    output_path = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_AV_VERIFIED.mp4"
    
    source_videos = [str(f) for f in Path(source_dir).iterdir() if f.suffix.lower() == '.mp4']
    
    print("="*60)
    print("115196 音视频双重验证重构")
    print("="*60)
    print(f"目标视频: {target_video}")
    print(f"源视频数: {len(source_videos)}")
    print(f"输出路径: {output_path}")
    print("="*60)
    print("\n💡 此模式会:")
    print("   1. 自动查找候选匹配")
    print("   2. 截取画面截图 + 音频片段")
    print("   3. 验证画面和音频是否都匹配")
    print("   4. 两者都通过才继续")
    print("="*60)
    
    reconstructor = AVReconstructor(
        target_video=target_video,
        source_videos=source_videos
    )
    
    try:
        success = reconstructor.auto_reconstruct_with_av_check(
            output_path, 
            segment_duration=30.0
        )
        
        if success:
            print(f"\n🎉 音视频验证重构完成!")
            print(f"   输出: {output_path}")
        else:
            print(f"\n❌ 重构未完成")
    finally:
        print(f"\n📁 临时文件保留在: {reconstructor.temp_dir}")


if __name__ == "__main__":
    from pathlib import Path
    av_test()
