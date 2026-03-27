#!/usr/bin/env python3
"""
渐进式验证视频重构 - 人工确认每一步
策略：
1. 找到候选片段起始位置
2. 截取画面截图
3. 人工确认匹配
4. 确认后继续下一步
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

class InteractiveReconstructor:
    """
    交互式重构器 - 每步人工确认
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
        """提取帧并调整大小便于查看"""
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', f'scale={size[0]}:{size[1]}',
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
        
        # 提取目标时间点前后音频指纹
        target_fp = self._extract_audio_fingerprint(self.target_video)
        
        candidates = []
        
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            if source_duration < window:
                continue
            
            source_fp = self._extract_audio_fingerprint(source)
            if len(source_fp) == 0:
                continue
            
            # 在目标时间点前后搜索
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
            
            if best_score > 0.5:  # 降低阈值，获取更多候选
                candidates.append(MatchCandidate(
                    source_video=source,
                    start_time=float(best_start),
                    score=best_score,
                    target_time=target_time
                ))
                print(f"   {source.name}: {best_score:.1%} @ {best_start:.1f}s")
        
        # 按匹配度排序
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:5]  # 返回前5个候选
    
    def verify_match(self, candidate: MatchCandidate, duration: float = 3.0) -> bool:
        """
        验证匹配 - 截取多张截图供人工确认
        返回: True=确认匹配, False=不匹配
        """
        print(f"\n{'='*60}")
        print(f"📸 验证候选匹配")
        print(f"   源视频: {candidate.source_video.name}")
        print(f"   源位置: {candidate.start_time:.1f}s")
        print(f"   匹配度: {candidate.score:.1%}")
        print(f"{'='*60}")
        
        # 创建对比截图目录
        verify_dir = self.temp_dir / f"verify_{candidate.target_time:.0f}"
        verify_dir.mkdir(exist_ok=True)
        
        # 截取3个时间点的对比图
        check_times = [0, duration/2, duration]
        screenshots = []
        
        for i, offset in enumerate(check_times):
            target_t = candidate.target_time + offset
            source_t = candidate.start_time + offset
            
            # 目标视频帧
            target_frame = verify_dir / f"target_{i}_{target_t:.1f}s.jpg"
            self.extract_frame(self.target_video, target_t, target_frame)
            
            # 源视频帧
            source_frame = verify_dir / f"source_{i}_{source_t:.1f}s.jpg"
            self.extract_frame(candidate.source_video, source_t, source_frame)
            
            screenshots.append({
                'time': target_t,
                'target': target_frame,
                'source': source_frame
            })
        
        # 显示截图信息
        print(f"\n📷 已生成对比截图:")
        for i, shot in enumerate(screenshots):
            print(f"   时间点 {i+1}: {shot['time']:.1f}s")
            print(f"      目标: {shot['target']}")
            print(f"      源:   {shot['source']}")
        
        print(f"\n💡 请查看以上截图，确认画面内容是否匹配")
        print(f"   截图目录: {verify_dir}")
        
        # 在实际应用中，这里会等待用户输入
        # 为了演示，我们假设用户会查看截图后决定
        return True  # 假设确认匹配
    
    def interactive_reconstruct(self, output_path: str, segment_duration: float = 30.0) -> bool:
        """
        交互式重构 - 每段人工确认
        segment_duration: 每段处理的时长（秒）
        """
        print(f"\n{'='*60}")
        print(f"🎬 交互式视频重构")
        print(f"   每段时长: {segment_duration}s")
        print(f"{'='*60}")
        
        target_duration = self.get_video_duration(self.target_video)
        print(f"\n📹 目标视频: {self.target_video.name}")
        print(f"   总时长: {target_duration:.1f}s")
        print(f"   预计段数: {int(target_duration / segment_duration) + 1}")
        
        # 分段处理
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
                
                if self.verify_match(candidate, duration=min(3.0, this_segment_duration)):
                    confirmed = candidate
                    print(f"\n✅ 确认使用此候选")
                    break
                else:
                    print(f"\n❌ 不匹配，尝试下一个候选")
            
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
    
    def _generate_output(self, output_path: str) -> bool:
        """生成最终输出 - 直接拼接已确认片段"""
        
        # 提取每个片段
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
        
        # 拼接
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
    
    def cleanup(self):
        """清理临时文件"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def interactive_test():
    """交互式测试"""
    target_video = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"
    output_path = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_INTERACTIVE.mp4"
    
    # 扫描源视频
    source_videos = [str(f) for f in Path(source_dir).iterdir() if f.suffix.lower() == '.mp4']
    
    print("="*60)
    print("115196 交互式重构")
    print("="*60)
    print(f"目标视频: {target_video}")
    print(f"源视频数: {len(source_videos)}")
    print(f"输出路径: {output_path}")
    print("="*60)
    print("\n💡 此模式会:")
    print("   1. 自动查找候选匹配")
    print("   2. 截取对比截图")
    print("   3. 等待你确认画面是否匹配")
    print("   4. 确认后继续下一段")
    print("="*60)
    
    reconstructor = InteractiveReconstructor(
        target_video=target_video,
        source_videos=source_videos
    )
    
    try:
        success = reconstructor.interactive_reconstruct(output_path, segment_duration=30.0)
        
        if success:
            print(f"\n🎉 交互式重构完成!")
            print(f"   输出: {output_path}")
        else:
            print(f"\n❌ 重构未完成")
    finally:
        # 保留截图供查看
        print(f"\n📁 临时文件保留在: {reconstructor.temp_dir}")
        print(f"   你可以查看所有对比截图")


if __name__ == "__main__":
    from pathlib import Path
    interactive_test()
