#!/usr/bin/env python3
"""
极速视频裁剪方案 - 3分钟目标
针对115196视频的极速重构方案
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
import time  # 添加这行

@dataclass
class VideoSegment:
    source_video: Path
    start_time: float
    end_time: float
    similarity_score: float

class FastVideoReconstructor:
    """
    极速重构器 - 3分钟目标
    策略：
    1. 仅使用音频指纹匹配（跳过视频帧验证）
    2. 降低采样率（fps从5降到2）
    3. 快速单源匹配（不尝试多源）
    4. 直接裁剪，不重新编码
    """
    
    def __init__(self, target_video: str, source_videos: List[str]):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.temp_dir = None
        
        # 极速配置
        self.fps = 2  # 从5降到2
        self.similarity_threshold = 0.80  # 从0.85降到0.80，更容易单源匹配

    def get_video_duration(self, video_path: Path) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def _extract_audio_fingerprint_fast(self, video_path: Path) -> np.ndarray:
        """极速音频指纹 - 低采样率"""
        temp_wav = self.temp_dir / f"{video_path.stem}_audio.wav"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(video_path), '-vn',
            '-acodec', 'pcm_s16le', '-ar', '4000', '-ac', '1',  # 从8000降到4000
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
        
        # 只取一半的采样点
        for i in range(0, min(n_blocks, 250), 2):  # 步长2，跳过一半
            block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
            fft = np.fft.rfft(block)
            magnitude = np.abs(fft)
            bands = np.array([np.mean(magnitude[j:j+len(magnitude)//10]) 
                            for j in range(0, len(magnitude), len(magnitude)//10)])
            features.append(bands[:10])
        
        return np.array(features)

    def _find_audio_match_fast(self, target_fp: np.ndarray, source_video: Path) -> Tuple[float, float]:
        """极速音频匹配 - 大步长搜索"""
        source_fp = self._extract_audio_fingerprint_fast(source_video)
        
        if len(target_fp) == 0 or len(source_fp) == 0 or len(target_fp) > len(source_fp):
            return 0, 0
        
        best_score = -1
        best_start = 0
        
        # 大步长搜索（每隔5秒）
        for start in range(0, len(source_fp) - len(target_fp) + 1, 5):
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

    def find_single_source(self, target_duration: float):
        """极速单源查找"""
        print(f"\n🔍 极速查找单源匹配...")
        
        target_fp = self._extract_audio_fingerprint_fast(self.target_video)
        
        best_result = None
        best_score = 0
        
        for source in self.source_videos:
            source_duration = self.get_video_duration(source)
            if source_duration < target_duration:
                continue
            
            score, start_time = self._find_audio_match_fast(target_fp, source)
            
            if score > best_score:
                best_score = score
                best_result = {
                    'source': source,
                    'start_time': start_time,
                    'score': score
                }
                
                print(f"   {source.name}: {score:.1%} @ {start_time:.1f}s")
                
                # 如果找到好的匹配，提前结束
                if score > 0.90:
                    print(f"   ✅ 找到高匹配度，提前结束搜索")
                    break
        
        return best_result

    def reconstruct_fast(self, output_path: str) -> bool:
        """极速重构 - 3分钟目标"""
        print(f"\n{'='*60}")
        print(f"🚀 极速重构模式")
        print(f"   目标: 3分钟内完成")
        print(f"{'='*60}")
        
        self.temp_dir = Path(tempfile.mkdtemp())
        start_total = time.time()
        
        try:
            target_duration = self.get_video_duration(self.target_video)
            print(f"\n📹 目标: {self.target_video.name} ({target_duration:.1f}s)")
            
            # 阶段1: 极速单源查找
            match = self.find_single_source(target_duration)
            
            if match and match['score'] > self.similarity_threshold:
                print(f"\n✅ 单源匹配成功!")
                print(f"   来源: {match['source'].name}")
                print(f"   位置: @{match['start_time']:.1f}s")
                print(f"   匹配度: {match['score']:.1%}")
                
                # 阶段2: 直接裁剪（不重新编码）
                print(f"\n✂️  直接裁剪（不重新编码）...")
                
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-ss', str(match['start_time']),
                    '-t', str(target_duration),
                    '-i', str(match['source']),
                    '-c', 'copy',  # 不重新编码，直接复制
                    str(output_path)
                ]
                subprocess.run(cmd, capture_output=True)
                
                if Path(output_path).exists():
                    elapsed = time.time() - start_total
                    output_duration = self.get_video_duration(output_path)
                    print(f"\n{'='*60}")
                    print(f"✅ 极速重构完成!")
                    print(f"   耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
                    print(f"   输出: {output_path}")
                    print(f"   时长: {output_duration:.1f}s")
                    print(f"{'='*60}")
                    return True
                else:
                    print(f"   ❌ 生成失败")
                    return False
            else:
                print(f"\n⚠️ 未找到足够匹配的单源")
                if match:
                    print(f"   最佳匹配: {match['score']:.1%} (需要>{self.similarity_threshold:.0%})")
                print(f"   建议: 降低阈值或使用完整版V6")
                return False
        
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)


def test_115196_fast():
    """测试115196极速重构"""
    import time
    import sys
    
    target_video = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"
    output_path = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196-1-363935819124715523_reconstructed_FAST.mp4"
    
    # 扫描源视频（只取前10个，加快速度）
    from pathlib import Path
    source_videos = [str(f) for f in Path(source_dir).iterdir() if f.suffix.lower() == '.mp4'][:10]
    
    print("="*60)
    print("115196 极速重构测试 (3分钟目标)")
    print("="*60)
    print(f"目标视频: {target_video}")
    print(f"源视频数: {len(source_videos)} (限制前10个)")
    print(f"输出路径: {output_path}")
    print("="*60)
    
    reconstructor = FastVideoReconstructor(
        target_video=target_video,
        source_videos=source_videos
    )
    
    success = reconstructor.reconstruct_fast(output_path)
    
    if success:
        # 验证结果
        print(f"\n📊 验证结果:")
        
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
        
        diff = abs(output_duration - target_duration)
        print(f"   原始时长: {target_duration:.2f}s")
        print(f"   输出时长: {output_duration:.2f}s")
        print(f"   差异: {diff:.2f}s")
        
        if diff < 1.0:
            print(f"   ✅ 时长对齐良好")
        else:
            print(f"   ⚠️ 时长有差异，但可能在可接受范围")
    else:
        print(f"\n❌ 极速重构失败")
        print(f"   建议: 运行完整版V6或检查源视频")


if __name__ == "__main__":
    test_115196_fast()
