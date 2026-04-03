#!/usr/bin/env python3
"""
极速高精度重构器 V4 - pHash + 预建索引优化
优化策略：
1. 预计算所有源视频的帧 pHash 索引（离线）
2. 查询时用 pHash 快速筛选候选
3. 对候选做精细验证
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
import pickle
import re
from PIL import Image
import imagehash

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

class FastHighPrecisionReconstructorV4:
    """
    极速高精度重构器 V4 - pHash 优化版
    """
    
    def __init__(self, target_video: str, source_videos: List[str], cache_dir: str = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = Path(cache_dir) if cache_dir else self.temp_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # 配置
        self.match_threshold = 0.95
        self.segment_duration = 5.0
        self.max_workers = 4
        
        # pHash 索引
        self.frame_index = {}  # {video_path: [(time, phash), ...]}
        
    def get_video_duration(self, video_path: Path) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
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
            img = Image.open(temp_frame)
            temp_frame.unlink()
            return img
        return None
    
    def compute_phash(self, img: Image.Image) -> imagehash.ImageHash:
        """计算感知哈希"""
        return imagehash.phash(img, hash_size=8)
    
    def build_frame_index(self, sample_interval: float = 1.0) -> Dict[Path, List[Tuple[float, imagehash.ImageHash]]]:
        """
        预建帧索引：提取所有源视频的帧 pHash
        sample_interval: 采样间隔（秒），默认每1秒取1帧
        """
        index_file = self.cache_dir / "frame_index_v4.pkl"
        
        # 检查缓存
        if index_file.exists():
            print(f"\n📂 加载已有帧索引: {index_file}")
            with open(index_file, 'rb') as f:
                self.frame_index = pickle.load(f)
            total_frames = sum(len(frames) for frames in self.frame_index.values())
            print(f"   ✅ 已索引 {len(self.frame_index)} 个视频，共 {total_frames} 帧")
            return self.frame_index
        
        print(f"\n🔨 构建帧索引 (采样间隔: {sample_interval}s)...")
        
        for i, video_path in enumerate(self.source_videos):
            print(f"   [{i+1}/{len(self.source_videos)}] {video_path.name}")
            
            duration = self.get_video_duration(video_path)
            frames = []
            
            # 每 sample_interval 秒取1帧
            for t in np.arange(0, duration, sample_interval):
                img = self.extract_frame_to_pil(video_path, t)
                if img:
                    phash = self.compute_phash(img)
                    frames.append((t, phash))
            
            self.frame_index[video_path] = frames
            print(f"      提取了 {len(frames)} 帧")
        
        # 保存索引
        with open(index_file, 'wb') as f:
            pickle.dump(self.frame_index, f)
        
        total_frames = sum(len(frames) for frames in self.frame_index.values())
        print(f"\n✅ 索引构建完成: {len(self.frame_index)} 个视频，共 {total_frames} 帧")
        print(f"   缓存文件: {index_file}")
        
        return self.frame_index
    
    def find_match_by_phash(self, target_start: float, duration: float, 
                            seg_index: int = 0, top_k: int = 10) -> List[Tuple[Path, float, float]]:
        """
        使用 pHash 快速查找匹配候选
        返回: [(source, start_time, similarity), ...]
        """
        # 提取目标帧的 pHash
        target_img = self.extract_frame_to_pil(self.target_video, target_start + duration * 0.5)
        if not target_img:
            return []
        
        target_phash = self.compute_phash(target_img)
        
        # 在所有源视频中搜索
        candidates = []
        
        for video_path, frames in self.frame_index.items():
            for time_sec, phash in frames:
                # 计算 pHash 距离（汉明距离）
                distance = target_phash - phash
                
                # 距离越小越相似，阈值从 12 放宽到 18
                if distance <= 18:
                    # 计算相似度分数（0-1）
                    similarity = 1.0 - (distance / 64.0)
                    candidates.append((video_path, time_sec, similarity))
        
        # 按相似度排序，取前 top_k（从 5 增加到 10）
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]
    
    def verify_match_by_visual(self, target_start: float, source: Path, 
                               source_start: float, duration: float) -> float:
        """用精细画面验证匹配质量"""
        # 提取目标帧
        target_frame = self.temp_dir / f"verify_target_{int(target_start*100)}.jpg"
        self.extract_frame(self.target_video, target_start + duration * 0.5, target_frame)
        
        # 提取源帧
        source_frame = self.temp_dir / f"verify_source_{int(source_start*100)}.jpg"
        self.extract_frame(source, source_start + duration * 0.5, source_frame)
        
        if target_frame.exists() and source_frame.exists():
            sim = self.calculate_frame_similarity(target_frame, source_frame)
            target_frame.unlink()
            source_frame.unlink()
            return sim
        
        return 0.0
    
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
        """处理单个段 - pHash 快速匹配 + 精细验证"""
        
        # 第一步：pHash 快速筛选候选
        candidates = self.find_match_by_phash(
            task.target_start, task.duration, task.index, top_k=5
        )
        
        if not candidates:
            return SegmentResult(index=task.index, success=False)
        
        # 第二步：精细验证候选
        best_source = None
        best_start = 0
        best_score = 0.0
        
        for source, start_time, phash_sim in candidates:
            # 精细画面验证
            visual_sim = self.verify_match_by_visual(
                task.target_start, source, start_time, task.duration
            )
            
            # 综合评分
            combined_score = 0.4 * phash_sim + 0.6 * visual_sim
            
            if combined_score > best_score:
                best_score = combined_score
                best_start = start_time
                best_source = source
        
        if best_score < 0.70:
            print(f"   段 {task.index}/44 ⚠️ 匹配分数不足 ({best_score:.2f})")
            return SegmentResult(index=task.index, success=False)
        
        print(f"   段 {task.index}/44 ✅ {best_source.name} @ {best_start}s (综合: {best_score:.2f})")
        
        return SegmentResult(
            index=task.index,
            success=True,
            source=best_source,
            source_start=best_start,
            quality={'combined': best_score}
        )
    
    def reconstruct_fast(self, output_path: str) -> bool:
        """极速重构"""
        import time
        
        print(f"\n{'='*70}")
        print(f"🚀 极速高精度重构 V4 - pHash 优化版")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # 预建帧索引（采样间隔从 0.5s 改为 0.25s，即每秒4帧）
        self.build_frame_index(sample_interval=0.25)
        
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
        """生成输出 - 视频从源视频提取（向前回退+去重），音频从目标视频提取"""
        
        video_clips = []
        audio_clips = []
        
        # 第一步：处理视频片段（向前回退2秒）
        print(f"   处理视频片段（向前回退2秒）...")
        for i, seg in enumerate(segments):
            seg_source = seg['source']
            seg_start = seg['start']
            target_start = seg['target_start']
            
            # 向前回退2秒，确保包含场景开始
            adjusted_start = max(0, seg_start - 2.0)
            adjusted_duration = seg['duration'] + 2.0
            
            # 提取视频片段（从源视频，使用回退后的时间戳）
            video_clip = self.temp_dir / f"seg_{seg['index']:03d}_v.mp4"
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(adjusted_start),
                '-t', str(adjusted_duration),
                '-i', str(seg_source),
                '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                str(video_clip)
            ]
            subprocess.run(cmd, capture_output=True)
            
            # 提取音频片段（从目标视频，保持原始时间轴）
            audio_clip = self.temp_dir / f"seg_{seg['index']:03d}_a.aac"
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(target_start),  # 使用目标视频的时间戳
                '-t', str(seg['duration']),
                '-i', str(self.target_video),  # 从目标视频提取
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
        
        # 第二步：对视频片段进行去重检测
        print(f"   检测并去除重复内容...")
        unique_video_clips = self._deduplicate_clips(video_clips)
        print(f"   去重后: {len(unique_video_clips)}/{len(video_clips)} 个片段")
        
        # 第三步：拼接视频
        video_concat = self.temp_dir / "video_concat.txt"
        with open(video_concat, 'w') as f:
            for clip in unique_video_clips:
                f.write(f"file '{clip}'\n")
        
        temp_video = self.temp_dir / "temp_video.mp4"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(video_concat),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            str(temp_video)
        ]
        subprocess.run(cmd, capture_output=True)
        
        # 第四步：拼接音频（保持与去重后视频片段对应）
        # 注意：音频也需要根据去重结果调整
        unique_audio_clips = self._get_corresponding_audio_clips(audio_clips, video_clips, unique_video_clips)
        
        audio_concat = self.temp_dir / "audio_concat.txt"
        with open(audio_concat, 'w') as f:
            for clip in unique_audio_clips:
                f.write(f"file '{clip}'\n")
        
        temp_audio = self.temp_dir / "temp_audio.aac"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(audio_concat),
            '-c:a', 'aac', '-b:a', '128k',
            str(temp_audio)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_video.exists() or not temp_audio.exists():
            print("❌ 音视频拼接失败")
            return False
        
        # 第五步：合并音视频
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(temp_video),
            '-i', str(temp_audio),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-shortest',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        
        return Path(output_path).exists()
    
    def _deduplicate_clips(self, clips: List[Path]) -> List[Path]:
        """检测并去除重复的视频片段"""
        if len(clips) <= 1:
            return clips
        
        unique_clips = [clips[0]]  # 保留第一个
        
        for i in range(1, len(clips)):
            current_clip = clips[i]
            prev_clip = unique_clips[-1]
            
            # 提取两个片段的中间帧进行比对
            if self._are_clips_similar(prev_clip, current_clip):
                print(f"   [去重] 跳过重复片段: {current_clip.name}")
                continue
            
            unique_clips.append(current_clip)
        
        return unique_clips
    
    def _are_clips_similar(self, clip1: Path, clip2: Path, threshold: float = 0.95) -> bool:
        """判断两个视频片段是否相似"""
        # 提取两个片段的中间帧
        frame1 = self.temp_dir / f"compare_{clip1.stem}.jpg"
        frame2 = self.temp_dir / f"compare_{clip2.stem}.jpg"
        
        # 获取视频时长
        duration1 = self._get_video_duration(clip1)
        duration2 = self._get_video_duration(clip2)
        
        # 提取中间帧
        cmd1 = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(duration1 * 0.5), '-i', str(clip1),
                '-vframes', '1', '-vf', 'scale=160:90', str(frame1)]
        cmd2 = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(duration2 * 0.5), '-i', str(clip2),
                '-vframes', '1', '-vf', 'scale=160:90', str(frame2)]
        
        subprocess.run(cmd1, capture_output=True)
        subprocess.run(cmd2, capture_output=True)
        
        if not frame1.exists() or not frame2.exists():
            return False
        
        # 计算相似度
        similarity = self.calculate_frame_similarity(frame1, frame2)
        
        # 清理临时帧
        frame1.unlink()
        frame2.unlink()
        
        return similarity >= threshold
    
    def _get_video_duration(self, video_path: Path) -> float:
        """获取视频时长"""
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _get_corresponding_audio_clips(self, audio_clips: List[Path], 
                                       video_clips: List[Path],
                                       unique_video_clips: List[Path]) -> List[Path]:
        """根据去重后的视频片段，获取对应的音频片段"""
        unique_audio_clips = []
        for v_clip in unique_video_clips:
            # 找到对应的音频片段（索引相同）
            try:
                idx = video_clips.index(v_clip)
                unique_audio_clips.append(audio_clips[idx])
            except ValueError:
                pass
        return unique_audio_clips


def main():
    target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    source_dir = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集"
    output = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_V4_PHASH.mp4"
    cache = "/Users/zhangxu/work/项目/cutvideo/cache_v4"
    
    source_videos = [str(f) for f in Path(source_dir).iterdir() if f.suffix.lower() == '.mp4']
    
    print("="*70)
    print("115196 极速高精度重构 V4 - pHash 优化版")
    print("="*70)
    
    Path(cache).mkdir(exist_ok=True)
    
    reconstructor = FastHighPrecisionReconstructorV4(target, source_videos, cache)
    
    try:
        success = reconstructor.reconstruct_fast(output)
        
        if success:
            print("\n🎉 极速重构完成!")
        else:
            print("\n❌ 重构失败")
    finally:
        print(f"\n📁 临时文件: {reconstructor.temp_dir}")


if __name__ == "__main__":
    from pathlib import Path
    main()
