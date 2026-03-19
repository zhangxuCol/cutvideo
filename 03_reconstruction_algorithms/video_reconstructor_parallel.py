#!/usr/bin/env python3
"""
视频混剪重构工具 - 多线程并行版
支持多源视频并行处理
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import json
import yaml
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
from queue import Queue

@dataclass
class VideoSegment:
    """视频片段信息"""
    source_video: Path
    start_time: float
    end_time: float
    similarity_score: float
    segment_idx: int

@dataclass
class FrameMatch:
    """帧匹配结果"""
    target_frame_idx: int
    source_video: Path
    source_frame_idx: int
    similarity: float
    timestamp: float

class VideoReconstructor:
    def __init__(self, target_video: str, source_videos: List[str], config: dict = None):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.config = config or {}
        self.temp_dir = None
        self.max_workers = config.get('max_workers', cpu_count())
        
    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def extract_frames(self, video_path: Path, output_dir: Path, fps: int = 2) -> List[Tuple[Path, float]]:
        """提取视频帧，返回(帧路径, 时间戳)列表"""
        output_pattern = output_dir / f"{video_path.stem}_%06d.jpg"
        scale = self.config.get('scale', '480:270')
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-i', str(video_path),
            '-vf', f'fps={fps},scale={scale}',
            '-q:v', '2',
            str(output_pattern)
        ]
        subprocess.run(cmd, capture_output=True)
        
        frames = []
        frame_files = sorted(output_dir.glob(f"{video_path.stem}_*.jpg"))
        for idx, frame_path in enumerate(frame_files):
            timestamp = idx / fps
            frames.append((frame_path, timestamp))
        
        return frames
    
    def calculate_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        """计算两帧的相似度"""
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 直方图相似度
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        # 综合评分
        return 0.5 * max(0, hist_sim) + 0.5 * template_sim
    
    def find_best_match_for_frame(self, target_frame_data: Tuple[int, Path, float], 
                                   source_frames_dict: Dict[Path, List[Tuple[Path, float]]]) -> Optional[FrameMatch]:
        """为单帧找到最佳匹配（用于并行处理）"""
        target_idx, target_frame, target_ts = target_frame_data
        match_threshold = self.config.get('match_threshold', 0.5)
        
        best_match = None
        best_score = 0.0
        
        for source_video, source_frames in source_frames_dict.items():
            for source_idx, (source_frame, source_ts) in enumerate(source_frames):
                similarity = self.calculate_similarity(target_frame, source_frame)
                
                if similarity > best_score and similarity > match_threshold:
                    best_score = similarity
                    best_match = FrameMatch(
                        target_frame_idx=target_idx,
                        source_video=source_video,
                        source_frame_idx=source_idx,
                        similarity=best_score,
                        timestamp=source_ts
                    )
        
        return best_match
    
    def find_matches_parallel(self, target_frames: List[Tuple[Path, float]], 
                              source_frames_dict: Dict[Path, List[Tuple[Path, float]]]) -> List[Optional[FrameMatch]]:
        """并行帧匹配"""
        print(f"🔍 正在并行比对 {len(target_frames)} 帧目标画面...")
        print(f"   使用 {self.max_workers} 个线程")
        
        matches = [None] * len(target_frames)
        target_frame_data = [(idx, frame, ts) for idx, (frame, ts) in enumerate(target_frames)]
        
        completed = 0
        total = len(target_frames)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.find_best_match_for_frame, frame_data, source_frames_dict): frame_data[0]
                for frame_data in target_frame_data
            }
            
            # 收集结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    match = future.result()
                    matches[idx] = match
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"   处理中: {completed}/{total} 帧 ({completed/total*100:.1f}%)")
                except Exception as e:
                    print(f"   处理帧 {idx} 时出错: {e}")
                    matches[idx] = None
        
        return matches
    
    def find_matches_threaded(self, target_frames: List[Tuple[Path, float]], 
                              source_frames_dict: Dict[Path, List[Tuple[Path, float]]]) -> List[Optional[FrameMatch]]:
        """使用线程池的帧匹配（适用于I/O密集型，但这里是CPU密集型）"""
        print(f"🔍 正在多线程比对 {len(target_frames)} 帧目标画面...")
        print(f"   使用 {self.max_workers} 个线程")
        
        matches = [None] * len(target_frames)
        match_threshold = self.config.get('match_threshold', 0.5)
        completed = 0
        lock = threading.Lock()
        
        def process_frame(target_idx, target_frame, target_ts):
            nonlocal completed
            
            best_match = None
            best_score = 0.0
            
            for source_video, source_frames in source_frames_dict.items():
                for source_idx, (source_frame, source_ts) in enumerate(source_frames):
                    similarity = self.calculate_similarity(target_frame, source_frame)
                    
                    if similarity > best_score and similarity > match_threshold:
                        best_score = similarity
                        best_match = FrameMatch(
                            target_frame_idx=target_idx,
                            source_video=source_video,
                            source_frame_idx=source_idx,
                            similarity=best_score,
                            timestamp=source_ts
                        )
            
            matches[target_idx] = best_match
            
            with lock:
                completed += 1
                if completed % 10 == 0 or completed == len(target_frames):
                    print(f"   处理中: {completed}/{len(target_frames)} 帧 ({completed/len(target_frames)*100:.1f}%)")
        
        # 使用线程池
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(lambda x: process_frame(x[0], x[1], x[2]), 
                        [(idx, frame, ts) for idx, (frame, ts) in enumerate(target_frames)])
        
        return matches
    
    def segment_matches(self, matches: List[Optional[FrameMatch]]) -> List[VideoSegment]:
        """将连续的匹配帧聚合成片段"""
        segments = []
        current_segment = []
        
        fps = self.config.get('fps', 2)
        min_duration = self.config.get('min_segment_duration', 0.5)
        
        for match in matches:
            if match is None:
                if len(current_segment) >= min_duration * fps:
                    self._save_segment(current_segment, segments)
                current_segment = []
            else:
                current_segment.append(match)
        
        if len(current_segment) >= min_duration * fps:
            self._save_segment(current_segment, segments)
        
        return segments
    
    def _save_segment(self, segment_matches: List[FrameMatch], segments: List[VideoSegment]):
        """保存一个片段"""
        if not segment_matches:
            return
        
        first_match = segment_matches[0]
        last_match = segment_matches[-1]
        avg_similarity = np.mean([m.similarity for m in segment_matches])
        
        segment = VideoSegment(
            source_video=first_match.source_video,
            start_time=first_match.timestamp,
            end_time=last_match.timestamp + 0.5,
            similarity_score=avg_similarity,
            segment_idx=len(segments)
        )
        segments.append(segment)
    
    def generate_ffmpeg_commands(self, segments: List[VideoSegment], output_path: str, 
                                 use_target_audio: bool = True) -> List[str]:
        """生成FFmpeg命令列表"""
        commands = []
        temp_files = []
        
        # 1. 裁剪每个片段
        for idx, seg in enumerate(segments):
            temp_file = f"temp_segment_{idx:03d}.mp4"
            temp_files.append(temp_file)
            
            cmd = f'ffmpeg -y -hide_banner -ss {seg.start_time:.3f} -to {seg.end_time:.3f} -i "{seg.source_video}" -an -c:v copy "{temp_file}"'
            commands.append(cmd)
        
        # 2. 生成concat列表
        concat_content = "\n".join([f"file '{f}'" for f in temp_files])
        with open("concat_list.txt", "w") as f:
            f.write(concat_content)
        
        # 3. 合并片段
        temp_video = "temp_video_no_audio.mp4"
        concat_cmd = f'ffmpeg -y -hide_banner -f concat -safe 0 -i concat_list.txt -an -c:v copy "{temp_video}"'
        commands.append(concat_cmd)
        temp_files.append(temp_video)
        
        # 4. 处理音频
        if use_target_audio:
            temp_audio = "temp_audio.aac"
            extract_cmd = f'ffmpeg -y -hide_banner -i "{self.target_video}" -vn -c:a aac "{temp_audio}"'
            commands.append(extract_cmd)
            temp_files.append(temp_audio)
            
            final_cmd = f'ffmpeg -y -hide_banner -i "{temp_video}" -i "{temp_audio}" -c:v copy -c:a copy -shortest "{output_path}"'
            commands.append(final_cmd)
        else:
            commands.append(f'mv "{temp_video}" "{output_path}"')
        
        # 5. 清理
        cleanup_cmd = f"rm -f {' '.join(temp_files)} concat_list.txt"
        commands.append(cleanup_cmd)
        
        return commands
    
    def reconstruct(self, output_path: str = "reconstructed.mp4", use_target_audio: bool = True):
        """主流程：重构视频"""
        
        target_duration = self.get_video_duration(self.target_video)
        print(f"🎯 目标视频时长: {target_duration:.2f}秒")
        print(f"🎬 源视频数量: {len(self.source_videos)}个")
        
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        target_frames_dir = self.temp_dir / "target"
        target_frames_dir.mkdir()
        
        source_frames_dirs = {}
        for source_video in self.source_videos:
            source_dir = self.temp_dir / f"source_{source_video.stem}"
            source_dir.mkdir()
            source_frames_dirs[source_video] = source_dir
        
        try:
            fps = self.config.get('fps', 2)
            
            # 提取帧
            print("🎬 提取目标视频帧...")
            target_frames = self.extract_frames(self.target_video, target_frames_dir, fps)
            print(f"   提取了 {len(target_frames)} 帧")
            
            source_frames_dict = {}
            for source_video in self.source_videos:
                print(f"🎬 提取源视频帧: {source_video.name}")
                frames = self.extract_frames(source_video, source_frames_dirs[source_video], fps)
                source_frames_dict[source_video] = frames
                print(f"   提取了 {len(frames)} 帧")
            
            # 并行帧匹配
            use_parallel = self.config.get('use_parallel', True)
            if use_parallel and len(target_frames) > 50:
                matches = self.find_matches_threaded(target_frames, source_frames_dict)
            else:
                matches = self.find_matches_sequential(target_frames, source_frames_dict)
            
            matched_frames = sum(1 for m in matches if m is not None)
            match_rate = matched_frames / len(matches) * 100
            print(f"\n📊 匹配率: {matched_frames}/{len(matches)} 帧 ({match_rate:.1f}%)")
            
            # 聚合成片段
            segments = self.segment_matches(matches)
            print(f"\n📦 找到 {len(segments)} 个连续片段:")
            
            total_duration = 0
            for seg in segments:
                duration = seg.end_time - seg.start_time
                total_duration += duration
                print(f"   片段{seg.segment_idx+1}: {seg.source_video.name}")
                print(f"      时间: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
                print(f"      相似度: {seg.similarity_score:.2%}")
            
            print(f"\n⏱️ 总时长: {total_duration:.2f}s / 目标: {target_duration:.2f}s")
            
            # 生成命令
            commands = self.generate_ffmpeg_commands(segments, output_path, use_target_audio)
            
            # 执行命令
            print("\n" + "=" * 60)
            print("⚙️  正在生成视频...")
            print("=" * 60)
            
            for cmd in commands[:-1]:
                print(f"执行: {cmd[:80]}...")
                result = subprocess.run(cmd, shell=True, capture_output=True)
                if result.returncode != 0:
                    print(f"❌ 命令失败: {result.stderr.decode()[:200]}")
                    return None
            
            # 清理
            subprocess.run(commands[-1], shell=True)
            
            print(f"\n✅ 视频重构完成: {output_path}")
            
            # 保存元数据
            metadata = {
                'target_video': str(self.target_video),
                'source_videos': [str(v) for v in self.source_videos],
                'output_video': output_path,
                'segments': [
                    {
                        'source_video': str(seg.source_video),
                        'source_video_name': seg.source_video.name,
                        'start_time': seg.start_time,
                        'end_time': seg.end_time,
                        'duration': seg.end_time - seg.start_time,
                        'similarity_score': float(seg.similarity_score),
                        'segment_idx': seg.segment_idx
                    }
                    for seg in segments
                ],
                'total_segments': len(segments),
                'total_duration': total_duration,
                'target_duration': target_duration
            }
            
            metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"📝 元数据已保存: {metadata_path}")
            
            return segments
            
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def find_matches_sequential(self, target_frames: List[Tuple[Path, float]], 
                                 source_frames_dict: Dict[Path, List[Tuple[Path, float]]]) -> List[Optional[FrameMatch]]:
        """顺序帧匹配（原始方法）"""
        print(f"🔍 正在顺序比对 {len(target_frames)} 帧目标画面...")
        
        matches = []
        match_threshold = self.config.get('match_threshold', 0.5)
        
        for target_idx, (target_frame, target_ts) in enumerate(target_frames):
            if target_idx % 10 == 0:
                print(f"   处理中: {target_idx}/{len(target_frames)} 帧")
            
            best_match = None
            best_score = 0.0
            
            for source_video, source_frames in source_frames_dict.items():
                for source_idx, (source_frame, source_ts) in enumerate(source_frames):
                    similarity = self.calculate_similarity(target_frame, source_frame)
                    
                    if similarity > best_score and similarity > match_threshold:
                        best_score = similarity
                        best_match = FrameMatch(
                            target_frame_idx=target_idx,
                            source_video=source_video,
                            source_frame_idx=source_idx,
                            similarity=best_score,
                            timestamp=source_ts
                        )
            
            matches.append(best_match)
        
        return matches


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='视频混剪重构工具 - 多线程版')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--target', '-t', help='目标视频路径')
    parser.add_argument('--sources', '-s', nargs='+', help='源视频路径列表')
    parser.add_argument('--output', '-o', help='输出视频路径')
    parser.add_argument('--fps', type=int, help='帧提取率（覆盖配置）')
    parser.add_argument('--threshold', type=float, help='匹配阈值（覆盖配置）')
    parser.add_argument('--workers', '-w', type=int, help='并行线程数（默认CPU核心数）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.fps is not None:
        config['fps'] = args.fps
    if args.threshold is not None:
        config['match_threshold'] = args.threshold
    if args.workers is not None:
        config['max_workers'] = args.workers
    
    # 获取视频路径
    if args.config:
        target_video = config.get('target_video')
        source_videos = config.get('source_videos', [])
        output_video = config.get('output_video', 'reconstructed.mp4')
    else:
        target_video = args.target
        source_videos = args.sources
        output_video = args.output or 'reconstructed.mp4'
    
    if not target_video or not source_videos:
        print("❌ 请提供目标视频和源视频")
        parser.print_help()
        return
    
    # 检查文件存在
    if not Path(target_video).exists():
        print(f"❌ 目标视频不存在: {target_video}")
        return
    
    for sv in source_videos:
        if not Path(sv).exists():
            print(f"❌ 源视频不存在: {sv}")
            return
    
    print("=" * 60)
    print("🎬 视频混剪重构工具 - 多线程版")
    print("=" * 60)
    print(f"目标视频: {target_video}")
    print(f"源视频: {source_videos}")
    print(f"输出视频: {output_video}")
    print(f"配置: {config}")
    print("=" * 60)
    
    # 执行重构
    reconstructor = VideoReconstructor(target_video, source_videos, config)
    segments = reconstructor.reconstruct(output_video, use_target_audio=True)
    
    if segments:
        print("\n✅ 重构成功!")
    else:
        print("\n❌ 重构失败")


if __name__ == "__main__":
    main()
