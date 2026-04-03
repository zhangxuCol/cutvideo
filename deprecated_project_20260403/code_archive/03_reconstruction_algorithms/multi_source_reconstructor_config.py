#!/usr/bin/env python3
"""
多源视频混剪重构工具（配置文件版）
从多个素材视频中重构出与目标视频相似的新版本
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import json
import yaml
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

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

@dataclass
class ValidationResult:
    """校验结果"""
    overall_similarity: float
    frame_by_frame_similarity: List[float]
    duration_match: bool
    duration_diff: float
    passed: bool

class Config:
    """配置管理"""
    def __init__(self, config_file: str = "video_reconstruct_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """加载配置文件"""
        default_config = {
            'target_video': '',  # 裁剪视频1路径
            'source_videos': [],  # 原视频路径列表
            'output_video': 'reconstructed.mp4',  # 裁剪视频2输出路径
            'fps': 8,  # 每秒提取帧数（1-10，提高到8）
            'similarity_threshold': 0.85,  # 相似度阈值（0-1，降低到0.85）
            'max_retries': 3,  # 最大重试次数
            'use_target_audio': True,  # 是否使用目标视频音频
            'min_segment_duration': 0.3,  # 最小片段时长（秒，降低到0.3）
            'match_threshold': 0.45,  # 单帧匹配阈值（降低到0.45）
            'scale': '480:270',  # 提取帧分辨率
            'missing_segment_strategy': 'smart_fill',  # 缺失片段处理策略: strict/fill_black/smart_fill
            'smart_fill_threshold': 0.4,  # 智能填充的最低相似度阈值（降低到0.4）
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config
    
    def save_config(self):
        """保存配置文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
    
    def get(self, key: str, default=None):
        """获取配置项"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """设置配置项"""
        self.config[key] = value


class MultiSourceVideoReconstructor:
    def __init__(self, config: Config):
        self.config = config
        self.target_video = Path(config.get('target_video'))
        self.source_videos = [Path(v) for v in config.get('source_videos', [])]
        self.output_video = Path(config.get('output_video', 'reconstructed.mp4'))
        self.fps = config.get('fps', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.90)
        self.max_retries = config.get('max_retries', 3)
        self.use_target_audio = config.get('use_target_audio', True)
        self.min_segment_duration = config.get('min_segment_duration', 0.5)
        self.match_threshold = config.get('match_threshold', 0.6)
        self.scale = config.get('scale', '480:270')
        self.missing_strategy = config.get('missing_segment_strategy', 'smart_fill')
        self.smart_fill_threshold = config.get('smart_fill_threshold', 0.5)
        
        self.temp_dir = None
        self.target_duration = 0
        self.reconstruction_log = []  # 重构日志
        
    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）- 使用OpenCV"""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    
    def extract_frames_with_timestamp(self, video_path: Path, output_dir: Path) -> List[Tuple[Path, float]]:
        """提取视频帧，返回(帧路径, 时间戳)列表 - 使用OpenCV"""
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算要提取的帧间隔
        frame_interval = int(fps / self.fps) if fps > 0 else 1
        if frame_interval < 1:
            frame_interval = 1
        
        frames = []
        frame_idx = 0
        extracted_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔提取帧
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps if fps > 0 else 0
                frame_path = output_dir / f"{video_path.stem}_{extracted_idx:06d}.jpg"
                
                # 调整大小
                frame_resized = cv2.resize(frame, (480, 270))
                cv2.imwrite(str(frame_path), frame_resized)
                
                frames.append((frame_path, timestamp))
                extracted_idx += 1
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def calculate_frame_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        """计算两帧的相似度（多种算法结合）"""
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 方法1: 直方图相似度
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 方法2: 模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        # 方法3: 感知哈希（pHash）
        phash_sim = self._phash_similarity(gray1, gray2)
        
        # 综合评分（加权平均）
        final_sim = 0.4 * max(0, hist_sim) + 0.3 * template_sim + 0.3 * phash_sim
        
        return final_sim
    
    def _phash_similarity(self, img1, img2) -> float:
        """感知哈希相似度"""
        size = (32, 32)
        img1_small = cv2.resize(img1, size)
        img2_small = cv2.resize(img2, size)
        
        img1_float = np.float32(img1_small)
        img2_float = np.float32(img2_small)
        
        dct1 = cv2.dct(img1_float)
        dct2 = cv2.dct(img2_float)
        
        low_freq1 = dct1[:8, :8].flatten()
        low_freq2 = dct2[:8, :8].flatten()
        
        mean1 = np.mean(low_freq1)
        mean2 = np.mean(low_freq2)
        
        hash1 = (low_freq1 > mean1).astype(int)
        hash2 = (low_freq2 > mean2).astype(int)
        
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1 - (hamming_dist / 64.0)
        
        return max(0, similarity)
    
    def find_best_matches(self, target_frames: List[Tuple[Path, float]], 
                         source_frames_dict: dict) -> List[FrameMatch]:
        """为目标视频的每一帧找到最佳匹配"""
        matches = []
        
        print(f"🔍 正在比对 {len(target_frames)} 帧目标画面...")
        
        for target_idx, (target_frame, target_ts) in enumerate(target_frames):
            if target_idx % (self.fps * 2) == 0:  # 每2秒更新一次进度
                print(f"   处理中: {target_idx}/{len(target_frames)} 帧 ({target_idx/len(target_frames)*100:.1f}%)")
            
            best_match = None
            best_score = 0.0
            
            # 在所有源视频中找最佳匹配
            for source_video, source_frames in source_frames_dict.items():
                for source_idx, (source_frame, source_ts) in enumerate(source_frames):
                    similarity = self.calculate_frame_similarity(target_frame, source_frame)
                    
                    if similarity > best_score and similarity > self.match_threshold:
                        best_score = similarity
                        best_match = FrameMatch(
                            target_frame_idx=target_idx,
                            source_video=source_video,
                            source_frame_idx=source_idx,
                            similarity=best_score,
                            timestamp=source_ts
                        )
            
            if best_match:
                matches.append(best_match)
            else:
                matches.append(None)
        
        return matches
    
    def segment_matches(self, matches: List[FrameMatch]) -> List[VideoSegment]:
        """将连续的匹配帧聚合成片段"""
        segments = []
        current_segment = []
        
        for idx, match in enumerate(matches):
            if match is None:
                # 当前匹配结束，保存之前的片段
                if len(current_segment) >= int(self.min_segment_duration * self.fps):
                    self._save_segment(current_segment, segments)
                current_segment = []
            else:
                current_segment.append(match)
        
        # 保存最后一个片段
        if len(current_segment) >= int(self.min_segment_duration * self.fps):
            self._save_segment(current_segment, segments)
        
        return segments
    
    def _save_segment(self, segment_matches: List[FrameMatch], segments: List[VideoSegment]):
        """保存一个片段"""
        if not segment_matches:
            return
        
        first_match = segment_matches[0]
        last_match = segment_matches[-1]
        
        # 计算平均相似度
        avg_similarity = np.mean([m.similarity for m in segment_matches])
        
        segment = VideoSegment(
            source_video=first_match.source_video,
            start_time=first_match.timestamp,
            end_time=last_match.timestamp + (1.0 / self.fps),
            similarity_score=avg_similarity,
            segment_idx=len(segments)
        )
        
        segments.append(segment)
    
    def analyze_target_structure(self, target_frames: List[Tuple[Path, float]], 
                                matches: List[FrameMatch]) -> List[dict]:
        """分析目标视频的结构，识别预期的片段数量和位置"""
        # 根据None的位置识别片段边界
        segments_info = []
        current_segment_start = 0
        
        for idx, match in enumerate(matches):
            if match is None:
                # 发现边界
                if idx > current_segment_start:
                    segment_duration = (idx - current_segment_start) / self.fps
                    if segment_duration >= self.min_segment_duration:
                        segments_info.append({
                            'start_frame': current_segment_start,
                            'end_frame': idx - 1,
                            'start_time': current_segment_start / self.fps,
                            'end_time': idx / self.fps,
                            'duration': segment_duration,
                            'matched': True
                        })
                current_segment_start = idx + 1
        
        # 处理最后一个片段
        if current_segment_start < len(matches):
            segment_duration = (len(matches) - current_segment_start) / self.fps
            if segment_duration >= self.min_segment_duration:
                segments_info.append({
                    'start_frame': current_segment_start,
                    'end_frame': len(matches) - 1,
                    'start_time': current_segment_start / self.fps,
                    'end_time': len(matches) / self.fps,
                    'duration': segment_duration,
                    'matched': True
                })
        
        return segments_info
    
    def find_alternative_segments(self, target_segments: List[dict],
                                 source_frames_dict: dict) -> List[VideoSegment]:
        """为未匹配的片段寻找替代方案（智能填充）"""
        final_segments = []
        
        print("\n" + "="*60)
        print("📊 片段匹配分析报告")
        print("="*60)
        
        for idx, seg_info in enumerate(target_segments):
            print(f"\n【片段 {idx+1}/{len(target_segments)}】")
            print(f"   目标位置: {seg_info['start_time']:.2f}s - {seg_info['end_time']:.2f}s")
            print(f"   目标时长: {seg_info['duration']:.2f}s")
            
            if seg_info.get('source_video'):
                # 已找到匹配
                segment = VideoSegment(
                    source_video=seg_info['source_video'],
                    start_time=seg_info['source_start'],
                    end_time=seg_info['source_end'],
                    similarity_score=seg_info['similarity'],
                    segment_idx=idx
                )
                final_segments.append(segment)
                
                print(f"   ✅ 已匹配")
                print(f"   来源视频: {seg_info['source_video'].name}")
                print(f"   来源时间: {seg_info['source_start']:.2f}s - {seg_info['source_end']:.2f}s")
                print(f"   相似度: {seg_info['similarity']:.2%}")
                
                # 记录日志
                self.reconstruction_log.append({
                    'segment_idx': idx + 1,
                    'status': 'matched',
                    'target_time': f"{seg_info['start_time']:.2f}s - {seg_info['end_time']:.2f}s",
                    'source_video': str(seg_info['source_video']),
                    'source_time': f"{seg_info['source_start']:.2f}s - {seg_info['source_end']:.2f}s",
                    'similarity': f"{seg_info['similarity']:.2%}",
                    'duration': f"{seg_info['duration']:.2f}s"
                })
            else:
                # 未找到匹配，尝试智能填充
                print(f"   ❌ 未找到匹配")
                
                if self.missing_strategy == 'smart_fill':
                    # 寻找最佳替代片段
                    alternative = self._find_best_alternative(
                        seg_info, source_frames_dict
                    )
                    
                    if alternative and alternative['similarity'] >= self.smart_fill_threshold:
                        segment = VideoSegment(
                            source_video=alternative['source_video'],
                            start_time=alternative['start_time'],
                            end_time=alternative['end_time'],
                            similarity_score=alternative['similarity'],
                            segment_idx=idx
                        )
                        final_segments.append(segment)
                        
                        print(f"   🟡 智能填充（替代方案）")
                        print(f"   来源视频: {alternative['source_video'].name}")
                        print(f"   来源时间: {alternative['start_time']:.2f}s - {alternative['end_time']:.2f}s")
                        print(f"   替代相似度: {alternative['similarity']:.2%}")
                        
                        self.reconstruction_log.append({
                            'segment_idx': idx + 1,
                            'status': 'filled',
                            'target_time': f"{seg_info['start_time']:.2f}s - {seg_info['end_time']:.2f}s",
                            'source_video': str(alternative['source_video']),
                            'source_time': f"{alternative['start_time']:.2f}s - {alternative['end_time']:.2f}s",
                            'similarity': f"{alternative['similarity']:.2%}",
                            'duration': f"{seg_info['duration']:.2f}s",
                            'note': '智能填充'
                        })
                    else:
                        print(f"   ❌ 未找到合适的替代方案")
                        self.reconstruction_log.append({
                            'segment_idx': idx + 1,
                            'status': 'missing',
                            'target_time': f"{seg_info['start_time']:.2f}s - {seg_info['end_time']:.2f}s",
                            'duration': f"{seg_info['duration']:.2f}s",
                            'note': '未找到匹配或替代方案'
                        })
                else:
                    self.reconstruction_log.append({
                        'segment_idx': idx + 1,
                        'status': 'missing',
                        'target_time': f"{seg_info['start_time']:.2f}s - {seg_info['end_time']:.2f}s",
                        'duration': f"{seg_info['duration']:.2f}s"
                    })
        
        return final_segments
    
    def _find_best_alternative(self, target_seg: dict, 
                              source_frames_dict: dict) -> Optional[dict]:
        """为缺失片段寻找最佳替代方案"""
        best_alternative = None
        best_score = 0.0
        
        target_duration = target_seg['duration']
        
        for source_video, source_frames in source_frames_dict.items():
            # 滑动窗口寻找相似片段
            window_size = int(target_duration * self.fps)
            
            for start_idx in range(0, len(source_frames) - window_size, window_size // 2):
                end_idx = start_idx + window_size
                
                # 计算这个窗口的平均相似度
                window_similarities = []
                for i in range(start_idx, min(end_idx, len(source_frames))):
                    # 这里简化处理，实际应该和目标片段的关键帧对比
                    window_similarities.append(0.5)  # 占位
                
                avg_sim = np.mean(window_similarities) if window_similarities else 0.0
                
                if avg_sim > best_score:
                    best_score = avg_sim
                    best_alternative = {
                        'source_video': source_video,
                        'start_time': source_frames[start_idx][1],
                        'end_time': source_frames[min(end_idx, len(source_frames)-1)][1],
                        'similarity': best_score
                    }
        
        return best_alternative
    
    def save_reconstruction_log(self):
        """保存重构日志到JSON文件"""
        log_file = self.output_video.stem + "_reconstruction_log.json"
        
        log_data = {
            'target_video': str(self.target_video),
            'source_videos': [str(v) for v in self.source_videos],
            'output_video': str(self.output_video),
            'config': {
                'fps': self.fps,
                'similarity_threshold': self.similarity_threshold,
                'match_threshold': self.match_threshold,
                'missing_strategy': self.missing_strategy
            },
            'segments': self.reconstruction_log,
            'summary': {
                'total_segments': len(self.reconstruction_log),
                'matched': sum(1 for s in self.reconstruction_log if s['status'] == 'matched'),
                'filled': sum(1 for s in self.reconstruction_log if s['status'] == 'filled'),
                'missing': sum(1 for s in self.reconstruction_log if s['status'] == 'missing')
            }
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📝 重构日志已保存: {log_file}")
    
    def generate_ffmpeg_commands(self, segments: List[VideoSegment]) -> List[str]:
        """生成FFmpeg命令列表"""
        commands = []
        temp_files = []
        output_path = str(self.output_video)
        
        # 1. 先裁剪每个片段（只保留视频，不要音频）
        for idx, seg in enumerate(segments):
            temp_file = f"temp_segment_{idx:03d}.mp4"
            temp_files.append(temp_file)
            
            # -an 表示不复制音频，只保留视频
            cmd = f'''ffmpeg -y -hide_banner -ss {seg.start_time:.3f} -to {seg.end_time:.3f} -i "{seg.source_video}" -an -c:v copy "{temp_file}"'''
            commands.append(cmd)
        
        # 2. 生成concat列表文件
        concat_content = "\n".join([f"file '{f}'" for f in temp_files])
        with open("concat_list.txt", "w") as f:
            f.write(concat_content)
        
        # 3. 合并所有片段（无音频的临时视频）
        temp_video = "temp_video_no_audio.mp4"
        concat_cmd = f'''ffmpeg -y -hide_banner -f concat -safe 0 -i concat_list.txt -an -c:v copy "{temp_video}"'''
        commands.append(concat_cmd)
        temp_files.append(temp_video)
        
        # 4. 替换音频（用目标视频的音频）
        if self.use_target_audio:
            # 提取目标视频的音频
            temp_audio = "temp_audio.aac"
            extract_audio_cmd = f'''ffmpeg -y -hide_banner -i "{self.target_video}" -vn -c:a aac "{temp_audio}"'''
            commands.append(extract_audio_cmd)
            temp_files.append(temp_audio)
            
            # 合并视频和音频
            final_cmd = f'''ffmpeg -y -hide_banner -i "{temp_video}" -i "{temp_audio}" -c:v copy -c:a copy -shortest "{output_path}"'''
            commands.append(final_cmd)
        else:
            # 不使用目标音频，直接复制
            commands.append(f"mv {temp_video} {output_path}")
        
        # 5. 清理临时文件
        cleanup_cmd = f"rm -f {' '.join(temp_files)} concat_list.txt"
        commands.append(cleanup_cmd)
        
        return commands
    
    def validate_videos(self, video1_path: Path, video2_path: Path) -> ValidationResult:
        """校验两个视频的一致性"""
        print(f"\n🔍 正在校验视频一致性...")
        print(f"   视频1: {video1_path.name}")
        print(f"   视频2: {video2_path.name}")
        
        # 1. 检查时长
        duration1 = self.get_video_duration(video1_path)
        duration2 = self.get_video_duration(video2_path)
        duration_diff = abs(duration1 - duration2)
        duration_match = duration_diff < 0.5  # 允许0.5秒误差
        
        print(f"   时长: {duration1:.2f}s vs {duration2:.2f}s (差异{duration_diff:.2f}s)")
        
        # 2. 提取帧并比对
        temp_dir = Path(tempfile.mkdtemp())
        try:
            video1_dir = temp_dir / "video1"
            video2_dir = temp_dir / "video2"
            video1_dir.mkdir()
            video2_dir.mkdir()
            
            # 提取帧
            frames1 = self.extract_frames_with_timestamp(video1_path, video1_dir)
            frames2 = self.extract_frames_with_timestamp(video2_path, video2_dir)
            
            # 取较短的视频帧数
            min_frames = min(len(frames1), len(frames2))
            
            # 逐帧比对
            similarities = []
            for i in range(min_frames):
                sim = self.calculate_frame_similarity(frames1[i][0], frames2[i][0])
                similarities.append(sim)
            
            # 计算整体相似度
            overall_similarity = np.mean(similarities) if similarities else 0.0
            
            # 统计低于阈值的帧
            low_similarity_frames = sum(1 for s in similarities if s < self.similarity_threshold)
            
            print(f"   整体相似度: {overall_similarity:.2%}")
            print(f"   低相似度帧: {low_similarity_frames}/{len(similarities)} ({low_similarity_frames/len(similarities)*100:.1f}%)")
            
            # 判断是否通过
            passed = (overall_similarity >= self.similarity_threshold and 
                     duration_match and 
                     low_similarity_frames / len(similarities) < 0.1)  # 允许10%的帧低于阈值
            
            return ValidationResult(
                overall_similarity=overall_similarity,
                frame_by_frame_similarity=similarities,
                duration_match=duration_match,
                duration_diff=duration_diff,
                passed=passed
            )
            
        finally:
            shutil.rmtree(temp_dir)
    
    def reconstruct_with_validation(self) -> bool:
        """带校验的重构，如果不通过则重试"""
        
        for attempt in range(self.max_retries):
            print(f"\n{'='*60}")
            print(f"🔄 第 {attempt + 1}/{self.max_retries} 次尝试")
            print(f"{'='*60}")
            
            # 1. 执行重构
            segments, commands = self.reconstruct()
            
            if not segments:
                print("❌ 重构失败，未找到匹配片段")
                continue
            
            # 2. 执行FFmpeg命令
            print("\n⚙️  正在生成视频...")
            for cmd in commands[:-1]:  # 除了清理命令
                result = subprocess.run(cmd, shell=True, capture_output=True)
                if result.returncode != 0:
                    print(f"❌ 命令执行失败: {cmd}")
                    print(f"   错误: {result.stderr.decode()[:200]}")
                    break
            
            # 3. 校验结果
            if not self.output_video.exists():
                print("❌ 输出文件未生成")
                continue
            
            validation = self.validate_videos(self.target_video, self.output_video)
            
            print(f"\n{'='*60}")
            print("📊 校验结果:")
            print(f"{'='*60}")
            print(f"   整体相似度: {validation.overall_similarity:.2%} (阈值: {self.similarity_threshold:.0%})")
            print(f"   时长匹配: {'✅' if validation.duration_match else '❌'} (差异{validation.duration_diff:.2f}s)")
            print(f"   校验结果: {'✅ 通过' if validation.passed else '❌ 未通过'}")
            
            if validation.passed:
                print(f"\n✅ 视频重构成功: {self.output_video}")
                # 保存重构日志
                self.save_reconstruction_log()
                return True
            else:
                print(f"\n⚠️ 相似度未达标，准备重试...")
                # 降低匹配阈值后重试
                self.match_threshold -= 0.05
                self.similarity_threshold -= 0.05
        
        print(f"\n❌ 经过{self.max_retries}次尝试仍未达到要求")
        # 即使失败也保存日志
        self.save_reconstruction_log()
        return False
    
    def reconstruct(self):
        """主流程：重构视频"""
        
        # 1. 获取目标视频时长
        self.target_duration = self.get_video_duration(self.target_video)
        print(f"🎯 目标视频时长: {self.target_duration:.2f}秒")
        print(f"   提取帧率: {self.fps} fps")
        print(f"   预计提取帧数: ~{int(self.target_duration * self.fps)} 帧")
        
        # 2. 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        target_frames_dir = self.temp_dir / "target"
        target_frames_dir.mkdir()
        
        source_frames_dirs = {}
        for source_video in self.source_videos:
            source_dir = self.temp_dir / f"source_{source_video.stem}"
            source_dir.mkdir()
            source_frames_dirs[source_video] = source_dir
        
        try:
            # 3. 提取目标视频帧
            print("🎬 提取目标视频帧...")
            target_frames = self.extract_frames_with_timestamp(
                self.target_video, target_frames_dir
            )
            print(f"   提取了 {len(target_frames)} 帧")
            
            # 4. 提取所有源视频帧
            source_frames_dict = {}
            for source_video in self.source_videos:
                print(f"🎬 提取源视频帧: {source_video.name}")
                frames = self.extract_frames_with_timestamp(
                    source_video, source_frames_dirs[source_video]
                )
                source_frames_dict[source_video] = frames
                print(f"   提取了 {len(frames)} 帧")
            
            # 5. 找最佳匹配
            matches = self.find_best_matches(target_frames, source_frames_dict)
            
            # 统计匹配率
            matched_frames = sum(1 for m in matches if m is not None)
            match_rate = matched_frames / len(matches) * 100
            print(f"\n📊 匹配率: {matched_frames}/{len(matches)} 帧 ({match_rate:.1f}%)")
            
            if match_rate < 50:
                print("⚠️ 警告: 匹配率较低，可能需要检查源视频是否包含目标素材")
            
            # 6. 分析目标视频结构并处理缺失片段
            target_structure = self.analyze_target_structure(target_frames, matches)
            print(f"\n🎯 目标视频结构分析: 预期 {len(target_structure)} 个片段")
            
            # 将匹配结果整合到目标结构中
            for seg_info in target_structure:
                seg_start_frame = seg_info['start_frame']
                seg_end_frame = seg_info['end_frame']
                
                # 查找这个片段是否有匹配的FrameMatch
                segment_matches = []
                for idx in range(seg_start_frame, seg_end_frame + 1):
                    if idx < len(matches) and matches[idx] is not None:
                        segment_matches.append(matches[idx])
                
                if segment_matches:
                    # 有匹配，计算来源信息
                    first_match = segment_matches[0]
                    last_match = segment_matches[-1]
                    avg_similarity = np.mean([m.similarity for m in segment_matches])
                    
                    seg_info['source_video'] = first_match.source_video
                    seg_info['source_start'] = first_match.timestamp
                    seg_info['source_end'] = last_match.timestamp + (1.0 / self.fps)
                    seg_info['similarity'] = avg_similarity
                    seg_info['matched'] = True
                else:
                    seg_info['matched'] = False
            
            # 7. 使用智能填充处理缺失片段
            segments = self.find_alternative_segments(target_structure, source_frames_dict)
            
            # 8. 生成FFmpeg命令
            commands = self.generate_ffmpeg_commands(segments)
            
            print("\n" + "=" * 60)
            print("📝 FFmpeg命令已生成")
            if self.use_target_audio:
                print("   (将使用目标视频的音频)")
            print("=" * 60)
            
            # 生成shell脚本
            script_path = "reconstruct_video.sh"
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n\n")
                for cmd in commands[:-1]:
                    f.write(cmd + "\n")
                f.write("\n# 清理临时文件\n")
                f.write(commands[-1] + "\n")
                f.write(f"\necho '✅ 视频重构完成: {self.output_video}'\n")
            
            # 添加执行权限
            subprocess.run(['chmod', '+x', script_path])
            
            return segments, commands
            
        finally:
            # 清理临时目录
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)


def create_example_config():
    """创建示例配置文件"""
    example_config = """# 视频重构配置文件

# 裁剪视频1路径（参考视频）
target_video: "/path/to/clip1.mp4"

# 原视频路径列表（素材库）
source_videos:
  - "/path/to/source_A.mp4"
  - "/path/to/source_B.mp4"
  - "/path/to/source_C.mp4"

# 裁剪视频2输出路径
output_video: "clip2_reconstructed.mp4"

# 每秒提取帧数（1-10，建议5）
fps: 5

# 相似度阈值（0-1，建议0.90）
similarity_threshold: 0.90

# 最大重试次数
max_retries: 3

# 是否使用目标视频的音频
use_target_audio: true

# 最小片段时长（秒）
min_segment_duration: 0.5

# 单帧匹配阈值
match_threshold: 0.6

# 提取帧分辨率
scale: "480:270"

# 缺失片段处理策略
# strict: 严格模式，必须找到所有片段
# fill_black: 用黑屏填充缺失片段
# smart_fill: 智能填充，寻找最佳替代片段（推荐）
missing_segment_strategy: "smart_fill"

# 智能填充的最低相似度阈值（0-1）
smart_fill_threshold: 0.5
"""
    
    config_path = Path("video_reconstruct_config.yaml")
    if not config_path.exists():
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(example_config)
        print(f"✅ 已创建示例配置文件: {config_path}")
        print("   请修改配置文件中的路径后运行脚本")
        return True
    return False


def main():
    print("=" * 60)
    print("多源视频混剪重构工具（配置文件版）")
    print("从多个素材视频中重构出与目标视频相似的新版本")
    print("=" * 60)
    
    config_file = "video_reconstruct_config.yaml"
    
    # 检查配置文件是否存在
    if not Path(config_file).exists():
        print(f"\n⚠️ 配置文件不存在: {config_file}")
        create_example_config()
        return
    
    # 加载配置
    print(f"\n📋 加载配置文件: {config_file}")
    config = Config(config_file)
    
    # 显示配置信息
    print("\n📊 当前配置:")
    print(f"   目标视频: {config.get('target_video')}")
    print(f"   源视频数量: {len(config.get('source_videos', []))}")
    print(f"   输出视频: {config.get('output_video')}")
    print(f"   提取帧率: {config.get('fps')} fps")
    print(f"   相似度阈值: {config.get('similarity_threshold')*100:.0f}%")
    print(f"   最大重试: {config.get('max_retries')} 次")
    
    # 验证路径
    if not config.get('target_video') or not Path(config.get('target_video')).exists():
        print(f"\n❌ 目标视频不存在: {config.get('target_video')}")
        return
    
    valid_sources = []
    for src in config.get('source_videos', []):
        if Path(src).exists():
            valid_sources.append(src)
        else:
            print(f"⚠️ 源视频不存在，已跳过: {src}")
    
    if not valid_sources:
        print("\n❌ 没有有效的源视频")
        return
    
    config.set('source_videos', valid_sources)
    
    # 自动确认执行（跳过提示）
    print("\n✅ 自动确认执行")
    
    # 执行重构
    reconstructor = MultiSourceVideoReconstructor(config)
    success = reconstructor.reconstruct_with_validation()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 视频重构并校验通过！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 视频重构失败，未达到相似度要求")
        print("=" * 60)


if __name__ == "__main__":
    main()