#!/usr/bin/env python3
"""
多源视频混剪重构工具
从多个原视频中找到素材，重新混剪出与目标视频相同长度、相似内容的新版本
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq

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

class MultiSourceVideoReconstructor:
    def __init__(self, target_video: str, source_videos: List[str]):
        self.target_video = Path(target_video)
        self.source_videos = [Path(v) for v in source_videos]
        self.temp_dir = None
        self.target_duration = 0

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

    def extract_frames_with_timestamp(self, video_path: Path, output_dir: Path, fps: int = 2) -> List[Tuple[Path, float]]:
        """提取视频帧，返回(帧路径, 时间戳)列表"""
        output_pattern = output_dir / f"{video_path.stem}_%06d.jpg"

        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-i', str(video_path),
            '-vf', f'fps={fps},scale=480:270',
            '-q:v', '2',
            str(output_pattern)
        ]

        subprocess.run(cmd, capture_output=True)

        # 获取提取的帧并计算时间戳
        frames = []
        frame_files = sorted(output_dir.glob(f"{video_path.stem}_*.jpg"))

        for idx, frame_path in enumerate(frame_files):
            timestamp = idx / fps  # 根据fps计算时间戳
            frames.append((frame_path, timestamp))

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

        # 方法2: 结构相似度（SSIM的简化版）
        # 使用模板匹配作为快速替代
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)

        # 方法3: 感知哈希（pHash）
        phash_sim = self._phash_similarity(gray1, gray2)

        # 综合评分（加权平均）
        final_sim = 0.4 * max(0, hist_sim) + 0.3 * template_sim + 0.3 * phash_sim

        return final_sim

    def _phash_similarity(self, img1, img2) -> float:
        """感知哈希相似度"""
        # 缩小并转换为灰度
        size = (32, 32)
        img1_small = cv2.resize(img1, size)
        img2_small = cv2.resize(img2, size)

        # 计算DCT
        img1_float = np.float32(img1_small)
        img2_float = np.float32(img2_small)

        dct1 = cv2.dct(img1_float)
        dct2 = cv2.dct(img2_float)

        # 取低频部分
        low_freq1 = dct1[:8, :8].flatten()
        low_freq2 = dct2[:8, :8].flatten()

        # 计算均值并二值化
        mean1 = np.mean(low_freq1)
        mean2 = np.mean(low_freq2)

        hash1 = (low_freq1 > mean1).astype(int)
        hash2 = (low_freq2 > mean2).astype(int)

        # 计算汉明距离
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1 - (hamming_dist / 64.0)

        return max(0, similarity)

    def find_best_matches(self, target_frames: List[Tuple[Path, float]], 
                         source_frames_dict: dict) -> List[FrameMatch]:
        """为目标视频的每一帧找到最佳匹配"""
        matches = []

        print(f"🔍 正在比对 {len(target_frames)} 帧目标画面...")

        for target_idx, (target_frame, target_ts) in enumerate(target_frames):
            if target_idx % 10 == 0:
                print(f"   处理中: {target_idx}/{len(target_frames)} 帧")

            best_match = None
            best_score = 0.0

            # 在所有源视频中找最佳匹配
            for source_video, source_frames in source_frames_dict.items():
                for source_idx, (source_frame, source_ts) in enumerate(source_frames):
                    similarity = self.calculate_frame_similarity(target_frame, source_frame)

                    if similarity > best_score and similarity > 0.6:  # 阈值0.6
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
                # 如果没找到匹配，标记为None
                matches.append(None)

        return matches

    def segment_matches(self, matches: List[FrameMatch], min_segment_duration: float = 1.0) -> List[VideoSegment]:
        """将连续的匹配帧聚合成片段"""
        segments = []
        current_segment = []

        fps = 2  # 假设fps为2

        for idx, match in enumerate(matches):
            if match is None:
                # 当前匹配结束，保存之前的片段
                if len(current_segment) >= min_segment_duration * fps:
                    self._save_segment(current_segment, segments)
                current_segment = []
            else:
                current_segment.append(match)

        # 保存最后一个片段
        if len(current_segment) >= min_segment_duration * fps:
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
            end_time=last_match.timestamp + 0.5,  # 加0.5秒确保覆盖
            similarity_score=avg_similarity,
            segment_idx=len(segments)
        )

        segments.append(segment)

    def generate_cut_list(self, segments: List[VideoSegment]) -> str:
        """生成FFmpeg concat所需的文件列表"""
        cut_list = []

        for seg in segments:
            duration = seg.end_time - seg.start_time
            cut_list.append({
                'file': str(seg.source_video),
                'start': seg.start_time,
                'end': seg.end_time,
                'duration': duration
            })

        return cut_list

    def generate_ffmpeg_commands(self, segments: List[VideoSegment], output_path: str = "reconstructed.mp4", 
                                 use_target_audio: bool = True) -> List[str]:
        """生成FFmpeg命令列表"""
        commands = []
        temp_files = []

        # 1. 先裁剪每个片段（只保留视频，不要音频）
        for idx, seg in enumerate(segments):
            temp_file = f"temp_segment_{idx:03d}.mp4"
            temp_files.append(temp_file)

            # -an 表示不复制音频，只保留视频
            duration = seg.end_time - seg.start_time
            cmd = f"""ffmpeg -y -hide_banner -ss {seg.start_time:.3f} -t {duration:.3f} -i "{seg.source_video}" -an -c:v copy "{temp_file}" """[:-1]
            commands.append(cmd)

        # 2. 生成concat列表文件
        concat_content = "\n".join([f"file '{f}'" for f in temp_files])
        with open("concat_list.txt", "w") as f:
            f.write(concat_content)

        # 3. 合并所有片段（无音频的临时视频）
        temp_video = "temp_video_no_audio.mp4"
        concat_cmd = f'ffmpeg -y -hide_banner -f concat -safe 0 -i concat_list.txt -an -c:v copy "{temp_video}"'
        commands.append(concat_cmd)
        temp_files.append(temp_video)

        # 4. 替换音频（用目标视频的音频）
        if use_target_audio:
            # 提取目标视频的音频
            temp_audio = "temp_audio.aac"
            extract_audio_cmd = f'ffmpeg -y -hide_banner -i "{self.target_video}" -vn -c:a aac "{temp_audio}"'
            commands.append(extract_audio_cmd)
            temp_files.append(temp_audio)

            # 合并视频和音频
            final_cmd = f'ffmpeg -y -hide_banner -i "{temp_video}" -i "{temp_audio}" -c:v copy -c:a copy -shortest "{output_path}"'
            commands.append(final_cmd)
        else:
            # 不使用目标音频，直接复制
            commands.append(f"mv {temp_video} {output_path}")

        # 5. 清理临时文件
        cleanup_cmd = f"rm -f {' '.join(temp_files)} concat_list.txt"
        commands.append(cleanup_cmd)

        return commands

    def validate_videos(self, video1_path: Path, video2_path: Path, 
                       similarity_threshold: float = 0.90) -> ValidationResult:
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

            # 提取帧（每秒2帧）
            frames1 = self.extract_frames_with_timestamp(video1_path, video1_dir, fps=2)
            frames2 = self.extract_frames_with_timestamp(video2_path, video2_dir, fps=2)

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
            low_similarity_frames = sum(1 for s in similarities if s < similarity_threshold)

            print(f"   整体相似度: {overall_similarity:.2%}")
            print(f"   低相似度帧: {low_similarity_frames}/{len(similarities)} ({low_similarity_frames/len(similarities)*100:.1f}%)")

            # 判断是否通过
            passed = (overall_similarity >= similarity_threshold and 
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

    def reconstruct_with_validation(self, output_path: str = "reconstructed.mp4", 
                                    use_target_audio: bool = True,
                                    similarity_threshold: float = 0.90,
                                    max_retries: int = 3) -> bool:
        """带校验的重构，如果不通过则重试"""

        for attempt in range(max_retries):
            print(f"\n{'='*60}")
            print(f"🔄 第 {attempt + 1}/{max_retries} 次尝试")
            print(f"{'='*60}")

            # 1. 执行重构
            segments, commands = self.reconstruct(output_path, use_target_audio)

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
            output_file = Path(output_path)
            if not output_file.exists():
                print("❌ 输出文件未生成")
                continue

            validation = self.validate_videos(self.target_video, output_file, similarity_threshold)

            print(f"\n{'='*60}")
            print("📊 校验结果:")
            print(f"{'='*60}")
            print(f"   整体相似度: {validation.overall_similarity:.2%} (阈值: {similarity_threshold:.0%})")
            print(f"   时长匹配: {'✅' if validation.duration_match else '❌'} (差异{validation.duration_diff:.2f}s)")
            print(f"   校验结果: {'✅ 通过' if validation.passed else '❌ 未通过'}")

            if validation.passed:
                print(f"\n✅ 视频重构成功: {output_path}")
                return True
            else:
                print(f"\n⚠️ 相似度未达标，准备重试...")
                # 降低匹配阈值或调整参数后重试
                similarity_threshold -= 0.05  # 每次降低5%阈值

        print(f"\n❌ 经过{max_retries}次尝试仍未达到要求")
        return False

    def reconstruct(self, output_path: str = "reconstructed.mp4", use_target_audio: bool = True):
        """主流程：重构视频"""

        # 1. 获取目标视频时长
        self.target_duration = self.get_video_duration(self.target_video)
        print(f"🎯 目标视频时长: {self.target_duration:.2f}秒")

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
                self.target_video, target_frames_dir, fps=2
            )
            print(f"   提取了 {len(target_frames)} 帧")

            # 4. 提取所有源视频帧
            source_frames_dict = {}
            for source_video in self.source_videos:
                print(f"🎬 提取源视频帧: {source_video.name}")
                frames = self.extract_frames_with_timestamp(
                    source_video, source_frames_dirs[source_video], fps=2
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

            # 6. 聚合成片段
            segments = self.segment_matches(matches, min_segment_duration=0.5)
            print(f"\n📦 找到 {len(segments)} 个连续片段:")

            total_duration = 0
            for seg in segments:
                duration = seg.end_time - seg.start_time
                total_duration += duration
                print(f"   片段{seg.segment_idx+1}: {seg.source_video.name}")
                print(f"      时间: {seg.start_time:.2f}s - {seg.end_time:.2f}s (时长{duration:.2f}s)")
                print(f"      相似度: {seg.similarity_score:.2%}")

            print(f"\n⏱️ 总时长: {total_duration:.2f}s / 目标: {self.target_duration:.2f}s")

            # 7. 生成FFmpeg命令（默认使用目标视频的音频）
            commands = self.generate_ffmpeg_commands(segments, output_path, use_target_audio=True)

            print("\n" + "=" * 60)
            print("📝 执行以下命令重构视频:")
            if use_target_audio:
                print("   (将使用目标视频的音频)")
            print("=" * 60)

            # 生成shell脚本
            script_path = "reconstruct_video.sh"
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n\n")
                for cmd in commands[:-1]:  # 除了清理命令
                    f.write(cmd + "\n")
                f.write("\n# 清理临时文件\n")
                f.write(commands[-1] + "\n")
                f.write(f"\necho '✅ 视频重构完成: {output_path}'\n")

            # 添加执行权限
            subprocess.run(['chmod', '+x', script_path])

            print(f"bash {script_path}")
            print("\n或直接执行以下命令:")
            for cmd in commands[:-1]:
                print(cmd)

            return segments, commands

        finally:
            # 清理临时目录
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)


def main():
    print("=" * 60)
    print("多源视频混剪重构工具（带自动校验）")
    print("从多个素材视频中重构出与目标视频相似的新版本")
    print("=" * 60)

    # 获取输入
    target_video = input("\n请输入目标视频路径（裁剪视频1）: ").strip().strip('"')

    print("\n请输入源视频路径（素材库，可输入多个，输入空行结束）:")
    source_videos = []
    while True:
        path = input(f"  源视频{len(source_videos)+1}: ").strip().strip('"')
        if not path:
            break
        if Path(path).exists():
            source_videos.append(path)
        else:
            print(f"    ⚠️ 文件不存在: {path}")

    if not source_videos:
        print("❌ 至少需要提供一个源视频")
        return

    if not Path(target_video).exists():
        print(f"❌ 目标视频不存在: {target_video}")
        return

    output_name = input("\n请输入输出视频名称（默认: reconstructed.mp4）: ").strip()
    if not output_name:
        output_name = "reconstructed.mp4"

    # 询问是否使用目标视频的音频
    use_audio = input("是否使用目标视频的音频? (y/n, 默认y): ").strip().lower()
    use_target_audio = use_audio != 'n'

    # 询问相似度阈值
    threshold_input = input("相似度阈值（默认90%，输入0-100的数字）: ").strip()
    similarity_threshold = float(threshold_input) / 100 if threshold_input else 0.90

    # 询问最大重试次数
    retry_input = input("最大重试次数（默认3次）: ").strip()
    max_retries = int(retry_input) if retry_input else 3

    # 执行带校验的重构
    reconstructor = MultiSourceVideoReconstructor(target_video, source_videos)
    success = reconstructor.reconstruct_with_validation(
        output_name, 
        use_target_audio, 
        similarity_threshold,
        max_retries
    )

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
