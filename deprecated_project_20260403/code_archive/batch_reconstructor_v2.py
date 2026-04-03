#!/usr/bin/env python3
"""
批量视频重构工具 V2
使用改进版重构器，支持片段去重和覆盖率检查
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v2 import VideoReconstructorHybridV2, load_config
from pathlib import Path
import json
from typing import List, Tuple
import subprocess

class BatchVideoReconstructorV2:
    def __init__(self, cut_videos_dir: str, source_videos_dir: str, output_dir: str):
        self.cut_videos_dir = Path(cut_videos_dir)
        self.source_videos_dir = Path(source_videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载默认配置
        self.config = load_config('/Users/zhangxu/work/项目/cutvideo/06_configurations/cut_reconstruction_config.yaml')

        # 获取源视频列表
        self.source_videos = self._get_source_videos()
        print(f"🎬 找到 {len(self.source_videos)} 个源视频")

    def _get_source_videos(self) -> List[Path]:
        """获取所有源视频"""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        videos = []
        for ext in video_extensions:
            videos.extend(self.source_videos_dir.glob(f'*{ext}'))
        return sorted(videos)

    def _get_cut_videos(self) -> List[Path]:
        """获取所有裁剪视频"""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        videos = []
        for ext in video_extensions:
            videos.extend(self.cut_videos_dir.glob(f'*{ext}'))
        return sorted(videos)

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

    def process_single_video(self, cut_video: Path) -> dict:
        """处理单个裁剪视频"""
        print(f"\n{'='*60}")
        print(f"🎬 处理裁剪视频: {cut_video.name}")
        print(f"{'='*60}")

        cut_duration = self.get_video_duration(cut_video)
        print(f"   时长: {cut_duration:.2f}秒")

        # 输出文件名: 裁剪视频名_cut.mp4
        output_name = f"{cut_video.stem}_cut.mp4"
        output_path = self.output_dir / output_name

        # 更新配置
        config = self.config.copy()
        config['target_video'] = str(cut_video)
        config['source_videos'] = [str(v) for v in self.source_videos]
        config['output_video'] = str(output_path)

        # 执行重构
        try:
            reconstructor = VideoReconstructorHybridV2(str(cut_video),
                                                      [str(v) for v in self.source_videos],
                                                      config)
            segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)

            if segments:
                # 计算匹配率
                total_matched_duration = sum(seg.end_time - seg.start_time for seg in segments)
                match_rate = total_matched_duration / cut_duration * 100

                # 验证二次裁剪视频时长
                if output_path.exists():
                    output_duration = self.get_video_duration(output_path)
                    duration_diff = abs(output_duration - cut_duration)

                    # 严格匹配模式：时长、内容、音频保持一致
                    # - 时长差异 < 1秒
                    # - 匹配率 > 95%
                    # - 音频已启用 use_target_audio=True
                    duration_match = duration_diff < 1.0  # 时长差异小于1秒
                    content_match = match_rate > 95.0     # 内容匹配率大于95%
                    success = duration_match and content_match

                    result = {
                        'cut_video': str(cut_video),
                        'cut_video_name': cut_video.name,
                        'output_video': str(output_path),
                        'output_video_name': output_name,
                        'cut_duration': cut_duration,
                        'output_duration': output_duration,
                        'duration_diff': duration_diff,
                        'match_rate': match_rate,
                        'segments': len(segments),
                        'success': success,
                        'error': None
                    }
                else:
                    result = {
                        'cut_video': str(cut_video),
                        'cut_video_name': cut_video.name,
                        'success': False,
                        'error': '输出视频未生成'
                    }
            else:
                result = {
                    'cut_video': str(cut_video),
                    'cut_video_name': cut_video.name,
                    'success': False,
                    'error': '未找到匹配片段'
                }

        except Exception as e:
            result = {
                'cut_video': str(cut_video),
                'cut_video_name': cut_video.name,
                'success': False,
                'error': str(e)
            }

        return result

    def process_all(self):
        """处理所有裁剪视频"""
        cut_videos = self._get_cut_videos()
        print(f"🎬 找到 {len(cut_videos)} 个裁剪视频")

        if not cut_videos:
            print("❌ 没有找到裁剪视频")
            return

        if not self.source_videos:
            print("❌ 没有找到源视频")
            return

        results = []
        success_count = 0

        for i, cut_video in enumerate(cut_videos, 1):
            print(f"\n📊 进度: {i}/{len(cut_videos)} ({i/len(cut_videos)*100:.1f}%)")
            result = self.process_single_video(cut_video)
            results.append(result)

            if result.get('success'):
                success_count += 1
                print(f"✅ 成功: {result['cut_video_name']}")
                print(f"   匹配率: {result['match_rate']:.1f}%")
                print(f"   时长差异: {result['duration_diff']:.2f}秒")
            else:
                print(f"❌ 失败: {result['cut_video_name']}")
                print(f"   错误: {result.get('error', '未知错误')}")

        # 保存结果报告
        report_path = self.output_dir / 'batch_reconstruction_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total': len(cut_videos),
                'success': success_count,
                'failed': len(cut_videos) - success_count,
                'success_rate': success_count / len(cut_videos) * 100 if cut_videos else 0,
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"📊 批量处理完成")
        print(f"{'='*60}")
        print(f"总视频数: {len(cut_videos)}")
        print(f"成功: {success_count}")
        print(f"失败: {len(cut_videos) - success_count}")
        print(f"成功率: {success_count / len(cut_videos) * 100:.1f}%")
        print(f"📋 报告已保存: {report_path}")

        return results


def main():
    # 配置路径
    CUT_VIDEOS_DIR = '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原'
    SOURCE_VIDEOS_DIR = '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集'
    OUTPUT_DIR = '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output'

    print("="*60)
    print("🎬 批量视频重构工具 V2")
    print("="*60)
    print(f"裁剪视频目录: {CUT_VIDEOS_DIR}")
    print(f"源视频目录: {SOURCE_VIDEOS_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)

    batch = BatchVideoReconstructorV2(CUT_VIDEOS_DIR, SOURCE_VIDEOS_DIR, OUTPUT_DIR)
    batch.process_all()


if __name__ == "__main__":
    main()
