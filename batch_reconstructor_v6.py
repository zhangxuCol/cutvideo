#!/usr/bin/env python3
"""
批量视频重构工具 V6 - 智能多源拼接版
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v6 import VideoReconstructorHybridV6, load_config
from pathlib import Path
import json
from typing import List, Dict
import subprocess
from dataclasses import dataclass

@dataclass
class VideoInfo:
    """视频信息"""
    path: Path
    duration: float
    name: str

class MultiSourceVideoReconstructor:
    """多源智能视频重构器 V6"""
    
    def __init__(self, cut_videos_dir: str, source_videos_dir: str, output_dir: str):
        self.cut_videos_dir = Path(cut_videos_dir)
        self.source_videos_dir = Path(source_videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有源视频
        self.source_videos = self._get_source_videos()
        total_duration = sum(self.get_video_duration(sv) for sv in self.source_videos)
        print(f"🎬 找到 {len(self.source_videos)} 个源视频，总时长 {total_duration:.1f}s")
        for sv in self.source_videos:
            print(f"   - {sv.name}: {self.get_video_duration(sv):.1f}s")

    def _get_source_videos(self) -> List[Path]:
        """获取所有源视频"""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        videos = []
        for ext in video_extensions:
            videos.extend(self.source_videos_dir.glob(f'*{ext}'))
        return sorted(videos)

    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())

    def process_single_video(self, cut_video: Path) -> dict:
        """处理单个裁剪视频"""
        print(f"\n{'='*70}")
        print(f"🎬 处理: {cut_video.name}")
        print(f"{'='*70}")
        
        cut_duration = self.get_video_duration(cut_video)
        print(f"   裁剪视频时长: {cut_duration:.2f}s")
        
        output_name = f"{cut_video.stem}_cut.mp4"
        output_path = self.output_dir / output_name
        
        # 使用 V6 重构器
        source_paths = [str(sv) for sv in self.source_videos]
        print(f"   使用 {len(source_paths)} 个源视频进行智能匹配")
        
        try:
            config = {
                'fps': 2,  # 降低帧率以加速处理
                'similarity_threshold': 0.85,
                'match_threshold': 0.6,
                'min_segment_duration': 1.0,
                'audio_weight': 0.4,
                'video_weight': 0.6
            }
            
            reconstructor = VideoReconstructorHybridV6(
                str(cut_video),
                source_paths,
                config
            )
            
            segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
            
            if segments and output_path.exists():
                # 计算匹配情况
                output_duration = self.get_video_duration(output_path)
                duration_diff = abs(output_duration - cut_duration)
                coverage = output_duration / cut_duration if cut_duration > 0 else 0
                
                # 判断是否成功
                is_success = coverage > 0.9  # 至少覆盖90%
                is_perfect = coverage > 0.95 and len(segments) == 1  # 单源完整匹配
                
                # 获取使用的源视频
                sources_used = list(set(str(seg.source_video) for seg in segments))
                
                result = {
                    'cut_video': str(cut_video),
                    'cut_video_name': cut_video.name,
                    'output_video': str(output_path),
                    'cut_duration': cut_duration,
                    'output_duration': output_duration,
                    'duration_diff': duration_diff,
                    'coverage': coverage,
                    'segments': len(segments),
                    'success': is_success,
                    'is_perfect': is_perfect,
                    'error': None,
                    'source_videos_used': sources_used,
                    'mode': 'single' if len(segments) == 1 else 'multi'
                }
                
                if is_success:
                    print(f"\n   ✅ 重构成功！")
                    print(f"      覆盖率: {coverage:.1%}")
                    print(f"      片段数: {len(segments)}")
                    print(f"      模式: {'单源' if len(segments) == 1 else '多源拼接'}")
                    if is_perfect:
                        print(f"      状态: 🌟 完美匹配")
                else:
                    print(f"\n   ⚠️ 覆盖不足: {coverage:.1%}")
            else:
                result = {
                    'cut_video': str(cut_video),
                    'cut_video_name': cut_video.name,
                    'success': False,
                    'error': '未生成输出文件',
                    'coverage': 0
                }
                print(f"\n   ❌ 重构失败: 未生成输出文件")
                
        except Exception as e:
            result = {
                'cut_video': str(cut_video),
                'cut_video_name': cut_video.name,
                'success': False,
                'error': str(e),
                'coverage': 0
            }
            print(f"\n   ❌ 重构失败: {e}")
        
        return result

    def process_all(self):
        """处理所有裁剪视频"""
        cut_videos = sorted(self.cut_videos_dir.glob('*.mp4'))
        print(f"🎬 找到 {len(cut_videos)} 个裁剪视频")
        
        if not cut_videos:
            print("❌ 没有找到裁剪视频")
            return []
        
        results = []
        success_count = 0
        perfect_count = 0
        
        for i, cut_video in enumerate(cut_videos, 1):
            print(f"\n📹 [{i}/{len(cut_videos)}]")
            result = self.process_single_video(cut_video)
            results.append(result)
            
            if result.get('success'):
                success_count += 1
                if result.get('is_perfect'):
                    perfect_count += 1
        
        # 生成报告
        report = {
            'total': len(cut_videos),
            'success': success_count,
            'perfect': perfect_count,
            'success_rate': success_count / len(cut_videos) * 100 if cut_videos else 0,
            'perfect_rate': perfect_count / len(cut_videos) * 100 if cut_videos else 0,
            'results': results
        }
        
        # 保存报告
        report_path = self.output_dir / 'reconstruction_report_v6.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印总结
        print(f"\n{'='*70}")
        print(f"📊 批量重构完成")
        print(f"{'='*70}")
        print(f"   总数: {report['total']}")
        print(f"   成功: {report['success']} ({report['success_rate']:.1f}%)")
        print(f"   完美: {report['perfect']} ({report['perfect_rate']:.1f}%)")
        print(f"   报告: {report_path}")
        
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量视频重构工具 V6 - 智能多源拼接版')
    parser.add_argument('--cut-dir', required=True, help='裁剪视频目录')
    parser.add_argument('--source-dir', required=True, help='源视频目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    
    args = parser.parse_args()
    
    reconstructor = MultiSourceVideoReconstructor(
        args.cut_dir,
        args.source_dir,
        args.output_dir
    )
    
    reconstructor.process_all()


if __name__ == '__main__':
    main()
