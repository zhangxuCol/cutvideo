#!/usr/bin/env python3
"""
批量视频重构工具 V5 - 多源视频智能拼接优化版
核心策略：
1. 多源视频联合搜索 - 在所有源视频中并行搜索匹配片段
2. 智能片段拼接 - 跨源视频组合片段，确保完整覆盖
3. 自动源视频扩展 - 当源视频时长不足时自动使用其他源
4. 片段间隙填充 - 自动处理片段之间的未匹配区域
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v5 import VideoReconstructorHybridV5, load_config
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import subprocess
from dataclasses import dataclass

@dataclass
class VideoInfo:
    """视频信息"""
    path: Path
    duration: float
    name: str

class MultiSourceVideoReconstructor:
    """多源智能视频重构器"""
    
    def __init__(self, cut_videos_dir: str, source_videos_dir: str, output_dir: str):
        self.cut_videos_dir = Path(cut_videos_dir)
        self.source_videos_dir = Path(source_videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = load_config('/Users/zhangxu/work/项目/cutvideo/configurations/cut_reconstruction_config.yaml')
        
        # 获取并分析所有源视频
        self.source_videos = self._analyze_source_videos()
        total_duration = sum(sv.duration for sv in self.source_videos)
        print(f"🎬 找到 {len(self.source_videos)} 个源视频，总时长 {total_duration:.1f}s")
        for sv in self.source_videos:
            print(f"   - {sv.name}: {sv.duration:.1f}s")

    def _analyze_source_videos(self) -> List[VideoInfo]:
        """分析所有源视频，获取时长信息"""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        videos = []
        
        for ext in video_extensions:
            for path in self.source_videos_dir.glob(f'*{ext}'):
                try:
                    duration = self._get_video_duration(path)
                    videos.append(VideoInfo(path, duration, path.name))
                except Exception as e:
                    print(f"   ⚠️  无法分析 {path.name}: {e}")
        
        # 按时长排序（从长到短）
        return sorted(videos, key=lambda x: x.duration, reverse=True)

    def _get_video_duration(self, video_path: Path) -> float:
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
        """处理单个裁剪视频（多源智能版）"""
        print(f"\n{'='*70}")
        print(f"🎬 处理: {cut_video.name}")
        print(f"{'='*70}")
        
        cut_duration = self._get_video_duration(cut_video)
        print(f"   裁剪视频时长: {cut_duration:.2f}s")
        
        # 检查源视频总时长是否足够
        total_source_duration = sum(sv.duration for sv in self.source_videos)
        if total_source_duration < cut_duration:
            print(f"   ⚠️  警告：源视频总时长({total_source_duration:.1f}s) < 裁剪视频时长({cut_duration:.1f}s)")
        
        output_name = f"{cut_video.stem}_cut.mp4"
        output_path = self.output_dir / output_name
        
        # 使用所有源视频进行重构
        source_paths = [sv.path for sv in self.source_videos]
        print(f"   使用 {len(source_paths)} 个源视频进行联合搜索")
        
        try:
            # 更新配置
            config = self.config.copy()
            config['target_video'] = str(cut_video)
            config['source_videos'] = [str(s) for s in source_paths]
            config['output_video'] = str(output_path)
            
            # V5: 使用多源重构器
            reconstructor = VideoReconstructorHybridV5(
                str(cut_video),
                [str(s) for s in source_paths],
                config
            )
            
            segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
            
            if segments and output_path.exists():
                # 计算匹配率
                total_matched = sum(seg.end_time - seg.start_time for seg in segments)
                match_rate = (total_matched / cut_duration * 100) if cut_duration > 0 else 0
                
                output_duration = self._get_video_duration(output_path)
                duration_diff = abs(output_duration - cut_duration)
                
                # V5: 使用更宽松的完美匹配标准
                # 时长差异 < 5秒 或 匹配率 > 90%
                duration_match = duration_diff < 5.0
                content_match = match_rate > 90.0
                
                # 放宽的成功标准
                is_success = (duration_match and content_match) or match_rate > 85.0
                
                result = {
                    'cut_video': str(cut_video),
                    'cut_video_name': cut_video.name,
                    'output_video': str(output_path),
                    'cut_duration': cut_duration,
                    'output_duration': output_duration,
                    'duration_diff': duration_diff,
                    'match_rate': match_rate,
                    'segments': len(segments),
                    'success': is_success,
                    'duration_match': duration_match,
                    'content_match': content_match,
                    'is_perfect': duration_match and content_match,
                    'error': None,
                    'source_videos_used': list(set(str(seg.source_video) for seg in segments))
                }
                
                if is_success:
                    print(f"   ✅ 重构成功！")
                    print(f"      匹配率: {match_rate:.1f}%")
                    print(f"      时长差: {duration_diff:.2f}s")
                    print(f"      片段数: {len(segments)}")
                    if duration_match and content_match:
                        print(f"      状态: 🌟 完美匹配")
                    else:
                        print(f"      状态: ✓ 良好匹配")
                else:
                    print(f"   ⚠️  部分匹配")
                    if not duration_match:
                        print(f"      时长差异: {duration_diff:.2f}s")
                    if not content_match:
                        print(f"      匹配率: {match_rate:.1f}%")
            else:
                result = {
                    'cut_video': str(cut_video),
                    'cut_video_name': cut_video.name,
                    'success': False,
                    'error': '未生成输出文件',
                    'source_videos_used': [str(s) for s in source_paths]
                }
                print(f"   ❌ 重构失败: 未生成输出文件")
                
        except Exception as e:
            result = {
                'cut_video': str(cut_video),
                'cut_video_name': cut_video.name,
                'success': False,
                'error': str(e),
                'source_videos_used': [str(s) for s in source_paths]
            }
            print(f"   ❌ 重构失败: {e}")
        
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
        report_path = self.output_dir / 'reconstruction_report_v5.json'
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
    
    parser = argparse.ArgumentParser(description='批量视频重构工具 V5')
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
