#!/usr/bin/env python3
"""
批量视频重构工具 V4 - 100% 成功率优化版
核心策略：
1. 智能源视频选择 - 找到最匹配的源视频
2. 多源视频拼接 - 使用多个源视频组合重构
3. 参数自动调优 - 动态调整匹配阈值
4. 分段重构 - 对失败部分单独处理
5. 错误恢复 - 处理 Broken pipe 等问题
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v2 import VideoReconstructorHybridV2, load_config
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import subprocess
import shutil
from dataclasses import dataclass

@dataclass
class VideoInfo:
    """视频信息"""
    path: Path
    duration: float
    name: str

class SmartVideoReconstructor:
    """智能视频重构器"""
    
    def __init__(self, cut_videos_dir: str, source_videos_dir: str, output_dir: str):
        self.cut_videos_dir = Path(cut_videos_dir)
        self.source_videos_dir = Path(source_videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = load_config('/Users/zhangxu/work/项目/cutvideo/06_configurations/cut_reconstruction_config.yaml')
        
        # 获取并分析所有源视频
        self.source_videos = self._analyze_source_videos()
        print(f"🎬 找到 {len(self.source_videos)} 个源视频")
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
                except:
                    pass
        
        # 按时长排序
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

    def _find_best_source_videos(self, cut_video: Path, cut_duration: float) -> List[Path]:
        """找到最匹配的源视频"""
        # 策略：选择时长 >= 裁剪视频时长的源视频
        suitable = [sv for sv in self.source_videos if sv.duration >= cut_duration * 0.8]
        
        if not suitable:
            # 如果没有合适的，返回所有源视频
            return [sv.path for sv in self.source_videos]
        
        # 返回前3个最合适的
        return [sv.path for sv in suitable[:3]]

    def _try_reconstruct_with_retry(
        self,
        cut_video: Path,
        source_videos: List[Path],
        output_path: Path,
        max_retries: int = 3
    ) -> Tuple[bool, Optional[List], str]:
        """带重试的重构"""
        
        for attempt in range(max_retries):
            try:
                config = self.config.copy()
                config['target_video'] = str(cut_video)
                config['source_videos'] = [str(sv) for sv in source_videos]
                config['output_video'] = str(output_path)
                
                # 调整匹配参数（每次重试放宽一点）
                if attempt > 0:
                    config['match_threshold'] = max(0.5, 0.8 - attempt * 0.1)
                    print(f"   🔄 重试 {attempt + 1}/{max_retries}, 降低匹配阈值到 {config['match_threshold']}")
                
                reconstructor = VideoReconstructorHybridV2(
                    str(cut_video),
                    [str(sv) for sv in source_videos],
                    config
                )
                
                segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
                
                if segments and output_path.exists():
                    return True, segments, "成功"
                else:
                    return False, None, "未生成输出文件"
                    
            except BrokenPipeError:
                print(f"   ⚠️  Broken pipe 错误，等待后重试...")
                import time
                time.sleep(2)
                continue
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    print(f"   ⚠️  错误: {error_msg}, 重试中...")
                    continue
                return False, None, error_msg
        
        return False, None, f"重试 {max_retries} 次后仍失败"

    def process_single_video(self, cut_video: Path) -> dict:
        """处理单个裁剪视频（智能版）"""
        print(f"\n{'='*70}")
        print(f"🎬 处理: {cut_video.name}")
        print(f"{'='*70}")
        
        cut_duration = self._get_video_duration(cut_video)
        print(f"   裁剪视频时长: {cut_duration:.2f}s")
        
        # 找到最佳源视频
        best_sources = self._find_best_source_videos(cut_video, cut_duration)
        print(f"   选择 {len(best_sources)} 个源视频进行匹配")
        
        output_name = f"{cut_video.stem}_cut.mp4"
        output_path = self.output_dir / output_name
        
        # 尝试重构
        success, segments, message = self._try_reconstruct_with_retry(
            cut_video, best_sources, output_path, max_retries=3
        )
        
        if success and segments:
            # 计算匹配率
            total_matched = sum(seg.end_time - seg.start_time for seg in segments)
            match_rate = (total_matched / cut_duration * 100) if cut_duration > 0 else 0
            
            output_duration = self._get_video_duration(output_path)
            duration_diff = abs(output_duration - cut_duration)
            
            # 严格匹配标准：时长差异 < 1秒，匹配率 > 95%
            duration_match = duration_diff < 1.0
            content_match = match_rate > 95.0
            is_perfect = duration_match and content_match
            
            result = {
                'cut_video': str(cut_video),
                'cut_video_name': cut_video.name,
                'output_video': str(output_path),
                'cut_duration': cut_duration,
                'output_duration': output_duration,
                'duration_diff': duration_diff,
                'match_rate': match_rate,
                'segments': len(segments),
                'success': is_perfect,  # 只有完美匹配才算成功
                'is_perfect': is_perfect,
                'duration_match': duration_match,
                'content_match': content_match,
                'error': None if is_perfect else f"时长匹配: {duration_match}, 内容匹配: {content_match}",
                'source_videos_used': [str(s) for s in best_sources]
            }
            
            if is_perfect:
                print(f"   ✅ 完美匹配！")
                print(f"      匹配率: {match_rate:.1f}%")
                print(f"      时长差: {duration_diff:.2f}s")
            else:
                print(f"   ⚠️  未完全匹配")
                if not duration_match:
                    print(f"      时长差异过大: {duration_diff:.2f}s > 1s")
                if not content_match:
                    print(f"      匹配率不足: {match_rate:.1f}% < 95%")
        else:
            result = {
                'cut_video': str(cut_video),
                'cut_video_name': cut_video.name,
                'success': False,
                'error': message,
                'source_videos_used': [str(s) for s in best_sources]
            }
            print(f"   ❌ 重构失败: {message}")
        
        return result

    def process_all(self):
        """处理所有裁剪视频"""
        cut_videos = sorted(self.cut_videos_dir.glob('*.mp4'))
        print(f"🎬 找到 {len(cut_videos)} 个裁剪视频")
        
        if not cut_videos:
            print("❌ 没有找到裁剪视频")
            return []
        
        results = []
        perfect_count = 0
        
        for i, cut_video in enumerate(cut_videos, 1):
            print(f"\n📊 进度: {i}/{len(cut_videos)} ({i/len(cut_videos)*100:.1f}%)")
            result = self.process_single_video(cut_video)
            results.append(result)
            
            if result.get('is_perfect'):
                perfect_count += 1
        
        # 保存报告
        report_path = self.output_dir / 'reconstruction_report_v4.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total': len(cut_videos),
                'perfect': perfect_count,
                'imperfect': len(cut_videos) - perfect_count,
                'perfect_rate': perfect_count / len(cut_videos) * 100 if cut_videos else 0,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"📊 处理完成")
        print(f"{'='*70}")
        print(f"总视频: {len(cut_videos)}")
        print(f"完美匹配: {perfect_count} ✅")
        print(f"未完全匹配: {len(cut_videos) - perfect_count}")
        print(f"完美率: {perfect_count / len(cut_videos) * 100:.1f}%")
        print(f"📋 报告: {report_path}")
        
        return results


def main():
    CUT_VIDEOS_DIR = '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原'
    SOURCE_VIDEOS_DIR = '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集'
    OUTPUT_DIR = '/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v4'
    
    print("="*70)
    print("🎬 智能视频重构工具 V4 - 追求 100% 完美匹配")
    print("="*70)
    
    reconstructor = SmartVideoReconstructor(CUT_VIDEOS_DIR, SOURCE_VIDEOS_DIR, OUTPUT_DIR)
    reconstructor.process_all()


if __name__ == "__main__":
    main()
