#!/usr/bin/env python3
"""
批量视频重构工具 V3 - 优化版
改进点：
1. 可配置的验证阈值
2. 详细的错误分类和日志
3. 支持部分重构模式
4. 批量重试机制
5. 调试模式
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from video_reconstructor_hybrid_v2 import VideoReconstructorHybridV2, load_config
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import subprocess
from dataclasses import dataclass, asdict
from enum import Enum

class ValidationLevel(Enum):
    """验证严格程度"""
    STRICT = "strict"      # 严格模式：匹配率>50%, 时长差异<20%
    NORMAL = "normal"      # 正常模式：匹配率>30%, 时长差异<30%
    LOOSE = "loose"        # 宽松模式：匹配率>10%, 时长差异<50%
    BEST_EFFORT = "best_effort"  # 尽力模式：只要有输出就算成功

@dataclass
class ValidationThresholds:
    """验证阈值配置"""
    min_match_rate: float = 50.0  # 最小匹配率 (%)
    max_duration_diff_ratio: float = 0.2  # 最大时长差异比例 (20%)
    
    @classmethod
    def from_level(cls, level: ValidationLevel) -> 'ValidationThresholds':
        """根据级别创建阈值"""
        thresholds = {
            ValidationLevel.STRICT: cls(50.0, 0.2),
            ValidationLevel.NORMAL: cls(30.0, 0.3),
            ValidationLevel.LOOSE: cls(10.0, 0.5),
            ValidationLevel.BEST_EFFORT: cls(0.0, 1.0),
        }
        return thresholds.get(level, cls())

@dataclass
class FailureReason:
    """失败原因"""
    code: str
    message: str
    details: Dict

class FailureAnalyzer:
    """失败分析器"""
    
    @staticmethod
    def analyze(
        match_rate: Optional[float],
        duration_diff: Optional[float],
        cut_duration: float,
        output_duration: Optional[float],
        has_output: bool,
        has_segments: bool,
        thresholds: ValidationThresholds
    ) -> Optional[FailureReason]:
        """分析失败原因"""
        
        if not has_output:
            return FailureReason(
                code="NO_OUTPUT",
                message="输出视频未生成",
                details={"check": "文件系统权限或重构过程异常"}
            )
        
        if not has_segments:
            return FailureReason(
                code="NO_SEGMENTS",
                message="未找到匹配片段",
                details={
                    "possible_causes": [
                        "源视频中不包含该裁剪片段",
                        "视频特征提取失败",
                        "匹配算法参数不合适"
                    ],
                    "suggestion": "检查源视频是否包含目标内容，或调整匹配参数"
                }
            )
        
        reasons = []
        
        # 检查匹配率
        if match_rate is not None and match_rate < thresholds.min_match_rate:
            reasons.append({
                "type": "LOW_MATCH_RATE",
                "message": f"匹配率过低 ({match_rate:.1f}% < {thresholds.min_match_rate}%)",
                "details": {
                    "actual_rate": match_rate,
                    "required_rate": thresholds.min_match_rate,
                    "suggestion": "尝试放宽匹配阈值或使用不同的源视频"
                }
            })
        
        # 检查时长差异
        if duration_diff is not None and cut_duration > 0:
            diff_ratio = duration_diff / cut_duration
            if diff_ratio > thresholds.max_duration_diff_ratio:
                reasons.append({
                    "type": "LARGE_DURATION_DIFF",
                    "message": f"时长差异过大 ({diff_ratio*100:.1f}% > {thresholds.max_duration_diff_ratio*100}%)",
                    "details": {
                        "cut_duration": cut_duration,
                        "output_duration": output_duration,
                        "diff_seconds": duration_diff,
                        "diff_ratio": diff_ratio,
                        "suggestion": "检查片段拼接逻辑或音频同步问题"
                    }
                })
        
        if reasons:
            return FailureReason(
                code="VALIDATION_FAILED",
                message="; ".join([r["message"] for r in reasons]),
                details={"reasons": reasons}
            )
        
        return None

class BatchVideoReconstructorV3:
    """批量视频重构工具 V3"""
    
    def __init__(
        self,
        cut_videos_dir: str,
        source_videos_dir: str,
        output_dir: str,
        validation_level: ValidationLevel = ValidationLevel.NORMAL,
        debug: bool = False
    ):
        self.cut_videos_dir = Path(cut_videos_dir)
        self.source_videos_dir = Path(source_videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_level = validation_level
        self.thresholds = ValidationThresholds.from_level(validation_level)
        self.debug = debug
        
        # 加载配置
        config_path = '/Users/zhangxu/work/项目/cutvideo/configurations/cut_reconstruction_config.yaml'
        self.config = load_config(config_path)
        
        # 获取源视频
        self.source_videos = self._get_source_videos()
        print(f"🎬 找到 {len(self.source_videos)} 个源视频")
        print(f"🔧 验证级别: {validation_level.value}")
        print(f"   - 最小匹配率: {self.thresholds.min_match_rate}%")
        print(f"   - 最大时长差异: {self.thresholds.max_duration_diff_ratio*100}%")

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

    def process_single_video(
        self,
        cut_video: Path,
        attempt: int = 1,
        max_attempts: int = 1
    ) -> dict:
        """处理单个裁剪视频（支持重试）"""
        
        print(f"\n{'='*70}")
        print(f"🎬 处理裁剪视频: {cut_video.name} (尝试 {attempt}/{max_attempts})")
        print(f"{'='*70}")

        cut_duration = self.get_video_duration(cut_video)
        print(f"   原始时长: {cut_duration:.2f}秒")

        output_name = f"{cut_video.stem}_cut.mp4"
        output_path = self.output_dir / output_name

        # 更新配置
        config = self.config.copy()
        config['target_video'] = str(cut_video)
        config['source_videos'] = [str(v) for v in self.source_videos]
        config['output_video'] = str(output_path)

        # 执行重构
        try:
            reconstructor = VideoReconstructorHybridV2(
                str(cut_video),
                [str(v) for v in self.source_videos],
                config
            )
            segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)

            # 分析结果
            result = self._analyze_result(
                cut_video, output_path, cut_duration, segments
            )
            
            # 如果失败且可以重试
            if not result['success'] and attempt < max_attempts:
                print(f"   ⚠️  处理失败，准备重试...")
                return self.process_single_video(cut_video, attempt + 1, max_attempts)
            
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ 异常: {error_msg}")
            
            if self.debug:
                import traceback
                traceback.print_exc()
            
            # 如果失败且可以重试
            if attempt < max_attempts:
                print(f"   🔄 发生异常，准备重试...")
                return self.process_single_video(cut_video, attempt + 1, max_attempts)
            
            return {
                'cut_video': str(cut_video),
                'cut_video_name': cut_video.name,
                'success': False,
                'error': error_msg,
                'error_code': 'EXCEPTION',
                'attempt': attempt
            }

    def _analyze_result(
        self,
        cut_video: Path,
        output_path: Path,
        cut_duration: float,
        segments: list
    ) -> dict:
        """分析处理结果"""
        
        has_output = output_path.exists()
        has_segments = bool(segments)
        
        # 计算匹配率
        match_rate = None
        if has_segments:
            total_matched = sum(seg.end_time - seg.start_time for seg in segments)
            match_rate = (total_matched / cut_duration * 100) if cut_duration > 0 else 0
        
        # 获取输出时长
        output_duration = None
        duration_diff = None
        if has_output:
            try:
                output_duration = self.get_video_duration(output_path)
                duration_diff = abs(output_duration - cut_duration)
            except:
                pass
        
        # 分析失败原因
        failure_reason = FailureAnalyzer.analyze(
            match_rate=match_rate,
            duration_diff=duration_diff,
            cut_duration=cut_duration,
            output_duration=output_duration,
            has_output=has_output,
            has_segments=has_segments,
            thresholds=self.thresholds
        )
        
        # 判断成功
        success = failure_reason is None
        
        result = {
            'cut_video': str(cut_video),
            'cut_video_name': cut_video.name,
            'output_video': str(output_path) if has_output else None,
            'output_video_name': output_path.name if has_output else None,
            'cut_duration': cut_duration,
            'output_duration': output_duration,
            'duration_diff': duration_diff,
            'match_rate': match_rate,
            'segments': len(segments) if segments else 0,
            'success': success,
            'error': failure_reason.message if failure_reason else None,
            'error_code': failure_reason.code if failure_reason else None,
            'error_details': failure_reason.details if failure_reason else None,
            'validation_level': self.validation_level.value
        }
        
        # 打印结果
        if success:
            print(f"   ✅ 成功")
            print(f"   📊 匹配率: {match_rate:.1f}%")
            print(f"   ⏱️  时长差异: {duration_diff:.2f}s")
        else:
            print(f"   ❌ 失败: {failure_reason.message}")
            if self.debug and failure_reason.details:
                print(f"   🔍 详情: {json.dumps(failure_reason.details, indent=2, ensure_ascii=False)}")
        
        return result

    def process_all(self, max_retries: int = 1) -> List[dict]:
        """处理所有裁剪视频"""
        
        cut_videos = self._get_cut_videos()
        print(f"🎬 找到 {len(cut_videos)} 个裁剪视频")
        print(f"🔄 最大重试次数: {max_retries}")

        if not cut_videos:
            print("❌ 没有找到裁剪视频")
            return []

        if not self.source_videos:
            print("❌ 没有找到源视频")
            return []

        results = []
        success_count = 0
        failure_reasons = {}

        for i, cut_video in enumerate(cut_videos, 1):
            print(f"\n📊 总进度: {i}/{len(cut_videos)} ({i/len(cut_videos)*100:.1f}%)")
            
            result = self.process_single_video(cut_video, max_attempts=max_retries)
            results.append(result)

            if result.get('success'):
                success_count += 1
            else:
                # 统计失败原因
                error_code = result.get('error_code', 'UNKNOWN')
                failure_reasons[error_code] = failure_reasons.get(error_code, 0) + 1

        # 保存详细报告
        self._save_report(results, success_count, len(cut_videos), failure_reasons)
        
        return results

    def _save_report(
        self,
        results: List[dict],
        success_count: int,
        total: int,
        failure_reasons: Dict[str, int]
    ):
        """保存详细报告"""
        
        report = {
            'total': total,
            'success': success_count,
            'failed': total - success_count,
            'success_rate': success_count / total * 100 if total else 0,
            'validation_level': self.validation_level.value,
            'thresholds': asdict(self.thresholds),
            'failure_analysis': failure_reasons,
            'results': results
        }
        
        report_path = self.output_dir / 'batch_reconstruction_report_v3.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print(f"\n{'='*70}")
        print(f"📊 批量处理完成")
        print(f"{'='*70}")
        print(f"总视频数: {total}")
        print(f"成功: {success_count}")
        print(f"失败: {total - success_count}")
        print(f"成功率: {success_count / total * 100:.1f}%")
        print(f"验证级别: {self.validation_level.value}")
        
        if failure_reasons:
            print(f"\n❌ 失败原因统计:")
            for code, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                print(f"   {code}: {count}")
        
        print(f"\n📋 详细报告: {report_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量视频重构工具 V3')
    parser.add_argument('--cut-dir', required=True, help='裁剪视频目录')
    parser.add_argument('--source-dir', required=True, help='源视频目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--validation', choices=['strict', 'normal', 'loose', 'best_effort'],
                       default='normal', help='验证级别')
    parser.add_argument('--retries', type=int, default=1, help='最大重试次数')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    validation_level = ValidationLevel(args.validation)
    
    print("="*70)
    print("🎬 批量视频重构工具 V3")
    print("="*70)
    print(f"裁剪视频目录: {args.cut_dir}")
    print(f"源视频目录: {args.source_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"验证级别: {args.validation}")
    print(f"重试次数: {args.retries}")
    print("="*70)

    batch = BatchVideoReconstructorV3(
        args.cut_dir,
        args.source_dir,
        args.output_dir,
        validation_level,
        args.debug
    )
    batch.process_all(max_retries=args.retries)


if __name__ == "__main__":
    main()
