#!/usr/bin/env python3
"""
音视频一致性验证工具
每隔5秒提取画面和音频，验证是否一致
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil
import wave
import struct
from typing import List, Tuple, Dict
import json

class AVConsistencyChecker:
    """
    音视频一致性检查器
    每隔指定间隔验证画面和音频
    """
    
    def __init__(self, target_video: str, reconstructed_video: str):
        self.target_video = Path(target_video)
        self.reconstructed_video = Path(reconstructed_video)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results = []
        
    def get_video_duration(self, video_path: Path) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def extract_frame(self, video_path: Path, time_sec: float, output_path: Path):
        """提取帧"""
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', 'scale=480:270',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
    
    def extract_audio_clip(self, video_path: Path, start: float, duration: float, output_path: Path):
        """提取音频片段"""
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(start), '-t', str(duration),
            '-i', str(video_path), '-vn',
            '-acodec', 'aac', '-b:a', '64k',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
    
    def calculate_frame_similarity(self, frame1_path: Path, frame2_path: Path) -> float:
        """计算两帧相似度"""
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 调整大小
        img1 = cv2.resize(img1, (320, 180))
        img2 = cv2.resize(img2, (320, 180))
        
        # 灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 直方图相似度
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        template_sim = np.max(result)
        
        return 0.5 * max(0, hist_sim) + 0.5 * template_sim
    
    def extract_audio_fingerprint(self, video_path: Path, start: float, duration: float) -> np.ndarray:
        """提取音频指纹"""
        temp_wav = self.temp_dir / f"fp_{start:.0f}.wav"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(start), '-t', str(duration),
            '-i', str(video_path), '-vn',
            '-acodec', 'pcm_s16le', '-ar', '8000', '-ac', '1',
            str(temp_wav)
        ]
        subprocess.run(cmd, capture_output=True)
        
        if not temp_wav.exists():
            return np.array([])
        
        with wave.open(str(temp_wav), 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            samples = struct.unpack(f'{n_frames}h', audio_data)
        
        # 提取特征（每秒一个特征）
        samples_per_sec = 8000
        n_blocks = len(samples) // samples_per_sec
        features = []
        
        for i in range(n_blocks):
            block = samples[i * samples_per_sec:(i + 1) * samples_per_sec]
            fft = np.fft.rfft(block)
            magnitude = np.abs(fft)
            # 取前20个频带均值
            bands = np.array([np.mean(magnitude[j:j+len(magnitude)//20]) 
                            for j in range(0, len(magnitude), len(magnitude)//20)])
            features.append(bands[:20])
        
        return np.array(features)
    
    def calculate_audio_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """计算音频指纹相似度"""
        if len(fp1) == 0 or len(fp2) == 0:
            return 0.0
        
        min_len = min(len(fp1), len(fp2))
        fp1 = fp1[:min_len]
        fp2 = fp2[:min_len]
        
        correlations = []
        for f1, f2 in zip(fp1, fp2):
            if len(f1) == len(f2) and np.std(f1) > 0 and np.std(f2) > 0:
                corr = np.corrcoef(f1, f2)[0, 1]
                correlations.append(corr)
            else:
                correlations.append(0)
        
        return np.mean(correlations) if correlations else 0.0
    
    def check_consistency(self, interval: float = 5.0) -> Dict:
        """
        每隔interval秒检查一次一致性
        """
        print(f"\n{'='*70}")
        print(f"🎬🔊 音视频一致性检查")
        print(f"   检查间隔: {interval}秒")
        print(f"{'='*70}")
        
        target_duration = self.get_video_duration(self.target_video)
        recon_duration = self.get_video_duration(self.reconstructed_video)
        
        print(f"\n📹 原始视频: {target_duration:.1f}s")
        print(f"📹 重构视频: {recon_duration:.1f}s")
        print(f"⏱️  时长差异: {abs(target_duration - recon_duration):.1f}s")
        
        # 确定检查点数
        check_duration = min(target_duration, recon_duration)
        check_times = np.arange(0, check_duration, interval)
        
        print(f"\n🔍 将检查 {len(check_times)} 个时间点")
        print(f"{'='*70}\n")
        
        results = []
        
        for i, check_time in enumerate(check_times):
            print(f"\n{'─'*70}")
            print(f"🔍 检查点 {i+1}/{len(check_times)}: {check_time:.1f}s")
            print(f"{'─'*70}")
            
            # 提取画面
            target_frame = self.temp_dir / f"t_{check_time:.0f}.jpg"
            recon_frame = self.temp_dir / f"r_{check_time:.0f}.jpg"
            
            self.extract_frame(self.target_video, check_time, target_frame)
            self.extract_frame(self.reconstructed_video, check_time, recon_frame)
            
            # 计算画面相似度
            if target_frame.exists() and recon_frame.exists():
                frame_sim = self.calculate_frame_similarity(target_frame, recon_frame)
            else:
                frame_sim = 0.0
            
            # 提取音频指纹（取2秒片段）
            target_fp = self.extract_audio_fingerprint(
                self.target_video, check_time, 2.0
            )
            recon_fp = self.extract_audio_fingerprint(
                self.reconstructed_video, check_time, 2.0
            )
            
            # 计算音频相似度
            audio_sim = self.calculate_audio_similarity(target_fp, recon_fp)
            
            # 综合评分
            combined_sim = 0.5 * frame_sim + 0.5 * audio_sim
            
            # 判定结果
            if combined_sim >= 0.95:
                status = "✅ 优秀"
            elif combined_sim >= 0.85:
                status = "✅ 良好"
            elif combined_sim >= 0.70:
                status = "⚠️  一般"
            else:
                status = "❌ 不匹配"
            
            result = {
                'time': float(check_time),
                'frame_similarity': float(frame_sim),
                'audio_similarity': float(audio_sim),
                'combined_similarity': float(combined_sim),
                'status': status,
                'target_frame': str(target_frame),
                'recon_frame': str(recon_frame)
            }
            results.append(result)
            
            # 输出结果
            print(f"   🖼️  画面相似度: {frame_sim:.1%}")
            print(f"   🔊 音频相似度: {audio_sim:.1%}")
            print(f"   📊 综合相似度: {combined_sim:.1%}")
            print(f"   {status}")
            
            # 如果不匹配，显示截图路径
            if combined_sim < 0.70:
                print(f"   ⚠️  请检查:")
                print(f"      原始: {target_frame}")
                print(f"      重构: {recon_frame}")
        
        # 统计
        excellent = sum(1 for r in results if r['combined_similarity'] >= 0.95)
        good = sum(1 for r in results if 0.85 <= r['combined_similarity'] < 0.95)
        fair = sum(1 for r in results if 0.70 <= r['combined_similarity'] < 0.85)
        poor = sum(1 for r in results if r['combined_similarity'] < 0.70)
        
        avg_frame = np.mean([r['frame_similarity'] for r in results])
        avg_audio = np.mean([r['audio_similarity'] for r in results])
        avg_combined = np.mean([r['combined_similarity'] for r in results])
        
        # 输出统计
        print(f"\n{'='*70}")
        print(f"📊 检查结果统计")
        print(f"{'='*70}")
        print(f"\n总检查点数: {len(results)}")
        print(f"\n✅ 优秀 (≥95%): {excellent} ({excellent/len(results)*100:.1f}%)")
        print(f"✅ 良好 (85-95%): {good} ({good/len(results)*100:.1f}%)")
        print(f"⚠️  一般 (70-85%): {fair} ({fair/len(results)*100:.1f}%)")
        print(f"❌ 不匹配 (<70%): {poor} ({poor/len(results)*100:.1f}%)")
        print(f"\n平均画面相似度: {avg_frame:.1%}")
        print(f"平均音频相似度: {avg_audio:.1%}")
        print(f"平均综合相似度: {avg_combined:.1%}")
        
        # 保存详细报告
        report_path = self.temp_dir / "consistency_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'target_video': str(self.target_video),
                'reconstructed_video': str(self.reconstructed_video),
                'target_duration': target_duration,
                'recon_duration': recon_duration,
                'check_interval': interval,
                'statistics': {
                    'total_checks': len(results),
                    'excellent': excellent,
                    'good': good,
                    'fair': fair,
                    'poor': poor,
                    'avg_frame_similarity': float(avg_frame),
                    'avg_audio_similarity': float(avg_audio),
                    'avg_combined_similarity': float(avg_combined)
                },
                'details': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存: {report_path}")
        print(f"{'='*70}\n")
        
        return {
            'statistics': {
                'total': len(results),
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor,
                'avg_frame': avg_frame,
                'avg_audio': avg_audio,
                'avg_combined': avg_combined
            },
            'details': results,
            'report_path': str(report_path)
        }
    
    def cleanup(self):
        """清理临时文件"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def main():
    """主函数"""
    target = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
    recon = "/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_AV_VERIFIED.mp4"
    
    print("="*70)
    print("115196 音视频一致性检查")
    print("="*70)
    print(f"原始视频: {target}")
    print(f"重构视频: {recon}")
    print("="*70)
    
    checker = AVConsistencyChecker(target, recon)
    
    try:
        results = checker.check_consistency(interval=5.0)
        
        # 输出关键时间点的不匹配项
        poor_results = [r for r in results['details'] if r['combined_similarity'] < 0.70]
        if poor_results:
            print(f"\n⚠️  发现 {len(poor_results)} 个不匹配的检查点:")
            for r in poor_results:
                print(f"   {r['time']:.1f}s: 综合相似度 {r['combined_similarity']:.1%}")
        else:
            print(f"\n✅ 所有检查点都达到良好或以上标准！")
            
    finally:
        print(f"\n📁 临时文件保留在: {checker.temp_dir}")
        print(f"   你可以查看所有对比截图和详细报告")


if __name__ == "__main__":
    main()
