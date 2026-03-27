#!/usr/bin/env python3
"""
V7 批量视频重构系统 - 优化版
确保输出视频与素材视频内容和音频完全一致
"""

import sys
sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/03_reconstruction_algorithms')

from pathlib import Path
import subprocess
import json
import tempfile
import shutil
from datetime import datetime
from video_reconstructor_hybrid_v7 import VideoReconstructorHybridV7

# 配置路径
ADX_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
SOURCE_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/剧集")
OUTPUT_DIR = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v7")
LOG_DIR = OUTPUT_DIR / "logs"

# 确保目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_duration(video_path):
    """获取视频时长"""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

def extract_audio_fingerprint(video_path, sample_rate=8000):
    """提取音频指纹用于对比"""
    import wave
    import struct
    import numpy as np
    
    temp_wav = Path(tempfile.gettempdir()) / f"audio_{Path(video_path).stem}.wav"
    
    # 提取音频
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-i', str(video_path), '-ar', str(sample_rate), '-ac', '1',
           str(temp_wav)]
    subprocess.run(cmd, capture_output=True)
    
    if not temp_wav.exists():
        return None
    
    # 读取音频数据
    with wave.open(str(temp_wav), 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        samples = struct.unpack(f'{n_frames}h', audio_data)
    
    # 清理临时文件
    temp_wav.unlink(missing_ok=True)
    
    return np.array(samples)

def compare_audio_fingerprints(fp1, fp2, threshold=0.85):
    """对比两个音频指纹的相似度"""
    if fp1 is None or fp2 is None:
        return 0.0
    
    # 对齐长度
    min_len = min(len(fp1), len(fp2))
    fp1 = fp1[:min_len]
    fp2 = fp2[:min_len]
    
    # 计算相关性
    if len(fp1) == 0:
        return 0.0
    
    correlation = np.corrcoef(fp1, fp2)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # 转换到 0-1 范围
    similarity = (correlation + 1) / 2
    
    return similarity

def extract_keyframes(video_path, num_frames=5):
    """提取关键帧用于对比"""
    duration = get_duration(video_path)
    if duration == 0:
        return []
    
    frames = []
    temp_dir = Path(tempfile.mkdtemp())
    
    # 在 0%, 25%, 50%, 75%, 100% 位置提取帧
    positions = [0, duration * 0.25, duration * 0.5, duration * 0.75, duration * 0.95]
    
    for i, pos in enumerate(positions):
        frame_path = temp_dir / f"frame_{i:03d}.jpg"
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-ss', str(pos), '-i', str(video_path),
               '-vframes', '1', '-q:v', '2', str(frame_path)]
        subprocess.run(cmd, capture_output=True)
        
        if frame_path.exists():
            frames.append(str(frame_path))
    
    return frames, temp_dir

def compare_frames_ssim(frames1, frames2):
    """使用 SSIM 对比两组帧"""
    from skimage.metrics import structural_similarity as ssim
    import cv2
    
    if len(frames1) != len(frames2):
        return 0.0
    
    scores = []
    for f1, f2 in zip(frames1, frames2):
        try:
            img1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                continue
            
            # 统一尺寸
            img1 = cv2.resize(img1, (320, 180))
            img2 = cv2.resize(img2, (320, 180))
            
            score = ssim(img1, img2)
            scores.append(score)
        except:
            continue
    
    return sum(scores) / len(scores) if scores else 0.0

def verify_output_quality(target_video, output_video):
    """
    验证输出视频与目标视频的一致性
    包括：时长、音频、关键帧
    """
    print(f"\n验证输出质量...")
    
    # 1. 时长检查
    duration1 = get_duration(target_video)
    duration2 = get_duration(output_video)
    duration_diff = abs(duration1 - duration2)
    
    if duration_diff > 1.0:
        return {
            "passed": False,
            "score": 0,
            "reason": f"时长不匹配: {duration1:.1f}s vs {duration2:.1f}s (差异 {duration_diff:.1f}s)",
            "checks": {"duration": False}
        }
    
    print(f"  ✓ 时长检查通过: {duration1:.1f}s vs {duration2:.1f}s")
    
    # 2. 音频对比
    print(f"  提取音频指纹...")
    fp1 = extract_audio_fingerprint(target_video)
    fp2 = extract_audio_fingerprint(output_video)
    
    audio_similarity = compare_audio_fingerprints(fp1, fp2)
    print(f"  音频相似度: {audio_similarity:.1%}")
    
    if audio_similarity < 0.85:
        return {
            "passed": False,
            "score": int(audio_similarity * 100),
            "reason": f"音频不匹配: 相似度 {audio_similarity:.1%} (需要 >85%)",
            "checks": {"duration": True, "audio": False}
        }
    
    print(f"  ✓ 音频检查通过")
    
    # 3. 关键帧对比
    print(f"  提取关键帧...")
    frames1, temp_dir1 = extract_keyframes(target_video)
    frames2, temp_dir2 = extract_keyframes(output_video)
    
    if len(frames1) == 0 or len(frames2) == 0:
        # 清理临时目录
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)
        return {
            "passed": False,
            "score": int(audio_similarity * 100),
            "reason": "无法提取关键帧",
            "checks": {"duration": True, "audio": True, "video": False}
        }
    
    video_similarity = compare_frames_ssim(frames1, frames2)
    print(f"  视频相似度: {video_similarity:.1%}")
    
    # 清理临时目录
    shutil.rmtree(temp_dir1, ignore_errors=True)
    shutil.rmtree(temp_dir2, ignore_errors=True)
    
    if video_similarity < 0.70:
        return {
            "passed": False,
            "score": int((audio_similarity + video_similarity) / 2 * 100),
            "reason": f"视频内容不匹配: 相似度 {video_similarity:.1%} (需要 >70%)",
            "checks": {"duration": True, "audio": True, "video": False}
        }
    
    print(f"  ✓ 视频检查通过")
    
    # 综合评分
    overall_score = (audio_similarity * 0.5 + video_similarity * 0.5) * 100
    
    return {
        "passed": True,
        "score": int(overall_score),
        "reason": f"验证通过: 音频 {audio_similarity:.1%}, 视频 {video_similarity:.1%}",
        "checks": {
            "duration": True,
            "audio": True,
            "video": True
        },
        "details": {
            "audio_similarity": audio_similarity,
            "video_similarity": video_similarity,
            "duration_diff": duration_diff
        }
    }

def process_single_video(target_video, output_path):
    """处理单个素材视频"""
    
    print(f"\n{'='*60}")
    print(f"处理: {target_video.name}")
    print(f"{'='*60}")
    
    target_duration = get_duration(target_video)
    print(f"目标时长: {target_duration:.1f}s")
    
    # 创建重构器
    reconstructor = VideoReconstructorHybridV7(
        target_video=str(target_video),
        source_videos=[str(SOURCE_DIR)],
        config={
            'audio_match_threshold': 0.6,
            'video_verify_threshold': 0.6
        }
    )
    
    try:
        # 执行重构
        segments = reconstructor.reconstruct(str(output_path), use_target_audio=True)
        
        if not segments:
            return {
                "success": False,
                "error": "未找到匹配片段",
                "segments": 0,
                "coverage": 0
            }
        
        # 计算覆盖率
        total_covered = sum(seg.end_time - seg.start_time for seg in segments)
        coverage = total_covered / target_duration if target_duration > 0 else 0
        
        print(f"\n重构完成:")
        print(f"  片段数: {len(segments)}")
        print(f"  覆盖率: {coverage:.1%}")
        
        # 验证输出质量
        verification = verify_output_quality(target_video, output_path)
        
        return {
            "success": True,
            "segments": len(segments),
            "coverage": coverage,
            "verification": verification,
            "output_path": str(output_path),
            "passed": verification.get("passed", False)
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "segments": 0,
            "coverage": 0
        }

def batch_process():
    """批量处理所有素材视频"""
    import numpy as np  # 确保导入
    
    print("="*60)
    print("V7 批量视频重构系统 - 优化版")
    print("="*60)
    print(f"素材目录: {ADX_DIR}")
    print(f"源视频目录: {SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 扫描所有素材视频
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    target_videos = [
        f for f in ADX_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    target_videos.sort()
    
    print(f"\n发现 {len(target_videos)} 个素材视频")
    
    # 批量处理
    results = []
    total = len(target_videos)
    passed_count = 0
    failed_count = 0
    
    for i, target_video in enumerate(target_videos, 1):
        print(f"\n\n[{i}/{total}] 处理: {target_video.name}")
        
        output_path = OUTPUT_DIR / f"{target_video.stem}_reconstructed.mp4"
        
        # 检查是否已处理过且通过验证
        if output_path.exists():
            print(f"  ⚠️  输出文件已存在，验证中...")
            verification = verify_output_quality(target_video, output_path)
            
            result = {
                "target": target_video.name,
                "status": "verified" if verification["passed"] else "failed",
                "verification": verification,
                "output_path": str(output_path)
            }
            
            if verification["passed"]:
                passed_count += 1
                print(f"  ✓ 验证通过")
            else:
                failed_count += 1
                print(f"  ✗ 验证失败: {verification['reason']}")
                # 重新处理
                print(f"  重新处理...")
                result = process_single_video(target_video, output_path)
                result["target"] = target_video.name
                result["status"] = "reprocessed"
                if result.get("passed"):
                    passed_count += 1
                else:
                    failed_count += 1
            
            results.append(result)
        else:
            # 新处理
            result = process_single_video(target_video, output_path)
            result["target"] = target_video.name
            result["status"] = "success" if result.get("passed") else "failed"
            
            if result.get("passed"):
                passed_count += 1
                print(f"  ✓ 处理完成并通过验证")
            else:
                failed_count += 1
                print(f"  ✗ 处理失败: {result.get('verification', {}).get('reason', result.get('error', 'Unknown'))}")
            
            results.append(result)
        
        # 保存进度
        save_progress(results, i, total, passed_count, failed_count)
    
    # 生成最终报告
    generate_report(results, passed_count, failed_count)
    
    return results

def save_progress(results, current, total, passed, failed):
    """保存处理进度"""
    progress_file = LOG_DIR / f"progress_{datetime.now().strftime('%Y%m%d')}.json"
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "current": current,
        "total": total,
        "passed": passed,
        "failed": failed,
        "results": results
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

def generate_report(results, passed_count, failed_count):
    """生成处理报告"""
    report_file = LOG_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("V7 批量处理报告 - 优化版\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计处理: {len(results)} 个视频\n")
        f.write(f"通过验证: {passed_count}\n")
        f.write(f"验证失败: {failed_count}\n")
        f.write(f"通过率: {passed_count/len(results)*100:.1f}%\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"\n素材: {result['target']}\n")
            f.write(f"状态: {result['status']}\n")
            
            if result.get("success"):
                f.write(f"片段数: {result.get('segments', 0)}\n")
                f.write(f"覆盖率: {result.get('coverage', 0):.1%}\n")
            
            if "verification" in result:
                v = result["verification"]
                f.write(f"验证: {v.get('reason', 'N/A')}\n")
                f.write(f"评分: {v.get('score', 0)}\n")
                if v.get("passed"):
                    f.write(f"结果: ✓ 通过\n")
                else:
                    f.write(f"结果: ✗ 失败\n")
            
            if "error" in result:
                f.write(f"错误: {result['error']}\n")
            
            f.write("-"*40 + "\n")
    
    print(f"\n{'='*60}")
    print("批量处理完成!")
    print(f"{'='*60}")
    print(f"总计: {len(results)}")
    print(f"通过: {passed_count} ({passed_count/len(results)*100:.1f}%)")
    print(f"失败: {failed_count}")
    print(f"报告: {report_file}")

if __name__ == "__main__":
    batch_process()
