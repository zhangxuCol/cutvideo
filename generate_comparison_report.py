#!/usr/bin/env python3
"""
生成最终对比报告 - 原素材 vs 二次裁剪素材
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, '/Users/zhangxu/work/项目/cutvideo/04_comparison_validation')
from compare_videos import compare_videos

def get_video_info(video_path):
    """获取视频信息"""
    import subprocess
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def main():
    adx_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原")
    output_dir = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output")
    
    # 读取 V5 报告
    report_path = Path("/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v5/reconstruction_report_v5.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("="*80)
    print("📊 原素材 vs 二次裁剪素材对比报告 (V5 版本)")
    print("="*80)
    print(f"\n生成时间: 2026-03-23")
    print(f"原素材目录: {adx_dir}")
    print(f"二次裁剪目录: {output_dir}")
    print(f"通过规则: 时长差异<5秒, 内容匹配率>90%")
    print()
    
    # 汇总统计
    total = report['total']
    success = report['success']
    perfect = report['perfect']
    
    print(f"{'='*80}")
    print(f"📈 总体统计")
    print(f"{'='*80}")
    print(f"   总视频数: {total}")
    print(f"   成功数: {success} ({report['success_rate']:.1f}%)")
    print(f"   完美匹配: {perfect} ({report['perfect_rate']:.1f}%)")
    print()
    
    # 详细对比表
    print(f"{'='*80}")
    print(f"📋 详细对比表")
    print(f"{'='*80}")
    print(f"{'序号':<4} {'原素材文件名':<45} {'原素材时长':<12} {'二次裁剪时长':<12} {'时长差异':<10} {'状态':<6}")
    print("-"*80)
    
    for i, result in enumerate(report['results'], 1):
        name = result['cut_video_name']
        orig_dur = result['cut_duration']
        out_dur = result['output_duration']
        diff = result['duration_diff']
        status = "✅通过" if result['success'] else "❌失败"
        
        print(f"{i:<4} {name:<45} {orig_dur:<12.2f} {out_dur:<12.2f} {diff:<10.2f} {status:<6}")
    
    # 失败项详情
    print(f"\n{'='*80}")
    print(f"❌ 失败项详情")
    print(f"{'='*80}")
    
    failed = [r for r in report['results'] if not r['success']]
    if failed:
        for result in failed:
            print(f"\n   视频: {result['cut_video_name']}")
            print(f"   原因: {result.get('error', '未知')}")
            print(f"   时长: {result['cut_duration']:.2f}s → {result['output_duration']:.2f}s")
            print(f"   差异: {result['duration_diff']:.2f}s")
    else:
        print("   无失败项")
    
    # 使用的源视频
    print(f"\n{'='*80}")
    print(f"📁 源视频使用情况")
    print(f"{'='*80}")
    
    all_sources = set()
    for result in report['results']:
        all_sources.update(result.get('source_videos_used', []))
    
    for source in sorted(all_sources):
        source_name = Path(source).name
        print(f"   • {source_name}")
    
    print(f"\n{'='*80}")
    print(f"✅ 报告生成完成")
    print(f"{'='*80}")
    print(f"\n注: V6 版本正在后台运行中，完成后将更新此报告。")

if __name__ == '__main__':
    main()
