#!/usr/bin/env python3
"""
自动运行视频重构脚本 - 跳过确认提示
"""

import sys
sys.path.insert(0, '/Users/zhangxu/.openclaw/workspace')

from pathlib import Path
import yaml

# 修改标准输入，自动回答 'y'
class AutoConfirmInput:
    def __init__(self):
        self.count = 0
    
    def readline(self):
        self.count += 1
        print("y")  # 输出到屏幕
        return "y\n"

# 替换 input 函数
def auto_input(prompt=""):
    print(prompt, end="")
    return "y"

# 保存原始 input
original_input = __builtins__.input if hasattr(__builtins__, 'input') else input

# 导入并运行主程序
if __name__ == "__main__":
    # 先修改配置文件确保路径正确
    config_file = Path("/Users/zhangxu/.openclaw/workspace/video_reconstruct_config.yaml")
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新为绝对路径
        test_dir = Path("/Users/zhangxu/.openclaw/workspace/test_videos").absolute()
        config['target_video'] = str(test_dir / "clip1.mp4")
        config['source_videos'] = [
            str(test_dir / "source_A.mp4"),
            str(test_dir / "source_B.mp4"),
            str(test_dir / "source_C.mp4")
        ]
        config['output_video'] = str(test_dir / "clip2_reconstructed.mp4")
        config['fps'] = 5
        config['similarity_threshold'] = 0.90
        config['max_retries'] = 3
        config['use_target_audio'] = True
        config['min_segment_duration'] = 0.5
        config['match_threshold'] = 0.6
        config['missing_segment_strategy'] = "smart_fill"
        config['smart_fill_threshold'] = 0.5
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        
        print(f"✅ 配置文件已更新")
        print(f"   目标视频: {config['target_video']}")
        print(f"   输出视频: {config['output_video']}")
    
    # 运行主程序
    print("\n" + "="*60)
    print("开始自动运行视频重构...")
    print("="*60 + "\n")
    
    exec(open('/Users/zhangxu/.openclaw/workspace/multi_source_reconstructor_config.py').read())
