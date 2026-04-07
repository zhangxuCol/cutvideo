#!/bin/bash
# 自动修复循环 - 直到100%通过

cd /Users/zhangxu/work/项目/cutvideo

echo "=========================================="
echo "自动修复循环 - 目标100%通过"
echo "=========================================="

TARGET="/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/adx原/115196-1-363935819124715523.mp4"
OUTPUT="/Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_FINAL.mp4"

attempt=0
max_attempts=5

while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "=========================================="
    echo "尝试 $attempt/$max_attempts"
    echo "=========================================="
    
    # 运行重构
    if [ $attempt -eq 1 ]; then
        echo "运行 V3 Fast..."
        python3 v6_fast.py 2>&1 | tee /tmp/attempt_$attempt.log
        cp /Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_V3_FAST.mp4 "$OUTPUT"
    else
        echo "运行 V4 Ultimate..."
        python3 v6_ultimate_v4.py 2>&1 | tee /tmp/attempt_$attempt.log
        if [ -f /Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_V4_ULTIMATE.mp4 ]; then
            cp /Users/zhangxu/work/项目/cutvideo/01_test_data_generation/source_videos/南城以北/output_v6_base/115196_V4_ULTIMATE.mp4 "$OUTPUT"
        fi
    fi
    
    # 验证
    echo ""
    echo "进行一致性验证..."
    python3 -c "
from av_consistency_checker import AVConsistencyChecker
import sys

checker = AVConsistencyChecker('$TARGET', '$OUTPUT')
results = checker.check_consistency(interval=5.0)

poor = results['statistics']['poor']
if poor == 0:
    print('SUCCESS: 100%通过')
    sys.exit(0)
else:
    print(f'FAILED: {poor} 个问题点')
    sys.exit(1)
" 2>&1 | tee -a /tmp/attempt_$attempt.log
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅✅✅ 成功！100%通过！✅✅✅"
        echo "输出: $OUTPUT"
        echo "=========================================="
        exit 0
    fi
    
    echo ""
    echo "未通过，继续优化..."
done

echo ""
echo "=========================================="
echo "⚠️ 达到最大尝试次数"
echo "建议手动检查或使用更严格的参数"
echo "=========================================="
