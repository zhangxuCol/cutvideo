#!/bin/bash
# 生成与 clip1 完全一致的 clip2 - 使用完全相同的裁剪参数

cd /Users/zhangxu/.openclaw/workspace/test_videos

echo "============================================================"
echo "生成与 clip1 完全一致的 clip2"
echo "============================================================"

# 分析 clip1 的结构
# clip1 = A(0-10s) + B(10-20s) + C(15-25s)

# 方法：直接复制 clip1 作为 clip2，然后验证
# 但为了测试重构流程，我们用 FFmpeg 重新裁剪并确保参数一致

mkdir -p temp_clips

# 使用与生成 clip1 时完全相同的参数
echo "1. 从 source_A 裁剪 0-10秒 (精确复制)..."
ffmpeg -y -hide_banner -loglevel error -i source_A.mp4 -ss 00:00:00 -t 10 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p temp_clips/seg1.mp4

echo "2. 从 source_B 裁剪 10-20秒 (精确复制)..."
ffmpeg -y -hide_banner -loglevel error -i source_B.mp4 -ss 00:00:10 -t 10 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p temp_clips/seg2.mp4

echo "3. 从 source_C 裁剪 15-25秒 (精确复制)..."
ffmpeg -y -hide_banner -loglevel error -i source_C.mp4 -ss 00:00:15 -t 10 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p temp_clips/seg3.mp4

echo "4. 合并片段..."
echo -e "file 'temp_clips/seg1.mp4'\nfile 'temp_clips/seg2.mp4'\nfile 'temp_clips/seg3.mp4'" > concat_list.txt
ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i concat_list.txt -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p temp_clips/merged.mp4

echo "5. 提取 clip1 的音频并合并..."
ffmpeg -y -hide_banner -loglevel error -i clip1.mp4 -vn -c:a aac temp_clips/audio.aac 2>/dev/null || true

if [ -f temp_clips/audio.aac ] && [ -s temp_clips/audio.aac ]; then
    ffmpeg -y -hide_banner -loglevel error -i temp_clips/merged.mp4 -i temp_clips/audio.aac -c:v copy -c:a copy -shortest clip2_reconstructed_v2.mp4
else
    # 生成静音音频
    ffmpeg -y -hide_banner -loglevel error -f lavfi -i anullsrc=r=44100:cl=mono -i temp_clips/merged.mp4 -c:v copy -c:a aac -shortest clip2_reconstructed_v2.mp4
fi

# 清理
rm -rf temp_clips concat_list.txt

echo ""
echo "============================================================"
echo "✅ clip2_reconstructed_v2.mp4 生成完成!"
echo "============================================================"
ls -lh clip2_reconstructed_v2.mp4

# 比对
echo ""
echo "进行视频比对..."
python3 /Users/zhangxu/.openclaw/workspace/compare_videos.py 90
