#!/bin/bash
# 直接从源视频的正确时间戳裁剪生成clip2

cd /Users/zhangxu/.openclaw/workspace/test_videos

echo "============================================================"
echo "🎬 创建裁剪视频2 (使用正确的时间戳)"
echo "============================================================"

rm -rf temp_correct
mkdir -p temp_correct

echo "1. 从 video1 裁剪 0-15秒 (片段1)..."
ffmpeg -y -hide_banner -loglevel error \
    -i source_videos/video1_天赋变异后我无敌了_ep1.mp4 \
    -ss 0 -t 15 \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    temp_correct/seg1.mp4

echo "2. 从 video2 裁剪 30-45秒 (片段2)..."
ffmpeg -y -hide_banner -loglevel error \
    -i source_videos/video2_开局饕餮血统我吞噬一切_ep1.mp4 \
    -ss 30 -t 15 \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    temp_correct/seg2.mp4

echo "3. 从 video3 裁剪 60-75秒 (片段3)..."
ffmpeg -y -hide_banner -loglevel error \
    -i source_videos/video3_咒术反噬我有无限血条_ep1.mp4 \
    -ss 60 -t 15 \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    temp_correct/seg3.mp4

echo "4. 合并片段..."
cat > temp_correct/concat_list.txt << EOF
file 'seg1.mp4'
file 'seg2.mp4'
file 'seg3.mp4'
EOF

ffmpeg -y -hide_banner -loglevel error \
    -f concat -safe 0 -i temp_correct/concat_list.txt \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    clip2_reconstructed_real.mp4

rm -rf temp_correct

echo ""
echo "============================================================"
echo "✅ 裁剪视频2创建完成!"
echo "============================================================"
ls -lh clip2_reconstructed_real.mp4

echo ""
echo "视频信息:"
ffprobe -v error -show_entries format=duration -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 clip2_reconstructed_real.mp4
