#!/bin/bash
# 重新创建clip1，使用正确的片段时长（每个片段约7.67秒）

cd /Users/zhangxu/.openclaw/workspace/test_videos

echo "============================================================"
echo "🎬 重新创建clip1（使用7.67秒片段）"
echo "============================================================"

rm -rf temp_clip1_v2
mkdir -p temp_clip1_v2

echo "1. 从 video1 裁剪 0-7.67秒..."
ffmpeg -y -hide_banner -loglevel error \
    -i source_videos/video1_天赋变异后我无敌了_ep1.mp4 \
    -ss 0 -t 7.67 \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    temp_clip1_v2/seg1.mp4

echo "2. 从 video2 裁剪 30-37.67秒..."
ffmpeg -y -hide_banner -loglevel error \
    -i source_videos/video2_开局饕餮血统我吞噬一切_ep1.mp4 \
    -ss 30 -t 7.67 \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    temp_clip1_v2/seg2.mp4

echo "3. 从 video3 裁剪 60-67.67秒..."
ffmpeg -y -hide_banner -loglevel error \
    -i source_videos/video3_咒术反噬我有无限血条_ep1.mp4 \
    -ss 60 -t 7.67 \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    temp_clip1_v2/seg3.mp4

echo "4. 合并片段..."
cat > temp_clip1_v2/concat_list.txt << EOF
file 'seg1.mp4'
file 'seg2.mp4'
file 'seg3.mp4'
EOF

ffmpeg -y -hide_banner -loglevel error \
    -f concat -safe 0 -i temp_clip1_v2/concat_list.txt \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    clip1_real_v2.mp4

rm -rf temp_clip1_v2

echo ""
echo "============================================================"
echo "✅ clip1_v2 创建完成!"
echo "============================================================"
ls -lh clip1_real_v2.mp4

echo ""
echo "视频信息:"
ffprobe -v error -show_entries format=duration -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 clip1_real_v2.mp4
