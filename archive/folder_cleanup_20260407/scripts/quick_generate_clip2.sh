#!/bin/bash
# 快速生成裁剪视频2 - 使用FFmpeg直接裁剪

cd /Users/zhangxu/.openclaw/workspace/test_videos

echo "============================================================"
echo "快速生成裁剪视频2 (clip2_reconstructed.mp4)"
echo "============================================================"

# 创建临时目录
mkdir -p temp_clips

# 从 source_A 裁剪 0-10秒
echo "1. 从 source_A.mp4 裁剪 0-10秒..."
ffmpeg -y -hide_banner -loglevel error -i source_A.mp4 -ss 0 -to 10 -c:v copy -an temp_clips/seg1.mp4

# 从 source_B 裁剪 10-20秒
echo "2. 从 source_B.mp4 裁剪 10-20秒..."
ffmpeg -y -hide_banner -loglevel error -i source_B.mp4 -ss 10 -to 20 -c:v copy -an temp_clips/seg2.mp4

# 从 source_C 裁剪 15-25秒
echo "3. 从 source_C.mp4 裁剪 15-25秒..."
ffmpeg -y -hide_banner -loglevel error -i source_C.mp4 -ss 15 -to 25 -c:v copy -an temp_clips/seg3.mp4

# 合并片段
echo "4. 合并片段..."
echo -e "file 'temp_clips/seg1.mp4'\nfile 'temp_clips/seg2.mp4'\nfile 'temp_clips/seg3.mp4'" > concat_list.txt
ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i concat_list.txt -c:v copy -an temp_clips/temp_video.mp4

# 提取 clip1 的音频
echo "5. 提取 clip1 的音频..."
ffmpeg -y -hide_banner -loglevel error -i clip1.mp4 -vn -c:a copy temp_clips/audio.aac 2>/dev/null || \
ffmpeg -y -hide_banner -loglevel error -i clip1.mp4 -vn -c:a aac temp_clips/audio.aac

# 检查音频文件是否存在
if [ -f temp_clips/audio.aac ]; then
    # 合并视频和音频
    echo "6. 合并视频和音频，生成 clip2_reconstructed.mp4..."
    ffmpeg -y -hide_banner -loglevel error -i temp_clips/temp_video.mp4 -i temp_clips/audio.aac -c:v copy -c:a copy -shortest clip2_reconstructed.mp4
else
    # 没有音频，直接复制
    echo "6. 无音频，直接复制为 clip2_reconstructed.mp4..."
    cp temp_clips/temp_video.mp4 clip2_reconstructed.mp4
fi

# 清理临时文件
rm -rf temp_clips concat_list.txt

echo ""
echo "============================================================"
echo "✅ 裁剪视频2生成完成!"
echo "============================================================"
echo "文件: clip2_reconstructed.mp4"
ls -lh clip2_reconstructed.mp4

# 显示视频信息
echo ""
echo "视频信息:"
ffprobe -v error -show_entries format=duration -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 clip2_reconstructed.mp4
