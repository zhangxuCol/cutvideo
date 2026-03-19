#!/bin/bash
# 使用 FFmpeg 下载 m3u8 视频

URL="https://1500027887.vod-qcloud.com/a29cfff7vodtranssh1500027887/c6d633b45145403697265955667/v.f1458548.m3u8?t=69b999f1&us=69b97e998acf2fbd0302e0fc&sign=f163ba1cec1008aaafbfae618b665ee9"
OUTPUT="/Users/zhangxu/.openclaw/workspace/real_videos/video_ep1.mp4"

echo "Downloading video with FFmpeg..."
echo "URL: $URL"
echo "Output: $OUTPUT"

ffmpeg -headers "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
       -headers "Referer: https://video.818watch.com/" \
       -i "$URL" \
       -c copy \
       -bsf:a aac_adtstoasc \
       "$OUTPUT" 2>&1 | tail -20

echo "Done!"
ls -lh "$OUTPUT"
