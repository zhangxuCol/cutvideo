#!/usr/bin/env python3
"""
从剧点数据库下载视频
"""

import sqlite3
import os
import requests
from pathlib import Path
from urllib.parse import urlparse

# 数据库路径
DB_PATH = os.path.expanduser("~/work/项目/爬虫/dianzhong/judian/judianiaaiap.db")

# 视频保存目录
DOWNLOAD_DIR = Path(os.path.expanduser("~/.openclaw/workspace/real_videos"))
DOWNLOAD_DIR.mkdir(exist_ok=True)


def download_video(url: str, output_path: Path) -> bool:
    """下载视频"""
    try:
        print(f"Downloading: {url[:80]}...")
        
        # 设置请求头 - 模拟浏览器
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://video.818watch.com/',
            'Origin': 'https://video.818watch.com'
        }
        
        # 创建 session
        session = requests.Session()
        session.headers.update(headers)
        
        # 流式下载
        response = session.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / 1024 / 1024:.2f} MB")
        
        # 保存文件
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024) == 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Progress: {progress:.1f}%")
        
        print(f"✅ Downloaded: {output_path.name}")
        print(f"Size: {downloaded / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def get_video_info(db_path: str):
    """从数据库获取一个视频信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取一个有视频链接的剧集
    cursor.execute("""
        SELECT 
            d.book_id,
            d.book_name,
            e.chapter_id,
            e.chapter_index,
            e.play_url
        FROM drama_episode e
        JOIN drama_info d ON e.book_id = d.book_id
        WHERE e.play_url != '' AND e.play_url IS NOT NULL
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'book_id': result[0],
            'book_name': result[1],
            'chapter_id': result[2],
            'chapter_index': result[3],
            'play_url': result[4]
        }
    return None


def main():
    print("="*60)
    print("🎬 剧点视频下载工具")
    print("="*60)
    
    # 检查数据库
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return 1
    
    print(f"Database: {DB_PATH}")
    print(f"Download dir: {DOWNLOAD_DIR}")
    
    # 获取视频信息
    print("\nFetching video info...")
    video_info = get_video_info(DB_PATH)
    
    if not video_info:
        print("❌ No video found in database")
        return 1
    
    print(f"\nFound video:")
    print(f"  Book: {video_info['book_name']}")
    print(f"  Chapter: {video_info['chapter_index']}")
    print(f"  URL: {video_info['play_url'][:60]}...")
    
    # 生成文件名
    safe_name = "".join(c for c in video_info['book_name'] if c.isalnum() or c in (' ', '-', '_')).strip()
    filename = f"{safe_name}_ep{video_info['chapter_index']}.mp4"
    output_path = DOWNLOAD_DIR / filename
    
    print(f"\nOutput: {output_path}")
    
    # 下载视频
    print("\n" + "="*60)
    success = download_video(video_info['play_url'], output_path)
    print("="*60)
    
    if success:
        print(f"\n✅ Download complete!")
        print(f"File: {output_path}")
        return 0
    else:
        print(f"\n❌ Download failed")
        return 1


if __name__ == "__main__":
    exit(main())
