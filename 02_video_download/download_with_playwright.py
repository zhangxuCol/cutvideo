#!/usr/bin/env python3
"""
使用 Playwright 重新获取视频链接并下载
"""

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from playwright.async_api import async_playwright

# 配置
DB_PATH = os.path.expanduser("~/work/项目/爬虫/dianzhong/judian/judianiaaiap.db")
DOWNLOAD_DIR = Path(os.path.expanduser("~/.openclaw/workspace/real_videos"))
DOWNLOAD_DIR.mkdir(exist_ok=True)


async def get_video_url_from_page(page, book_id: int, chapter_id: int) -> str:
    """从页面获取视频真实地址"""
    try:
        # 构造视频播放页面URL
        video_url = f"https://video.818watch.com/#/video-manage/video-list/video-detail?bookId={book_id}&chapterId={chapter_id}"
        
        print(f"Navigating to: {video_url}")
        await page.goto(video_url, wait_until="networkidle")
        
        # 等待视频加载
        await page.wait_for_timeout(3000)
        
        # 尝试获取视频地址
        # 方法1: 从 video 标签获取
        video_src = await page.eval_on_selector("video", "el => el.src")
        if video_src:
            return video_src
        
        # 方法2: 从网络请求拦截（需要提前设置）
        # 这里简化处理，返回None
        return None
        
    except Exception as e:
        print(f"Error getting video URL: {e}")
        return None


async def download_one_video():
    """下载一个视频"""
    # 从数据库获取一个视频信息
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT d.book_id, d.book_name, e.chapter_id, e.chapter_index
        FROM drama_episode e
        JOIN drama_info d ON e.book_id = d.book_id
        WHERE e.play_url != '' AND e.play_url IS NOT NULL
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        print("No video found in database")
        return
    
    book_id, book_name, chapter_id, chapter_index = result
    print(f"Found: {book_name} - Chapter {chapter_index}")
    
    # 启动浏览器获取真实视频地址
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        # 登录
        print("Logging in...")
        await page.goto("https://video.818watch.com/login")
        await page.fill('input[placeholder="请输入账号"]', "中文在线巨量端原生-取链接")
        await page.fill('input[placeholder="请输入密码"]', "8y529c7C")
        await page.click("button:has-text('登录')")
        await page.wait_for_timeout(3000)
        
        # 获取视频地址
        video_url = await get_video_url_from_page(page, book_id, chapter_id)
        
        await browser.close()
        
        if video_url:
            print(f"Got video URL: {video_url}")
            # 这里可以添加下载逻辑
        else:
            print("Failed to get video URL")


if __name__ == "__main__":
    asyncio.run(download_one_video())
