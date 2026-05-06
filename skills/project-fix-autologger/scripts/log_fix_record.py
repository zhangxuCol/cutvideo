#!/usr/bin/env python3
"""写入带时间戳的项目修复记录。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


def _now() -> tuple[str, str]:
    dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.strftime("%Y%m%d_%H%M%S")


def _get_git_changed_files(cwd: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    files: list[str] = []
    for raw in proc.stdout.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        path = line[3:].strip() if len(line) >= 4 else line
        if path and path not in files:
            files.append(path)
    return files


def _contains_english_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _render_block(
    *,
    ts_text: str,
    issue_id: str,
    status: str,
    title: str,
    problem: str,
    root_cause: str,
    solution: str,
    result: str,
    verification: str,
    files: list[str],
    video_file: str,
    issue_time: str,
    clip_range: str,
    issue_detail: str,
    tags: list[str],
) -> str:
    files_text = ", ".join(files) if files else "无"
    tags_text = ", ".join(tags) if tags else "无"
    return (
        f"## [{ts_text}] {title}\n\n"
        f"- 时间戳: `{ts_text}`\n"
        f"- 问题编号: `{issue_id}`\n"
        f"- 状态: `{status}`\n"
        f"- 问题现象: {problem}\n"
        f"- 根因分析: {root_cause}\n"
        f"- 修改方法: {solution}\n"
        f"- 修复结果: {result}\n"
        f"- 验证方式: {verification}\n"
        f"- 涉及文件: `{files_text}`\n"
        f"- 视频文件: `{video_file}`\n"
        f"- 问题时间点: `{issue_time}`\n"
        f"- 裁剪区间: `{clip_range}`\n"
        f"- 视频问题详情: {issue_detail}\n"
        f"- 标签: `{tags_text}`\n\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="写入一条带时间戳的修复记录。")
    parser.add_argument("--workspace", default=".", help="项目根目录")
    parser.add_argument(
        "--log-dir",
        default="project_fix_records",
        help="记录目录（相对或绝对路径）",
    )
    parser.add_argument("--title", required=True, help="修复标题")
    parser.add_argument("--issue-id", default="", help="问题编号；不传时自动生成")
    parser.add_argument(
        "--status",
        default="已修复",
        choices=["待修复", "已修复"],
        help="记录状态：待修复 或 已修复",
    )
    parser.add_argument("--problem", required=True, help="问题现象")
    parser.add_argument("--root-cause", default="无", help="根因分析")
    parser.add_argument("--solution", required=True, help="修改方法")
    parser.add_argument("--result", required=True, help="修复结果")
    parser.add_argument("--verification", default="无", help="验证方式")
    parser.add_argument("--files", nargs="*", default=[], help="涉及文件")
    parser.add_argument(
        "--no-git-auto-files",
        action="store_true",
        help="不从 git status 自动推断改动文件",
    )
    parser.add_argument("--video-file", default="无", help="视频文件路径（如适用）")
    parser.add_argument(
        "--issue-time",
        default="无",
        help="问题出现时间点，例如 00:03:25",
    )
    parser.add_argument(
        "--clip-range",
        default="无",
        help="裁剪区间，例如 00:03:20-00:03:40",
    )
    parser.add_argument(
        "--issue-detail",
        default="无",
        help="视频/音频/字幕问题细节",
    )
    parser.add_argument("--tags", nargs="*", default=[], help="标签（可选）")
    parser.add_argument(
        "--allow-english",
        action="store_true",
        help="允许记录正文出现英文（默认不允许）",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = (workspace / log_dir).resolve()
    else:
        log_dir = log_dir.resolve()

    entries_dir = log_dir / "entries"
    by_date_dir = log_dir / "by-date"
    entries_dir.mkdir(parents=True, exist_ok=True)
    by_date_dir.mkdir(parents=True, exist_ok=True)

    ts_text, ts_compact = _now()
    day = ts_text.split(" ")[0]
    issue_id = args.issue_id.strip() or ts_compact

    files = [f for f in args.files if f.strip()]
    if not files and not args.no_git_auto_files:
        files = _get_git_changed_files(workspace)

    chinese_fields = [
        args.title.strip(),
        args.problem.strip(),
        (args.root_cause.strip() or "无"),
        args.solution.strip(),
        args.result.strip(),
        (args.verification.strip() or "无"),
        (args.issue_detail.strip() or "无"),
        " ".join([x for x in args.tags if x.strip()]),
    ]
    if not args.allow_english and any(_contains_english_text(x) for x in chinese_fields):
        print(
            "[修复日志] 检测到英文内容。根据要求，记录正文与示例需使用中文。"
            "如确需英文，请显式传入 --allow-english。",
        )
        return 4

    entry_md = _render_block(
        ts_text=ts_text,
        issue_id=issue_id,
        status=args.status.strip(),
        title=args.title.strip(),
        problem=args.problem.strip(),
        root_cause=args.root_cause.strip() or "无",
        solution=args.solution.strip(),
        result=args.result.strip(),
        verification=args.verification.strip() or "无",
        files=files,
        video_file=args.video_file.strip() or "无",
        issue_time=args.issue_time.strip() or "无",
        clip_range=args.clip_range.strip() or "无",
        issue_detail=args.issue_detail.strip() or "无",
        tags=[x for x in args.tags if x.strip()],
    )

    entry_path = entries_dir / f"{ts_compact}.md"
    history_md_path = log_dir / "fix-history.md"
    history_jsonl_path = log_dir / "fix-history.jsonl"
    daily_path = by_date_dir / f"{day}.md"

    entry_path.write_text(entry_md, encoding="utf-8")

    with history_md_path.open("a", encoding="utf-8") as f:
        if history_md_path.stat().st_size == 0:
            f.write("# 项目修复记录汇总\n\n")
        f.write(entry_md)

    record = {
        "时间戳": ts_text,
        "问题编号": issue_id,
        "状态": args.status.strip(),
        "标题": args.title.strip(),
        "问题现象": args.problem.strip(),
        "根因分析": args.root_cause.strip() or "无",
        "修改方法": args.solution.strip(),
        "修复结果": args.result.strip(),
        "验证方式": args.verification.strip() or "无",
        "涉及文件": files,
        "视频文件": args.video_file.strip() or "无",
        "问题时间点": args.issue_time.strip() or "无",
        "裁剪区间": args.clip_range.strip() or "无",
        "视频问题详情": args.issue_detail.strip() or "无",
        "标签": [x for x in args.tags if x.strip()],
        "记录文件": str(entry_path),
    }
    with history_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with daily_path.open("a", encoding="utf-8") as f:
        if daily_path.stat().st_size == 0:
            f.write(f"# {day} 修复记录\n\n")
        f.write(entry_md)

    print(f"[修复日志] 单条记录={entry_path}")
    print(f"[修复日志] 问题编号={issue_id}")
    print(f"[修复日志] 汇总Markdown={history_md_path}")
    print(f"[修复日志] 汇总JSONL={history_jsonl_path}")
    print(f"[修复日志] 按天归档={daily_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
