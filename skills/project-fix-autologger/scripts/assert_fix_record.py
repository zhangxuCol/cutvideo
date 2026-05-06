#!/usr/bin/env python3
"""检查最近修复记录是否覆盖预期文件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _normalize_path(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("\\", "/")
    return text.lstrip("./")


def _load_last_record(jsonl_path: Path) -> dict | None:
    if not jsonl_path.exists():
        return None
    lines = [line.strip() for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="检查最近一条修复记录是否覆盖指定文件。")
    parser.add_argument("--workspace", default=".", help="项目根目录")
    parser.add_argument("--log-dir", default="project_fix_records", help="记录目录（相对或绝对路径）")
    parser.add_argument("--expect-files", nargs="+", required=True, help="本次修复应覆盖的文件")
    parser.add_argument(
        "--expect-status",
        default="已修复",
        choices=["待修复", "已修复"],
        help="期望最近一条记录的状态",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = (workspace / log_dir).resolve()
    else:
        log_dir = log_dir.resolve()

    jsonl_path = log_dir / "fix-history.jsonl"
    last_record = _load_last_record(jsonl_path)
    if last_record is None:
        print("[修复日志检查] 未找到任何修复记录。")
        return 2

    last_status = str(last_record.get("状态", "")).strip()
    if last_status != str(args.expect_status).strip():
        print(
            f"[修复日志检查] 最近一条记录状态不符合预期："
            f"当前为 {last_status or '无'}，期望为 {args.expect_status}。"
        )
        print(f"[修复日志检查] 最近记录：{last_record.get('记录文件', '无')}")
        return 4

    expected = [_normalize_path(item) for item in args.expect_files if str(item).strip()]
    recorded_files = [_normalize_path(item) for item in last_record.get("涉及文件", [])]

    missing: list[str] = []
    for item in expected:
        if item in recorded_files:
            continue
        if any(rec.endswith(item) for rec in recorded_files):
            continue
        missing.append(item)

    if missing:
        print("[修复日志检查] 最近一条记录未覆盖以下文件：")
        for item in missing:
            print(f"- {item}")
        print(f"[修复日志检查] 最近记录：{last_record.get('记录文件', '无')}")
        return 3

    print("[修复日志检查] 通过")
    print(f"[修复日志检查] 最近记录：{last_record.get('记录文件', '无')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
