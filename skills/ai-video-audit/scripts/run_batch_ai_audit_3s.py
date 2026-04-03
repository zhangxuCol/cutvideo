#!/usr/bin/env python3
"""
批量执行 3 秒间隔 AI 审片，并生成总汇总报告页面。
"""

import argparse
import html
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_config import (
    load_section_config,
    cfg_str,
    cfg_int,
    cfg_float,
    cfg_bool,
)


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_materials(material_dir: Path) -> List[Path]:
    return sorted([p for p in material_dir.glob("*.mp4") if p.is_file()])


def find_candidate(material: Path, candidate_dir: Path) -> Optional[Path]:
    direct = candidate_dir / material.name
    if direct.exists():
        return direct

    candidates = [
        candidate_dir / f"{material.stem}_V3_FAST.mp4",
        candidate_dir / f"{material.stem}_cut.mp4",
        candidate_dir / f"{material.stem}_reconstructed.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def reconstruct_with_v6_fast(
    repo_root: Path,
    target: Path,
    source_dir: Path,
    output: Path,
    cache_dir: Path,
    workers: int,
    segment_duration: float,
    low_score_threshold: float,
    force_target_audio: bool,
    config_path: str = "",
) -> Tuple[bool, str]:
    ensure_dir(output.parent)
    ensure_dir(cache_dir)
    cmd = [
        "python", str(repo_root / "v6_fast.py"),
        "--target", str(target),
        "--source-dir", str(source_dir),
        "--output", str(output),
        "--cache", str(cache_dir),
        "--workers", str(max(0, int(workers))),
        "--segment-duration", str(float(segment_duration)),
        "--low-score-threshold", str(float(low_score_threshold)),
    ]
    if force_target_audio:
        cmd.append("--force-target-audio")
    if config_path:
        cmd.extend(["--config", str(config_path)])
    result = run_cmd(cmd)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "v6_fast_failed").strip()
        return False, msg
    if not output.exists():
        return False, "v6_fast_no_output"
    return True, ""


def run_single_audit(
    repo_root: Path,
    target: Path,
    candidate: Path,
    output_dir: Path,
    interval: float,
    clip_duration: float,
    max_points: int,
    asr_mode: str,
    asr_cmd: str,
    asr_python: str,
    asr_model: str,
    language: str,
    target_sub: str,
    candidate_sub: str,
    clip_elapsed_sec: float = 0.0,
    candidate_mode: str = "existing",
    config_path: str = "",
) -> Tuple[bool, str]:
    ensure_dir(output_dir)
    script = repo_root / "skills" / "ai-video-audit" / "scripts" / "build_ai_video_audit_bundle.py"
    cmd = [
        "python", str(script),
        "--target", str(target),
        "--candidate", str(candidate),
        "--interval", str(interval),
        "--clip-duration", str(clip_duration),
        "--max-points", str(max_points),
        "--asr", asr_mode,
        "--asr-model", asr_model,
        "--language", language,
        "--clip-elapsed-sec", str(float(clip_elapsed_sec)),
        "--candidate-mode", str(candidate_mode),
        "--output-dir", str(output_dir),
    ]
    if asr_cmd:
        cmd.extend(["--asr-cmd", asr_cmd])
    if asr_python:
        cmd.extend(["--asr-python", asr_python])
    if target_sub:
        cmd.extend(["--target-sub", target_sub])
    if candidate_sub:
        cmd.extend(["--candidate-sub", candidate_sub])
    if config_path:
        cmd.extend(["--config", str(config_path)])

    result = run_cmd(cmd)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "audit_failed").strip()

    manifest = output_dir / "audit_manifest.json"
    if not manifest.exists():
        return False, "audit_manifest_missing"
    return True, ""


def build_total_report(
    output_root: Path,
    rows: List[Dict],
    started_at: str,
    finished_at: str,
    interval: float,
    material_dir: Path,
    candidate_dir: Path,
    source_dir: Optional[Path],
) -> Tuple[Path, Path]:
    summary_json = output_root / "audit_total_summary.json"
    summary_html = output_root / "audit_total_summary.html"

    total = len(rows)
    success = sum(1 for r in rows if r.get("status") == "ok")
    failed = total - success

    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "interval": interval,
        "material_dir": str(material_dir),
        "candidate_dir": str(candidate_dir),
        "source_dir": str(source_dir) if source_dir else None,
        "report_root": str(output_root),
        "total": total,
        "success": success,
        "failed": failed,
        "rows": rows,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    trs: List[str] = []
    for i, row in enumerate(rows, start=1):
        if row.get("status") == "ok":
            summ = row.get("summary", {})
            report_rel = row.get("report_rel", "")
            manifest_rel = row.get("manifest_rel", "")
            trs.append(
                f"""
                <tr>
                  <td>{i}</td>
                  <td>{html.escape(row.get("name", ""))}</td>
                  <td>OK</td>
                  <td>{summ.get("total_points", 0)}</td>
                  <td>{(summ.get("avg_visual") or 0) * 100:.1f}%</td>
                  <td>{(summ.get("avg_subtitle") or 0) * 100:.1f}%</td>
                  <td>{((summ.get("avg_audio") if summ.get("avg_audio") is not None else 0) * 100):.1f}%</td>
                  <td>{(summ.get("avg_overall") or 0) * 100:.1f}%</td>
                  <td>{summ.get("mismatch_points", 0)}</td>
                  <td>{html.escape(summ.get("overall_verdict", "未知"))}</td>
                  <td>{row.get("clip_elapsed_sec") if row.get("clip_elapsed_sec") is not None else "-"}</td>
                  <td>{row.get("audit_elapsed_sec") if row.get("audit_elapsed_sec") is not None else "-"}</td>
                  <td>{row.get("total_elapsed_sec") if row.get("total_elapsed_sec") is not None else "-"}</td>
                  <td><a href="{html.escape(report_rel)}">详细页面</a></td>
                  <td><a href="{html.escape(manifest_rel)}">manifest</a></td>
                </tr>
                """
            )
        else:
            trs.append(
                f"""
                <tr>
                  <td>{i}</td>
                  <td>{html.escape(row.get("name", ""))}</td>
                  <td>FAILED</td>
                  <td colspan="7">{html.escape(row.get("error", ""))}</td>
                  <td>{row.get("clip_elapsed_sec") if row.get("clip_elapsed_sec") is not None else "-"}</td>
                  <td>{row.get("audit_elapsed_sec") if row.get("audit_elapsed_sec") is not None else "-"}</td>
                  <td>{row.get("total_elapsed_sec") if row.get("total_elapsed_sec") is not None else "-"}</td>
                  <td>-</td>
                  <td>-</td>
                </tr>
                """
            )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>3秒间隔 AI 审片总报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft Yahei", sans-serif; margin: 20px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; font-size: 12px; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .meta {{ line-height: 1.8; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>3秒间隔 AI 审片总报告</h1>
  <div class="meta">
    <div>原素材目录: {html.escape(str(material_dir))}</div>
    <div>二剪目录: {html.escape(str(candidate_dir))}</div>
    <div>源剧集目录: {html.escape(str(source_dir)) if source_dir else "-"}</div>
    <div>测试报告目录: {html.escape(str(output_root))}</div>
    <div>开始时间: {html.escape(started_at)}</div>
    <div>结束时间: {html.escape(finished_at)}</div>
    <div>检查间隔: {interval} 秒</div>
    <div>总数: {total}，成功: {success}，失败: {failed}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>素材</th>
        <th>状态</th>
        <th>检查点</th>
        <th>画面均分</th>
        <th>字幕均分</th>
        <th>音频均分</th>
        <th>综合均分</th>
        <th>不一致点</th>
        <th>结论</th>
        <th>二剪耗时(s)</th>
        <th>审片耗时(s)</th>
        <th>总耗时(s)</th>
        <th>详细页</th>
        <th>清单</th>
      </tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</body>
</html>
"""
    summary_html.write_text(html_text, encoding="utf-8")
    return summary_json, summary_html


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", help="配置文件路径（JSON）")
    pre_args, _ = pre_parser.parse_known_args()

    try:
        cfg, cfg_path = load_section_config(REPO_ROOT, "batch_ai_audit_3s", explicit_path=pre_args.config)
        source_dir_default = cfg_str(cfg, "source_dir", "")
        interval_default = cfg_float(cfg, "interval", 3.0)
        clip_duration_default = cfg_float(cfg, "clip_duration", 2.0)
        max_points_default = cfg_int(cfg, "max_points", 1200)
        asr_default = cfg_str(cfg, "asr", "auto")
        asr_cmd_default = cfg_str(cfg, "asr_cmd", "")
        asr_python_default = cfg_str(cfg, "asr_python", "")
        asr_model_default = cfg_str(cfg, "asr_model", "base")
        language_default = cfg_str(cfg, "language", "zh")
        target_sub_default = cfg_str(cfg, "target_sub", "")
        candidate_sub_default = cfg_str(cfg, "candidate_sub", "")
        output_root_default = cfg_str(cfg, "output_root", "")
        cache_dir_default = cfg_str(cfg, "cache_dir", "")
        reconstruct_workers_default = cfg_int(cfg, "reconstruct_workers", 5)
        reconstruct_segment_duration_default = cfg_float(cfg, "reconstruct_segment_duration", 5.0)
        reconstruct_low_score_threshold_default = cfg_float(cfg, "reconstruct_low_score_threshold", 0.82)
        reconstruct_force_target_audio_default = cfg_bool(cfg, "reconstruct_force_target_audio", True)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 2

    parser = argparse.ArgumentParser(description="批量执行 3 秒间隔 AI 审片")
    parser.add_argument("--config", default=str(cfg_path) if cfg_path else "", help="配置文件路径（JSON）")
    parser.add_argument("--material-dir", required=True, help="原素材目录")
    parser.add_argument("--candidate-dir", required=True, help="二剪目录")
    parser.add_argument("--source-dir", default=source_dir_default, help="源剧集目录（缺失候选时用于重构）")
    parser.add_argument("--reconstruct-missing", action="store_true", help="候选缺失时自动用 v6_fast 重构")
    parser.add_argument("--interval", type=float, default=interval_default, help="抽检间隔秒数")
    parser.add_argument("--clip-duration", type=float, default=clip_duration_default, help="每个点音频切片时长")
    parser.add_argument("--max-points", type=int, default=max_points_default, help="每条视频最大检查点")
    parser.add_argument("--asr", default=asr_default, choices=["auto", "none", "faster_whisper", "whisper"])
    parser.add_argument("--asr-cmd", default=asr_cmd_default, help="whisper 命令路径")
    parser.add_argument("--asr-python", default=asr_python_default, help="whisper 所在 python 路径")
    parser.add_argument("--asr-model", default=asr_model_default, help="ASR 模型")
    parser.add_argument("--language", default=language_default, help="ASR 语种")
    parser.add_argument("--target-sub", default=target_sub_default, help="原素材字幕文件（可选）")
    parser.add_argument("--candidate-sub", default=candidate_sub_default, help="二剪字幕文件（可选）")
    parser.add_argument("--output-root", default=output_root_default, help="报告输出根目录")
    parser.add_argument("--cache-dir", default=cache_dir_default, help="v6_fast 缓存目录")
    parser.add_argument("--reconstruct-workers", type=int, default=reconstruct_workers_default, help="缺失候选时 v6_fast 并发数")
    parser.add_argument("--reconstruct-segment-duration", type=float, default=reconstruct_segment_duration_default, help="缺失候选时 v6_fast 分段时长")
    parser.add_argument("--reconstruct-low-score-threshold", type=float, default=reconstruct_low_score_threshold_default, help="缺失候选时 v6_fast 低分阈值")
    add_bool_arg(
        parser,
        "--reconstruct-force-target-audio",
        reconstruct_force_target_audio_default,
        "缺失候选时重构是否强制使用目标音轨",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    material_dir = Path(args.material_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    source_dir = Path(args.source_dir).resolve() if args.source_dir else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() if args.output_root else (repo_root / "runtime" / "temp_outputs" / "ai_video_audit_batch" / ts)
    ensure_dir(output_root)
    ensure_dir(candidate_dir)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (repo_root / "runtime" / "cache")

    materials = discover_materials(material_dir)
    if not materials:
        raise RuntimeError(f"素材目录无 mp4: {material_dir}")

    started_at = datetime.now().isoformat()
    rows: List[Dict] = []

    print("=" * 72)
    print("批量 3 秒 AI 审片")
    print("=" * 72)
    print(f"素材目录: {material_dir}")
    print(f"二剪目录: {candidate_dir}")
    print(f"素材数量: {len(materials)}")
    print(f"输出目录: {output_root}")
    if args.config:
        print(f"配置文件: {args.config}")

    for idx, material in enumerate(materials, start=1):
        print(f"\n[{idx}/{len(materials)}] {material.name}")
        item_start = time.perf_counter()
        clip_elapsed_sec: Optional[float] = None
        audit_elapsed_sec: Optional[float] = None
        candidate_mode = "existing"
        candidate = find_candidate(material, candidate_dir)
        if candidate is None and args.reconstruct_missing and source_dir:
            candidate = candidate_dir / f"{material.stem}_V3_FAST.mp4"
            print(f"  - 候选缺失，启动重构: {candidate.name}")
            clip_start = time.perf_counter()
            ok, err = reconstruct_with_v6_fast(
                repo_root=repo_root,
                target=material,
                source_dir=source_dir,
                output=candidate,
                cache_dir=cache_dir,
                workers=args.reconstruct_workers,
                segment_duration=args.reconstruct_segment_duration,
                low_score_threshold=args.reconstruct_low_score_threshold,
                force_target_audio=bool(args.reconstruct_force_target_audio),
                config_path=args.config,
            )
            clip_elapsed_sec = round(time.perf_counter() - clip_start, 3)
            candidate_mode = "reconstructed"
            if not ok:
                rows.append({
                    "name": material.name,
                    "status": "failed",
                    "error": f"reconstruct_failed: {err}",
                    "candidate_mode": candidate_mode,
                    "clip_elapsed_sec": clip_elapsed_sec,
                    "audit_elapsed_sec": audit_elapsed_sec,
                    "total_elapsed_sec": round(time.perf_counter() - item_start, 3),
                })
                continue

        if candidate is None or not candidate.exists():
            rows.append({
                "name": material.name,
                "status": "failed",
                "error": "candidate_not_found",
                "candidate_mode": candidate_mode,
                "clip_elapsed_sec": clip_elapsed_sec,
                "audit_elapsed_sec": audit_elapsed_sec,
                "total_elapsed_sec": round(time.perf_counter() - item_start, 3),
            })
            continue

        if clip_elapsed_sec is None:
            clip_elapsed_sec = 0.0

        report_dir = output_root / f"{material.stem}_3s"
        audit_start = time.perf_counter()
        ok, err = run_single_audit(
            repo_root=repo_root,
            target=material,
            candidate=candidate,
            output_dir=report_dir,
            interval=args.interval,
            clip_duration=args.clip_duration,
            max_points=args.max_points,
            asr_mode=args.asr,
            asr_cmd=args.asr_cmd,
            asr_python=args.asr_python,
            asr_model=args.asr_model,
            language=args.language,
            target_sub=args.target_sub,
            candidate_sub=args.candidate_sub,
            clip_elapsed_sec=clip_elapsed_sec,
            candidate_mode=candidate_mode,
            config_path=args.config,
        )
        audit_elapsed_sec = round(time.perf_counter() - audit_start, 3)
        total_elapsed_sec = round(time.perf_counter() - item_start, 3)
        if not ok:
            rows.append({
                "name": material.name,
                "status": "failed",
                "error": f"audit_failed: {err}",
                "candidate": str(candidate),
                "candidate_mode": candidate_mode,
                "clip_elapsed_sec": clip_elapsed_sec,
                "audit_elapsed_sec": audit_elapsed_sec,
                "total_elapsed_sec": total_elapsed_sec,
            })
            continue

        manifest_path = report_dir / "audit_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = manifest.get("summary", {})
        report_html = report_dir / "comparison_report.html"
        rows.append({
            "name": material.name,
            "status": "ok",
            "material": str(material),
            "candidate": str(candidate),
            "candidate_mode": candidate_mode,
            "manifest": str(manifest_path),
            "report_html": str(report_html),
            "manifest_rel": str(manifest_path.relative_to(output_root)),
            "report_rel": str(report_html.relative_to(output_root)),
            "clip_elapsed_sec": clip_elapsed_sec,
            "audit_elapsed_sec": audit_elapsed_sec,
            "total_elapsed_sec": total_elapsed_sec,
            "summary": summary,
        })
        print(f"  - 完成: {report_html}")

    finished_at = datetime.now().isoformat()
    summary_json, summary_html = build_total_report(
        output_root=output_root,
        rows=rows,
        started_at=started_at,
        finished_at=finished_at,
        interval=args.interval,
        material_dir=material_dir,
        candidate_dir=candidate_dir,
        source_dir=source_dir,
    )

    print("\n" + "=" * 72)
    print("批量审片完成")
    print(f"汇总 JSON: {summary_json}")
    print(f"汇总 HTML: {summary_html}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
