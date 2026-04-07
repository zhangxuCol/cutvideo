#!/usr/bin/env python3
"""
批量执行 3 秒间隔 AI 审片，并生成总汇总报告页面。
增强点：
1) 支持并发裁剪与并发审片
2) 默认先裁剪后审片（不影响单视频出片速度）
3) 不一致点可触发“速度守护”自动优化重跑
"""

import argparse
import html
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


def run_cmd(cmd: List[str], timeout_sec: Optional[float] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timeout_exceeded:{timeout_sec}")


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


def read_clip_elapsed_from_quality_report(candidate: Optional[Path]) -> Optional[float]:
    if candidate is None:
        return None
    report = candidate.with_suffix(".quality_report.json")
    if not report.exists():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return None
    timing = data.get("timing", {}) or {}
    val = timing.get("total_elapsed_sec")
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    return None


def reconstruct_with_engine(
    repo_root: Path,
    script_name: str,
    target: Path,
    source_dir: Path,
    output: Path,
    cache_dir: Path,
    workers: int,
    render_workers: int,
    segment_duration: float,
    low_score_threshold: float,
    force_target_audio: bool,
    strict_visual_verify: bool,
    boundary_glitch_fix: bool,
    adjacent_overlap_trigger: Optional[float],
    adjacent_lag_trigger: Optional[float],
    isolated_drift_trigger: Optional[float],
    cross_source_mapping_jump_trigger: Optional[float],
    timeout_sec: Optional[float] = None,
    config_path: str = "",
) -> Tuple[bool, str]:
    ensure_dir(output.parent)
    ensure_dir(cache_dir)

    script_path = repo_root / script_name
    if not script_path.exists():
        return False, f"reconstruct_script_not_found:{script_path}"

    cmd = [
        "python", str(script_path),
        "--target", str(target),
        "--source-dir", str(source_dir),
        "--output", str(output),
        "--cache", str(cache_dir),
        "--workers", str(max(0, int(workers))),
        "--render-workers", str(max(0, int(render_workers))),
        "--segment-duration", str(float(segment_duration)),
        "--low-score-threshold", str(float(low_score_threshold)),
        "--no-run-evidence-validation",
        "--no-run-ai-verify-snapshots",
    ]

    cmd.append("--strict-visual-verify" if strict_visual_verify else "--no-strict-visual-verify")
    cmd.append("--boundary-glitch-fix" if boundary_glitch_fix else "--no-boundary-glitch-fix")
    cmd.append("--force-target-audio" if force_target_audio else "--no-force-target-audio")

    if adjacent_overlap_trigger is not None:
        cmd.extend(["--adjacent-overlap-trigger", str(float(adjacent_overlap_trigger))])
    if adjacent_lag_trigger is not None:
        cmd.extend(["--adjacent-lag-trigger", str(float(adjacent_lag_trigger))])
    if isolated_drift_trigger is not None:
        cmd.extend(["--isolated-drift-trigger", str(float(isolated_drift_trigger))])
    if cross_source_mapping_jump_trigger is not None:
        cmd.extend(["--cross-source-mapping-jump-trigger", str(float(cross_source_mapping_jump_trigger))])

    if config_path:
        cmd.extend(["--config", str(config_path)])

    result = run_cmd(cmd, timeout_sec=timeout_sec)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "reconstruct_failed").strip()
        return False, msg
    if not output.exists():
        return False, "reconstruct_no_output"
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


def load_manifest(manifest_path: Optional[Path]) -> Dict:
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def audio_only_mismatch_stats(manifest_path: Optional[Path], min_visual: float = 0.98) -> Tuple[int, int, bool]:
    """
    返回:
    - mismatch_total: 明显不一致点数量
    - audio_only_count: 同时满足「仅 audio 硬不一致 + 视觉高一致」的点数量
    - eligible: 所有 mismatch 都满足 audio-only 条件（可优先音轨快修）
    """
    manifest = load_manifest(manifest_path)
    checkpoints = manifest.get("checkpoints", []) or []
    mismatch_total = 0
    audio_only_count = 0

    for p in checkpoints:
        if p.get("verdict") != "明显不一致":
            continue
        mismatch_total += 1
        reasons = p.get("hard_mismatch_reasons", []) or []
        visual = p.get("visual_similarity")
        visual_ok = isinstance(visual, (int, float)) and float(visual) >= float(min_visual)
        if reasons == ["audio"] and visual_ok:
            audio_only_count += 1

    eligible = mismatch_total > 0 and audio_only_count == mismatch_total
    return mismatch_total, audio_only_count, eligible


def visual_mismatch_overlay_plan(
    manifest_path: Optional[Path],
    window_sec: float = 1.5,
    merge_gap_sec: float = 0.5,
    max_points: int = 8,
    max_total_window_sec: float = 30.0,
) -> Tuple[int, int, List[Tuple[float, float]], bool, float]:
    """
    从 manifest 提取“视觉硬不一致”点，生成覆盖修复窗口。
    返回:
    - mismatch_total: 全部硬不一致点数量
    - visual_mismatch_count: 视觉硬不一致点数量
    - windows: 覆盖窗口 [(start, end), ...]
    - eligible: 是否满足快修条件
    - total_window_sec: 覆盖总时长
    """
    manifest = load_manifest(manifest_path)
    checkpoints = manifest.get("checkpoints", []) or []

    mismatch_total = 0
    visual_times: List[float] = []
    half = max(0.1, float(window_sec))
    merge_gap = max(0.0, float(merge_gap_sec))
    max_pts = max(1, int(max_points))

    for p in checkpoints:
        if p.get("verdict") != "明显不一致":
            continue
        mismatch_total += 1
        reasons = p.get("hard_mismatch_reasons", []) or []
        if "visual" not in reasons:
            continue
        t = p.get("sample_time_sec")
        if not isinstance(t, (int, float)):
            t = p.get("time_sec")
        if isinstance(t, (int, float)):
            visual_times.append(float(t))

    visual_times = sorted(set(visual_times))
    visual_count = len(visual_times)
    if visual_count <= 0:
        return mismatch_total, visual_count, [], False, 0.0
    if visual_count > max_pts:
        return mismatch_total, visual_count, [], False, 0.0

    windows: List[Tuple[float, float]] = []
    for t in visual_times:
        start = max(0.0, t - half)
        end = max(start, t + half)
        if not windows:
            windows.append((start, end))
            continue
        last_s, last_e = windows[-1]
        if start <= (last_e + merge_gap):
            windows[-1] = (last_s, max(last_e, end))
        else:
            windows.append((start, end))

    total_window_sec = sum(max(0.0, e - s) for s, e in windows)
    eligible = bool(windows)
    if max_total_window_sec > 0:
        eligible = eligible and (total_window_sec <= float(max_total_window_sec))
    return mismatch_total, visual_count, windows, eligible, total_window_sec


def probe_video_size(video: Path) -> Optional[Tuple[int, int]]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(video),
    ]
    ret = run_cmd(cmd)
    if ret.returncode != 0:
        return None
    txt = (ret.stdout or "").strip()
    if "x" not in txt:
        return None
    try:
        w_str, h_str = txt.split("x", 1)
        w = int(w_str.strip())
        h = int(h_str.strip())
        if w > 0 and h > 0:
            return w, h
    except Exception:
        return None
    return None


def overlay_target_visual(
    candidate: Path,
    target: Path,
    output: Path,
    windows: List[Tuple[float, float]],
    preset: str = "veryfast",
    crf: int = 22,
) -> Tuple[bool, str]:
    ensure_dir(output.parent)
    if output.exists():
        output.unlink()
    if not windows:
        return False, "visual_fix_no_windows"

    size = probe_video_size(candidate)
    if not size:
        return False, "visual_fix_probe_size_failed"
    width, height = size

    enable_expr = "+".join([f"between(t,{s:.3f},{e:.3f})" for s, e in windows])
    filter_complex = (
        f"[1:v]scale={width}:{height},setsar=1[src];"
        f"[0:v][src]overlay=enable='{enable_expr}'[vout]"
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(candidate),
        "-i", str(target),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", str(preset or "veryfast"),
        "-crf", str(int(crf)),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output),
    ]
    ret = run_cmd(cmd)
    if ret.returncode == 0 and output.exists():
        return True, f"windows={len(windows)}"

    msg = (ret.stderr or ret.stdout or "visual_fix_failed").strip()
    return False, msg


def remux_target_audio(candidate: Path, target: Path, output: Path) -> Tuple[bool, str]:
    ensure_dir(output.parent)
    if output.exists():
        output.unlink()

    copy_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(candidate),
        "-i", str(target),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        str(output),
    ]
    copy_ret = run_cmd(copy_cmd)
    if copy_ret.returncode == 0 and output.exists():
        return True, "copy"

    reenc_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(candidate),
        "-i", str(target),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output),
    ]
    reenc_ret = run_cmd(reenc_cmd)
    if reenc_ret.returncode == 0 and output.exists():
        return True, "aac_reencode"

    msg = (reenc_ret.stderr or reenc_ret.stdout or copy_ret.stderr or copy_ret.stdout or "audio_remux_failed").strip()
    return False, msg


def build_total_report(
    output_root: Path,
    rows: List[Dict],
    started_at: str,
    finished_at: str,
    clip_phase_elapsed_sec: Optional[float],
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
    clip_values = [float(r["clip_elapsed_sec"]) for r in rows if isinstance(r.get("clip_elapsed_sec"), (int, float))]
    clip_elapsed_sum_sec = round(sum(clip_values), 3) if clip_values else None
    clip_elapsed_max_sec = round(max(clip_values), 3) if clip_values else None

    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "clip_phase_elapsed_sec": clip_phase_elapsed_sec,
        "clip_elapsed_sum_sec": clip_elapsed_sum_sec,
        "clip_elapsed_max_sec": clip_elapsed_max_sec,
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
                  <td>{row.get("clip_budget_sec") if row.get("clip_budget_sec") is not None else "-"}</td>
                  <td>{row.get("clip_elapsed_sec") if row.get("clip_elapsed_sec") is not None else "-"}</td>
                  <td>{row.get("audit_elapsed_sec") if row.get("audit_elapsed_sec") is not None else "-"}</td>
                  <td>{row.get("total_elapsed_sec") if row.get("total_elapsed_sec") is not None else "-"}</td>
                  <td>{"是" if row.get("optimization_applied") else "否"}</td>
                  <td>{html.escape(str(row.get("optimization_note", "")))}</td>
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
                  <td>-</td>
                  <td>{row.get("clip_budget_sec") if row.get("clip_budget_sec") is not None else "-"}</td>
                  <td>{row.get("clip_elapsed_sec") if row.get("clip_elapsed_sec") is not None else "-"}</td>
                  <td>{row.get("audit_elapsed_sec") if row.get("audit_elapsed_sec") is not None else "-"}</td>
                  <td>{row.get("total_elapsed_sec") if row.get("total_elapsed_sec") is not None else "-"}</td>
                  <td>{"是" if row.get("optimization_applied") else "否"}</td>
                  <td>{html.escape(str(row.get("optimization_note", "")))}</td>
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
    <div>裁剪阶段总耗时(墙钟): {clip_phase_elapsed_sec if clip_phase_elapsed_sec is not None else "-"} 秒</div>
    <div>裁剪耗时汇总(按素材累加): {clip_elapsed_sum_sec if clip_elapsed_sum_sec is not None else "-"} 秒</div>
    <div>单素材裁剪最大耗时: {clip_elapsed_max_sec if clip_elapsed_max_sec is not None else "-"} 秒</div>
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
        <th>裁剪基准(s)</th>
        <th>二剪耗时(s)</th>
        <th>审片耗时(s)</th>
        <th>总耗时(s)</th>
        <th>优化已应用</th>
        <th>优化备注</th>
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


@dataclass
class ItemState:
    material: Path
    candidate: Optional[Path] = None
    candidate_mode: str = "existing"
    status: str = "pending"
    error: str = ""
    clip_elapsed_sec: Optional[float] = None
    clip_budget_sec: Optional[float] = None
    audit_elapsed_sec: Optional[float] = None
    total_elapsed_sec: Optional[float] = None
    summary: Optional[Dict] = None
    manifest_path: Optional[Path] = None
    report_html: Optional[Path] = None
    optimization_applied: bool = False
    optimization_attempted: bool = False
    optimization_note: str = ""


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

        jobs_default = max(1, cfg_int(cfg, "jobs", 2))
        clip_jobs_default = max(1, cfg_int(cfg, "clip_jobs", jobs_default))
        audit_jobs_default = max(1, cfg_int(cfg, "audit_jobs", jobs_default))

        reconstruct_script_default = cfg_str(cfg, "reconstruct_script", "fast_v7.py")
        reconstruct_workers_default = cfg_int(cfg, "reconstruct_workers", 8)
        reconstruct_render_workers_default = cfg_int(cfg, "reconstruct_render_workers", 4)
        reconstruct_segment_duration_default = cfg_float(cfg, "reconstruct_segment_duration", 5.0)
        reconstruct_low_score_threshold_default = cfg_float(cfg, "reconstruct_low_score_threshold", 0.82)
        reconstruct_force_target_audio_default = cfg_bool(cfg, "reconstruct_force_target_audio", True)
        reconstruct_strict_visual_verify_default = cfg_bool(cfg, "reconstruct_strict_visual_verify", True)
        reconstruct_boundary_glitch_fix_default = cfg_bool(cfg, "reconstruct_boundary_glitch_fix", True)
        reconstruct_timeout_sec_default = max(0.0, cfg_float(cfg, "reconstruct_timeout_sec", 0.0))

        optimize_on_mismatch_default = cfg_bool(cfg, "optimize_on_mismatch", True)
        optimize_max_retries_default = max(0, cfg_int(cfg, "optimize_max_retries", 1))
        optimize_max_clip_increase_ratio_default = max(0.0, cfg_float(cfg, "optimize_max_clip_increase_ratio", 0.0))
        optimize_max_clip_seconds_cap_default = max(0.0, cfg_float(cfg, "optimize_max_clip_seconds_cap", 180.0))
        optimize_audio_remux_on_audio_mismatch_default = cfg_bool(cfg, "optimize_audio_remux_on_audio_mismatch", True)
        optimize_audio_remux_min_visual_default = cfg_float(cfg, "optimize_audio_remux_min_visual", 0.98)
        optimize_visual_overlay_on_visual_mismatch_default = cfg_bool(cfg, "optimize_visual_overlay_on_visual_mismatch", True)
        optimize_visual_overlay_window_sec_default = max(0.1, cfg_float(cfg, "optimize_visual_overlay_window_sec", 1.5))
        optimize_visual_overlay_merge_gap_sec_default = max(0.0, cfg_float(cfg, "optimize_visual_overlay_merge_gap_sec", 0.5))
        optimize_visual_overlay_max_points_default = max(1, cfg_int(cfg, "optimize_visual_overlay_max_points", 8))
        optimize_visual_overlay_max_total_window_sec_default = max(0.0, cfg_float(cfg, "optimize_visual_overlay_max_total_window_sec", 30.0))
        optimize_visual_overlay_crf_default = cfg_int(cfg, "optimize_visual_overlay_crf", 22)
        optimize_visual_overlay_preset_default = cfg_str(cfg, "optimize_visual_overlay_preset", "veryfast")
        optimize_adjacent_overlap_trigger_default = cfg_float(cfg, "optimize_adjacent_overlap_trigger", 0.45)
        optimize_adjacent_lag_trigger_default = cfg_float(cfg, "optimize_adjacent_lag_trigger", 0.55)
        optimize_isolated_drift_trigger_default = cfg_float(cfg, "optimize_isolated_drift_trigger", 0.55)
        optimize_cross_source_mapping_jump_trigger_default = cfg_float(cfg, "optimize_cross_source_mapping_jump_trigger", 0.50)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 2

    parser = argparse.ArgumentParser(description="批量执行 3 秒间隔 AI 审片（并发+优化闭环）")
    parser.add_argument("--config", default=str(cfg_path) if cfg_path else "", help="配置文件路径（JSON）")
    parser.add_argument("--material-dir", required=True, help="原素材目录")
    parser.add_argument("--candidate-dir", required=True, help="二剪目录")
    parser.add_argument("--source-dir", default=source_dir_default, help="源剧集目录（重构时必需）")

    parser.add_argument("--reconstruct-missing", action="store_true", help="候选缺失时自动重构")
    parser.add_argument("--reconstruct-all", action="store_true", help="忽略现有候选，全部重构后再审片")

    parser.add_argument("--jobs", type=int, default=jobs_default, help="并发总开关（未显式指定 clip/audit 时作为默认）")
    parser.add_argument("--clip-jobs", type=int, default=clip_jobs_default, help="并发裁剪任务数")
    parser.add_argument("--audit-jobs", type=int, default=audit_jobs_default, help="并发审片任务数")

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

    parser.add_argument("--cache-dir", default=cache_dir_default, help="重构缓存目录")
    parser.add_argument("--reconstruct-script", default=reconstruct_script_default, help="重构脚本（默认 fast_v7.py）")
    parser.add_argument("--reconstruct-workers", type=int, default=reconstruct_workers_default, help="重构匹配并发数")
    parser.add_argument("--reconstruct-render-workers", type=int, default=reconstruct_render_workers_default, help="重构渲染并发数")
    parser.add_argument("--reconstruct-segment-duration", type=float, default=reconstruct_segment_duration_default, help="重构分段时长")
    parser.add_argument("--reconstruct-low-score-threshold", type=float, default=reconstruct_low_score_threshold_default, help="重构低分阈值")
    parser.add_argument("--reconstruct-timeout-sec", type=float, default=reconstruct_timeout_sec_default, help="首轮重构单条超时上限（秒，<=0 表示不限制）")
    add_bool_arg(parser, "--reconstruct-force-target-audio", reconstruct_force_target_audio_default, "重构是否强制目标音轨")
    add_bool_arg(parser, "--reconstruct-strict-visual-verify", reconstruct_strict_visual_verify_default, "重构是否启用严格画面核验")
    add_bool_arg(parser, "--reconstruct-boundary-glitch-fix", reconstruct_boundary_glitch_fix_default, "重构是否启用边界单帧修复")

    add_bool_arg(parser, "--optimize-on-mismatch", optimize_on_mismatch_default, "不一致时自动优化重构并复审")
    parser.add_argument("--optimize-max-retries", type=int, default=optimize_max_retries_default, help="自动优化最大重试次数")
    parser.add_argument("--optimize-max-clip-increase-ratio", type=float, default=optimize_max_clip_increase_ratio_default, help="自动优化允许的裁剪耗时增幅比例（0 表示不允许变慢）")
    parser.add_argument("--optimize-max-clip-seconds-cap", type=float, default=optimize_max_clip_seconds_cap_default, help="自动优化单次重构绝对耗时上限（秒，<=0 表示不限制）")
    add_bool_arg(parser, "--optimize-audio-remux-on-audio-mismatch", optimize_audio_remux_on_audio_mismatch_default, "仅音频硬不一致时优先做目标音轨快修")
    parser.add_argument("--optimize-audio-remux-min-visual", type=float, default=optimize_audio_remux_min_visual_default, help="音轨快修触发的最低视觉一致度")
    add_bool_arg(parser, "--optimize-visual-overlay-on-visual-mismatch", optimize_visual_overlay_on_visual_mismatch_default, "视觉硬不一致时优先做局部画面覆盖快修")
    parser.add_argument("--optimize-visual-overlay-window-sec", type=float, default=optimize_visual_overlay_window_sec_default, help="视觉快修每个异常点前后覆盖窗口（秒）")
    parser.add_argument("--optimize-visual-overlay-merge-gap-sec", type=float, default=optimize_visual_overlay_merge_gap_sec_default, help="视觉快修相邻窗口合并间隔（秒）")
    parser.add_argument("--optimize-visual-overlay-max-points", type=int, default=optimize_visual_overlay_max_points_default, help="视觉快修可处理的最大视觉硬不一致点数")
    parser.add_argument("--optimize-visual-overlay-max-total-window-sec", type=float, default=optimize_visual_overlay_max_total_window_sec_default, help="视觉快修允许的窗口总时长上限（秒）")
    parser.add_argument("--optimize-visual-overlay-crf", type=int, default=optimize_visual_overlay_crf_default, help="视觉快修重编码 CRF（越大体积越小）")
    parser.add_argument("--optimize-visual-overlay-preset", default=optimize_visual_overlay_preset_default, help="视觉快修重编码 preset（ultrafast/veryfast/fast/...）")
    parser.add_argument("--optimize-adjacent-overlap-trigger", type=float, default=optimize_adjacent_overlap_trigger_default)
    parser.add_argument("--optimize-adjacent-lag-trigger", type=float, default=optimize_adjacent_lag_trigger_default)
    parser.add_argument("--optimize-isolated-drift-trigger", type=float, default=optimize_isolated_drift_trigger_default)
    parser.add_argument("--optimize-cross-source-mapping-jump-trigger", type=float, default=optimize_cross_source_mapping_jump_trigger_default)

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
    jobs = max(1, int(args.jobs))
    clip_jobs = max(1, int(args.clip_jobs or jobs))
    audit_jobs = max(1, int(args.audit_jobs or jobs))

    materials = discover_materials(material_dir)
    if not materials:
        raise RuntimeError(f"素材目录无 mp4: {material_dir}")

    started_at = datetime.now().isoformat()
    states: Dict[str, ItemState] = {m.name: ItemState(material=m) for m in materials}

    print("=" * 72)
    print("批量 3 秒 AI 审片（并发 + 自动优化）")
    print("=" * 72)
    print(f"素材目录: {material_dir}")
    print(f"二剪目录: {candidate_dir}")
    print(f"素材数量: {len(materials)}")
    print(f"输出目录: {output_root}")
    print(f"并发配置: clip_jobs={clip_jobs}, audit_jobs={audit_jobs}")
    if args.config:
        print(f"配置文件: {args.config}")

    # 阶段1：重构（若需要）
    clip_phase_start_perf = time.perf_counter()
    reconstruct_timeout_sec = max(0.0, float(args.reconstruct_timeout_sec))
    reconstruct_targets: List[ItemState] = []
    for st in states.values():
        found = find_candidate(st.material, candidate_dir)
        if bool(args.reconstruct_all):
            st.candidate = candidate_dir / f"{st.material.stem}_V3_FAST.mp4"
            st.candidate_mode = "reconstructed"
            reconstruct_targets.append(st)
        elif found is None and bool(args.reconstruct_missing):
            st.candidate = candidate_dir / f"{st.material.stem}_V3_FAST.mp4"
            st.candidate_mode = "reconstructed"
            reconstruct_targets.append(st)
        else:
            st.candidate = found
            st.candidate_mode = "existing"
            st.clip_budget_sec = read_clip_elapsed_from_quality_report(found)

    if reconstruct_targets:
        if source_dir is None:
            for st in reconstruct_targets:
                st.status = "failed"
                st.error = "source_dir_required_for_reconstruct"
        else:
            print(f"\n[阶段1] 并发重构 {len(reconstruct_targets)} 条")

            def _run_reconstruct(st: ItemState) -> Tuple[str, bool, str, float]:
                cstart = time.perf_counter()
                per_cache = cache_dir / st.material.stem
                ok, err = reconstruct_with_engine(
                    repo_root=repo_root,
                    script_name=args.reconstruct_script,
                    target=st.material,
                    source_dir=source_dir,
                    output=st.candidate,
                    cache_dir=per_cache,
                    workers=args.reconstruct_workers,
                    render_workers=args.reconstruct_render_workers,
                    segment_duration=args.reconstruct_segment_duration,
                    low_score_threshold=args.reconstruct_low_score_threshold,
                    force_target_audio=bool(args.reconstruct_force_target_audio),
                    strict_visual_verify=bool(args.reconstruct_strict_visual_verify),
                    boundary_glitch_fix=bool(args.reconstruct_boundary_glitch_fix),
                    adjacent_overlap_trigger=None,
                    adjacent_lag_trigger=None,
                    isolated_drift_trigger=None,
                    cross_source_mapping_jump_trigger=None,
                    timeout_sec=(reconstruct_timeout_sec if reconstruct_timeout_sec > 0 else None),
                    config_path=args.config,
                )
                return st.material.name, ok, err, round(time.perf_counter() - cstart, 3)

            with ThreadPoolExecutor(max_workers=clip_jobs) as pool:
                futs = [pool.submit(_run_reconstruct, st) for st in reconstruct_targets]
                for fut in as_completed(futs):
                    name, ok, err, elapsed = fut.result()
                    if ok and reconstruct_timeout_sec > 0 and elapsed > (reconstruct_timeout_sec + 0.5):
                        ok = False
                        err = f"reconstruct_elapsed_exceeds_cap:{elapsed:.3f}s>{reconstruct_timeout_sec:.3f}s"
                    st = states[name]
                    st.clip_elapsed_sec = elapsed
                    st.clip_budget_sec = elapsed
                    if ok and st.candidate and st.candidate.exists():
                        st.status = "clip_ok"
                        print(f"  - 出片完成: {name} ({elapsed:.3f}s)")
                    else:
                        st.status = "failed"
                        st.error = f"reconstruct_failed:{err}"
                        print(f"  - 出片失败: {name} ({elapsed:.3f}s) -> {err[:160]}")

    # 给未重构项补状态
    for st in states.values():
        if st.status == "failed":
            continue
        if st.candidate is None or not st.candidate.exists():
            st.status = "failed"
            st.error = "candidate_not_found"
            continue
        if st.status == "pending":
            st.status = "clip_ok"
    clip_phase_elapsed_sec = round(time.perf_counter() - clip_phase_start_perf, 3)

    # 阶段2：审片
    audit_targets = [st for st in states.values() if st.status == "clip_ok"]
    print(f"\n[阶段2] 并发审片 {len(audit_targets)} 条")

    def _run_audit(st: ItemState, retry_tag: str = "") -> Tuple[bool, str, Optional[Dict], Optional[Path], Optional[Path], float]:
        suffix = f"_3s{retry_tag}" if retry_tag else "_3s"
        report_dir = output_root / f"{st.material.stem}{suffix}"
        astart = time.perf_counter()
        ok, err = run_single_audit(
            repo_root=repo_root,
            target=st.material,
            candidate=st.candidate,
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
            clip_elapsed_sec=float(st.clip_elapsed_sec or 0.0),
            candidate_mode=st.candidate_mode,
            config_path=args.config,
        )
        elapsed = round(time.perf_counter() - astart, 3)
        if not ok:
            return False, err, None, None, None, elapsed

        manifest_path = report_dir / "audit_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"manifest_parse_failed:{exc}", None, manifest_path, report_dir / "comparison_report.html", elapsed
        return True, "", manifest.get("summary", {}), manifest_path, report_dir / "comparison_report.html", elapsed

    with ThreadPoolExecutor(max_workers=audit_jobs) as pool:
        futs = {pool.submit(_run_audit, st): st.material.name for st in audit_targets}
        for fut in as_completed(futs):
            name = futs[fut]
            st = states[name]
            ok, err, summary, manifest_path, report_html, audit_elapsed = fut.result()
            st.audit_elapsed_sec = audit_elapsed
            st.total_elapsed_sec = round(float(st.clip_elapsed_sec or 0.0) + float(st.audit_elapsed_sec or 0.0), 3)
            if not ok:
                st.status = "failed"
                st.error = f"audit_failed:{err}"
                print(f"  - 审片失败: {name} ({audit_elapsed:.3f}s) -> {err[:160]}")
            else:
                st.status = "ok"
                st.summary = summary or {}
                st.manifest_path = manifest_path
                st.report_html = report_html
                print(f"  - 审片完成: {name} ({audit_elapsed:.3f}s), mismatch={st.summary.get('mismatch_points', 0)}")

    # 阶段3：自动优化（不一致）
    if bool(args.optimize_on_mismatch) and source_dir and int(args.optimize_max_retries) > 0:
        print("\n[阶段3] 不一致自动优化（速度守护）")
        for st in states.values():
            if st.status != "ok":
                continue
            mismatch_points = int((st.summary or {}).get("mismatch_points", 0) or 0)
            if mismatch_points <= 0:
                continue

            st.optimization_attempted = True
            baseline_mismatch = mismatch_points
            baseline_budget = st.clip_budget_sec
            if baseline_budget is None:
                baseline_budget = read_clip_elapsed_from_quality_report(st.candidate)
            if baseline_budget is None:
                st.optimization_note = "opt_clip_budget_missing_skip"
                print(f"  - 跳过优化: {st.material.name} (缺少基准耗时)")
                continue

            accepted = False
            for retry in range(1, int(args.optimize_max_retries) + 1):
                allowed = float(baseline_budget) * (1.0 + max(0.0, float(args.optimize_max_clip_increase_ratio)))
                absolute_cap = max(0.0, float(args.optimize_max_clip_seconds_cap))
                if absolute_cap > 0.0:
                    allowed = min(allowed, absolute_cap)

                # 快修优先：仅音频硬不一致且画面高一致时，优先替换目标音轨（极快且不影响裁剪主链路）。
                if bool(args.optimize_audio_remux_on_audio_mismatch):
                    mismatch_total, audio_only_count, audio_only_eligible = audio_only_mismatch_stats(
                        st.manifest_path,
                        min_visual=float(args.optimize_audio_remux_min_visual),
                    )
                    if audio_only_eligible:
                        audiofix_output = candidate_dir / f"{st.material.stem}_V3_FAST_audiofix{retry}.mp4"
                        af_start = time.perf_counter()
                        ok_af, af_msg = remux_target_audio(st.candidate, st.material, audiofix_output)
                        af_elapsed = round(time.perf_counter() - af_start, 3)
                        if not ok_af or not audiofix_output.exists():
                            st.optimization_note = f"audio_fix_failed:{af_msg}"
                        elif af_elapsed > allowed:
                            st.optimization_note = f"audio_fix_slower_rejected:{af_elapsed:.3f}s>{allowed:.3f}s"
                        else:
                            old_candidate = st.candidate
                            st.candidate = audiofix_output
                            ok2, err2, sum2, manifest2, report2, audit2 = _run_audit(st, retry_tag=f"_audiofix{retry}")
                            st.candidate = old_candidate
                            if not ok2:
                                st.optimization_note = f"audio_fix_audit_failed:{err2}"
                            else:
                                new_mismatch = int((sum2 or {}).get("mismatch_points", 0) or 0)
                                if new_mismatch < baseline_mismatch:
                                    try:
                                        if old_candidate and old_candidate.exists():
                                            bak = old_candidate.with_suffix(".pre_opt.bak.mp4")
                                            if bak.exists():
                                                bak.unlink()
                                            old_candidate.rename(bak)
                                            audiofix_output.rename(old_candidate)
                                            bak.unlink(missing_ok=True)
                                            st.candidate = old_candidate
                                        else:
                                            st.candidate = audiofix_output
                                    except Exception:
                                        st.optimization_note = "audio_fix_apply_failed_keep_audiofix_file"
                                        continue

                                    st.optimization_applied = True
                                    st.clip_elapsed_sec = float(af_elapsed)
                                    st.summary = sum2 or {}
                                    st.manifest_path = manifest2
                                    st.report_html = report2
                                    st.audit_elapsed_sec = float(audit2)
                                    st.total_elapsed_sec = round(float(st.clip_elapsed_sec or 0.0) + float(st.audit_elapsed_sec or 0.0), 3)
                                    st.optimization_note = (
                                        f"audio_fix mismatch:{baseline_mismatch}->{new_mismatch}, "
                                        f"clip={af_elapsed:.3f}s<=baseline:{baseline_budget:.3f}s "
                                        f"(audio_only={audio_only_count}/{mismatch_total}, mode={af_msg})"
                                    )
                                    accepted = True
                                    break
                                else:
                                    st.optimization_note = (
                                        f"audio_fix_no_improve:{baseline_mismatch}->{new_mismatch} "
                                        f"(audio_only={audio_only_count}/{mismatch_total})"
                                    )

                # 快修补充：少量视觉硬不一致时，局部覆盖目标画面（同时使用目标音轨）再复审。
                if (not accepted) and bool(args.optimize_visual_overlay_on_visual_mismatch):
                    mismatch_total_v, visual_count, windows, visual_eligible, total_window_sec = visual_mismatch_overlay_plan(
                        st.manifest_path,
                        window_sec=float(args.optimize_visual_overlay_window_sec),
                        merge_gap_sec=float(args.optimize_visual_overlay_merge_gap_sec),
                        max_points=int(args.optimize_visual_overlay_max_points),
                        max_total_window_sec=float(args.optimize_visual_overlay_max_total_window_sec),
                    )
                    if visual_eligible:
                        vis_output = candidate_dir / f"{st.material.stem}_V3_FAST_visfix{retry}.mp4"
                        vis_start = time.perf_counter()
                        ok_vf, vf_msg = overlay_target_visual(
                            st.candidate,
                            st.material,
                            vis_output,
                            windows,
                            preset=str(args.optimize_visual_overlay_preset),
                            crf=int(args.optimize_visual_overlay_crf),
                        )
                        vis_elapsed = round(time.perf_counter() - vis_start, 3)
                        if not ok_vf or not vis_output.exists():
                            st.optimization_note = f"visual_fix_failed:{vf_msg}"
                        elif vis_elapsed > allowed:
                            st.optimization_note = f"visual_fix_slower_rejected:{vis_elapsed:.3f}s>{allowed:.3f}s"
                        else:
                            old_candidate = st.candidate
                            st.candidate = vis_output
                            ok2, err2, sum2, manifest2, report2, audit2 = _run_audit(st, retry_tag=f"_visfix{retry}")
                            st.candidate = old_candidate
                            if not ok2:
                                st.optimization_note = f"visual_fix_audit_failed:{err2}"
                            else:
                                new_mismatch = int((sum2 or {}).get("mismatch_points", 0) or 0)
                                if new_mismatch < baseline_mismatch:
                                    try:
                                        if old_candidate and old_candidate.exists():
                                            bak = old_candidate.with_suffix(".pre_opt.bak.mp4")
                                            if bak.exists():
                                                bak.unlink()
                                            old_candidate.rename(bak)
                                            vis_output.rename(old_candidate)
                                            bak.unlink(missing_ok=True)
                                            st.candidate = old_candidate
                                        else:
                                            st.candidate = vis_output
                                    except Exception:
                                        st.optimization_note = "visual_fix_apply_failed_keep_visfix_file"
                                        continue

                                    st.optimization_applied = True
                                    st.clip_elapsed_sec = float(vis_elapsed)
                                    st.summary = sum2 or {}
                                    st.manifest_path = manifest2
                                    st.report_html = report2
                                    st.audit_elapsed_sec = float(audit2)
                                    st.total_elapsed_sec = round(float(st.clip_elapsed_sec or 0.0) + float(st.audit_elapsed_sec or 0.0), 3)
                                    st.optimization_note = (
                                        f"visual_fix mismatch:{baseline_mismatch}->{new_mismatch}, "
                                        f"clip={vis_elapsed:.3f}s<=baseline:{baseline_budget:.3f}s "
                                        f"(visual={visual_count}/{mismatch_total_v}, windows={len(windows)}, span={total_window_sec:.3f}s)"
                                    )
                                    accepted = True
                                    break
                                else:
                                    st.optimization_note = (
                                        f"visual_fix_no_improve:{baseline_mismatch}->{new_mismatch} "
                                        f"(visual={visual_count}/{mismatch_total_v}, windows={len(windows)})"
                                    )
                    elif visual_count > 0:
                        st.optimization_note = (
                            f"visual_fix_not_eligible:visual={visual_count}/{mismatch_total_v}, "
                            f"max_points={int(args.optimize_visual_overlay_max_points)}, "
                            f"max_span={float(args.optimize_visual_overlay_max_total_window_sec):.3f}s"
                        )

                if accepted:
                    break

                opt_output = candidate_dir / f"{st.material.stem}_V3_FAST_opt{retry}.mp4"
                cstart = time.perf_counter()
                ok, err = reconstruct_with_engine(
                    repo_root=repo_root,
                    script_name=args.reconstruct_script,
                    target=st.material,
                    source_dir=source_dir,
                    output=opt_output,
                    cache_dir=cache_dir / f"{st.material.stem}_opt{retry}",
                    workers=args.reconstruct_workers,
                    render_workers=args.reconstruct_render_workers,
                    segment_duration=args.reconstruct_segment_duration,
                    low_score_threshold=args.reconstruct_low_score_threshold,
                    force_target_audio=bool(args.reconstruct_force_target_audio),
                    strict_visual_verify=bool(args.reconstruct_strict_visual_verify),
                    boundary_glitch_fix=bool(args.reconstruct_boundary_glitch_fix),
                    adjacent_overlap_trigger=float(args.optimize_adjacent_overlap_trigger),
                    adjacent_lag_trigger=float(args.optimize_adjacent_lag_trigger),
                    isolated_drift_trigger=float(args.optimize_isolated_drift_trigger),
                    cross_source_mapping_jump_trigger=float(args.optimize_cross_source_mapping_jump_trigger),
                    timeout_sec=allowed,
                    config_path=args.config,
                )
                opt_clip_elapsed = round(time.perf_counter() - cstart, 3)
                if not ok or not opt_output.exists():
                    st.optimization_note = f"opt_reconstruct_failed:{err}"
                    continue

                if opt_clip_elapsed > allowed:
                    st.optimization_note = f"opt_clip_slower_rejected:{opt_clip_elapsed:.3f}s>{allowed:.3f}s"
                    continue

                old_candidate = st.candidate
                st.candidate = opt_output
                ok2, err2, sum2, manifest2, report2, audit2 = _run_audit(st, retry_tag=f"_opt{retry}")
                st.candidate = old_candidate
                if not ok2:
                    st.optimization_note = f"opt_audit_failed:{err2}"
                    continue

                new_mismatch = int((sum2 or {}).get("mismatch_points", 0) or 0)
                if new_mismatch < baseline_mismatch:
                    try:
                        if old_candidate and old_candidate.exists():
                            bak = old_candidate.with_suffix(".pre_opt.bak.mp4")
                            if bak.exists():
                                bak.unlink()
                            old_candidate.rename(bak)
                            opt_output.rename(old_candidate)
                            bak.unlink(missing_ok=True)
                            st.candidate = old_candidate
                        else:
                            st.candidate = opt_output
                    except Exception:
                        st.optimization_note = "opt_apply_failed_keep_opt_file"
                        continue

                    st.optimization_applied = True
                    st.clip_elapsed_sec = float(opt_clip_elapsed)
                    st.summary = sum2 or {}
                    st.manifest_path = manifest2
                    st.report_html = report2
                    st.audit_elapsed_sec = float(audit2)
                    st.total_elapsed_sec = round(float(st.clip_elapsed_sec or 0.0) + float(st.audit_elapsed_sec or 0.0), 3)
                    st.optimization_note = f"mismatch:{baseline_mismatch}->{new_mismatch}, clip={opt_clip_elapsed:.3f}s<=baseline:{baseline_budget:.3f}s"
                    accepted = True
                    break
                else:
                    st.optimization_note = f"no_improve:{baseline_mismatch}->{new_mismatch}"

            if accepted:
                print(f"  - 优化已应用: {st.material.name} ({st.optimization_note})")
            else:
                print(f"  - 优化未应用: {st.material.name} ({st.optimization_note or 'no_acceptable_result'})")

    # 汇总输出
    rows: List[Dict] = []
    for st in states.values():
        if st.status == "ok":
            rows.append({
                "name": st.material.name,
                "status": "ok",
                "material": str(st.material),
                "candidate": str(st.candidate) if st.candidate else "",
                "candidate_mode": st.candidate_mode,
                "manifest": str(st.manifest_path) if st.manifest_path else "",
                "report_html": str(st.report_html) if st.report_html else "",
                "manifest_rel": str(st.manifest_path.relative_to(output_root)) if st.manifest_path else "",
                "report_rel": str(st.report_html.relative_to(output_root)) if st.report_html else "",
                "clip_budget_sec": st.clip_budget_sec,
                "clip_elapsed_sec": st.clip_elapsed_sec,
                "audit_elapsed_sec": st.audit_elapsed_sec,
                "total_elapsed_sec": st.total_elapsed_sec,
                "optimization_applied": bool(st.optimization_applied),
                "optimization_attempted": bool(st.optimization_attempted),
                "optimization_note": st.optimization_note,
                "summary": st.summary or {},
            })
        else:
            rows.append({
                "name": st.material.name,
                "status": "failed",
                "error": st.error,
                "candidate": str(st.candidate) if st.candidate else "",
                "candidate_mode": st.candidate_mode,
                "clip_budget_sec": st.clip_budget_sec,
                "clip_elapsed_sec": st.clip_elapsed_sec,
                "audit_elapsed_sec": st.audit_elapsed_sec,
                "total_elapsed_sec": st.total_elapsed_sec,
                "optimization_applied": bool(st.optimization_applied),
                "optimization_attempted": bool(st.optimization_attempted),
                "optimization_note": st.optimization_note,
            })

    finished_at = datetime.now().isoformat()
    summary_json, summary_html = build_total_report(
        output_root=output_root,
        rows=rows,
        started_at=started_at,
        finished_at=finished_at,
        clip_phase_elapsed_sec=clip_phase_elapsed_sec,
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
