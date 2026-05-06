#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

VIDEO_EXTS = {".mp4", ".mov", ".mkv"}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class Sample:
    sample_id: str
    target_video: Path
    source_project: str
    expected_shape: str
    acceptance_checks: dict


def derive_project_dirs(target_video: Path) -> tuple[Path, Path, Path]:
    target_dir = target_video.parent
    if target_dir.name not in {"素材", "adx原"}:
        raise ValueError(f"unsupported target path: {target_video}")
    project_dir = target_dir.parent
    source_dir = project_dir / "剧集"
    if not source_dir.is_dir():
        raise ValueError(f"source dir not found: {source_dir}")
    output_dir = project_dir / "output"
    return project_dir, source_dir, output_dir


def load_samples(config_path: Path) -> List[Sample]:
    payload = read_json(config_path)
    samples: List[Sample] = []
    for item in payload.get("samples") or []:
        samples.append(
            Sample(
                sample_id=str(item["id"]),
                target_video=Path(str(item["target_video"])).resolve(),
                source_project=str(item.get("source_project") or ""),
                expected_shape=str(item.get("expected_shape") or ""),
                acceptance_checks=dict(item.get("acceptance_checks") or {}),
            )
        )
    return samples


def nearest_boundary(boundaries: List[dict], target_sec: float) -> Optional[dict]:
    if not boundaries:
        return None
    return min(boundaries, key=lambda item: abs(float(item.get("target_second") or 0.0) - float(target_sec)))


def tail_file(path: Path, line_count: int = 20) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def video_path_for_report(report_path: Path) -> Path:
    name = report_path.name
    suffix = ".quality_report.json"
    if name.endswith(suffix):
        return report_path.with_name(name[: -len(suffix)] + ".mp4")
    return report_path.with_suffix(".mp4")


def find_existing_reports(sample: Sample, root_dir: Path, output_root: Path) -> List[Path]:
    reports: List[Path] = []
    current_report = output_root / sample.sample_id / f"{sample.target_video.stem}_V3_FAST.quality_report.json"
    if current_report.is_file():
        reports.append(current_report)
    search_root = root_dir / "runtime" / "temp_outputs" / "regression_guard"
    if search_root.is_dir():
        pattern = f"**/{sample.sample_id}/{sample.target_video.stem}_V3_FAST.quality_report.json"
        for report_path in search_root.glob(pattern):
            if report_path not in reports and report_path.is_file():
                reports.append(report_path)
    return sorted(reports, key=lambda path: path.stat().st_mtime, reverse=True)


def evaluate_sample_output(
    sample: Sample,
    root_dir: Path,
    config_path: Path,
    project_dir: Path,
    out_video: Path,
    report_path: Path,
    returncode: int,
    stdout_tail: str,
    stderr_tail: str,
    validation_mode: str,
) -> dict:
    result = {
        "sample_id": sample.sample_id,
        "target_video": str(sample.target_video),
        "project_dir": str(project_dir),
        "output_video": str(out_video),
        "report_path": str(report_path),
        "returncode": int(returncode),
        "validation_mode": str(validation_mode),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "passed": False,
        "violations": [],
    }

    if not report_path.is_file():
        result["violations"].append("quality_report_missing")
        return result

    report = read_json(report_path)
    result["report_summary"] = {
        "reconstruct_status": report.get("reconstruct_status"),
        "render_status": (report.get("render_metrics") or {}).get("status"),
        "available_segments": report.get("available_segments"),
        "total_segments": report.get("total_segments"),
        "avg_combined_score": report.get("avg_combined_score"),
        "low_score_segments": report.get("low_score_segments"),
        "material_shape": ((report.get("material_analysis") or {}).get("material_shape")),
    }

    checks = sample.acceptance_checks
    if not out_video.is_file():
        result["violations"].append("output_missing")
    if str(report.get("reconstruct_status") or "") != "ok":
        result["violations"].append("reconstruct_status_not_ok")
    if str((report.get("render_metrics") or {}).get("status") or "") != "ok":
        result["violations"].append("render_status_not_ok")

    available_segments = int(report.get("available_segments") or 0)
    total_segments = int(report.get("total_segments") or 0)
    if total_segments > 0:
        available_ratio = float(available_segments) / float(total_segments)
        result["report_summary"]["available_segment_ratio"] = available_ratio
        min_ratio = checks.get("min_available_segment_ratio")
        if min_ratio is not None and available_ratio < float(min_ratio):
            result["violations"].append(f"available_segment_ratio<{min_ratio}")

    min_avg = checks.get("min_avg_combined_score")
    if min_avg is not None and float(report.get("avg_combined_score") or 0.0) < float(min_avg):
        result["violations"].append(f"avg_combined_score<{min_avg}")

    max_low = checks.get("max_low_score_segments")
    if max_low is not None and int(report.get("low_score_segments") or 0) > int(max_low):
        result["violations"].append(f"low_score_segments>{max_low}")

    expected_shapes = [
        str(shape)
        for shape in (checks.get("expected_shapes") or [])
        if str(shape).strip()
    ]
    if sample.expected_shape:
        expected_shapes.append(str(sample.expected_shape))
    if expected_shapes:
        actual_shape = str((report.get("material_analysis") or {}).get("material_shape") or "")
        if actual_shape not in set(expected_shapes):
            result["violations"].append(f"material_shape_not_in:{','.join(expected_shapes)}")

    boundaries = list(report.get("boundary_details") or [])
    for point in checks.get("must_hold_points") or []:
        boundary = nearest_boundary(boundaries, float(point.get("time_sec") or 0.0))
        if boundary is None:
            result["violations"].append(f"missing_boundary:{point.get('label') or point.get('time_sec')}")
            continue
        gap_sec = boundary.get("gap_sec")
        if gap_sec is None:
            continue
        gap_sec = float(gap_sec)
        max_neg = point.get("max_negative_overlap_sec")
        max_pos = point.get("max_positive_gap_sec")
        if max_neg is not None and gap_sec < -float(max_neg):
            result["violations"].append(f"boundary_overlap:{point.get('label') or point.get('time_sec')}")
        if max_pos is not None and gap_sec > float(max_pos):
            result["violations"].append(f"boundary_gap:{point.get('label') or point.get('time_sec')}")

    # 预留：需要更严时可在样本清单中打开 run_3s_audit。
    if bool(checks.get("run_3s_audit", False)):
        audit_script = root_dir / "skills" / "ai-video-audit" / "scripts" / "build_ai_video_audit_bundle.py"
        audit_dir = sample_root / "ai_audit_3s"
        audit_cmd = [
            sys.executable,
            str(audit_script),
            "--config",
            str(config_path),
            "--target",
            str(sample.target_video),
            "--candidate",
            str(out_video),
            "--interval",
            "3",
            "--clip-duration",
            "2",
            "--output-dir",
            str(audit_dir),
        ]
        audit_proc = subprocess.run(audit_cmd, check=False, capture_output=True, text=True)
        manifest = audit_dir / "audit_manifest.json"
        result["audit_manifest"] = str(manifest)
        result["audit_returncode"] = int(audit_proc.returncode)
        if manifest.is_file():
            audit_payload = read_json(manifest)
            mismatch_points = int(((audit_payload.get("summary") or {}).get("mismatch_points") or 0))
            result["report_summary"]["mismatch_points"] = mismatch_points
            max_mismatch = checks.get("max_mismatch_points")
            if max_mismatch is not None and mismatch_points > int(max_mismatch):
                result["violations"].append(f"mismatch_points>{max_mismatch}")
        else:
            result["violations"].append("audit_manifest_missing")

    result["passed"] = len(result["violations"]) == 0
    return result


def run_sample(sample: Sample, root_dir: Path, config_path: Path, output_root: Path) -> dict:
    project_dir, source_dir, _ = derive_project_dirs(sample.target_video)
    checks = sample.acceptance_checks

    if bool(checks.get("allow_existing_report_validation", False)):
        for existing_report in find_existing_reports(sample, root_dir, output_root):
            existing_video = video_path_for_report(existing_report)
            result = evaluate_sample_output(
                sample,
                root_dir,
                config_path,
                project_dir,
                existing_video,
                existing_report,
                0,
                "",
                "",
                "existing_report",
            )
            if result["passed"]:
                return result

    sample_root = output_root / sample.sample_id
    sample_root.mkdir(parents=True, exist_ok=True)
    out_video = sample_root / f"{sample.target_video.stem}_V3_FAST.mp4"
    cache_dir = sample_root / "cache"
    frame_index_cache_dir = project_dir / ".cache_fast_v7" / "frame_index_shared"
    frame_index_cache_dir.mkdir(parents=True, exist_ok=True)
    if out_video.exists():
        out_video.unlink()
    report_path = out_video.with_suffix(".quality_report.json")
    if report_path.exists():
        report_path.unlink()

    stdout_log = sample_root / "regression_rebuild.stdout.log"
    stderr_log = sample_root / "regression_rebuild.stderr.log"
    cmd = [
        sys.executable,
        str(root_dir / "fast_v7.py"),
        "--config",
        str(config_path),
        "--target",
        str(sample.target_video),
        "--source-dir",
        str(source_dir),
        "--output",
        str(out_video),
        "--cache",
        str(cache_dir),
        "--frame-index-cache-dir",
        str(frame_index_cache_dir),
        "--no-run-evidence-validation",
        "--no-run-ai-verify-snapshots",
    ]
    timeout_sec = checks.get("max_rebuild_seconds")
    started = time.time()
    try:
        with stdout_log.open("w", encoding="utf-8") as stdout_fh, stderr_log.open("w", encoding="utf-8") as stderr_fh:
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                timeout=float(timeout_sec) if timeout_sec is not None else None,
            )
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired:
        returncode = 124

    result = evaluate_sample_output(
        sample,
        root_dir,
        config_path,
        project_dir,
        out_video,
        report_path,
        returncode,
        tail_file(stdout_log),
        tail_file(stderr_log),
        "rebuild",
    )
    result["elapsed_sec"] = int(time.time() - started)
    if returncode == 124:
        result["violations"].append(f"rebuild_timeout>{timeout_sec}")
        result["passed"] = False
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量修复后的外置回归护栏")
    parser.add_argument("--config", required=True, help="主配置文件")
    parser.add_argument("--samples-config", required=True, help="回归样本清单")
    parser.add_argument("--batch-state", default="", help="关联的批次状态文件")
    parser.add_argument("--output-root", required=True, help="回归输出目录")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    root_dir = Path(__file__).resolve().parent.parent
    config_path = Path(args.config).resolve()
    samples_config = Path(args.samples_config).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples = load_samples(samples_config)
    if not samples:
        print("[regression] no samples configured")
        return 0

    results = [run_sample(sample, root_dir, config_path, output_root) for sample in samples]
    summary = {
        "generated_at": now_stamp(),
        "samples_config": str(samples_config),
        "batch_state": str(Path(args.batch_state).resolve()) if args.batch_state else "",
        "total_samples": len(results),
        "passed_samples": sum(1 for item in results if item["passed"]),
        "failed_samples": sum(1 for item in results if not item["passed"]),
        "results": results,
    }
    summary_path = output_root / "regression_guard.summary.json"
    write_json_atomic(summary_path, summary)
    print(f"[regression] summary={summary_path}")
    print(f"[regression] passed={summary['passed_samples']} failed={summary['failed_samples']}")
    return 0 if summary["failed_samples"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
