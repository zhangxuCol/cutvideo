#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
PAUSE_EXIT_CODE = 10
REGRESSION_BLOCK_EXIT_CODE = 11
STATE_VERSION = 1
SOURCE_POOL_GAP_FAILURE_REASONS = {"insufficient_coverage_in_source_pool"}
ALL_UNMATCHED_FAILURE_REASONS = {"target_sequence_low_confidence"}
ISOLATED_BOUNDARY_GAP_FAILURE_REASONS = {"too_many_unresolved_multi_source_boundaries"}
NON_REPAIRABLE_FAILURE_REASONS = set()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")


def load_cfg(root_dir: Path, config_path: Optional[Path]) -> dict:
    cfg_file = config_path or (root_dir / "configurations" / "ai_pipeline.defaults.json")
    if not cfg_file.is_file():
        return {}
    return read_json(cfg_file)


def bool_cfg(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_project_layout(project_dir: Path) -> Tuple[Path, Path]:
    if (project_dir / "素材").is_dir():
        return project_dir / "素材", project_dir / "output"
    if (project_dir / "adx原").is_dir():
        return project_dir / "adx原", project_dir / "output"
    raise SystemExit(f"material dir not found under: {project_dir} (expect 素材 or adx原)")


def archive_existing_output_dir(output_dir: Path, stamp: str) -> Optional[Path]:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return None
    entries = [p for p in output_dir.iterdir()]
    if not entries:
        return None
    backup_dir = output_dir.parent / f"{output_dir.name}_backup_{stamp}"
    output_dir.rename(backup_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def list_materials(material_dir: Path, output_dir: Path) -> List[Path]:
    files = sorted(
        p for p in material_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in VIDEO_EXTS
            and not p.name.startswith(".")
            and not p.name.startswith("._")
        )
    )
    todo: List[Path] = []
    for file_path in files:
        stem = file_path.stem
        output_video = output_dir / f"{stem}_V3_FAST.mp4"
        if output_video.exists():
            continue
        todo.append(file_path)
    return todo


def stem_for(material_path: Path) -> str:
    return material_path.stem


def tail_missing_meta(report: dict) -> dict:
    failure = report.get("failure") or {}
    details = failure.get("details") if isinstance(failure.get("details"), dict) else {}
    missing_indices = sorted({int(x) for x in (details.get("missing_indices") or [])})
    total = int(details.get("expected_total_segments") or report.get("total_segments") or 0)
    if total <= 0:
        return {"tail_missing_count": 0, "tail_missing_start_index": -1, "tail_missing_dominant": False}
    missing_set = set(missing_indices)
    tail_count = 0
    cursor = total - 1
    while cursor >= 0 and cursor in missing_set:
        tail_count += 1
        cursor -= 1
    missing_count = int(details.get("missing_segments_count") or failure.get("missing_segments_count") or len(missing_indices))
    segment_duration = 5.0
    try:
        target_duration = float(report.get("target_duration") or 0.0)
        if total > 0 and target_duration > 0.0:
            segment_duration = max(0.1, float(target_duration) / float(total))
    except Exception:
        segment_duration = 5.0
    dominant = bool(
        (
            tail_count >= max(3, int(total * 0.12))
            or tail_count * float(segment_duration) >= max(30.0, float(segment_duration) * 3.0)
        )
        and tail_count >= max(1, int(missing_count * 0.65))
    )
    return {
        "tail_missing_count": int(tail_count),
        "tail_missing_start_index": int(total - tail_count) if tail_count > 0 else -1,
        "tail_missing_dominant": bool(dominant),
    }


def classify_failure_reason(report: dict) -> str:
    failure = report.get("failure") or {}
    details = failure.get("details") if isinstance(failure.get("details"), dict) else {}
    render_metrics = report.get("render_metrics") or {}
    reason = str(failure.get("reason") or render_metrics.get("error") or "unknown")
    if reason == "too_many_unresolved_multi_source_boundaries":
        material_analysis = report.get("material_analysis") or {}
        selected_strategy = str(material_analysis.get("selected_strategy") or details.get("selected_strategy") or "")
        tail_meta = tail_missing_meta(report)
        if selected_strategy == "sequence_model" and bool(tail_meta.get("tail_missing_dominant")):
            return "insufficient_coverage_in_source_pool"
    return reason


def report_is_success(report: dict) -> bool:
    render_metrics = report.get("render_metrics") or {}
    if str(report.get("reconstruct_status") or "").lower() != "ok":
        return False
    if str(render_metrics.get("status") or "ok").lower() != "ok":
        return False
    output_video = str(report.get("output_video") or "")
    if output_video and not Path(output_video).is_file():
        return False
    return True


def is_non_repairable_failure(result: dict) -> bool:
    return str(result.get("final_failure_reason") or "") in NON_REPAIRABLE_FAILURE_REASONS


def is_source_pool_gap_failure(result: dict) -> bool:
    reason = str(result.get("final_failure_reason") or result.get("reason") or "")
    return bool(result.get("source_pool_gap_failure")) or reason in SOURCE_POOL_GAP_FAILURE_REASONS


def is_all_unmatched_failure(result: dict) -> bool:
    if bool(result.get("all_unmatched_failure")) or bool(result.get("all_segments_unmatched")):
        return True
    reason = str(result.get("final_failure_reason") or result.get("reason") or "")
    expected_total = int(result.get("expected_total_segments") or 0)
    available = int(result.get("available_segments") or 0)
    missing = int(result.get("missing_segments_count") or result.get("missing") or 0)
    return bool(reason in ALL_UNMATCHED_FAILURE_REASONS and expected_total > 0 and available == 0 and missing >= expected_total)


def is_isolated_boundary_gap_failure(result: dict) -> bool:
    if bool(result.get("isolated_boundary_gap_failure")):
        return True
    reason = str(result.get("final_failure_reason") or result.get("reason") or "")
    if reason not in ISOLATED_BOUNDARY_GAP_FAILURE_REASONS:
        return False
    expected_total = int(result.get("expected_total_segments") or 0)
    available = int(result.get("available_segments") or 0)
    missing = int(result.get("missing_segments_count") or result.get("missing") or 0)
    if expected_total <= 0 or missing <= 0:
        return False
    max_missing = max(1, min(2, int(expected_total * 0.03) + 1))
    return bool(missing <= max_missing and available + missing >= expected_total)


def should_run_source_pool_gap_fallback(report_info: dict) -> bool:
    return bool(
        is_source_pool_gap_failure(report_info)
        or is_all_unmatched_failure(report_info)
        or is_isolated_boundary_gap_failure(report_info)
    )


def set_failure_pause_baseline(state: dict) -> None:
    summary = state.get("summary") or {}
    state["failure_pause_baseline"] = {
        "final_fail": int(summary.get("final_fail") or 0),
        "repairable_final_fail": int(summary.get("repairable_final_fail") or 0),
        "source_pool_gap_final_fail": int(summary.get("source_pool_gap_final_fail") or 0),
        "non_repairable_final_fail": int(summary.get("non_repairable_final_fail") or 0),
        "set_at": now_iso(),
    }


def new_final_failures_since_baseline(state: dict) -> int:
    summary = state.get("summary") or {}
    baseline = state.get("failure_pause_baseline") or {}
    return max(0, int(summary.get("final_fail") or 0) - int(baseline.get("final_fail") or 0))


def normalize_existing_failure_results(state: dict) -> None:
    """Refresh existing failed results from reports before deciding what to rerun."""
    results = state.get("results_by_stem") or {}
    for result in results.values():
        if result.get("final_status") != "FAIL":
            continue
        report_path = Path(str(result.get("report_path") or ""))
        if not report_path.is_file():
            continue
        try:
            report = read_json(report_path)
        except Exception:
            continue
        info = read_report_info(report_path)
        tail_meta = tail_missing_meta(report)
        if report_is_success(report):
            result["final_status"] = "OK"
            result["stage"] = result.get("stage") or "normalized_existing_ok"
            result["failure_stage"] = ""
            result["final_failure_reason"] = ""
            result["missing_segments_count"] = int(info.get("missing") or 0)
            result["expected_total_segments"] = int(info.get("expected_total_segments") or 0)
            result["available_segments"] = int(info.get("available_segments") or 0)
            result["source_pool_gap_failure"] = False
            result["all_unmatched_failure"] = False
            result["isolated_boundary_gap_failure"] = False
            result["source_pool_gap_target_fallback"] = bool(info.get("source_pool_gap_target_fallback"))
            result["non_repairable_failure"] = False
            result["reconstruct_status"] = info.get("reconstruct_status", result.get("reconstruct_status", "unknown"))
            continue
        result["shape"] = info.get("shape", result.get("shape", "unknown"))
        result["final_failure_reason"] = info.get("reason", result.get("final_failure_reason", "unknown"))
        result["missing_segments_count"] = int(info.get("missing") or result.get("missing_segments_count") or 0)
        result["expected_total_segments"] = int(info.get("expected_total_segments") or result.get("expected_total_segments") or 0)
        result["available_segments"] = int(info.get("available_segments") or result.get("available_segments") or 0)
        result["source_pool_gap_failure"] = bool(is_source_pool_gap_failure(info))
        result["all_unmatched_failure"] = bool(is_all_unmatched_failure(info))
        result["isolated_boundary_gap_failure"] = bool(is_isolated_boundary_gap_failure(info))
        result["source_pool_gap_target_fallback"] = bool(info.get("source_pool_gap_target_fallback"))
        result["non_repairable_failure"] = bool(is_non_repairable_failure(result))
        if int(tail_meta.get("tail_missing_count") or 0) > 0:
            result["tail_missing_count"] = int(tail_meta["tail_missing_count"])
            result["tail_missing_start_index"] = int(tail_meta["tail_missing_start_index"])
            result["tail_missing_dominant"] = bool(tail_meta["tail_missing_dominant"])
    state["results_by_stem"] = results


def default_report_info(report_path: Path, reason: str) -> dict:
    return {
        "shape": "unknown",
        "strategy": "unknown",
        "reason": str(reason),
        "missing": 0,
        "reconstruct_status": "unknown",
        "report_path": str(report_path),
        "tail_missing_count": 0,
        "tail_missing_start_index": -1,
        "tail_missing_dominant": False,
        "expected_total_segments": 0,
        "available_segments": 0,
        "source_pool_gap_failure": False,
        "all_unmatched_failure": False,
        "isolated_boundary_gap_failure": False,
        "source_pool_gap_target_fallback": False,
        "all_segments_unmatched": False,
    }


def read_report_info(report_path: Path) -> dict:
    if not report_path.is_file():
        return default_report_info(report_path, "missing_report_after_failure")
    try:
        report = read_json(report_path)
    except Exception:
        return default_report_info(report_path, "invalid_report_after_failure")
    material_analysis = report.get("material_analysis") or {}
    failure = report.get("failure") or {}
    details = failure.get("details") if isinstance(failure.get("details"), dict) else {}
    render_metrics = report.get("render_metrics") or {}
    reason = classify_failure_reason(report)
    tail_meta = tail_missing_meta(report)
    source_pool_gap_target_fallback = report.get("source_pool_gap_target_fallback")
    if not isinstance(source_pool_gap_target_fallback, dict):
        source_pool_gap_target_fallback = {}
    expected_total_segments = int(
        details.get("expected_total_segments")
        or report.get("total_segments")
        or 0
    )
    available_segments = int(report.get("available_segments") or 0)
    missing = int(
        details.get("missing_segments_count")
        or failure.get("missing_segments_count")
        or 0
    )
    all_unmatched = bool(
        str(reason) in ALL_UNMATCHED_FAILURE_REASONS
        and expected_total_segments > 0
        and available_segments == 0
        and missing >= expected_total_segments
    )
    all_unmatched = bool(all_unmatched or source_pool_gap_target_fallback.get("all_segments_unmatched"))
    source_pool_gap = bool(str(reason) in SOURCE_POOL_GAP_FAILURE_REASONS)
    isolated_boundary_gap = bool(
        str(reason) in ISOLATED_BOUNDARY_GAP_FAILURE_REASONS
        and expected_total_segments > 0
        and missing > 0
        and missing <= max(1, min(2, int(expected_total_segments * 0.03) + 1))
        and available_segments + missing >= expected_total_segments
    )
    return {
        "shape": str(material_analysis.get("material_shape") or "unknown"),
        "strategy": str(material_analysis.get("selected_strategy") or "unknown"),
        "reason": str(reason or failure.get("reason") or render_metrics.get("error") or "unknown"),
        "missing": int(missing),
        "reconstruct_status": str(report.get("reconstruct_status") or "unknown"),
        "report_path": str(report_path),
        "tail_missing_count": int(tail_meta.get("tail_missing_count") or 0),
        "tail_missing_start_index": int(tail_meta.get("tail_missing_start_index") or -1),
        "tail_missing_dominant": bool(tail_meta.get("tail_missing_dominant")),
        "expected_total_segments": int(expected_total_segments),
        "available_segments": int(available_segments),
        "source_pool_gap_failure": bool(source_pool_gap),
        "all_unmatched_failure": bool(all_unmatched),
        "isolated_boundary_gap_failure": bool(isolated_boundary_gap),
        "source_pool_gap_target_fallback": bool(source_pool_gap_target_fallback.get("enabled", False)),
        "all_segments_unmatched": bool(all_unmatched),
    }


def should_retry(report_info: dict, retry_enabled: bool, retry_on_fail: bool, missing_threshold: int) -> bool:
    if not retry_enabled or not retry_on_fail:
        return False
    shape = str(report_info.get("shape") or "unknown")
    reason = str(report_info.get("reason") or "unknown")
    missing = int(report_info.get("missing") or 0)
    if shape in {"stitched_subset", "full_library_mixed_sequence"}:
        return True
    if missing >= int(missing_threshold):
        return True
    return reason in {
        "too_many_unresolved_multi_source_boundaries",
        "too_many_missing_segments_without_target_fallback",
        "target_sequence_low_confidence",
        "insufficient_coverage_in_source_pool",
    }


def run_once(run_one: Path, material_path: Path, runtime_log: Path, extra_args: Optional[List[str]] = None) -> int:
    cmd = ["zsh", str(run_one), str(material_path)]
    if extra_args:
        cmd.extend(extra_args)
    with runtime_log.open("a", encoding="utf-8") as log_fh:
        return subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, check=False).returncode


def fallback_stage_name(report_info: dict) -> str:
    if is_all_unmatched_failure(report_info):
        return "all_unmatched_fallback"
    return "source_pool_gap_fallback"


def result_from_info(
    material_path: Path,
    stem: str,
    final_status: str,
    stage: str,
    failure_stage: str,
    elapsed: int,
    info: dict,
    first_info: Optional[dict] = None,
    retry_strategy: str = "",
) -> dict:
    final_failure_reason = "" if final_status == "OK" else str(info.get("reason") or "unknown")
    result = {
        "material_path": str(material_path),
        "stem": stem,
        "final_status": final_status,
        "stage": stage,
        "failure_stage": failure_stage,
        "elapsed_sec": int(elapsed),
        "shape": info.get("shape", "unknown"),
        "first_shape": (first_info or info).get("shape", "unknown"),
        "first_pass_strategy": (first_info or info).get("strategy", "unknown"),
        "retry_strategy": retry_strategy,
        "final_failure_reason": final_failure_reason,
        "missing_segments_count": int(info.get("missing") or 0),
        "expected_total_segments": int(info.get("expected_total_segments") or 0),
        "available_segments": int(info.get("available_segments") or 0),
        "source_pool_gap_failure": bool(is_source_pool_gap_failure(info)),
        "all_unmatched_failure": bool(is_all_unmatched_failure(info)),
        "isolated_boundary_gap_failure": bool(is_isolated_boundary_gap_failure(info)),
        "source_pool_gap_target_fallback": bool(info.get("source_pool_gap_target_fallback", False)),
        "non_repairable_failure": False,
        "reconstruct_status": info.get("reconstruct_status", "unknown"),
        "report_path": info.get("report_path", ""),
        "tail_missing_count": int(info.get("tail_missing_count") or 0),
        "tail_missing_start_index": int(info.get("tail_missing_start_index") or -1),
        "tail_missing_dominant": bool(info.get("tail_missing_dominant")),
    }
    if final_status != "OK":
        result["non_repairable_failure"] = bool(is_non_repairable_failure(result))
    return result


def run_material(
    material_path: Path,
    run_one: Path,
    runtime_log: Path,
    output_dir: Path,
    retry_enabled: bool,
    retry_on_fail: bool,
    retry_missing_threshold: int,
) -> dict:
    stem = stem_for(material_path)
    report_path = output_dir / f"{stem}_V3_FAST.quality_report.json"
    started = time.time()
    append_line(runtime_log, f"[start] {stem} {now_iso()}")

    first_rc = run_once(run_one, material_path, runtime_log)
    if first_rc == 0:
        info = read_report_info(report_path)
        elapsed = int(time.time() - started)
        result = result_from_info(material_path, stem, "OK", "first", "", elapsed, info)
        append_line(runtime_log, f"[done] OK {stem} ({elapsed}s) stage=first shape={info['shape']} strategy={info['strategy']}")
        return result

    first_info = read_report_info(report_path)
    if should_run_source_pool_gap_fallback(first_info):
        append_line(
            runtime_log,
            f"[fallback] {stem} stage={fallback_stage_name(first_info)} reason={first_info['reason']} missing={first_info['missing']} "
            f"available={first_info.get('available_segments', 0)} total={first_info.get('expected_total_segments', 0)}",
        )
        fallback_rc = run_once(
            run_one,
            material_path,
            runtime_log,
            ["--force-strategy", "sequence_model", "--source-pool-gap-target-fallback"],
        )
        fallback_info = read_report_info(report_path)
        elapsed = int(time.time() - started)
        if fallback_rc == 0:
            result = result_from_info(
                material_path,
                stem,
                "OK",
                fallback_stage_name(first_info),
                "",
                elapsed,
                fallback_info,
                first_info,
                "sequence_model+source_pool_gap_target_fallback",
            )
            append_line(runtime_log, f"[done] OK {stem} ({elapsed}s) stage={result['stage']} shape={fallback_info['shape']} strategy={fallback_info['strategy']}")
            return result
        result = result_from_info(
            material_path,
            stem,
            "FAIL",
            fallback_stage_name(first_info),
            "source_pool_gap_fallback",
            elapsed,
            fallback_info,
            first_info,
            "sequence_model+source_pool_gap_target_fallback",
        )
        append_line(runtime_log, f"[done] FAIL {stem} ({elapsed}s) stage={result['stage']} reason={fallback_info['reason']} shape={fallback_info['shape']} strategy={fallback_info['strategy']} missing={fallback_info['missing']}")
        return result

    if should_retry(first_info, retry_enabled, retry_on_fail, retry_missing_threshold):
        append_line(
            runtime_log,
            f"[retry] {stem} shape={first_info['shape']} strategy={first_info['strategy']} reason={first_info['reason']} missing={first_info['missing']}",
        )
        retry_rc = run_once(run_one, material_path, runtime_log, ["--force-strategy", "sequence_model"])
        retry_info = read_report_info(report_path)
        if retry_rc == 0:
            elapsed = int(time.time() - started)
            result = result_from_info(material_path, stem, "OK", "retry", "", elapsed, retry_info, first_info, "sequence_model")
            append_line(runtime_log, f"[done] OK {stem} ({elapsed}s) stage=retry shape={retry_info['shape']} strategy={retry_info['strategy']}")
            return result

        if should_run_source_pool_gap_fallback(retry_info):
            append_line(
                runtime_log,
                f"[fallback] {stem} stage={fallback_stage_name(retry_info)} reason={retry_info['reason']} missing={retry_info['missing']} "
                f"available={retry_info.get('available_segments', 0)} total={retry_info.get('expected_total_segments', 0)}",
            )
            fallback_rc = run_once(
                run_one,
                material_path,
                runtime_log,
                ["--force-strategy", "sequence_model", "--source-pool-gap-target-fallback"],
            )
            fallback_info = read_report_info(report_path)
            elapsed = int(time.time() - started)
            if fallback_rc == 0:
                result = result_from_info(
                    material_path,
                    stem,
                    "OK",
                    fallback_stage_name(retry_info),
                    "",
                    elapsed,
                    fallback_info,
                    first_info,
                    "sequence_model+source_pool_gap_target_fallback",
                )
                append_line(runtime_log, f"[done] OK {stem} ({elapsed}s) stage={result['stage']} shape={fallback_info['shape']} strategy={fallback_info['strategy']}")
                return result
            result = result_from_info(
                material_path,
                stem,
                "FAIL",
                fallback_stage_name(retry_info),
                "source_pool_gap_fallback",
                elapsed,
                fallback_info,
                first_info,
                "sequence_model+source_pool_gap_target_fallback",
            )
            append_line(runtime_log, f"[done] FAIL {stem} ({elapsed}s) stage={result['stage']} reason={fallback_info['reason']} shape={fallback_info['shape']} strategy={fallback_info['strategy']} missing={fallback_info['missing']}")
            return result

        elapsed = int(time.time() - started)
        result = result_from_info(material_path, stem, "FAIL", "retry", "retry_pass", elapsed, retry_info, first_info, "sequence_model")
        append_line(runtime_log, f"[done] FAIL {stem} ({elapsed}s) stage=retry reason={retry_info['reason']} shape={retry_info['shape']} strategy={retry_info['strategy']} missing={retry_info['missing']}")
        return result

    elapsed = int(time.time() - started)
    result = result_from_info(material_path, stem, "FAIL", "first", "first_pass", elapsed, first_info)
    append_line(runtime_log, f"[done] FAIL {stem} ({elapsed}s) stage=first reason={first_info['reason']} shape={first_info['shape']} strategy={first_info['strategy']} missing={first_info['missing']}")
    return result


def result_to_progress_line(result: dict) -> str:
    base = [
        result["final_status"],
        result["stem"],
        f"{int(result['elapsed_sec'])}s",
        f"stage={result['stage']}",
        f"first_pass_strategy={result.get('first_pass_strategy', '')}",
        f"shape={result.get('shape', 'unknown')}",
        f"retry_strategy={result.get('retry_strategy', '')}",
        f"failure_stage={result.get('failure_stage', '')}",
        f"final_failure_reason={result.get('final_failure_reason', '')}",
        f"source_pool_gap_failure={bool(result.get('source_pool_gap_failure', False))}",
        f"all_unmatched_failure={bool(result.get('all_unmatched_failure', False))}",
        f"isolated_boundary_gap_failure={bool(result.get('isolated_boundary_gap_failure', False))}",
        f"source_pool_gap_target_fallback={bool(result.get('source_pool_gap_target_fallback', False))}",
        f"non_repairable_failure={bool(result.get('non_repairable_failure', False))}",
        f"missing_segments_count={int(result.get('missing_segments_count', 0) or 0)}",
        f"available_segments={int(result.get('available_segments', 0) or 0)}",
        f"expected_total_segments={int(result.get('expected_total_segments', 0) or 0)}",
        f"reconstruct_status={result.get('reconstruct_status', 'unknown')}",
        f"report_path={result.get('report_path', '')}",
    ]
    return "|".join(base)


def refresh_state(state: dict, pending: List[str], inflight: List[str]) -> None:
    results = state.get("results_by_stem") or {}
    success = sum(1 for item in results.values() if item.get("final_status") == "OK")
    failures = [item for item in results.values() if item.get("final_status") == "FAIL"]
    source_pool_gap_failures = [
        item for item in failures
        if is_source_pool_gap_failure(item) or is_all_unmatched_failure(item)
    ]
    non_repairable_failures = [item for item in failures if is_non_repairable_failure(item)]
    repairable_failures = [
        item for item in failures
        if not is_non_repairable_failure(item)
        and not (is_source_pool_gap_failure(item) or is_all_unmatched_failure(item))
    ]
    reasons = [str(item.get("final_failure_reason") or "unknown") for item in failures]
    repairable_reasons = [
        str(item.get("final_failure_reason") or "unknown")
        for item in repairable_failures + source_pool_gap_failures
    ]
    dominant_reason = Counter(reasons).most_common(1)[0][0] if reasons else ""
    repairable_dominant_reason = Counter(repairable_reasons).most_common(1)[0][0] if repairable_reasons else ""
    state["remaining_materials"] = list(pending)
    state["inflight_materials"] = list(inflight)
    state["failed_materials"] = [item["material_path"] for item in repairable_failures]
    state["source_pool_gap_failed_materials"] = [item["material_path"] for item in source_pool_gap_failures]
    state["non_repairable_failed_materials"] = [item["material_path"] for item in non_repairable_failures]
    state["dominant_failure_reason"] = dominant_reason
    state["repairable_dominant_failure_reason"] = repairable_dominant_reason
    state["summary"] = {
        "total": int(len(state.get("all_materials") or [])),
        "success": int(success),
        "final_fail": int(len(failures)),
        "repairable_final_fail": int(len(repairable_failures) + len(source_pool_gap_failures)),
        "source_pool_gap_final_fail": int(len(source_pool_gap_failures)),
        "non_repairable_final_fail": int(len(non_repairable_failures)),
        "remaining": int(len(pending)),
        "inflight": int(len(inflight)),
    }
    state["updated_at"] = now_iso()


def build_repair_summary(state: dict) -> dict:
    failed = [item for item in (state.get("results_by_stem") or {}).values() if item.get("final_status") == "FAIL"]
    source_pool_gap_failed = [
        item for item in failed
        if is_source_pool_gap_failure(item) or is_all_unmatched_failure(item)
    ]
    repairable_failed = [
        item for item in failed
        if not is_non_repairable_failure(item)
        and not (is_source_pool_gap_failure(item) or is_all_unmatched_failure(item))
    ]
    non_repairable_failed = [item for item in failed if is_non_repairable_failure(item)]
    failure_reasons = Counter(str(item.get("final_failure_reason") or "unknown") for item in failed)
    repairable_failure_reasons = Counter(
        str(item.get("final_failure_reason") or "unknown")
        for item in repairable_failed + source_pool_gap_failed
    )
    material_shapes = Counter(str(item.get("shape") or "unknown") for item in failed)
    summary = {
        "batch_stamp": state.get("batch_stamp"),
        "project_dir": state.get("project_dir"),
        "pause_round": int(state.get("pause_round") or 0),
        "generated_at": now_iso(),
        "failure_pause_baseline": dict(state.get("failure_pause_baseline") or {}),
        "new_final_failures_since_baseline": int(new_final_failures_since_baseline(state)),
        "failed_materials": [item.get("material_path") for item in repairable_failed],
        "repairable_failed_materials": [item.get("material_path") for item in repairable_failed],
        "source_pool_gap_failed_materials": [item.get("material_path") for item in source_pool_gap_failed],
        "non_repairable_failed_materials": [item.get("material_path") for item in non_repairable_failed],
        "all_failed_materials": [item.get("material_path") for item in failed],
        "recent_failed_materials": [item.get("material_path") for item in (repairable_failed + source_pool_gap_failed)[-3:]],
        "inflight_materials_at_pause": list(state.get("inflight_materials_at_pause") or []),
        "dominant_failure_reason": state.get("dominant_failure_reason") or "",
        "repairable_dominant_failure_reason": state.get("repairable_dominant_failure_reason") or "",
        "final_failure_reason_histogram": dict(failure_reasons),
        "repairable_failure_reason_histogram": dict(repairable_failure_reasons),
        "material_shape_histogram": dict(material_shapes),
        "failures": repairable_failed,
        "source_pool_gap_failures": source_pool_gap_failed,
        "all_unmatched_failures": [item for item in source_pool_gap_failed if is_all_unmatched_failure(item)],
        "all_failures": failed,
        "non_repairable_failures": non_repairable_failed,
        "source_pool_diagnostics": build_source_pool_diagnostics(state, failed),
        "heartbeat": {
            "interval_minutes": int(state.get("paused_repair_heartbeat_minutes") or 30),
            "instruction": "检查最近一轮 paused_for_repair 状态，若当前没有活跃批量进程则在本线程继续修复；修复后先跑回归护栏，再恢复批量。",
        },
    }
    return summary


def build_source_pool_diagnostics(state: dict, failures: List[dict]) -> dict:
    source_dir = Path(str(state.get("project_dir") or "")) / "剧集"
    source_files = []
    if source_dir.is_dir():
        source_files = sorted(
            p for p in source_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in VIDEO_EXTS
            and not p.name.startswith(".")
            and not p.name.startswith("._")
        )

    source_durations: Dict[str, float] = {}
    for source in source_files:
        source_durations[source.name] = probe_duration_sec(source)
    source_total = sum(float(v) for v in source_durations.values() if float(v) > 0.0)

    items: List[dict] = []
    for result in failures:
        if not (is_source_pool_gap_failure(result) or is_all_unmatched_failure(result)):
            continue
        reason = str(result.get("final_failure_reason") or "")
        report_path = Path(str(result.get("report_path") or ""))
        report = read_json(report_path) if report_path.is_file() else {}
        failure = report.get("failure") or {}
        details = failure.get("details") if isinstance(failure.get("details"), dict) else {}
        material_analysis = report.get("material_analysis") or {}
        missing_count = int(
            details.get("missing_segments_count")
            or failure.get("missing_segments_count")
            or result.get("missing_segments_count")
            or 0
        )
        total_segments = int(details.get("expected_total_segments") or report.get("total_segments") or 0)
        target_duration = float(report.get("target_duration") or 0.0)
        used_sources = []
        for seg in report.get("segments") or []:
            source_name = Path(str(seg.get("source_video") or seg.get("source") or "")).name
            if source_name and source_name not in used_sources:
                used_sources.append(source_name)
        if not used_sources:
            for seg in material_analysis.get("target_sequence_segments") or []:
                source_name = Path(str(seg.get("source") or "")).name
                if source_name and source_name not in used_sources:
                    used_sources.append(source_name)
        tail_meta = tail_missing_meta(report) if report else {}
        items.append(
            {
                "material": Path(str(result.get("material_path") or "")).name,
                "report_path": str(report_path),
                "target_duration_sec": round(float(target_duration), 3),
                "source_pool_total_duration_sec": round(float(source_total), 3),
                "target_vs_source_pool_ratio": (
                    round(float(target_duration) / float(source_total), 3)
                    if source_total > 1e-6 and target_duration > 0.0
                    else None
                ),
                "material_shape": str(material_analysis.get("material_shape") or result.get("shape") or "unknown"),
                "selected_strategy": str(material_analysis.get("selected_strategy") or result.get("retry_strategy") or "unknown"),
                "available_segments": int(report.get("available_segments") or 0),
                "total_segments": int(total_segments),
                "missing_segments_count": int(missing_count),
                "missing_ratio": (
                    round(float(missing_count) / float(total_segments), 3)
                    if total_segments > 0
                    else None
                ),
                "tail_missing_count": int(tail_meta.get("tail_missing_count") or result.get("tail_missing_count") or 0),
                "tail_missing_start_index": int(tail_meta.get("tail_missing_start_index") or result.get("tail_missing_start_index") or -1),
                "tail_missing_dominant": bool(tail_meta.get("tail_missing_dominant") or result.get("tail_missing_dominant") or False),
                "used_sources": used_sources,
                "diagnosis": (
                    "all_segments_unmatched_target_fallback_candidate"
                    if is_all_unmatched_failure(result)
                    else "source_pool_or_unmatched_tail_content_not_stably_available"
                ),
            }
        )

    return {
        "source_dir": str(source_dir),
        "source_count": int(len(source_files)),
        "source_pool_total_duration_sec": round(float(source_total), 3),
        "source_durations_sec": {name: round(float(value), 3) for name, value in source_durations.items()},
        "failure_count": int(len(items)),
        "items": items,
    }


def probe_duration_sec(path: Path) -> float:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        if proc.returncode != 0:
            return 0.0
        return max(0.0, float((proc.stdout or "0").strip() or 0.0))
    except Exception:
        return 0.0


def run_regression_guard(
    regression_script: Path,
    config_path: Path,
    regression_guard_config: Path,
    state_path: Path,
    output_root: Path,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(regression_script),
        "--config",
        str(config_path),
        "--samples-config",
        str(regression_guard_config),
        "--batch-state",
        str(state_path),
        "--output-root",
        str(output_root),
    ]
    return subprocess.run(cmd, check=False).returncode


def prepare_resume_queue(state: dict) -> List[str]:
    ordered: List[str] = []
    seen = set()
    skip = set(state.get("non_repairable_failed_materials") or [])
    completed = {
        str(item.get("material_path") or "")
        for item in (state.get("results_by_stem") or {}).values()
        if item.get("final_status") == "OK"
    }
    for group in (
        state.get("failed_materials") or [],
        state.get("source_pool_gap_failed_materials") or [],
        state.get("inflight_materials") or [],
        state.get("inflight_materials_at_pause") or [],
        state.get("remaining_materials") or [],
    ):
        for item in group:
            if item in completed:
                continue
            if item in skip:
                continue
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
    return ordered


def reset_results_for_rerun(state: dict, rerun_materials: List[str]) -> None:
    results = state.get("results_by_stem") or {}
    rerun_stems = {stem_for(Path(path)) for path in rerun_materials}
    for stem in rerun_stems:
        results.pop(stem, None)
    state["results_by_stem"] = results


def process_queue(
    state: dict,
    run_one: Path,
    runtime_log: Path,
    progress_log: Path,
    output_dir: Path,
    workers: int,
    auto_pause_on_failures: bool,
    fail_threshold_count: int,
    retry_enabled: bool,
    retry_on_fail: bool,
    retry_missing_threshold: int,
) -> int:
    pending: List[str] = list(state.get("remaining_materials") or [])
    state.setdefault("results_by_stem", {})
    pause_requested = False
    pause_snapshot: Optional[List[str]] = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        inflight: Dict[concurrent.futures.Future, str] = {}

        def current_inflight_materials() -> List[str]:
            return [path for path in inflight.values()]

        while pending or inflight:
            while pending and len(inflight) < max(1, workers) and not pause_requested:
                material = pending.pop(0)
                future = pool.submit(
                    run_material,
                    Path(material),
                    run_one,
                    runtime_log,
                    output_dir,
                    retry_enabled,
                    retry_on_fail,
                    retry_missing_threshold,
                )
                inflight[future] = material
                refresh_state(state, pending, current_inflight_materials())
                write_json_atomic(Path(state["state_path"]), state)

            if not inflight:
                break

            done, _ = concurrent.futures.wait(
                list(inflight.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                material = inflight.pop(future)
                result = future.result()
                state["results_by_stem"][result["stem"]] = result
                append_line(progress_log, result_to_progress_line(result))
                refresh_state(state, pending, current_inflight_materials())
                write_json_atomic(Path(state["state_path"]), state)

                if (
                    not pause_requested
                    and auto_pause_on_failures
                    and new_final_failures_since_baseline(state) >= int(fail_threshold_count)
                ):
                    pause_requested = True
                    pause_snapshot = sorted(current_inflight_materials())
                    state["pause_round"] = int(state.get("pause_round") or 0) + 1
                    state["inflight_materials_at_pause"] = list(pause_snapshot)
                    refresh_state(state, pending, current_inflight_materials())
                    write_json_atomic(Path(state["state_path"]), state)

            if pause_requested and inflight:
                continue

        if pause_requested:
            state["status"] = "paused_for_repair"
            state["paused_at"] = now_iso()
            if pause_snapshot is not None:
                state["inflight_materials_at_pause"] = list(pause_snapshot)
            refresh_state(state, pending, [])
            repair_summary = build_repair_summary(state)
            repair_summary_path = Path(state["output_dir"]) / f"batch_clip_{state['batch_stamp']}.repair_summary.round{int(state['pause_round'])}.json"
            write_json_atomic(repair_summary_path, repair_summary)
            state["repair_summary_path"] = str(repair_summary_path)
            write_json_atomic(Path(state["state_path"]), state)
            return PAUSE_EXIT_CODE

    state["status"] = "completed"
    state["completed_at"] = now_iso()
    refresh_state(state, [], [])
    write_json_atomic(Path(state["state_path"]), state)
    return 0


def create_initial_state(
    project_dir: Path,
    material_dir: Path,
    output_dir: Path,
    stamp: str,
    config_path: Path,
    workers: int,
    fail_threshold_count: int,
    auto_pause_on_failures: bool,
    retry_enabled: bool,
    retry_on_fail: bool,
    retry_missing_threshold: int,
    heartbeat_minutes: int,
    all_materials: List[str],
) -> dict:
    state_path = output_dir / f"batch_clip_{stamp}.state.json"
    state = {
        "version": STATE_VERSION,
        "batch_stamp": stamp,
        "project_dir": str(project_dir),
        "material_dir": str(material_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "state_path": str(state_path),
        "status": "running",
        "workers": int(workers),
        "fail_threshold_count": int(fail_threshold_count),
        "auto_pause_on_failures": bool(auto_pause_on_failures),
        "retry_enabled": bool(retry_enabled),
        "retry_on_fail": bool(retry_on_fail),
        "retry_missing_threshold": int(retry_missing_threshold),
        "paused_repair_heartbeat_minutes": int(heartbeat_minutes),
        "pause_round": 0,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "paused_at": "",
        "resumed_at": "",
        "completed_at": "",
        "all_materials": list(all_materials),
        "remaining_materials": list(all_materials),
        "failed_materials": [],
        "source_pool_gap_failed_materials": [],
        "non_repairable_failed_materials": [],
        "inflight_materials": [],
        "inflight_materials_at_pause": [],
        "results_by_stem": {},
        "dominant_failure_reason": "",
        "repairable_dominant_failure_reason": "",
        "repair_summary_path": "",
        "failure_pause_baseline": {
            "final_fail": 0,
            "repairable_final_fail": 0,
            "source_pool_gap_final_fail": 0,
            "non_repairable_final_fail": 0,
            "set_at": now_iso(),
        },
        "summary": {
            "total": int(len(all_materials)),
            "success": 0,
            "final_fail": 0,
            "repairable_final_fail": 0,
            "source_pool_gap_final_fail": 0,
            "non_repairable_final_fail": 0,
            "remaining": int(len(all_materials)),
            "inflight": 0,
        },
    }
    return state


def print_batch_header(state: dict, todo_list: Path, runtime_log: Path, progress_log: Path) -> None:
    print(f"[batch] project={state['project_dir']}")
    print(f"[batch] material_dir={state['material_dir']}")
    print(f"[batch] output_dir={state['output_dir']}")
    print(f"[batch] workers={state['workers']}")
    print(f"[batch] todo={len(state.get('remaining_materials') or [])}")
    print(f"[batch] todo_list={todo_list}")
    print(f"[batch] runtime_log={runtime_log}")
    print(f"[batch] progress_log={progress_log}")
    print(f"[batch] state_file={state['state_path']}")
    print(f"[batch] fail_threshold_count={state['fail_threshold_count']}")
    print(f"[batch] auto_pause_on_failures={state['auto_pause_on_failures']}")
    print(f"[batch] stitched_retry_enabled={state['retry_enabled']}")
    print(f"[batch] stitched_retry_on_fail={state['retry_on_fail']}")
    print(f"[batch] retry_missing_threshold={state['retry_missing_threshold']}")


def print_summary(state: dict, progress_log: Path, runtime_log: Path) -> None:
    summary = state.get("summary") or {}
    print(
        "[summary] "
        f"total={summary.get('total', 0)} "
        f"ok={summary.get('success', 0)} "
        f"fail={summary.get('final_fail', 0)} "
        f"repairable_fail={summary.get('repairable_final_fail', summary.get('final_fail', 0))} "
        f"source_pool_gap_fail={summary.get('source_pool_gap_final_fail', 0)} "
        f"non_repairable_fail={summary.get('non_repairable_final_fail', 0)} "
        f"remaining={summary.get('remaining', 0)} "
        f"inflight={summary.get('inflight', 0)}"
    )
    print(f"[summary] runtime_log={runtime_log}")
    print(f"[summary] progress_log={progress_log}")
    if state.get("repair_summary_path"):
        print(f"[summary] repair_summary={state['repair_summary_path']}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可暂停、可恢复的批量二次裁剪控制器")
    parser.add_argument("project_dir", nargs="?", help="项目目录")
    parser.add_argument("workers", nargs="?", type=int, default=None, help="并发数")
    parser.add_argument("--resume-from-state", dest="resume_from_state", default="", help="从状态文件恢复批次")
    parser.add_argument("--workers", dest="workers_opt", type=int, default=None, help="并发数（恢复批次时推荐使用）")
    parser.add_argument("--config", dest="config", default="", help="配置文件路径")
    parser.add_argument("--fail-threshold-count", dest="fail_threshold_count", type=int, default=None, help="最终失败数达到该阈值时暂停")
    parser.add_argument("--disable-auto-pause-on-failures", action="store_true", help="禁用失败阈值自动暂停")
    parser.add_argument("--disable-auto-archive-existing-output", action="store_true", help="禁用新批次启动前自动归档旧 output")
    parser.add_argument("--run-one-script", dest="run_one_script", default="", help="覆盖单条裁剪脚本路径（测试用）")
    parser.add_argument("--regression-guard-script", dest="regression_guard_script", default="", help="覆盖回归护栏脚本路径（测试用）")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    config_path = Path(args.config).resolve() if args.config else root_dir / "configurations" / "ai_pipeline.defaults.json"
    cfg = load_cfg(root_dir, config_path)
    batch_cfg = cfg.get("batch_project_clip") or {}
    v6_cfg = cfg.get("v6_fast") or {}

    run_one = Path(args.run_one_script).resolve() if args.run_one_script else script_dir / "run_clip_one.sh"
    regression_guard_script = Path(args.regression_guard_script).resolve() if args.regression_guard_script else script_dir / "run_regression_guard.py"
    resume_positional_workers: Optional[int] = None
    if args.resume_from_state and args.project_dir and str(args.project_dir).isdigit():
        resume_positional_workers = int(str(args.project_dir))
    workers = int(args.workers_opt or args.workers or resume_positional_workers or 2)
    fail_threshold_count = int(args.fail_threshold_count or batch_cfg.get("final_fail_threshold_count") or 3)
    auto_pause_on_failures = not args.disable_auto_pause_on_failures and bool_cfg(batch_cfg.get("auto_pause_on_failures"), True)
    retry_enabled = bool_cfg(v6_cfg.get("stitched_material_retry_enabled"), True)
    retry_on_fail = bool_cfg(v6_cfg.get("stitched_material_retry_on_fail"), True)
    retry_missing_threshold = int(v6_cfg.get("retry_on_missing_segments_threshold") or 2)
    regression_guard_enabled = bool_cfg(batch_cfg.get("regression_guard_enabled"), True)
    regression_guard_config = root_dir / str(batch_cfg.get("regression_guard_config") or "configurations/regression_guard.samples.json")
    heartbeat_minutes = int(batch_cfg.get("paused_repair_heartbeat_minutes") or 30)
    auto_archive_existing_output = (
        not args.disable_auto_archive_existing_output
        and bool_cfg(batch_cfg.get("archive_existing_output_on_fresh_start"), True)
    )

    if args.resume_from_state:
        state_path = Path(args.resume_from_state).resolve()
        if not state_path.is_file():
            raise SystemExit(f"state file not found: {state_path}")
        state = read_json(state_path)
        previous_status = str(state.get("status") or "")
        project_dir = Path(state["project_dir"])
        output_dir = Path(state["output_dir"])
        stamp = str(state["batch_stamp"])
        runtime_log = output_dir / f"batch_clip_{stamp}.runtime.log"
        progress_log = output_dir / f"batch_clip_{stamp}.progress.log"
        state["status"] = "repairing"
        state["resumed_at"] = now_iso()
        normalize_existing_failure_results(state)
        refresh_state(
            state,
            list(state.get("remaining_materials") or []),
            list(state.get("inflight_materials") or []),
        )
        write_json_atomic(state_path, state)

        if regression_guard_enabled:
            regression_output = root_dir / "runtime" / "temp_outputs" / "regression_guard" / f"batch_{stamp}" / f"round_{int(state.get('pause_round') or 0)}"
            state["status"] = "regression_check"
            write_json_atomic(state_path, state)
            rc = run_regression_guard(regression_guard_script, config_path, regression_guard_config, state_path, regression_output)
            if rc != 0:
                state["status"] = "paused_for_repair"
                state["updated_at"] = now_iso()
                write_json_atomic(state_path, state)
                return REGRESSION_BLOCK_EXIT_CODE

        rerun_materials = prepare_resume_queue(state)
        reset_results_for_rerun(state, rerun_materials)
        state["remaining_materials"] = list(rerun_materials)
        state["failed_materials"] = []
        state["source_pool_gap_failed_materials"] = []
        state["inflight_materials"] = []
        state["status"] = "resuming"
        refresh_state(state, list(rerun_materials), [])
        if previous_status in {"paused_for_repair", "repairing", "regression_check"}:
            set_failure_pause_baseline(state)
        else:
            state.setdefault("failure_pause_baseline", {
                "final_fail": 0,
                "repairable_final_fail": 0,
                "source_pool_gap_final_fail": 0,
                "non_repairable_final_fail": 0,
                "set_at": now_iso(),
            })
        write_json_atomic(state_path, state)

        rc = process_queue(
            state,
            run_one,
            runtime_log,
            progress_log,
            output_dir,
            workers,
            auto_pause_on_failures,
            fail_threshold_count,
            retry_enabled,
            retry_on_fail,
            retry_missing_threshold,
        )
        print_summary(state, progress_log, runtime_log)
        return rc

    if not args.project_dir:
        raise SystemExit("project_dir is required unless --resume-from-state is used")

    project_dir = Path(args.project_dir).resolve()
    if not project_dir.is_dir():
        raise SystemExit(f"project dir not found: {project_dir}")
    material_dir, output_dir = resolve_project_layout(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_output_dir: Optional[Path] = None
    if auto_archive_existing_output:
        archived_output_dir = archive_existing_output_dir(output_dir, stamp)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    materials = list_materials(material_dir, output_dir)
    if not materials:
        print(f"[batch] project={project_dir}")
        print(f"[batch] material_dir={material_dir}")
        print(f"[batch] output_dir={output_dir}")
        print("[batch] nothing to do")
        return 0

    todo_list = output_dir / f"batch_clip_{stamp}.todo.txt"
    runtime_log = output_dir / f"batch_clip_{stamp}.runtime.log"
    progress_log = output_dir / f"batch_clip_{stamp}.progress.log"
    todo_list.write_text("\n".join(str(p) for p in materials) + "\n", encoding="utf-8")

    state = create_initial_state(
        project_dir,
        material_dir,
        output_dir,
        stamp,
        config_path,
        workers,
        fail_threshold_count,
        auto_pause_on_failures,
        retry_enabled,
        retry_on_fail,
        retry_missing_threshold,
        heartbeat_minutes,
        [str(p) for p in materials],
    )
    state["archived_output_dir"] = str(archived_output_dir) if archived_output_dir else ""
    write_json_atomic(Path(state["state_path"]), state)
    print_batch_header(state, todo_list, runtime_log, progress_log)
    if archived_output_dir is not None:
        print(f"[batch] archived_existing_output={archived_output_dir}")

    rc = process_queue(
        state,
        run_one,
        runtime_log,
        progress_log,
        output_dir,
        workers,
        auto_pause_on_failures,
        fail_threshold_count,
        retry_enabled,
        retry_on_fail,
        retry_missing_threshold,
    )
    print_summary(state, progress_log, runtime_log)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
