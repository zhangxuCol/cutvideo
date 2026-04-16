#!/usr/bin/env python3
"""Auto-iterate: gate -> auto-modify -> gate until pass or stop rule."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _tail_lines(text: str, limit: int = 80) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[-limit:])


def _run_cmd(command: str, timeout_sec: int | None = None) -> dict[str, Any]:
    started = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or f"Timed out after {timeout_sec}s"

    return {
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_sec": round(time.time() - started, 3),
        "stdout_tail": _tail_lines(stdout),
        "stderr_tail": _tail_lines(stderr),
    }


def _render_command(template: str, values: dict[str, str]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"unknown placeholder in auto_modify.command: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_goal_gate(goal_gate_py: Path, spec_path: Path, report_path: Path) -> dict[str, Any]:
    cmd = (
        f"python3 {shlex.quote(str(goal_gate_py))} "
        f"--spec {shlex.quote(str(spec_path))} "
        f"--report {shlex.quote(str(report_path))}"
    )
    result = _run_cmd(cmd)
    gate_report = _read_json(report_path) if report_path.exists() else {}
    result["report_path"] = str(report_path)
    result["passed_all"] = bool(gate_report.get("passed_all"))
    result["first_failed_hard_gate"] = gate_report.get("first_failed_hard_gate")
    result["summary"] = gate_report.get("summary")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto iterate until hard gates pass or max iterations reached."
    )
    parser.add_argument("--spec", required=True, help="Path to goal spec JSON")
    parser.add_argument(
        "--out-dir",
        default="runtime/temp_outputs/goal_self_improve",
        help="Directory for iterative reports",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec).expanduser().resolve()
    if not spec_path.exists():
        print(f"[auto_improve] spec not found: {spec_path}", file=sys.stderr)
        return 2

    try:
        spec = _read_json(spec_path)
    except Exception as exc:
        print(f"[auto_improve] invalid JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(spec, dict):
        print("[auto_improve] spec root must be an object", file=sys.stderr)
        return 2

    goal = spec.get("goal")
    checks = spec.get("checks")
    if not isinstance(goal, str) or not goal.strip():
        print("[auto_improve] goal must be a non-empty string", file=sys.stderr)
        return 2
    if not isinstance(checks, list) or not checks:
        print("[auto_improve] checks must be a non-empty list", file=sys.stderr)
        return 2

    max_iterations = spec.get("max_iterations", 10)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        print("[auto_improve] max_iterations must be a positive integer", file=sys.stderr)
        return 2

    auto_modify = spec.get("auto_modify")
    if not isinstance(auto_modify, dict):
        print(
            "[auto_improve] missing auto_modify object (needs auto_modify.command)",
            file=sys.stderr,
        )
        return 3

    command_template = auto_modify.get("command")
    if not isinstance(command_template, str) or not command_template.strip():
        print(
            "[auto_improve] auto_modify.command must be a non-empty string",
            file=sys.stderr,
        )
        return 3

    modify_timeout = auto_modify.get("timeout_sec", 900)
    if not isinstance(modify_timeout, int) or modify_timeout <= 0:
        print("[auto_improve] auto_modify.timeout_sec must be positive int", file=sys.stderr)
        return 2

    stop_on_failure = auto_modify.get("stop_on_failure", False)
    if not isinstance(stop_on_failure, bool):
        print("[auto_improve] auto_modify.stop_on_failure must be true/false", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir).expanduser().resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    goal_gate_py = (Path(__file__).resolve().parent / "goal_gate.py").resolve()
    if not goal_gate_py.exists():
        print(f"[auto_improve] goal_gate.py missing: {goal_gate_py}", file=sys.stderr)
        return 2

    history: list[dict[str, Any]] = []
    iteration = 0
    last_gate_report_path = run_dir / "iteration_000_gate.json"
    gate_result = _run_goal_gate(goal_gate_py, spec_path, last_gate_report_path)
    history.append({"iteration": iteration, "phase": "gate", **gate_result})

    if gate_result["passed_all"]:
        summary = {
            "goal": goal,
            "status": "PASS",
            "max_iterations": max_iterations,
            "iterations_used": 0,
            "last_gate_report": str(last_gate_report_path),
            "history": history,
        }
        out_path = run_dir / "summary.json"
        out_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[auto_improve] PASS with baseline, summary={out_path}")
        return 0

    while iteration < max_iterations:
        iteration += 1
        gate_report_path = run_dir / f"iteration_{iteration:03d}_gate.json"

        vars_map = {
            "iteration": str(iteration),
            "spec": str(spec_path),
            "spec_q": shlex.quote(str(spec_path)),
            "last_gate_report": str(last_gate_report_path),
            "last_gate_report_q": shlex.quote(str(last_gate_report_path)),
            "gate_report": str(gate_report_path),
            "gate_report_q": shlex.quote(str(gate_report_path)),
            "run_dir": str(run_dir),
            "run_dir_q": shlex.quote(str(run_dir)),
            "workspace": str(Path.cwd().resolve()),
            "workspace_q": shlex.quote(str(Path.cwd().resolve())),
        }

        try:
            modify_cmd = _render_command(command_template, vars_map)
        except Exception as exc:
            print(f"[auto_improve] bad auto_modify.command: {exc}", file=sys.stderr)
            return 2

        modify_result = _run_cmd(modify_cmd, timeout_sec=modify_timeout)
        history.append({"iteration": iteration, "phase": "modify", **modify_result})

        if modify_result["exit_code"] != 0 and stop_on_failure:
            summary = {
                "goal": goal,
                "status": "STOPPED",
                "reason": "auto_modify_failed",
                "max_iterations": max_iterations,
                "iterations_used": iteration,
                "last_gate_report": str(last_gate_report_path),
                "history": history,
            }
            out_path = run_dir / "summary.json"
            out_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[auto_improve] STOPPED on modify failure, summary={out_path}")
            return 1

        gate_result = _run_goal_gate(goal_gate_py, spec_path, gate_report_path)
        history.append({"iteration": iteration, "phase": "gate", **gate_result})
        last_gate_report_path = gate_report_path

        if gate_result["passed_all"]:
            summary = {
                "goal": goal,
                "status": "PASS",
                "max_iterations": max_iterations,
                "iterations_used": iteration,
                "last_gate_report": str(last_gate_report_path),
                "history": history,
            }
            out_path = run_dir / "summary.json"
            out_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[auto_improve] PASS at iteration={iteration}, summary={out_path}")
            return 0

    summary = {
        "goal": goal,
        "status": "STOPPED",
        "reason": "max_iterations_reached",
        "max_iterations": max_iterations,
        "iterations_used": max_iterations,
        "last_gate_report": str(last_gate_report_path),
        "history": history,
    }
    out_path = run_dir / "summary.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[auto_improve] STOPPED max_iterations reached, summary={out_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
