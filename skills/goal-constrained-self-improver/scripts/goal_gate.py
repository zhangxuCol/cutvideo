#!/usr/bin/env python3
"""Run acceptance checks from a goal spec and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Check:
    name: str
    command: str
    timeout_sec: int
    expect_exit: int
    hard_gate: bool


def _ensure_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _ensure_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _tail_lines(text: str, limit: int = 80) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[-limit:])


def _load_checks(raw_checks: list[Any], default_timeout: int) -> list[Check]:
    checks: list[Check] = []
    for idx, raw in enumerate(raw_checks):
        if isinstance(raw, str):
            checks.append(
                Check(
                    name=f"check_{idx + 1}",
                    command=raw,
                    timeout_sec=default_timeout,
                    expect_exit=0,
                    hard_gate=True,
                )
            )
            continue

        item = _ensure_dict(raw, f"checks[{idx}]")
        cmd = item.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError(f"checks[{idx}].command must be a non-empty string")

        name = item.get("name", f"check_{idx + 1}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"checks[{idx}].name must be a non-empty string")

        timeout_sec = item.get("timeout_sec", default_timeout)
        if not isinstance(timeout_sec, int) or timeout_sec <= 0:
            raise ValueError(f"checks[{idx}].timeout_sec must be a positive integer")

        expect_exit = item.get("expect_exit", 0)
        if not isinstance(expect_exit, int):
            raise ValueError(f"checks[{idx}].expect_exit must be an integer")

        hard_gate = item.get("hard_gate", True)
        if not isinstance(hard_gate, bool):
            raise ValueError(f"checks[{idx}].hard_gate must be true/false")

        checks.append(
            Check(
                name=name.strip(),
                command=cmd.strip(),
                timeout_sec=timeout_sec,
                expect_exit=expect_exit,
                hard_gate=hard_gate,
            )
        )
    return checks


def load_spec(path: Path) -> tuple[dict[str, Any], list[Check], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    spec = _ensure_dict(data, "spec")

    goal = spec.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a non-empty string")

    constraints = spec.get("constraints", [])
    if not isinstance(constraints, list) or not all(
        isinstance(x, str) and x.strip() for x in constraints
    ):
        raise ValueError("constraints must be an array of non-empty strings")

    default_timeout = spec.get("default_timeout_sec", 600)
    if not isinstance(default_timeout, int) or default_timeout <= 0:
        raise ValueError("default_timeout_sec must be a positive integer")

    max_iterations = spec.get("max_iterations")
    if max_iterations is not None and (
        not isinstance(max_iterations, int) or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer when provided")

    raw_checks = _ensure_list(spec.get("checks"), "checks")
    if not raw_checks:
        raise ValueError("checks must contain at least one command")

    checks = _load_checks(raw_checks, default_timeout)
    return spec, checks, default_timeout


def run_check(check: Check) -> dict[str, Any]:
    started = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            check.command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=check.timeout_sec,
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or f"Timed out after {check.timeout_sec}s"

    duration = round(time.time() - started, 3)
    passed = (exit_code == check.expect_exit) and (not timed_out)
    return {
        "name": check.name,
        "command": check.command,
        "hard_gate": check.hard_gate,
        "timeout_sec": check.timeout_sec,
        "expect_exit": check.expect_exit,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "passed": passed,
        "duration_sec": duration,
        "stdout_tail": _tail_lines(stdout),
        "stderr_tail": _tail_lines(stderr),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run acceptance checks from a goal spec and emit a JSON report."
    )
    parser.add_argument("--spec", required=True, help="Path to spec JSON file")
    parser.add_argument(
        "--report",
        default="runtime/temp_outputs/goal_gate_report.json",
        help="Output report JSON path",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        spec, checks, _ = load_spec(spec_path)
    except Exception as exc:
        print(f"[goal_gate] invalid spec: {exc}", file=sys.stderr)
        return 2

    started = time.time()
    results = [run_check(check) for check in checks]

    hard_failures = [r for r in results if r["hard_gate"] and not r["passed"]]
    soft_failures = [r for r in results if (not r["hard_gate"]) and not r["passed"]]
    passed_all = len(hard_failures) == 0

    report = {
        "goal": spec["goal"],
        "constraints": spec.get("constraints", []),
        "max_iterations": spec.get("max_iterations"),
        "passed_all": passed_all,
        "summary": {
            "total_checks": len(results),
            "hard_failures": len(hard_failures),
            "soft_failures": len(soft_failures),
            "duration_sec": round(time.time() - started, 3),
        },
        "first_failed_hard_gate": hard_failures[0]["name"] if hard_failures else None,
        "results": results,
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[goal_gate] report: {report_path}")
    if passed_all:
        print("[goal_gate] PASS: all hard gates satisfied")
        if soft_failures:
            print(f"[goal_gate] NOTE: {len(soft_failures)} soft checks failed")
        return 0

    print(
        "[goal_gate] FAIL: "
        f"{len(hard_failures)} hard gate(s) failed; "
        f"first={hard_failures[0]['name']}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
