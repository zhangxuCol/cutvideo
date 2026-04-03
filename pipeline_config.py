#!/usr/bin/env python3
"""
Pipeline config helpers.

Design:
- JSON config file provides stable defaults for automation.
- CLI arguments always override config values.
- Config path resolution order:
  1) explicit --config
  2) CUTVIDEO_CONFIG env var
  3) default file candidates under repo
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CONFIG_RELATIVE_PATHS = (
    "06_configurations/ai_pipeline.defaults.json",
    "06_configurations/ai_pipeline.local.json",
)


def _to_abs_path(repo_root: Path, raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def resolve_config_path(
    repo_root: Path,
    explicit_path: str = "",
    env_var: str = "CUTVIDEO_CONFIG",
) -> Optional[Path]:
    explicit = (explicit_path or "").strip()
    env_path = os.getenv(env_var, "").strip()

    if explicit:
        p = _to_abs_path(repo_root, explicit)
        if not p.exists():
            raise RuntimeError(f"--config 指定的配置文件不存在: {p}")
        return p

    if env_path:
        p = _to_abs_path(repo_root, env_path)
        if not p.exists():
            raise RuntimeError(f"{env_var} 指定的配置文件不存在: {p}")
        return p

    for rel in DEFAULT_CONFIG_RELATIVE_PATHS:
        p = (repo_root / rel).resolve()
        if p.exists():
            return p
    return None


def load_section_config(
    repo_root: Path,
    section: str,
    explicit_path: str = "",
    env_var: str = "CUTVIDEO_CONFIG",
) -> Tuple[Dict[str, Any], Optional[Path]]:
    cfg_path = resolve_config_path(repo_root, explicit_path=explicit_path, env_var=env_var)
    if cfg_path is None:
        return {}, None

    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"读取配置文件失败: {cfg_path} ({exc})") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"配置文件根节点必须是 object: {cfg_path}")

    section_data = raw.get(section, {})
    if section_data is None:
        section_data = {}
    if not isinstance(section_data, dict):
        raise RuntimeError(f"配置节 {section} 必须是 object: {cfg_path}")

    return section_data, cfg_path


def cfg_str(cfg: Dict[str, Any], key: str, default: str) -> str:
    v = cfg.get(key, default)
    if v is None:
        return default
    return str(v)


def cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key, default)
    try:
        return int(v)
    except Exception as exc:
        raise RuntimeError(f"配置项 {key} 需要 int，当前值: {v!r}") from exc


def cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    try:
        return float(v)
    except Exception as exc:
        raise RuntimeError(f"配置项 {key} 需要 float，当前值: {v!r}") from exc


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(value)


def cfg_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    v = cfg.get(key, default)
    try:
        return _parse_bool(v)
    except Exception as exc:
        raise RuntimeError(f"配置项 {key} 需要 bool，当前值: {v!r}") from exc


def cfg_str_list(cfg: Dict[str, Any], key: str, default: List[str]) -> List[str]:
    v = cfg.get(key, default)
    if v is None:
        return list(default)
    if isinstance(v, list):
        out = [str(x).strip() for x in v if str(x).strip()]
        return out
    if isinstance(v, str):
        out = [x.strip() for x in v.split(",") if x.strip()]
        return out
    raise RuntimeError(f"配置项 {key} 需要 list[str] 或逗号分隔字符串，当前值: {v!r}")


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]
