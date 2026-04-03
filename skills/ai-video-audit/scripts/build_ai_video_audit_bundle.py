#!/usr/bin/env python3
"""
构建 AI 审片证据包：画面抽帧 + 音频切片 + OCR 字幕 + 可选 ASR 听写
"""

import argparse
import html
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_config import (
    load_section_config,
    cfg_str,
    cfg_int,
    cfg_float,
)


@dataclass
class Cue:
    start: float
    end: float
    text: str


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def ensure_bin(name: str) -> None:
    result = run_cmd(["which", name])
    if result.returncode != 0:
        raise RuntimeError(f"缺少命令: {name}")


def get_duration(video: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe失败: {result.stderr}")
    return float((result.stdout or "0").strip())


def extract_frame(video: Path, t: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{t:.3f}", "-i", str(video),
        "-vframes", "1", "-vf", "scale=960:-2",
        str(out_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"抽帧失败({video}@{t:.3f}s): {result.stderr}")


def extract_audio_clip(video: Path, start: float, duration: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(video), "-vn",
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(out_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"切音频失败({video}@{start:.3f}s): {result.stderr}")


def extract_full_audio_wav(video: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vn",
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(out_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"抽取整段音频失败({video}): {result.stderr}")


def normalize_text(text: str) -> str:
    text = re.sub(r"\{\\.*?\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\\N", " ").replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_srt(path: Path) -> List[Cue]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", txt)
    cues: List[Cue] = []
    for blk in blocks:
        lines = [ln.strip("\ufeff") for ln in blk.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        if re.match(r"^\d+$", lines[0]):
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        left, right = [x.strip() for x in lines[0].split("-->", 1)]
        right = right.split(" ")[0]
        try:
            h1, m1, s1 = left.split(":")
            sec1, ms1 = s1.split(",")
            start = int(h1) * 3600 + int(m1) * 60 + int(sec1) + int(ms1) / 1000.0

            h2, m2, s2 = right.split(":")
            sec2, ms2 = s2.split(",")
            end = int(h2) * 3600 + int(m2) * 60 + int(sec2) + int(ms2) / 1000.0
        except Exception:
            continue
        text = normalize_text(" ".join(lines[1:]))
        if text:
            cues.append(Cue(start=start, end=end, text=text))
    return cues


def parse_vtt_ts(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s_ms = parts
    else:
        h = "0"
        m, s_ms = parts
    s, ms = s_ms.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: Path) -> List[Cue]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cues: List[Cue] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper().startswith("WEBVTT") or line.startswith("NOTE"):
            i += 1
            continue
        if "-->" in line:
            ts_line = line
        elif i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1
            ts_line = lines[i]
        else:
            i += 1
            continue

        left, right = [x.strip() for x in ts_line.split("-->", 1)]
        right = right.split(" ")[0]
        try:
            start = parse_vtt_ts(left)
            end = parse_vtt_ts(right)
        except Exception:
            i += 1
            continue

        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = normalize_text(" ".join(text_lines))
        if text:
            cues.append(Cue(start=start, end=end, text=text))
        i += 1
    return cues


def parse_ass(path: Path) -> List[Cue]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cues: List[Cue] = []
    in_events = False
    for line in lines:
        ln = line.strip()
        if ln.startswith("[Events]"):
            in_events = True
            continue
        if not in_events or not ln.startswith("Dialogue:"):
            continue
        payload = ln.split(":", 1)[1].strip()
        parts = payload.split(",", 9)
        if len(parts) < 10:
            continue
        start_s, end_s, text = parts[1], parts[2], parts[9]
        try:
            h1, m1, s1 = start_s.split(":")
            sec1, cs1 = s1.split(".")
            start = int(h1) * 3600 + int(m1) * 60 + int(sec1) + int(cs1) / 100.0
            h2, m2, s2 = end_s.split(":")
            sec2, cs2 = s2.split(".")
            end = int(h2) * 3600 + int(m2) * 60 + int(sec2) + int(cs2) / 100.0
        except Exception:
            continue
        norm = normalize_text(text)
        if norm:
            cues.append(Cue(start=start, end=end, text=norm))
    return cues


def parse_whisper_json(path: Path) -> List[Cue]:
    raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    segs = raw.get("segments", [])
    cues: List[Cue] = []
    for seg in segs:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = normalize_text(seg.get("text", ""))
            if text:
                cues.append(Cue(start=start, end=end, text=text))
        except Exception:
            continue
    return cues


def load_cues(path: Optional[Path]) -> List[Cue]:
    if path is None or not path.exists():
        return []
    ext = path.suffix.lower()
    if ext == ".srt":
        return parse_srt(path)
    if ext == ".vtt":
        return parse_vtt(path)
    if ext in {".ass", ".ssa"}:
        return parse_ass(path)
    return []


def cue_text_at(cues: List[Cue], t: float) -> str:
    hits = [c.text for c in cues if (c.start - 0.3) <= t <= (c.end + 0.3)]
    return " | ".join(hits[:2]) if hits else ""


def cue_text_window(cues: List[Cue], start: float, end: float) -> str:
    if end < start:
        end = start
    hits = [c.text for c in cues if not (c.end < (start - 0.2) or c.start > (end + 0.2))]
    dedup: List[str] = []
    for t in hits:
        if t and (not dedup or dedup[-1] != t):
            dedup.append(t)
    return " ".join(dedup[:4]).strip()


def ocr_bottom_subtitle(frame_path: Path, ocr_lang: str) -> str:
    img = cv2.imread(str(frame_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    crop = img[int(h * 0.62):h, 0:w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tmp = frame_path.with_suffix(".ocr.png")
    cv2.imwrite(str(tmp), bw)

    cmd = [
        "tesseract", str(tmp), "stdout",
        "-l", ocr_lang,
        "--psm", "6",
    ]
    result = run_cmd(cmd)
    tmp.unlink(missing_ok=True)
    if result.returncode != 0:
        return ""
    return normalize_text(result.stdout or "")


def detect_asr_backend() -> str:
    try:
        import faster_whisper  # noqa: F401
        return "faster_whisper"
    except Exception:
        pass
    try:
        import whisper  # noqa: F401
        return "whisper"
    except Exception:
        pass
    return "none"


def detect_whisper_cli_cmd() -> Optional[List[str]]:
    result = run_cmd(["which", "whisper"])
    cmd = (result.stdout or "").strip()
    if result.returncode == 0 and cmd:
        return [cmd]
    return None


def validate_cmd_help(cmd: List[str]) -> bool:
    result = run_cmd(cmd + ["--help"])
    return result.returncode == 0


def select_asr_backend(asr_mode: str, asr_cmd: str, asr_python: str) -> Tuple[str, Optional[List[str]], Optional[str]]:
    """
    返回 (backend, cmd_base, error)
    backend:
    - none
    - faster_whisper
    - whisper
    - whisper_cli
    - whisper_py_cli
    """
    if asr_mode == "none":
        return "none", None, None

    if asr_cmd:
        cmd_base = shlex.split(asr_cmd)
        if cmd_base and validate_cmd_help(cmd_base):
            return "whisper_cli", cmd_base, None
        return "none", None, f"asr_cmd_not_working:{asr_cmd}"

    if asr_python:
        if validate_cmd_help([asr_python, "-m", "whisper"]):
            return "whisper_py_cli", [asr_python, "-m", "whisper"], None
        return "none", None, f"asr_python_not_working:{asr_python}"

    if asr_mode == "auto":
        cli = detect_whisper_cli_cmd()
        if cli and validate_cmd_help(cli):
            return "whisper_cli", cli, None
        module_backend = detect_asr_backend()
        if module_backend in {"faster_whisper", "whisper"}:
            return module_backend, None, None
        return "none", None, "no_asr_backend_detected"

    if asr_mode in {"faster_whisper", "whisper"}:
        module_backend = detect_asr_backend()
        if module_backend == asr_mode:
            return module_backend, None, None
        return "none", None, f"requested_backend_unavailable:{asr_mode}"

    return "none", None, "unknown_asr_mode"


def transcribe_with_asr(audio_path: Path, backend: str, model: str, language: str) -> str:
    if backend == "faster_whisper":
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel(model, device="cpu", compute_type="int8")
        segments, _ = whisper_model.transcribe(str(audio_path), language=language)
        return normalize_text(" ".join(seg.text for seg in segments))

    if backend == "whisper":
        import whisper

        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(str(audio_path), language=language, fp16=False)
        return normalize_text(result.get("text", ""))

    return ""


def transcribe_with_whisper_cli(audio_path: Path, cmd_base: List[str], model: str, language: str, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = cmd_base + [
        str(audio_path),
        "--model", model,
        "--language", language,
        "--task", "transcribe",
        "--output_format", "txt",
        "--output_dir", str(out_dir),
        "--fp16", "False",
        "--verbose", "False",
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "whisper_cli_failed").strip())

    txt_path = out_dir / f"{audio_path.stem}.txt"
    if not txt_path.exists():
        candidates = sorted(out_dir.glob("*.txt"))
        if not candidates:
            raise RuntimeError("whisper_cli_no_output_txt")
        txt_path = candidates[-1]
    return normalize_text(txt_path.read_text(encoding="utf-8", errors="ignore"))


_FW_MODEL_CACHE: Dict[str, object] = {}
_WHISPER_MODEL_CACHE: Dict[str, object] = {}


def transcribe_video_to_cues(
    video_path: Path,
    backend: str,
    model: str,
    language: str,
    asr_dir: Path,
    label: str,
    cmd_base: Optional[List[str]],
) -> Tuple[List[Cue], Optional[str]]:
    if backend == "none":
        return [], None

    audio_path = asr_dir / f"{label}_full.wav"
    extract_full_audio_wav(video_path, audio_path)

    if backend in {"whisper_cli", "whisper_py_cli"}:
        cmd = (cmd_base or []) + [
            str(audio_path),
            "--model", model,
            "--language", language,
            "--task", "transcribe",
            "--output_format", "json",
            "--output_dir", str(asr_dir),
            "--fp16", "False",
            "--verbose", "False",
        ]
        result = run_cmd(cmd)
        if result.returncode != 0:
            return [], (result.stderr or result.stdout or "whisper_cli_failed").strip()

        json_path = asr_dir / f"{audio_path.stem}.json"
        if not json_path.exists():
            candidates = sorted(asr_dir.glob("*.json"))
            if not candidates:
                return [], "whisper_cli_no_output_json"
            json_path = candidates[-1]

        try:
            cues = parse_whisper_json(json_path)
            transcript = normalize_text(" ".join(c.text for c in cues))
            (asr_dir / f"{label}_full_transcript.txt").write_text(transcript, encoding="utf-8")
            return cues, None
        except Exception as exc:
            return [], f"whisper_json_parse_error:{exc}"

    if backend == "faster_whisper":
        from faster_whisper import WhisperModel

        if model not in _FW_MODEL_CACHE:
            _FW_MODEL_CACHE[model] = WhisperModel(model, device="cpu", compute_type="int8")
        whisper_model = _FW_MODEL_CACHE[model]

        segments, _ = whisper_model.transcribe(str(audio_path), language=language)
        cues: List[Cue] = []
        for seg in segments:
            text = normalize_text(getattr(seg, "text", "") or "")
            if not text:
                continue
            cues.append(Cue(start=float(seg.start), end=float(seg.end), text=text))
        transcript = normalize_text(" ".join(c.text for c in cues))
        (asr_dir / f"{label}_full_transcript.txt").write_text(transcript, encoding="utf-8")
        return cues, None

    if backend == "whisper":
        import whisper

        if model not in _WHISPER_MODEL_CACHE:
            _WHISPER_MODEL_CACHE[model] = whisper.load_model(model)
        whisper_model = _WHISPER_MODEL_CACHE[model]

        result = whisper_model.transcribe(str(audio_path), language=language, fp16=False, verbose=False)
        segs = result.get("segments", [])
        cues: List[Cue] = []
        for seg in segs:
            text = normalize_text(seg.get("text", ""))
            if not text:
                continue
            cues.append(Cue(start=float(seg.get("start", 0.0)), end=float(seg.get("end", 0.0)), text=text))
        transcript = normalize_text(" ".join(c.text for c in cues))
        (asr_dir / f"{label}_full_transcript.txt").write_text(transcript, encoding="utf-8")
        return cues, None

    return [], "unsupported_asr_backend"


def calculate_frame_similarity(frame1_path: Path, frame2_path: Path) -> Optional[float]:
    img1 = cv2.imread(str(frame1_path))
    img2 = cv2.imread(str(frame2_path))
    if img1 is None or img2 is None:
        return None

    img1 = cv2.resize(img1, (320, 180))
    img2 = cv2.resize(img2, (320, 180))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
    hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    template = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    template_sim = float(template.max()) if template.size > 0 else 0.0
    return max(0.0, min(1.0, 0.5 * max(0.0, float(hist_sim)) + 0.5 * template_sim))


def text_similarity(a: str, b: str) -> Optional[float]:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def valid_asr_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t == "unavailable":
        return False
    if t.startswith("asr_error:"):
        return False
    return True


def combine_scores(weighted_scores: List[Tuple[Optional[float], float]]) -> Optional[float]:
    used: List[Tuple[float, float]] = []
    for score, weight in weighted_scores:
        if score is None:
            continue
        used.append((score, weight))
    if not used:
        return None
    weight_sum = sum(w for _, w in used)
    if weight_sum <= 0:
        return None
    return sum(s * w for s, w in used) / weight_sum


def verdict_from_score(score: Optional[float]) -> str:
    if score is None:
        return "未知"
    if score >= 0.90:
        return "高度一致"
    if score >= 0.75:
        return "基本一致"
    if score >= 0.55:
        return "存在差异"
    return "明显不一致"


def modality_flag(score: Optional[float], ok_threshold: float = 0.75, mismatch_threshold: float = 0.55) -> str:
    if score is None:
        return "unknown"
    if score < mismatch_threshold:
        return "mismatch"
    if score >= ok_threshold:
        return "ok"
    return "warning"


def score_to_pct(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    return f"{score * 100:.1f}%"


def generate_html_report(
    output_dir: Path,
    target_video: Path,
    candidate_video: Path,
    points: List[Dict],
    summary: Dict,
) -> Path:
    def rel(p: str) -> str:
        return os.path.relpath(p, output_dir)

    rows = []
    for p in points:
        subtitle_source = p.get("subtitle_source_used", "ocr")
        rows.append(
            f"""
            <tr>
              <td>{p['index']}</td>
              <td>{p['time_sec']:.3f}</td>
              <td>{p.get('sample_time_sec', p['time_sec']):.3f}</td>
              <td>
                <img src="{html.escape(rel(p['target_frame']))}" width="220"><br>
                <audio controls preload="none" src="{html.escape(rel(p['target_audio']))}"></audio>
              </td>
              <td>
                <img src="{html.escape(rel(p['candidate_frame']))}" width="220"><br>
                <audio controls preload="none" src="{html.escape(rel(p['candidate_audio']))}"></audio>
              </td>
              <td>{score_to_pct(p.get('visual_similarity'))}</td>
              <td>{score_to_pct(p.get('subtitle_similarity'))}<br><small>{html.escape(subtitle_source)}</small></td>
              <td>{score_to_pct(p.get('audio_similarity'))}</td>
              <td>{score_to_pct(p.get('overall_similarity'))}</td>
              <td>{html.escape(p.get('verdict', '未知'))}</td>
              <td>
                <b>字幕(原):</b> {html.escape((p.get('subtitle_target_used') or '')[:200])}<br>
                <b>字幕(二剪):</b> {html.escape((p.get('subtitle_candidate_used') or '')[:200])}<br>
                <b>ASR(原):</b> {html.escape((p.get('asr_target') or '')[:200])}<br>
                <b>ASR(二剪):</b> {html.escape((p.get('asr_candidate') or '')[:200])}
              </td>
            </tr>
            """
        )

    low_points = summary.get("lowest_points", [])
    low_html = "".join(
        [
            f"<li>#{pt['index']} @ {pt['time_sec']:.3f}s: overall={score_to_pct(pt.get('overall_similarity'))}, verdict={html.escape(pt.get('verdict', '未知'))}</li>"
            for pt in low_points
        ]
    )

    html_body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>视频对比审片报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft Yahei", sans-serif; margin: 20px; color: #1f2937; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .meta {{ margin-bottom: 16px; line-height: 1.6; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; margin: 12px 0 18px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #f9fafb; }}
    .card .k {{ font-size: 12px; color: #6b7280; }}
    .card .v {{ font-size: 20px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 12px; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    .wrap {{ overflow-x: auto; }}
    .section {{ margin: 16px 0; }}
    audio {{ width: 220px; margin-top: 4px; }}
    small {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>视频对比审片报告</h1>
  <div class="meta">
    <div><b>原素材:</b> {html.escape(str(target_video))}</div>
    <div><b>二次裁剪:</b> {html.escape(str(candidate_video))}</div>
    <div><b>候选来源:</b> {html.escape(str(summary.get('candidate_mode', 'unknown')))}</div>
    <div><b>二次裁剪耗时:</b> {summary.get('clip_elapsed_sec', 0.0)} 秒</div>
    <div><b>时间间隔:</b> {summary.get('interval')} 秒</div>
    <div><b>检查点:</b> {summary.get('total_points')}</div>
  </div>

  <div class="cards">
    <div class="card"><div class="k">画面平均</div><div class="v">{score_to_pct(summary.get('avg_visual'))}</div></div>
    <div class="card"><div class="k">字幕平均</div><div class="v">{score_to_pct(summary.get('avg_subtitle'))}</div></div>
    <div class="card"><div class="k">音频平均</div><div class="v">{score_to_pct(summary.get('avg_audio'))}</div></div>
    <div class="card"><div class="k">综合平均</div><div class="v">{score_to_pct(summary.get('avg_overall'))}</div></div>
  </div>

  <div class="section">
    <h2>汇总结论</h2>
    <div>总体判定: <b>{html.escape(summary.get('overall_verdict', '未知'))}</b></div>
    <div>高度一致点数: {summary.get('high_match_points')} / {summary.get('total_points')}</div>
    <div>明显不一致点数: {summary.get('mismatch_points')} / {summary.get('total_points')}</div>
    <div>最差检查点:</div>
    <ul>{low_html}</ul>
  </div>

  <div class="section wrap">
    <h2>逐点详细对比（画面+字幕+音频）</h2>
    <table>
      <thead>
        <tr>
          <th style="width:40px">#</th>
          <th style="width:70px">时间点</th>
          <th style="width:80px">采样时间</th>
          <th style="width:240px">原素材（画面+音频）</th>
          <th style="width:240px">二次裁剪（画面+音频）</th>
          <th style="width:70px">画面</th>
          <th style="width:80px">字幕</th>
          <th style="width:70px">音频</th>
          <th style="width:80px">综合</th>
          <th style="width:90px">判定</th>
          <th>文本证据（字幕/OCR/ASR）</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    report_path = output_dir / "comparison_report.html"
    report_path.write_text(html_body, encoding="utf-8")
    return report_path


def build_checkpoints(duration: float, interval: float, max_points: int) -> List[float]:
    points = []
    t = 0.0
    while t < duration + 1e-6:
        points.append(round(t, 3))
        t += interval
    if points and points[-1] < duration - 0.5:
        points.append(round(duration, 3))

    if len(points) <= max_points:
        return points

    step = max(1, len(points) // max_points)
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled[-1] = points[-1]
    return sampled[:max_points]


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", help="配置文件路径（JSON）")
    pre_args, _ = pre_parser.parse_known_args()

    try:
        cfg, cfg_path = load_section_config(REPO_ROOT, "build_ai_video_audit_bundle", explicit_path=pre_args.config)
        interval_default = cfg_float(cfg, "interval", 8.0)
        clip_duration_default = cfg_float(cfg, "clip_duration", 6.0)
        max_points_default = cfg_int(cfg, "max_points", 40)
        ocr_lang_default = cfg_str(cfg, "ocr_lang", "chi_sim+eng")
        language_default = cfg_str(cfg, "language", "zh")
        asr_model_default = cfg_str(cfg, "asr_model", "small")
        asr_default = cfg_str(cfg, "asr", "auto")
        asr_cmd_default = cfg_str(cfg, "asr_cmd", "")
        asr_python_default = cfg_str(cfg, "asr_python", "")
        output_dir_default = cfg_str(cfg, "output_dir", "")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 2

    parser = argparse.ArgumentParser(description="构建 AI 视频审片证据包")
    parser.add_argument("--config", default=str(cfg_path) if cfg_path else "", help="配置文件路径（JSON）")
    parser.add_argument("--target", required=True, help="基准视频")
    parser.add_argument("--candidate", required=True, help="候选视频")
    parser.add_argument("--target-sub", help="基准字幕文件（可选）")
    parser.add_argument("--candidate-sub", help="候选字幕文件（可选）")

    parser.add_argument("--interval", type=float, default=interval_default, help="抽检间隔（秒）")
    parser.add_argument("--clip-duration", type=float, default=clip_duration_default, help="每个检查点音频切片长度（秒）")
    parser.add_argument("--max-points", type=int, default=max_points_default, help="最大检查点数量")

    parser.add_argument("--ocr-lang", default=ocr_lang_default, help="Tesseract 语言")
    parser.add_argument("--language", default=language_default, help="ASR 语种")
    parser.add_argument("--asr-model", default=asr_model_default, help="ASR 模型名")
    parser.add_argument("--asr", choices=["auto", "none", "faster_whisper", "whisper"], default=asr_default)
    parser.add_argument("--asr-cmd", default=asr_cmd_default, help="指定 whisper 命令路径或命令串（跨环境）")
    parser.add_argument("--asr-python", default=asr_python_default, help="指定装有 whisper 的 python 解释器路径")
    parser.add_argument("--clip-elapsed-sec", type=float, default=0.0, help="二次裁剪耗时（秒，可由批处理注入）")
    parser.add_argument("--candidate-mode", default="existing", help="候选来源模式（existing/reconstructed）")

    parser.add_argument("--output-dir", default=output_dir_default, help="输出目录，默认 runtime/temp_outputs/ai_video_audit/<timestamp>")
    args = parser.parse_args()
    repo_root = REPO_ROOT

    ensure_bin("ffmpeg")
    ensure_bin("ffprobe")
    ensure_bin("tesseract")

    target = Path(args.target).resolve()
    candidate = Path(args.candidate).resolve()
    if not target.exists():
        raise FileNotFoundError(f"target 不存在: {target}")
    if not candidate.exists():
        raise FileNotFoundError(f"candidate 不存在: {candidate}")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (repo_root / "runtime" / "temp_outputs" / "ai_video_audit" / ts).resolve()

    frames_dir = out_dir / "frames"
    audio_dir = out_dir / "audio"
    ocr_dir = out_dir / "ocr"
    asr_dir = out_dir / "asr"

    for d in [out_dir, frames_dir, audio_dir, ocr_dir, asr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    duration = min(get_duration(target), get_duration(candidate))
    checkpoints = build_checkpoints(duration, max(0.5, args.interval), max(1, args.max_points))

    target_cues = load_cues(Path(args.target_sub).resolve()) if args.target_sub else []
    candidate_cues = load_cues(Path(args.candidate_sub).resolve()) if args.candidate_sub else []

    backend, asr_cmd_base, asr_error = select_asr_backend(args.asr, args.asr_cmd, args.asr_python)

    print("=" * 70)
    print("AI审片证据包构建")
    print("=" * 70)
    print(f"target: {target}")
    print(f"candidate: {candidate}")
    print(f"duration(min): {duration:.2f}s")
    print(f"checkpoints: {len(checkpoints)}")
    print(f"asr_backend: {backend}")
    if args.config:
        print(f"config: {args.config}")
    if asr_error:
        print(f"asr_note: {asr_error}")

    asr_target_cues: List[Cue] = []
    asr_candidate_cues: List[Cue] = []
    asr_init_error: Optional[str] = None
    if backend != "none":
        asr_target_cues, err_t = transcribe_video_to_cues(
            target, backend, args.asr_model, args.language, asr_dir, "target", asr_cmd_base
        )
        asr_candidate_cues, err_c = transcribe_video_to_cues(
            candidate, backend, args.asr_model, args.language, asr_dir, "candidate", asr_cmd_base
        )
        errs = [e for e in [err_t, err_c] if e]
        if errs:
            asr_init_error = ";".join(errs)
            print(f"asr_init_error: {asr_init_error}")

    report_points = []
    for i, t in enumerate(checkpoints, start=1):
        t_sample = min(max(0.0, t), max(0.0, duration - 0.2))
        print(f"[{i}/{len(checkpoints)}] t={t:.3f}s sample={t_sample:.3f}s")

        tf = frames_dir / f"target_{i:03d}_{int(t_sample*1000):09d}.jpg"
        cf = frames_dir / f"candidate_{i:03d}_{int(t_sample*1000):09d}.jpg"
        ta = audio_dir / f"target_{i:03d}_{int(t_sample*1000):09d}.wav"
        ca = audio_dir / f"candidate_{i:03d}_{int(t_sample*1000):09d}.wav"

        extract_frame(target, t_sample, tf)
        extract_frame(candidate, t_sample, cf)
        extract_audio_clip(target, t_sample, args.clip_duration, ta)
        extract_audio_clip(candidate, t_sample, args.clip_duration, ca)

        ocr_t = ocr_bottom_subtitle(tf, args.ocr_lang)
        ocr_c = ocr_bottom_subtitle(cf, args.ocr_lang)

        (ocr_dir / f"target_{i:03d}.txt").write_text(ocr_t, encoding="utf-8")
        (ocr_dir / f"candidate_{i:03d}.txt").write_text(ocr_c, encoding="utf-8")

        asr_t = ""
        asr_c = ""
        if backend != "none" and not asr_init_error:
            asr_t = cue_text_window(asr_target_cues, t_sample, t_sample + args.clip_duration)
            asr_c = cue_text_window(asr_candidate_cues, t_sample, t_sample + args.clip_duration)

        (asr_dir / f"target_{i:03d}.txt").write_text(asr_t, encoding="utf-8")
        (asr_dir / f"candidate_{i:03d}.txt").write_text(asr_c, encoding="utf-8")

        subtitle_target_file = cue_text_at(target_cues, t)
        subtitle_candidate_file = cue_text_at(candidate_cues, t)
        # 只有两边都命中字幕文件时才使用字幕文件对比，避免“一边字幕文件一边OCR”导致误判。
        has_target_sub_file = bool((subtitle_target_file or "").strip())
        has_candidate_sub_file = bool((subtitle_candidate_file or "").strip())
        use_subtitle_file_pair = has_target_sub_file and has_candidate_sub_file
        subtitle_source = "subtitle_file_pair" if use_subtitle_file_pair else "ocr"
        subtitle_target_used = subtitle_target_file if use_subtitle_file_pair else ocr_t
        subtitle_candidate_used = subtitle_candidate_file if use_subtitle_file_pair else ocr_c

        visual_similarity = calculate_frame_similarity(tf, cf)
        subtitle_similarity = text_similarity(subtitle_target_used, subtitle_candidate_used)

        audio_similarity = None
        if valid_asr_text(asr_t) and valid_asr_text(asr_c):
            audio_similarity = text_similarity(asr_t, asr_c)

        overall_similarity = combine_scores(
            [
                (visual_similarity, 0.50),
                (subtitle_similarity, 0.25),
                (audio_similarity, 0.25),
            ]
        )
        visual_flag = modality_flag(visual_similarity)
        subtitle_flag = modality_flag(subtitle_similarity)
        audio_flag = modality_flag(audio_similarity)
        if "mismatch" in [visual_flag, subtitle_flag, audio_flag]:
            verdict = "明显不一致"
        else:
            verdict = verdict_from_score(overall_similarity)

        report_points.append({
            "index": i,
            "time_sec": t,
            "sample_time_sec": t_sample,
            "target_frame": str(tf),
            "candidate_frame": str(cf),
            "target_audio": str(ta),
            "candidate_audio": str(ca),
            "ocr_target": ocr_t,
            "ocr_candidate": ocr_c,
            "subtitle_target_from_file": subtitle_target_file,
            "subtitle_candidate_from_file": subtitle_candidate_file,
            "subtitle_source_used": subtitle_source,
            "subtitle_target_used": subtitle_target_used,
            "subtitle_candidate_used": subtitle_candidate_used,
            "asr_target": asr_t if backend != "none" else "unavailable",
            "asr_candidate": asr_c if backend != "none" else "unavailable",
            "visual_similarity": visual_similarity,
            "subtitle_similarity": subtitle_similarity,
            "audio_similarity": audio_similarity,
            "overall_similarity": overall_similarity,
            "visual_flag": visual_flag,
            "subtitle_flag": subtitle_flag,
            "audio_flag": audio_flag,
            "verdict": verdict,
        })

    def avg_score(key: str) -> Optional[float]:
        vals = [p[key] for p in report_points if p.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    avg_visual = avg_score("visual_similarity")
    avg_subtitle = avg_score("subtitle_similarity")
    avg_audio = avg_score("audio_similarity")
    avg_overall = avg_score("overall_similarity")

    high_match_points = sum(1 for p in report_points if p.get("verdict") == "高度一致")
    mismatch_points = sum(1 for p in report_points if p.get("verdict") == "明显不一致")
    lowest_points = sorted(report_points, key=lambda x: x.get("overall_similarity") if x.get("overall_similarity") is not None else 9.0)[:10]
    overall_verdict = verdict_from_score(avg_overall)

    summary = {
        "interval": args.interval,
        "total_points": len(report_points),
        "avg_visual": avg_visual,
        "avg_subtitle": avg_subtitle,
        "avg_audio": avg_audio,
        "avg_overall": avg_overall,
        "overall_verdict": overall_verdict,
        "high_match_points": high_match_points,
        "mismatch_points": mismatch_points,
        "lowest_points": lowest_points,
        "clip_elapsed_sec": round(float(args.clip_elapsed_sec), 3),
        "candidate_mode": str(args.candidate_mode),
    }

    html_report_path = generate_html_report(out_dir, target, candidate, report_points, summary)

    manifest = {
        "target": str(target),
        "candidate": str(candidate),
        "duration_used": duration,
        "interval": args.interval,
        "clip_duration": args.clip_duration,
        "asr_backend": backend,
        "asr_error": asr_error or asr_init_error,
        "asr_cmd_used": " ".join(asr_cmd_base) if asr_cmd_base else None,
        "asr_model": args.asr_model,
        "candidate_mode": str(args.candidate_mode),
        "clip_elapsed_sec": round(float(args.clip_elapsed_sec), 3),
        "summary": summary,
        "comparison_report_html": str(html_report_path),
        "subtitle_inputs": {
            "target_sub": str(Path(args.target_sub).resolve()) if args.target_sub else None,
            "candidate_sub": str(Path(args.candidate_sub).resolve()) if args.candidate_sub else None,
        },
        "checkpoints": report_points,
        "review_guide": {
            "steps": [
                "逐条查看 target_frame 与 candidate_frame，确认画面语义是否一致",
                "对比 OCR 与字幕文件文本，确认字幕是否一致",
                "若 asr_* 可用，检查语音语义是否一致；否则对问题点人工复听",
                "输出问题清单：time_sec + 问题类型 + 证据路径",
            ]
        }
    }

    manifest_path = out_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("证据包已生成")
    print(f"输出目录: {out_dir}")
    print(f"清单文件: {manifest_path}")
    print(f"详细报告页: {html_report_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
