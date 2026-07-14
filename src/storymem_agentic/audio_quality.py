from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

_DB_RE = re.compile(r"(?P<name>mean_volume|max_volume):\s*(?P<value>-?[0-9.]+) dB")


def evaluate_audio_metrics(
    metrics: dict[str, Any],
    *,
    duration_tolerance_seconds: float = 0.1,
    maximum_peak_db: float = -0.1,
    minimum_mean_db: float = -45.0,
) -> dict[str, Any]:
    failures = []
    if not metrics.get("has_audio", False):
        failures.append("missing_audio_stream")
    observed = float(metrics.get("duration_seconds") or 0.0)
    expected = float(metrics.get("expected_duration_seconds") or 0.0)
    if expected and abs(observed - expected) > duration_tolerance_seconds:
        failures.append("audio_duration_out_of_tolerance")
    peak = metrics.get("max_volume_db")
    if isinstance(peak, (int, float)) and float(peak) > maximum_peak_db:
        failures.append("audio_clipping_risk")
    mean = metrics.get("mean_volume_db")
    if isinstance(mean, (int, float)) and float(mean) < minimum_mean_db:
        failures.append("audio_effectively_silent")
    return {"passed": not failures, "failure_reasons": failures, "metrics": metrics}


def probe_audio_quality(
    media_path: str | Path,
    *,
    expected_duration_seconds: float,
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, Any]:
    path = Path(media_path)
    metrics: dict[str, Any] = {
        "has_audio": False,
        "duration_seconds": 0.0,
        "expected_duration_seconds": expected_duration_seconds,
        "mean_volume_db": None,
        "max_volume_db": None,
    }
    if not path.exists():
        return evaluate_audio_metrics(metrics)
    probe = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    metadata = probe.stdout + probe.stderr
    metrics["has_audio"] = "Audio:" in metadata
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):([0-9.]+)", metadata)
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        metrics["duration_seconds"] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if metrics["has_audio"]:
        volume = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for match in _DB_RE.finditer(volume.stdout + volume.stderr):
            metrics[f"{match.group('name')}_db"] = float(match.group("value"))
    return evaluate_audio_metrics(metrics)
