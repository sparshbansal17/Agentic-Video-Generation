from __future__ import annotations

from pathlib import Path
from typing import Any


PRIMARY_SYSTEMS = ("storymem_agentic", "automv", "mavin", "movieagent")
TRACKS = ("conditioned_song_to_video", "end_to_end_prompt_to_song_video")


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark manifest or submission violates the public contract."""


def _require(value: Any, expected_type: type, label: str) -> None:
    if not isinstance(value, expected_type) or (expected_type in {str, list, dict} and not value):
        raise BenchmarkValidationError(f"{label} must be a non-empty {expected_type.__name__}")


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    _require(manifest.get("benchmark_id"), str, "benchmark_id")
    _require(manifest.get("version"), str, "version")
    _require(manifest.get("systems"), list, "systems")
    _require(manifest.get("cases"), list, "cases")
    _require(manifest.get("seeds"), list, "seeds")
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in manifest["seeds"]):
        raise BenchmarkValidationError("seeds must contain only integers")
    if len(set(manifest["seeds"])) != len(manifest["seeds"]):
        raise BenchmarkValidationError("seeds must be unique")
    systems = set(manifest["systems"])
    missing_systems = set(PRIMARY_SYSTEMS) - systems
    if missing_systems:
        raise BenchmarkValidationError(
            f"manifest is missing primary systems: {sorted(missing_systems)}"
        )
    seen: set[str] = set()
    for index, case in enumerate(manifest["cases"]):
        label = f"cases[{index}]"
        _require(case, dict, label)
        _require(case.get("case_id"), str, f"{label}.case_id")
        if case["case_id"] in seen:
            raise BenchmarkValidationError(f"duplicate case_id: {case['case_id']}")
        seen.add(case["case_id"])
        if case.get("track") not in TRACKS:
            raise BenchmarkValidationError(f"{label}.track must be one of {TRACKS}")
        _require(case.get("prompt"), str, f"{label}.prompt")
        _require(case.get("lyrics"), list, f"{label}.lyrics")
        if not all(isinstance(line, str) and line.strip() for line in case["lyrics"]):
            raise BenchmarkValidationError(f"{label}.lyrics contains an empty/non-string line")
        duration = case.get("target_duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise BenchmarkValidationError(f"{label}.target_duration_seconds must be positive")
        scenes = case.get("expected_scenes")
        if not isinstance(scenes, int) or scenes <= 0:
            raise BenchmarkValidationError(f"{label}.expected_scenes must be a positive integer")
    return manifest


def validate_submission(
    submission: dict[str, Any],
    *,
    manifest: dict[str, Any],
    base_dir: Path | None = None,
    require_media: bool = False,
) -> dict[str, Any]:
    _require(submission.get("system"), str, "system")
    _require(submission.get("case_id"), str, "case_id")
    if submission["system"] not in manifest["systems"]:
        raise BenchmarkValidationError(f"unknown system: {submission['system']}")
    case_ids = {case["case_id"] for case in manifest["cases"]}
    if submission["case_id"] not in case_ids:
        raise BenchmarkValidationError(f"unknown case_id: {submission['case_id']}")
    _require(submission.get("final_video"), str, "final_video")
    if require_media:
        root = base_dir or Path.cwd()
        video = Path(submission["final_video"])
        if not video.is_absolute():
            video = root / video
        if not video.is_file() or video.stat().st_size == 0:
            raise BenchmarkValidationError(f"final_video is missing or empty: {video}")
    return submission
