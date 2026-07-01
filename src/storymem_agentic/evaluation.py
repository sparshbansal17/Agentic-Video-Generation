from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .alignment import transcript_from_words, word_error_rate
from .schemas import AudioPlan


def evaluate_alignment(plan: AudioPlan, aligned_words: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    expected = "\n".join(plan.lyrics)
    actual = transcript_from_words(aligned_words or [])
    has_expected = bool(expected.strip())
    has_observed = bool(actual.strip())
    wer = word_error_rate(expected, actual) if has_observed else None
    passes_wer = has_observed and (wer is not None and wer <= 0.08)
    failure_reasons: list[str] = []
    if has_expected and not has_observed:
        failure_reasons.append("missing_observed_lyrics")
    elif not passes_wer:
        failure_reasons.append("wer_above_threshold")
    return {
        "expected_text": expected,
        "observed_text": actual,
        "word_error_rate": wer,
        "passes_wer": passes_wer,
        "failure_reasons": failure_reasons,
    }


def evaluate_manifest(plan: AudioPlan, mix_manifest: dict, aligned_words: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    duration_delta_ms = abs(
        float(mix_manifest.get("target_duration_seconds", 0.0)) - plan.target_duration_seconds
    ) * 1000
    duration_tolerance = float(plan.mix.get("duration_tolerance_ms", 250))
    alignment = evaluate_alignment(plan, aligned_words)
    passed = duration_delta_ms <= duration_tolerance and alignment["passes_wer"]
    return {
        "passed": passed,
        "duration_delta_ms": round(duration_delta_ms, 3),
        "duration_tolerance_ms": duration_tolerance,
        "alignment": alignment,
        "recommended_action": "accept" if passed else "regenerate_voice_or_adjust_timing",
    }


def write_evaluation(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return target
