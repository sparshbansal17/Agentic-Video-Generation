from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .alignment import repeated_word_requirements, transcript_from_words, word_error_rate, words
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
    observed_tokens = words(actual)
    expected_lines = list(plan.lyrics)
    line_reports = []
    cursor = 0
    for line_index, line in enumerate(expected_lines, start=1):
        expected_tokens = words(line)
        matched = 0
        matched_counts: dict[str, int] = {}
        for token in expected_tokens:
            found_index = None
            for observed_index in range(cursor, len(observed_tokens)):
                if observed_tokens[observed_index] == token:
                    found_index = observed_index
                    break
            if found_index is None:
                continue
            cursor = found_index + 1
            matched += 1
            matched_counts[token] = matched_counts.get(token, 0) + 1
        repeated = repeated_word_requirements(line)
        repeated_omissions = {
            token: {"expected": expected_count, "matched": matched_counts.get(token, 0)}
            for token, expected_count in repeated.items()
            if matched_counts.get(token, 0) < expected_count
        }
        if matched < len(expected_tokens):
            failure_reasons.append(f"line_{line_index}_missing_words")
        for token, counts in repeated_omissions.items():
            failure_reasons.append(
                f"line_{line_index}_omitted_repeated_{token}_{counts['matched']}_of_{counts['expected']}"
            )
        if line_index == len(expected_lines) and matched < len(expected_tokens):
            failure_reasons.append(f"line_{line_index}_final_line_incomplete")
        line_reports.append(
            {
                "line_index": line_index,
                "text": line,
                "matched_word_count": matched,
                "expected_word_count": len(expected_tokens),
                "matched_ratio": matched / len(expected_tokens) if expected_tokens else 1.0,
                "repeated_word_omissions": repeated_omissions,
            }
        )
    failure_reasons = list(dict.fromkeys(failure_reasons))
    return {
        "expected_text": expected,
        "observed_text": actual,
        "word_error_rate": wer,
        "passes_wer": passes_wer and not failure_reasons,
        "failure_reasons": failure_reasons,
        "lines": line_reports,
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
