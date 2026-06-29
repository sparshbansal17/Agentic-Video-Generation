from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = words(reference)
    hyp = words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def load_whisperx_words(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    output: list[dict[str, Any]] = []
    for segment in data.get("segments", []):
        for word in segment.get("words", []):
            if "word" in word:
                output.append(word)
    return output


def transcript_from_words(aligned_words: list[dict[str, Any]]) -> str:
    return " ".join(str(item.get("word", "")).strip() for item in aligned_words).strip()


def normalized_word_tokens(aligned_words: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    tokens: list[tuple[str, dict[str, Any]]] = []
    for item in aligned_words:
        for token in words(str(item.get("word", ""))):
            tokens.append((token, item))
    return tokens


def _token_seconds(item: dict[str, Any]) -> tuple[float | None, float | None]:
    start = item.get("start")
    end = item.get("end")
    return (
        float(start) if start is not None else None,
        float(end) if end is not None else None,
    )


def _tokens_for_window(
    observed_tokens: list[tuple[str, dict[str, Any]]],
    window: tuple[float, float] | None,
    tolerance: float,
) -> list[tuple[str, dict[str, Any]]]:
    if window is None:
        return observed_tokens
    start, end = window
    window_start = start - tolerance
    window_end = end + tolerance
    scoped = []
    for token, item in observed_tokens:
        token_start, token_end = _token_seconds(item)
        if token_start is None or token_end is None:
            scoped.append((token, item))
            continue
        if token_end >= window_start and token_start <= window_end:
            scoped.append((token, item))
    return scoped


def _ordered_line_match(wanted: list[str], observed_tokens: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    cursor = 0
    matched = []
    for wanted_word in wanted:
        found_index = None
        for token_index in range(cursor, len(observed_tokens)):
            if observed_tokens[token_index][0] == wanted_word:
                found_index = token_index
                break
        if found_index is None:
            continue
        matched.append(observed_tokens[found_index][1])
        cursor = found_index + 1
    return matched


def _timing_items(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reliable = [
        item
        for item in matched
        if "start" in item and "end" in item and float(item.get("score", 1.0) or 0.0) >= 0.15
    ]
    return reliable or [item for item in matched if "start" in item and "end" in item]


def line_timestamps(
    reference_lines: list[str],
    aligned_words: list[dict[str, Any]],
    planned_windows: list[tuple[float, float]] | None = None,
    *,
    window_tolerance_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    observed_tokens = normalized_word_tokens(aligned_words)
    output = []
    for index, line in enumerate(reference_lines, start=1):
        wanted = words(line)
        window = planned_windows[index - 1] if planned_windows and index <= len(planned_windows) else None
        candidates = _tokens_for_window(observed_tokens, window, window_tolerance_seconds)
        matched = _ordered_line_match(wanted, candidates)
        timing_items = _timing_items(matched)
        starts = [float(item["start"]) for item in timing_items if "start" in item]
        ends = [float(item["end"]) for item in timing_items if "end" in item]
        output.append(
            {
                "line_index": index,
                "text": line,
                "observed_start_seconds": min(starts) if starts else None,
                "observed_end_seconds": max(ends) if ends else None,
                "matched_word_count": len(matched),
                "expected_word_count": len(wanted),
                "matched_ratio": (len(matched) / len(wanted)) if wanted else 1.0,
            }
        )
    return output


def analyze_whisperx_alignment(
    reference_lines: list[str],
    planned_windows: list[tuple[float, float]],
    aligned_words: list[dict[str, Any]],
    *,
    wer_threshold: float = 0.25,
    drift_tolerance_seconds: float = 1.0,
) -> dict[str, Any]:
    reference = " ".join(reference_lines)
    hypothesis = transcript_from_words(aligned_words)
    wer = word_error_rate(reference, hypothesis)
    lines = line_timestamps(reference_lines, aligned_words, planned_windows)
    failures = []
    for item, (planned_start, planned_end) in zip(lines, planned_windows):
        observed_start = item["observed_start_seconds"]
        observed_end = item["observed_end_seconds"]
        if observed_start is None or observed_end is None:
            failures.append(f"line_{item['line_index']}_missing_words")
            continue
        if item["matched_ratio"] < 0.8:
            failures.append(f"line_{item['line_index']}_partial_words")
        start_drift = observed_start - planned_start
        end_drift = observed_end - planned_end
        item["start_drift_seconds"] = round(start_drift, 3)
        item["end_drift_seconds"] = round(end_drift, 3)
        if observed_start < planned_start - drift_tolerance_seconds:
            failures.append(f"line_{item['line_index']}_starts_before_scene")
        if observed_end > planned_end + drift_tolerance_seconds:
            failures.append(f"line_{item['line_index']}_ends_after_scene")
    if wer > wer_threshold:
        failures.append("wer_above_threshold")
    return {
        "passed": not failures,
        "word_error_rate": wer,
        "wer_threshold": wer_threshold,
        "drift_tolerance_seconds": drift_tolerance_seconds,
        "transcript": hypothesis,
        "lines": lines,
        "failure_reasons": failures,
    }
