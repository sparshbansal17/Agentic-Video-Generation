from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _iteration_number(path: Path) -> int:
    try:
        return int(path.parent.name)
    except ValueError:
        return 0


def _run_id(path: Path, results_root: Path) -> str:
    parts = path.relative_to(results_root).parts
    try:
        index = parts.index("iterations")
    except ValueError:
        return str(path.parent.relative_to(results_root))
    return str(Path(*parts[:index]))


def _run_manifest(results_root: Path, run_id: str) -> dict[str, Any]:
    path = results_root / run_id / "run_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _media_files(iteration_dir: Path) -> list[Path]:
    return [path for path in iteration_dir.rglob("*.mp4") if path.is_file() and path.stat().st_size]


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _extract_alignment(report: dict[str, Any]) -> dict[str, Any] | None:
    alignment = report.get("whisperx_alignment")
    if isinstance(alignment, dict) and alignment:
        return alignment
    for reviewer in report.get("reviewer_reports", []):
        if reviewer.get("reviewer") == "WhisperXLyricTimingAgent":
            evidence = reviewer.get("evidence")
            if isinstance(evidence, dict) and evidence:
                return evidence
    return None


def _report_record(path: Path, results_root: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    run_id = _run_id(path, results_root)
    run_manifest = _run_manifest(results_root, run_id)
    media = _media_files(path.parent)
    scene_reports = report.get("scene_reports", [])
    reviewer_reports = report.get("reviewer_reports", [])
    alignment = _extract_alignment(report)
    scene_scores = [
        _as_float(scene.get("scores", {}).get("vlm_prompt_adherence")) for scene in scene_reports
    ]
    scene_scores = [value for value in scene_scores if value is not None]
    line_ratios: list[float] = []
    absolute_drifts: list[float] = []
    if alignment:
        for line in alignment.get("lines", []):
            ratio = _as_float(line.get("matched_ratio"))
            if ratio is not None:
                line_ratios.append(ratio)
            for name in ("start_drift_seconds", "end_drift_seconds"):
                drift = _as_float(line.get(name))
                if drift is not None:
                    absolute_drifts.append(abs(drift))
    regeneration_targets = report.get("regeneration_targets", [])
    targeted_scenes = {
        int(value)
        for value in regeneration_targets
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return {
        "run_id": run_id,
        "iteration": _iteration_number(path),
        "path": str(path),
        "passed": bool(report.get("passed")),
        "media_file_count": len(media),
        "media_bytes": sum(item.stat().st_size for item in media),
        "scene_count": len(scene_reports),
        "scene_pass_rate": _mean(float(bool(scene.get("passed"))) for scene in scene_reports),
        "prompt_adherence": _mean(scene_scores),
        "reviewer_count": len(reviewer_reports),
        "reviewers": {
            str(item.get("reviewer")): bool(item.get("passed")) for item in reviewer_reports
        },
        "artifact_checks": {
            str(name): bool(value)
            for name, value in report.get("artifact_checks", {}).items()
            if isinstance(value, bool)
        },
        "configuration": {
            "mode": str(run_manifest.get("mode", "unknown")),
            "planner_backend": str(run_manifest.get("planner_backend", "unknown")),
            "review_backend": str(run_manifest.get("review_backend", "unknown")),
            "audio_aligner": str(run_manifest.get("audio_aligner", "unknown")),
        },
        "wer": _as_float(alignment.get("word_error_rate")) if alignment else None,
        "line_completeness": _mean(line_ratios),
        "absolute_line_drift_seconds": _mean(absolute_drifts),
        "targeted_scene_count": len(targeted_scenes),
        "target_fraction": (len(targeted_scenes) / len(scene_reports) if scene_reports else None),
    }


def evaluate_history(results_root: str | Path) -> dict[str, Any]:
    """Aggregate only reports backed by non-empty local MP4 files.

    Scores in historical reports are pipeline-internal measurements, not human ratings and
    not directly comparable to numbers published on another paper's dataset.
    """
    root = Path(results_root).resolve()
    discovered = sorted(root.glob("**/iterations/*/evaluation_report.json"))
    unreadable: list[str] = []
    records: list[dict[str, Any]] = []
    for path in discovered:
        try:
            record = _report_record(path, root)
        except (OSError, ValueError, json.JSONDecodeError):
            unreadable.append(str(path))
            continue
        if record["media_file_count"]:
            records.append(record)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["run_id"]].append(record)
    for run_records in grouped.values():
        run_records.sort(key=lambda item: item["iteration"])
    initial = [items[0] for items in grouped.values()]
    terminal = [items[-1] for items in grouped.values()]
    attempted_repairs = [
        items for items in grouped.values() if len(items) > 1 and not items[0]["passed"]
    ]
    recovered = [items for items in attempted_repairs if items[-1]["passed"]]

    reviewer_names = sorted(
        {name for record in terminal for name in record["reviewers"] if name != "None"}
    )
    reviewer_pass_rates = {
        name: _round(
            _mean(
                float(record["reviewers"][name])
                for record in terminal
                if name in record["reviewers"]
            )
        )
        for name in reviewer_names
    }
    with_wer = [record for record in terminal if record["wer"] is not None]
    with_lines = [record for record in terminal if record["line_completeness"] is not None]
    with_drift = [
        record for record in terminal if record["absolute_line_drift_seconds"] is not None
    ]
    targeted = [record for record in records if record["target_fraction"] is not None]
    artifact_names = sorted({name for record in terminal for name in record["artifact_checks"]})
    artifact_pass_rates = {
        name: _round(
            _mean(
                float(record["artifact_checks"][name])
                for record in terminal
                if name in record["artifact_checks"]
            )
        )
        for name in artifact_names
    }
    configurations: dict[str, dict[str, Any]] = {}
    for field in ("mode", "planner_backend", "review_backend", "audio_aligner"):
        values = sorted({record["configuration"][field] for record in terminal})
        configurations[field] = {
            value: sum(record["configuration"][field] == value for record in terminal)
            for value in values
        }
    command_review = [
        record for record in terminal if record["configuration"]["review_backend"] == "command"
    ]

    return {
        "schema_version": "1.0",
        "evaluation_scope": "retrospective_media_backed_storymem_agentic_runs",
        "comparability_warning": (
            "Pipeline-internal scores are diagnostic only. They are not direct AutoMV, MAVIN, "
            "or MovieAgent comparisons until all systems run the locked benchmark inputs."
        ),
        "provenance": {
            "results_root": str(root),
            "reports_discovered": len(discovered),
            "reports_with_nonempty_mp4": len(records),
            "reports_excluded_without_media": len(discovered) - len(records) - len(unreadable),
            "unreadable_reports": unreadable,
        },
        "run_metrics": {
            "media_backed_runs": len(grouped),
            "media_backed_iterations": len(records),
            "first_media_backed_iteration_pass_rate": _round(
                _mean(float(item["passed"]) for item in initial)
            ),
            "latest_media_backed_iteration_pass_rate": _round(
                _mean(float(item["passed"]) for item in terminal)
            ),
            "mean_iterations_per_run": _round(
                _mean(float(len(items)) for items in grouped.values())
            ),
            "repair_attempts_after_media_backed_failure": len(attempted_repairs),
            "repair_success_rate": _round(
                len(recovered) / len(attempted_repairs) if attempted_repairs else None
            ),
        },
        "configuration_breakdown": {
            **configurations,
            "command_review_terminal_runs": len(command_review),
            "command_review_terminal_pass_rate": _round(
                _mean(float(record["passed"]) for record in command_review)
            ),
        },
        "terminal_quality_diagnostics": {
            "mean_scene_pass_rate": _round(
                _mean(
                    item["scene_pass_rate"]
                    for item in terminal
                    if item["scene_pass_rate"] is not None
                )
            ),
            "mean_heuristic_prompt_adherence_proxy": _round(
                _mean(
                    item["prompt_adherence"]
                    for item in terminal
                    if item["prompt_adherence"] is not None
                )
            ),
            "prompt_adherence_note": (
                "Legacy non-command reports assign 0.8 when a clip exists; this is an artifact "
                "proxy, not a semantic model or human quality score."
            ),
            "mean_reviewer_count": _round(
                _mean(float(item["reviewer_count"]) for item in terminal)
            ),
            "reviewer_pass_rates": reviewer_pass_rates,
            "artifact_pass_rates": artifact_pass_rates,
        },
        "audio_diagnostics": {
            "terminal_runs_with_wer": len(with_wer),
            "mean_wer": _round(_mean(item["wer"] for item in with_wer)),
            "terminal_runs_with_line_completeness": len(with_lines),
            "mean_line_completeness": _round(
                _mean(item["line_completeness"] for item in with_lines)
            ),
            "terminal_runs_with_drift": len(with_drift),
            "mean_absolute_line_boundary_drift_seconds": _round(
                _mean(item["absolute_line_drift_seconds"] for item in with_drift)
            ),
        },
        "repair_localization_diagnostics": {
            "iterations_with_scene_reports": len(targeted),
            "mean_fraction_of_scenes_targeted": _round(
                _mean(item["target_fraction"] for item in targeted)
            ),
            "note": "A lower fraction is efficient only when the reviewer found every real fault.",
        },
        "terminal_runs": terminal,
    }
