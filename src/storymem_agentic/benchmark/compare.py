from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import validate_submission


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _planning_path(plans_root: Path, system: str, case_id: str, seed: int) -> Path:
    return plans_root / system / case_id / f"seed_{seed:03d}" / "planning_metrics.json"


def compare_submissions(
    *, manifest: dict[str, Any], submissions_root: Path, plans_root: Path, seed: int = 0
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for submission_path in sorted(submissions_root.glob("**/submission.json")):
        try:
            submission = _json(submission_path)
            validate_submission(
                submission,
                manifest=manifest,
                base_dir=submission_path.parent,
                require_media=True,
            )
            delivery_path = submission_path.parent / str(submission.get("evaluation_report"))
            delivery = _json(delivery_path)
            planning_path = _planning_path(
                plans_root, submission["system"], submission["case_id"], seed
            )
            if not planning_path.is_file():
                local_planning = submission_path.parent / "planning_metrics.json"
                planning_path = local_planning if local_planning.is_file() else planning_path
            planning = _json(planning_path) if planning_path.is_file() else {}
            media_path = submission_path.parent / "media_metrics.json"
            media = _json(media_path) if media_path.is_file() else {}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(submission_path), "error": str(exc)})
            continue
        rows.append(
            {
                "system": submission["system"],
                "case_id": submission["case_id"],
                "delivery_pass": bool(
                    delivery.get("has_video_stream")
                    and delivery.get("has_audio_stream")
                    and float(delivery.get("absolute_duration_error_seconds", 1e9)) <= 0.25
                ),
                "duration_error_seconds": delivery.get("absolute_duration_error_seconds"),
                "audio_checksum_matches_manifest": delivery.get("locked_audio_sha256")
                == next(
                    case.get("locked_audio_sha256")
                    for case in manifest["cases"]
                    if case["case_id"] == submission["case_id"]
                ),
                "scene_count_accuracy": planning.get("scene_count_accuracy"),
                "exact_lyric_and_subtitle_rate": planning.get(
                    "exact_lyric_and_subtitle_rate"
                ),
                "required_entity_exact_phrase_coverage": planning.get(
                    "required_entity_exact_phrase_coverage"
                ),
                "required_entity_key_token_coverage": planning.get(
                    "required_entity_key_token_coverage"
                ),
                "primary_entity_scene_coverage": planning.get(
                    "primary_entity_scene_coverage"
                ),
                "camera_vocabulary_per_expected_scene": planning.get(
                    "camera_vocabulary_per_expected_scene"
                ),
                "safety_language_scene_rate": planning.get("safety_language_scene_rate"),
                "unsafe_term_occurrences": planning.get("unsafe_term_occurrences"),
                "schema_placeholder_occurrences": planning.get(
                    "schema_placeholder_occurrences"
                ),
                "agent_calls": planning.get("agent_calls"),
                "planning_seconds": planning.get("planning_seconds"),
                "generation_seconds": submission.get("wall_time_seconds"),
                "clip_assigned_scene_similarity_mean": media.get(
                    "clip_assigned_scene_similarity_mean"
                ),
                "clip_global_prompt_similarity_mean": media.get(
                    "clip_global_prompt_similarity_mean"
                ),
                "clip_lyric_retrieval_order_accuracy": media.get(
                    "clip_lyric_retrieval_order_accuracy"
                ),
                "clip_adjacent_scene_similarity_mean": media.get(
                    "clip_adjacent_scene_similarity_mean"
                ),
                "notes": submission.get("notes", ""),
            }
        )
    return {
        "benchmark_id": manifest["benchmark_id"],
        "seed": seed,
        "rows": sorted(rows, key=lambda row: row["system"]),
        "validation_errors": errors,
        "interpretation": (
            "Planning values are contract/proxy metrics; delivery values are measured from media. "
            "Do not interpret either family as blinded perceptual quality."
        ),
    }


def render_comparison_markdown(report: dict[str, Any]) -> str:
    columns = (
        "system",
        "delivery_pass",
        "duration_error_seconds",
        "scene_count_accuracy",
        "exact_lyric_and_subtitle_rate",
        "required_entity_key_token_coverage",
        "primary_entity_scene_coverage",
        "safety_language_scene_rate",
        "planning_seconds",
        "generation_seconds",
        "clip_assigned_scene_similarity_mean",
        "clip_lyric_retrieval_order_accuracy",
    )
    lines = [
        "# Agentic AV benchmark comparison",
        "",
        report["interpretation"],
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in report["rows"]:
        lines.append("| " + " | ".join(str(row.get(column)) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def write_comparison(
    *,
    manifest_path: str | Path,
    submissions_root: str | Path,
    plans_root: str | Path,
    output_dir: str | Path,
    seed: int = 0,
) -> tuple[Path, Path]:
    manifest = _json(Path(manifest_path))
    report = compare_submissions(
        manifest=manifest,
        submissions_root=Path(submissions_root),
        plans_root=Path(plans_root),
        seed=seed,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "comparison.json"
    markdown_path = destination / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_comparison_markdown(report), encoding="utf-8")
    return json_path, markdown_path
