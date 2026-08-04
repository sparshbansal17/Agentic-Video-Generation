from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import validate_submission


PRIMARY_ROW_KEYS = {
    "semantic_contrastive_margin_mean": "advanced_semantic_contrastive_margin_mean",
    "semantic_retrieval_mrr": "advanced_semantic_retrieval_mrr",
    "dense_frame_margin_p10": "advanced_dense_frame_margin_p10",
    "dinov2_within_scene_consistency": "advanced_dinov2_within_scene_consistency",
    "flow_warp_error_mean": "advanced_flow_warp_error_mean",
    "panel_weighted_overall_score": "panel_weighted_overall_score",
    "panel_semantic_grounding": "panel_semantic_grounding",
    "panel_narrative_progression": "panel_narrative_progression",
    "panel_subject_identity_consistency": "panel_subject_identity_consistency",
    "panel_temporal_coherence": "panel_temporal_coherence",
    "panel_visual_quality": "panel_visual_quality",
}


def _ranking_summary(
    rows: list[dict[str, Any]], metric_directions: dict[str, str]
) -> dict[str, Any]:
    systems = [str(row["system"]) for row in rows]
    wins = {system: 0.0 for system in systems}
    rankings = []
    saturated = []
    for metric, direction in metric_directions.items():
        row_key = PRIMARY_ROW_KEYS.get(metric, metric)
        values = {str(row["system"]): row.get(row_key) for row in rows}
        if not values or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values.values()
        ):
            continue
        target = min(values.values()) if direction == "lower" else max(values.values())
        tolerance = max(1e-12, abs(float(target)) * 1e-9)
        spread = max(values.values()) - min(values.values())
        if spread <= tolerance:
            saturated.append(
                {
                    "metric": metric,
                    "row_key": row_key,
                    "values": values,
                    "reason": "zero spread across systems",
                }
            )
            continue
        winners = [
            system for system, value in values.items() if abs(float(value) - target) <= tolerance
        ]
        credit = 1.0 / len(winners)
        for system in winners:
            wins[system] += credit
        rankings.append(
            {
                "metric": metric,
                "row_key": row_key,
                "direction": direction,
                "values": values,
                "winners": winners,
                "spread": spread,
            }
        )
    return {
        "evaluated_metric_count": len(rankings),
        "excluded_saturated_metric_count": len(saturated),
        "excluded_saturated_metrics": saturated,
        "win_credit": wins,
        "rankings": rankings,
    }


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
    seen: set[tuple[str, str]] = set()
    seed_dir = f"seed_{seed:03d}"
    for submission_path in sorted(submissions_root.glob("**/submission.json")):
        if submission_path.parent.name != seed_dir:
            continue
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
            plan_vlm_path = planning_path.parent / "advanced_plan_metrics.json"
            plan_vlm = _json(plan_vlm_path) if plan_vlm_path.is_file() else {}
            media_path = submission_path.parent / "media_metrics.json"
            media = _json(media_path) if media_path.is_file() else {}
            advanced_path = submission_path.parent / "advanced_media_metrics.json"
            advanced = _json(advanced_path) if advanced_path.is_file() else {}
            vlm_path = submission_path.parent / "vlm_media_metrics.json"
            vlm = _json(vlm_path) if vlm_path.is_file() else {}
            panel_path = submission_path.parent / "vlm_panel_metrics.json"
            panel = _json(panel_path) if panel_path.is_file() else {}
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
                "plan_vlm_weighted_overall_score": plan_vlm.get(
                    "weighted_overall_score"
                ),
                "plan_vlm_semantic_role_fidelity": plan_vlm.get("scores", {}).get(
                    "semantic_role_fidelity"
                ),
                "plan_vlm_observable_action_grounding": plan_vlm.get(
                    "scores", {}
                ).get("observable_action_grounding"),
                "plan_vlm_narrative_progression": plan_vlm.get("scores", {}).get(
                    "narrative_progression"
                ),
                "plan_vlm_cross_scene_continuity": plan_vlm.get("scores", {}).get(
                    "cross_scene_continuity"
                ),
                "plan_vlm_shot_design": plan_vlm.get("scores", {}).get("shot_design"),
                "plan_vlm_safety_adaptation": plan_vlm.get("scores", {}).get(
                    "safety_adaptation"
                ),
                "plan_vlm_renderability": plan_vlm.get("scores", {}).get(
                    "renderability"
                ),
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
                "advanced_semantic_alignment_mean": advanced.get(
                    "semantic_alignment_mean"
                ),
                "advanced_semantic_contrastive_margin_mean": advanced.get(
                    "semantic_contrastive_margin_mean"
                ),
                "advanced_semantic_retrieval_mrr": advanced.get(
                    "semantic_retrieval_mrr"
                ),
                "advanced_dense_frame_margin_mean": advanced.get(
                    "dense_frame_margin_mean"
                ),
                "advanced_dense_frame_margin_p10": advanced.get(
                    "dense_frame_margin_p10"
                ),
                "advanced_dinov2_within_scene_consistency": advanced.get(
                    "dinov2_temporal", {}
                ).get("within_scene_embedding_consistency_mean"),
                "advanced_dinov2_scene_change_separation": advanced.get(
                    "dinov2_temporal", {}
                ).get("scene_change_separation"),
                "advanced_flow_warp_error_mean": advanced.get(
                    "flow_temporal", {}
                ).get("flow_warp_error_mean"),
                "vlm_weighted_overall_score": vlm.get("weighted_overall_score"),
                "vlm_semantic_grounding": vlm.get("scores", {}).get(
                    "semantic_grounding"
                ),
                "vlm_narrative_progression": vlm.get("scores", {}).get(
                    "narrative_progression"
                ),
                "vlm_subject_identity_consistency": vlm.get("scores", {}).get(
                    "subject_identity_consistency"
                ),
                "vlm_temporal_coherence": vlm.get("scores", {}).get(
                    "temporal_coherence"
                ),
                "vlm_visual_quality": vlm.get("scores", {}).get("visual_quality"),
                "panel_weighted_overall_score": panel.get("weighted_overall_score"),
                "panel_weighted_rank_points": panel.get("weighted_rank_points"),
                "panel_semantic_grounding": panel.get("scores", {}).get(
                    "semantic_grounding"
                ),
                "panel_narrative_progression": panel.get("scores", {}).get(
                    "narrative_progression"
                ),
                "panel_subject_identity_consistency": panel.get("scores", {}).get(
                    "subject_identity_consistency"
                ),
                "panel_temporal_coherence": panel.get("scores", {}).get(
                    "temporal_coherence"
                ),
                "panel_visual_quality": panel.get("scores", {}).get("visual_quality"),
                "notes": submission.get("notes", ""),
            }
        )
        seen.add((submission["system"], submission["case_id"]))
    for planning_path in sorted(
        plans_root.glob(f"*/*/seed_{seed:03d}/planning_metrics.json")
    ):
        system = planning_path.parents[2].name
        case_id = planning_path.parents[1].name
        if (system, case_id) in seen:
            continue
        try:
            planning = _json(planning_path)
            plan_vlm_path = planning_path.parent / "advanced_plan_metrics.json"
            plan_vlm = _json(plan_vlm_path) if plan_vlm_path.is_file() else {}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(planning_path), "error": str(exc)})
            continue
        rows.append(
            {
                "system": system,
                "case_id": case_id,
                "delivery_pass": None,
                "duration_error_seconds": None,
                "audio_checksum_matches_manifest": None,
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
                "safety_language_scene_rate": planning.get(
                    "safety_language_scene_rate"
                ),
                "unsafe_term_occurrences": planning.get("unsafe_term_occurrences"),
                "schema_placeholder_occurrences": planning.get(
                    "schema_placeholder_occurrences"
                ),
                "agent_calls": planning.get("agent_calls"),
                "planning_seconds": planning.get("planning_seconds"),
                "plan_vlm_weighted_overall_score": plan_vlm.get(
                    "weighted_overall_score"
                ),
                "plan_vlm_semantic_role_fidelity": plan_vlm.get("scores", {}).get(
                    "semantic_role_fidelity"
                ),
                "plan_vlm_observable_action_grounding": plan_vlm.get(
                    "scores", {}
                ).get("observable_action_grounding"),
                "plan_vlm_narrative_progression": plan_vlm.get("scores", {}).get(
                    "narrative_progression"
                ),
                "plan_vlm_cross_scene_continuity": plan_vlm.get("scores", {}).get(
                    "cross_scene_continuity"
                ),
                "plan_vlm_shot_design": plan_vlm.get("scores", {}).get("shot_design"),
                "plan_vlm_safety_adaptation": plan_vlm.get("scores", {}).get(
                    "safety_adaptation"
                ),
                "plan_vlm_renderability": plan_vlm.get("scores", {}).get(
                    "renderability"
                ),
                "generation_seconds": None,
                "clip_assigned_scene_similarity_mean": None,
                "clip_global_prompt_similarity_mean": None,
                "clip_lyric_retrieval_order_accuracy": None,
                "clip_adjacent_scene_similarity_mean": None,
                "advanced_semantic_alignment_mean": None,
                "advanced_semantic_contrastive_margin_mean": None,
                "advanced_semantic_retrieval_mrr": None,
                "advanced_dense_frame_margin_mean": None,
                "advanced_dense_frame_margin_p10": None,
                "advanced_dinov2_within_scene_consistency": None,
                "advanced_dinov2_scene_change_separation": None,
                "advanced_flow_warp_error_mean": None,
                "vlm_weighted_overall_score": None,
                "vlm_semantic_grounding": None,
                "vlm_narrative_progression": None,
                "vlm_subject_identity_consistency": None,
                "vlm_temporal_coherence": None,
                "vlm_visual_quality": None,
                "panel_weighted_overall_score": None,
                "panel_weighted_rank_points": None,
                "panel_semantic_grounding": None,
                "panel_narrative_progression": None,
                "panel_subject_identity_consistency": None,
                "panel_temporal_coherence": None,
                "panel_visual_quality": None,
                "notes": (
                    "Planning-only result; no validated media submission is available "
                    "for this system and case."
                ),
            }
        )
    advanced = manifest.get("advanced_evaluation", {})
    media_summary = _ranking_summary(rows, advanced.get("primary_metrics", {}))
    plan_summary = _ranking_summary(rows, advanced.get("plan_primary_metrics", {}))
    return {
        "benchmark_id": manifest["benchmark_id"],
        "seed": seed,
        "rows": sorted(rows, key=lambda row: row["system"]),
        "validation_errors": errors,
        "advanced_primary_summary": media_summary,
        "plan_primary_summary": plan_summary,
        "interpretation": (
            "Planning values are contract/proxy metrics; delivery values are measured from media. "
            "Advanced summaries exclude zero-spread metrics and do not interpret them as wins. "
            "VLM panels are deterministic model judgments, not blinded human perceptual quality."
        ),
    }


def render_comparison_markdown(report: dict[str, Any]) -> str:
    columns = (
        "system",
        "delivery_pass",
        "duration_error_seconds",
        "planning_seconds",
        "plan_vlm_weighted_overall_score",
        "plan_vlm_semantic_role_fidelity",
        "plan_vlm_observable_action_grounding",
        "plan_vlm_narrative_progression",
        "plan_vlm_cross_scene_continuity",
        "plan_vlm_shot_design",
        "plan_vlm_safety_adaptation",
        "plan_vlm_renderability",
        "generation_seconds",
        "advanced_semantic_contrastive_margin_mean",
        "advanced_semantic_retrieval_mrr",
        "advanced_dense_frame_margin_mean",
        "advanced_dense_frame_margin_p10",
        "advanced_dinov2_within_scene_consistency",
        "advanced_dinov2_scene_change_separation",
        "advanced_flow_warp_error_mean",
        "panel_weighted_overall_score",
        "panel_weighted_rank_points",
        "panel_semantic_grounding",
        "panel_narrative_progression",
        "panel_subject_identity_consistency",
        "panel_temporal_coherence",
        "panel_visual_quality",
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
