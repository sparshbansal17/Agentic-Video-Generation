from __future__ import annotations

import json
import shutil
from pathlib import Path

from storymem_agentic.benchmark.compare import compare_submissions
from storymem_agentic.benchmark.compare import _ranking_summary


def test_compare_submissions_joins_delivery_and_planning(tmp_path: Path) -> None:
    manifest = {
        "benchmark_id": "test",
        "systems": ["storymem_agentic", "automv", "mavin", "movieagent"],
        "cases": [
            {
                "case_id": "case",
                "locked_audio_sha256": "audio-hash",
            }
        ],
    }
    submission_dir = tmp_path / "submissions" / "automv" / "case" / "seed_000"
    submission_dir.mkdir(parents=True)
    (submission_dir / "final.mp4").write_bytes(b"media")
    (submission_dir / "delivery.json").write_text(
        json.dumps(
            {
                "has_video_stream": True,
                "has_audio_stream": True,
                "absolute_duration_error_seconds": 0.0,
                "locked_audio_sha256": "audio-hash",
            }
        )
    )
    (submission_dir / "submission.json").write_text(
        json.dumps(
            {
                "system": "automv",
                "case_id": "case",
                "final_video": "final.mp4",
                "evaluation_report": "delivery.json",
            }
        )
    )
    (submission_dir / "advanced_media_metrics.json").write_text(
        json.dumps(
            {
                "semantic_contrastive_margin_mean": 0.031,
                "semantic_retrieval_mrr": 0.75,
                "dense_frame_margin_mean": 0.02,
                "dense_frame_margin_p10": -0.01,
                "dinov2_temporal": {
                    "within_scene_embedding_consistency_mean": 0.91,
                    "scene_change_separation": 0.08,
                },
                "flow_temporal": {"flow_warp_error_mean": 0.04},
            }
        )
    )
    (submission_dir / "vlm_media_metrics.json").write_text(
        json.dumps(
            {
                "weighted_overall_score": 76.5,
                "scores": {
                    "semantic_grounding": 81,
                    "narrative_progression": 77,
                    "subject_identity_consistency": 74,
                    "temporal_coherence": 71,
                    "visual_quality": 80,
                },
            }
        )
    )
    (submission_dir / "vlm_panel_metrics.json").write_text(
        json.dumps(
            {
                "weighted_overall_score": 78.25,
                "weighted_rank_points": 1.67,
                "scores": {
                    "semantic_grounding": 84,
                    "narrative_progression": 79,
                    "subject_identity_consistency": 77,
                    "temporal_coherence": 72,
                    "visual_quality": 81,
                },
            }
        )
    )
    plan_dir = tmp_path / "plans" / "automv" / "case" / "seed_000"
    plan_dir.mkdir(parents=True)
    (plan_dir / "planning_metrics.json").write_text(
        json.dumps({"scene_count_accuracy": 1.0, "planning_seconds": 12.0})
    )
    (plan_dir / "advanced_plan_metrics.json").write_text(
        json.dumps(
            {
                "weighted_overall_score": 82.25,
                "scores": {
                    "semantic_role_fidelity": 90,
                    "observable_action_grounding": 85,
                    "narrative_progression": 80,
                    "cross_scene_continuity": 79,
                    "shot_design": 77,
                    "safety_adaptation": 88,
                    "renderability": 75,
                },
            }
        )
    )
    shutil.copytree(
        submission_dir,
        tmp_path / "submissions" / "automv" / "case" / "renderer_v1_seed_000",
    )

    report = compare_submissions(
        manifest=manifest,
        submissions_root=tmp_path / "submissions",
        plans_root=tmp_path / "plans",
    )

    assert report["validation_errors"] == []
    assert len(report["rows"]) == 1
    assert report["rows"][0]["delivery_pass"] is True
    assert report["rows"][0]["audio_checksum_matches_manifest"] is True
    assert report["rows"][0]["scene_count_accuracy"] == 1.0
    assert report["rows"][0]["planning_seconds"] == 12.0
    assert report["rows"][0]["advanced_semantic_retrieval_mrr"] == 0.75
    assert report["rows"][0]["advanced_dense_frame_margin_p10"] == -0.01
    assert report["rows"][0]["advanced_dinov2_scene_change_separation"] == 0.08
    assert report["rows"][0]["vlm_weighted_overall_score"] == 76.5
    assert report["rows"][0]["vlm_semantic_grounding"] == 81
    assert report["rows"][0]["plan_vlm_weighted_overall_score"] == 82.25
    assert report["rows"][0]["plan_vlm_semantic_role_fidelity"] == 90
    assert report["rows"][0]["panel_weighted_overall_score"] == 78.25
    assert report["rows"][0]["panel_semantic_grounding"] == 84


def test_compare_submissions_includes_planning_only_systems(tmp_path: Path) -> None:
    manifest = {
        "benchmark_id": "test",
        "systems": ["storymem_agentic", "automv", "movieagent"],
        "cases": [{"case_id": "case", "locked_audio_sha256": "audio-hash"}],
    }
    for system, seconds in (("automv", 12.0), ("movieagent", 34.0)):
        plan_dir = tmp_path / "plans" / system / "case" / "seed_000"
        plan_dir.mkdir(parents=True)
        (plan_dir / "planning_metrics.json").write_text(
            json.dumps(
                {
                    "scene_count_accuracy": 1.0,
                    "planning_seconds": seconds,
                }
            )
        )

    report = compare_submissions(
        manifest=manifest,
        submissions_root=tmp_path / "submissions",
        plans_root=tmp_path / "plans",
    )

    assert [row["system"] for row in report["rows"]] == ["automv", "movieagent"]
    assert all(row["delivery_pass"] is None for row in report["rows"])
    assert all(row["generation_seconds"] is None for row in report["rows"])
    assert report["rows"][1]["planning_seconds"] == 34.0
    assert "Planning-only" in report["rows"][0]["notes"]


def test_ranking_summary_respects_direction_and_fractional_ties() -> None:
    rows = [
        {
            "system": "ours",
            "advanced_semantic_retrieval_mrr": 0.8,
            "advanced_flow_warp_error_mean": 0.02,
            "panel_visual_quality": 80,
        },
        {
            "system": "auto",
            "advanced_semantic_retrieval_mrr": 0.7,
            "advanced_flow_warp_error_mean": 0.02,
            "panel_visual_quality": 80,
        },
        {
            "system": "movie",
            "advanced_semantic_retrieval_mrr": 0.5,
            "advanced_flow_warp_error_mean": 0.04,
            "panel_visual_quality": 80,
        },
    ]
    result = _ranking_summary(
        rows,
        {
            "semantic_retrieval_mrr": "higher",
            "flow_warp_error_mean": "lower",
            "panel_visual_quality": "higher",
        },
    )
    assert result["evaluated_metric_count"] == 2
    assert result["excluded_saturated_metric_count"] == 1
    assert result["excluded_saturated_metrics"][0]["metric"] == "panel_visual_quality"
    assert result["win_credit"] == {"ours": 1.5, "auto": 0.5, "movie": 0.0}
    assert result["rankings"][1]["winners"] == ["ours", "auto"]
