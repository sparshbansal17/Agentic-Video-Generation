from __future__ import annotations

import json
from pathlib import Path

import pytest

from storymem_agentic.benchmark.aggregate import aggregate_advanced_results


def _write_seed(root: Path, system: str, seed: int, margin: float, panel: float) -> None:
    destination = root / system / "case" / f"seed_{seed:03d}"
    destination.mkdir(parents=True)
    (destination / "advanced_media_metrics.json").write_text(
        json.dumps(
            {
                "semantic_contrastive_margin_mean": margin,
                "semantic_retrieval_mrr": 0.5 + margin,
                "dense_frame_margin_p10": margin - 0.01,
                "dinov2_temporal": {
                    "within_scene_embedding_consistency_mean": 0.9 + margin
                },
                "flow_temporal": {"flow_warp_error_mean": 0.1 - margin},
            }
        )
    )
    (destination / "vlm_panel_metrics.json").write_text(
        json.dumps(
            {
                "weighted_overall_score": panel,
                "scores": {
                    "semantic_grounding": panel - 1,
                    "narrative_progression": panel - 2,
                    "subject_identity_consistency": 80,
                    "temporal_coherence": 80,
                    "visual_quality": 80,
                },
            }
        )
    )


def test_aggregate_advanced_results_counts_seed_wins_and_saturation(tmp_path: Path) -> None:
    systems = ["storymem_agentic", "automv", "movieagent"]
    manifest = {
        "benchmark_id": "test",
        "systems": systems,
        "advanced_evaluation": {
            "primary_metrics": {
                "semantic_contrastive_margin_mean": "higher",
                "flow_warp_error_mean": "lower",
                "panel_weighted_overall_score": "higher",
                "panel_visual_quality": "higher",
            }
        },
    }
    for seed in (0, 1):
        _write_seed(tmp_path, "storymem_agentic", seed, 0.03, 85)
        _write_seed(tmp_path, "automv", seed, 0.02, 75)
        _write_seed(tmp_path, "movieagent", seed, 0.01, 65)

    result = aggregate_advanced_results(
        manifest=manifest, submissions_root=tmp_path, case_id="case", seeds=[0, 1]
    )

    assert result["complete"] is True
    assert result["aggregate_valid_seed_metric_count"] == 6
    assert result["aggregate_win_credit"]["storymem_agentic"] == 6
    assert result["aggregate_win_rate"]["storymem_agentic"] == 1
    assert result["per_seed"][0]["excluded_saturated_metrics"][0]["metric"] == (
        "panel_visual_quality"
    )
    metric = result["metric_summary"]["storymem_agentic"][
        "semantic_contrastive_margin_mean"
    ]
    assert metric["mean"] == pytest.approx(0.03)
    assert metric["sample_std"] == pytest.approx(0)


def test_aggregate_reports_missing_seed_artifacts(tmp_path: Path) -> None:
    manifest = {
        "benchmark_id": "test",
        "systems": ["storymem_agentic", "automv", "movieagent"],
        "advanced_evaluation": {
            "primary_metrics": {"semantic_retrieval_mrr": "higher"}
        },
    }
    result = aggregate_advanced_results(
        manifest=manifest, submissions_root=tmp_path, case_id="case", seeds=[0]
    )
    assert result["complete"] is False
    assert len(result["errors"]) == 3
