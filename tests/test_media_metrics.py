from __future__ import annotations

import pytest
import torch

from storymem_agentic.benchmark.advanced_media import (
    _bootstrap_ci,
    contrastive_semantic_metrics,
    temporal_embedding_metrics,
)
from storymem_agentic.benchmark.media import _case
from storymem_agentic.benchmark.vlm_media import (
    _extract_json,
    validate_rubric_scores,
    weighted_rubric_score,
)
from storymem_agentic.benchmark.vlm_plan import (
    PLAN_RUBRIC_WEIGHTS,
    _normalized_scenes,
    validate_plan_rubric,
)
from storymem_agentic.benchmark.vlm_plan_panel import validate_plan_panel
from storymem_agentic.benchmark.vlm_panel import (
    aggregate_panel_runs,
    validate_panel_result,
)


def test_media_case_selects_locked_case() -> None:
    manifest = {"cases": [{"case_id": "one"}, {"case_id": "two", "value": 2}]}
    assert _case(manifest, "two") == {"case_id": "two", "value": 2}
    with pytest.raises(ValueError, match="unknown benchmark case"):
        _case(manifest, "missing")


def test_contrastive_metrics_reward_correct_non_saturated_retrieval() -> None:
    text = torch.eye(3)
    scenes = torch.tensor(
        [[0.9, 0.1, 0.0], [0.2, 0.8, 0.1], [0.4, 0.5, 0.6]], dtype=torch.float32
    )
    frames = torch.stack([scenes, scenes * 0.95], dim=1)

    result = contrastive_semantic_metrics(scenes, frames, text)

    assert result["semantic_retrieval_recall_at_1"] == 1.0
    assert result["semantic_retrieval_mrr"] == 1.0
    assert result["semantic_contrastive_margin_mean"] == pytest.approx(0.5)
    assert result["semantic_contrastive_margin_min"] == pytest.approx(0.1)
    assert len(result["semantic_contrastive_margin_ci95"]) == 2


def test_contrastive_metrics_expose_wrong_scene_rank() -> None:
    text = torch.eye(3)
    scenes = torch.tensor(
        [[0.9, 0.1, 0.0], [0.8, 0.2, 0.1], [0.1, 0.2, 0.9]], dtype=torch.float32
    )
    frames = torch.stack([scenes, scenes], dim=1)

    result = contrastive_semantic_metrics(scenes, frames, text)

    assert result["semantic_retrieval_recall_at_1"] == pytest.approx(2 / 3)
    assert result["semantic_retrieval_mrr"] == pytest.approx(5 / 6)
    assert result["per_scene_reference_rank"] == [1, 2, 1]
    assert result["semantic_contrastive_margin_min"] < 0


def test_temporal_metrics_separate_stability_from_scene_change() -> None:
    frames = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    result = temporal_embedding_metrics(frames)
    assert result["within_scene_embedding_consistency_mean"] == pytest.approx(1.0)
    assert result["adjacent_scene_embedding_similarity_mean"] == pytest.approx(0.0)
    assert result["scene_change_separation"] == pytest.approx(1.0)


def test_bootstrap_ci_is_deterministic() -> None:
    assert _bootstrap_ci([0.1, 0.2, 0.3]) == _bootstrap_ci([0.1, 0.2, 0.3])


def test_vlm_rubric_parsing_and_weighting() -> None:
    names = (
        "semantic_grounding",
        "narrative_progression",
        "subject_identity_consistency",
        "setting_continuity",
        "temporal_coherence",
        "visual_quality",
        "lullaby_suitability",
        "safety",
    )
    parsed = _extract_json("```json\n" + __import__("json").dumps(
        {"scores": {name: 80 for name in names}, "evidence": {}}
    ) + "\n```")
    scores = validate_rubric_scores(parsed)
    assert weighted_rubric_score(scores) == pytest.approx(80.0)


def test_vlm_rubric_rejects_saturated_out_of_range_value() -> None:
    scores = {
        name: 80
        for name in (
            "semantic_grounding",
            "narrative_progression",
            "subject_identity_consistency",
            "setting_continuity",
            "temporal_coherence",
            "visual_quality",
            "lullaby_suitability",
            "safety",
        )
    }
    scores["safety"] = 101
    with pytest.raises(ValueError, match="between 0 and 100"):
        validate_rubric_scores({"scores": scores})


def test_plan_judge_uses_only_equal_normalized_fields() -> None:
    plan = {
        "private_agent_metadata": "ignored",
        "scenes": [
            {
                "lyric_line": "line one",
                "video_prompts": ["visible action"],
                "hidden_reasoning": "ignored",
            }
        ],
    }
    assert _normalized_scenes(plan, 1) == [
        {"scene_num": 1, "lyric_line": "line one", "render_prompt": "visible action"}
    ]


def test_plan_rubric_validation_is_continuous() -> None:
    expected = {
        name: 70.5 + index
        for index, name in enumerate(PLAN_RUBRIC_WEIGHTS)
    }
    assert validate_plan_rubric({"scores": expected}) == expected


def test_plan_rubric_rejects_default_score_collapse() -> None:
    with pytest.raises(ValueError, match="at least three distinct"):
        validate_plan_rubric(
            {"scores": {name: 60 for name in PLAN_RUBRIC_WEIGHTS}}
        )


def test_panel_validation_preserves_evidence_justified_ties() -> None:
    value = {
        dimension: {
            "scores": {"A": 81, "B": 73, "C": 64},
            "ranking": ["A", "B", "C"],
            "evidence": "A is visibly stronger",
        }
        for dimension in (
            "semantic_grounding",
            "narrative_progression",
            "subject_identity_consistency",
            "setting_continuity",
            "temporal_coherence",
            "visual_quality",
        )
    }
    assert validate_panel_result(value)["semantic_grounding"]["scores"]["A"] == 81
    value["visual_quality"]["scores"] = {"A": 80, "B": 80, "C": 70}
    value["visual_quality"]["ranking"] = ["A, B", "C"]
    result = validate_panel_result(value)
    assert result["visual_quality"]["ranking_groups"] == [["A", "B"], ["C"]]


def test_panel_aggregation_unblinds_and_averages_positions() -> None:
    dimensions = (
        "semantic_grounding",
        "narrative_progression",
        "subject_identity_consistency",
        "setting_continuity",
        "temporal_coherence",
        "visual_quality",
    )
    runs = []
    for mapping in (
        {"A": "ours", "B": "auto", "C": "movie"},
        {"A": "auto", "B": "movie", "C": "ours"},
        {"A": "movie", "B": "ours", "C": "auto"},
    ):
        label_for = {system: label for label, system in mapping.items()}
        scores = {
            label_for["ours"]: 90,
            label_for["auto"]: 70,
            label_for["movie"]: 50,
        }
        ranking = sorted(scores, key=scores.get, reverse=True)
        runs.append(
            {
                "label_to_system": mapping,
                "result": {
                    dimension: {"scores": scores, "ranking": ranking, "evidence": ""}
                    for dimension in dimensions
                },
            }
        )
    result = aggregate_panel_runs(["ours", "auto", "movie"], runs)
    assert result["ours"]["weighted_overall_score"] == pytest.approx(90)
    assert result["ours"]["weighted_rank_points"] == pytest.approx(2)
    assert result["movie"]["weighted_rank_points"] == pytest.approx(0)


def test_panel_aggregation_assigns_fractional_tie_rank_points() -> None:
    dimensions = (
        "semantic_grounding",
        "narrative_progression",
        "subject_identity_consistency",
        "setting_continuity",
        "temporal_coherence",
        "visual_quality",
    )
    run = {
        "label_to_system": {"A": "ours", "B": "auto", "C": "movie"},
        "result": {
            dimension: {
                "scores": {"A": 80, "B": 80, "C": 70},
                "ranking_groups": [["A", "B"], ["C"]],
                "evidence": "",
            }
            for dimension in dimensions
        },
    }
    result = aggregate_panel_runs(["ours", "auto", "movie"], [run])
    assert result["ours"]["weighted_rank_points"] == pytest.approx(1.5)
    assert result["auto"]["weighted_rank_points"] == pytest.approx(1.5)
    assert result["movie"]["weighted_rank_points"] == pytest.approx(0)


def test_plan_panel_requires_distinct_scores_and_consistent_rank() -> None:
    value = {
        dimension: {
            "scores": {"A": 88, "B": 72, "C": 51},
            "ranking": ["A", "B", "C"],
            "evidence": "A has the clearest observable beat",
        }
        for dimension in PLAN_RUBRIC_WEIGHTS
    }
    assert validate_plan_panel(value)["semantic_role_fidelity"]["scores"]["A"] == 88
    value["renderability"]["ranking"] = ["B", "A", "C"]
    reconciled = validate_plan_panel(value)["renderability"]
    assert reconciled["score_order_reconciled"] is True
    assert reconciled["scores"] == {"B": 88.0, "A": 72.0, "C": 51.0}


def test_plan_panel_accepts_positional_score_arrays() -> None:
    value = {
        dimension: {
            "scores": [88, 72, 51],
            "ranking": ["A", "B", "C"],
            "evidence": "",
        }
        for dimension in PLAN_RUBRIC_WEIGHTS
    }
    assert validate_plan_panel(value)["shot_design"]["scores"] == {
        "A": 88.0,
        "B": 72.0,
        "C": 51.0,
    }
