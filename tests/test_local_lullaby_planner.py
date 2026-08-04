from __future__ import annotations

from scripts.local_lullaby_planner import apply_binding_replacements


def _revision_map(decision: dict) -> dict[tuple[int, str], object]:
    return {
        (item["scene_num"], item["field_to_change"]): item["replacement_value"]
        for item in decision["scene_revisions"]
    }


def test_binding_replacements_inject_an_omitted_required_action() -> None:
    decision = {
        "scene_revisions": [
            {
                "scene_num": 4,
                "field_to_change": "camera",
                "replacement_value": "A held wide payoff",
            }
        ],
        "plan_updates": {},
    }
    issues = [
        {
            "scene_num": 4,
            "field": "action",
            "replacement_value": "The unchanged star remains visible beside a separate diamond sparkle",
        }
    ]

    result = apply_binding_replacements(decision, issues)

    assert _revision_map(result) == {
        (4, "action"): "The unchanged star remains visible beside a separate diamond sparkle",
        (4, "camera"): "A held wide payoff",
    }


def test_binding_replacements_prefer_first_specific_value_over_stale_diagnostic() -> None:
    decision = {
        "scene_revisions": [
            {
                "scene_num": 2,
                "field_to_change": "relationship_change",
                "replacement_value": ["model guess"],
            }
        ],
        "plan_updates": {},
    }
    issues = [
        {
            "scene_num": 2,
            "field": "action",
            "replacement_fields": {
                "action": "The villager points up at the star",
                "relationship_change": ["the villager points up at the star"],
            },
        },
        {
            "scene_num": 2,
            "field": "relationship_change",
            "replacement_value": ["the star looks curious"],
        },
    ]

    result = apply_binding_replacements(decision, issues)

    assert _revision_map(result)[(2, "relationship_change")] == [
        "the villager points up at the star"
    ]


def test_binding_replacements_preserve_unbound_model_edits_and_ignore_derived_fields() -> None:
    decision = {
        "scene_revisions": [
            {"scene_num": 3, "field_to_change": "camera", "replacement_value": "Low angle"}
        ],
        "plan_updates": {"arc_summary": "model-authored update"},
    }
    issues = [
        {
            "scene_num": 3,
            "replacement_fields": {
                "selected_characters": [{"label": "Star"}],
                "description": "derived text must not be patched",
            },
        }
    ]

    result = apply_binding_replacements(decision, issues)

    assert _revision_map(result) == {
        (3, "selected_characters"): [{"label": "Star"}],
        (3, "camera"): "Low angle",
    }
    assert result["plan_updates"] == {"arc_summary": "model-authored update"}
