from __future__ import annotations

from storymem_agentic.benchmark.planning import score_planning_artifact


def test_score_planning_artifact_reports_contract_proxies() -> None:
    case = {
        "case_id": "test",
        "expected_scenes": 2,
        "lyrics": ["line one", "line two"],
        "required_entities": ["gold star", "village"],
    }
    story = {
        "scenes": [
            {
                "lyric_line": "line one",
                "subtitle_text": "line one",
                "video_prompts": ["Wide child-safe view of a gold star over a village"],
            },
            {
                "lyric_line": "line two",
                "subtitle_text": "line two",
                "video_prompts": ["Gentle close-up of the same gold star"],
            },
        ]
    }

    score = score_planning_artifact(story, case, {"system": "automv", "agent_calls": 1})

    assert score["scene_count_accuracy"] == 1.0
    assert score["exact_lyric_and_subtitle_rate"] == 1.0
    assert score["required_entity_exact_phrase_coverage"] == 1.0
    assert score["required_entity_key_token_coverage"] == 1.0
    assert score["primary_entity_scene_coverage"] == 1.0
    assert score["camera_vocabulary"] == ["close-up", "wide"]
    assert score["safety_language_scene_rate"] == 1.0
    assert score["schema_placeholder_occurrences"] == 0
