from __future__ import annotations

from scripts.run_published_baseline_planner import (
    automv_plan_issues,
    movieagent_shot_issues,
    normalize_story,
    object_values,
    parse_json,
)


def test_parse_json_accepts_fenced_arrays_and_objects() -> None:
    assert parse_json('```json\n[{"number": 1}]\n```') == [{"number": 1}]
    assert parse_json('preface {"Shot": {"Shot 1": {"x": 1}}} suffix') == {
        "Shot": {"Shot 1": {"x": 1}}
    }


def test_object_values_normalizes_movieagent_named_objects() -> None:
    value = {"Shot": {"Shot 1": {"prompt": "one"}, "Shot 2": {"prompt": "two"}}}
    assert object_values(value, ("Shot",)) == [{"prompt": "one"}, {"prompt": "two"}]


def test_normalize_story_writes_storymem_contract() -> None:
    case = {
        "case_id": "tiny",
        "prompt": "A tiny test",
        "lyrics": ["one", "two"],
        "expected_scenes": 2,
    }
    native = {
        "character_bank": [{"name": "star", "visual_description": "round and gold"}],
        "storyboard": [
            {"story": "first action", "camera": "wide"},
            {"story": "second action", "camera": "close"},
        ],
    }

    story = normalize_story("automv", case, native)

    assert story["story_name"] == "benchmark_automv_tiny"
    assert story["scenes"][0] == {
        "scene_num": 1,
        "lyric_line": "one",
        "video_prompts": ["Continuity anchors: star round and gold. first action wide"],
        "first_frame_prompt": ["Continuity anchors: star round and gold. first action wide"],
        "cut": [True],
        "subtitle_text": "one",
    }


def test_automv_plan_issues_rejects_schema_placeholders() -> None:
    case = {
        "lyrics": ["one"],
        "target_duration_seconds": 6,
    }
    native = {
        "character_bank": [
            {"name": "stable identifier", "visual_description": "unchanging appearance"}
        ],
        "storyboard": [
            {"number": 1, "text": "one", "story": "beat", "camera": "wide", "verification": "safe"}
        ],
    }

    issues = automv_plan_issues(native, case)

    assert "character_bank[1].name is a schema placeholder" in issues
    assert "character_bank[1].visual_description is a schema placeholder" in issues


def test_movieagent_shot_issues_requires_one_shot_per_scene() -> None:
    shots = {
        "Shot": {
            "Shot 1": {
                "Plot/Visual Description": "star over village",
                "Shot Type": "wide",
                "Camera Movement": "static",
            },
            "Shot 2": {
                "Plot/Visual Description": "star close-up",
                "Shot Type": "close-up",
                "Camera Movement": "push",
            },
        }
    }

    assert movieagent_shot_issues(shots, {"expected_scenes": 4}) == [
        "shot list has 2 entries; expected exactly 4"
    ]
