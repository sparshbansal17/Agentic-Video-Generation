from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CAMERA_TERMS = (
    "close-up",
    "wide",
    "medium",
    "overhead",
    "low angle",
    "high angle",
    "tracking",
    "pan",
    "tilt",
    "dolly",
    "static",
    "push",
    "orbit",
    "crane",
)
SAFETY_TERMS = ("child-safe", "child safe", "safe", "no text", "no scary", "gentle")
UNSAFE_TERMS = ("blood", "gore", "weapon", "falling", "crashing", "frightening")
PLACEHOLDER_TERMS = (
    "stable identifier",
    "unchanging appearance",
    "story role",
    "character1",
    "description here",
)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _entity_key(entity: str) -> str:
    tokens = _normalized(entity).split()
    return tokens[-1].removesuffix("s") if tokens else ""


def _contains_key(prompt: str, key: str) -> bool:
    return key in {token.removesuffix("s") for token in prompt.split()} if key else False


def score_planning_artifact(
    story: dict[str, Any], case: dict[str, Any], provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score deterministic, auditable properties of a normalized agent plan.

    These are contract/proxy metrics. They intentionally do not claim visual quality before the
    renderer has produced media.
    """
    scenes = story.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []
    prompts = [
        str(scene.get("video_prompts", [""])[0])
        if isinstance(scene, dict) and isinstance(scene.get("video_prompts"), list)
        else ""
        for scene in scenes
    ]
    normalized_prompts = [_normalized(prompt) for prompt in prompts]
    entities = [str(entity) for entity in case.get("required_entities", [])]
    entity_exact_hits = {
        entity: any(_normalized(entity) in prompt for prompt in normalized_prompts)
        for entity in entities
    }
    entity_key_hits = {
        entity: any(_contains_key(prompt, _entity_key(entity)) for prompt in normalized_prompts)
        for entity in entities
    }
    primary = _entity_key(entities[0]) if entities else ""
    camera_vocab = {
        term for prompt in normalized_prompts for term in CAMERA_TERMS if _normalized(term) in prompt
    }
    expected = int(case["expected_scenes"])
    lyrics = list(case["lyrics"])
    lyric_exact = [
        isinstance(scene, dict)
        and scene.get("lyric_line") == lyrics[index]
        and scene.get("subtitle_text") == lyrics[index]
        for index, scene in enumerate(scenes[: len(lyrics)])
    ]
    if len(lyric_exact) < len(lyrics):
        lyric_exact.extend([False] * (len(lyrics) - len(lyric_exact)))
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "metric_scope": "planning_contract_proxy",
        "expected_scenes": expected,
        "actual_scenes": len(scenes),
        "scene_count_accuracy": float(len(scenes) == expected),
        "nonempty_prompt_rate": _rate([bool(prompt.strip()) for prompt in prompts]),
        "exact_lyric_and_subtitle_rate": _rate(lyric_exact),
        "required_entity_exact_phrase_coverage": _rate(list(entity_exact_hits.values())),
        "required_entity_exact_phrase_hits": entity_exact_hits,
        "required_entity_key_token_coverage": _rate(list(entity_key_hits.values())),
        "required_entity_key_token_hits": entity_key_hits,
        "primary_entity_scene_coverage": _rate(
            [_contains_key(prompt, primary) for prompt in normalized_prompts]
        )
        if primary
        else None,
        "camera_vocabulary": sorted(camera_vocab),
        "camera_vocabulary_per_expected_scene": len(camera_vocab) / expected,
        "safety_language_scene_rate": _rate(
            [any(_normalized(term) in prompt for term in SAFETY_TERMS) for prompt in normalized_prompts]
        ),
        "unsafe_term_occurrences": sum(
            prompt.count(_normalized(term))
            for prompt in normalized_prompts
            for term in UNSAFE_TERMS
        ),
        "schema_placeholder_occurrences": sum(
            prompt.count(_normalized(term))
            for prompt in normalized_prompts
            for term in PLACEHOLDER_TERMS
        ),
        "mean_prompt_characters": sum(map(len, prompts)) / len(prompts) if prompts else None,
    }
    if provenance:
        result.update(
            {
                "system": provenance.get("system"),
                "agent_calls": provenance.get("agent_calls"),
                "planning_seconds": provenance.get("planning_seconds"),
                "source_sha256": provenance.get("source_sha256"),
            }
        )
    return result


def write_planning_score(
    *,
    manifest_path: str | Path,
    case_id: str,
    plan_path: str | Path,
    provenance_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    case = next((case for case in manifest["cases"] if case["case_id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown benchmark case: {case_id}")
    story = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    provenance = (
        json.loads(Path(provenance_path).read_text(encoding="utf-8"))
        if provenance_path
        else None
    )
    result = score_planning_artifact(story, case, provenance)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
