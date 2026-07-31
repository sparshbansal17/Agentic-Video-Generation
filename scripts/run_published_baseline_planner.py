#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.local_lullaby_planner import generate_with_transformers, load_transformers_runtime


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [index for token in ("{", "[") if (index := cleaned.find(token)) >= 0]
        if not starts:
            raise
        start = min(starts)
        closing = "}" if cleaned[start] == "{" else "]"
        end = cleaned.rfind(closing)
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_movieagent_prompts(repo: Path) -> dict[str, str]:
    source = repo / "movie_agent" / "system_prompts.py"
    spec = importlib.util.spec_from_file_location("published_movieagent_prompts", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MovieAgent prompts: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.sys_prompts)


def case_from_manifest(manifest: Path, case_id: str) -> dict[str, Any]:
    value = load_json(manifest)
    for case in value.get("cases", []):
        if case.get("case_id") == case_id:
            return dict(case)
    raise ValueError(f"unknown benchmark case: {case_id}")


def timed_lines(case: dict[str, Any]) -> list[dict[str, Any]]:
    lyrics = list(case["lyrics"])
    duration = float(case["target_duration_seconds"])
    step = duration / len(lyrics)
    return [
        {
            "number": index,
            "start": round((index - 1) * step, 2),
            "end": round(index * step, 2),
            "text": lyric,
        }
        for index, lyric in enumerate(lyrics, start=1)
    ]


def run_json_model(
    *,
    runtime: tuple[Any, Any, Any],
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> tuple[Any, str, float]:
    started = time.monotonic()
    raw = generate_with_transformers(
        model_name,
        user_prompt,
        max_new_tokens,
        system_prompt=system_prompt,
        runtime=runtime,
    )
    elapsed = time.monotonic() - started
    return parse_json(raw), raw, elapsed


def automv_plan(
    case: dict[str, Any], runtime: tuple[Any, Any, Any], model_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = timed_lines(case)
    system_prompt = (
        "You are AutoMV's screenwriter and director agents. Produce strict JSON only. Design a "
        "cohesive full-song music-video storyboard, a persistent character bank, camera coverage, "
        "and child-safe verification notes. Preserve every supplied lyric and timestamp exactly."
    )
    user_prompt = f"""
The locked benchmark input is:
{json.dumps({"prompt": case["prompt"], "timed_lyrics": lines, "required_entities": case.get("required_entities", []), "stressors": case.get("stressors", [])}, indent=2)}

Return exactly this JSON shape:
{{
  "character_bank": [{{"name": "stable identifier", "visual_description": "unchanging appearance", "role": "story role"}}],
  "storyboard": [
    {{"number": 1, "start": 0.0, "end": 6.0, "label": "story", "text": "exact lyric", "story": "cinematic visual beat", "camera": "shot size and one camera movement", "verification": "alignment, safety, and continuity checks"}}
  ]
}}
The storyboard must contain exactly {len(lines)} entries, one for each timed lyric in order. Use
one readable action per entry, alternate shot scale, keep recurring entities visually identical,
avoid generated text and frightening imagery, and make the last entry a visible narrative payoff.
"""
    parsed, raw, elapsed = run_json_model(
        runtime=runtime,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=2600,
    )
    if not isinstance(parsed, dict) or not isinstance(parsed.get("storyboard"), list):
        raise ValueError("AutoMV adapter did not return a storyboard")
    trace = [{"agent": "AutoMVScreenwriterDirector", "elapsed_seconds": elapsed, "raw": raw}]
    return parsed, trace


def automv_plan_issues(native: dict[str, Any], case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    bank = object_values(native.get("character_bank"), ())
    placeholders = {"stable identifier", "unchanging appearance", "story role", "character"}
    if not bank:
        issues.append("character_bank is empty")
    for index, entry in enumerate(bank, 1):
        name = str(entry.get("name", "")).strip().lower()
        description = str(entry.get("visual_description", "")).strip().lower()
        if not name or name in placeholders:
            issues.append(f"character_bank[{index}].name is a schema placeholder")
        if not description or description in placeholders:
            issues.append(f"character_bank[{index}].visual_description is a schema placeholder")
    lines = timed_lines(case)
    storyboard = object_values(native.get("storyboard"), ())
    if len(storyboard) != len(lines):
        issues.append(f"storyboard has {len(storyboard)} entries; expected {len(lines)}")
    for expected, item in zip(lines, storyboard):
        if item.get("number") != expected["number"]:
            issues.append(f"shot {expected['number']} has the wrong sequence number")
        if item.get("start") != expected["start"] or item.get("end") != expected["end"]:
            issues.append(f"shot {expected['number']} does not preserve the locked time window")
        if item.get("text") != expected["text"]:
            issues.append(f"shot {expected['number']} does not preserve the exact lyric")
        for field in ("story", "camera", "verification"):
            if not str(item.get(field, "")).strip():
                issues.append(f"shot {expected['number']} has an empty {field}")
    return issues


def revise_automv_plan(
    *,
    native: dict[str, Any],
    case: dict[str, Any],
    runtime: tuple[Any, Any, Any],
    model_name: str,
    trace: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    system_prompt = (
        "You are AutoMV's verifier and director. Return strict corrected JSON only. Preserve valid "
        "storyboard work while correcting every listed contract defect."
    )
    for _attempt in range(2):
        issues = automv_plan_issues(native, case)
        if not issues:
            return native, trace
        request = f"""
The verifier rejected this plan:
{json.dumps(native, ensure_ascii=False, indent=2)}

Defects:
{json.dumps(issues, indent=2)}

Return the complete corrected object with character_bank and storyboard. Use concrete visual
identities grounded in these required entities: {json.dumps(case.get("required_entities", []))}.
Never emit the literal values "stable identifier", "unchanging appearance", "story role", or
other schema/example placeholders. Keep exactly {len(case["lyrics"])} shots and preserve every
locked lyric, timestamp, story beat, camera field, and verification field unless it is defective.
"""
        parsed, raw, elapsed = run_json_model(
            runtime=runtime,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=request,
            max_new_tokens=2200,
        )
        if not isinstance(parsed, dict):
            raise ValueError("AutoMV verifier did not return an object")
        native = parsed
        trace.append({"agent": "AutoMVVerifierRevision", "elapsed_seconds": elapsed, "raw": raw})
    remaining = automv_plan_issues(native, case)
    if remaining:
        raise ValueError(f"AutoMV verifier left contract defects: {remaining}")
    return native, trace


def movieagent_plan(
    case: dict[str, Any],
    repo: Path,
    runtime: tuple[Any, Any, Any],
    model_name: str,
    checkpoint_dir: Path | None = None,
    resume_stages: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = load_movieagent_prompts(repo)
    entities = list(case.get("required_entities", []))
    synopsis = f"{case['prompt']} Exact ordered lyric beats: " + " | ".join(
        f"{index}. {line}" for index, line in enumerate(case["lyrics"], 1)
    )
    trace: list[dict[str, Any]] = []
    partial_trace_path = checkpoint_dir / "movieagent_trace.partial.json" if checkpoint_dir else None
    if resume_stages and partial_trace_path and partial_trace_path.is_file():
        partial = json.loads(partial_trace_path.read_text(encoding="utf-8"))
        if isinstance(partial, list):
            trace = list(partial)

    def save_trace() -> None:
        if partial_trace_path:
            partial_trace_path.write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    screenplay_request = f"""
Script Synopsis: {synopsis}
Character: {json.dumps(entities)}
For this short benchmark, return one compact Sub-Script containing all four lyric beats in order.
Keep exact lyric lines as quoted text and use only the supplied character/entity names.
"""
    screenplay_path = checkpoint_dir / "movieagent_screenplay.json" if checkpoint_dir else None
    if resume_stages and screenplay_path and screenplay_path.is_file():
        screenplay = load_json(screenplay_path)
    else:
        screenplay, raw, elapsed = run_json_model(
            runtime=runtime,
            model_name=model_name,
            system_prompt=prompts["screenwriterCoT-sys"],
            user_prompt=screenplay_request,
            max_new_tokens=1800,
        )
        trace.append({"agent": "MovieAgentScreenwriter", "elapsed_seconds": elapsed, "raw": raw})
        save_trace()
        if screenplay_path:
            screenplay_path.write_text(
                json.dumps(screenplay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    scene_request = f"""
Create exactly four ordered scenes from this benchmark screenplay, one per exact lyric line.
Use only these characters/entities: {json.dumps(entities)}.
Benchmark screenplay JSON:
{json.dumps(screenplay, ensure_ascii=False)}
"""
    scenes_path = checkpoint_dir / "movieagent_scenes.json" if checkpoint_dir else None
    if resume_stages and scenes_path and scenes_path.is_file():
        scenes = load_json(scenes_path)
    else:
        scenes, raw, elapsed = run_json_model(
            runtime=runtime,
            model_name=model_name,
            system_prompt=prompts["ScenePlanningCoT-sys"],
            user_prompt=scene_request,
            max_new_tokens=2200,
        )
        trace.append({"agent": "MovieAgentScenePlanner", "elapsed_seconds": elapsed, "raw": raw})
        save_trace()
        if scenes_path:
            scenes_path.write_text(
                json.dumps(scenes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    shot_request = f"""
Turn the four ordered scenes below into exactly four shots total, one shot per scene and one exact
lyric subtitle per shot. Preserve entity appearance and setting continuity. Use one camera movement
at most, child-safe actions, and no visible generated text.
Scene plan JSON:
{json.dumps(scenes, ensure_ascii=False)}
Exact lyrics:
{json.dumps(case["lyrics"], ensure_ascii=False)}
"""
    shots_path = checkpoint_dir / "movieagent_shots.json" if checkpoint_dir else None
    if resume_stages and shots_path and shots_path.is_file():
        shots = load_json(shots_path)
    else:
        shots, raw, elapsed = run_json_model(
            runtime=runtime,
            model_name=model_name,
            system_prompt=prompts["ShotPlotCreateCoT-sys"],
            user_prompt=shot_request,
            max_new_tokens=2600,
        )
        trace.append({"agent": "MovieAgentShotPlanner", "elapsed_seconds": elapsed, "raw": raw})
        save_trace()
        if shots_path:
            shots_path.write_text(
                json.dumps(shots, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    shots, schema_adapted = canonicalize_movieagent_scene_shots(shots, case)
    if schema_adapted:
        if checkpoint_dir:
            raw_path = checkpoint_dir / "movieagent_shots_raw.json"
            if not raw_path.exists() and shots_path:
                raw_path.write_text(shots_path.read_text(encoding="utf-8"), encoding="utf-8")
        trace.append(
            {
                "agent": "MovieAgentSchemaAdapter",
                "elapsed_seconds": 0.0,
                "raw": "Deterministically mapped Scene Description, Plot, Emotional Tone, "
                "Cinematography Notes, and Lyrics into the published Shot keys.",
            }
        )
    shots, trace = revise_movieagent_shots(
        shots=shots,
        scenes=scenes,
        case=case,
        system_prompt=prompts["ShotPlotCreateCoT-sys"],
        runtime=runtime,
        model_name=model_name,
        trace=trace,
    )
    if shots_path:
        shots_path.write_text(
            json.dumps(shots, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    save_trace()
    return {"screenplay": screenplay, "scenes": scenes, "shots": shots}, trace


def movieagent_shot_issues(shots: Any, case: dict[str, Any]) -> list[str]:
    items = object_values(shots, ("Shot", "shots"))
    expected = int(case["expected_scenes"])
    issues = []
    if len(items) != expected:
        issues.append(f"shot list has {len(items)} entries; expected exactly {expected}")
    for index, item in enumerate(items, 1):
        if not str(item.get("Plot/Visual Description", "")).strip():
            issues.append(f"shot {index} has no Plot/Visual Description")
        if not str(item.get("Shot Type", "")).strip():
            issues.append(f"shot {index} has no Shot Type")
        if not str(item.get("Camera Movement", "")).strip():
            issues.append(f"shot {index} has no Camera Movement")
    return issues


def canonicalize_movieagent_scene_shots(
    shots: Any, case: dict[str, Any]
) -> tuple[Any, bool]:
    """Convert a four-entry MovieAgent Scene response into its Shot field contract.

    Qwen sometimes preserves the upstream ScenePlanning keys at the shot boundary. This adapter
    only renames/splits fields already present in that response; it does not author visual content.
    """
    if not isinstance(shots, dict) or "Shot" in shots or "shots" in shots:
        return shots, False
    items = object_values(shots.get("Scene"), ())
    if len(items) != int(case["expected_scenes"]):
        return shots, False
    canonical: dict[str, Any] = {}
    for index, item in enumerate(items, 1):
        notes = str(item.get("Cinematography Notes", "")).strip()
        parts = [part.strip() for part in notes.split(",", 1)]
        shot_type = parts[0] if parts and parts[0] else "static medium shot"
        movement = parts[1] if len(parts) > 1 and parts[1] else "static camera"
        visual = " ".join(
            str(item.get(field, "")).strip()
            for field in ("Scene Description", "Plot", "Visual Style")
            if str(item.get(field, "")).strip()
        )
        canonical[f"Shot {index}"] = {
            "Involving Characters": item.get("Involving Characters", []),
            "Plot/Visual Description": visual,
            "Coarse Plot": str(item.get("Plot", "")).strip(),
            "Emotional Enhancement": str(item.get("Emotional Tone", "")).strip(),
            "Shot Type": shot_type,
            "Camera Movement": movement,
            "Subtitles": {"Narration": case["lyrics"][index - 1]},
        }
    result = {
        "Internal Chain-of-Thought": shots.get("Internal Chain-of-Thought", {}),
        "Shot": canonical,
    }
    return result, True


def revise_movieagent_shots(
    *,
    shots: Any,
    scenes: Any,
    case: dict[str, Any],
    system_prompt: str,
    runtime: tuple[Any, Any, Any],
    model_name: str,
    trace: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    for _attempt in range(2):
        issues = movieagent_shot_issues(shots, case)
        if not issues:
            return shots, trace
        request = f"""
The MovieAgent shot supervisor rejected this shot list:
{json.dumps(shots, ensure_ascii=False)}

Defects:
{json.dumps(issues)}

Rebuild it as exactly {case["expected_scenes"]} ordered shots total, one shot for each scene and
one for each exact lyric in this order: {json.dumps(case["lyrics"], ensure_ascii=False)}. Preserve
the published Shot JSON fields. Every shot must contain Plot/Visual Description, Coarse Plot,
Emotional Enhancement, Shot Type, Camera Movement, and Subtitles. Keep recurring entity appearance
consistent and use only these scene-plan facts: {json.dumps(scenes, ensure_ascii=False)}
"""
        parsed, raw, elapsed = run_json_model(
            runtime=runtime,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=request,
            max_new_tokens=2600,
        )
        shots = parsed
        trace.append(
            {"agent": "MovieAgentShotSupervisorRevision", "elapsed_seconds": elapsed, "raw": raw}
        )
    remaining = movieagent_shot_issues(shots, case)
    if remaining:
        raise ValueError(f"MovieAgent shot supervisor left contract defects: {remaining}")
    return shots, trace


def object_values(value: Any, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in preferred_keys:
        nested = value.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
        if isinstance(nested, dict):
            return [item for item in nested.values() if isinstance(item, dict)]
    return [item for item in value.values() if isinstance(item, dict)]


def normalize_story(system: str, case: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    if system == "automv":
        items = object_values(native.get("storyboard"), ())
        character_bank = object_values(native.get("character_bank"), ())
        continuity = "; ".join(
            " ".join(
                str(value).strip()
                for value in (entry.get("name"), entry.get("visual_description"))
                if str(value or "").strip()
            )
            for entry in character_bank
        )
        prompts = [
            " ".join(
                str(value).strip()
                for value in (
                    f"Continuity anchors: {continuity}." if continuity else "",
                    item.get("story"),
                    item.get("camera"),
                    item.get("verification"),
                )
                if str(value or "").strip()
            )
            for item in items
        ]
    else:
        shot_root = native.get("shots", {})
        items = object_values(shot_root, ("Shot", "shots"))
        prompts = []
        for item in items:
            characters = item.get("Involving Characters", [])
            if isinstance(characters, dict):
                character_names = list(characters)
            elif isinstance(characters, list):
                character_names = [str(value) for value in characters]
            else:
                character_names = [str(characters)] if str(characters or "").strip() else []
            continuity = ", ".join(name.strip() for name in character_names if name.strip())
            prompts.append(
                " ".join(
                    str(value).strip()
                    for value in (
                        f"Continuity anchors: {continuity}." if continuity else "",
                        item.get("Plot/Visual Description"),
                        item.get("Coarse Plot"),
                        item.get("Emotional Enhancement"),
                        item.get("Shot Type"),
                        item.get("Camera Movement"),
                    )
                    if str(value or "").strip()
                )
            )
    expected = int(case["expected_scenes"])
    if len(prompts) < expected:
        raise ValueError(f"{system} produced {len(prompts)} usable shots; expected {expected}")
    prompts = prompts[:expected]
    return {
        "story_name": f"benchmark_{system}_{case['case_id']}",
        "story_overview": case["prompt"],
        "scenes": [
            {
                "scene_num": index,
                "lyric_line": case["lyrics"][index - 1],
                "video_prompts": [prompt],
                "first_frame_prompt": [prompt],
                "cut": [True],
                "subtitle_text": case["lyrics"][index - 1],
            }
            for index, prompt in enumerate(prompts, start=1)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run published baseline planning on a locked case")
    parser.add_argument("--system", choices=("automv", "movieagent"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument(
        "--refresh-derived",
        action="store_true",
        help="rebuild StoryMem JSON from an existing native_plan.json without model inference",
    )
    parser.add_argument(
        "--resume-stages",
        action="store_true",
        help="reuse completed MovieAgent stage checkpoints from the output directory",
    )
    parser.add_argument(
        "--revise-existing",
        action="store_true",
        help="run verifier correction on an existing native plan without repeating initial planning",
    )
    args = parser.parse_args()

    case = case_from_manifest(args.manifest, args.case_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_derived:
        native = load_json(args.output_dir / "native_plan.json")
        story = normalize_story(args.system, case, native)
        (args.output_dir / "storymem_story.json").write_text(
            json.dumps(story, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(args.output_dir / "storymem_story.json")
        return 0
    runtime = load_transformers_runtime(args.model)
    if args.system == "automv":
        if args.revise_existing:
            native = load_json(args.output_dir / "native_plan.json")
            trace_value = json.loads(
                (args.output_dir / "planning_trace.json").read_text(encoding="utf-8")
            )
            trace = list(trace_value) if isinstance(trace_value, list) else []
        else:
            native, trace = automv_plan(case, runtime, args.model)
        native, trace = revise_automv_plan(
            native=native,
            case=case,
            runtime=runtime,
            model_name=args.model,
            trace=trace,
        )
        source = args.external_repo / "picture_generate" / "picture.py"
    else:
        if args.revise_existing:
            raise ValueError("--revise-existing is currently supported only for AutoMV")
        native, trace = movieagent_plan(
            case,
            args.external_repo,
            runtime,
            args.model,
            checkpoint_dir=args.output_dir,
            resume_stages=args.resume_stages,
        )
        source = args.external_repo / "movie_agent" / "system_prompts.py"
    story = normalize_story(args.system, case, native)
    provenance = {
        "system": args.system,
        "case_id": args.case_id,
        "model": args.model,
        "external_repo": str(args.external_repo.resolve()),
        "source_contract": str(source.resolve()),
        "source_sha256": source_digest(source),
        "agent_calls": len(trace),
        "planning_seconds": sum(item["elapsed_seconds"] for item in trace),
    }
    for name, value in (
        ("native_plan.json", native),
        ("storymem_story.json", story),
        ("planning_trace.json", trace),
        ("provenance.json", provenance),
    ):
        (args.output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
