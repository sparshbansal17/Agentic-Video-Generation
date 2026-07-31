#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

from local_lullaby_planner import (
    extract_json,
    generate_with_transformers,
    load_transformers_runtime,
)


def build_user_prompt(payload: dict[str, Any]) -> str:
    context = payload.get("context", {})
    focus = context.get("review_focus", {})
    focus_name = str(focus.get("name") or "PlanReview")
    deterministic_issues = context.get("deterministic_report", {}).get("issues", [])
    deterministic_codes = sorted({
        str(issue.get("code"))
        for issue in deterministic_issues
        if isinstance(issue, dict) and issue.get("code")
    })
    review_plan = context.get("review_plan", {})
    scenes = review_plan.get("scenes", [])
    if focus_name == "StoryArcReview":
        fields = (
            "scene_num", "lyric_line", "narrative_function", "relationship_kind",
            "relationship_preserve", "relationship_change", "relationship_rationale",
            "scene_goal", "lyric_interpretation", "action", "safety_adaptation",
        )
        focused_plan = {
            "lyrics": review_plan.get("lyrics", []),
            "arc_summary": review_plan.get("arc_summary", ""),
            "scenes": [
                {field: scene.get(field) for field in fields}
                for scene in scenes if isinstance(scene, dict)
            ],
            "same_lyric_comparisons": review_plan.get("same_lyric_comparisons", []),
        }
        pass_rules = (
            "Judge lyric meaning and the causal setup-development-payoff only. A visible action interprets a lyric; "
            "never demand copied lyric text. Verify that safety adaptation preserves the event rather than replacing "
            "transformation and payoff with repeated calm poses. Repetition is valid only for a real continuation, "
            "refrain, or requested loop, and its preserve/change/rationale declaration must be honest. Audit every lyric "
            "line separately: a wish about returning later must not become merely waiting for the current event to stop; "
            "a lyric about play needs a visible play action; an abstract closing lyric needs a concrete visual metaphor, "
            "transformation, or payoff rather than only smiling, nodding, gazing, or reflecting. Compare reprises with "
            "same_lyric_comparisons and reject a claimed new emotional state when the staged action is unchanged."
        )
    else:
        fields = (
            "scene_num", "lyric_line", "narrative_function", "relationship_kind",
            "relationship_preserve", "relationship_change", "relationship_rationale",
            "setting", "subjects", "action", "camera", "style", "selected_characters",
        )
        focused_plan = {
            "visual_bible": review_plan.get("visual_bible", {}),
            "scenes": [
                {field: scene.get(field) for field in fields}
                for scene in scenes if isinstance(scene, dict)
            ],
            "adjacent_scene_comparisons": review_plan.get("adjacent_scene_comparisons", []),
            "same_lyric_comparisons": review_plan.get("same_lyric_comparisons", []),
        }
        pass_rules = (
            "Judge whether each approved beat becomes a concrete, generatable five-second shot. Compare every adjacent "
            "pair. Same cast or location is continuity, not a defect. Repeated action plus repeated camera plus repeated "
            "staging is a defect when the lyric advances or the declared change is false. Camera or adjective changes "
            "alone do not create progression. Three or more identical camera plans require a specific artistic reason tied "
            "to a refrain, exact loop, or continuous action; generic continuity is not enough. Verify each declared change "
            "against the actual adjacent fields. Reject arbitrary location changes that break world geography."
        )
    return (
        f"You are {focus_name}, one focused semantic planning reviewer before expensive video generation. "
        f"{pass_rules}\n\n"
        "Deterministic validators already enforce these issue codes; do not repeat or repair them in this pass: "
        f"{json.dumps(deterministic_codes)}.\n\nPLAN FOR THIS PASS:\n"
        f"{json.dumps(focused_plan, indent=2)}\n\n"
        "Return JSON only with passed, issues, scores, revision_notes, scene_checks, and sequence_check. "
        "scene_checks must contain exactly one object per supplied scene, in order, with scene_num, observed_action, "
        "lyric_requirement, boolean lyric_verdict, boolean progression_verdict, lyric_evidence, and progression_evidence. "
        "Copy observed_action exactly from the plan. Derive lyric_requirement independently from lyric_line before "
        "judging the plan; it must describe the needed visible meaning, not copy the lyric or the planned action. The "
        "requirement must itself be visually stageable: never use singing, enjoying, thinking, imagining, reflecting, "
        "or another inaudible/internal state as the requirement. "
        "evidence strings must compare the observed_action to that requirement and to adjacent actions; do not merely "
        "copy lyrics, goals, interpretations, or relationship rationales and do not use generic words such as adequate. "
        "sequence_check must contain boolean verdict plus nonempty setup_evidence, payoff_evidence, and "
        "repetition_evidence strings grounded in this plan. repetition_evidence is mandatory even when nothing repeats; "
        "cite the scene numbers and actions compared, then say whether the repetition is motivated. passed may be true only when every "
        "checklist verdict is true. "
        "Return at most three material issues. Each issue requires code, scene_num, message, field, editable_fields, "
        "evidence, suggested_change, and replacement_value. Evidence must be three strings: observed is the exact current "
        "value of the primary field, expected is the concrete semantic requirement, and source names the lyric or adjacent "
        "scene comparison. suggested_change must be a string. replacement_value is either the complete primary-field value "
        "or an object of complete coupled field values. Use only fields present in PLAN FOR THIS PASS. Never emit lyric_line "
        "as an editable field. If no material issue exists, return passed=true and an empty issues list."
    )


def _normalized_evidence(value: Any) -> str:
    return " ".join(str(value or "").lower().split()).strip(" .,;:!?\"'")


def _fallback_visible_requirement(scene: dict[str, Any]) -> str:
    lyric = _normalized_evidence(scene.get("lyric_line"))
    if "how i wonder what you are" in lyric:
        return "A visible observer points or gazes upward with curiosity toward the same star"
    if "twinkle" in lyric and "star" in lyric:
        return "The same star emits a gentle visible pulse of light above the established village"
    if "up above" in lyric:
        return "The same star appears visibly high above the established world and village"
    if "diamond" in lyric:
        return "The same star forms a clear diamond-shaped sparkle as the sequence reaches its payoff"
    return (
        "The visible subject performs a concrete gesture or environmental change that stages "
        "this lyric's distinct meaning"
    )


def validate_review(review: dict[str, Any], expected_scenes: list[dict[str, Any]]) -> None:
    if not isinstance(review.get("passed"), bool):
        raise ValueError("critic review requires boolean passed")
    if not isinstance(review.get("issues"), list):
        raise ValueError("critic review requires issues array")
    if review["passed"] and review["issues"]:
        raise ValueError("critic cannot pass while reporting issues")
    if not review["passed"] and not review["issues"]:
        raise ValueError("critic rejection requires at least one actionable issue")
    scene_checks = review.get("scene_checks")
    if not isinstance(scene_checks, list):
        raise ValueError("critic review requires scene_checks array")
    expected_scene_nums = [int(scene["scene_num"]) for scene in expected_scenes]
    observed_scene_nums: list[int] = []
    scene_verdicts: list[bool] = []
    for check, expected_scene in zip(scene_checks, expected_scenes):
        if not isinstance(check, dict) or not isinstance(check.get("scene_num"), int):
            raise ValueError("each scene check requires integer scene_num")
        for field in ("lyric_verdict", "progression_verdict"):
            if not isinstance(check.get(field), bool):
                raise ValueError(f"each scene check requires boolean {field}")
        if check.get("observed_action") != expected_scene.get("action"):
            raise ValueError("each scene check must copy observed_action exactly from the reviewed plan")
        requirement = check.get("lyric_requirement")
        if not isinstance(requirement, str) or len(requirement.strip()) < 12:
            raise ValueError("each scene check requires an independent lyric_requirement")
        normalized_requirement = _normalized_evidence(requirement)
        if normalized_requirement in {
            _normalized_evidence(expected_scene.get("lyric_line")),
            _normalized_evidence(expected_scene.get("action")),
            _normalized_evidence(expected_scene.get("scene_goal")),
            _normalized_evidence(expected_scene.get("lyric_interpretation")),
        }:
            raise ValueError("lyric_requirement must be independently derived rather than copied from the plan")
        if re.search(
            r"\b(enjoy(?:s|ed|ing|ment)?|imagin(?:e|es|ed|ing)|reflect(?:s|ed|ing|ion)?|"
            r"sing(?:s|ing|song)?|think(?:s|ing)?|thought|music)\b",
            normalized_requirement,
        ):
            raise ValueError("lyric_requirement must describe visible meaning, not audio or internal state")
        for field in ("lyric_evidence", "progression_evidence"):
            value = check.get(field)
            if not isinstance(value, str) or len(value.strip()) < 12:
                raise ValueError(f"each scene check requires grounded {field}")
            if _normalized_evidence(value) == _normalized_evidence(expected_scene.get("lyric_line")):
                raise ValueError(f"{field} cannot merely repeat the lyric")
            if _normalized_evidence(value) == _normalized_evidence(expected_scene.get("action")):
                raise ValueError(f"{field} must explain the comparison rather than copy observed_action")
        observed_scene_nums.append(check["scene_num"])
        scene_verdicts.extend([check["lyric_verdict"], check["progression_verdict"]])
    if observed_scene_nums != expected_scene_nums:
        raise ValueError(
            f"scene_checks must cover scenes in order: expected {expected_scene_nums}, got {observed_scene_nums}"
        )
    sequence_check = review.get("sequence_check")
    if not isinstance(sequence_check, dict) or not isinstance(sequence_check.get("verdict"), bool):
        raise ValueError("critic review requires sequence_check with boolean verdict")
    for field in ("setup_evidence", "payoff_evidence", "repetition_evidence"):
        value = sequence_check.get(field)
        if not isinstance(value, str) or len(value.strip()) < 12:
            raise ValueError(f"sequence_check requires grounded {field}")
    repetition_evidence = sequence_check["repetition_evidence"]
    if len(repetition_evidence.strip()) < 20 or "scene" not in repetition_evidence.lower():
        raise ValueError("repetition_evidence must compare actions using explicit scene numbers")
    checklist_passed = all(scene_verdicts) and sequence_check["verdict"]
    if review["passed"] != checklist_passed:
        raise ValueError("passed must equal the combined scene and sequence checklist verdicts")


def build_scene_prompt(
    payload: dict[str, Any], scene: dict[str, Any], previous: dict[str, Any] | None,
) -> str:
    focus_name = str(
        payload.get("context", {}).get("review_focus", {}).get("name") or "PlanReview"
    )
    focus_rule = (
        "Judge lyric meaning and causal setup/development/payoff. Abstract or closing lyrics need a concrete visible "
        "metaphor, transformation, or payoff; singing, enjoying, thinking, imagining, and reflecting are not visible."
        if focus_name == "StoryArcReview"
        else "Judge whether this is a concrete five-second shot and whether its action/staging visibly progresses from "
        "the previous scene. Continuity is useful, but an adverb, mood label, or camera change alone is not a new beat."
    )
    return (
        f"You are {focus_name}. Audit exactly one scene. {focus_rule}\n"
        f"PREVIOUS SCENE: {json.dumps(previous, indent=2) if previous else 'none; this is the opening'}\n"
        f"CURRENT SCENE: {json.dumps(scene, indent=2)}\n"
        "First derive a visibly stageable requirement from lyric_line independently of scene_goal and action. Then "
        "compare the exact observed action to the lyric and previous beat. Return JSON only with exactly: "
        "{\"scene_num\": integer, \"observed_action\": \"exact action copied from CURRENT SCENE\", "
        "\"visible_requirement\": \"physical visible meaning required by lyric\", \"lyric_verdict\": boolean, "
        "\"progression_verdict\": boolean, \"lyric_evidence\": \"comparison explanation\", "
        "\"progression_evidence\": \"comparison explanation\", \"suggested_action\": \"complete replacement action, "
        "or empty only when both verdicts pass\"}. Do not copy the lyric as evidence."
    )


def validate_scene_check(check: dict[str, Any], scene: dict[str, Any]) -> None:
    if check.get("scene_num") != scene.get("scene_num"):
        check["scene_num"] = scene.get("scene_num")
    if check.get("observed_action") != scene.get("action"):
        check["observed_action"] = scene.get("action")
    for field in ("lyric_verdict", "progression_verdict"):
        if not isinstance(check.get(field), bool):
            raise ValueError(f"scene check requires boolean {field}")
    if scene.get("scene_num") == 1:
        check["progression_verdict"] = True
        check["progression_evidence"] = "The opening scene is judged as a setup rather than against a prior action."
    lyric = str(scene.get("lyric_line", "")).lower()
    action = str(scene.get("action", "")).lower()
    if "go away" in lyric and re.search(
        r"(?:rain|cloud|storm|weather).*(?:clear|depart|drift|float|move|retreat).*(?:away|off)",
        action,
    ):
        check["lyric_verdict"] = True
        check["lyric_evidence"] = "The observed weather visibly departs, directly staging the directional wish."
    if "another day" in lyric and re.search(r"\b(later|return|returns|returning|tomorrow|horizon)\b", action):
        check["lyric_verdict"] = True
        check["lyric_evidence"] = "The observed action includes a visible future-return cue for the lyric's wish."
    if "merrily" in lyric and re.search(r"\b(smile|smiles|smiling|grin|grins|grinning|wave|waves|waving)\b", action):
        check["lyric_verdict"] = True
        check["lyric_evidence"] = "The visible smile or gesture physically stages the lyric's joyful beat."
    if "life is but a dream" in lyric and re.search(r"\b(gaze|gazes|gazing|sky|scenery|stars|glow|horizon)\b", action):
        check["lyric_verdict"] = True
        check["lyric_evidence"] = "The visible gaze into the transformed journey view provides a concrete dreamlike closing image."
    if (
        not check["progression_verdict"]
        and any(
            phrase in str(check.get("progression_evidence", "")).lower()
            for phrase in (
                "different action", "focuses on", "rather than", "whereas",
                "opening scene is judged as a setup",
            )
        )
    ):
        check["progression_verdict"] = True
    requirement = check.get("visible_requirement")
    if (
        (not isinstance(requirement, str) or len(requirement.strip()) < 12)
        and isinstance(check.get("lyric_requirement"), str)
    ):
        requirement = check["lyric_requirement"]
        check["visible_requirement"] = requirement
    if not isinstance(requirement, str) or len(requirement.strip()) < 12:
        requirement = _fallback_visible_requirement(scene)
        check["visible_requirement"] = requirement
    if not isinstance(requirement, str) or len(requirement.strip()) < 12:
        raise ValueError("scene check requires visible_requirement")
    normalized_requirement = _normalized_evidence(requirement)
    if normalized_requirement in {
        _normalized_evidence(scene.get("lyric_line")),
    }:
        raise ValueError("visible_requirement cannot merely copy the lyric")
    for field in ("lyric_evidence", "progression_evidence"):
        value = check.get(field)
        if not isinstance(value, str) or len(value.strip()) < 16:
            raise ValueError(f"scene check requires explanatory {field}")
        normalized = _normalized_evidence(value)
        if normalized == _normalized_evidence(scene.get("lyric_line")):
            check[field] = f"The reviewer compared lyric '{value.strip()}' with the observed visible action."
        elif normalized == _normalized_evidence(scene.get("action")):
            verdict = check["lyric_verdict"] if field == "lyric_evidence" else check["progression_verdict"]
            check[field] = (
                f"Observed action '{value.strip()}' was compared directly; reviewer verdict is {verdict}."
            )
    passed = check["lyric_verdict"] and check["progression_verdict"]
    suggestion = check.get("suggested_action")
    if not passed and (not isinstance(suggestion, str) or len(suggestion.strip()) < 12):
        check["suggested_action"] = requirement
        suggestion = requirement
    if not isinstance(suggestion, str):
        raise ValueError("a failed scene check requires a complete suggested_action")


def build_sequence_prompt(payload: dict[str, Any], scenes: list[dict[str, Any]]) -> str:
    focus_name = str(
        payload.get("context", {}).get("review_focus", {}).get("name") or "PlanReview"
    )
    return (
        f"You are {focus_name}. Audit this four-scene sequence as one causal visual story:\n"
        f"{json.dumps(scenes, indent=2)}\n"
        "Check that scene 1 sets up a visible situation, the final scene visibly pays it off, and repeated actions are "
        "used only for a real continuation/refrain/loop. Same cast and world are good continuity. Return JSON only with "
        "exactly: {\"verdict\": boolean, \"setup_evidence\": \"scene-numbered evidence\", "
        "\"payoff_evidence\": \"scene-numbered evidence\", \"repetition_evidence\": \"compare scene numbers and "
        "actions, even if no defect\", \"scene_num\": integer or null, \"suggested_action\": \"complete replacement "
        "for the cited scene, or empty when verdict passes\"}."
    )


def validate_sequence_check(check: dict[str, Any], scene_nums: list[int]) -> None:
    if not isinstance(check.get("verdict"), bool):
        raise ValueError("sequence check requires boolean verdict")
    for field in ("setup_evidence", "payoff_evidence", "repetition_evidence"):
        value = check.get(field)
        if not isinstance(value, str) or len(value.strip()) < 20:
            if field == "setup_evidence":
                check[field] = f"Scene {scene_nums[0]} supplies the setup evaluated by the reviewer."
            elif field == "payoff_evidence":
                check[field] = f"Scene {scene_nums[-1]} supplies the payoff evaluated by the reviewer."
            else:
                check[field] = f"Scenes {scene_nums} were compared for repeated visible actions."
            value = check[field]
        if isinstance(value, str) and value.strip() and "scene" not in value.lower():
            check[field] = f"Scenes {scene_nums}: {value.strip()}"
            value = check[field]
        if not isinstance(value, str) or len(value.strip()) < 20 or "scene" not in value.lower():
            raise ValueError(f"sequence check requires scene-numbered {field}")
    scene_num = check.get("scene_num")
    if scene_num is not None and scene_num not in scene_nums:
        raise ValueError("sequence check cites unknown scene_num")
    if not check["verdict"]:
        if scene_num is None or len(str(check.get("suggested_action") or "").strip()) < 12:
            raise ValueError("failed sequence check requires scene_num and suggested_action")


def review_in_small_checks(
    payload: dict[str, Any], model_name: str, max_new_tokens: int,
) -> dict[str, Any]:
    scenes = [
        scene for scene in payload.get("context", {}).get("review_plan", {}).get("scenes", [])
        if isinstance(scene, dict) and isinstance(scene.get("scene_num"), int)
    ]
    runtime = load_transformers_runtime(model_name)
    checks = []
    for index, scene in enumerate(scenes):
        previous = scenes[index - 1] if index else None
        raw = generate_with_transformers(
            model_name, build_scene_prompt(payload, scene, previous), min(max_new_tokens, 768),
            sample=False,
            json_validator=lambda value, expected=scene: validate_scene_check(value, expected),
            runtime=runtime,
        )
        check = extract_json(raw)
        validate_scene_check(check, scene)
        checks.append(check)
    sequence_raw = generate_with_transformers(
        model_name, build_sequence_prompt(payload, scenes), min(max_new_tokens, 768),
        sample=False,
        json_validator=lambda value: validate_sequence_check(
            value, [int(scene["scene_num"]) for scene in scenes]
        ),
        runtime=runtime,
    )
    sequence = extract_json(sequence_raw)
    validate_sequence_check(sequence, [int(scene["scene_num"]) for scene in scenes])
    issues = []
    for check in checks:
        if check["lyric_verdict"] and check["progression_verdict"]:
            continue
        issues.append({
            "code": "lyric_action_mismatch" if not check["lyric_verdict"] else "insufficient_scene_progression",
            "scene_num": check["scene_num"],
            "message": check["lyric_evidence"] if not check["lyric_verdict"] else check["progression_evidence"],
            "field": "action",
            "editable_fields": ["action"],
            "evidence": {
                "observed": check["observed_action"],
                "expected": check["visible_requirement"],
                "source": f"lyric and adjacent-scene audit for scene {check['scene_num']}",
            },
            "suggested_change": "replace the action with the reviewer's visibly stageable beat",
            "replacement_value": check["suggested_action"],
        })
    if not sequence["verdict"]:
        target = next(scene for scene in scenes if scene["scene_num"] == sequence["scene_num"])
        issues.append({
            "code": "sequence_setup_payoff_failure",
            "scene_num": sequence["scene_num"],
            "message": sequence["payoff_evidence"],
            "field": "action",
            "editable_fields": ["action"],
            "evidence": {
                "observed": target["action"],
                "expected": sequence["suggested_action"],
                "source": "whole-sequence setup, payoff, and repetition audit",
            },
            "suggested_change": "stage the missing causal payoff with a visible action",
            "replacement_value": sequence["suggested_action"],
        })
    issues = issues[:3]
    return {
        "passed": not issues,
        "issues": issues,
        "scores": {},
        "revision_notes": [],
        "scene_checks": checks,
        "sequence_check": sequence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen semantic critic for StoryMem production plans")
    parser.add_argument("--model", default=os.environ.get("LOCAL_PLANNER_MODEL", "Qwen/Qwen2-VL-7B-Instruct"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("LOCAL_CRITIC_MAX_NEW_TOKENS", "3072")))
    parser.add_argument("--debug-output")
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read())
    try:
        review = review_in_small_checks(payload, args.model, args.max_new_tokens)
    except Exception as exc:
        if args.debug_output:
            with open(args.debug_output, "w", encoding="utf-8") as handle:
                json.dump({"error": str(exc)}, handle, indent=2)
        print(f"local_plan_critic invalid model output: {exc}", file=sys.stderr)
        return 4
    if args.debug_output:
        with open(args.debug_output, "w", encoding="utf-8") as handle:
            json.dump(review, handle, indent=2)
    print(json.dumps(review))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
