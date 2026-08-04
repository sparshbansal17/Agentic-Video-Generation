#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import re
import sys
from typing import Any, Callable


EDITABLE_SCENE_FIELDS = {
    "scene_goal",
    "lyric_interpretation",
    "setting",
    "subjects",
    "action",
    "camera",
    "style",
    "safety_adaptation",
    "selected_characters",
    "expected_mood",
    "boundary_behavior",
    "cut",
    "narrative_function",
    "relationship_kind",
    "relationship_preserve",
    "relationship_change",
    "relationship_rationale",
}


def apply_binding_replacements(
    decision: dict[str, Any], validation_issues: list[Any]
) -> dict[str, Any]:
    """Merge validator-provided corrections into a model-generated revision patch.

    Validation issues are ordered by specificity.  The first supplied value for a
    scene/field therefore wins when a later, more generic diagnostic also mentions
    that field.  Model-authored edits remain useful for fields without a binding,
    but cannot silently omit or override an exact acceptance criterion.
    """
    if "scene_revisions" not in decision:
        return decision

    bindings: dict[tuple[int, str], dict[str, Any]] = {}
    for issue in validation_issues:
        if not isinstance(issue, dict):
            continue
        scene_num = issue.get("scene_num")
        if not isinstance(scene_num, int):
            continue

        replacement_fields = issue.get("replacement_fields")
        candidates: list[tuple[str, Any]] = []
        if isinstance(replacement_fields, dict):
            candidates.extend(replacement_fields.items())
        field = issue.get("field")
        if isinstance(field, str) and "replacement_value" in issue:
            candidates.append((field, issue["replacement_value"]))

        for field_name, replacement_value in candidates:
            key = (scene_num, field_name)
            if field_name not in EDITABLE_SCENE_FIELDS or key in bindings:
                continue
            bindings[key] = {
                "scene_num": scene_num,
                "field_to_change": field_name,
                "replacement_value": deepcopy(replacement_value),
            }

    model_revisions = decision.get("scene_revisions")
    if not isinstance(model_revisions, list):
        return decision
    retained: list[dict[str, Any]] = []
    for revision in model_revisions:
        if not isinstance(revision, dict):
            retained.append(revision)
            continue
        key = (revision.get("scene_num"), revision.get("field_to_change"))
        if key not in bindings:
            retained.append(revision)

    decision["scene_revisions"] = [*bindings.values(), *retained]
    return decision


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            decoded, _end = json.JSONDecoder().raw_decode(cleaned.lstrip())
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = re.sub(r"}\]\s*,\s*(?=\{\s*\"scene_num\")", "}, ", candidate)
                if repaired != candidate:
                    return json.loads(repaired)
        raise


def build_user_prompt(payload: dict[str, Any]) -> str:
    prompt = str(payload.get("prompt", ""))
    context = payload.get("context", {})
    validation_issues = context.get("validation_issues") or []
    previous_decision = context.get("previous_decision")
    rejected_transactions = context.get("rejected_transactions") or []
    revision_block = ""
    if validation_issues:
        issue_codes = {
            str(issue.get("code", ""))
            for issue in validation_issues
            if isinstance(issue, dict)
        }
        issue_codes.update(
            str(introduced[0])
            for transaction in rejected_transactions
            if isinstance(transaction, dict)
            for introduced in transaction.get("introduced_issues", [])
            if isinstance(introduced, (list, tuple)) and introduced
        )
        unsafe_instruction = ""
        if "unsafe_visual_action" in issue_codes:
            unsafe_instruction = (
                "\nCRITICAL SAFETY REVISION REQUIRED:\n"
                "- For every unsafe_visual_action scene, rewrite scene_goal, lyric_interpretation, action, setting, "
                "subjects, and safety_adaptation.\n"
                "- Use the evidence excerpts in validation_issues to identify the exact hazardous visual wording.\n"
                "- Replace unsupported descent, breakage, dropping, crashing, or impact with a calm safe adaptation "
                "that preserves the lyric's causal meaning without depicting danger.\n"
                "- Compare the neighboring beats and retain a distinct setup, supported transformation, and payoff; "
                "do not turn every affected scene into the same stationary safe pose.\n"
                "- The replacement scene must use only visibly safe supported motion such as supported safely, "
                "gently lowers, floating gently, caught safely, lands safely, or settles safely.\n"
                "- Remove all remaining hazardous wording from setting, subjects, action, lyric_interpretation, "
                "camera, style, and safety_adaptation; do not leave unsafe wording and merely add a negation later.\n"
                "- Do not use fall, falls, falling, drop, break, crash, impact, or synonyms anywhere in replacement "
                "values, even if modified by gently/safely or followed by being caught. The depicted motion itself must "
                "be different, such as remaining securely supported, swaying in place, or floating horizontally.\n"
                "- Patch the editable structured fields action, lyric_interpretation, setting, subjects, camera, and "
                "safety_adaptation as needed; never patch the derived description field.\n"
            )
        diversity_instruction = ""
        if "repeated_scene_staging" in issue_codes:
            diversity_instruction = (
                "\nCRITICAL SCENE-DIVERSITY REVISION REQUIRED:\n"
                "- Rewrite every repeated_scene_staging scene identified by scene_num.\n"
                "- First decide whether the lyric truly calls for a continuation or refrain. If so, declare relationship_kind "
                "continuation or reprise and explain what is intentionally preserved, what story state changes, and why.\n"
                "- Otherwise give it a materially different visible beat and motivated staging/coverage. Never introduce an "
                "unrelated location merely to look different. Preserve character identity and the supplied lyric.\n"
            )
        narrative_instruction = ""
        if issue_codes.intersection({
            "adverb_only_action_change", "directional_wish_not_staged", "future_return_wish_not_staged",
            "meta_action_placeholder", "misstaged_reprise", "named_prop_event_not_staged",
            "repeated_narrative_beat", "reused_nonadjacent_action", "wish_outcome_reversed",
        }):
            narrative_instruction = (
                "\nCRITICAL NARRATIVE-BEAT REVISION REQUIRED:\n"
                "- Use that scene's lyric_line, scene_goal, and lyric_interpretation to create a distinct visible gesture, "
                "expression, interaction, reaction, or payoff, or explicitly justify a lyric-motivated continuation/reprise.\n"
                "- Revise coupled action, staging, coverage, and relationship fields together when needed.\n"
                "- The replacement must name what the visible subject physically does within five seconds.\n"
                "- Compare against every action named in issue evidence, including nonadjacent scenes. Do not copy the "
                "same base action and add an adverb, emotion, singing, thinking, enjoying, or reflecting.\n"
                "- For a joy lyric use a readable physical celebration or interaction. For an abstract or closing lyric, "
                "use a visible prop/environment transformation plus a character reaction that pays off the sequence.\n"
            )
        visual_action_instruction = ""
        if "audio_or_internal_action" in issue_codes or "camera_direction_in_action" in issue_codes:
            visual_action_instruction = (
                "\nCRITICAL VISIBLE-ACTION REVISION REQUIRED:\n"
                "- Edit only each cited action field. Do not return camera edits.\n"
                "- Preserve any physical activity already present, but replace singing, music, thinking, imagining, reflecting, "
                "spoken dialogue, or camera language with a visible pose, gaze, facial expression, or gesture.\n"
                "- When repetition issues are also present, the visible replacement must differ physically from all actions "
                "named in their evidence; never solve visibility by reverting to an earlier repeated action.\n"
            )
        camera_instruction = ""
        if "repeated_camera_coverage" in issue_codes or "undeclared_repeated_coverage" in issue_codes:
            camera_instruction = (
                "\nCRITICAL CAMERA-COVERAGE REVISION REQUIRED:\n"
                "- Edit the camera field of every cited scene. The replacement must not equal the current camera text.\n"
                "- Change the actual coverage: choose a motivated wide establishing view, close-up reaction/detail, "
                "low child-eye angle, overhead composition, gentle reveal, or another clearly different shot size/angle.\n"
                "- Do not merely rephrase the same medium/static/tracking shot. Keep at most one simple movement.\n"
            )
        relationship_instruction = ""
        if "false_relationship_change_claim" in issue_codes:
            relationship_instruction = (
                "\nCRITICAL RELATIONSHIP-HONESTY REVISION REQUIRED:\n"
                "- A relationship_change entry must describe an observable difference from the preceding scene.\n"
                "- If a cited setting, action, or camera truly changes, patch that concrete field. If it is intentionally "
                "preserved, move that fact to relationship_preserve and name the actual changed action/state instead.\n"
                "- Never claim setting, location, camera, or action changed when the corresponding text is identical.\n"
                "- For every cited scene you MUST patch relationship_change itself. When the location is intentionally "
                "continuous, also patch relationship_preserve to name that location continuity, and make "
                "relationship_change a concrete list such as ['wind begins moving the cradle'] or ['the supported cradle "
                "settles onto the cushion']. Do not answer with action-only patches.\n"
            )
        crowd_instruction = ""
        if "overcrowded_five_second_shot" in issue_codes:
            crowd_instruction = (
                "\nCRITICAL CAST-SIMPLIFICATION REVISION REQUIRED:\n"
                "- Keep at most three visible characters in each cited five-second scene.\n"
                "- Choose only characters needed for the lyric and story beat; do not preserve irrelevant bank entries.\n"
                "- Rewrite selected_characters, subjects, setting, and action together so removed characters are not "
                "named, depicted, or assigned actions anywhere in that scene.\n"
                "- Preserve the lead character and the scene's lyric meaning while simplifying the interaction.\n"
            )
        wish_instruction = ""
        if issue_codes.intersection({"directional_wish_not_staged", "future_return_wish_not_staged"}):
            wish_instruction = (
                "\nCRITICAL WISH-STAGING REVISION REQUIRED:\n"
                "- Stage the requested outcome, not activity inside the unwanted condition. For departure, show the "
                "subject making a visible goodbye gesture while the unwanted weather/object visibly drifts away or clears.\n"
                "- For a future-return wish, show the departing object receding toward the horizon while the subject "
                "points toward that later destination; do not show puddle play, dancing, waiting, or only a hopeful look.\n"
                "- The replacement action must literally name the visible departure/clearing and future-return cue.\n"
            )
        revision_block = (
            "\nThis is a revision request. Fix every validation issue below with a targeted patch. "
            "Preserve supplied lyrics exactly and edit only authorized structured fields before prompt compilation.\n"
            f"Validation issues JSON:\n{json.dumps(validation_issues, indent=2)}\n"
            f"Rejected prior revision transactions (do not repeat defects they introduced):\n"
            f"{json.dumps(rejected_transactions, indent=2)}\n"
            f"{unsafe_instruction}{diversity_instruction}{narrative_instruction}{visual_action_instruction}{camera_instruction}{relationship_instruction}{crowd_instruction}{wish_instruction}\n"
            "GENERAL CORRECTION CONTRACT:\n"
            "- Treat every issue object, regardless of code, as a binding acceptance criterion.\n"
            "- When an issue supplies replacement_value, use that exact complete value for its cited field unless a "
            "coupled edit is needed to keep adjacent structured fields consistent.\n"
            "- Use scene_num to edit the affected scene and use field/evidence/message to determine the required change.\n"
            "- Compare the replacement against previous_decision before answering; every cited defect must have a visible JSON-field change.\n"
            "- Do not insert canned topic imagery or substitute a generic bedtime scene unrelated to the supplied input.\n"
        )
        if rejected_transactions:
            revision_block += (
                "Previous patches were rejected transactionally. Do not repeat them; correct the reported reason and "
                "return different, type-compatible values for every still-cited field:\n"
                f"{json.dumps(rejected_transactions, indent=2)}\n"
            )
        if previous_decision:
            revision_block += f"Previous planner decision JSON:\n{json.dumps(previous_decision, indent=2)}\n"
        return (
            "Revise a structured video plan by returning a SMALL JSON PATCH, not the complete plan.\n"
            "Each scene_revisions item must contain exactly scene_num, field_to_change, and replacement_value; "
            "plan_updates must be an object. "
            "Include every field needed to resolve each issue, "
            "but do not include unchanged scenes or explanatory prose. Preserve lyrics and character identity.\n"
            "field_to_change MUST be one of: scene_goal, lyric_interpretation, setting, subjects, action, camera, "
            "style, safety_adaptation, selected_characters, expected_mood, boundary_behavior, cut, narrative_function, "
            "relationship_kind, relationship_preserve, relationship_change, relationship_rationale. Never patch "
            "description, video_prompt, first_frame_prompt, lyric_line, subtitle text, or other derived fields because "
            "they are compiled later and the patch will be ignored. Use multiple patch entries when several structured "
            "fields or scenes must change.\n"
            f"{revision_block}\n"
            "Return only the JSON patch. Verify that each replacement differs from previous_decision and directly "
            "satisfies the corresponding issue message and evidence."
        )
    return (
        "Create the lullaby production-planning JSON now.\n\n"
        "Planning directions:\n"
        f"{prompt}\n\n"
        "User input JSON. Treat null values as not supplied by the user:\n"
        f"{json.dumps(context.get('input', {}), indent=2)}\n\n"
        "Return exactly one JSON object with this shape and concrete values, not a schema:\n"
        "{\n"
        '  "lyrics": ["line 1", "line 2"],\n'
        '  "clip_count": 2,\n'
        '  "target_duration_seconds": 10,\n'
        '  "arc_summary": "specific character action that sets up, develops, and visibly pays off these exact lyrics",\n'
        '  "visual_bible": {"primary_world": "concise world name", "allowed_locations": ["location"]},\n'
        '  "selected_characters": [\n'
        '    {"label": "character_id", "role": "role", "description": "concise consistent visual description", '
        '"selection_rationale": "why this character fits the lyrics"}\n'
        "  ],\n"
        '  "scenes": [\n'
        '    {"scene_num": 1, "lyric_line": "line 1", '
        '"narrative_function": "setup", '
        '"relationship_to_previous": {"kind": "opening", "preserve": [], "change": [], "rationale": "establishes the lyric world"}, '
        '"scene_goal": "what this scene must communicate", '
        '"lyric_interpretation": "visual interpretation of the lyric", '
        '"setting": "specific full-frame setting with foreground and background", '
        '"subjects": "specific selected characters and visible anchors", '
        '"action": "specific child-safe action", '
        '"camera": "specific shot and camera movement", '
        '"style": "visual style, lighting, color palette, mood", '
        '"safety_adaptation": "no text, no dialogue, no inset frame, no scary imagery, safe adaptation details", '
        '"selected_characters": [{"label": "character_id", "selection_rationale": "why used here"}], '
        '"expected_mood": "calm bedtime mood", "boundary_behavior": "hold", "cut": true, "review_status": "pending"}\n'
        "  ],\n"
        '  "music_prompt": "continuous full-song lullaby audio prompt"\n'
        "}\n\n"
        "Rules: if the topic/name is a known traditional or public-domain lullaby, use the appropriate complete "
        "short singable lyrics you know, including repeated closing lines when those are part of the common verse. "
        "If it is an original topic, write complete calm lullaby lyrics. "
        "Create enough scenes to cover every lyric line unless clip_count is explicitly present in the user input. "
        "If clip_count is not supplied, choose 4 to 8 lyric lines and 4 to 8 clips; do not repeat verses just to "
        "increase length. If target_duration_seconds is supplied, choose a clip_count that covers it with about "
        "5 seconds per clip, capped at 12 clips unless the user explicitly requests more. "
        "Each lyric line must be unique unless repetition is essential to the known lullaby ending; never output "
        "the same stanza more than twice. "
        "If the user input includes character_bank_entries, inspect all entries and choose relevant characters. "
        "The bank is a menu, not an ensemble: select the smallest lyric-relevant cast, normally one or two visible "
        "characters per scene, and never place unrelated bank entries into subjects, setting, or action merely because "
        "they are available. Every character named as visible must have matching selected_characters metadata. "
        "Do not use keyword aliases supplied by downstream code; make and justify the selection yourself. "
        "Keep chosen concise visual descriptions unchanged in "
        "every scene where they appear, and only create a missing generic character when no bank entry fits a needed "
        "role. "
        "Direct a polished preschool animated sing-along: colorful, playful, emotionally clear, funny or wondrous when "
        "appropriate, with expressive faces, rhythmic gestures, simple cause-and-effect, and a satisfying final payoff. "
        "Use general preschool entertainment principles without copying any named show, franchise, character, or design. "
        "Follow StoryMem's rule that every clip is one five-second shot: plan one readable action or interaction that can "
        "start, register, and settle in that time, with one to three visible characters and at most one camera movement. "
        "Avoid complex choreography, simultaneous events, extreme motion, text rendering, and audio-dependent visuals. "
        "Build a mini arc: establish, develop/anticipate, react/pay off, then finish on a satisfying image. "
        "State that arc in arc_summary before designing shots. For every scene include narrative_function and "
        "relationship_to_previous with kind, preserve, change, and rationale. Use opening for scene one, then continuation, "
        "reprise, contrast, or payoff. Same cast/location is continuity. A refrain may reprise earlier staging when motivated, "
        "but normally show changed story state or closure unless an exact loop was requested. Reject arbitrary variety that "
        "breaks world geography. "
        "Arc_summary must name concrete characters, events, and the final visual payoff from the supplied lyrics; never "
        "copy generic wording from the JSON format example. Relationship change must name an observable difference, "
        "never 'no change', unless the lyric itself repeats and the user explicitly requests an exact visual loop. "
        "Never emit placeholder metadata values such as 'variant', 'anchor', 'avoidance', or 'constraint'. Omit optional "
        "character metadata when no concrete value is needed. "
        "visual_bible.allowed_locations must contain one to four unique stageable places only. Clock parts, props, "
        "characters, and repeated names are not locations; never enumerate them in allowed_locations. "
        "Follow the Wan prompt recipe for scene planning. Each clip must support the Advanced Formula: subject plus "
        "subject description, scene plus foreground/background description, motion plus motion description, aesthetic "
        "control, and stylization. Plan this like a normal edited video storyboard, not one continuous shot. Make every "
        "clip a distinct full-frame shot: vary location or staging, action, camera angle, shot size, subject distance, "
        "foreground/background layout, lens, composition, color tone, lighting, and motion. Use cut=true for each "
        "lyric-scene clip unless the user explicitly asks for a single continuous one-shot. Do not repeat the same "
        "generic medium-shot camera language for all clips. Use wide coverage to establish relationships, medium coverage "
        "for interactions, close-ups for expressions or props, and moving/reveal angles only when motivated. Preserve "
        "world geography instead of jumping to unrelated locations merely for variety. Each scene description must be a concrete paragraph in "
        "this format: 'Opening shot: [specific setting with foreground/background]. [specific subject] [specific "
        "action]. Camera [specific movement]. [visual style, lighting, color palette, mood]. [continuity and "
        "child-safety constraints].' Do not output vague descriptions like 'child looks at moon' or 'star shines'. "
        "Do not copy literal lyric text into scene descriptions; "
        "describe the visual meaning instead. Put any needed character or setting continuity details directly in the "
        "scene description; do not assume a repeated character-bank prefix will be added later. "
        "Give the lead a clear pose, gaze target, expression, and playful child-readable gesture. Set cut=true for the first "
        "shot and real changes; use cut=false only for a deliberate same-action continuation with compatible subjects, "
        "screen direction, setting, and camera axis. Never describe a small framed box, inset image, picture-in-picture, poster, border, title card, or screen-within-screen. "
        "Adapt unsafe literal rhyme events into calm child-safe visuals. Never show babies, children, characters, "
        "props, vehicles, furniture, or supports falling, breaking, crashing, dropping, or striking the ground. "
        "If validation reports unsafe_visual_action, use the evidence excerpts to rewrite the affected structured "
        "fields into visibly safe supported motion, and remove the hazardous wording instead of only negating it. "
        "Do not include dialogue or background music in visual scenes; audio is generated separately. "
        "Return only valid JSON. Do not return a JSON schema. Do not wrap it in markdown.\n"
        f"{revision_block}"
        "If this is a revision request, the final validation instructions above override the previous decision. "
        "Do not copy an affected scene unchanged. Return the complete corrected JSON object now."
    )


def validate_decision(decision: dict[str, Any]) -> None:
    if "scene_revisions" in decision:
        if not isinstance(decision["scene_revisions"], list):
            raise ValueError("revision patch scene_revisions must be an array")
        if not decision["scene_revisions"] and not decision.get("plan_updates"):
            raise ValueError("revision patch requires a scene revision or plan update")
        if any(not isinstance(item, dict) or not item.get("scene_num") for item in decision["scene_revisions"]):
            raise ValueError("each scene revision requires scene_num and changed fields")
        return
    if decision.get("type") == "object" and "properties" in decision and "lyrics" not in decision:
        raise ValueError("planner returned a JSON schema instead of a planner decision")
    required = ["lyrics", "clip_count", "target_duration_seconds", "arc_summary", "visual_bible", "selected_characters", "scenes", "music_prompt"]
    missing = [key for key in required if key not in decision]
    if missing:
        raise ValueError(f"planner decision missing required keys: {', '.join(missing)}")
    if not isinstance(decision["lyrics"], list) or not [line for line in decision["lyrics"] if str(line).strip()]:
        raise ValueError("planner decision requires non-empty lyrics array")
    if not isinstance(decision["scenes"], list) or not decision["scenes"]:
        raise ValueError("planner decision requires non-empty scenes array")
    if not isinstance(decision["selected_characters"], list) or not decision["selected_characters"]:
        raise ValueError("planner decision requires non-empty selected_characters array")
    visual_bible = decision.get("visual_bible")
    allowed_locations = visual_bible.get("allowed_locations") if isinstance(visual_bible, dict) else None
    if not isinstance(allowed_locations, list) or not 1 <= len(allowed_locations) <= 4:
        raise ValueError("visual_bible requires one to four allowed locations")
    normalized_locations = {str(value).strip().lower() for value in allowed_locations if str(value).strip()}
    if len(normalized_locations) != len(allowed_locations):
        raise ValueError("visual_bible allowed locations must be unique and non-empty")
    placeholder_values = {"anchor", "avoidance", "constraint", "variant"}
    for character in decision["selected_characters"]:
        if not isinstance(character, dict):
            continue
        metadata_values = [
            character.get("allowed_variant"),
            *(character.get("allowed_variants") or []),
            *(character.get("visual_anchors") or []),
            *(character.get("continuity_constraints") or []),
            *(character.get("negative_constraints") or []),
        ]
        if any(str(value).strip().lower() in placeholder_values for value in metadata_values if value is not None):
            raise ValueError("planner decision contains placeholder character metadata")
    required_scene_fields = {
        "scene_goal",
        "narrative_function",
        "relationship_to_previous",
        "lyric_interpretation",
        "setting",
        "subjects",
        "action",
        "camera",
        "style",
        "safety_adaptation",
        "selected_characters",
    }
    for index, scene in enumerate(decision["scenes"], start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"scene {index} must be an object")
        missing_scene = [field for field in required_scene_fields if not scene.get(field)]
        if missing_scene:
            raise ValueError(f"scene {index} missing required structured fields: {', '.join(missing_scene)}")
        relationship = scene.get("relationship_to_previous")
        if not isinstance(relationship, dict):
            raise ValueError(f"scene {index} relationship_to_previous must be an object")
        relationship_required = ["kind", "preserve", "change", "rationale"]
        if any(field not in relationship for field in relationship_required):
            raise ValueError(f"scene {index} relationship_to_previous is incomplete")
        if relationship.get("kind") not in {
            "opening", "establishing", "continuation", "continue", "continued", "reprise", "return",
            "refrain", "contrast", "payoff", "final", "finale", "ending", "resolution",
        }:
            raise ValueError(f"scene {index} relationship kind is invalid")
        for character in scene.get("selected_characters", []):
            if isinstance(character, dict) and str(character.get("allowed_variant", "")).strip().lower() in placeholder_values:
                raise ValueError(f"scene {index} contains placeholder character metadata")


def load_transformers_runtime(model_name: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=os.environ.get("PLANNER_LOCAL_FILES_ONLY", "1") != "0",
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=os.environ.get("PLANNER_LOCAL_FILES_ONLY", "1") != "0",
    )
    return torch, model, processor


def generate_with_transformers(
    model_name: str,
    user_prompt: str,
    max_new_tokens: int,
    *,
    system_prompt: str = "You produce strict JSON for video-production planning. No markdown, no commentary.",
    sample: bool = False,
    forbidden_words: list[str] | None = None,
    sample_seed: int | None = None,
    json_validator: Callable[[dict[str, Any]], None] | None = None,
    runtime: tuple[Any, Any, Any] | None = None,
) -> str:
    torch, model, processor = runtime or load_transformers_runtime(model_name)
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    bad_words_ids = None
    if forbidden_words:
        variants = {
            variant
            for word in forbidden_words
            for variant in (word, f" {word}", word.capitalize(), f" {word.capitalize()}")
        }
        bad_words_ids = [
            token_ids
            for variant in sorted(variants)
            if (token_ids := processor.tokenizer.encode(variant, add_special_tokens=False))
        ]
    last_output = ""
    if sample_seed is not None:
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
    for _attempt in range(5 if sample else 1):
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=0.35 if sample else None,
            top_p=0.85 if sample else None,
            repetition_penalty=1.06,
            bad_words_ids=bad_words_ids,
        )
        output_ids = generated[:, prompt_len:]
        last_output = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if not sample:
            break
        try:
            parsed = extract_json(last_output)
            if json_validator is not None:
                json_validator(parsed)
            break
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            continue
    return last_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen planner backend for storymem_agentic CommandAgentBackend")
    parser.add_argument("--model", default=os.environ.get("LOCAL_PLANNER_MODEL", "Qwen/Qwen2-VL-7B-Instruct"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("LOCAL_PLANNER_MAX_NEW_TOKENS", "4096")))
    parser.add_argument("--debug-output")
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read())
    user_prompt = build_user_prompt(payload)
    validation_issues = payload.get("context", {}).get("validation_issues") or []
    is_revision = bool(validation_issues)
    issue_codes = {
        str(issue.get("code", "")) for issue in validation_issues if isinstance(issue, dict)
    }
    forbidden_words = None
    if "unsafe_visual_action" in issue_codes:
        forbidden_words = [
            "fall", "falls", "falling", "fell", "drop", "drops", "dropping", "dropped",
            "break", "breaks", "breaking", "broke", "crash", "crashes", "crashing", "impact",
            "descend", "descends", "descending", "descended",
        ]
    if "repeated_camera_coverage" in issue_codes:
        forbidden_words = [*(forbidden_words or []), "medium shot"]
    if "audio_or_internal_action" in issue_codes:
        forbidden_words = [
            *(forbidden_words or []), "sing", "sings", "singing", "music", "think", "thinks",
            "thinking", "reflect", "reflects", "reflecting", "imagine", "imagines", "imagining",
            "enjoy", "enjoys", "enjoying", "ponder", "ponders", "pondering", "contemplate",
            "contemplates", "contemplating", "talk", "talks", "talking",
        ]
    if "adverb_only_action_change" in issue_codes:
        forbidden_words = [
            *(forbidden_words or []), "gently", "merrily", "peacefully", "joyfully", "happily",
        ]
    if "nonvisual_relationship_change" in issue_codes:
        forbidden_words = [
            *(forbidden_words or []), "contentment", "anticipation", "excitement", "joy", "patience",
            "delight", "feeling", "thought", "reflection",
        ]
    if issue_codes.intersection({"directional_wish_not_staged", "future_return_wish_not_staged"}):
        forbidden_words = [
            *(forbidden_words or []), "hope", "hopeful", "dance", "dances", "dancing", "spin",
            "spins", "spinning", "excitement", "anticipation", "gesture", "hands", "face", "puddle",
            "puddles",
        ]
    response_key = str(payload.get("context", {}).get("response_key", "planner_revision_1"))
    revision_number_match = re.search(r"(\d+)$", response_key)
    sample_seed = 1701 + (int(revision_number_match.group(1)) if revision_number_match else 0) * 7919

    directional_scenes = {
        int(issue["scene_num"])
        for issue in validation_issues
        if isinstance(issue, dict) and issue.get("code") == "directional_wish_not_staged"
        and isinstance(issue.get("scene_num"), int)
    }
    future_scenes = {
        int(issue["scene_num"])
        for issue in validation_issues
        if isinstance(issue, dict) and issue.get("code") == "future_return_wish_not_staged"
        and isinstance(issue.get("scene_num"), int)
    }
    nonvisual_relationship_scenes = {
        int(issue["scene_num"])
        for issue in validation_issues
        if isinstance(issue, dict) and issue.get("code") == "nonvisual_relationship_change"
        and isinstance(issue.get("scene_num"), int)
    }
    action_revision_scenes = {
        int(issue["scene_num"])
        for issue in validation_issues
        if isinstance(issue, dict)
        and issue.get("field") == "action"
        and isinstance(issue.get("scene_num"), int)
    }

    def validate_generated_decision(decision: dict[str, Any]) -> None:
        apply_binding_replacements(decision, validation_issues)
        validate_decision(decision)
        if not (
            directional_scenes
            or future_scenes
            or nonvisual_relationship_scenes
            or action_revision_scenes
        ):
            return
        action_revisions = {
            int(item["scene_num"]): str(item.get("replacement_value", ""))
            for item in decision.get("scene_revisions", [])
            if isinstance(item, dict) and item.get("field_to_change") == "action"
            and isinstance(item.get("scene_num"), int)
        }
        for scene_num in directional_scenes:
            replacement = action_revisions.get(scene_num, "").lower()
            if not re.search(
                r"(?:cloud|rain|storm|weather).*(?:clear|depart|drift|float|move|retreat).*(?:away|off)|"
                r"(?:clear|depart|drift|float|move|retreat).*(?:cloud|rain|storm|weather)",
                replacement,
            ):
                raise ValueError(f"scene {scene_num} must visibly stage the requested departure")
        for scene_num in future_scenes:
            replacement = action_revisions.get(scene_num, "").lower()
            if not re.search(r"\b(later|return|returns|returning|tomorrow|horizon)\b", replacement):
                raise ValueError(f"scene {scene_num} must include a visible future-return cue")
        relationship_revisions = {
            int(item["scene_num"]): item.get("replacement_value")
            for item in decision.get("scene_revisions", [])
            if isinstance(item, dict) and item.get("field_to_change") == "relationship_change"
            and isinstance(item.get("scene_num"), int)
        }
        for scene_num in nonvisual_relationship_scenes:
            replacement = relationship_revisions.get(scene_num, "")
            text = " ".join(replacement) if isinstance(replacement, list) else str(replacement)
            if re.search(
                r"\b(contentment|anticipation|excitement|joy|patience|delight|feeling|thought|reflection|"
                r"eager|eagerness|happy|happiness|sad|sadness)\b",
                text.lower(),
            ) or len(text.strip()) < 12:
                raise ValueError(f"scene {scene_num} relationship_change must name a visible action")
        for scene_num in action_revision_scenes:
            replacement = action_revisions.get(scene_num, "")
            if (
                len(replacement.split()) < 6
                or re.search(
                    r"\b(?:reveal(?:s|ed|ing)?|show(?:s|ed|ing)?|depict(?:s|ed|ing)?)\s+"
                    r"(?:the\s+|a\s+)?(?:subject|scene|lyric|beat|payoff)\b",
                    replacement.lower(),
                )
            ):
                raise ValueError(
                    f"scene {scene_num} action must name a concrete visible subject and event"
                )

    raw = generate_with_transformers(
        args.model,
        user_prompt,
        args.max_new_tokens,
        sample=True,
        forbidden_words=forbidden_words,
        sample_seed=sample_seed if is_revision else 1701,
        json_validator=validate_generated_decision,
    )
    if args.debug_output:
        with open(args.debug_output, "w", encoding="utf-8") as handle:
            handle.write(raw)
    try:
        decision = extract_json(raw)
        validate_generated_decision(decision)
    except Exception as exc:
        print(f"local_lullaby_planner invalid model output: {exc}", file=sys.stderr)
        print(raw[:4000], file=sys.stderr)
        return 4
    print(json.dumps(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
