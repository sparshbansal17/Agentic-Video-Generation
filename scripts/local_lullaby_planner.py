#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def build_user_prompt(payload: dict[str, Any]) -> str:
    prompt = str(payload.get("prompt", ""))
    context = payload.get("context", {})
    validation_issues = context.get("validation_issues") or []
    previous_decision = context.get("previous_decision")
    revision_block = ""
    if validation_issues:
        issue_codes = {
            str(issue.get("code", ""))
            for issue in validation_issues
            if isinstance(issue, dict)
        }
        unsafe_instruction = ""
        if "unsafe_visual_action" in issue_codes:
            unsafe_instruction = (
                "\nCRITICAL SAFETY REVISION REQUIRED:\n"
                "- For every unsafe_visual_action scene, rewrite scene_goal, lyric_interpretation, action, setting, "
                "subjects, and safety_adaptation.\n"
                "- Use the evidence excerpts in validation_issues to identify the exact hazardous visual wording.\n"
                "- Replace unsupported descent, breakage, dropping, crashing, or impact with a calm safe adaptation "
                "that preserves the lyric's emotional meaning without depicting danger.\n"
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
                "- Give it a materially different setting or foreground/background staging, a different visible action, "
                "and a different camera composition from matches_scene_num.\n"
                "- Preserve character identity and the supplied lyric; do not return the previous scene unchanged.\n"
            )
        camera_instruction = ""
        if "repeated_camera_coverage" in issue_codes:
            camera_instruction = (
                "\nCRITICAL CAMERA-COVERAGE REVISION REQUIRED:\n"
                "- Edit the camera field of every cited scene. The replacement must not equal the current camera text.\n"
                "- Change the actual coverage: choose a motivated wide establishing view, close-up reaction/detail, "
                "low child-eye angle, overhead composition, gentle reveal, or another clearly different shot size/angle.\n"
                "- Do not merely rephrase the same medium/static/tracking shot. Keep at most one simple movement.\n"
            )
        revision_block = (
            "\nThis is a revision request. Fix every validation issue below by returning a complete replacement "
            "planner decision JSON. Preserve supplied lyrics exactly and edit the structured scene descriptions, "
            "camera fields, characters, and continuity fields before prompt compilation.\n"
            f"Validation issues JSON:\n{json.dumps(validation_issues, indent=2)}\n"
            f"{unsafe_instruction}{diversity_instruction}{camera_instruction}\n"
            "GENERAL CORRECTION CONTRACT:\n"
            "- Treat every issue object, regardless of code, as a binding acceptance criterion.\n"
            "- Use scene_num to edit the affected scene and use field/evidence/message to determine the required change.\n"
            "- Compare the replacement against previous_decision before answering; every cited defect must have a visible JSON-field change.\n"
            "- Do not insert canned topic imagery or substitute a generic bedtime scene unrelated to the supplied input.\n"
        )
        if previous_decision:
            revision_block += f"Previous planner decision JSON:\n{json.dumps(previous_decision, indent=2)}\n"
        return (
            "Revise a structured video plan by returning a SMALL JSON PATCH, not the complete plan.\n"
            "The patch shape is: {\"scene_revisions\": [{\"scene_num\": 1, \"field_to_change\": "
            "\"replacement value\"}], \"plan_updates\": {}}. Include every field needed to resolve each issue, "
            "but do not include unchanged scenes or explanatory prose. Preserve lyrics and character identity.\n"
            "field_to_change MUST be one of: scene_goal, lyric_interpretation, setting, subjects, action, camera, "
            "style, safety_adaptation, selected_characters, expected_mood, boundary_behavior, cut. Never patch "
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
        '  "visual_bible": {"primary_world": "concise world name", "allowed_locations": ["location"]},\n'
        '  "selected_characters": [\n'
        '    {"label": "character_id", "role": "role", "description": "concise consistent visual description", '
        '"selection_rationale": "why this character fits the lyrics", "visual_anchors": ["anchor"], "allowed_variants": ["variant"], '
        '"continuity_constraints": ["constraint"], "negative_constraints": ["avoidance"]}\n'
        "  ],\n"
        '  "scenes": [\n'
        '    {"scene_num": 1, "lyric_line": "line 1", '
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
        if not isinstance(decision["scene_revisions"], list) or not decision["scene_revisions"]:
            raise ValueError("revision patch requires non-empty scene_revisions array")
        if any(not isinstance(item, dict) or not item.get("scene_num") for item in decision["scene_revisions"]):
            raise ValueError("each scene revision requires scene_num and changed fields")
        return
    if decision.get("type") == "object" and "properties" in decision and "lyrics" not in decision:
        raise ValueError("planner returned a JSON schema instead of a planner decision")
    required = ["lyrics", "clip_count", "target_duration_seconds", "visual_bible", "selected_characters", "scenes", "music_prompt"]
    missing = [key for key in required if key not in decision]
    if missing:
        raise ValueError(f"planner decision missing required keys: {', '.join(missing)}")
    if not isinstance(decision["lyrics"], list) or not [line for line in decision["lyrics"] if str(line).strip()]:
        raise ValueError("planner decision requires non-empty lyrics array")
    if not isinstance(decision["scenes"], list) or not decision["scenes"]:
        raise ValueError("planner decision requires non-empty scenes array")
    if not isinstance(decision["selected_characters"], list) or not decision["selected_characters"]:
        raise ValueError("planner decision requires non-empty selected_characters array")
    required_scene_fields = {
        "scene_goal",
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


def generate_with_transformers(
    model_name: str,
    user_prompt: str,
    max_new_tokens: int,
    *,
    sample: bool = False,
    forbidden_words: list[str] | None = None,
    sample_seed: int | None = None,
) -> str:
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
    messages = [
        {
            "role": "system",
            "content": "You produce strict JSON for video-production planning. No markdown, no commentary.",
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
    for _attempt in range(3 if sample else 1):
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=0.35 if sample else None,
            top_p=0.85 if sample else None,
            bad_words_ids=bad_words_ids,
        )
        output_ids = generated[:, prompt_len:]
        last_output = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if not sample:
            break
        try:
            extract_json(last_output)
            break
        except (json.JSONDecodeError, ValueError):
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
        ]
    if "repeated_camera_coverage" in issue_codes:
        forbidden_words = [*(forbidden_words or []), "medium shot"]
    response_key = str(payload.get("context", {}).get("response_key", "planner_revision_1"))
    revision_number_match = re.search(r"(\d+)$", response_key)
    sample_seed = 1701 + (int(revision_number_match.group(1)) if revision_number_match else 0) * 7919
    raw = generate_with_transformers(
        args.model,
        user_prompt,
        args.max_new_tokens,
        sample=is_revision,
        forbidden_words=forbidden_words,
        sample_seed=sample_seed if is_revision else None,
    )
    if args.debug_output:
        with open(args.debug_output, "w", encoding="utf-8") as handle:
            handle.write(raw)
    try:
        decision = extract_json(raw)
        validate_decision(decision)
    except Exception as exc:
        print(f"local_lullaby_planner invalid model output: {exc}", file=sys.stderr)
        print(raw[:4000], file=sys.stderr)
        return 4
    print(json.dumps(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
