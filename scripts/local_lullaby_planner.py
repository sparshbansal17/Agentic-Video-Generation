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
        '  "characters": [\n'
        '    {"label": "character_id", "description": "visual description", '
        '"continuity_constraints": ["constraint"], "negative_constraints": ["avoidance"]}\n'
        "  ],\n"
        '  "scenes": [\n'
        '    {"scene_num": 1, "lyric_line": "line 1", '
        '"description": "Opening shot: specific full-frame setting with foreground and background. Specific child-safe subject action. Camera specific movement. Visual style, lighting, color palette, mood, continuity, no text.", '
        '"camera": "shot size, camera angle, lens, composition, and camera movement", '
        '"expected_mood": "calm bedtime mood", "boundary_behavior": "hold", "cut": true}\n'
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
        "Follow the Wan prompt recipe for scene planning. Each clip must support the Advanced Formula: subject plus "
        "subject description, scene plus foreground/background description, motion plus motion description, aesthetic "
        "control, and stylization. Plan this like a normal edited video storyboard, not one continuous shot. Make every "
        "clip a distinct full-frame shot: vary location or staging, action, camera angle, shot size, subject distance, "
        "foreground/background layout, lens, composition, color tone, lighting, and motion. Use cut=true for each "
        "lyric-scene clip unless the user explicitly asks for a single continuous one-shot. Do not repeat the same "
        "generic medium-shot camera language for all clips. Each scene description must be a concrete paragraph in "
        "this format: 'Opening shot: [specific setting with foreground/background]. [specific subject] [specific "
        "action]. Camera [specific movement]. [visual style, lighting, color palette, mood]. [continuity and "
        "child-safety constraints].' Do not output vague descriptions like 'child looks at moon' or 'star shines'. "
        "Do not copy literal lyric text into scene descriptions; "
        "describe the visual meaning instead. Put any needed character or setting continuity details directly in the "
        "scene description; do not assume a repeated character-bank prefix will be added later. "
        "Never describe a small framed box, inset image, picture-in-picture, poster, border, title card, or screen-within-screen. "
        "Do not include dialogue or background music in visual scenes; audio is generated separately. "
        "Return only valid JSON. Do not return a JSON schema. Do not wrap it in markdown."
    )


def validate_decision(decision: dict[str, Any]) -> None:
    if decision.get("type") == "object" and "properties" in decision and "lyrics" not in decision:
        raise ValueError("planner returned a JSON schema instead of a planner decision")
    required = ["lyrics", "clip_count", "target_duration_seconds", "characters", "scenes", "music_prompt"]
    missing = [key for key in required if key not in decision]
    if missing:
        raise ValueError(f"planner decision missing required keys: {', '.join(missing)}")
    if not isinstance(decision["lyrics"], list) or not [line for line in decision["lyrics"] if str(line).strip()]:
        raise ValueError("planner decision requires non-empty lyrics array")
    if not isinstance(decision["scenes"], list) or not decision["scenes"]:
        raise ValueError("planner decision requires non-empty scenes array")
    if not isinstance(decision["characters"], list) or not decision["characters"]:
        raise ValueError("planner decision requires non-empty characters array")


def generate_with_transformers(model_name: str, user_prompt: str, max_new_tokens: int) -> str:
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
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    output_ids = generated[:, prompt_len:]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen planner backend for storymem_agentic CommandAgentBackend")
    parser.add_argument("--model", default=os.environ.get("LOCAL_PLANNER_MODEL", "Qwen/Qwen2-VL-7B-Instruct"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("LOCAL_PLANNER_MAX_NEW_TOKENS", "4096")))
    parser.add_argument("--debug-output")
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read())
    user_prompt = build_user_prompt(payload)
    raw = generate_with_transformers(args.model, user_prompt, args.max_new_tokens)
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
