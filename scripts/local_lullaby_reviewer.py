#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


EXPECTED_REVIEWERS = [
    "VisualSafetyReviewAgent",
    "StoryAlignmentReviewAgent",
    "ContinuityReviewAgent",
    "AudioReviewAgent",
    "AudioVisualSyncReviewAgent",
]


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


def compact_context(context: dict[str, Any], max_frames: int) -> tuple[dict[str, Any], list[str]]:
    frames: list[str] = []
    scenes = []
    image_index = 1
    for scene in context.get("scenes", []):
        frame_paths = [path for path in scene.get("frame_paths", []) if Path(path).exists()]
        selected = frame_paths[:max(1, max_frames // max(1, len(context.get("scenes", []))))]
        frames.extend(selected)
        labels = []
        for _path in selected:
            labels.append(f"attached_image_{image_index}")
            image_index += 1
        scenes.append(
            {
                "scene_num": scene.get("scene_num"),
                "lyric": scene.get("lyric"),
                "planned_start_seconds": scene.get("planned_start_seconds"),
                "planned_end_seconds": scene.get("planned_end_seconds"),
                "video_prompt": scene.get("video_prompt"),
                "first_frame_prompt": scene.get("first_frame_prompt"),
                "attached_frame_labels": labels,
            }
        )
    frames = frames[:max_frames]
    compact = {
        "strict_lullaby_review": context.get("strict_lullaby_review"),
        "final_video_path": context.get("final_video_path"),
        "subtitle_path": context.get("subtitle_path"),
        "artifact_checks": context.get("artifact_checks"),
        "streams": context.get("streams"),
        "duration_seconds": context.get("duration_seconds"),
        "planned_duration_seconds": context.get("planned_duration_seconds"),
        "characters": context.get("characters"),
        "scenes": scenes,
        "whisperx_alignment": context.get("whisperx_alignment"),
        "acceptance_thresholds": context.get("acceptance_thresholds"),
    }
    return compact, frames


def build_prompt(payload: dict[str, Any], compact: dict[str, Any]) -> str:
    return (
        "Review this generated toddler lullaby video using the attached sampled frames and metadata. "
        "Return one concise JSON object only, with a 'reports' array containing exactly these reviewers: "
        + ", ".join(EXPECTED_REVIEWERS)
        + ". Each report must contain only reviewer, passed, scores, failure_reasons, and evidence. "
        "Scores must contain only one numeric 'overall' key. Evidence must contain at most three keys: "
        "sampled_frames_checked (boolean), observation (one sentence), and optionally target_scenes "
        "(an array of scene numbers). Never create nested evidence, recursive category names, exhaustive checklists, "
        "or unlisted keys. Keep the entire response below 900 words. "
        "If any fix is needed, include target_scenes, prompt_revisions, first_frame_prompt_revisions, "
        "audio_prompt_revision, subtitle_timing_adjustments, or mix_adjustments. "
        "Reject generated in-frame text, scary imagery, unsafe content, clutter, poor lyric-scene match, "
        "character/style drift, harsh or low-quality music, unclear vocals, missing lyrics, or bad audio-video pairing. "
        "Be concrete and concise. Example: "
        "{\"reviewer\":\"VisualSafetyReviewAgent\",\"passed\":true,\"scores\":{\"overall\":1.0},"
        "\"failure_reasons\":[],\"evidence\":{\"sampled_frames_checked\":true,"
        "\"observation\":\"Frames are safe and clear.\"}}. Context JSON:\n"
        f"{json.dumps(compact, indent=2)[:24000]}\n\n"
        "Original review instructions:\n"
        f"{str(payload.get('prompt', ''))[:4000]}"
    )


def validate_reports(response: dict[str, Any]) -> None:
    reports = response.get("reports")
    if not isinstance(reports, list):
        raise ValueError("review output requires reports array")
    names = {item.get("reviewer") for item in reports if isinstance(item, dict)}
    missing = [name for name in EXPECTED_REVIEWERS if name not in names]
    if missing:
        raise ValueError(f"review output missing reviewers: {', '.join(missing)}")


def generate_with_qwen_vl(model_name: str, prompt: str, image_paths: list[str], max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device_map = os.environ.get("LOCAL_REVIEWER_DEVICE_MAP")
    if device_map is None:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        local_files_only=os.environ.get("REVIEWER_LOCAL_FILES_ONLY", "1") != "0",
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=os.environ.get("REVIEWER_LOCAL_FILES_ONLY", "1") != "0",
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({"type": "image", "image": path})
    messages = [
        {"role": "system", "content": "You are a strict multimodal QA panel. Output valid JSON only."},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    output_ids = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen-VL reviewer backend for storymem_agentic.")
    parser.add_argument("--model", default=os.environ.get("LOCAL_REVIEWER_MODEL", "Qwen/Qwen2-VL-7B-Instruct"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("LOCAL_REVIEWER_MAX_NEW_TOKENS", "2048")))
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("LOCAL_REVIEWER_MAX_FRAMES", "6")))
    parser.add_argument("--debug-output")
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read())
    compact, image_paths = compact_context(payload.get("context", {}), args.max_frames)
    prompt = build_prompt(payload, compact)
    raw = generate_with_qwen_vl(args.model, prompt, image_paths, args.max_new_tokens)
    if args.debug_output:
        Path(args.debug_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.debug_output).write_text(raw, encoding="utf-8")
    try:
        response = extract_json(raw)
        validate_reports(response)
    except Exception as exc:
        print(f"local_lullaby_reviewer invalid model output: {exc}", file=sys.stderr)
        print(raw[:4000], file=sys.stderr)
        return 4
    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
