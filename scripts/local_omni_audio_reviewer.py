#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(cleaned[cleaned.index("{") : cleaned.rindex("}") + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Local audio/video reviewer using Qwen2.5-Omni.")
    parser.add_argument("--model", default=os.getenv("LOCAL_AUDIO_REVIEWER_MODEL", "Qwen/Qwen2.5-Omni-7B"))
    parser.add_argument("--max-new-tokens", type=int, default=700)
    args = parser.parse_args()
    payload = json.loads(sys.stdin.read())
    context = payload.get("context", {})
    media_path = Path(context.get("review_media_path") or context.get("final_video_path") or "")
    if not media_path.exists():
        print(f"audio reviewer media does not exist: {media_path}", file=sys.stderr)
        return 4

    from qwen_omni_utils import process_mm_info
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

    reviewer = str(context.get("response_key") or "AudioReviewAgent")
    prompt = (
        f"Act as {reviewer}. Listen to the complete audio and inspect the video. "
        "Judge clear pleasant singing, complete supplied lyrics, continuous music, mix quality, child safety, "
        "story mood fit, and audio/visual timing. Metadata is supporting evidence only; base audio-quality claims "
        "on what you hear. Return one JSON object with passed, scores containing numeric overall, "
        "failure_reasons, and evidence containing a concise observation. Context: "
        + json.dumps({
            "lyrics": [scene.get("lyric") for scene in context.get("scenes", [])],
            "planned_windows": [
                [scene.get("planned_start_seconds"), scene.get("planned_end_seconds")]
                for scene in context.get("scenes", [])
            ],
            "whisperx_alignment": context.get("whisperx_alignment"),
        })[:12000]
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(media_path), "use_audio_in_video": True},
            {"type": "text", "text": prompt},
        ],
    }]
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=os.getenv("REVIEWER_LOCAL_FILES_ONLY", "1") != "0",
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=os.getenv("REVIEWER_LOCAL_FILES_ONLY", "1") != "0",
    )
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(
        text=rendered,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    ).to(model.device)
    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, return_audio=False)
    if isinstance(generated, tuple):
        generated = generated[0]
    generated = generated[:, inputs.input_ids.shape[1] :]
    response = extract_json(processor.batch_decode(generated, skip_special_tokens=True)[0])
    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
