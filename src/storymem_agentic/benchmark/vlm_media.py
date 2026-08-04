from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .advanced_media import sample_scene_frames
from .media import _case


RUBRIC_WEIGHTS = {
    "semantic_grounding": 0.25,
    "narrative_progression": 0.15,
    "subject_identity_consistency": 0.15,
    "setting_continuity": 0.10,
    "temporal_coherence": 0.10,
    "visual_quality": 0.10,
    "lullaby_suitability": 0.10,
    "safety": 0.05,
}


def _mosaic(frames: list[list[Any]]) -> Any:
    from PIL import Image, ImageDraw

    cell_width, cell_height = 384, 216
    label_height = 28
    canvas = Image.new(
        "RGB", (cell_width * 3, (cell_height + label_height) * len(frames)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    positions = ("early", "middle", "late")
    for scene_index, scene in enumerate(frames, 1):
        if len(scene) != 3:
            raise ValueError("VLM mosaic requires exactly three frames per scene")
        top = (scene_index - 1) * (cell_height + label_height)
        for column, (position, image) in enumerate(zip(positions, scene)):
            resized = image.convert("RGB").resize((cell_width, cell_height))
            left = column * cell_width
            canvas.paste(resized, (left, top + label_height))
            draw.text((left + 8, top + 7), f"Scene {scene_index} — {position}", fill="black")
    return canvas


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("VLM output did not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM output must be a JSON object")
    return value


def validate_rubric_scores(value: dict[str, Any]) -> dict[str, float]:
    scores = value.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("VLM output is missing scores")
    normalized: dict[str, float] = {}
    for name in RUBRIC_WEIGHTS:
        raw = scores.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"VLM score {name} must be numeric")
        score = float(raw)
        if not 0 <= score <= 100:
            raise ValueError(f"VLM score {name} must be between 0 and 100")
        normalized[name] = score
    return normalized


def weighted_rubric_score(scores: dict[str, float]) -> float:
    return sum(scores[name] * weight for name, weight in RUBRIC_WEIGHTS.items())


def _rubric_prompt(case: dict[str, Any]) -> str:
    descriptions = case.get("evaluation_scene_descriptions")
    if not isinstance(descriptions, list) or len(descriptions) != int(case["expected_scenes"]):
        raise ValueError("VLM scoring requires locked evaluation_scene_descriptions")
    numbered = "\n".join(
        f"{index}. Lyric: {lyric}\n   Observable target: {description}"
        for index, (lyric, description) in enumerate(
            zip(case["lyrics"], descriptions), 1
        )
    )
    dimensions = ", ".join(RUBRIC_WEIGHTS)
    return f"""You are a blinded evaluator of a generated children's lullaby video. The image is a
4-row contact sheet: one row per chronological scene and three frames per scene. Judge only visible
evidence. Do not infer success from the task description, and do not reward the row labels.

Task: {case['prompt']}
Required chronological evidence:
{numbered}

Score these dimensions from 0 to 100: {dimensions}.
Use the full range. 90-100 means nearly all visible requirements are clearly realized with excellent
consistency; 70-89 means most are clear with minor defects; 50-69 means partial or generic realization;
30-49 means major omissions or contradictions; 0-29 means absent, severely broken, or unsafe. For
semantic_grounding, explicitly penalize missing observer curiosity in scene 2 and a missing single
diamond-shaped sparkle in scene 4. For identity, compare the star's shape, face, color, and proportions
across rows. For temporal coherence, compare early/middle/late frames within each row. For narrative
progression, require four observably different beats rather than camera-only changes.

Return JSON only in this exact structure:
{{"scores": {{"semantic_grounding": 0, "narrative_progression": 0,
"subject_identity_consistency": 0, "setting_continuity": 0, "temporal_coherence": 0,
"visual_quality": 0, "lullaby_suitability": 0, "safety": 0}},
"evidence": {{"semantic_grounding": "brief visible evidence", "narrative_progression": "brief visible evidence",
"subject_identity_consistency": "brief visible evidence", "setting_continuity": "brief visible evidence",
"temporal_coherence": "brief visible evidence", "visual_quality": "brief visible evidence",
"lullaby_suitability": "brief visible evidence", "safety": "brief visible evidence"}}}}"""


def score_vlm_media(
    *,
    video_path: Path,
    case: dict[str, Any],
    model_path: str,
    device_map: str = "auto",
    mosaic_output: Path | None = None,
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    frames = sample_scene_frames(video_path, int(case["expected_scenes"]), 3)
    mosaic = _mosaic(frames)
    if mosaic_output:
        mosaic_output.parent.mkdir(parents=True, exist_ok=True)
        mosaic.save(mosaic_output)
    prompt = _rubric_prompt(case)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": mosaic},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            num_beams=1,
        )
    trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parsed = _extract_json(raw)
    scores = validate_rubric_scores(parsed)
    evidence = parsed.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    mosaic_bytes = mosaic.tobytes()
    return {
        "case_id": case["case_id"],
        "metric_scope": "blinded_qwen2_vl_contact_sheet_rubric_v1",
        "model": Path(model_path).name,
        "sampling": "three uniformly spaced interior frames per scene",
        "scores": scores,
        "weighted_overall_score": weighted_rubric_score(scores),
        "evidence": {name: str(evidence.get(name, "")) for name in RUBRIC_WEIGHTS},
        "rubric_weights": RUBRIC_WEIGHTS,
        "mosaic_rgb_sha256": hashlib.sha256(mosaic_bytes).hexdigest(),
        "raw_model_output": raw,
        "note": (
            "Deterministic single-model rubric judgment over a system-blind contact sheet. "
            "Treat as complementary to embedding metrics and blinded human evaluation."
        ),
    }


def write_vlm_media_score(
    *,
    manifest_path: str | Path,
    case_id: str,
    video_path: str | Path,
    output_path: str | Path,
    model_path: str,
    device_map: str = "auto",
    mosaic_output: str | Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    result = score_vlm_media(
        video_path=Path(video_path),
        case=_case(manifest, case_id),
        model_path=model_path,
        device_map=device_map,
        mosaic_output=Path(mosaic_output) if mosaic_output else None,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
