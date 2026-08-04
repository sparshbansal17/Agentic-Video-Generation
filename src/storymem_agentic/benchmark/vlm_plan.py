from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .media import _case
from .vlm_media import _extract_json


PLAN_RUBRIC_WEIGHTS = {
    "semantic_role_fidelity": 0.25,
    "observable_action_grounding": 0.15,
    "narrative_progression": 0.15,
    "cross_scene_continuity": 0.15,
    "shot_design": 0.10,
    "safety_adaptation": 0.10,
    "renderability": 0.10,
}


def _normalized_scenes(plan: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    raw_scenes = plan.get("scenes")
    if not isinstance(raw_scenes, list) or len(raw_scenes) != expected:
        raise ValueError(f"normalized plan must contain exactly {expected} scenes")
    scenes = []
    for index, raw in enumerate(raw_scenes, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"scene {index} must be an object")
        prompts = raw.get("video_prompts")
        prompt = prompts[0] if isinstance(prompts, list) and prompts else ""
        scenes.append(
            {
                "scene_num": index,
                "lyric_line": str(raw.get("lyric_line") or raw.get("subtitle_text") or ""),
                "render_prompt": str(prompt),
            }
        )
    return scenes


def _plan_prompt(case: dict[str, Any], scenes: list[dict[str, Any]]) -> str:
    targets = case.get("evaluation_scene_descriptions")
    if not isinstance(targets, list) or len(targets) != len(scenes):
        raise ValueError("plan judging requires locked evaluation_scene_descriptions")
    return f"""You are a blinded evaluator of normalized plans for a children's lullaby video.
Every system is represented only by four lyric/render-prompt pairs. Judge semantic and cinematic
content, not wording length, formatting, or hidden metadata.

Task: {case['prompt']}
Locked observable targets: {json.dumps(targets, ensure_ascii=False)}
Candidate plan: {json.dumps(scenes, ensure_ascii=False, indent=2)}

Score 0-100 for: {', '.join(PLAN_RUBRIC_WEIGHTS)}. Use the full range: 90-100 excellent and
specific, 70-89 mostly correct, 50-69 generic/partial, 30-49 major omissions or contradictions,
0-29 unusable. semantic_role_fidelity must treat “How I wonder what you are” as an observer
wondering about the star; a star merely becoming mysterious or inviting curiosity without a visible
observer is incomplete. observable_action_grounding requires visible actions, not abstract moods.
narrative_progression requires four different beats and a concrete diamond-shaped final payoff.
cross_scene_continuity requires the same star and village without erasing intended scene changes.
shot_design rewards motivated varied coverage. safety_adaptation rewards specific child-safe visual
handling, not boilerplate alone. renderability penalizes contradictions, duplicated directives, and
overloaded prompts, but does not reward brevity by itself.

Return JSON only with two objects named scores and evidence. Both objects must contain exactly the
seven dimension names listed above. Every scores value must be a numeric 0-100 judgment chosen from
the evidence; every evidence value must be a brief string. Do not use a shared default score and do
not copy a schema placeholder."""


def validate_plan_rubric(value: dict[str, Any]) -> dict[str, float]:
    scores = value.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("plan judge output is missing scores")
    result = {}
    for name in PLAN_RUBRIC_WEIGHTS:
        raw = scores.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"plan judge score {name} must be numeric")
        score = float(raw)
        if not 0 <= score <= 100:
            raise ValueError(f"plan judge score {name} must be between 0 and 100")
        result[name] = score
    if len(set(result.values())) < 3:
        raise ValueError("plan judge must use at least three distinct scores across dimensions")
    return result


def score_plan_vlm(
    *, case: dict[str, Any], plan: dict[str, Any], model_path: str, device_map: str = "auto"
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    scenes = _normalized_scenes(plan, int(case["expected_scenes"]))
    prompt = _plan_prompt(case, scenes)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=900, do_sample=False, num_beams=1
        )
    trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parsed = _extract_json(raw)
    scores = validate_plan_rubric(parsed)
    evidence = parsed.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    return {
        "case_id": case["case_id"],
        "metric_scope": "blinded_qwen2_vl_normalized_plan_rubric_v2",
        "model": Path(model_path).name,
        "scores": scores,
        "weighted_overall_score": sum(
            scores[name] * weight for name, weight in PLAN_RUBRIC_WEIGHTS.items()
        ),
        "rubric_weights": PLAN_RUBRIC_WEIGHTS,
        "evidence": {name: str(evidence.get(name, "")) for name in PLAN_RUBRIC_WEIGHTS},
        "raw_model_output": raw,
        "note": "Deterministic judge over the same normalized lyric/render-prompt fields for every system.",
    }


def write_plan_vlm_score(
    *,
    manifest_path: str | Path,
    case_id: str,
    plan_path: str | Path,
    output_path: str | Path,
    model_path: str,
    device_map: str = "auto",
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    result = score_plan_vlm(
        case=_case(manifest, case_id),
        plan=plan,
        model_path=model_path,
        device_map=device_map,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
