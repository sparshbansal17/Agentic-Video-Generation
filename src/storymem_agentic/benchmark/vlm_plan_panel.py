from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .media import _case
from .vlm_media import _extract_json
from .vlm_panel import LABELS, aggregate_panel_runs
from .vlm_plan import PLAN_RUBRIC_WEIGHTS, _normalized_scenes


def _prompt(case: dict[str, Any], labeled_plans: dict[str, list[dict[str, Any]]]) -> str:
    targets = case.get("evaluation_scene_descriptions")
    if not isinstance(targets, list) or len(targets) != int(case["expected_scenes"]):
        raise ValueError("plan panel requires locked evaluation_scene_descriptions")
    return f"""You are a blinded comparative evaluator of three normalized plans for a children's
lullaby video. Each candidate contains exactly the same public fields: lyric line and render prompt.
System identities and private metadata are hidden. Judge semantic/cinematic content, not length.

Task: {case['prompt']}
Locked observable targets: {json.dumps(targets, ensure_ascii=False)}
Candidates: {json.dumps(labeled_plans, ensure_ascii=False, indent=2)}

For every dimension ({', '.join(PLAN_RUBRIC_WEIGHTS)}), give A, B, and C distinct continuous
0-100 scores and a strict best-to-worst ranking. semantic_role_fidelity must treat “How I wonder
what you are” as an observer visibly wondering about the star; a star becoming mysterious or
inviting curiosity without an observer is incomplete. observable_action_grounding requires visible
actions, not abstract moods. narrative_progression requires four different beats and a concrete
diamond-shaped final payoff. cross_scene_continuity requires the same star and village while
preserving intended changes. shot_design rewards motivated varied coverage. safety_adaptation
rewards specific visual handling, not boilerplate alone. renderability penalizes contradictions,
duplication, and overload without rewarding brevity by itself.

Return JSON only. The top-level object must contain exactly the seven dimension names. Every
dimension must contain: scores (numeric A/B/C values), ranking (a strict array of A/B/C ordered best
to worst), and evidence (one brief decisive comparison). Scores within each dimension must differ.
Do not use shared defaults or schema placeholders."""


def validate_plan_panel(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != set(PLAN_RUBRIC_WEIGHTS):
        raise ValueError("plan panel must contain exactly the seven rubric dimensions")
    normalized = {}
    for dimension in PLAN_RUBRIC_WEIGHTS:
        item = value[dimension]
        if not isinstance(item, dict):
            raise ValueError(f"plan panel result {dimension} must be an object")
        raw_scores = item.get("scores")
        ranking = item.get("ranking")
        if isinstance(raw_scores, list) and len(raw_scores) == 3:
            raw_scores = dict(zip(LABELS, raw_scores))
        if not isinstance(raw_scores, dict) or set(raw_scores) != set(LABELS):
            raise ValueError(f"plan panel scores for {dimension} must contain A, B, and C")
        scores = {}
        for label in LABELS:
            raw = raw_scores[label]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"plan panel score {dimension}.{label} must be numeric")
            score = float(raw)
            if not 0 <= score <= 100:
                raise ValueError(f"plan panel score {dimension}.{label} must be 0-100")
            scores[label] = score
        if len(set(scores.values())) != 3:
            raise ValueError(f"plan panel scores for {dimension} must be distinct")
        if not isinstance(ranking, list) or len(ranking) != 3 or set(ranking) != set(LABELS):
            raise ValueError(f"plan panel ranking for {dimension} must contain A, B, and C")
        expected = sorted(LABELS, key=lambda label: scores[label], reverse=True)
        reconciled = ranking != expected
        if reconciled:
            ordered_scores = sorted(scores.values(), reverse=True)
            scores = dict(zip(ranking, ordered_scores))
        normalized[dimension] = {
            "scores": scores,
            "ranking": list(ranking),
            "evidence": str(item.get("evidence", "")),
            "score_order_reconciled": reconciled,
        }
    return normalized


def score_plan_vlm_panel(
    *,
    candidates: dict[str, dict[str, Any]],
    case: dict[str, Any],
    model_path: str,
    device_map: str = "auto",
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    systems = list(candidates)
    if len(systems) != 3:
        raise ValueError("plan panel requires exactly three candidates")
    normalized = {
        system: _normalized_scenes(plan, int(case["expected_scenes"]))
        for system, plan in candidates.items()
    }
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    runs = []
    for shift in range(3):
        order = systems[shift:] + systems[:shift]
        mapping = dict(zip(LABELS, order))
        labeled = {label: normalized[system] for label, system in mapping.items()}
        messages = [
            {"role": "user", "content": [{"type": "text", "text": _prompt(case, labeled)}]}
        ]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=2200, do_sample=False, num_beams=1
            )
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        raw = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            validated = validate_plan_panel(_extract_json(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid plan panel response: {exc}; raw={raw}") from exc
        runs.append(
            {
                "position_shift": shift,
                "label_to_system": mapping,
                "result": validated,
                "raw_model_output": raw,
            }
        )
    return {
        "case_id": case["case_id"],
        "metric_scope": "blinded_qwen2_vl_three_way_cyclic_plan_panel_v3",
        "model": Path(model_path).name,
        "systems": systems,
        "runs": runs,
        "aggregated": aggregate_panel_runs(systems, runs, PLAN_RUBRIC_WEIGHTS),
        "rubric_weights": PLAN_RUBRIC_WEIGHTS,
        "note": "Three comparative plan judgments rotate every system through A, B, and C.",
    }


def write_plan_vlm_panel_score(
    *,
    manifest_path: str | Path,
    case_id: str,
    candidates: dict[str, str | Path],
    output_path: str | Path,
    per_system_output_root: str | Path,
    seed_dir: str,
    model_path: str,
    device_map: str = "auto",
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    result = score_plan_vlm_panel(
        candidates={
            name: json.loads(Path(path).read_text(encoding="utf-8"))
            for name, path in candidates.items()
        },
        case=_case(manifest, case_id),
        model_path=model_path,
        device_map=device_map,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    root = Path(per_system_output_root)
    for system, metrics in result["aggregated"].items():
        path = root / system / case_id / seed_dir / "advanced_plan_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "metric_scope": result["metric_scope"],
                    **metrics,
                    "panel_report": str(destination.resolve()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return result
