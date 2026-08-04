from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .advanced_media import sample_scene_frames
from .media import _case
from .vlm_media import _extract_json, _mosaic


LABELS = ("A", "B", "C")
PANEL_WEIGHTS = {
    "semantic_grounding": 0.30,
    "narrative_progression": 0.20,
    "subject_identity_consistency": 0.20,
    "setting_continuity": 0.10,
    "temporal_coherence": 0.10,
    "visual_quality": 0.10,
}


def _panel_prompt(case: dict[str, Any]) -> str:
    targets = case.get("evaluation_scene_descriptions")
    if not isinstance(targets, list) or len(targets) != int(case["expected_scenes"]):
        raise ValueError("panel judging requires locked evaluation_scene_descriptions")
    return f"""You are a blinded comparative evaluator of three generated children's lullaby
videos, labeled A, B, and C in one composite image. Each labeled candidate section is a four-row
chronological contact sheet with early/middle/late frames per scene. Compare visible evidence
across candidates; system identities are hidden.

Task: {case['prompt']}
Locked observable scene targets: {json.dumps(targets, ensure_ascii=False)}

For every dimension ({', '.join(PANEL_WEIGHTS)}), assign each candidate a continuous 0-100 score
and a best-to-worst ranking. Use visible differences and the full scale. Evidence-justified ties
are allowed, especially for common-renderer visual properties, but do not use a shared default
without comparing the images. semantic_grounding
must penalize a missing visible observer wondering about the star in scene 2 and a missing single
diamond-shaped sparkle in scene 4. narrative_progression requires four observably different beats.
subject_identity_consistency compares star shape, face, color, and proportions across rows.
temporal_coherence compares the three frames within each row. Explain the decisive visible
difference briefly.

Return JSON only. The top-level object must contain exactly the six dimension names. Every
dimension value must be an object with: scores (an object with numeric A, B, and C values), ranking
(a best-to-worst array that may group tied labels), and evidence (a brief decisive comparison).
Choose the numbers from visible evidence. Do not copy a placeholder."""


def _combined_labeled_panel(label_to_mosaic: dict[str, Any]) -> Any:
    from PIL import Image, ImageDraw

    first = label_to_mosaic[LABELS[0]]
    header = 52
    canvas = Image.new(
        "RGB", (first.width, (first.height + header) * len(LABELS)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(LABELS):
        top = index * (first.height + header)
        draw.rectangle((0, top, first.width, top + header), fill=(25, 25, 25))
        draw.text((20, top + 16), f"CANDIDATE {label}", fill="white")
        canvas.paste(label_to_mosaic[label], (0, top + header))
    return canvas


def validate_panel_result(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != set(PANEL_WEIGHTS):
        raise ValueError("panel result must contain exactly the six rubric dimensions")
    normalized = {}
    for dimension in PANEL_WEIGHTS:
        item = value.get(dimension)
        if not isinstance(item, dict):
            raise ValueError(f"panel result missing {dimension}")
        raw_scores = item.get("scores")
        ranking = item.get("ranking")
        if isinstance(raw_scores, list) and len(raw_scores) == 3:
            raw_scores = dict(zip(LABELS, raw_scores))
        if not isinstance(raw_scores, dict) or set(raw_scores) != set(LABELS):
            raise ValueError(f"panel scores for {dimension} must contain A, B, and C")
        scores = {}
        for label in LABELS:
            raw = raw_scores[label]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"panel score {dimension}.{label} must be numeric")
            score = float(raw)
            if not 0 <= score <= 100:
                raise ValueError(f"panel score {dimension}.{label} must be between 0 and 100")
            scores[label] = score
        grouped: list[list[str]] = []
        for score in sorted(set(scores.values()), reverse=True):
            grouped.append([label for label in LABELS if scores[label] == score])
        normalized[dimension] = {
            "scores": scores,
            "ranking_groups": grouped,
            "raw_ranking": ranking,
            "evidence": str(item.get("evidence", "")),
        }
    return normalized


def aggregate_panel_runs(
    systems: list[str],
    runs: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    weights = weights or PANEL_WEIGHTS
    totals = {
        system: {
            "dimension_scores": {dimension: [] for dimension in weights},
            "dimension_rank_points": {dimension: [] for dimension in weights},
        }
        for system in systems
    }
    for run in runs:
        mapping = run["label_to_system"]
        result = run["result"]
        for dimension in weights:
            groups = result[dimension].get("ranking_groups")
            if groups:
                position = 0
                for group in groups:
                    available = [2 - index for index in range(position, position + len(group))]
                    points = sum(available) / len(available)
                    for label in group:
                        totals[mapping[label]]["dimension_rank_points"][dimension].append(
                            points
                        )
                    position += len(group)
            else:
                for rank_index, label in enumerate(result[dimension]["ranking"]):
                    system = mapping[label]
                    totals[system]["dimension_rank_points"][dimension].append(2 - rank_index)
            for label, score in result[dimension]["scores"].items():
                totals[mapping[label]]["dimension_scores"][dimension].append(score)
    aggregated = {}
    for system, values in totals.items():
        scores = {
            dimension: sum(items) / len(items)
            for dimension, items in values["dimension_scores"].items()
        }
        rank_points = {
            dimension: sum(items) / len(items)
            for dimension, items in values["dimension_rank_points"].items()
        }
        aggregated[system] = {
            "scores": scores,
            "weighted_overall_score": sum(
                scores[dimension] * weight
                for dimension, weight in weights.items()
            ),
            "rank_points": rank_points,
            "weighted_rank_points": sum(
                rank_points[dimension] * weight
                for dimension, weight in weights.items()
            ),
        }
    return aggregated


def score_vlm_panel(
    *,
    candidates: dict[str, Path],
    case: dict[str, Any],
    model_path: str,
    device_map: str = "auto",
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    systems = list(candidates)
    if len(systems) != 3:
        raise ValueError("panel scoring requires exactly three candidates")
    mosaics = {
        system: _mosaic(
            sample_scene_frames(path, int(case["expected_scenes"]), frames_per_scene=3)
        )
        for system, path in candidates.items()
    }
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    raw_runs = []
    for shift in range(3):
        order = systems[shift:] + systems[:shift]
        label_to_system = dict(zip(LABELS, order))
        panel = _combined_labeled_panel(
            {label: mosaics[label_to_system[label]] for label in LABELS}
        )
        content = [
            {"type": "image", "image": panel},
            {"type": "text", "text": _panel_prompt(case)},
        ]
        messages = [{"role": "user", "content": content}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=2200, do_sample=False, num_beams=1
            )
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        raw = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            validated = validate_panel_result(_extract_json(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid media panel response: {exc}; raw={raw}") from exc
        raw_runs.append(
            {
                "position_shift": shift,
                "label_to_system": label_to_system,
                "result": validated,
                "raw_model_output": raw,
            }
        )
    return {
        "case_id": case["case_id"],
        "metric_scope": "blinded_qwen2_vl_three_way_cyclic_panel_v2",
        "model": Path(model_path).name,
        "systems": systems,
        "runs": raw_runs,
        "aggregated": aggregate_panel_runs(systems, raw_runs),
        "rubric_weights": PANEL_WEIGHTS,
        "note": (
            "Three strict comparative judgments rotate every system through A, B, and C. "
            "Scores and rank points are averaged across positions to reduce order bias."
        ),
    }


def write_vlm_panel_score(
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
    result = score_vlm_panel(
        candidates={name: Path(path) for name, path in candidates.items()},
        case=_case(manifest, case_id),
        model_path=model_path,
        device_map=device_map,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    root = Path(per_system_output_root)
    for system, metrics in result["aggregated"].items():
        path = root / system / case_id / seed_dir / "vlm_panel_metrics.json"
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
