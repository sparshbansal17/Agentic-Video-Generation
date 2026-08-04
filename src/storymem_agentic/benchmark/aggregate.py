from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric_value(
    metric: str, advanced: dict[str, Any], panel: dict[str, Any]
) -> float | None:
    paths = {
        "semantic_contrastive_margin_mean": (advanced, "semantic_contrastive_margin_mean"),
        "semantic_retrieval_mrr": (advanced, "semantic_retrieval_mrr"),
        "dense_frame_margin_p10": (advanced, "dense_frame_margin_p10"),
        "dinov2_within_scene_consistency": (
            advanced.get("dinov2_temporal", {}),
            "within_scene_embedding_consistency_mean",
        ),
        "flow_warp_error_mean": (
            advanced.get("flow_temporal", {}),
            "flow_warp_error_mean",
        ),
        "panel_weighted_overall_score": (panel, "weighted_overall_score"),
        "panel_semantic_grounding": (panel.get("scores", {}), "semantic_grounding"),
        "panel_narrative_progression": (
            panel.get("scores", {}),
            "narrative_progression",
        ),
        "panel_subject_identity_consistency": (
            panel.get("scores", {}),
            "subject_identity_consistency",
        ),
        "panel_temporal_coherence": (panel.get("scores", {}), "temporal_coherence"),
        "panel_visual_quality": (panel.get("scores", {}), "visual_quality"),
    }
    container, key = paths[metric]
    value = container.get(key) if isinstance(container, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int = 0) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def aggregate_advanced_results(
    *,
    manifest: dict[str, Any],
    submissions_root: Path,
    case_id: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    directions = dict(manifest["advanced_evaluation"]["primary_metrics"])
    systems = [
        system
        for system in manifest["systems"]
        if system in {"storymem_agentic", "automv", "movieagent"}
    ]
    values: dict[str, dict[str, list[float]]] = {
        system: {metric: [] for metric in directions} for system in systems
    }
    per_seed = []
    errors = []
    total_wins = {system: 0.0 for system in systems}
    total_valid = 0
    for seed in seeds:
        seed_values: dict[str, dict[str, float]] = {system: {} for system in systems}
        for system in systems:
            root = submissions_root / system / case_id / f"seed_{seed:03d}"
            advanced_path = root / "advanced_media_metrics.json"
            panel_path = root / "vlm_panel_metrics.json"
            if not advanced_path.is_file() or not panel_path.is_file():
                errors.append(
                    {
                        "system": system,
                        "seed": seed,
                        "error": "missing advanced_media_metrics.json or vlm_panel_metrics.json",
                    }
                )
                continue
            advanced = _json(advanced_path)
            panel = _json(panel_path)
            for metric in directions:
                value = _metric_value(metric, advanced, panel)
                if value is not None:
                    seed_values[system][metric] = value
                    values[system][metric].append(value)
        rankings = []
        saturated = []
        win_credit = {system: 0.0 for system in systems}
        for metric, direction in directions.items():
            metric_values = {
                system: seed_values[system].get(metric) for system in systems
            }
            if not all(value is not None for value in metric_values.values()):
                continue
            numeric = {system: float(value) for system, value in metric_values.items()}
            spread = max(numeric.values()) - min(numeric.values())
            scale = max(abs(value) for value in numeric.values())
            tolerance = max(1e-12, scale * 1e-9)
            if spread <= tolerance:
                saturated.append({"metric": metric, "values": numeric})
                continue
            target = min(numeric.values()) if direction == "lower" else max(numeric.values())
            winners = [
                system
                for system, value in numeric.items()
                if abs(value - target) <= tolerance
            ]
            credit = 1.0 / len(winners)
            for system in winners:
                win_credit[system] += credit
                total_wins[system] += credit
            total_valid += 1
            rankings.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "values": numeric,
                    "winners": winners,
                    "spread": spread,
                }
            )
        per_seed.append(
            {
                "seed": seed,
                "evaluated_metric_count": len(rankings),
                "excluded_saturated_metrics": saturated,
                "win_credit": win_credit,
                "rankings": rankings,
            }
        )
    summary = {}
    for system in systems:
        summary[system] = {}
        for metric, metric_values in values[system].items():
            if not metric_values:
                continue
            summary[system][metric] = {
                "count": len(metric_values),
                "mean": float(np.mean(metric_values)),
                "sample_std": float(np.std(metric_values, ddof=1))
                if len(metric_values) > 1
                else None,
                "bootstrap_mean_ci95": _bootstrap_mean_ci(metric_values),
                "values": metric_values,
            }
    return {
        "benchmark_id": manifest["benchmark_id"],
        "case_id": case_id,
        "seeds": list(seeds),
        "systems": systems,
        "metric_directions": directions,
        "complete": not errors,
        "errors": errors,
        "per_seed": per_seed,
        "aggregate_win_credit": total_wins,
        "aggregate_valid_seed_metric_count": total_valid,
        "aggregate_win_rate": {
            system: total_wins[system] / total_valid if total_valid else None
            for system in systems
        },
        "metric_summary": summary,
        "note": (
            "Confidence intervals bootstrap renderer seeds, not independent prompts or human raters. "
            "Zero-spread seed endpoints are excluded from win counts."
        ),
    }


def write_advanced_aggregate(
    *,
    manifest_path: str | Path,
    submissions_root: str | Path,
    case_id: str,
    seeds: Sequence[int],
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = _json(Path(manifest_path))
    result = aggregate_advanced_results(
        manifest=manifest,
        submissions_root=Path(submissions_root),
        case_id=case_id,
        seeds=seeds,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
