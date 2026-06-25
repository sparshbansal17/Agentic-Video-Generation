from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - optional in minimal test envs
    cv2 = None

from .agents import AgentBackend, CommandAgentBackend
from .schemas import ProductionPlan, ReviewerReport


REVIEW_SCHEMA: dict[str, Any] = {
    "name": "lullaby_review",
    "type": "object",
    "required": ["passed", "scores", "failure_reasons"],
    "properties": {
        "passed": {"type": "boolean"},
        "scores": {"type": "object"},
        "failure_reasons": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "object"},
        "target_scenes": {"type": "array", "items": {"type": "integer"}},
        "prompt_revisions": {"type": "object"},
        "first_frame_prompt_revisions": {"type": "object"},
        "audio_prompt_revision": {"type": "string"},
        "subtitle_timing_adjustments": {"type": "object"},
        "mix_adjustments": {"type": "object"},
    },
}

AGGREGATED_REVIEW_SCHEMA: dict[str, Any] = {
    "name": "lullaby_aggregated_reviews",
    "type": "object",
    "required": ["reports"],
    "properties": {
        "reports": {
            "type": "array",
            "items": REVIEW_SCHEMA,
        }
    },
}


@dataclass(slots=True)
class ModelReviewConfig:
    review_backend: str = "mock"
    vlm_command: str | None = None
    ffmpeg_bin: str = "ffmpeg"
    strict_lullaby_review: bool = True


def _frame_path(review_frames_dir: Path, scene_num: int, label: str) -> Path:
    return review_frames_dir / f"scene_{scene_num:02d}_{label}.jpg"


def sample_scene_frames(
    plan: ProductionPlan,
    generated_dir: Path,
    review_frames_dir: Path,
) -> dict[int, list[str]]:
    review_frames_dir.mkdir(parents=True, exist_ok=True)
    samples: dict[int, list[str]] = {}
    if cv2 is None:
        return samples
    for scene in plan.scenes:
        clip = generated_dir / f"{scene.scene_num:02d}_01.mp4"
        if not clip.exists():
            samples[scene.scene_num] = []
            continue
        cap = cv2.VideoCapture(str(clip))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frames <= 0:
            cap.release()
            samples[scene.scene_num] = []
            continue
        indices = {
            "first": 0,
            "middle": max(0, frames // 2),
            "last": max(0, frames - 1),
        }
        scene_samples = []
        for label, frame_index in indices.items():
            output = _frame_path(review_frames_dir, scene.scene_num, label)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok and frame is not None:
                cv2.imwrite(str(output), frame)
                scene_samples.append(str(output))
        cap.release()
        samples[scene.scene_num] = scene_samples
    return samples


def _scene_context(plan: ProductionPlan, frame_samples: dict[int, list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_num": scene.scene_num,
            "lyric": scene.subtitle_text,
            "planned_start_seconds": scene.start_seconds,
            "planned_end_seconds": scene.end_seconds,
            "video_prompt": scene.video_prompt,
            "first_frame_prompt": scene.first_frame_prompt,
            "characters": [asdict(profile) for profile in plan.character_bank],
            "frame_paths": frame_samples.get(scene.scene_num, []),
        }
        for scene in plan.scenes
    ]


def _prompt_for_reviewer(reviewer: str) -> str:
    base = (
        "You are a strict reviewer for generated toddler lullaby videos. Return only JSON matching the schema. "
        "Use the provided frame paths and context as evidence. Scores are 0.0 to 1.0. "
        "If a problem can be fixed by changing prompts, include prompt_revisions keyed by scene number. "
        "If audio should be regenerated, include audio_prompt_revision. If visual clips should be regenerated, "
        "include target_scenes."
    )
    reviewer_focus = {
        "VisualSafetyReviewAgent": (
            "Review frame images for child safety, scary content, inappropriate imagery, visual clutter, "
            "generated text/letters, and whether the visuals are calm, warm, and engaging for toddlers."
        ),
        "StoryAlignmentReviewAgent": (
            "Review whether each scene's frames visibly match the assigned lyric and story beat, and whether "
            "the sequence is coherent as a lullaby video."
        ),
        "ContinuityReviewAgent": (
            "Review character identity, setting, palette, and visual-style continuity across scenes. Use the "
            "character bank and reference metadata as constraints."
        ),
        "AudioReviewAgent": (
            "Review audio evidence for lullaby quality: clear pleasant vocals, complete lyrics, gentle music, "
            "good mix/loudness, no harsh or scary sound, and a child-friendly bedtime mood."
        ),
        "AudioVisualSyncReviewAgent": (
            "Review the pairing of video frames, lyric timing, subtitles, and audio transcript. Check whether "
            "lyrics occur near the intended scene, whether scene mood fits the sung line, and whether fixes "
            "should target video prompts, audio prompt, subtitles, or clip timing."
        ),
    }
    return f"{base} {reviewer_focus[reviewer]}"


def _aggregated_review_prompt() -> str:
    return (
        "You are a panel of strict reviewers for generated toddler lullaby videos. Return only JSON with a "
        "'reports' array. Include exactly one report for each reviewer: VisualSafetyReviewAgent, "
        "StoryAlignmentReviewAgent, ContinuityReviewAgent, AudioReviewAgent, AudioVisualSyncReviewAgent. "
        "Use the provided sampled frame paths, final video path, subtitles, scene prompts, character bank, "
        "WhisperX transcript/timing, and stream/duration evidence. Do not pass unsafe, scary, cluttered, "
        "off-lyric, visually inconsistent, harsh-sounding, badly mixed, mistimed, or non-child-friendly output. "
        "If a fix is needed, include target_scenes, prompt_revisions, first_frame_prompt_revisions, "
        "audio_prompt_revision, subtitle_timing_adjustments, or mix_adjustments as appropriate. Scores are 0.0 to 1.0."
    )


def _normalize_model_report(reviewer: str, response: dict[str, Any]) -> ReviewerReport:
    raw_evidence = response.get("evidence")
    if isinstance(raw_evidence, dict):
        evidence = dict(raw_evidence)
    elif raw_evidence is None:
        evidence = {}
    else:
        evidence = {"model_evidence": raw_evidence}
    for key in [
        "target_scenes",
        "prompt_revisions",
        "first_frame_prompt_revisions",
        "audio_prompt_revision",
        "subtitle_timing_adjustments",
        "mix_adjustments",
    ]:
        if key in response and key not in evidence:
            evidence[key] = response[key]
    failures = [str(item) for item in response.get("failure_reasons", [])]
    return ReviewerReport(
        reviewer=reviewer,
        passed=bool(response.get("passed", False)),
        scores={str(key): float(value) for key, value in dict(response.get("scores") or {}).items() if isinstance(value, (int, float))},
        failure_reasons=failures,
        evidence=evidence,
    )


def _normalize_aggregated_reports(response: dict[str, Any]) -> list[ReviewerReport]:
    reports = []
    for item in response.get("reports", []) or []:
        if not isinstance(item, dict):
            continue
        reviewer = str(item.get("reviewer") or "")
        if reviewer:
            reports.append(_normalize_model_report(reviewer, item))
    expected = {
        "VisualSafetyReviewAgent",
        "StoryAlignmentReviewAgent",
        "ContinuityReviewAgent",
        "AudioReviewAgent",
        "AudioVisualSyncReviewAgent",
    }
    present = {item.reviewer for item in reports}
    for reviewer in sorted(expected - present):
        reports.append(
            ReviewerReport(
                reviewer=reviewer,
                passed=False,
                scores={"model_response": 0.0},
                failure_reasons=["missing_model_report"],
                evidence={"expected_reviewers": sorted(expected), "present_reviewers": sorted(present)},
            )
        )
    return reports


def _call_reviewer(
    backend: AgentBackend,
    reviewer: str,
    context: dict[str, Any],
) -> ReviewerReport:
    try:
        response = backend.generate_json(_prompt_for_reviewer(reviewer), REVIEW_SCHEMA, {**context, "response_key": reviewer})
    except Exception as exc:  # noqa: BLE001 - reviewer failure is review evidence
        return ReviewerReport(
            reviewer=reviewer,
            passed=False,
            scores={"model_call": 0.0},
            failure_reasons=["model_review_failed"],
            evidence={"error": str(exc)},
        )
    return _normalize_model_report(reviewer, response)


def model_review_reports(
    plan: ProductionPlan,
    *,
    generated_dir: Path,
    final_video_path: Path,
    subtitle_path: Path | None,
    review_frames_dir: Path,
    config: ModelReviewConfig,
    artifact_checks: dict[str, bool],
    streams: dict[str, bool],
    duration: float,
    planned_duration: float,
    whisperx_alignment: dict[str, Any] | None,
) -> list[ReviewerReport]:
    if config.review_backend != "command":
        return []
    if not config.vlm_command:
        return [
            ReviewerReport(
                reviewer=reviewer,
                passed=False,
                scores={"model_config": 0.0},
                failure_reasons=["missing_vlm_command"],
                evidence={"review_backend": config.review_backend},
            )
            for reviewer in [
                "VisualSafetyReviewAgent",
                "StoryAlignmentReviewAgent",
                "ContinuityReviewAgent",
                "AudioReviewAgent",
                "AudioVisualSyncReviewAgent",
            ]
        ]

    frame_samples = sample_scene_frames(plan, generated_dir, review_frames_dir)
    context = {
        "strict_lullaby_review": config.strict_lullaby_review,
        "final_video_path": str(final_video_path),
        "subtitle_path": str(subtitle_path) if subtitle_path else None,
        "artifact_checks": artifact_checks,
        "streams": streams,
        "duration_seconds": duration,
        "planned_duration_seconds": planned_duration,
        "whisperx_alignment": whisperx_alignment,
        "characters": [asdict(profile) for profile in plan.character_bank],
        "scenes": _scene_context(plan, frame_samples),
        "acceptance_thresholds": {
            "minimum_visual_safety": 0.9,
            "minimum_story_alignment": 0.8,
            "minimum_continuity": 0.8,
            "minimum_audio_quality": 0.8,
            "minimum_audio_visual_sync": 0.8,
        },
    }
    backend = CommandAgentBackend(config.vlm_command)
    try:
        response = backend.generate_json(_aggregated_review_prompt(), AGGREGATED_REVIEW_SCHEMA, {**context, "response_key": "ReviewAggregatorAgent"})
        return _normalize_aggregated_reports(response)
    except Exception as exc:  # noqa: BLE001 - reviewer failure is review evidence
        return [
            ReviewerReport(
                reviewer=reviewer,
                passed=False,
                scores={"model_call": 0.0},
                failure_reasons=["model_review_failed"],
                evidence={"error": str(exc)},
            )
            for reviewer in [
                "VisualSafetyReviewAgent",
                "StoryAlignmentReviewAgent",
                "ContinuityReviewAgent",
                "AudioReviewAgent",
                "AudioVisualSyncReviewAgent",
            ]
        ]
