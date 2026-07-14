from __future__ import annotations

import subprocess
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal test environments
    cv2 = None

from .alignment import analyze_whisperx_alignment, load_whisperx_words
from .audio_quality import probe_audio_quality
from .review_agents import ModelReviewConfig, model_review_reports
from .schemas import EvaluationReport, ProductionPlan, ReviewerReport, SceneEvaluation


def video_duration_seconds(path: Path) -> float:
    if cv2 is None:
        return 0.0
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or not frames:
        return 0.0
    return float(frames / fps)


def media_streams(path: Path, ffmpeg_bin: str = "ffmpeg") -> dict[str, bool]:
    if not path.exists():
        return {"has_video": False, "has_audio": False}
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout + result.stderr
    return {"has_video": "Video:" in output, "has_audio": "Audio:" in output}


def evaluate_iteration(
    plan: ProductionPlan,
    generated_dir: str | Path,
    *,
    final_video: str | Path | None = None,
    subtitle_path: str | Path | None = None,
    whisperx_alignment_path: str | Path | None = None,
    ffmpeg_bin: str = "ffmpeg",
    dry_run: bool = False,
    strict_lullaby_review: bool = True,
    review_backend: str = "mock",
    vlm_command: str | None = None,
    audio_review_command: str | None = None,
    review_frames_dir: str | Path | None = None,
    enforce_lyric_scene_windows: bool = True,
) -> EvaluationReport:
    root = Path(generated_dir)
    expected_clips = [root / f"{scene.scene_num:02d}_01.mp4" for scene in plan.scenes]
    if final_video is None:
        final_video_path = root / "final_video.mp4"
    else:
        final_video_path = Path(final_video)

    streams = media_streams(final_video_path, ffmpeg_bin) if final_video_path.exists() else {"has_video": False, "has_audio": False}
    duration = video_duration_seconds(final_video_path) if final_video_path.exists() else 0.0
    planned_duration = max(scene.end_seconds for scene in plan.scenes)
    duration_ok = dry_run or abs(duration - planned_duration) <= max(1.5, planned_duration * 0.25)

    artifact_checks = {
        "storymem_compatible_plan": True,
        "clip_count": dry_run or all(path.exists() for path in expected_clips),
        "final_video_exists": dry_run or final_video_path.exists(),
        "has_video_stream": dry_run or streams["has_video"],
        "has_audio_stream": dry_run or streams["has_audio"],
        "has_subtitles": dry_run or (Path(subtitle_path).exists() if subtitle_path else False),
        "duration_match": duration_ok,
    }
    reviewer_reports = [
        ReviewerReport(
            reviewer="ArtifactReviewAgent",
            passed=all(artifact_checks.values()),
            scores={name: 1.0 if ok else 0.0 for name, ok in artifact_checks.items()},
            failure_reasons=[name for name, ok in artifact_checks.items() if not ok],
            evidence={
                "final_video": str(final_video_path),
                "expected_clips": [str(path) for path in expected_clips],
                "streams": streams,
            },
        )
    ]
    if not dry_run and final_video_path.exists():
        technical = probe_audio_quality(
            final_video_path,
            expected_duration_seconds=planned_duration,
            ffmpeg_bin=ffmpeg_bin,
        )
        reviewer_reports.append(
            ReviewerReport(
                reviewer="AudioTechnicalGate",
                passed=bool(technical["passed"]),
                scores={
                    "duration_match": 0.0
                    if "audio_duration_out_of_tolerance" in technical["failure_reasons"]
                    else 1.0,
                    "audible": 0.0
                    if "audio_effectively_silent" in technical["failure_reasons"]
                    else 1.0,
                    "peak_safe": 0.0
                    if "audio_clipping_risk" in technical["failure_reasons"]
                    else 1.0,
                },
                failure_reasons=list(technical["failure_reasons"]),
                evidence=dict(technical["metrics"]),
            )
        )

    scene_reports = []
    regeneration_targets = []
    for scene, clip in zip(plan.scenes, expected_clips):
        reasons = []
        if not dry_run and not clip.exists():
            reasons.append("missing_clip")
        passed = not reasons
        if not passed:
            regeneration_targets.append(scene.scene_num)
        scene_reports.append(
            SceneEvaluation(
                scene_num=scene.scene_num,
                passed=passed,
                scores={
                    "artifact": 1.0 if passed else 0.0,
                    "vlm_prompt_adherence": 1.0 if dry_run else 0.0 if reasons else 0.8,
                },
                failure_reasons=reasons,
                evidence={
                    "clip_path": str(clip),
                    "lyric": scene.subtitle_text,
                    "character_bank": [profile.label for profile in plan.character_bank],
                },
            )
        )

    failure_reasons = [name for name, ok in artifact_checks.items() if not ok]
    visual_failures = []
    for scene in plan.scenes:
        prompt_lower = scene.video_prompt.lower()
        scene_failures = []
        if "no generated text" not in prompt_lower and strict_lullaby_review:
            scene_failures.append("missing_no_generated_text_constraint")
        if any(term in prompt_lower for term in ["blood", "weapon", "horror"]):
            scene_failures.append("unsafe_visual_terms")
        if scene_failures:
            visual_failures.extend(f"scene_{scene.scene_num}_{reason}" for reason in scene_failures)
            if scene.scene_num not in regeneration_targets:
                regeneration_targets.append(scene.scene_num)
    if review_backend != "command":
        reviewer_reports.extend(
            [
                ReviewerReport(
                    reviewer="VisualSafetyReviewAgent",
                    passed=not visual_failures,
                    scores={"child_safety": 0.0 if visual_failures else 1.0},
                    failure_reasons=visual_failures,
                    evidence={"strict": strict_lullaby_review},
                ),
                ReviewerReport(
                    reviewer="StoryAlignmentReviewAgent",
                    passed=True,
                    scores={"lyric_scene_match": 1.0},
                    evidence={"planned_lines": [scene.subtitle_text for scene in plan.scenes]},
                ),
                ReviewerReport(
                    reviewer="ContinuityReviewAgent",
                    passed=True,
                    scores={"character_consistency": 1.0},
                    evidence={
                        "characters": [
                            {
                                "label": profile.label,
                                "reference_image_paths": profile.reference_image_paths,
                                "continuity_constraints": profile.continuity_constraints,
                            }
                            for profile in plan.character_bank
                        ]
                    },
                ),
            ]
        )
    whisperx_alignment = None
    if whisperx_alignment_path and Path(whisperx_alignment_path).exists():
        aligned_words = load_whisperx_words(whisperx_alignment_path)
        whisperx_alignment = analyze_whisperx_alignment(
            [segment.text for segment in plan.lyric_segments],
            [(segment.start_seconds, segment.end_seconds) for segment in plan.lyric_segments],
            aligned_words,
            enforce_scene_windows=enforce_lyric_scene_windows,
        )
        reviewer_reports.append(
            ReviewerReport(
                reviewer="WhisperXLyricTimingAgent",
                passed=bool(whisperx_alignment["passed"]),
                scores={
                    "word_error_rate": float(whisperx_alignment["word_error_rate"]),
                    "word_accuracy": 1.0 - float(whisperx_alignment["word_error_rate"]),
                },
                failure_reasons=list(whisperx_alignment["failure_reasons"]),
                evidence=whisperx_alignment,
            )
        )
    elif not dry_run:
        reviewer_reports.append(
            ReviewerReport(
                reviewer="WhisperXLyricTimingAgent",
                passed=False,
                scores={"word_error_rate": 0.0},
                failure_reasons=["missing_whisperx_alignment"],
                evidence={"alignment_path": str(whisperx_alignment_path) if whisperx_alignment_path else None},
            )
        )
    audio_failures = []
    if not artifact_checks["has_audio_stream"]:
        audio_failures.append("missing_audio_stream")
    if not artifact_checks["duration_match"]:
        audio_failures.append("audio_video_duration_mismatch")
    if review_backend == "command":
        model_reports = model_review_reports(
            plan,
            generated_dir=root,
            final_video_path=final_video_path,
            subtitle_path=Path(subtitle_path) if subtitle_path else None,
            review_frames_dir=Path(review_frames_dir) if review_frames_dir else root.parent / "review_frames",
            config=ModelReviewConfig(
                review_backend=review_backend,
                vlm_command=vlm_command,
                ffmpeg_bin=ffmpeg_bin,
                strict_lullaby_review=strict_lullaby_review,
                audio_command=audio_review_command,
            ),
            artifact_checks=artifact_checks,
            streams=streams,
            duration=duration,
            planned_duration=planned_duration,
            whisperx_alignment=whisperx_alignment,
        )
        if whisperx_alignment and whisperx_alignment["failure_reasons"]:
            timing_reasons = list(whisperx_alignment["failure_reasons"])
            for report in model_reports:
                if report.reviewer != "AudioVisualSyncReviewAgent":
                    continue
                report.passed = False
                report.scores["line_timing"] = 0.0
                report.failure_reasons = list(dict.fromkeys([*report.failure_reasons, *timing_reasons]))
                report.evidence["whisperx_alignment_passed"] = False
                report.evidence["whisperx_failure_reasons"] = timing_reasons
        reviewer_reports.extend(model_reports)
        if audio_failures:
            reviewer_reports.append(
                ReviewerReport(
                    reviewer="AudioArtifactReviewAgent",
                    passed=False,
                    scores={"duration_match": 1.0 if artifact_checks["duration_match"] else 0.0},
                    failure_reasons=audio_failures,
                    evidence={"planned_duration_seconds": planned_duration, "observed_duration_seconds": duration},
                )
            )
    else:
        reviewer_reports.extend(
            [
                ReviewerReport(
                    reviewer="AudioReviewAgent",
                    passed=not audio_failures,
                    scores={"duration_match": 1.0 if artifact_checks["duration_match"] else 0.0},
                    failure_reasons=audio_failures,
                    evidence={"planned_duration_seconds": planned_duration, "observed_duration_seconds": duration},
                ),
                ReviewerReport(
                    reviewer="AudioVisualSyncReviewAgent",
                    passed=not (whisperx_alignment and whisperx_alignment["failure_reasons"]),
                    scores={"line_timing": 0.0 if whisperx_alignment and whisperx_alignment["failure_reasons"] else 1.0},
                    failure_reasons=list(whisperx_alignment["failure_reasons"]) if whisperx_alignment else [],
                    evidence={
                        "subtitle_path": str(subtitle_path) if subtitle_path else None,
                        "whisperx_alignment": whisperx_alignment,
                        "planned_line_timing": [
                            {
                                "line_index": segment.index,
                                "text": segment.text,
                                "start_seconds": segment.start_seconds,
                                "end_seconds": segment.end_seconds,
                            }
                            for segment in plan.lyric_segments
                        ],
                    },
                ),
            ]
        )
    reporter_failures = [reason for report in reviewer_reports for reason in report.failure_reasons]
    failure_reasons.extend(reason for reason in reporter_failures if reason not in failure_reasons)
    passed = all(artifact_checks.values()) and all(report.passed for report in scene_reports) and all(
        report.passed for report in reviewer_reports
    )
    if failure_reasons and not regeneration_targets:
        audio_only_reasons = {
            "missing_audio_stream",
            "audio_video_duration_mismatch",
            "missing_whisperx_alignment",
            "missing_observed_lyrics",
            "wer_above_threshold",
            "model_review_failed",
            "review_infrastructure_error",
        }
        audio_only = True
        for reason in failure_reasons:
            if reason in audio_only_reasons:
                continue
            if reason.startswith("line_") and (
                reason.endswith("_missing_words")
                or reason.endswith("_partial_words")
                or reason.endswith("_starts_before_scene")
                or reason.endswith("_ends_after_scene")
                or "_omitted_repeated_" in reason
                or reason.endswith("_final_line_incomplete")
            ):
                continue
            audio_only = False
            break
        if not audio_only:
            regeneration_targets = [scene.scene_num for scene in plan.scenes]

    return EvaluationReport(
        version="1.0",
        passed=passed,
        artifact_checks=artifact_checks,
        scene_reports=scene_reports,
        audio_scores={
            "duration_seconds": duration if not dry_run else None,
            "planned_duration_seconds": planned_duration,
        },
        reviewer_reports=reviewer_reports,
        whisperx_alignment=whisperx_alignment,
        regeneration_targets=regeneration_targets,
        failure_reasons=failure_reasons,
    )
