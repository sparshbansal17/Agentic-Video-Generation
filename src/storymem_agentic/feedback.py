from __future__ import annotations

from .schemas import EvaluationReport, ProductionPlan, RevisionPlan, quantize_seconds


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _dedupe_guidance(parts: list[str]) -> str:
    seen = set()
    output = []
    for part in parts:
        for sentence in str(part).replace("|", ". ").split("."):
            compact = " ".join(sentence.split()).strip()
            if not compact:
                continue
            key = compact.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(compact)
    return ". ".join(output)


def build_revision_plan(plan: ProductionPlan, report: EvaluationReport) -> RevisionPlan:
    failed = set(report.regeneration_targets)
    reviewer_evidence = {item.reviewer: item.evidence for item in report.reviewer_reports}

    if report.passed:
        return RevisionPlan(
            version="1.0",
            status="accepted",
            preserve_scenes=[scene.scene_num for scene in plan.scenes],
            rationale=["evaluation passed"],
            reviewer_evidence=reviewer_evidence,
        )

    model_prompt_revisions: dict[str, str] = {}
    model_first_frame_revisions: dict[str, str] = {}
    model_subtitle_adjustments: dict[str, dict[str, float]] = {}
    model_mix_adjustments: dict[str, object] = {}
    model_audio_revisions: list[str] = []
    for reviewer in report.reviewer_reports:
        evidence = reviewer.evidence or {}
        for scene_num in evidence.get("target_scenes", []) or []:
            try:
                failed.add(int(scene_num))
            except (TypeError, ValueError):
                continue
        for key, value in _mapping(evidence.get("prompt_revisions")).items():
            model_prompt_revisions[str(key)] = str(value)
        for key, value in _mapping(evidence.get("first_frame_prompt_revisions")).items():
            model_first_frame_revisions[str(key)] = str(value)
        for key, value in _mapping(evidence.get("subtitle_timing_adjustments")).items():
            if isinstance(value, dict):
                model_subtitle_adjustments[str(key)] = value
        model_mix_adjustments.update(_mapping(evidence.get("mix_adjustments")))
        if evidence.get("audio_prompt_revision"):
            model_audio_revisions.append(str(evidence["audio_prompt_revision"]))
    for scene in plan.scenes:
        if any(dep in failed for dep in scene.regeneration_dependencies):
            failed.add(scene.scene_num)

    failed_reviewers = {item.reviewer for item in report.reviewer_reports if not item.passed}
    whisperx_failed = "WhisperXLyricTimingAgent" in failed_reviewers
    audio_failed = bool({"AudioReviewAgent", "AudioArtifactReviewAgent", "AudioVisualSyncReviewAgent", "WhisperXLyricTimingAgent"} & failed_reviewers)
    visual_or_artifact_failed = bool(failed_reviewers - {"AudioReviewAgent", "AudioVisualSyncReviewAgent", "WhisperXLyricTimingAgent"})
    video_artifacts_ok = all(
        report.artifact_checks.get(name, False)
        for name in ["clip_count", "final_video_exists", "has_video_stream", "has_subtitles", "duration_match"]
    )
    audio_only_failure = audio_failed and not visual_or_artifact_failed and video_artifacts_ok and not failed
    target_scenes = [] if audio_only_failure else sorted(failed or [scene.scene_num for scene in plan.scenes])
    preserve = [scene.scene_num for scene in plan.scenes if scene.scene_num not in target_scenes]
    prompt_revisions = {
        str(scene_num): (
            "Revise prompt for clearer lyric-scene match, stronger character-bank repetition, "
            "less clutter, and no generated text."
        )
        for scene_num in target_scenes
    }
    prompt_revisions.update(model_prompt_revisions)
    first_frame_prompt_revisions = {
        str(scene_num): "Keep the accepted character design visible in the first frame; avoid text and clutter."
        for scene_num in target_scenes
    }
    first_frame_prompt_revisions.update(model_first_frame_revisions)
    failed_audio_lines = []
    for item in (report.whisperx_alignment or {}).get("lines", []):
        line_index = item.get("line_index")
        matched = int(item.get("matched_word_count") or 0)
        expected = int(item.get("expected_word_count") or 0)
        if item.get("observed_start_seconds") is None or item.get("observed_end_seconds") is None or matched < expected:
            failed_audio_lines.append(f"line {line_index}: {item.get('text')}")
        for token, counts in (item.get("repeated_word_omissions") or {}).items():
            failed_audio_lines.append(
                f'line {line_index} omitted repeated "{token}" ({counts.get("matched")} of {counts.get("expected")}); regenerate line {line_index} with each "{token}" clearly separated'
            )
    exact_line_guidance = (
        " Pay special attention to these exact lines: " + " | ".join(failed_audio_lines) + "."
        if failed_audio_lines
        else ""
    )

    audio_revision = (
        "Regenerate one continuous full-song lullaby with exact lyrics, clearer diction, gentler timing, "
        "and no per-scene song fragments."
        + exact_line_guidance
        if audio_failed
        else None
    )
    if model_audio_revisions:
        audio_revision = _dedupe_guidance([item for item in [audio_revision, *model_audio_revisions] if item])

    return RevisionPlan(
        version="1.0",
        status="needs_iteration",
        target_scenes=target_scenes,
        regenerate_audio=audio_failed or not report.artifact_checks.get("has_audio_stream", True),
        prompt_revisions=prompt_revisions,
        first_frame_prompt_revisions=first_frame_prompt_revisions,
        audio_prompt_revision=audio_revision,
        lyric_timing_adjustments={
            str(item.get("line_index")): {
                "observed_start_seconds": item.get("observed_start_seconds"),
                "observed_end_seconds": item.get("observed_end_seconds"),
            }
            for item in (report.whisperx_alignment or {}).get("lines", [])
            if whisperx_failed
        },
        subtitle_timing_adjustments={
            str(item.get("line_index")): {
                "start_seconds": item.get("observed_start_seconds"),
                "end_seconds": item.get("observed_end_seconds"),
            }
            for item in (report.whisperx_alignment or {}).get("lines", [])
            if whisperx_failed and item.get("observed_start_seconds") is not None and item.get("observed_end_seconds") is not None
        }
        | model_subtitle_adjustments,
        mix_adjustments=({"check_fades": True, "avoid_cutting_lyric_boundaries": True} if audio_failed else {}) | model_mix_adjustments,
        rationale=report.failure_reasons or ["scene-level verifier failure"],
        reviewer_evidence=reviewer_evidence,
        preserve_scenes=preserve,
    )


def apply_revision_plan(plan: ProductionPlan, revision: RevisionPlan) -> ProductionPlan:
    revised = ProductionPlan.from_dict(plan.to_dict())
    if revision.status == "accepted":
        return revised

    for scene in revised.scenes:
        key = str(scene.scene_num)
        prompt_revision = revision.prompt_revisions.get(key)
        if prompt_revision:
            scene.video_prompt = f"{scene.video_prompt} Revision guidance: {prompt_revision}"
        first_frame_revision = revision.first_frame_prompt_revisions.get(key)
        if first_frame_revision:
            scene.first_frame_prompt = f"{scene.first_frame_prompt} Revision guidance: {first_frame_revision}"

    if revision.audio_prompt_revision:
        revised.music_prompt = f"{revised.music_prompt}. Revision guidance: {revision.audio_prompt_revision}"

    if revision.clip_duration_adjustments:
        start = 0.0
        by_scene = {scene.scene_num: scene for scene in revised.scenes}
        by_segment = {segment.index: segment for segment in revised.lyric_segments}
        for scene in revised.scenes:
            key = str(scene.scene_num)
            current_duration = scene.end_seconds - scene.start_seconds
            duration = float(revision.clip_duration_adjustments.get(key, current_duration))
            duration = max(0.5, duration)
            scene.start_seconds = quantize_seconds(start, revised.target_fps)
            scene.end_seconds = quantize_seconds(start + duration, revised.target_fps)
            segment = by_segment[scene.lyric_segment_index]
            segment.start_seconds = scene.start_seconds
            segment.end_seconds = scene.end_seconds
            start = scene.end_seconds

    revised.validate()
    return revised
