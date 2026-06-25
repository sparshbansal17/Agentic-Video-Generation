from __future__ import annotations

import argparse
import json
from pathlib import Path

from storymem_agentic.media_evaluator import evaluate_iteration
from storymem_agentic.orchestrator import run_whisperx_command, write_json
from storymem_agentic.feedback import apply_revision_plan, build_revision_plan
from storymem_agentic.schemas import ProductionPlan


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun WhisperX and evaluate an existing agentic iteration.")
    parser.add_argument("--iteration-dir", required=True)
    parser.add_argument("--whisperx-command", required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--media-file", default="generated/generated_subtitled_with_music.mp4")
    parser.add_argument("--output-name", default="whisperx_alignment_fresh.json")
    parser.add_argument("--review-backend", choices=["mock", "command"], default="mock")
    parser.add_argument("--vlm-command")
    parser.add_argument("--exit-zero", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    iteration_dir = Path(args.iteration_dir).resolve()
    plan_path = iteration_dir / "production_plan.json"
    media_path = iteration_dir / args.media_file
    subtitle_path = iteration_dir / "generated" / "subtitles.ass"
    output_path = iteration_dir / args.output_name
    whisperx_dir = iteration_dir / "whisperx_fresh"

    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    if not media_path.exists():
        raise FileNotFoundError(media_path)

    if output_path.exists():
        output_path.unlink()

    alignment_path = run_whisperx_command(
        command_template=args.whisperx_command,
        audio_file=media_path,
        output_dir=whisperx_dir,
        output_file=output_path,
    )
    if alignment_path is None or not alignment_path.exists():
        raise RuntimeError("WhisperX command did not produce an alignment JSON")

    plan = ProductionPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
    report = evaluate_iteration(
        plan,
        iteration_dir / "generated",
        final_video=media_path,
        subtitle_path=subtitle_path,
        whisperx_alignment_path=alignment_path,
        ffmpeg_bin=args.ffmpeg_bin,
        review_backend=args.review_backend,
        vlm_command=args.vlm_command,
        review_frames_dir=iteration_dir / "review_frames_fresh",
    )
    report_path = iteration_dir / "evaluation_report_fresh_whisperx.json"
    write_json(report_path, report.to_dict())
    revision = build_revision_plan(plan, report)
    revision_path = iteration_dir / "revision_plan_fresh_review.json"
    write_json(revision_path, revision.to_dict())
    revised_plan = apply_revision_plan(plan, revision)
    revised_plan_path = iteration_dir / "production_plan_revised_from_fresh_review.json"
    write_json(revised_plan_path, revised_plan.to_dict())
    preview_path = iteration_dir / "planner_update_preview_fresh_review.json"
    write_json(
        preview_path,
        {
            "source_production_plan": str(plan_path),
            "revised_production_plan": str(revised_plan_path),
            "target_scenes": revision.target_scenes,
            "preserve_scenes": revision.preserve_scenes,
            "regenerate_audio": revision.regenerate_audio,
            "prompt_revisions": revision.prompt_revisions,
            "first_frame_prompt_revisions": revision.first_frame_prompt_revisions,
            "audio_prompt_revision": revision.audio_prompt_revision,
            "subtitle_timing_adjustments": revision.subtitle_timing_adjustments,
            "mix_adjustments": revision.mix_adjustments,
            "rationale": revision.rationale,
        },
    )

    whisperx = report.whisperx_alignment or {}
    summary = {
        "passed": report.passed,
        "failure_reasons": report.failure_reasons,
        "word_error_rate": whisperx.get("word_error_rate"),
        "wer_threshold": whisperx.get("wer_threshold"),
        "drift_tolerance_seconds": whisperx.get("drift_tolerance_seconds"),
        "transcript": whisperx.get("transcript"),
        "alignment_path": str(alignment_path),
        "report_path": str(report_path),
        "revision_path": str(revision_path),
        "revised_plan_path": str(revised_plan_path),
        "planner_update_preview_path": str(preview_path),
    }
    print(json.dumps(summary, indent=2))
    return 0 if args.exit_zero or report.passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
