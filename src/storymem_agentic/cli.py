from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .orchestrator import run_workflow
from .runner import run_agentic, run_audio_postprocess, write_audio_artifacts


def plan_audio(args: argparse.Namespace) -> int:
    rhyme_text = Path(args.rhyme_file).read_text(encoding="utf-8")
    result = write_audio_artifacts(
        rhyme_text=rhyme_text,
        story_json=args.story_json,
        output_dir=args.output_dir,
        target_duration=args.target_duration,
        mode=args.mode,
        voice_backend=args.voice_backend,
        music_backend=args.music_backend,
        nested_audio_dir=False,
    )
    print(result.audio_plan_path)
    return 0


def agentic_run(args: argparse.Namespace) -> int:
    result = run_agentic(
        rhyme_file=args.rhyme_file,
        output_dir=args.output_dir,
        story_json=args.story_json,
        target_duration=args.target_duration,
        mode=args.mode,
        voice_backend=args.voice_backend,
        music_backend=args.music_backend,
        dry_run=args.dry_run,
    )
    print(result.output_dir / "run_manifest.json")
    return 0


def postprocess_audio(args: argparse.Namespace) -> int:
    result = run_audio_postprocess(
        rhyme_file=args.rhyme_file,
        story_json=args.story_json,
        output_dir=args.output_dir,
        final_video=args.final_video,
        mode=args.mode,
        voice_backend=args.voice_backend,
        music_backend=args.music_backend,
        media_audio_mode=args.media_audio_mode,
        audio_output_suffix=args.audio_output_suffix,
        audio_voice_style=args.audio_voice_style,
        ace_step_cmd=args.ace_step_cmd,
        vocal_cmd=args.vocal_cmd,
        backing_cmd=args.backing_cmd,
        musicgen_cmd=args.musicgen_cmd,
        song_cmd=args.song_cmd,
        voice_ref_audio=args.voice_ref_audio,
        voice_ref_text=args.voice_ref_text,
        ffmpeg_bin=args.ffmpeg_bin,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(result["media_output"] or result["manifest_path"])
    return 0


def workflow(args: argparse.Namespace) -> int:
    if (
        args.workflow_mode == "iterate"
        and args.review_backend != "command"
        and not args.allow_mock_review
    ):
        print(
            "error: iterate requires --review-backend command so revision decisions use real reviewers. "
            "Pass --allow-mock-review only for tests or dry-run debugging.",
            file=sys.stderr,
        )
        return 2
    result = run_workflow(
        rhyme_file=args.rhyme_file,
        topic_or_name=args.topic or "",
        lyrics=args.lyrics,
        lyrics_file=args.lyrics_file,
        output_dir=args.output_dir,
        mode=args.workflow_mode,
        target_duration=args.target_duration,
        target_audience=args.target_audience,
        visual_style=args.visual_style,
        audio_style=args.audio_style,
        clip_count=args.clip_count,
        max_iterations=args.max_iterations,
        seed=args.seed,
        storymem_dir=args.storymem_dir,
        t2v_model_path=args.t2v_model_path,
        i2v_model_path=args.i2v_model_path,
        lora_weight_path=args.lora_weight_path,
        nproc_per_node=args.nproc_per_node,
        ffmpeg_bin=args.ffmpeg_bin,
        storymem_t5_cpu=args.storymem_t5_cpu,
        storymem_offload_model=args.storymem_offload_model,
        storymem_sample_steps=args.storymem_sample_steps,
        storymem_frame_num=args.storymem_frame_num,
        storymem_keyframe_mode=args.storymem_keyframe_mode,
        execute_video=args.execute_video,
        character_db_path=args.character_db,
        character_bank_path=args.character_bank,
        production_plan_path=args.production_plan,
        planner_backend=args.planner_backend,
        planner_command=args.planner_command,
        plan_critic_command=args.plan_critic_command,
        max_plan_revisions=args.max_plan_revisions,
        plan_validation_policy=args.plan_validation_policy,
        review_backend=args.review_backend,
        vlm_command=args.vlm_command,
        audio_review_command=args.audio_review_command,
        whisperx_command=args.whisperx_command,
        audio_aligner=args.audio_aligner,
        strict_lullaby_review=args.strict_lullaby_review,
        generate_audio=args.generate_audio,
        media_audio_mode=args.media_audio_mode,
        voice_backend=args.voice_backend,
        music_backend=args.music_backend,
        audio_output_suffix=args.audio_output_suffix,
        audio_voice_style=args.audio_voice_style,
        ace_step_cmd=args.ace_step_cmd,
        vocal_cmd=args.vocal_cmd,
        backing_cmd=args.backing_cmd,
        musicgen_cmd=args.musicgen_cmd,
        song_cmd=args.song_cmd,
        voice_ref_audio=args.voice_ref_audio,
        voice_ref_text=args.voice_ref_text,
        allow_scene_mix_debug=args.allow_scene_mix_debug,
        full_song_candidate_count=args.full_song_candidates,
        voice_candidate_count=args.voice_candidates,
        music_candidate_count=args.music_candidates,
    )
    print(result["latest_iteration"])
    return 0 if result["passed"] or args.workflow_mode in {"dry_run", "generate"} else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StoryMem Agentic orchestration CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan-audio", help="Create a timing-first audio plan and dry-run manifests")
    plan.add_argument("--rhyme-file", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--story-json")
    plan.add_argument("--target-duration", type=float)
    plan.add_argument("--mode", choices=["voice_bed", "full_song"], default="voice_bed")
    plan.add_argument("--voice-backend", default="f5_tts")
    plan.add_argument("--music-backend", default="musicgen")
    plan.set_defaults(func=plan_audio)
    run = sub.add_parser("run", help="Create an agentic dry-run manifest across story, video, audio, and evaluation stages")
    run.add_argument("--rhyme-file", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--story-json")
    run.add_argument("--target-duration", type=float)
    run.add_argument("--mode", choices=["voice_bed", "full_song"], default="voice_bed")
    run.add_argument("--voice-backend", default="f5_tts")
    run.add_argument("--music-backend", default="musicgen")
    run.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    run.set_defaults(func=agentic_run)
    post = sub.add_parser("postprocess-audio", help="Plan, render, mux, and evaluate audio for an existing StoryMem video")
    post.add_argument("--rhyme-file", required=True)
    post.add_argument("--story-json", required=True)
    post.add_argument("--output-dir", required=True)
    post.add_argument("--final-video", required=True)
    post.add_argument("--mode", choices=["voice_bed", "full_song"], default="full_song")
    post.add_argument("--voice-backend", default="ace_step_full_song")
    post.add_argument("--music-backend", default="ace_step_full_song")
    post.add_argument("--media-audio-mode", choices=["full_song", "separate_stems", "scene_lyrics_mix", "hybrid_voice_bed"], default="full_song")
    post.add_argument("--audio-output-suffix", default="_with_music")
    post.add_argument("--audio-voice-style", default="gentle adult lullaby")
    post.add_argument("--ace-step-cmd")
    post.add_argument("--vocal-cmd")
    post.add_argument("--backing-cmd")
    post.add_argument("--musicgen-cmd")
    post.add_argument("--song-cmd")
    post.add_argument("--voice-ref-audio")
    post.add_argument("--voice-ref-text")
    post.add_argument("--ffmpeg-bin")
    post.add_argument("--seed", type=int, default=0)
    post.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    post.set_defaults(func=postprocess_audio)
    for name, help_text, mode in [
        ("plan", "Create production/audio planning artifacts without GPU generation", "dry_run"),
        ("dry-run", "Create production/audio/evaluation/revision artifacts without GPU generation", "dry_run"),
        ("generate", "Create one iteration and optionally execute StoryMem video generation", "generate"),
        ("iterate", "Run generation/evaluation/revision loop until pass or max iterations", "iterate"),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--topic")
        command.add_argument("--rhyme-file")
        command.add_argument("--lyrics")
        command.add_argument("--lyrics-file")
        command.add_argument("--output-dir", required=True)
        command.add_argument("--target-duration", type=float)
        command.add_argument("--target-audience", default="toddlers")
        command.add_argument("--visual-style", default="bright rounded toddler-safe bedtime storybook animation")
        command.add_argument("--audio-style", default="warm clear nursery singalong, soft music box and celesta")
        command.add_argument("--clip-count", type=int)
        command.add_argument("--max-iterations", type=int, default=1)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--character-db")
        command.add_argument("--character-bank")
        command.add_argument("--production-plan", help="Reuse an approved production_plan.json without replanning")
        command.add_argument("--planner-backend", choices=["mock", "command"], default="mock")
        command.add_argument("--planner-command")
        command.add_argument("--plan-critic-command")
        command.add_argument("--max-plan-revisions", type=int, default=8)
        command.add_argument("--plan-validation-policy", choices=["block", "warn"], default="block")
        command.add_argument("--review-backend", choices=["mock", "command"], default="mock")
        command.add_argument("--allow-mock-review", action="store_true")
        command.add_argument("--vlm-command")
        command.add_argument("--audio-review-command", help="Audio/video-capable reviewer command, e.g. Qwen2.5-Omni")
        command.add_argument("--whisperx-command")
        command.add_argument("--audio-aligner", choices=["whisperx", "none"], default="whisperx")
        command.add_argument("--strict-lullaby-review", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--media-audio-mode", choices=["full_song", "separate_stems", "scene_lyrics_mix", "hybrid_voice_bed"], default="full_song")
        command.add_argument("--voice-backend", choices=["ace_step_full_song", "ace_step", "f5_tts", "cosyvoice", "yue_full_song"], default=None)
        command.add_argument("--music-backend", choices=["ace_step_full_song", "ace_step", "musicgen", "stable_audio"], default=None)
        command.add_argument("--audio-output-suffix", default="_with_music")
        command.add_argument("--audio-voice-style", default="gentle adult lullaby")
        command.add_argument("--ace-step-cmd")
        command.add_argument("--vocal-cmd")
        command.add_argument("--backing-cmd")
        command.add_argument("--musicgen-cmd")
        command.add_argument("--song-cmd")
        command.add_argument("--voice-ref-audio")
        command.add_argument("--voice-ref-text")
        command.add_argument(
            "--allow-scene-mix-debug",
            action="store_true",
            help=argparse.SUPPRESS,  # deprecated compatibility flag; scene mixing is disabled
        )
        command.add_argument("--full-song-candidates", type=int, default=8)
        command.add_argument("--voice-candidates", type=int, default=4)
        command.add_argument("--music-candidates", type=int, default=1)
        command.add_argument("--storymem-dir")
        command.add_argument("--t2v-model-path")
        command.add_argument("--i2v-model-path")
        command.add_argument("--lora-weight-path")
        command.add_argument("--nproc-per-node", type=int, default=8)
        command.add_argument("--ffmpeg-bin", default="ffmpeg")
        command.add_argument("--storymem-t5-cpu", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument("--storymem-offload-model", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument("--storymem-sample-steps", type=int)
        command.add_argument("--storymem-frame-num", type=int)
        command.add_argument("--storymem-keyframe-mode", choices=["hps", "simple", "off"], default="hps")
        command.add_argument("--execute-video", action=argparse.BooleanOptionalAction, default=False)
        command.set_defaults(func=workflow, workflow_mode=mode)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
