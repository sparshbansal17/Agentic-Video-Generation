from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_agentic, write_audio_artifacts


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
