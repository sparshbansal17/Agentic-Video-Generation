from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .audio_director import build_audio_plan, load_story_hints
from .backends import write_backend_manifest
from .evaluation import evaluate_manifest, write_evaluation
from .mixer import build_mix_manifest, write_mix_manifest


@dataclass(slots=True)
class RunStage:
    name: str
    status: str
    outputs: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class AgenticRunResult:
    output_dir: Path
    run_manifest: dict[str, Any]
    audio_plan_path: Path
    mix_manifest_path: Path
    backend_manifest_path: Path
    evaluation_path: Path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_audio_artifacts(
    *,
    rhyme_text: str,
    story_json: str | Path | None,
    output_dir: str | Path,
    target_duration: float | None,
    mode: str = "voice_bed",
    voice_backend: str = "f5_tts",
    music_backend: str = "musicgen",
    nested_audio_dir: bool = True,
) -> AgenticRunResult:
    story_summary, scene_hints = load_story_hints(story_json)
    plan = build_audio_plan(
        rhyme_text,
        scene_hints=scene_hints,
        story_summary=story_summary,
        target_duration_seconds=target_duration,
        mode=mode,  # type: ignore[arg-type]
        voice_backend=voice_backend,
        music_backend=music_backend,
    )

    root = Path(output_dir)
    audio_dir = root / "audio" if nested_audio_dir else root
    audio_plan_path = _write_json(audio_dir / "audio_plan.json", plan.to_dict())
    mix_manifest = build_mix_manifest(plan)
    mix_manifest_path = write_mix_manifest(audio_dir / "mix_manifest.json", mix_manifest)
    backend_manifest_path = write_backend_manifest(
        audio_dir / "backend_invocations.json",
        {
            "duration": plan.target_duration_seconds,
            "music_prompt": plan.music_prompt,
            "lyrics_file": "lyrics.txt",
            "prompt_file": "audio_plan.json",
            "output_file": "out.wav",
        },
    )
    evaluation_path = write_evaluation(
        audio_dir / "audio_evaluation_report.json",
        evaluate_manifest(plan, mix_manifest),
    )
    (audio_dir / "lyrics.txt").write_text("\n".join(plan.lyrics) + "\n", encoding="utf-8")

    return AgenticRunResult(
        output_dir=root,
        run_manifest={},
        audio_plan_path=audio_plan_path,
        mix_manifest_path=mix_manifest_path,
        backend_manifest_path=backend_manifest_path,
        evaluation_path=evaluation_path,
    )


def run_agentic(
    *,
    rhyme_file: str | Path,
    output_dir: str | Path,
    story_json: str | Path | None = None,
    target_duration: float | None = None,
    mode: str = "voice_bed",
    voice_backend: str = "f5_tts",
    music_backend: str = "musicgen",
    dry_run: bool = True,
) -> AgenticRunResult:
    if not dry_run:
        raise NotImplementedError("real backend execution is not implemented; run with dry_run=True")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rhyme_path = Path(rhyme_file)
    rhyme_text = rhyme_path.read_text(encoding="utf-8")

    audio_result = write_audio_artifacts(
        rhyme_text=rhyme_text,
        story_json=story_json,
        output_dir=root,
        target_duration=target_duration,
        mode=mode,
        voice_backend=voice_backend,
        music_backend=music_backend,
        nested_audio_dir=True,
    )

    video_stage = RunStage(
        name="storymem_video",
        status="pending",
        outputs=["video/final_video.mp4"],
        command=[
            "python",
            "pipeline.py",
            "--story-json",
            str(story_json or ""),
            "--output-dir",
            str(root / "video"),
        ],
        notes="Recorded for orchestration only; StoryMem generation is not invoked in dry-run mode.",
    )
    stages = [
        RunStage(
            name="audio_plan",
            status="complete",
            outputs=[
                str(audio_result.audio_plan_path.relative_to(root)),
                str(audio_result.mix_manifest_path.relative_to(root)),
                str(audio_result.backend_manifest_path.relative_to(root)),
                str(audio_result.evaluation_path.relative_to(root)),
            ],
        ),
        video_stage,
        RunStage(
            name="evaluation",
            status="complete",
            outputs=[str(audio_result.evaluation_path.relative_to(root))],
            notes="Audio manifest evaluation only; media quality scoring waits for generated assets.",
        ),
    ]
    manifest = {
        "version": "1.0",
        "dry_run": dry_run,
        "inputs": {
            "rhyme_file": str(rhyme_path),
            "story_json": str(story_json) if story_json else None,
            "target_duration_seconds": target_duration,
            "mode": mode,
        },
        "stages": [asdict(stage) for stage in stages],
        "next_actions": [
            "implement StoryMem video runner",
            "implement selected audio backend runner",
            "run WhisperX alignment after voice generation",
        ],
    }
    run_manifest = _write_json(root / "run_manifest.json", manifest)
    audio_result.run_manifest.update({"path": str(run_manifest), **manifest})
    return audio_result
