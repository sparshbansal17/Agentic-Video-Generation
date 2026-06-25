from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .backends import load_backend_config
from .schemas import AudioLinePlan, AudioMode, AudioPlan, SceneHint

_CONFIG_MIX = load_backend_config().get("mix", {})

DEFAULT_MIX = {
    "sample_rate": 48000,
    "target_lufs": -15.0,
    "true_peak_db": -1.5,
    "music_ducking_ratio": 6.0,
    "max_time_stretch_percent": 8.0,
    "line_boundary_tolerance_ms": 350,
    "duration_tolerance_ms": 250,
} | {str(key): value for key, value in _CONFIG_MIX.items()}

DEFAULT_REGENERATION_POLICY = {
    "voice_bed": ["regenerate_voice", "adjust_pauses", "regenerate_music", "adjust_mix"],
    "full_song": ["regenerate_full_song", "revise_song_prompt", "adjust_mix"],
    "never_segment_sung_song": True,
}


def normalize_lyrics(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def load_story_hints(path: str | Path | None) -> tuple[str, list[SceneHint]]:
    if not path:
        return "", []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = [SceneHint.from_story_scene(scene) for scene in data.get("scenes", [])]
    return str(data.get("story_overview", "")).strip(), scenes


def _allocate_lines(
    lyric_lines: list[str],
    scene_hints: list[SceneHint],
    target_duration: float,
) -> list[AudioLinePlan]:
    line_count = len(lyric_lines)
    pad = min(0.35, target_duration / max(line_count, 1) * 0.08)
    slot = target_duration / line_count
    plans: list[AudioLinePlan] = []
    for index, text in enumerate(lyric_lines, start=1):
        hint = scene_hints[index - 1] if index - 1 < len(scene_hints) else None
        start = round((index - 1) * slot + pad, 3)
        end = round(index * slot - pad, 3)
        plans.append(
            AudioLinePlan(
                index=index,
                text=text,
                scene_num=hint.scene_num if hint else index,
                start_seconds=start,
                end_seconds=max(end, start + 0.5),
                subtitle=hint.subtitle_text if hint and hint.subtitle_text else text,
                clip_duration_seconds=hint.duration_seconds if hint else None,
                vocal_style="gentle lullaby vocal" if text else "instrumental",
                music_style="soft music box, celesta, glockenspiel, warm strings",
                expected_scene_mood="calm bedtime wonder",
                boundary_behavior="fade" if index == line_count else "hold",
            )
        )
    return plans


def _scene_arc(scene_hints: Iterable[SceneHint]) -> str:
    notes = []
    for hint in scene_hints:
        summary = " ".join(hint.summary.split())[:140]
        if summary:
            notes.append(f"scene {hint.scene_num}: {summary}")
    return "; ".join(notes)


def build_audio_plan(
    rhyme_text: str,
    *,
    scene_hints: list[SceneHint] | None = None,
    story_summary: str = "",
    target_duration_seconds: float | None = None,
    mode: AudioMode = "voice_bed",
    voice_backend: str = "f5_tts",
    music_backend: str = "musicgen",
    aligner_backend: str = "whisperx",
    voice_style: str = "warm clear nursery-rhyme narrator, gentle bedtime delivery",
    music_style: str = "soft music box, celesta, glockenspiel, harp, warm strings, child-safe lullaby",
) -> AudioPlan:
    lyric_lines = normalize_lyrics(rhyme_text)
    if not lyric_lines:
        raise ValueError("rhyme_text did not contain any lyric lines")
    hints = scene_hints or []
    target_duration = float(target_duration_seconds or max(8.0, len(lyric_lines) * 4.0))
    arc = _scene_arc(hints)
    music_prompt = ", ".join(
        item for item in [music_style, voice_style if mode == "full_song" else "instrumental bed under clear voice", story_summary, arc] if item
    )
    plan = AudioPlan(
        version="1.0",
        mode=mode,
        target_duration_seconds=target_duration,
        lyrics=lyric_lines,
        story_summary=story_summary,
        music_prompt=music_prompt[:1200],
        voice_backend=voice_backend,
        music_backend=music_backend,
        aligner_backend=aligner_backend,
        lines=_allocate_lines(lyric_lines, hints, target_duration),
        regeneration_policy=DEFAULT_REGENERATION_POLICY,
        mix=DEFAULT_MIX,
    )
    plan.validate()
    return plan
