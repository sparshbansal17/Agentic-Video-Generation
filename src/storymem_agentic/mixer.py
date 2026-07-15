from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schemas import AudioPlan


def build_mix_manifest(
    plan: AudioPlan,
    *,
    voice_stems: list[str] | None = None,
    music_bed: str | None = None,
    output_file: str | None = None,
) -> dict:
    if plan.mode == "full_song":
        default_voice_stems = ["song.wav"]
        default_music_bed = "backing.wav"
        default_output = "mixed_song.wav"
    else:
        default_voice_stems = [f"voice/line_{line.index:02d}.wav" for line in plan.lines]
        default_music_bed = "music/bed.wav"
        default_output = "final_audio.wav"
    return {
        "version": "1.0",
        "mode": plan.mode,
        "target_duration_seconds": plan.target_duration_seconds,
        "voice_stems": default_voice_stems if voice_stems is None else voice_stems,
        "music_bed": music_bed or default_music_bed,
        "output_file": output_file or default_output,
        "ducking": {
            "enabled": plan.mode in {"voice_bed", "full_song"},
            "ratio": plan.mix.get("music_ducking_ratio", 6.0),
        },
        "loudness": {
            "target_lufs": plan.mix.get("target_lufs", -15.0),
            "true_peak_db": plan.mix.get("true_peak_db", -1.5),
        },
        "line_timing": [asdict(line) for line in plan.lines],
    }


def write_mix_manifest(path: str | Path, manifest: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target
