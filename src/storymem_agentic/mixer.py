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
    output_file: str = "final_audio.wav",
) -> dict:
    return {
        "version": "1.0",
        "mode": plan.mode,
        "target_duration_seconds": plan.target_duration_seconds,
        "voice_stems": voice_stems or [f"voice/line_{line.index:02d}.wav" for line in plan.lines],
        "music_bed": music_bed or "music/bed.wav",
        "output_file": output_file,
        "ducking": {
            "enabled": plan.mode == "voice_bed",
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
