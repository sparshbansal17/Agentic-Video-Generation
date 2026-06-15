from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Mapping


@dataclass(slots=True)
class BackendCommand:
    name: str
    kind: str
    template: str
    description: str = ""

    def render(self, values: Mapping[str, object]) -> list[str]:
        rendered = Template(self.template).safe_substitute({k: str(v) for k, v in values.items()})
        return shlex.split(rendered)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def default_backends() -> dict[str, BackendCommand]:
    return {
        "f5_tts": BackendCommand(
            name="f5_tts",
            kind="voice",
            template="f5-tts_infer-cli --ref_audio '${ref_audio}' --ref_text '${ref_text}' --gen_text '${text}' --output_file '${output_file}'",
            description="Exact spoken nursery-rhyme line generation with a stable reference voice.",
        ),
        "cosyvoice": BackendCommand(
            name="cosyvoice",
            kind="voice",
            template="python engines/cosyvoice_infer.py --text '${text}' --style '${voice_style}' --output '${output_file}'",
            description="Instruction-controlled TTS alternative for style, speed, and emotion control.",
        ),
        "musicgen": BackendCommand(
            name="musicgen",
            kind="music",
            template="python engines/musicgen_infer.py --prompt '${music_prompt}' --duration ${duration} --output '${output_file}'",
            description="Continuous instrumental bed generation.",
        ),
        "ace_step_full_song": BackendCommand(
            name="ace_step_full_song",
            kind="song",
            template="python engines/acestep_song.py --lyrics-file '${lyrics_file}' --prompt-file '${prompt_file}' --duration ${duration} --output '${output_file}'",
            description="Whole-song generation only; do not use for independent scene segments.",
        ),
        "yue_full_song": BackendCommand(
            name="yue_full_song",
            kind="song",
            template="python engines/yue_song.py --genre-txt '${genre_file}' --lyrics-txt '${lyrics_file}' --output-dir '${output_dir}'",
            description="Optional high-resource full-song lyrics-to-song backend.",
        ),
        "whisperx": BackendCommand(
            name="whisperx",
            kind="aligner",
            template="whisperx '${audio_file}' --model small --language en --output_dir '${output_dir}'",
            description="Word-level alignment and lyric timing evaluation.",
        ),
    }


def write_backend_manifest(path: str | Path, values: Mapping[str, object]) -> Path:
    manifest = {name: command.to_dict() for name, command in default_backends().items()}
    payload = {"values": dict(values), "backends": manifest}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
