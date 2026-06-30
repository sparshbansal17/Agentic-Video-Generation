from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal installs without PyYAML
    yaml = None


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


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "audio" / "default.yaml"


def _fallback_config() -> dict[str, Any]:
    return {
        "voice_backend": "f5_tts",
        "music_backend": "musicgen",
        "aligner_backend": "whisperx",
        "media_audio_mode": "hybrid_voice_bed",
        "mix": {
            "sample_rate": 48000,
            "target_lufs": -15.0,
            "true_peak_db": -1.5,
            "duration_tolerance_ms": 250,
            "line_boundary_tolerance_ms": 350,
            "max_time_stretch_percent": 8,
        },
        "backends": {
            "f5_tts": {
                "kind": "voice",
                "command": "f5-tts_infer-cli --ref_audio '${ref_audio}' --ref_text '${ref_text}' --gen_text '${text}' --output_file '${output_file}'",
            },
            "cosyvoice": {
                "kind": "voice",
                "command": "cosyvoice-infer --text '${text}' --prompt-audio '${ref_audio}' --prompt-text '${ref_text}' --style '${voice_style}' --output '${output_file}'",
            },
            "musicgen": {
                "kind": "music",
                "command": "musicgen-generate --prompt '${music_prompt}' --duration ${duration} --output '${output_file}'",
            },
            "stable_audio": {
                "kind": "music",
                "command": "stable-audio-generate --prompt '${music_prompt}' --duration ${duration} --output '${output_file}'",
            },
            "ace_step_full_song": {
                "kind": "song",
                "command": "acestep-generate --lyrics-file '${lyrics_file}' --prompt-file '${prompt_file}' --duration ${duration} --output '${output_file}'",
            },
            "yue_full_song": {
                "kind": "song",
                "command": "yue-generate --genre-txt '${genre_file}' --lyrics-txt '${lyrics_file}' --output-dir '${output_dir}'",
            },
            "whisperx": {
                "kind": "aligner",
                "command": "whisperx '${audio_file}' --model small --language en --output_dir '${output_dir}'",
            },
        },
    }


def load_backend_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.exists() and yaml is not None:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict) and isinstance(loaded.get("backends"), dict):
            return loaded
    return _fallback_config()


def default_backends(config_path: str | Path | None = None) -> dict[str, BackendCommand]:
    config = load_backend_config(config_path)
    configured = config.get("backends", {})
    descriptions = {
        "f5_tts": "Exact spoken nursery-rhyme line generation with a stable reference voice.",
        "cosyvoice": "Instruction-controlled TTS alternative for style, speed, and emotion control.",
        "musicgen": "Continuous instrumental bed generation.",
        "stable_audio": "Continuous instrumental bed alternative.",
        "ace_step_full_song": "Whole-song generation only; do not use for independent scene segments.",
        "yue_full_song": "Optional high-resource full-song lyrics-to-song backend.",
        "whisperx": "Word-level alignment and lyric timing evaluation.",
    }
    return {
        name: BackendCommand(
            name=name,
            kind=str(item.get("kind", "")),
            template=str(item.get("command", "")),
            description=descriptions.get(name, ""),
        )
        for name, item in configured.items()
        if isinstance(item, dict)
    }

def write_backend_manifest(path: str | Path, values: Mapping[str, object]) -> Path:
    manifest = {name: command.to_dict() for name, command in default_backends().items()}
    payload = {
        "manifest_type": "planned_backend_templates",
        "execution_status": "not_executed",
        "execution_note": (
            "These are dry-run command templates for planning and audit. Real media rendering is "
            "performed by story_audio.py through the postprocess-audio workflow."
        ),
        "config_path": str(DEFAULT_CONFIG_PATH),
        "values": dict(values),
        "backends": manifest,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
