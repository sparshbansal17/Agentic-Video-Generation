from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Protocol

from .schemas import AudioRepairAction, SongSpec


class SongBackend(Protocol):
    name: str
    model_version: str

    def generate(self, spec: SongSpec, output: Path, *, seed: int) -> Path: ...

    def repaint(
        self,
        spec: SongSpec,
        source: Path,
        output: Path,
        action: AudioRepairAction,
        *,
        seed: int,
    ) -> Path: ...


@dataclass(slots=True)
class CommandSongBackend:
    """Adapter for ACE-Step 1.5, SongGeneration 2, or another local CLI."""

    name: str
    model_version: str
    generate_template: str
    repaint_template: str | None = None

    @staticmethod
    def _run(template: str, values: dict[str, str], output: Path) -> Path:
        rendered = Template(template).safe_substitute(
            {key: shlex.quote(value) for key, value in values.items()}
        )
        subprocess.run(shlex.split(rendered), check=True)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"song backend completed without a usable output: {output}")
        return output

    @staticmethod
    def _write_inputs(spec: SongSpec, output: Path) -> tuple[Path, Path]:
        output.parent.mkdir(parents=True, exist_ok=True)
        stem = output.with_suffix("")
        spec_path = stem.with_name(stem.name + "_song_spec.json")
        lyrics_path = stem.with_name(stem.name + "_lyrics.txt")
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n", encoding="utf-8")
        lyrics_path.write_text("\n".join(spec.lyrics) + "\n", encoding="utf-8")
        return spec_path, lyrics_path

    def generate(self, spec: SongSpec, output: Path, *, seed: int) -> Path:
        spec_path, lyrics_path = self._write_inputs(spec, output)
        return self._run(self.generate_template, {
            "song_spec_file": str(spec_path),
            "lyrics_file": str(lyrics_path),
            "output_file": str(output),
            "duration": f"{spec.duration_seconds:.3f}",
            "caption": spec.caption,
            "bpm": str(spec.bpm),
            "key_scale": spec.key_scale,
            "time_signature": spec.time_signature,
            "seed": str(seed),
        }, output)

    def repaint(
        self,
        spec: SongSpec,
        source: Path,
        output: Path,
        action: AudioRepairAction,
        *,
        seed: int,
    ) -> Path:
        if action.kind != "repaint_region" or not self.repaint_template:
            raise ValueError(f"{self.name} does not support requested repaint action")
        spec_path, lyrics_path = self._write_inputs(spec, output)
        return self._run(self.repaint_template, {
            "song_spec_file": str(spec_path),
            "lyrics_file": str(lyrics_path),
            "source_audio": str(source),
            "output_file": str(output),
            "repaint_start": f"{float(action.start_seconds or 0.0):.3f}",
            "repaint_end": f"{float(action.end_seconds or spec.duration_seconds):.3f}",
            "seed": str(seed),
        }, output)
