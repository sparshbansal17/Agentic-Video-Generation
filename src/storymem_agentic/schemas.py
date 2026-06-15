from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AudioMode = Literal["voice_bed", "full_song"]


@dataclass(slots=True)
class SceneHint:
    scene_num: int
    summary: str
    duration_seconds: float | None = None
    subtitle_text: str | None = None

    @classmethod
    def from_story_scene(cls, scene: dict[str, Any]) -> "SceneHint":
        prompts = scene.get("video_prompts") or []
        summary = str(prompts[0] if prompts else scene.get("story_overview", "")).strip()
        return cls(
            scene_num=int(scene.get("scene_num", 0)),
            summary=summary,
            duration_seconds=scene.get("duration_seconds"),
            subtitle_text=scene.get("subtitle_text") or scene.get("lyric_line"),
        )


@dataclass(slots=True)
class AudioLinePlan:
    index: int
    text: str
    scene_num: int
    start_seconds: float
    end_seconds: float
    voice: str = "narrator"
    subtitle: str | None = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass(slots=True)
class AudioPlan:
    version: str
    mode: AudioMode
    target_duration_seconds: float
    lyrics: list[str]
    story_summary: str
    music_prompt: str
    voice_backend: str
    music_backend: str
    aligner_backend: str
    lines: list[AudioLinePlan]
    regeneration_policy: dict[str, Any] = field(default_factory=dict)
    mix: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in {"voice_bed", "full_song"}:
            raise ValueError(f"Unsupported audio mode: {self.mode}")
        if self.target_duration_seconds <= 0:
            raise ValueError("target_duration_seconds must be positive")
        if not self.lyrics:
            raise ValueError("audio plan requires at least one lyric line")
        if not self.lines:
            raise ValueError("audio plan requires at least one timed line")
        previous_end = -1.0
        for line in self.lines:
            if not line.text.strip():
                raise ValueError(f"line {line.index} has empty text")
            if line.start_seconds < previous_end - 0.001:
                raise ValueError("line timings must be monotonic")
            if line.end_seconds <= line.start_seconds:
                raise ValueError(f"line {line.index} has non-positive duration")
            previous_end = line.end_seconds

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioPlan":
        lines = [AudioLinePlan(**line) for line in data.get("lines", [])]
        plan = cls(
            version=data["version"],
            mode=data["mode"],
            target_duration_seconds=float(data["target_duration_seconds"]),
            lyrics=list(data["lyrics"]),
            story_summary=data.get("story_summary", ""),
            music_prompt=data["music_prompt"],
            voice_backend=data["voice_backend"],
            music_backend=data["music_backend"],
            aligner_backend=data["aligner_backend"],
            lines=lines,
            regeneration_policy=dict(data.get("regeneration_policy", {})),
            mix=dict(data.get("mix", {})),
        )
        plan.validate()
        return plan
