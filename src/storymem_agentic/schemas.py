from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AudioMode = Literal["voice_bed", "full_song"]
RunMode = Literal["dry_run", "generate", "iterate"]


def quantize_seconds(value: float, fps: int = 24) -> float:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return round(round(float(value) * fps) / fps, 9)


@dataclass(slots=True)
class NurseryRhymeInput:
    rhyme_text: str = ""
    topic_or_name: str = ""
    lyrics: str | None = None
    rhyme_file: str | None = None
    target_audience: str = "toddlers"
    target_duration_seconds: float | None = None
    visual_style: str = "bright rounded toddler-safe bedtime storybook animation"
    audio_style: str = "warm clear nursery singalong, soft music box and celesta"
    max_iterations: int = 1
    seed: int = 0
    output_root: str = "results/agentic_run"
    clip_count: int | None = None
    character_db_path: str | None = None
    character_bank_path: str | None = None

    def validate(self) -> None:
        if not (self.rhyme_text.strip() or (self.lyrics or "").strip() or self.topic_or_name.strip()):
            raise ValueError("topic_or_name, lyrics, or rhyme_text is required")
        if self.target_duration_seconds is not None and self.target_duration_seconds <= 0:
            raise ValueError("target_duration_seconds must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.clip_count is not None and self.clip_count <= 0:
            raise ValueError("clip_count must be positive")

    @property
    def source_lyrics(self) -> str:
        return (self.lyrics if self.lyrics is not None else self.rhyme_text).strip()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


LullabyPromptInput = NurseryRhymeInput


@dataclass(slots=True)
class CharacterProfile:
    label: str
    description: str
    role: str | None = None
    visual_anchors: list[str] = field(default_factory=list)
    allowed_variants: list[str] = field(default_factory=list)
    continuity_constraints: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    reference_image_paths: list[str] = field(default_factory=list)
    voice_notes: str | None = None


@dataclass(slots=True)
class LyricSegment:
    index: int
    text: str
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass(slots=True)
class SceneBeat:
    scene_num: int
    lyric_segment_index: int
    start_seconds: float
    end_seconds: float
    description: str
    video_prompt: str
    first_frame_prompt: str
    subtitle_text: str
    cut: bool = False
    audio_description: str = ""
    expected_mood: str = "calm bedtime wonder"
    boundary_behavior: str = "hold"
    regeneration_dependencies: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ProductionPlan:
    version: str
    rhyme: NurseryRhymeInput
    clip_count: int
    target_fps: int
    lyric_segments: list[LyricSegment]
    character_bank: list[CharacterProfile]
    scenes: list[SceneBeat]
    audio_mode: AudioMode
    music_prompt: str
    evaluation_rubric: dict[str, Any]

    def validate(self) -> None:
        self.rhyme.validate()
        if self.clip_count <= 0:
            raise ValueError("clip_count must be positive")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if len(self.scenes) != self.clip_count:
            raise ValueError("scene count must match clip_count")
        if not self.lyric_segments:
            raise ValueError("production plan requires lyric segments")
        previous_end = -1.0
        segment_ids = {segment.index for segment in self.lyric_segments}
        for segment in self.lyric_segments:
            if not segment.text.strip():
                raise ValueError(f"lyric segment {segment.index} has empty text")
            if segment.start_seconds < previous_end - 0.001:
                raise ValueError("lyric segment timings must be monotonic")
            if segment.end_seconds <= segment.start_seconds:
                raise ValueError(f"lyric segment {segment.index} has non-positive duration")
            previous_end = segment.end_seconds
        seen = set()
        for scene in self.scenes:
            if scene.scene_num in seen:
                raise ValueError("scene_num values must be unique")
            seen.add(scene.scene_num)
            if scene.lyric_segment_index not in segment_ids:
                raise ValueError(f"scene {scene.scene_num} references missing lyric segment")
            if scene.end_seconds <= scene.start_seconds:
                raise ValueError(f"scene {scene.scene_num} has non-positive duration")
            if not scene.video_prompt.strip():
                raise ValueError(f"scene {scene.scene_num} has empty video_prompt")
            matching_segment = next(segment for segment in self.lyric_segments if segment.index == scene.lyric_segment_index)
            if not scene.subtitle_text.strip():
                raise ValueError(f"scene {scene.scene_num} has empty subtitle_text")
            if scene.start_seconds < matching_segment.start_seconds - 0.001 or scene.end_seconds > matching_segment.end_seconds + 0.001:
                raise ValueError(f"scene {scene.scene_num} falls outside lyric timing window")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProductionPlan":
        plan = cls(
            version=data["version"],
            rhyme=NurseryRhymeInput(**data["rhyme"]),
            clip_count=int(data["clip_count"]),
            target_fps=int(data.get("target_fps", 24)),
            lyric_segments=[LyricSegment(**item) for item in data.get("lyric_segments", [])],
            character_bank=[CharacterProfile(**item) for item in data.get("character_bank", [])],
            scenes=[SceneBeat(**item) for item in data.get("scenes", [])],
            audio_mode=data.get("audio_mode", "full_song"),
            music_prompt=data.get("music_prompt", ""),
            evaluation_rubric=dict(data.get("evaluation_rubric", {})),
        )
        plan.validate()
        return plan


@dataclass(slots=True)
class SceneEvaluation:
    scene_num: int
    passed: bool
    scores: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReviewerReport:
    reviewer: str
    passed: bool
    scores: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationReport:
    version: str
    passed: bool
    artifact_checks: dict[str, bool]
    scene_reports: list[SceneEvaluation]
    audio_scores: dict[str, float | None] = field(default_factory=dict)
    reviewer_reports: list[ReviewerReport] = field(default_factory=list)
    whisperx_alignment: dict[str, Any] | None = None
    regeneration_targets: list[int] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationReport":
        return cls(
            version=data["version"],
            passed=bool(data["passed"]),
            artifact_checks=dict(data.get("artifact_checks", {})),
            scene_reports=[SceneEvaluation(**item) for item in data.get("scene_reports", [])],
            audio_scores=dict(data.get("audio_scores", {})),
            reviewer_reports=[ReviewerReport(**item) for item in data.get("reviewer_reports", [])],
            whisperx_alignment=data.get("whisperx_alignment"),
            regeneration_targets=list(data.get("regeneration_targets", [])),
            failure_reasons=list(data.get("failure_reasons", [])),
        )


@dataclass(slots=True)
class RevisionPlan:
    version: str
    status: str
    target_scenes: list[int] = field(default_factory=list)
    regenerate_audio: bool = False
    prompt_revisions: dict[str, str] = field(default_factory=dict)
    first_frame_prompt_revisions: dict[str, str] = field(default_factory=dict)
    audio_prompt_revision: str | None = None
    clip_duration_adjustments: dict[str, float] = field(default_factory=dict)
    lyric_timing_adjustments: dict[str, dict[str, float]] = field(default_factory=dict)
    subtitle_timing_adjustments: dict[str, dict[str, float]] = field(default_factory=dict)
    mix_adjustments: dict[str, Any] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)
    reviewer_evidence: dict[str, Any] = field(default_factory=dict)
    preserve_scenes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    clip_duration_seconds: float | None = None
    vocal_style: str = "gentle lullaby vocal"
    music_style: str = "soft music box and celesta"
    expected_scene_mood: str = "calm bedtime wonder"
    boundary_behavior: str = "hold"

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
