from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents import AgentBackend
from .audio_director import normalize_lyrics
from .schemas import (
    CharacterProfile,
    LyricSegment,
    NurseryRhymeInput,
    ProductionPlan,
    SceneBeat,
    quantize_seconds,
)


DEFAULT_RUBRIC = {
    "artifact_checks_required": True,
    "min_scene_score": 0.75,
    "require_audio_stream": True,
    "require_subtitles": True,
    "visual_checks": [
        "character_consistency",
        "lyric_scene_match",
        "child_safety",
        "no_generated_text",
        "motion_plausibility",
    ],
}

STORYMEM_CLIP_SECONDS = 5.0
SHOT_PATTERNS = [
    "wide establishing shot with a clearly changed location and full-body staging",
    "intimate close-up focused on the main character's face and one meaningful prop",
    "low-angle wonder shot with foreground shapes and a deep background",
    "overhead dreamlike view with characters moving through the scene",
    "side-tracking shot following a gentle action across the frame",
    "peaceful final wide shot with the characters small but still clear in the environment",
]
COMPOSITIONS = [
    "balanced composition",
    "center composition",
    "left-weighted composition",
    "symmetrical composition",
    "short-side composition",
    "right-weighted composition",
]
LENSES = [
    "wide-angle lens",
    "medium lens",
    "wide lens",
    "telephoto lens",
    "medium lens",
    "wide-angle lens",
]
LIGHTING_CONTROLS = [
    "moonlight and warm practical nightlight, soft lighting, low contrast, warm pastel colors",
    "warm practical lighting, soft side lighting, shallow depth of field, gentle warm colors",
    "cool moonlighting with warm rim light, soft lighting, low-angle perspective, desaturated blue-gold colors",
    "diffused moonlight, soft top lighting, calm cool colors with warm highlights",
    "warm bedside lamp light, edge lighting, low contrast, soft peach and blue colors",
    "dawn-like soft light, side lighting, low contrast, warm gentle colors",
]

PLANNER_DECISION_SCHEMA: dict[str, Any] = {
    "name": "lullaby_planner_decision",
    "type": "object",
    "required": ["lyrics", "clip_count", "target_duration_seconds", "characters", "scenes", "music_prompt"],
    "properties": {
        "lyrics": {"type": "array", "items": {"type": "string"}},
        "clip_count": {"type": "integer"},
        "target_duration_seconds": {"type": "number"},
        "characters": {"type": "array", "items": {"type": "object"}},
        "scenes": {"type": "array", "items": {"type": "object"}},
        "music_prompt": {"type": "string"},
    },
}


def _character_bank(visual_style: str, character_db_path: str | None = None) -> list[CharacterProfile]:
    if character_db_path:
        data = json.loads(Path(character_db_path).read_text(encoding="utf-8"))
        raw_profiles = data.get("characters", data if isinstance(data, list) else [])
        profiles = []
        for item in raw_profiles:
            label = item.get("id") or item.get("label") or item.get("name")
            description = item.get("visual_description") or item.get("description")
            if not label or not description:
                continue
            profiles.append(
                CharacterProfile(
                    label=str(label),
                    description=str(description),
                    continuity_constraints=list(item.get("continuity_constraints", [])),
                    negative_constraints=list(item.get("negative_constraints", [])),
                    reference_image_paths=list(item.get("reference_image_paths", [])),
                    voice_notes=item.get("voice_notes") or item.get("personality_notes"),
                )
            )
        if profiles:
            return profiles

    return [
        CharacterProfile(
            label="pajama_child",
            description=(
                "same toddler-safe pajama child in every scene: round friendly face, sleepy happy eyes, "
                "soft blue pajamas with tiny star pattern, warm expression, simple rounded storybook design"
            ),
            continuity_constraints=["same pajamas", "same round friendly face", "calm bedtime expression"],
            negative_constraints=["no scary expression", "no sharp features"],
        ),
        CharacterProfile(
            label="smiling_star",
            description=(
                "same small rounded golden star in every scene: friendly smiling face, soft glow, "
                "gentle twinkle, no sharp edges, bedtime-safe toy-like appearance"
            ),
            continuity_constraints=["same small rounded golden star", "same soft glow"],
            negative_constraints=["no sharp points", "no harsh flare"],
        ),
        CharacterProfile(
            label="playroom_setting",
            description=(
                f"consistent cozy toddler playroom bedroom: round window, warm peach lamp glow, mint toys, "
                f"soft blue night sky, uncluttered {visual_style}"
            ),
            continuity_constraints=["round window remains visible", "warm peach, mint, soft blue palette"],
            negative_constraints=["no clutter", "no in-frame text"],
        ),
    ]


def _fallback_topic_lyrics(topic_or_name: str) -> str:
    topic = " ".join(topic_or_name.split()) or "sleepy stars"
    return "\n".join(
        [
            f"Sleep softly now, {topic},",
            "Moonlight hums a gentle tune,",
            "Stars are rocking clouds to sleep,",
            "Dreams will glow until the moon.",
        ]
    )


def _segment_lines(lines: list[str], target_duration: float, fps: int) -> list[LyricSegment]:
    slot = target_duration / max(len(lines), 1)
    pad = min(quantize_seconds(slot * 0.08, fps), quantize_seconds(0.35, fps))
    segments = []
    for index, text in enumerate(lines, start=1):
        start = quantize_seconds((index - 1) * slot + pad, fps)
        end = quantize_seconds(index * slot - pad, fps)
        if end <= start:
            end = quantize_seconds(start + 0.5, fps)
        segments.append(LyricSegment(index=index, text=text, start_seconds=start, end_seconds=end))
    return segments


def _shot_pattern(index: int) -> str:
    return SHOT_PATTERNS[(index - 1) % len(SHOT_PATTERNS)]


def _composition(index: int) -> str:
    return COMPOSITIONS[(index - 1) % len(COMPOSITIONS)]


def _lens(index: int) -> str:
    return LENSES[(index - 1) % len(LENSES)]


def _lighting_control(index: int) -> str:
    return LIGHTING_CONTROLS[(index - 1) % len(LIGHTING_CONTROLS)]


def _storyboard_action(topic: str, lyric_line: str, index: int, clip_count: int) -> tuple[str, str]:
    clean_topic = " ".join(topic.split()) or "the lullaby"
    if index == 1:
        return (
            f"a new opening scene introduces {clean_topic} through the meaning of the first lyric line, "
            "with the main character entering a cozy bedtime world",
            "wide establishing shot, gentle camera drift, full environment visible",
        )
    if index == clip_count:
        return (
            "a separate closing scene resolves the final lyric line with a calm goodnight image, "
            "the main character settles peacefully as the world grows quiet",
            "peaceful final wide shot, slow pullback, soft fade-ready motion",
        )
    pattern = _shot_pattern(index)
    return (
        "a distinct storyboard scene visualizes the current lyric line with a new action and changed setting, "
        "keeping the same soft character design but not repeating the previous composition",
        f"{pattern}, gentle lullaby motion",
    )


def _storyboard_prompt(
    *,
    index: int,
    clip_count: int,
    description: str,
    camera: str,
    visual_style: str,
    no_text_constraint: str,
) -> str:
    start = (index - 1) * STORYMEM_CLIP_SECONDS
    end = index * STORYMEM_CLIP_SECONDS
    subject = (
        "same toddler-safe lullaby characters implied by this scene, rounded friendly faces, soft expressive eyes, "
        "gentle child-safe proportions, consistent appearance and palette across clips"
    )
    scene = (
        f"{description}; full-frame environment with clear foreground, midground, and background; "
        "changed staging from the previous clip"
    )
    motion = (
        f"{camera}; visible gentle subject motion, subtle environmental motion, smooth stable timing; "
        "no frozen pose, no static tableau"
    )
    aesthetic = (
        f"{_lighting_control(index)}, night time bedtime atmosphere, {_shot_pattern(index)}, "
        f"{_composition(index)}, {_lens(index)}"
    )
    storyboard = (
        f"Shot {index} [{start:.1f}-{end:.1f}s]: hard cut transition into this clip; "
        f"{description}; {camera}; calm child-friendly emotional performance."
    )
    return (
        "Wan scene-clip prompt using the Advanced Formula. "
        f"Overall description: calm toddler lullaby video, edited storyboard clip {index} of {clip_count}, "
        "consistent characters and style across clips, but this clip has its own composition and action. "
        f"Subject: {subject}. "
        f"Scene: {scene}. "
        f"Motion: {motion}. "
        f"Aesthetic Control: {aesthetic}. "
        f"Stylization: {visual_style}, bright rounded bedtime storybook animation, soft textures, clean silhouettes. "
        f"Storyboard script: {storyboard} "
        "Use the entire image for the scene; no picture-in-picture, no framed inset, no small box, "
        "no border, no poster, no screen-within-screen composition. "
        "No dialogue. No background music. No subtitles in the image. "
        f"{no_text_constraint} No generated text, no letters, no scary elements, clear toddler-safe composition."
    )


def build_production_plan(rhyme: NurseryRhymeInput, *, target_fps: int = 24) -> ProductionPlan:
    rhyme.validate()
    source_lyrics = rhyme.source_lyrics or _fallback_topic_lyrics(rhyme.topic_or_name)
    lines = normalize_lyrics(source_lyrics)
    clip_count = min(rhyme.clip_count or len(lines), len(lines))
    target_duration = rhyme.target_duration_seconds or clip_count * STORYMEM_CLIP_SECONDS
    segments = _segment_lines(lines[:clip_count], target_duration, target_fps)
    bank = _character_bank(rhyme.visual_style, rhyme.character_db_path)
    no_text_constraint = (
        "Do not show any written lyrics, subtitles, captions, letters, words, signs, labels, title cards, "
        "book pages, handwriting, or readable text inside the image; the lyric is for meaning only."
    )

    scenes = []
    for index, segment in enumerate(segments, start=1):
        action, camera = _storyboard_action(rhyme.topic_or_name, segment.text, index, clip_count)
        prompt = _storyboard_prompt(
            index=index,
            clip_count=clip_count,
            description=action,
            camera=camera,
            visual_style=rhyme.visual_style,
            no_text_constraint=no_text_constraint,
        )
        first_frame = (
            f"Full-frame opening shot for a new edited scene: {action}; {_shot_pattern(index)}; "
            "clean first frame, no written words, no letters, no captions, no inset frame, rounded bedtime storybook style."
        )
        scenes.append(
            SceneBeat(
                scene_num=index,
                lyric_segment_index=segment.index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                description=action,
                video_prompt=prompt,
                first_frame_prompt=first_frame,
                subtitle_text=segment.text,
                cut=True,
                audio_description=f"Sing exactly this lyric line in the continuous full-song track: {segment.text}",
                expected_mood="calm child-safe bedtime wonder",
                boundary_behavior="fade" if index == clip_count else "hold",
                regeneration_dependencies=[],
            )
        )

    music_prompt = (
        f"{rhyme.audio_style}, full-song nursery-rhyme performance, exact lyrics, steady timing, "
        "music box, celesta, glockenspiel star twinkles, soft strings, gentle fade-out"
    )
    plan = ProductionPlan(
        version="1.0",
        rhyme=rhyme,
        clip_count=clip_count,
        target_fps=target_fps,
        lyric_segments=segments,
        character_bank=bank,
        scenes=scenes,
        audio_mode="full_song",
        music_prompt=music_prompt,
        evaluation_rubric=DEFAULT_RUBRIC,
    )
    plan.validate()
    return plan


def _planner_prompt() -> str:
    return (
        "You are PromptPlannerAgent for a local-first StoryMem lullaby video pipeline. "
        "Given the user input, produce only strict JSON. If lyrics are supplied, preserve them exactly. "
        "If only a lullaby name/topic is supplied, generate or select the appropriate full lullaby lyrics; "
        "do not ask for missing clip count, duration, style, characters, timing, or scenes. "
        "Honor optional user overrides exactly when present: target_duration_seconds, clip_count, target_audience, "
        "visual_style, audio_style, lyrics, and character_db_path. "
        "StoryMem generates one short video clip per scene prompt; each generated clip is approximately five seconds. "
        "If the user does not explicitly provide target_duration_seconds, set total duration to clip_count * 5 seconds. "
        "If the user explicitly provides target_duration_seconds, choose enough clips to cover that duration at about five seconds per clip. "
        "Choose a calm child-safe bedtime visual plan, complete clip count, total duration, lyric-to-scene plan, "
        "characters, continuity constraints, negative constraints, scene descriptions, camera/motion notes, boundary behavior, "
        "and music prompt. Sung lullabies must be one continuous full-song track, never per-scene song fragments. "
        "Follow the Wan prompt recipe for each scene clip. Plan every clip with the Advanced Formula: "
        "Subject plus subject description, Scene plus foreground/background description, Motion plus motion description, "
        "Aesthetic Control including light source, lighting type, time of day, shot size, composition, lens, color tone, "
        "camera angle and camera movement, and Stylization. Plan this like a normal edited video storyboard, not like "
        "one continuous shot. Every clip should be a new shot with its own location or staging, action, shot size, "
        "camera angle, subject distance, foreground/background layout, and motion. Use hard edited cuts between "
        "lyric-scene clips unless the user explicitly asks for a single continuous one-shot. Do not repeat generic "
        "camera language such as the same medium shot for all clips. "
        "Put any character or setting continuity details needed for that shot directly into that scene description; "
        "do not rely on a repeated character-bank prefix being added later. "
        "Every visual scene must prohibit generated text, letters, scary imagery, clutter, unsafe content, dialogue, "
        "and background music, because audio and subtitles are handled separately. "
        "Return JSON matching the schema: lyrics as an array of lyric lines; clip_count; target_duration_seconds; "
        "characters with label, description, continuity_constraints, negative_constraints, optional reference_image_paths; "
        "scenes with scene_num, lyric_line, description, camera, expected_mood, boundary_behavior, and optional cut. "
        "Scene descriptions should describe visual meaning without copying literal lyric text into the visual prompt. "
        "Return music_prompt for the separate continuous full-song audio. "
        "Use enough clips to cover the generated or supplied lyrics unless the input explicitly gives clip_count."
    )


def _list_from_decision(value: Any) -> list[str]:
    if isinstance(value, str):
        return normalize_lyrics(value)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _profiles_from_decision(decision: dict[str, Any], visual_style: str, character_db_path: str | None) -> list[CharacterProfile]:
    if character_db_path:
        return _character_bank(visual_style, character_db_path)
    raw_characters = decision.get("characters") or decision.get("character_bank") or []
    profiles: list[CharacterProfile] = []
    if isinstance(raw_characters, list):
        for index, item in enumerate(raw_characters, start=1):
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("id") or item.get("name") or f"character_{index}"
            description = item.get("description") or item.get("visual_description") or ""
            if not str(description).strip():
                continue
            profiles.append(
                CharacterProfile(
                    label=str(label),
                    description=str(description),
                    continuity_constraints=[str(value) for value in item.get("continuity_constraints", [])],
                    negative_constraints=[str(value) for value in item.get("negative_constraints", [])],
                    reference_image_paths=[str(value) for value in item.get("reference_image_paths", [])],
                    voice_notes=item.get("voice_notes") or item.get("personality_notes"),
                )
            )
    return profiles or _character_bank(visual_style, None)


def production_plan_from_planner_decision(
    rhyme: NurseryRhymeInput,
    decision: dict[str, Any],
    *,
    target_fps: int = 24,
) -> ProductionPlan:
    source_lines = normalize_lyrics(rhyme.source_lyrics) if rhyme.source_lyrics else []
    lines = source_lines or _list_from_decision(decision.get("lyrics"))
    if not lines:
        raise ValueError("planner decision did not provide lyrics")
    requested_clip_count = rhyme.clip_count or len(lines)
    clip_count = min(max(1, requested_clip_count), len(lines))
    if rhyme.target_duration_seconds is not None:
        target_duration = float(rhyme.target_duration_seconds)
    else:
        target_duration = clip_count * STORYMEM_CLIP_SECONDS
    segments = _segment_lines(lines[:clip_count], target_duration, target_fps)
    bank = _profiles_from_decision(decision, rhyme.visual_style, rhyme.character_db_path)
    no_text_constraint = (
        "Do not show any written lyrics, subtitles, captions, letters, words, signs, labels, title cards, "
        "book pages, handwriting, or readable text inside the image; the lyric is for meaning only."
    )
    raw_scenes = decision.get("scenes") if isinstance(decision.get("scenes"), list) else []
    scenes = []
    for index, segment in enumerate(segments, start=1):
        raw = raw_scenes[index - 1] if index - 1 < len(raw_scenes) and isinstance(raw_scenes[index - 1], dict) else {}
        description = str(raw.get("description") or raw.get("action") or "gentle bedtime scene matching the lullaby line meaning")
        camera = str(raw.get("camera") or raw.get("motion") or "slow gentle bedtime camera movement")
        expected_mood = str(raw.get("expected_mood") or "calm child-safe bedtime wonder")
        boundary_behavior = str(raw.get("boundary_behavior") or ("fade" if index == clip_count else "hold"))
        prompt = _storyboard_prompt(
            index=index,
            clip_count=clip_count,
            description=description,
            camera=camera,
            visual_style=rhyme.visual_style,
            no_text_constraint=no_text_constraint,
        )
        first_frame = (
            f"Full-frame opening shot for a new edited scene: {description}; {_shot_pattern(index)}; "
            "clean first frame, no written words, no letters, no captions, no inset frame, rounded bedtime storybook style."
        )
        cut = bool(raw.get("cut", True))
        scenes.append(
            SceneBeat(
                scene_num=index,
                lyric_segment_index=segment.index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                description=description,
                video_prompt=prompt,
                first_frame_prompt=first_frame,
                subtitle_text=segment.text,
                cut=cut,
                audio_description=f"Sing exactly this lyric line in the continuous full-song track: {segment.text}",
                expected_mood=expected_mood,
                boundary_behavior=boundary_behavior,
                regeneration_dependencies=[] if cut else list(range(1, index)),
            )
        )
    music_prompt = str(decision.get("music_prompt") or "").strip() or (
        f"{rhyme.audio_style}, full-song nursery-rhyme performance, exact lyrics, steady timing, "
        "music box, celesta, glockenspiel star twinkles, soft strings, gentle fade-out"
    )
    plan = ProductionPlan(
        version="1.0",
        rhyme=rhyme,
        clip_count=clip_count,
        target_fps=target_fps,
        lyric_segments=segments,
        character_bank=bank,
        scenes=scenes,
        audio_mode="full_song",
        music_prompt=music_prompt,
        evaluation_rubric=DEFAULT_RUBRIC,
    )
    plan.validate()
    return plan


class PromptPlannerAgent:
    def __init__(self, backend: AgentBackend | None = None, *, target_fps: int = 24) -> None:
        self.backend = backend
        self.target_fps = target_fps
        self.last_prompt: str | None = None
        self.last_schema: dict[str, Any] | None = None
        self.last_context: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.used_fallback: bool = False

    def plan(self, rhyme: NurseryRhymeInput) -> ProductionPlan:
        self.last_error = None
        self.last_response = None
        self.used_fallback = False
        if not self.backend:
            self.used_fallback = True
            return build_production_plan(rhyme, target_fps=self.target_fps)
        self.last_prompt = _planner_prompt()
        self.last_schema = PLANNER_DECISION_SCHEMA
        self.last_context = {"response_key": "planner", "input": rhyme.to_dict()}
        try:
            response = self.backend.generate_json(
                self.last_prompt,
                self.last_schema,
                self.last_context,
            )
        except Exception as exc:
            self.last_error = str(exc)
            self.used_fallback = True
            return build_production_plan(rhyme, target_fps=self.target_fps)
        self.last_response = response
        candidate = response.get("planner_decision", response.get("production_plan", response))
        try:
            if "lyric_segments" in candidate and "rhyme" in candidate:
                return ProductionPlan.from_dict(candidate)
            return production_plan_from_planner_decision(rhyme, candidate, target_fps=self.target_fps)
        except Exception as exc:
            self.last_error = str(exc)
            self.used_fallback = True
            return build_production_plan(rhyme, target_fps=self.target_fps)
