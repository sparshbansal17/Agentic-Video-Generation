from __future__ import annotations

import json
import re
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
SCENE_CAMERA_PLANS = [
    "Camera makes a slow diagonal push from a soft foreground object toward the main subject, revealing the scene depth",
    "Camera glides sideways at a sleepy walking pace, letting foreground shapes pass gently across the frame",
    "Camera rises in a shallow crane move, opening from character-scale detail into the larger bedtime setting",
    "Camera eases into a slow half-orbit, keeping the main subject centered while the background changes clearly",
    "Camera pulls back from a close detail into a wide calm tableau, preserving smooth parallax and stable focus",
    "Camera tilts down from a soft overhead detail to the main action, ending on a peaceful balanced composition",
    "Camera tracks forward low and slow, following the subject's gentle motion through the environment",
    "Camera drifts backward like a floating lullaby note, giving the final image space to breathe",
]
SCENE_AESTHETIC_PLANS = [
    "soft pastel 3D animation, warm practical glow, rounded forms, calm lullaby mood",
    "pastel children's-book style, soft moonlight, uncluttered shapes, calm motion, whimsical bedtime magic",
    "dreamlike 3D animation, soft focus, gentle blue-violet and honey-gold palette, safe soothing atmosphere",
    "magical bedtime tone, soft cinematic lighting, low contrast, pastel colors, smooth parallax movement",
    "warm moonlight mixed with a small practical light, soft lullaby ending, slow fade-ready glow",
    "rounded storybook animation, soft practical light, uncluttered foreground, gentle bedtime palette",
    "plush toy-like materials, soft edge lighting, low contrast, clean child-safe silhouettes",
    "watercolor-soft 3D storybook style, delicate highlights, calm blue-gold color harmony",
]
SETTING_FRAMES = [
    "a cozy rounded bedroom playroom at night, with a warm nightlight, round window, and soft toys",
    "a small handmade storybook landscape, with rounded paths, toy-like props, and warm distant lights",
    "a floating plush-cloud sky above a tiny toy town, with foreground clouds and open blue space",
    "a wide magical miniature world, with tiny safe details glowing below the characters",
    "a quiet closing space that returns to the lullaby's core image, with the final subject clearly visible and the background simplified",
    "a gentle outdoor bedtime scene inspired by the topic, with slow-moving natural elements and uncluttered child-safe staging",
    "a cozy interior nook built around the rhyme's main object, with plush textures, soft curtains, and a readable foreground-to-background path",
    "a slow-moving dream corridor of soft shapes from the lyric, arranged for a clear edited video shot rather than a one-shot scene",
]
PLANNER_DECISION_SCHEMA: dict[str, Any] = {
    "name": "scene_planner_decision",
    "type": "object",
    "required": [
        "lyrics",
        "clip_count",
        "target_duration_seconds",
        "visual_bible",
        "selected_characters",
        "scenes",
        "music_prompt",
    ],
    "properties": {
        "lyrics": {"type": "array", "items": {"type": "string"}},
        "clip_count": {"type": "integer"},
        "target_duration_seconds": {"type": "number"},
        "visual_bible": {"type": "object"},
        "selected_characters": {"type": "array", "items": {"type": "object"}},
        "characters": {"type": "array", "items": {"type": "object"}},
        "scenes": {"type": "array", "items": {"type": "object"}},
        "music_prompt": {"type": "string"},
    },
}

PLAN_CRITIC_SCHEMA: dict[str, Any] = {
    "name": "scene_plan_review_report",
    "type": "object",
    "required": ["passed", "issues"],
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["code", "message", "scene_num"],
                "properties": {
                    "code": {"type": "string"},
                    "message": {"type": "string"},
                    "scene_num": {"type": ["integer", "null"]},
                    "field": {"type": ["string", "null"]},
                    "evidence": {
                        "type": "object",
                        "required": ["observed", "expected", "source"],
                        "properties": {
                            "observed": {"type": "string"},
                            "expected": {"type": "string"},
                            "source": {"type": "string"},
                        },
                    },
                    "suggested_change": {"type": "string"},
                    "replacement_value": {},
                },
            },
        },
        "scores": {"type": "object"},
        "revision_notes": {"type": "array", "items": {"type": "string"}},
    },
}


def _load_structured_file(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError("YAML character banks require PyYAML; use JSON or install PyYAML") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _profile_from_item(item: dict[str, Any], index: int) -> CharacterProfile | None:
    label = item.get("id") or item.get("label") or item.get("name")
    description = item.get("visual_description") or item.get("description")
    if not label or not description:
        return None
    return CharacterProfile(
        label=str(label),
        description=_clean_sentence(str(description))[:220],
        role=str(item.get("role")) if item.get("role") else None,
        visual_anchors=[str(value) for value in item.get("visual_anchors", item.get("anchors", []))],
        allowed_variants=[str(value) for value in item.get("allowed_variants", [])],
        continuity_constraints=[str(value) for value in item.get("continuity_constraints", [])],
        negative_constraints=[str(value) for value in item.get("negative_constraints", item.get("avoid", []))],
        reference_image_paths=[str(value) for value in item.get("reference_image_paths", [])],
        voice_notes=item.get("voice_notes") or item.get("personality_notes"),
    )


def _character_bank(visual_style: str, character_db_path: str | None = None) -> list[CharacterProfile]:
    if character_db_path:
        data = _load_structured_file(character_db_path)
        raw_profiles = data.get("characters", data if isinstance(data, list) else [])
        profiles = []
        for index, item in enumerate(raw_profiles, start=1):
            if not isinstance(item, dict):
                continue
            profile = _profile_from_item(item, index)
            if profile:
                profiles.append(profile)
        if profiles:
            return profiles
        raise ValueError(f"character bank contains no valid character profiles: {character_db_path}")

    return [
        CharacterProfile(
            label="bedtime_child",
            description=(
                "same toddler-safe pajama child in every scene: round friendly face, sleepy happy eyes, "
                "soft blue pajamas with tiny dot pattern, warm expression, simple rounded storybook design"
            ),
            continuity_constraints=["same pajamas", "same round friendly face", "calm bedtime expression"],
            negative_constraints=["no scary expression", "no sharp features"],
        ),
        CharacterProfile(
            label="soft_glow_companion",
            description=(
                "same small rounded glowing companion in every scene: friendly smiling face, soft glow, "
                "no sharp edges, bedtime-safe toy-like appearance"
            ),
            continuity_constraints=["same small rounded glowing companion", "same soft glow"],
            negative_constraints=["no harsh flare", "no sharp features"],
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
    topic = " ".join(topic_or_name.split()) or "sleepy dreams"
    return "\n".join(
        [
            f"Sleep softly now, {topic},",
            "Soft light hums a gentle tune,",
            "Dreams are rocking clouds to sleep,",
            "Gentle light will carry you.",
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


def _scene_camera_plan(index: int) -> str:
    return SCENE_CAMERA_PLANS[(index - 1) % len(SCENE_CAMERA_PLANS)]


def _scene_aesthetic_plan(index: int) -> str:
    return SCENE_AESTHETIC_PLANS[(index - 1) % len(SCENE_AESTHETIC_PLANS)]


def _text_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _character_source_path(rhyme: NurseryRhymeInput) -> str | None:
    return rhyme.character_bank_path or rhyme.character_db_path


def _planner_input_context(rhyme: NurseryRhymeInput) -> dict[str, Any]:
    payload = rhyme.to_dict()
    source = _character_source_path(rhyme)
    if source:
        payload["character_bank_entries"] = [
            {
                "label": profile.label,
                "role": profile.role,
                "description": profile.description,
                "visual_anchors": profile.visual_anchors,
                "allowed_variants": profile.allowed_variants,
                "continuity_constraints": profile.continuity_constraints,
                "negative_constraints": profile.negative_constraints,
                "reference_image_paths": profile.reference_image_paths,
            }
            for profile in _character_bank(rhyme.visual_style, source)
        ]
    return payload




def _scene_description_is_specific(text: str) -> bool:
    lowered = text.lower()
    word_count = len(text.split())
    camera_terms = len(
        re.findall(
            r"\b(camera|camera angle|camera movement|shot size|composition|lens|color tone|lighting)\b",
            lowered,
        )
    )
    required_craft = ["opening shot", "camera", "soft"]
    has_action = any(word in lowered for word in ["moves", "drifts", "floats", "rises", "glows", "smiles", "walks", "pulls", "dollies", "cranes", "orbits", "tilts"])
    has_setting = any(word in lowered for word in ["nursery", "window", "village", "cloud", "sky", "bedroom", "meadow", "forest", "rooftop", "world"])
    has_spatial_detail = any(
        term in lowered
        for term in [
            "foreground",
            "midground",
            "background",
            "layered",
            "prop",
            "window",
            "path",
            "room",
            "landscape",
            "setting",
        ]
    )
    too_generic = any(
        phrase in lowered
        for phrase in [
            "opening shot: scene ",
            "wide shot of the night sky",
            "child is looking up",
            "camera moves slightly",
            "visual style: bright",
            "inspired by the input prompt",
            "from the input prompt",
            "the current lyric meaning",
            "the prompt's main imagery",
            "the lullaby theme",
            "clear main subject area",
            "generic lullaby scene",
        ]
    )
    return (
        word_count >= 60
        and word_count <= 170
        and camera_terms <= 5
        and all(term in lowered for term in required_craft)
        and has_action
        and has_setting
        and has_spatial_detail
        and not too_generic
    )


def _scene_description_has_concrete_content(text: str) -> bool:
    lowered = text.lower()
    words = _text_words(text)
    generic_phrases = [
        "visualizes the lyric",
        "lyric meaning",
        "inspired by the topic",
        "main subject",
        "generic lullaby",
    ]
    production_words = {
        "camera", "shot", "scene", "visual", "style", "lighting", "color", "palette",
        "mood", "continuity", "text", "soft", "warm", "bright", "calm",
    }
    content_words = words - production_words
    has_action = bool(
        {
            "runs", "run", "rows", "row", "rocks", "rocking", "sleeps", "sleeping",
            "looks", "sits", "stands", "walks", "floats", "glides", "falls", "rises",
            "climbs", "plays", "drifts", "holds", "opens", "closes", "shines",
        }
        & words
    )
    return (
        len(text.split()) >= 12
        and len(content_words) >= 8
        and has_action
        and not any(phrase in lowered for phrase in generic_phrases)
    )


def _clean_sentence(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split()).strip(" ,;.")


def _strip_leading_scene_label(text: str) -> str:
    cleaned = " ".join(str(text or "").replace("\n", " ").split()).strip(" ,;")
    cleaned = re.sub(r"(?i)^opening shot:\s*scene\s+\d+\s*:\s*", "Opening shot: ", cleaned)
    cleaned = re.sub(r"(?i)^scene\s+\d+\s*:\s*", "", cleaned)
    return cleaned


def _sentence_case(text: str) -> str:
    text = _clean_sentence(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def _strip_scene_production_notes(description: str) -> str:
    scene_text = _strip_leading_scene_label(description)
    if scene_text.startswith("Opening shot:"):
        scene_text = scene_text[len("Opening shot:"):].strip()
    scene_text = scene_text.replace(" Camera ", ". Camera ")
    split_pattern = re.compile(
        r"(?i)(?:\.\s+)?(?:camera|medium shot|wide shot|close-up shot|closeup shot|shot size|composition|lens|"
        r"color tone|lighting|time of day|camera angle|camera movement|visual style|stylization)\b"
    )
    match = split_pattern.search(scene_text)
    if match:
        scene_text = scene_text[: match.start()].strip(" ,;.")
    if len(scene_text) > 280:
        scene_text = scene_text[:280].rsplit(" ", 1)[0].rstrip(" ,;.")
    return _sentence_case(scene_text) + "."


def _camera_is_noisy(camera: str) -> bool:
    lowered = camera.lower()
    if not lowered.strip():
        return True
    if re.match(r"^(medium|wide|close-up|closeup|overhead|low-angle|high-angle)\s+shot\b", lowered):
        return True
    if len(camera.split()) > 18 or len(camera) > 130:
        return True
    if lowered in {"medium shot", "wide shot", "close-up", "close up", "slow push", "gentle pan", "soft fade"}:
        return True
    if "smooth camera movement" in lowered or "camera moves from left to right" in lowered:
        return True
    repeated_labels = len(
        re.findall(
            r"\b(camera angle|camera movement|shot size|composition|lens|color tone|lighting|medium shot)\b",
            lowered,
        )
    )
    return repeated_labels > 2


def _normalize_camera(camera: str, index: int) -> str:
    cleaned = _clean_sentence(camera)
    if _camera_is_noisy(cleaned):
        return _scene_camera_plan(index)
    cleaned = re.sub(
        r"(?i)\b(camera angle|camera movement|shot size|composition|lens|color tone|lighting)\s*:\s*",
        "",
        cleaned,
    )
    if not cleaned.lower().startswith("camera"):
        cleaned = f"Camera {cleaned}"
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rsplit(" ", 1)[0].rstrip(" ,;.")
    return _sentence_case(cleaned)


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
    text_guard = "no generated text or readable words"
    scene_text = _strip_scene_production_notes(description)
    camera_text = _clean_sentence(camera) or "Camera unspecified by planner"
    if camera_text and not camera_text.lower().startswith("camera"):
        camera_text = f"Camera {camera_text}"
    style = visual_style
    if len(style) > 90:
        style = style[:90].rsplit(" ", 1)[0].rstrip(" ,;")
    # The planner's camera field is authoritative. Only the offline diagnostic
    # scaffold needs a synthetic staging cue to keep its placeholder clips distinct.
    staging = ""
    if "non-agentic dry-run scaffold only" in description.lower():
        staging = f"; {_shot_pattern(index)}"
    return (
        f"Scene {index} of {clip_count}, [{start:.1f}-{end:.1f}s], toddler-safe lullaby animation: "
        f"{scene_text} {camera_text}. "
        f"{style}{staging}; soft bedtime mood, rounded shapes, gentle motion. "
        "Use StoryMem keyframe memory for character/style consistency; this prompt controls the new scene layout. "
        f"No dialogue or background music, {text_guard}, no picture-in-picture, no inset frame, no scary elements."
    )


def build_production_plan(rhyme: NurseryRhymeInput, *, target_fps: int = 24) -> ProductionPlan:
    rhyme.validate()
    source_lyrics = rhyme.source_lyrics or _fallback_topic_lyrics(rhyme.topic_or_name)
    lines = normalize_lyrics(source_lyrics)
    clip_count = min(rhyme.clip_count or len(lines), len(lines))
    target_duration = rhyme.target_duration_seconds or clip_count * STORYMEM_CLIP_SECONDS
    segments = _segment_lines(lines[:clip_count], target_duration, target_fps)
    bank = _character_bank(rhyme.visual_style, _character_source_path(rhyme))
    no_text_constraint = (
        "Do not show any written lyrics, subtitles, captions, letters, words, signs, labels, title cards, "
        "book pages, handwriting, or readable text inside the image; the lyric is for meaning only."
    )

    scenes = []
    for index, segment in enumerate(segments, start=1):
        selected_characters = [
            {
                "label": profile.label,
                "selection_rationale": "non-agentic local dry-run scaffold",
                "description": profile.description,
            }
            for profile in bank[:1]
        ]
        setting = "a calm full-frame bedtime storybook scene with foreground, midground, and background clearly separated"
        subjects = selected_characters[0]["description"] if selected_characters else "a simple child-safe bedtime subject"
        action_text = "performs a gentle visual action reserved for an external planner to define semantically"
        camera = _scene_camera_plan(index)
        action = (
            f"Opening shot: {setting}. {subjects} {action_text}. {camera}. "
            f"{rhyme.visual_style}, soft lighting, rounded shapes, uncluttered staging. "
            "Non-agentic dry-run scaffold only; no generated text, no dialogue, no inset frame, no scary imagery."
        )
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
                scene_goal="non-agentic dry-run scaffold",
                lyric_interpretation="requires agentic planner review for semantic interpretation",
                setting=setting,
                subjects=subjects,
                action=action_text,
                camera=camera,
                style=rhyme.visual_style,
                safety_adaptation="no generated text, no dialogue, no inset frame, no scary imagery",
                selected_characters=selected_characters,
                review_status="approved",
            )
        )

    music_prompt = (
        f"{rhyme.audio_style}, full-song nursery-rhyme performance, exact lyrics, steady timing, "
        "music box, celesta, glockenspiel shimmer, soft strings, gentle fade-out"
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
        evaluation_rubric={**DEFAULT_RUBRIC, "planning_mode": "non_agentic_local_dry_run"},
    )
    plan.validate()
    return plan


def _planner_prompt() -> str:
    return (
        "You are ScenePlannerAgent for a local-first StoryMem lullaby video pipeline. "
        "Given the user input, produce only strict JSON. If lyrics are supplied, preserve them exactly. "
        "If only a lullaby name/topic is supplied, generate or select the appropriate full lullaby lyrics; "
        "do not ask for missing clip count, duration, style, characters, timing, or scenes. "
        "Keep the plan practical for generation: if clip_count is not supplied, use 4 to 8 clips and 4 to 8 "
        "lyric lines. Do not repeat verses just to fill space. If target_duration_seconds is supplied, choose "
        "roughly one clip per five seconds, capped at 12 clips unless the user explicitly asks for more. "
        "Never output the same stanza more than twice. "
        "Honor optional user overrides exactly when present: target_duration_seconds, clip_count, target_audience, "
        "visual_style, audio_style, lyrics, character_bank_path, and character_db_path. "
        "If a character bank is supplied, inspect all entries and choose relevant characters agentically. "
        "For every selected character include label, description, and selection_rationale. For every scene include "
        "selected_characters with label and selection_rationale. Do not rely on downstream keyword matching. "
        "Only invent a generic missing character when no bank entry fits a needed role, and make that character "
        "simple, named, and consistent. "
        "StoryMem generates one short video clip per scene prompt; each generated clip is approximately five seconds. "
        "If the user does not explicitly provide target_duration_seconds, set total duration to clip_count * 5 seconds. "
        "If the user explicitly provides target_duration_seconds, choose enough clips to cover that duration at about five seconds per clip. "
        "Direct the result as a polished preschool animated music video: joyful, colorful, emotionally readable, funny or "
        "wonder-filled where appropriate, and always child-safe. Use the energy principles of successful preschool "
        "sing-alongs—clear silhouettes, expressive faces, rhythmic gestures, playful cause-and-effect, appealing props, "
        "and satisfying visual payoffs—without copying any named show, franchise, character, or signature design. "
        "Choose a child-safe visual plan, complete clip count, total duration, visual_bible, lyric-to-scene plan, "
        "characters, continuity constraints, negative constraints, scene descriptions, camera/motion notes, boundary behavior, "
        "and music prompt. Sung lullabies must be one continuous full-song track, never per-scene song fragments. "
        "Follow StoryMem's native story-input rules. Each scene is one approximately five-second generated shot, so give "
        "it one primary action or interaction that can begin, read clearly, and settle within five seconds. Avoid complex "
        "choreography, multiple simultaneous events, extreme motion, text rendering, or audio-dependent visual events. "
        "Each shot prompt must be concise but sufficiently detailed, equivalent to 1-4 sentences, with explicit recurring "
        "character appearance, action, setting layout, mood, shot size, and at most one simple camera movement. "
        "Follow the Wan prompt recipe for each scene clip. Plan every clip with the Advanced Formula: "
        "Subject plus subject description, Scene plus foreground/background description, Motion plus motion description, "
        "Aesthetic Control including light source, lighting type, time of day, shot size, composition, lens, color tone, "
        "camera angle and camera movement, and Stylization. Plan this like a normal edited video storyboard, not like "
        "one continuous shot. Build a miniature visual arc across the clips: establish the world and goal, develop a "
        "playful action or anticipation, deliver a clear reaction/payoff, and end on a satisfying final image. Every clip "
        "should advance that arc with its own staging, action, shot size, "
        "camera angle, subject distance, foreground/background layout, and motion. Use hard edited cuts between "
        "lyric-scene clips unless the user explicitly asks for a single continuous one-shot. Do not repeat generic "
        "camera language such as the same medium shot for all clips. Choose coverage like a real storyboard: use a wide "
        "shot to establish spatial relationships, medium shots for readable interaction, close-ups for an expression or "
        "important prop, and tracking/reveal/overhead angles only when motivated by the action. Do not change locations "
        "randomly merely to appear different; preserve world geography while changing foreground, subject distance, "
        "screen direction, action, or point of view. "
        "Keep each scene description between 70 and 120 words. Keep the camera field to one concise camera-movement "
        "sentence under 18 words. Do not place shot size, lens, color tone, composition, or repeated camera labels "
        "inside the camera field; those belong in the scene description once. "
        "Choose settings from the rhyme's own world. Do not mix unrelated default bedtime locations into a topic "
        "where they do not belong; let the planner/reviewer decide the lyric world from the supplied text. "
        "Each scene description must be a concrete paragraph in this format: 'Opening shot: [specific setting with "
        "foreground/background]. [specific subject] [specific action]. Camera [specific movement]. [visual style, "
        "lighting, color palette, mood]. [continuity and child-safety constraints].' The description must be rich "
        "enough to stand alone as a video-generation scene, similar to a director's storyboard card. "
        "Put any character or setting continuity details needed for that shot directly into that scene description; "
        "do not rely on a repeated character-bank prefix being added later. "
        "Keep each shot focused on one to three visible characters. Give the lead a clear pose, gaze target, facial "
        "expression, and child-readable action. Prefer playful gestures such as waving, bouncing, clapping, pointing, "
        "peeking, gentle dancing, rowing, stacking, chasing bubbles, or reacting with delight when they fit the lyric; "
        "do not force generic bedtime stillness onto an energetic song. "
        "Set cut=true for the first shot and whenever location, time, composition, or action changes. Use cut=false only "
        "when the next shot deliberately continues the same action from the prior last frame with compatible subjects, "
        "screen direction, setting, and camera axis. "
        "Every visual scene must prohibit generated text, letters, scary imagery, clutter, unsafe content, dialogue, "
        "and background music, because audio and subtitles are handled separately. "
        "For unsafe lyric events involving unsupported descent, breakage, dropping, crashing, impact, or dangerous "
        "motion by a child, character, prop, vehicle, support, or object, do not visualize the hazard. Adapt the "
        "scene into visibly safe supported motion and remove hazardous wording from structured fields. "
        "Return JSON matching the schema: lyrics as an array of lyric lines; clip_count; target_duration_seconds; "
        "visual_bible; selected_characters with label, description, selection_rationale, continuity_constraints, "
        "negative_constraints, optional reference_image_paths; "
        "scenes with scene_num, lyric_line, scene_goal, lyric_interpretation, setting, subjects, action, camera, "
        "style, safety_adaptation, selected_characters, expected_mood, boundary_behavior, optional cut, and review_status='pending'. "
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


def _profiles_from_decision(decision: dict[str, Any], visual_style: str, character_path: str | None) -> list[CharacterProfile]:
    if character_path:
        return _character_bank(visual_style, character_path)
    raw_characters = decision.get("selected_characters") or decision.get("characters") or decision.get("character_bank") or []
    profiles: list[CharacterProfile] = []
    if isinstance(raw_characters, list):
        for index, item in enumerate(raw_characters, start=1):
            if not isinstance(item, dict):
                continue
            item = {**item, "label": item.get("label") or item.get("id") or item.get("name") or f"character_{index}"}
            profile = _profile_from_item(item, index)
            if profile:
                profiles.append(profile)
    if not profiles:
        raise ValueError("planner decision requires at least one character with a label and description")
    return profiles


def build_visual_bible(plan: ProductionPlan) -> dict[str, Any]:
    selected_labels = {
        _character_lookup_key(str(item.get("label", "")))
        for scene in plan.scenes
        for item in scene.selected_characters
        if isinstance(item, dict) and str(item.get("label", "")).strip()
    }
    return {
        "primary_world": str(plan.evaluation_rubric.get("visual_bible", {}).get("primary_world") or "agent_designed_world"),
        "visual_style": plan.rhyme.visual_style,
        "allowed_locations": list(plan.evaluation_rubric.get("visual_bible", {}).get("allowed_locations", [])),
        "recurring_characters": [
            {
                "label": profile.label,
                "role": profile.role,
                "description": profile.description,
                "visual_anchors": profile.visual_anchors,
                "negative_constraints": profile.negative_constraints,
            }
            for profile in plan.character_bank
            if _character_lookup_key(profile.label) in selected_labels
        ],
        "prohibited_imagery": [
            "generated text",
            "readable words",
            "picture-in-picture",
            "inset frame",
            "scary imagery",
            "unsafe falls or impacts",
            "dialogue in visual prompt",
        ],
    }


def _visible_action_replacement(action: str) -> str | None:
    enjoying = re.match(r"(?i)^(.*?)\s+is\s+enjoying\s+(.+)$", action.strip())
    if enjoying:
        return f"{enjoying.group(1)} smiles brightly during {enjoying.group(2)}".strip(" ,;.")
    singing_and = re.sub(r"(?i)\bsinging\s+and\s+", "", action).strip(" ,;.")
    if singing_and != action.strip(" ,;."):
        return singing_and
    singing_while = re.match(
        r"(?i)^(.*?\b(?:is|are))\s+.*?\bsinging\b.*?\bwhile\s+(.+)$",
        action.strip(),
    )
    if singing_while:
        return f"{singing_while.group(1)} {singing_while.group(2)}".strip(" ,;.")
    cleaned = re.sub(
        r"(?i)(?:,|\bwhile\b|\band\b|\bwith\b)\s*(?:a\s+)?(?:sing(?:s|ing)?|tune|music|think(?:s|ing)?|reflect(?:s|ing)?|enjoy(?:s|ing)?)\b[^,;.]*",
        "",
        action,
    ).strip(" ,;.")
    return cleaned if cleaned and cleaned != action.strip(" ,;.") else None


def validate_plan_semantics(plan: ProductionPlan, *, require_reviewer_approval: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    plan.validate()
    required_scene_fields = [
        "scene_goal",
        "lyric_interpretation",
        "setting",
        "subjects",
        "action",
        "camera",
        "style",
        "safety_adaptation",
    ]
    for scene in plan.scenes:
        lower_prompt = scene.video_prompt.lower()
        for field_name in required_scene_fields:
            if not str(getattr(scene, field_name, "")).strip():
                issues.append({"code": "missing_structured_scene_field", "field": field_name, "scene_num": scene.scene_num, "message": f"scene missing {field_name}"})
        if not scene.selected_characters:
            issues.append({"code": "missing_character_selection_metadata", "scene_num": scene.scene_num, "message": "scene must include selected character metadata"})
        elif any(not str(item.get("label", "")).strip() or not str(item.get("selection_rationale", item.get("rationale", ""))).strip() for item in scene.selected_characters if isinstance(item, dict)):
            issues.append({"code": "incomplete_character_selection_metadata", "scene_num": scene.scene_num, "message": "selected characters require label and rationale"})
        if len(scene.selected_characters) > 3:
            issues.append(
                {
                    "code": "overcrowded_five_second_shot",
                    "scene_num": scene.scene_num,
                    "field": "selected_characters",
                    "message": "a five-second preschool shot should focus on no more than three visible characters",
                    "evidence": {"selected_character_count": len(scene.selected_characters)},
                }
            )
        selected_labels = {
            _character_lookup_key(str(item.get("label", "")))
            for item in scene.selected_characters
            if isinstance(item, dict) and str(item.get("label", "")).strip()
        }
        visible_text = " ".join((scene.setting, scene.subjects, scene.action)).lower().replace("_", " ")
        missing_profiles = [
            profile
            for profile in plan.character_bank
            if _character_lookup_key(profile.label).replace("_", " ") in visible_text
            and _character_lookup_key(profile.label) not in selected_labels
        ]
        if missing_profiles and len(selected_labels) + len(missing_profiles) > 3:
            issues.append(
                {
                    "code": "overcrowded_five_second_shot",
                    "scene_num": scene.scene_num,
                    "field": "selected_characters",
                    "message": "the scene names more than three visible bank characters; simplify the cast and all references together",
                    "evidence": {
                        "observed": json.dumps(sorted(selected_labels | {_character_lookup_key(p.label) for p in missing_profiles})),
                        "expected": "one to three lyric-relevant visible characters",
                        "source": "setting, subjects, action, and selected_characters",
                    },
                }
            )
        elif missing_profiles:
            replacement = [dict(item) for item in scene.selected_characters if isinstance(item, dict)]
            replacement.extend(
                {
                    "label": profile.label,
                    "selection_rationale": "visible subject explicitly named in this scene",
                    "description": profile.description,
                }
                for profile in missing_profiles
            )
            issues.append(
                {
                    "code": "visible_character_not_selected",
                    "scene_num": scene.scene_num,
                    "field": "selected_characters",
                    "message": "every visible bank character must appear in scene selection metadata",
                    "evidence": {
                        "observed": json.dumps(sorted(selected_labels)),
                        "expected": json.dumps([profile.label for profile in missing_profiles]),
                        "source": "character_bank labels explicitly named in setting, subjects, or action",
                    },
                    "suggested_change": "add the missing visible characters without changing the scene semantics",
                    "replacement_value": replacement,
                }
            )
        if require_reviewer_approval and scene.review_status != "approved":
            issues.append({"code": "missing_reviewer_approval", "scene_num": scene.scene_num, "message": "scene requires reviewer approval before generation"})
        if len(scene.video_prompt) >= 900:
            issues.append({"code": "prompt_too_long", "scene_num": scene.scene_num, "message": "StoryMem prompt exceeds length budget"})
        if any(term not in lower_prompt for term in ["no generated text", "no picture-in-picture", "no inset frame", "no scary"]):
            issues.append({"code": "missing_visual_guard", "scene_num": scene.scene_num, "message": "prompt is missing required visual guardrails"})
        if "no dialogue" not in lower_prompt or "background music" not in lower_prompt:
            issues.append({"code": "missing_audio_guard", "scene_num": scene.scene_num, "message": "prompt must keep dialogue and music out of visual generation"})
        safety_hits = []
        for field_name in (
            "scene_goal", "lyric_interpretation", "setting", "subjects", "action", "camera", "safety_adaptation"
        ):
            safety_hits.extend(_unsafe_safety_hits(getattr(scene, field_name), field=field_name))
        if safety_hits:
            issues.append(
                {
                    "code": "unsafe_visual_action",
                    "scene_num": scene.scene_num,
                    "message": "scene contains an unsafe visual action that must be adapted before generation",
                    "evidence": safety_hits,
                }
            )
        action_lower = scene.action.lower().strip()
        if re.match(r"^camera\b", action_lower):
            issues.append(
                {
                    "code": "camera_direction_in_action",
                    "scene_num": scene.scene_num,
                    "field": "action",
                    "message": "action must describe a visible subject action, not camera direction",
                    "evidence": {"observed": scene.action, "expected": "visible subject action", "source": "action"},
                }
            )
        if re.search(r"\b(?:sing|sings|singing|sang|tune|music|think|thinks|thinking|reflect|reflects|reflecting|enjoy|enjoys|enjoying)\b", action_lower):
            replacement = _visible_action_replacement(scene.action)
            issues.append(
                {
                    "code": "audio_or_internal_action",
                    "scene_num": scene.scene_num,
                    "field": "action",
                    "message": "replace audio-only or internal-state action with a readable pose, gaze, expression, or gesture",
                    "evidence": {"observed": scene.action, "expected": "directly visible action", "source": "action"},
                    **(
                        {
                            "suggested_change": "remove the audio or internal clause while preserving the visible action",
                            "replacement_value": replacement,
                        }
                        if replacement
                        else {}
                    ),
                }
            )
    if plan.scenes and not plan.scenes[0].cut:
        issues.append(
            {
                "code": "first_storymem_shot_requires_cut",
                "scene_num": 1,
                "field": "cut",
                "message": "the first StoryMem shot must start a new scene with cut=true",
            }
        )
    camera_groups: dict[str, list[int]] = {}
    for scene in plan.scenes:
        key = re.sub(r"\b(camera|shot|the|a|an)\b", " ", _clean_sentence(scene.camera).lower())
        key = " ".join(key.split())
        camera_groups.setdefault(key, []).append(scene.scene_num)
    for camera, scene_nums in camera_groups.items():
        if camera and len(scene_nums) >= 3:
            for scene_num in scene_nums[2:]:
                issues.append(
                    {
                        "code": "repeated_camera_coverage",
                        "scene_num": scene_num,
                        "field": "camera",
                        "message": "the storyboard repeats the same camera coverage across three or more shots",
                        "evidence": {
                            "observed": camera,
                            "expected": "distinct motivated coverage for this shot",
                            "source": f"camera fields in scenes {scene_nums}",
                        },
                        "suggested_change": "use distinct coverage without changing scene semantics",
                        "replacement_value": _scene_camera_plan(scene_num),
                    }
                )
    for current, previous in zip(plan.scenes[1:], plan.scenes):
        agentic_pair = not any(
            "non-agentic" in str(value).lower()
            for value in (current.scene_goal, previous.scene_goal, current.description, previous.description)
        )
        if agentic_pair and _clean_sentence(current.action).lower() == _clean_sentence(previous.action).lower():
            issues.append(
                {
                    "code": "repeated_narrative_beat",
                    "scene_num": current.scene_num,
                    "field": "action",
                    "message": "consecutive lyric scenes need distinct visible actions or reactions that advance the story",
                    "evidence": {
                        "observed": current.action,
                        "expected": "a distinct visible development, reaction, or payoff",
                        "source": f"action in preceding scene {previous.scene_num}",
                    },
                }
            )
        unchanged = [
            field_name
            for field_name in ("setting", "action", "camera")
            if _clean_sentence(getattr(current, field_name, "")).lower()
            == _clean_sentence(getattr(previous, field_name, "")).lower()
        ]
        if len(unchanged) == 3:
            issues.append(
                {
                    "code": "repeated_scene_staging",
                    "scene_num": current.scene_num,
                    "message": "consecutive scenes must materially change setting, action, or camera composition",
                    "evidence": {"matches_scene_num": previous.scene_num, "unchanged_fields": unchanged},
                }
            )
    return {
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "visual_bible": build_visual_bible(plan),
    }


def _critic_prompt() -> str:
    return (
        "You are PlanCriticAgent, the semantic reviewer for a StoryMem video plan before costly GPU generation. "
        "Review the supplied plan independently of its topic; never substitute canned nursery-rhyme scenes. Judge it "
        "as a human story editor and director for an engaging preschool animated sing-along and according to StoryMem's "
        "one-five-second-shot-per-prompt constraint. Return "
        "strict JSON with passed, issues, scores, and revision_notes. Score 0.0 to 1.0 for lyric_alignment, "
        "scene_progression, visual_continuity, child_safety, prompt_generatability, and prompt_hygiene. Review both "
        "individual shots and the overall plan: speaker/subject meaning, setup-development-payoff, redundant beats, "
        "character motivation and plurality, world geography, emotional progression, visual-versus-audio actions, "
        "camera motivation, cut continuity, and whether a revised action still agrees with its scene_goal and "
        "lyric_interpretation. Use the lyric_line embedded in each scene as the source of truth. Audit every scene in "
        "order and compare lyric_line independently against scene_goal, lyric_interpretation, setting, subjects, and "
        "action; never assume the plan's interpretation is correct. Then audit the sequence as one causal visual story. "
        "Reject an action that depicts the opposite of a stated wish, merely paraphrases a lyric without a visible "
        "story beat, uses an abstract inner state such as reflecting or enjoying without a readable pose or expression, "
        "assigns singing to visual generation, introduces irrelevant characters or locations, lists a visible bank "
        "character absent from selected_characters, or leaves a safety-adapted action contradicting scene_goal or "
        "lyric_interpretation. Do not approve merely because fields are populated or child-safe. Reject when "
        "any scene misrepresents its lyric, fails to advance a clear setup-development-payoff arc, repeats prior staging "
        "without narrative purpose, lacks a readable expression/gaze/action, overloads the five-second beat, drifts outside the visual "
        "bible, changes a recurring character's anchors, depicts unsafe literal action, is vague or internally "
        "contradictory, overloads a five-second clip, conflicts on camera direction, or requests generated text, "
        "dialogue, inset frames, or background music inside a visual scene. Audio planning is reviewed separately; "
        "do not review or reinterpret music_prompt. For each semantic issue return a stable snake_case code, exact "
        "scene_num (or null for plan-wide issues), affected field when known, concise evidence, and a suggested_change "
        "that preserves the user's input. Also return replacement_value: the exact complete JSON value that should "
        "replace the cited editable field, not an instruction and not a whole regenerated plan. For plan-wide issues, "
        "target only visual_bible, selected_characters, or scenes. Evidence must be an object with observed (an exact plan fact), expected "
        "(the conflicting requirement), and source (the field, lyric, visual-bible rule, or comparison scene). Only "
        "review planner-owned fields present in review_plan; do not review derived prompts, timing, audio_description, "
        "first_frame_prompt, subtitle metadata, or guardrail wording. A prohibition such as 'no generated text' is a "
        "valid guardrail, not a request to generate text. Deterministic issues in context are already enforced and will be merged by "
        "the orchestrator; do not omit or contradict them, but do not duplicate them. Set passed=false whenever you "
        "return an issue. A safe adaptation must change the depicted event itself rather than merely negate hazardous "
        "words. Approval means the plan is specific, coherent, distinct, safe, and directly generatable."
    )


def _raw_scene_field(raw: dict[str, Any], *names: str) -> str:
    for name in names:
        value = raw.get(name)
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    label = item.get("description") or item.get("label") or item.get("name") or item.get("id")
                    if label:
                        parts.append(str(label))
                elif str(item).strip():
                    parts.append(str(item))
            value = ", ".join(parts)
        elif isinstance(value, dict):
            value = value.get("description") or value.get("label") or value.get("name") or value.get("id") or ""
        if value is not None and str(value).strip():
            return _clean_sentence(str(value))
    return ""


def _character_lookup_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")


def _selected_character_metadata(
    raw: dict[str, Any],
    character_bank: list[CharacterProfile],
) -> list[dict[str, Any]]:
    bank_by_label = {
        _character_lookup_key(profile.label): profile
        for profile in character_bank
    }
    raw_selection = raw.get("selected_characters")
    if raw_selection is None:
        raw_selection = raw.get("characters")
    if not isinstance(raw_selection, list):
        return []
    selected: list[dict[str, Any]] = []
    for item in raw_selection:
        if isinstance(item, str):
            selected.append({"label": item, "selection_rationale": "selected by planner"})
        elif isinstance(item, dict):
            label = item.get("label") or item.get("id") or item.get("name")
            rationale = item.get("selection_rationale") or item.get("rationale") or item.get("reason")
            if label:
                bank_profile = bank_by_label.get(_character_lookup_key(str(label)))
                description = item.get("description") or (bank_profile.description if bank_profile else None)
                selected.append(
                    {
                        "label": str(label),
                        "selection_rationale": str(rationale or "selected by planner"),
                        **({"description": str(description)} if description else {}),
                    }
                )
    return selected


def _compile_scene_description(
    raw: dict[str, Any],
    visual_style: str,
    selected_characters: list[dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    fields = {
        "scene_goal": _raw_scene_field(raw, "scene_goal", "goal"),
        "lyric_interpretation": _raw_scene_field(raw, "lyric_interpretation", "interpretation"),
        "setting": _raw_scene_field(raw, "setting"),
        "subjects": _raw_scene_field(raw, "subjects", "subject"),
        "action": _raw_scene_field(raw, "action"),
        "camera": _raw_scene_field(raw, "camera"),
        "style": _raw_scene_field(raw, "style", "visual_style") or visual_style,
        "safety_adaptation": _raw_scene_field(raw, "safety_adaptation", "safety"),
    }
    character_descriptions = [
        str(item.get("description", "")).strip()
        for item in selected_characters
        if isinstance(item, dict) and str(item.get("description", "")).strip()
    ]
    missing_descriptions = [
        description for description in character_descriptions
        if description.lower() not in fields["subjects"].lower()
    ]
    if missing_descriptions:
        references = "; ".join(missing_descriptions)
        fields["subjects"] = (
            f"{fields['subjects']}, featuring {references}"
            if fields["subjects"]
            else references
        )
    description = _strip_leading_scene_label(str(raw.get("description") or ""))
    if all(fields[name] for name in ["setting", "subjects", "action", "camera", "style", "safety_adaptation"]):
        description = (
            f"Opening shot: {fields['setting']}. {fields['subjects']} {fields['action']}. "
            f"{fields['camera']}. {fields['style']}. {fields['safety_adaptation']}."
        )
    elif description and not description.startswith("Opening shot:"):
        description = f"Opening shot: {description}"
    return description, fields


def _unsafe_safety_hits(text: str, *, field: str) -> list[dict[str, str]]:
    lowered = f" {str(text or '').lower()} "
    lowered = re.sub(r"\b(?:rain|raindrops|snow|snowflakes|leaves|petals|water)\s+(?:is\s+|are\s+)?falling\b", " ", lowered)
    safe_adaptation_terms = [
        "no falling",
        "never falling",
        "not falling",
        "not fall",
        "settles safely",
        "lands safely",
        "caught safely",
        "supported safely",
        "floating gently",
        "gently lowers",
    ]
    for term in safe_adaptation_terms:
        lowered = lowered.replace(term, " ")
    unsafe_patterns = [
        r"\bfall\b",
        r"\bfalling\b",
        r"\bfalls\b",
        r"\bfall down\b",
        r"\bfalling down\b",
        r"\bfall toward\b",
        r"\babout to fall\b",
        r"\bdrop toward\b",
        r"\bdropping\b",
        r"\bdescend(?:s|ed|ing)?\b",
        r"\bcrash(?:es|ing)?\b",
        r"\bimpact(?:s|ing)?\b",
        r"\b(?:break|breaks|breaking)\b.{0,40}\b(?:cradle|branch|bough|support|rope|rail)\b",
        r"\b(?:cradle|branch|bough|support|rope|rail)\b.{0,40}\b(?:break|breaks|breaking)\b",
    ]
    hits: list[dict[str, str]] = []
    for pattern in unsafe_patterns:
        match = re.search(pattern, lowered)
        if match:
            excerpt = " ".join(lowered[max(0, match.start() - 80): match.end() + 80].split())
            hits.append({"field": field, "pattern": pattern, "excerpt": excerpt})
    return hits


def _normalize_critic_issue(issue: Any, index: int) -> dict[str, Any]:
    if not isinstance(issue, dict):
        return {
            "code": "critic_invalid_issue",
            "scene_num": None,
            "message": f"critic issue {index} must be an object",
            "evidence": {"received_type": type(issue).__name__},
        }
    code = re.sub(r"[^a-z0-9]+", "_", str(issue.get("code") or "semantic_review_issue").lower()).strip("_")
    message = _clean_sentence(str(issue.get("message") or issue.get("reason") or "semantic review rejected the plan"))
    raw_scene_num = issue.get("scene_num")
    try:
        scene_num = int(raw_scene_num) if raw_scene_num is not None else None
    except (TypeError, ValueError):
        scene_num = None
    normalized = {
        "code": code or "semantic_review_issue",
        "scene_num": scene_num,
        "message": message,
    }
    for key in ("field", "evidence", "suggested_change", "replacement_value"):
        if issue.get(key) not in (None, "", [], {}):
            normalized[key] = issue[key]
    field = normalized.get("field")
    replacement = normalized.get("replacement_value")
    if isinstance(field, str) and isinstance(replacement, dict) and field in replacement:
        normalized["replacement_value"] = replacement[field]
    evidence = normalized.get("evidence")
    if isinstance(field, str) and isinstance(evidence, dict) and not evidence.get("observed") and evidence.get(field):
        normalized["evidence"] = {**evidence, "observed": evidence[field]}
    return normalized


def _normalize_critic_scores(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores: dict[str, float] = {}
    for key, raw_score in value.items():
        try:
            scores[str(key)] = max(0.0, min(1.0, float(raw_score)))
        except (TypeError, ValueError):
            continue
    return scores


def _critic_review_plan(plan: ProductionPlan) -> dict[str, Any]:
    scene_fields = (
        "scene_num",
        "scene_goal",
        "lyric_interpretation",
        "setting",
        "subjects",
        "action",
        "camera",
        "style",
        "safety_adaptation",
        "selected_characters",
        "expected_mood",
        "boundary_behavior",
        "cut",
    )
    return {
        "lyrics": [segment.text for segment in plan.lyric_segments],
        "visual_bible": build_visual_bible(plan),
        "available_character_options": [
            {
                "label": profile.label,
                "role": profile.role,
                "description": profile.description,
                "visual_anchors": profile.visual_anchors,
                "allowed_variants": profile.allowed_variants,
                "continuity_constraints": profile.continuity_constraints,
                "negative_constraints": profile.negative_constraints,
            }
            for profile in plan.character_bank
        ],
        "scenes": [
            {
                "lyric_line": plan.lyric_segments[scene.lyric_segment_index - 1].text,
                **{field: getattr(scene, field) for field in scene_fields},
            }
            for scene in plan.scenes
        ],
    }


def _critic_issue_is_actionable(issue: dict[str, Any], review_plan: dict[str, Any]) -> bool:
    editable_scene_fields = {
        "scene_goal", "lyric_interpretation", "setting", "subjects", "action", "camera", "style",
        "safety_adaptation", "selected_characters", "expected_mood", "boundary_behavior", "cut",
    }
    plan_fields = {"visual_bible", "selected_characters"}
    field = str(issue.get("field") or "").split("[", 1)[0]
    # Hygiene is already enforced deterministically. Small semantic critics tend
    # to misuse this label for harmless wording preferences, so keep it advisory.
    if issue.get("code") == "prompt_hygiene":
        return False
    scene_num = issue.get("scene_num")
    if scene_num is None:
        if field not in plan_fields:
            return False
    elif not isinstance(scene_num, int) or not 1 <= scene_num <= len(review_plan.get("scenes", [])) or field not in editable_scene_fields:
        return False
    evidence = issue.get("evidence")
    if not isinstance(evidence, dict):
        return False
    if not all(str(evidence.get(key) or "").strip() for key in ("observed", "expected", "source")):
        return False
    observed = str(evidence["observed"]).strip().lower()
    expected = str(evidence["expected"]).strip().lower()
    if observed == expected:
        return False
    if scene_num is not None:
        scene_value = review_plan["scenes"][scene_num - 1].get(field)
        if observed not in json.dumps(scene_value, sort_keys=True).lower():
            return False
    source = str(evidence["source"]).strip().lower()
    if source.startswith("visual_bible") and expected not in json.dumps(review_plan.get("visual_bible", {}), sort_keys=True).lower():
        return False
    suggested_change = issue.get("suggested_change")
    if not isinstance(suggested_change, str) or not suggested_change.strip():
        return False
    if "replacement_value" not in issue:
        return False
    replacement = issue["replacement_value"]
    if scene_num is not None and replacement == review_plan["scenes"][scene_num - 1].get(field):
        return False
    return True


class PlanCriticAgent:
    def __init__(self, backend: AgentBackend | None = None) -> None:
        self.backend = backend
        self.last_prompt: str | None = None
        self.last_schema: dict[str, Any] | None = None
        self.last_context: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None
        self.last_error: str | None = None

    def review(self, plan: ProductionPlan, deterministic_report: dict[str, Any]) -> dict[str, Any]:
        self.last_error = None
        self.last_response = None
        deterministic_issues = deterministic_report.get("issues", [])
        if not isinstance(deterministic_issues, list):
            deterministic_issues = []
        if not self.backend:
            return {
                "passed": not deterministic_issues,
                "issues": deterministic_issues,
                "scores": {},
                "revision_notes": ["Resolve every deterministic validation issue before approval."] if deterministic_issues else [],
            }
        self.last_prompt = _critic_prompt()
        self.last_schema = PLAN_CRITIC_SCHEMA
        review_plan = _critic_review_plan(plan)
        self.last_context = {
            "response_key": "plan_critic",
            "review_plan": review_plan,
            "deterministic_report": deterministic_report,
            "review_contract": {
                "independent_dimensions": [
                    "lyric_alignment",
                    "scene_progression",
                    "visual_continuity",
                    "child_safety",
                    "prompt_generatability",
                    "prompt_hygiene",
                ],
                "deterministic_issues_are_binding": True,
                "issue_fields": ["code", "scene_num", "message", "field", "evidence", "suggested_change"],
            },
        }
        try:
            response = self.backend.generate_json(self.last_prompt, self.last_schema, self.last_context)
        except Exception as exc:
            self.last_error = str(exc)
            return {
                "passed": False,
                "issues": [{"code": "critic_backend_error", "message": str(exc), "scene_num": None}],
                "scores": {},
                "revision_notes": [],
            }
        self.last_response = response
        raw_issues = response.get("issues", [])
        protocol_issues: list[dict[str, Any]] = []
        if not isinstance(raw_issues, list):
            protocol_issues.append(
                {"code": "critic_invalid_issues", "message": "critic issues must be a list", "scene_num": None}
            )
            raw_issues = []
        normalized_issues = [_normalize_critic_issue(issue, index) for index, issue in enumerate(raw_issues, start=1)]
        issues = [issue for issue in normalized_issues if _critic_issue_is_actionable(issue, review_plan)]
        warnings = [
            {**issue, "warning": "critic claim was not grounded in editable planner fields with comparative evidence"}
            for issue in normalized_issues
            if issue not in issues
        ]
        issues.extend(protocol_issues)
        if response.get("passed") is False and not raw_issues and not protocol_issues:
            issues.append(
                {
                    "code": "critic_rejected_without_issues",
                    "scene_num": None,
                    "message": "critic rejected the plan without actionable issue details",
                    "suggested_change": "review the plan again and identify exact fields and scenes requiring revision",
                }
            )
        known = {
            (str(issue.get("code")), issue.get("scene_num"))
            for issue in issues
            if isinstance(issue, dict)
        }
        for issue in deterministic_issues:
            if isinstance(issue, dict) and (str(issue.get("code")), issue.get("scene_num")) not in known:
                issues.append(issue)
        scene_revisions = [
            {
                "scene_num": issue["scene_num"],
                "field_to_change": issue["field"],
                "replacement_value": issue["replacement_value"],
            }
            for issue in issues
            if isinstance(issue, dict)
            and issue.get("scene_num") is not None
            and issue.get("field")
            and "replacement_value" in issue
        ]
        plan_updates = {
            str(issue["field"]): issue["replacement_value"]
            for issue in issues
            if isinstance(issue, dict)
            and issue.get("scene_num") is None
            and issue.get("field") in {"visual_bible", "selected_characters"}
            and "replacement_value" in issue
        }
        return {
            "passed": not issues,
            "issues": issues,
            "scores": _normalize_critic_scores(response.get("scores", {})),
            "revision_notes": [
                _clean_sentence(str(note)) for note in response.get("revision_notes", []) if str(note).strip()
            ] if isinstance(response.get("revision_notes", []), list) else [],
            "warnings": warnings,
            "targeted_revision": {"scene_revisions": scene_revisions, "plan_updates": plan_updates},
        }


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
    bank = _profiles_from_decision(decision, rhyme.visual_style, _character_source_path(rhyme))
    no_text_constraint = (
        "Do not show any written lyrics, subtitles, captions, letters, words, signs, labels, title cards, "
        "book pages, handwriting, or readable text inside the image; the lyric is for meaning only."
    )
    raw_scenes = decision.get("scenes") if isinstance(decision.get("scenes"), list) else []
    scenes = []
    for index, segment in enumerate(segments, start=1):
        raw = raw_scenes[index - 1] if index - 1 < len(raw_scenes) and isinstance(raw_scenes[index - 1], dict) else {}
        selected_characters = _selected_character_metadata(raw, bank)
        description, structured = _compile_scene_description(raw, rhyme.visual_style, selected_characters)
        camera = structured["camera"]
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
            f"Full-frame opening shot for a new edited scene: {description}; "
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
                scene_goal=structured["scene_goal"],
                lyric_interpretation=structured["lyric_interpretation"],
                setting=structured["setting"],
                subjects=structured["subjects"],
                action=structured["action"],
                camera=structured["camera"],
                style=structured["style"],
                safety_adaptation=structured["safety_adaptation"],
                selected_characters=selected_characters,
                review_status=str(raw.get("review_status") or "pending"),
            )
        )
    music_prompt = str(decision.get("music_prompt") or "").strip() or (
        f"{rhyme.audio_style}, full-song nursery-rhyme performance, exact lyrics, steady timing, "
        "music box, celesta, glockenspiel shimmer, soft strings, gentle fade-out"
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
        evaluation_rubric={**DEFAULT_RUBRIC, "visual_bible": decision.get("visual_bible", {}) if isinstance(decision.get("visual_bible"), dict) else {}},
    )
    plan.validate()
    return plan


def _apply_planner_revision(previous: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    revisions = response.get("scene_revisions")
    if not isinstance(revisions, list):
        return response.get("planner_decision", response.get("production_plan", response))
    revised = json.loads(json.dumps(previous))
    scenes = revised.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("scene revision requires a previous decision with scenes")
    by_number = {
        int(scene.get("scene_num", index)): scene
        for index, scene in enumerate(scenes, start=1)
        if isinstance(scene, dict)
    }
    editable = {
        "scene_goal", "lyric_interpretation", "setting", "subjects", "action", "camera", "style",
        "safety_adaptation", "selected_characters", "expected_mood", "boundary_behavior", "cut",
    }
    changed = False
    for revision in revisions:
        if not isinstance(revision, dict):
            continue
        try:
            scene_num = int(revision.get("scene_num"))
        except (TypeError, ValueError):
            continue
        target = by_number.get(scene_num)
        if target is None:
            continue
        field_to_change = revision.get("field_to_change")
        if field_to_change in editable and "replacement_value" in revision:
            replacement = revision["replacement_value"]
            if replacement != target.get(field_to_change):
                target[field_to_change] = replacement
                changed = True
        for field in editable:
            if field in revision and revision[field] != target.get(field):
                target[field] = revision[field]
                changed = True
    plan_updates = response.get("plan_updates", {})
    if isinstance(plan_updates, dict):
        for field in ("visual_bible", "selected_characters", "music_prompt"):
            if field in plan_updates and plan_updates[field] != revised.get(field):
                revised[field] = plan_updates[field]
                changed = True
    if not changed:
        return revised
    return revised


def _constrain_revision_to_issues(response: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    revisions = response.get("scene_revisions")
    if not isinstance(revisions, list):
        return response
    exact = {
        (issue.get("scene_num"), str(issue.get("field")))
        for issue in issues
        if isinstance(issue, dict) and issue.get("scene_num") is not None and issue.get("field")
    }
    broad_scenes = {
        issue.get("scene_num")
        for issue in issues
        if isinstance(issue, dict)
        and issue.get("scene_num") is not None
        and issue.get("code") in {"unsafe_visual_action", "repeated_scene_staging", "overcrowded_five_second_shot"}
    }
    filtered = [
        revision
        for revision in revisions
        if isinstance(revision, dict)
        and revision.get("replacement_value") not in (None, "", [], {})
        and (
            revision.get("scene_num") in broad_scenes
            or (revision.get("scene_num"), str(revision.get("field_to_change"))) in exact
        )
    ]
    return {**response, "scene_revisions": filtered}


class PromptPlannerAgent:
    def __init__(
        self,
        backend: AgentBackend | None = None,
        *,
        target_fps: int = 24,
        critic: PlanCriticAgent | None = None,
        max_plan_revisions: int = 4,
    ) -> None:
        self.backend = backend
        self.target_fps = target_fps
        self.critic = critic or PlanCriticAgent(None)
        self.max_plan_revisions = max(0, int(max_plan_revisions))
        self.last_prompt: str | None = None
        self.last_schema: dict[str, Any] | None = None
        self.last_context: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.used_fallback: bool = False
        self.plan_attempts: list[dict[str, Any]] = []
        self.agent_steps: list[dict[str, Any]] = []
        self.last_validation_report: dict[str, Any] | None = None
        self.last_critic_report: dict[str, Any] | None = None

    def _diagnostic_rejected_plan(self, rhyme: NurseryRhymeInput, reason: str) -> ProductionPlan:
        plan = build_production_plan(rhyme, target_fps=self.target_fps)
        for scene in plan.scenes:
            scene.review_status = "rejected"
            scene.lyric_interpretation = ""
        plan.evaluation_rubric = {
            **plan.evaluation_rubric,
            "planning_mode": "diagnostic_rejected_scaffold",
            "planner_failure_reason": reason,
        }
        return plan

    def plan(self, rhyme: NurseryRhymeInput) -> ProductionPlan:
        self.last_error = None
        self.last_response = None
        self.used_fallback = False
        self.plan_attempts = []
        self.agent_steps = []
        self.last_validation_report = None
        self.last_critic_report = None
        if not self.backend:
            self.used_fallback = True
            plan = build_production_plan(rhyme, target_fps=self.target_fps)
            self.last_validation_report = validate_plan_semantics(plan)
            self.last_critic_report = self.critic.review(plan, self.last_validation_report)
            return plan
        self.last_prompt = _planner_prompt()
        self.last_schema = PLANNER_DECISION_SCHEMA
        self.last_context = {"response_key": "planner", "input": _planner_input_context(rhyme)}
        try:
            response = self.backend.generate_json(
                self.last_prompt,
                self.last_schema,
                self.last_context,
            )
        except Exception as exc:
            self.last_error = str(exc)
            plan = self._diagnostic_rejected_plan(rhyme, str(exc))
            self.last_validation_report = validate_plan_semantics(plan)
            self.last_critic_report = {
                "passed": False,
                "issues": [{"code": "planner_backend_error", "message": str(exc), "scene_num": None}],
                "scores": {},
                "revision_notes": [],
            }
            self.agent_steps.append({"kind": "planner_draft", "attempt": 1, "status": "backend_error", "error": str(exc)})
            return plan
        self.last_response = response
        candidate = response.get("planner_decision", response.get("production_plan", response))
        last_plan: ProductionPlan | None = None
        for attempt in range(0, self.max_plan_revisions + 1):
            try:
                if "lyric_segments" in candidate and "rhyme" in candidate:
                    plan = ProductionPlan.from_dict(candidate)
                else:
                    plan = production_plan_from_planner_decision(rhyme, candidate, target_fps=self.target_fps)
            except Exception as exc:
                self.last_error = str(exc)
                self.plan_attempts.append({"attempt": attempt + 1, "status": "conversion_failed", "error": str(exc), "candidate": candidate})
                self.agent_steps.append({"kind": "planner_draft" if attempt == 0 else "planner_revision", "attempt": attempt + 1, "status": "conversion_failed", "error": str(exc), "candidate": candidate})
                break
            last_plan = plan
            self.agent_steps.append({"kind": "planner_draft" if attempt == 0 else "planner_revision", "attempt": attempt + 1, "status": "converted", "candidate": candidate, "production_plan": plan.to_dict()})
            deterministic_report = validate_plan_semantics(plan, require_reviewer_approval=False)
            critic_report = self.critic.review(plan, deterministic_report)
            self.agent_steps.append({"kind": "plan_review", "attempt": attempt + 1, "status": "passed" if critic_report.get("passed") else "rejected", "review": critic_report})
            merged_issues = []
            seen_issues: set[tuple[str, Any]] = set()
            for issue in [*deterministic_report.get("issues", []), *critic_report.get("issues", [])]:
                key = (str(issue.get("code")), issue.get("scene_num")) if isinstance(issue, dict) else (str(issue), None)
                if key not in seen_issues:
                    seen_issues.add(key)
                    merged_issues.append(issue)
            self.last_critic_report = critic_report
            if not merged_issues:
                for scene in plan.scenes:
                    scene.review_status = "approved"
                self.last_validation_report = {
                    **validate_plan_semantics(plan, require_reviewer_approval=True),
                    "critic_passed": True,
                    "critic_issue_count": 0,
                }
            else:
                for scene in plan.scenes:
                    scene.review_status = "rejected" if critic_report.get("issues") else scene.review_status
                self.last_validation_report = {
                    **validate_plan_semantics(plan, require_reviewer_approval=True),
                    "critic_passed": critic_report.get("passed"),
                    "critic_issue_count": len(critic_report.get("issues", [])),
                }
            self.plan_attempts.append(
                {
                    "attempt": attempt + 1,
                    "status": "passed" if not merged_issues else "needs_revision",
                    "deterministic_report": deterministic_report,
                    "critic_report": critic_report,
                    "candidate": candidate,
                }
            )
            if not merged_issues:
                return plan
            if attempt >= self.max_plan_revisions:
                return plan
            targeted_revision = critic_report.get("targeted_revision", {})
            if isinstance(targeted_revision, dict) and (
                targeted_revision.get("scene_revisions") or targeted_revision.get("plan_updates")
            ):
                reviewer_candidate = _apply_planner_revision(candidate, targeted_revision)
                if reviewer_candidate != candidate:
                    self.agent_steps.append(
                        {
                            "kind": "planner_revision",
                            "attempt": attempt + 2,
                            "status": "reviewer_targeted_patch_applied",
                            "source": "PlanCriticAgent",
                            "targeted_revision": targeted_revision,
                        }
                    )
                    candidate = reviewer_candidate
                    continue
            revision_context = {
                "response_key": f"planner_revision_{attempt + 1}",
                "input": _planner_input_context(rhyme),
                "previous_decision": candidate,
                "production_plan": plan.to_dict(),
                "validation_issues": merged_issues,
                "instruction": (
                    "Return a complete replacement planner decision JSON. Preserve supplied lyrics exactly. "
                    "Edit structured scene descriptions/camera fields to fix only the listed issues before prompt compilation. "
                    "For unsafe_visual_action, use issue evidence excerpts to rewrite scene_goal, lyric_interpretation, "
                    "action, setting, subjects, camera, and safety_adaptation so unsupported descent, breakage, "
                    "dropping, crashing, impact, or other hazardous motion is replaced by visibly safe supported motion. "
                    "Remove hazardous wording from the scene instead of only adding a negation. For repeated_scene_staging, "
                    "materially change setting, action, and composition while preserving character continuity. For every "
                    "other issue code, use its message, scene_num, field, and evidence as binding acceptance criteria."
                ),
            }
            try:
                response = self.backend.generate_json(self.last_prompt, self.last_schema, revision_context)
            except Exception as exc:
                self.last_error = str(exc)
                self.agent_steps.append({"kind": "planner_revision", "attempt": attempt + 2, "status": "backend_error", "error": str(exc), "context": revision_context})
                break
            response = _constrain_revision_to_issues(response, merged_issues)
            candidate = _apply_planner_revision(candidate, response)
        if last_plan is not None:
            return last_plan
        fallback = self._diagnostic_rejected_plan(rhyme, self.last_error or "planner did not produce a convertible plan")
        self.last_validation_report = validate_plan_semantics(fallback)
        self.last_critic_report = {
            "passed": False,
            "issues": [{"code": "planner_conversion_failed", "message": self.last_error or "planner did not produce a convertible plan", "scene_num": None}],
            "scores": {},
            "revision_notes": [],
        }
        return fallback
