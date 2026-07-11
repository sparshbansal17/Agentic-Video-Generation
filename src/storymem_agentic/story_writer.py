from __future__ import annotations

from dataclasses import asdict

from .schemas import ProductionPlan


def story_from_plan(plan: ProductionPlan) -> dict:
    plan.validate()
    return {
        "story_name": "agentic_nursery_rhyme",
        "story_overview": (
            f"Agentic nursery-rhyme StoryMem script for {plan.rhyme.target_audience}. "
            "Planner-authored scene prompts carry per-shot visual detail and no in-frame text."
        ),
        "agentic_metadata": {
            "plan_version": plan.version,
            "target_fps": plan.target_fps,
            "audio_mode": plan.audio_mode,
            "music_prompt": plan.music_prompt,
            "clip_count": plan.clip_count,
            "character_bank": [asdict(profile) for profile in plan.character_bank],
        },
        "scenes": [
            {
                "scene_num": scene.scene_num,
                "lyric_line": scene.subtitle_text,
                "audio_description": scene.audio_description,
                "planned_start_seconds": scene.start_seconds,
                "planned_end_seconds": scene.end_seconds,
                "duration_seconds": scene.end_seconds - scene.start_seconds,
                "video_prompts": [scene.video_prompt],
                "first_frame_prompt": [scene.first_frame_prompt],
                "cut": [scene.cut],
                "subtitle_text": scene.subtitle_text,
            }
            for scene in plan.scenes
        ],
    }


def storymem_script_from_plan(plan: ProductionPlan) -> dict:
    plan.validate()
    return {
        "story_overview": (
            f"A polished child-safe preschool animated sing-along for {plan.rhyme.target_audience}. "
            "Each five-second StoryMem shot advances a clear, playful visual arc with readable action and expression."
        ),
        "scenes": [
            {
                "scene_num": scene.scene_num,
                "video_prompts": [scene.video_prompt],
                "cut": [scene.cut],
                # The parent pipeline uses this only when burning subtitles after generation.
                "subtitle_text": scene.subtitle_text,
            }
            for scene in plan.scenes
        ],
    }


def rhyme_text_from_plan(plan: ProductionPlan) -> str:
    return "\n".join(segment.text for segment in plan.lyric_segments) + "\n"
