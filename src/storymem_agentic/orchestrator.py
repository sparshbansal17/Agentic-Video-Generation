from __future__ import annotations

import json
import shutil
import shlex
import subprocess
from pathlib import Path
from string import Template
from typing import Any

from .alignment import analyze_whisperx_alignment, load_whisperx_words
from .audio_director import build_audio_plan
from .agents import CommandAgentBackend
from .backends import write_backend_manifest
from .feedback import apply_revision_plan, build_revision_plan
from .media_evaluator import evaluate_iteration
from .mixer import build_mix_manifest, write_mix_manifest
from .planner import PlanCriticAgent, PromptPlannerAgent, build_visual_bible, validate_plan_semantics
from .runner import run_audio_postprocess
from .schemas import AudioCandidate, EvaluationReport, NurseryRhymeInput, ProductionPlan, RevisionPlan, SceneHint, SongSpec
from .song_pipeline import candidate_manifest
from .story_writer import rhyme_text_from_plan, story_from_plan, storymem_script_from_plan
from .subtitles import add_whisperx_subtitles


REPO_ROOT = Path(__file__).resolve().parents[2]


def default_storymem_dir() -> Path:
    return REPO_ROOT


def storymem_dir_path(storymem_dir: str | Path | None) -> Path:
    return Path(storymem_dir).expanduser().resolve() if storymem_dir else default_storymem_dir().resolve()


def resolve_storymem_dir(storymem_dir: str | Path | None) -> Path:
    resolved = storymem_dir_path(storymem_dir)
    required = [resolved / "pipeline.py", resolved / "wan", resolved / "extract_keyframes.py"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "StoryMem runtime directory is missing required in-repo files: " + ", ".join(missing)
        )
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_rhyme_input(
    *,
    rhyme_file: str | Path | None = None,
    topic_or_name: str = "",
    lyrics: str | None = None,
    lyrics_file: str | Path | None = None,
    output_root: str | Path,
    target_duration: float | None = None,
    target_audience: str = "toddlers",
    visual_style: str = "bright rounded toddler-safe bedtime storybook animation",
    audio_style: str = "warm clear nursery singalong, soft music box and celesta",
    max_iterations: int = 1,
    seed: int = 0,
    clip_count: int | None = None,
    character_db_path: str | Path | None = None,
    character_bank_path: str | Path | None = None,
) -> NurseryRhymeInput:
    rhyme_text = Path(rhyme_file).read_text(encoding="utf-8") if rhyme_file else ""
    supplied_lyrics = Path(lyrics_file).read_text(encoding="utf-8") if lyrics_file else lyrics
    input_data = NurseryRhymeInput(
        rhyme_text=rhyme_text,
        topic_or_name=topic_or_name,
        lyrics=supplied_lyrics,
        rhyme_file=str(rhyme_file) if rhyme_file else None,
        target_audience=target_audience,
        target_duration_seconds=target_duration,
        visual_style=visual_style,
        audio_style=audio_style,
        max_iterations=max_iterations,
        seed=seed,
        output_root=str(output_root),
        clip_count=clip_count,
        character_db_path=str(character_db_path) if character_db_path else None,
        character_bank_path=str(character_bank_path) if character_bank_path else None,
    )
    input_data.validate()
    return input_data


def write_iteration_artifacts(root: Path, iteration: int, plan: ProductionPlan, *, dry_run: bool) -> dict[str, Path]:
    iteration_dir = root / "iterations" / f"{iteration:03d}"
    generated_dir = iteration_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    story = story_from_plan(plan)
    storymem_story = storymem_script_from_plan(plan)
    first_shot_story = {
        **storymem_story,
        "scenes": [
            {
                **storymem_story["scenes"][0],
                "video_prompts": [storymem_story["scenes"][0]["video_prompts"][0]],
                "cut": [storymem_story["scenes"][0]["cut"][0]],
            }
        ],
    }
    rhyme_text = rhyme_text_from_plan(plan)
    planned_duration = max(scene.end_seconds for scene in plan.scenes)
    audio_voice_backend = "ace_step_full_song" if plan.audio_mode == "full_song" else "f5_tts"
    audio_music_backend = "ace_step_full_song" if plan.audio_mode == "full_song" else "musicgen"
    audio_plan = build_audio_plan(
        rhyme_text,
        scene_hints=[SceneHint.from_story_scene(scene) for scene in story["scenes"]],
        story_summary=story["story_overview"],
        target_duration_seconds=planned_duration,
        mode=plan.audio_mode,
        voice_backend=audio_voice_backend,
        music_backend=audio_music_backend,
        music_style=plan.music_prompt,
    )
    mix_manifest = build_mix_manifest(audio_plan)

    paths = {
        "iteration_dir": iteration_dir,
        "generated_dir": generated_dir,
        "production_plan": write_json(iteration_dir / "production_plan.json", plan.to_dict()),
        "story": write_json(iteration_dir / "story.json", story),
        "storymem_story": write_json(iteration_dir / "storymem_story.json", storymem_story),
        "story_t2v_first_shot": write_json(iteration_dir / "story_t2v_first_shot.json", first_shot_story),
        "audio_plan": write_json(iteration_dir / "audio_plan.json", audio_plan.to_dict()),
        "mix_manifest": write_mix_manifest(iteration_dir / "mix_manifest.json", mix_manifest),
        "backend_manifest": write_backend_manifest(
            iteration_dir / "backend_invocations.json",
            {
                "duration": audio_plan.target_duration_seconds,
                "music_prompt": audio_plan.music_prompt,
                "lyrics_file": "lyrics.txt",
                "prompt_file": "audio_plan.json",
                "output_file": "out.wav",
            },
        ),
    }
    (iteration_dir / "lyrics.txt").write_text(rhyme_text, encoding="utf-8")
    (iteration_dir / "review_frames").mkdir(exist_ok=True)
    (iteration_dir / "review_reports").mkdir(exist_ok=True)
    write_json(
        iteration_dir / "run_context.json",
        {
            "dry_run": dry_run,
            "iteration": iteration,
            "generated_dir": str(generated_dir),
        },
    )
    return paths


def existing_iteration_paths(root: Path, iteration: int) -> dict[str, Path]:
    iteration_dir = root / "iterations" / f"{iteration:03d}"
    generated_dir = iteration_dir / "generated"
    return {
        "iteration_dir": iteration_dir,
        "generated_dir": generated_dir,
        "production_plan": iteration_dir / "production_plan.json",
        "story": iteration_dir / "story.json",
        "storymem_story": iteration_dir / "storymem_story.json",
        "story_t2v_first_shot": iteration_dir / "story_t2v_first_shot.json",
        "audio_plan": iteration_dir / "audio_plan.json",
        "mix_manifest": iteration_dir / "mix_manifest.json",
        "backend_manifest": iteration_dir / "backend_invocations.json",
    }


def build_storymem_commands(
    *,
    story_json: Path,
    first_shot_story_json: Path | None = None,
    output_dir: Path,
    storymem_dir: str | Path,
    t2v_model_path: str | Path,
    i2v_model_path: str | Path,
    lora_weight_path: str | Path,
    nproc_per_node: int = 8,
    size: str = "832*480",
    max_memory_size: int = 10,
    ffmpeg_bin: str = "ffmpeg",
    seed: int = 0,
    t5_cpu: bool = False,
    offload_model: bool = False,
    sample_steps: int | None = None,
    frame_num: int | None = None,
    keyframe_mode: str = "hps",
) -> list[list[str]]:
    storymem_dir = storymem_dir_path(storymem_dir)
    story_json = story_json.resolve()
    first_shot_story_json = first_shot_story_json.resolve() if first_shot_story_json else None
    output_dir = output_dir.resolve()
    t2v_model_path = Path(t2v_model_path).resolve()
    i2v_model_path = Path(i2v_model_path).resolve()
    lora_weight_path = Path(lora_weight_path).resolve()

    def command_for(script_path: Path, *, mi2v: bool) -> list[str]:
        command = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "pipeline.py",
        "--story_script_path",
        str(script_path),
        "--t2v_model_path",
        str(t2v_model_path),
        "--i2v_model_path",
        str(i2v_model_path),
        "--lora_weight_path",
        str(lora_weight_path),
        "--size",
        size,
        "--max_memory_size",
        str(max_memory_size),
        "--output_dir",
        str(output_dir),
        "--ffmpeg_bin",
        ffmpeg_bin,
        "--seed",
        str(seed),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        str(nproc_per_node),
        "--lora_rank",
        "128",
        ]
        if keyframe_mode != "hps":
            command.extend(["--keyframe_mode", keyframe_mode])
        if offload_model:
            command.append("--offload_model")
        if sample_steps is not None:
            command.extend(["--sample_steps", str(sample_steps)])
        if frame_num is not None:
            command.extend(["--frame_num", str(frame_num)])
        if t5_cpu:
            command.append("--t5_cpu")
        if mi2v:
            command.append("--mi2v")
        return command

    first = [*command_for(first_shot_story_json or story_json, mi2v=False), "--t2v_first_shot"]
    second = command_for(story_json, mi2v=True)
    return [first, second]


def build_storymem_continuation_command(
    *,
    story_json: Path,
    output_dir: Path,
    storymem_dir: str | Path,
    t2v_model_path: str | Path,
    i2v_model_path: str | Path,
    lora_weight_path: str | Path,
    nproc_per_node: int = 8,
    size: str = "832*480",
    max_memory_size: int = 10,
    ffmpeg_bin: str = "ffmpeg",
    seed: int = 0,
    t5_cpu: bool = False,
    offload_model: bool = False,
    sample_steps: int | None = None,
    frame_num: int | None = None,
    keyframe_mode: str = "hps",
) -> list[str]:
    storymem_dir = storymem_dir_path(storymem_dir)
    story_json = story_json.resolve()
    output_dir = output_dir.resolve()
    t2v_model_path = Path(t2v_model_path).resolve()
    i2v_model_path = Path(i2v_model_path).resolve()
    lora_weight_path = Path(lora_weight_path).resolve()
    command = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "pipeline.py",
        "--story_script_path",
        str(story_json),
        "--t2v_model_path",
        str(t2v_model_path),
        "--i2v_model_path",
        str(i2v_model_path),
        "--lora_weight_path",
        str(lora_weight_path),
        "--size",
        size,
        "--max_memory_size",
        str(max_memory_size),
        "--output_dir",
        str(output_dir),
        "--ffmpeg_bin",
        ffmpeg_bin,
        "--seed",
        str(seed),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        str(nproc_per_node),
        "--lora_rank",
        "128",
    ]
    if keyframe_mode != "hps":
        command.extend(["--keyframe_mode", keyframe_mode])
    command.append("--mi2v")
    if offload_model:
        command.append("--offload_model")
    if sample_steps is not None:
        command.extend(["--sample_steps", str(sample_steps)])
    if frame_num is not None:
        command.extend(["--frame_num", str(frame_num)])
    if t5_cpu:
        command.append("--t5_cpu")
    return command


def write_targeted_story_json(full_story_path: Path, output_path: Path, start_scene_num: int) -> Path:
    story = json.loads(full_story_path.read_text(encoding="utf-8"))
    story["scenes"] = [
        scene for scene in story.get("scenes", []) if int(scene.get("scene_num", 0)) >= start_scene_num
    ]
    return write_json(output_path, story)


def _storymem_story_path(paths: dict[str, Path]) -> Path:
    storymem_story = paths.get("storymem_story")
    if storymem_story and storymem_story.exists():
        return storymem_story
    return paths["story"]


def _command_option(command: list[str], option: str) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def ensure_first_shot_memory_keyframe(output_dir: str | Path) -> Path | None:
    generated_dir = Path(output_dir)
    if list(generated_dir.glob("*keyframe*.jpg")):
        return None
    last_frame = generated_dir / "last_frame.jpg"
    if not last_frame.exists():
        return None
    fallback_keyframe = generated_dir / "01_01_keyframe0.jpg"
    shutil.copyfile(last_frame, fallback_keyframe)
    return fallback_keyframe


def run_storymem_commands(commands: list[list[str]], *, cwd: str | Path) -> None:
    for command in commands:
        subprocess.run(command, cwd=str(cwd), check=True)
        if "--t2v_first_shot" in command:
            output_dir = _command_option(command, "--output_dir")
            if output_dir:
                ensure_first_shot_memory_keyframe(output_dir)


def resolve_storymem_video(generated_dir: Path) -> Path:
    candidates = [
        generated_dir / "final_video.mp4",
        generated_dir / f"{generated_dir.name}_subtitled.mp4",
        generated_dir / f"{generated_dir.name}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_whisperx_command(
    *,
    command_template: str | None,
    audio_file: Path,
    output_dir: Path,
    output_file: Path,
) -> Path | None:
    if not command_template:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "audio_file": str(audio_file),
        "output_dir": str(output_dir),
        "output_file": str(output_file),
    }
    rendered = Template(command_template).safe_substitute(
        {key: shlex.quote(value) for key, value in values.items()}
    )
    try:
        subprocess.run(shlex.split(rendered), check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        write_json(
            output_dir / "whisperx_command_error.json",
            {
                "command": rendered,
                "audio_file": str(audio_file),
                "output_file": str(output_file),
                "error": str(exc),
            },
        )
        return None
    if output_file.exists():
        return output_file
    json_outputs = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if json_outputs:
        shutil.copy2(json_outputs[0], output_file)
        return output_file
    return None


def _is_whisperx_timing_failure(report: EvaluationReport | None) -> bool:
    if report is None:
        return False
    if not report.whisperx_alignment:
        return False
    return any(
        str(reason).startswith("line_") or str(reason) in {"wer_above_threshold", "missing_observed_lyrics"}
        for reason in report.whisperx_alignment.get("failure_reasons", [])
    )


def analyze_whisperx_for_plan(
    plan: ProductionPlan,
    whisperx_alignment_path: Path,
    *,
    enforce_scene_windows: bool = True,
) -> dict[str, Any]:
    aligned_words = load_whisperx_words(whisperx_alignment_path)
    return analyze_whisperx_alignment(
        [segment.text for segment in plan.lyric_segments],
        [(segment.start_seconds, segment.end_seconds) for segment in plan.lyric_segments],
        aligned_words,
        enforce_scene_windows=enforce_scene_windows,
    )


def _needs_scene_lyrics_audio_fallback(alignment: dict[str, Any] | None) -> bool:
    if not alignment or alignment.get("passed"):
        return False
    return any(
        str(reason).startswith("line_")
        or str(reason) in {"wer_above_threshold", "missing_observed_lyrics", "lyrics_start_too_late"}
        for reason in alignment.get("failure_reasons", [])
    )


def _alignment_rank(alignment: dict[str, Any] | None) -> tuple[int, float, int, int, int, float, float, int]:
    if not alignment:
        return (1, 999.0, 999, 999, 999, 1.0, 999.0, 999)
    lines = alignment.get("lines", []) or []
    missing_words = sum(int(item.get("missing_word_count") or 0) for item in lines)
    final_line_missing = 0
    if lines:
        final = lines[-1]
        final_line_missing = max(
            int(final.get("expected_word_count") or 0) - int(final.get("matched_word_count") or 0),
            0,
        )
    repeated_omissions = int(alignment.get("repeated_word_omission_count") or 0)
    drift = 0.0
    for item in lines:
        drift += abs(float(item.get("start_drift_seconds") or 0.0))
        drift += abs(float(item.get("end_drift_seconds") or 0.0))
    return (
        0 if alignment.get("passed") else 1,
        (
            float(alignment["initial_lyric_start_seconds"])
            if alignment.get("initial_lyric_start_seconds") is not None
            else 999.0
        ),
        final_line_missing,
        repeated_omissions,
        missing_words,
        float(alignment.get("word_error_rate", 1.0) or 1.0),
        drift,
        len(alignment.get("failure_reasons", []) or []),
    )


def _bounded_candidate_count(*counts: int) -> int:
    return max(1, min(max(1, int(count)) for count in counts))


def _retry_seed_offsets(candidate_count: int, seed_offsets: list[int]) -> list[int]:
    return seed_offsets[: max(0, candidate_count - 1)]


def run_workflow(
    *,
    rhyme_file: str | Path | None = None,
    topic_or_name: str = "",
    lyrics: str | None = None,
    lyrics_file: str | Path | None = None,
    output_dir: str | Path,
    mode: str = "dry_run",
    target_duration: float | None = None,
    target_audience: str = "toddlers",
    visual_style: str = "bright rounded toddler-safe bedtime storybook animation",
    audio_style: str = "warm clear nursery singalong, soft music box and celesta",
    clip_count: int | None = None,
    max_iterations: int = 1,
    seed: int = 0,
    storymem_dir: str | Path | None = None,
    t2v_model_path: str | Path | None = None,
    i2v_model_path: str | Path | None = None,
    lora_weight_path: str | Path | None = None,
    nproc_per_node: int = 8,
    ffmpeg_bin: str = "ffmpeg",
    storymem_t5_cpu: bool = False,
    storymem_offload_model: bool = False,
    storymem_sample_steps: int | None = None,
    storymem_frame_num: int | None = None,
    storymem_keyframe_mode: str = "hps",
    execute_video: bool = False,
    character_db_path: str | Path | None = None,
    character_bank_path: str | Path | None = None,
    production_plan_path: str | Path | None = None,
    planner_backend: str = "mock",
    planner_command: str | None = None,
    plan_critic_command: str | None = None,
    max_plan_revisions: int = 8,
    plan_validation_policy: str = "block",
    review_backend: str = "mock",
    vlm_command: str | None = None,
    audio_review_command: str | None = None,
    whisperx_command: str | None = None,
    audio_aligner: str = "whisperx",
    add_transcribed_subtitles: bool = True,
    strict_lullaby_review: bool = True,
    generate_audio: bool = True,
    media_audio_mode: str = "full_song",
    voice_backend: str | None = None,
    music_backend: str | None = None,
    audio_output_suffix: str = "_with_music",
    audio_voice_style: str = "warm clear toddler nursery singalong vocal, exact lyrics, gentle adult lead with soft childlike brightness, music box, celesta, glockenspiel star twinkles, soft strings, light harp, slow 3/4 sway, vocals forward",
    ace_step_cmd: str | None = None,
    vocal_cmd: str | None = None,
    backing_cmd: str | None = None,
    musicgen_cmd: str | None = None,
    song_cmd: str | None = None,
    voice_ref_audio: str | None = None,
    voice_ref_text: str | None = None,
    allow_scene_mix_debug: bool = False,
    full_song_candidate_count: int = 8,
    voice_candidate_count: int = 4,
    music_candidate_count: int = 1,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    nursery_input = load_rhyme_input(
        rhyme_file=rhyme_file,
        topic_or_name=topic_or_name,
        lyrics=lyrics,
        lyrics_file=lyrics_file,
        output_root=root,
        target_duration=target_duration,
        target_audience=target_audience,
        visual_style=visual_style,
        audio_style=audio_style,
        max_iterations=max_iterations,
        seed=seed,
        clip_count=clip_count,
        character_db_path=character_db_path,
        character_bank_path=character_bank_path,
    )
    critic_backend = CommandAgentBackend(plan_critic_command) if plan_critic_command else None
    planner = PromptPlannerAgent(
        CommandAgentBackend(planner_command) if planner_backend == "command" and planner_command else None,
        critic=PlanCriticAgent(critic_backend),
        max_plan_revisions=max_plan_revisions,
    )
    if production_plan_path:
        source_plan_path = Path(production_plan_path)
        plan = ProductionPlan.from_dict(json.loads(source_plan_path.read_text(encoding="utf-8")))
        if plan.version != "1.1":
            if planner.backend is None:
                raise ValueError(
                    f"legacy production plan requires agentic replanning under the 1.1 semantic contract: {source_plan_path}"
                )
            plan = planner.plan(nursery_input)
            planner.last_context = {
                **(planner.last_context or {}),
                "production_plan_path": str(source_plan_path),
                "legacy_plan_replanned": True,
                "reused_finalized_plan": False,
            }
        else:
            deterministic_report = validate_plan_semantics(plan, require_reviewer_approval=False)
            critic_report = planner.critic.review(plan, deterministic_report)
            if deterministic_report.get("issues") or not critic_report.get("passed"):
                raise ValueError(f"supplied production plan failed current semantic review: {source_plan_path}")
            for scene in plan.scenes:
                scene.review_status = "approved"
            validation_report = validate_plan_semantics(plan, require_reviewer_approval=True)
            planner.last_validation_report = validation_report
            planner.last_critic_report = critic_report
            planner.last_context = {
                "production_plan_path": str(source_plan_path),
                "reused_finalized_plan": True,
                "semantic_review_rerun": True,
            }
    else:
        plan = planner.plan(nursery_input)
    write_json(root / "nursery_rhyme_input.json", nursery_input.to_dict())
    validation_report = planner.last_validation_report or validate_plan_semantics(plan)
    critic_report = planner.last_critic_report or {"passed": True, "issues": [], "scores": {}, "revision_notes": []}
    visual_bible = validation_report.get("visual_bible") or build_visual_bible(plan)
    write_json(root / "lyrics_resolution.json", {"source": "supplied" if nursery_input.source_lyrics else "planner_or_fallback", "lines": [segment.text for segment in plan.lyric_segments]})
    write_json(
        root / "character_bank_resolution.json",
        {
            "source_path": nursery_input.character_bank_path or nursery_input.character_db_path,
            "characters": [
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
                for profile in plan.character_bank
            ],
        },
    )
    write_json(root / "visual_bible.json", visual_bible)
    write_json(root / "plan_validation_report.json", validation_report)
    write_json(root / "plan_critic_report.json", critic_report)
    for index, attempt in enumerate(planner.plan_attempts, start=1):
        write_json(root / f"planner_attempt_{index:02d}.json", attempt)
    step_counters: dict[str, int] = {}
    for step in getattr(planner, "agent_steps", []):
        kind = str(step.get("kind", "planner_step"))
        if kind not in {"planner_draft", "plan_review", "planner_revision"}:
            kind = "planner_step"
        step_counters[kind] = step_counters.get(kind, 0) + 1
        write_json(root / f"{kind}_{step_counters[kind]:02d}.json", step)
    planner_output = {
        "backend": planner_backend,
        "production_plan_path": str(production_plan_path) if production_plan_path else None,
        "reused_finalized_plan": bool((planner.last_context or {}).get("reused_finalized_plan", False)),
        "command": planner_command,
        "plan_critic_command": plan_critic_command,
        "max_plan_revisions": max_plan_revisions,
        "plan_validation_policy": plan_validation_policy,
        "used_fallback": planner.used_fallback,
        "error": planner.last_error,
        "validation_passed": bool(validation_report.get("passed")),
        "validation_issue_count": int(validation_report.get("issue_count", 0)),
        "critic_passed": bool(critic_report.get("passed", True)),
        "critic_issue_count": len(critic_report.get("issues", [])),
        "fallback_policy": (
            "non_agentic_local_dry_run_only"
            if planner_backend != "command" and planner.used_fallback
            else None
        ),
        "prompt": planner.last_prompt,
        "schema": planner.last_schema,
        "context": planner.last_context,
        "response": planner.last_response,
        "attempt_count": len(planner.plan_attempts),
    }
    write_json(
        root / "planner_agent_output.json",
        planner_output,
    )
    validation_failed = not bool(validation_report.get("passed")) or not bool(critic_report.get("passed", True))
    if planner.used_fallback and mode in {"generate", "iterate"} and execute_video:
        validation_failed = True
        validation_report = {
            **validation_report,
            "passed": False,
            "issues": [
                *validation_report.get("issues", []),
                {
                    "code": "non_agentic_plan_cannot_launch_gpu_generation",
                    "scene_num": None,
                    "message": "local non-agentic planner scaffold is allowed only for dry-run/offline artifacts",
                },
            ],
            "issue_count": int(validation_report.get("issue_count", 0)) + 1,
        }
        write_json(root / "plan_validation_report.json", validation_report)
        planner_output["validation_passed"] = False
        planner_output["validation_issue_count"] = int(validation_report.get("issue_count", 0))
        write_json(root / "planner_agent_output.json", planner_output)
    if validation_failed and plan_validation_policy == "block" and mode in {"generate", "iterate"}:
        raise ValueError(
            "planner validation failed before GPU generation; see plan_validation_report.json and plan_critic_report.json"
        )

    iteration_count = 1 if mode in {"dry_run", "generate"} else max_iterations
    latest_report: EvaluationReport | None = None
    latest_paths: dict[str, Path] = {}
    latest_final_candidate: Path | None = None
    reuse_generated_video_dir: Path | None = None
    visual_regeneration_start_scene: int | None = None
    current_media_audio_mode = media_audio_mode
    if execute_video:
        active_storymem_dir = resolve_storymem_dir(storymem_dir)
    elif storymem_dir or (t2v_model_path and i2v_model_path and lora_weight_path):
        active_storymem_dir = storymem_dir_path(storymem_dir)
    else:
        active_storymem_dir = None

    for iteration in range(1, iteration_count + 1):
        cached_paths = existing_iteration_paths(root, iteration)
        cached_report_path = cached_paths["iteration_dir"] / "evaluation_report.json"
        cached_revision_path = cached_paths["iteration_dir"] / "revision_plan.json"
        if mode == "iterate" and cached_report_path.exists():
            latest_paths = cached_paths
            if cached_paths["production_plan"].exists():
                plan = ProductionPlan.from_dict(json.loads(cached_paths["production_plan"].read_text(encoding="utf-8")))
            latest_report = EvaluationReport.from_dict(json.loads(cached_report_path.read_text(encoding="utf-8")))
            if cached_revision_path.exists():
                cached_revision = RevisionPlan(**json.loads(cached_revision_path.read_text(encoding="utf-8")))
            else:
                cached_revision = build_revision_plan(plan, latest_report)
                write_json(cached_revision_path, cached_revision.to_dict())
            latest_final_candidate = (
                cached_paths["generated_dir"] / "generated_subtitled_with_music.mp4"
                if (cached_paths["generated_dir"] / "generated_subtitled_with_music.mp4").exists()
                else resolve_storymem_video(cached_paths["generated_dir"])
            )
            cached_audio_result_path = cached_paths["iteration_dir"] / "audio_postprocess_result.json"
            if cached_audio_result_path.exists():
                cached_audio_result = json.loads(cached_audio_result_path.read_text(encoding="utf-8"))
                if cached_audio_result.get("fallback_from_full_song"):
                    current_media_audio_mode = "hybrid_voice_bed" if media_audio_mode == "hybrid_voice_bed" else "scene_lyrics_mix"
            if latest_report.passed:
                break
            if cached_revision.target_scenes:
                first_target = min(cached_revision.target_scenes)
                reuse_generated_video_dir = cached_paths["generated_dir"] if first_target > 1 else None
                visual_regeneration_start_scene = first_target if first_target > 1 else None
            elif cached_revision.regenerate_audio:
                reuse_generated_video_dir = cached_paths["generated_dir"]
                visual_regeneration_start_scene = None
                if current_media_audio_mode == "full_song" and _is_whisperx_timing_failure(latest_report):
                    current_media_audio_mode = "hybrid_voice_bed" if media_audio_mode == "hybrid_voice_bed" else "scene_lyrics_mix"
            else:
                reuse_generated_video_dir = None
                visual_regeneration_start_scene = None
            plan = apply_revision_plan(plan, cached_revision)
            continue

        latest_paths = write_iteration_artifacts(root, iteration, plan, dry_run=mode == "dry_run")
        commands = []
        if mode in {"generate", "iterate"}:
            if reuse_generated_video_dir is not None:
                shutil.copytree(reuse_generated_video_dir, latest_paths["generated_dir"], dirs_exist_ok=True)
                if visual_regeneration_start_scene is not None and active_storymem_dir and t2v_model_path and i2v_model_path and lora_weight_path:
                    targeted_story = write_targeted_story_json(
                        _storymem_story_path(latest_paths),
                        latest_paths["iteration_dir"] / f"story_from_scene_{visual_regeneration_start_scene:02d}.json",
                        visual_regeneration_start_scene,
                    )
                    commands = [
                        build_storymem_continuation_command(
                            story_json=targeted_story,
                            output_dir=latest_paths["generated_dir"],
                            storymem_dir=active_storymem_dir,
                            t2v_model_path=t2v_model_path,
                            i2v_model_path=i2v_model_path,
                            lora_weight_path=lora_weight_path,
                            nproc_per_node=nproc_per_node,
                            ffmpeg_bin=ffmpeg_bin,
                            seed=seed + ((iteration - 1) * 9973),
                            t5_cpu=storymem_t5_cpu,
                            offload_model=storymem_offload_model,
                            sample_steps=storymem_sample_steps,
                            frame_num=storymem_frame_num,
                            keyframe_mode=storymem_keyframe_mode,
                        )
                    ]
                    write_json(
                        latest_paths["iteration_dir"] / "storymem_commands.json",
                        {
                            "commands": commands,
                            "notes": (
                                f"reused preserved video artifacts from {reuse_generated_video_dir}; "
                                f"regenerating from scene {visual_regeneration_start_scene} with MI2V memory"
                            ),
                        },
                    )
                    if execute_video:
                        run_storymem_commands(commands, cwd=active_storymem_dir)
                else:
                    write_json(
                        latest_paths["iteration_dir"] / "storymem_commands.json",
                        {
                            "commands": [],
                            "notes": f"reused accepted video artifacts from {reuse_generated_video_dir} for audio-only revision",
                        },
                    )
            elif active_storymem_dir and t2v_model_path and i2v_model_path and lora_weight_path:
                commands = build_storymem_commands(
                    story_json=_storymem_story_path(latest_paths),
                    first_shot_story_json=latest_paths["story_t2v_first_shot"],
                    output_dir=latest_paths["generated_dir"],
                    storymem_dir=active_storymem_dir,
                    t2v_model_path=t2v_model_path,
                    i2v_model_path=i2v_model_path,
                    lora_weight_path=lora_weight_path,
                    nproc_per_node=nproc_per_node,
                    ffmpeg_bin=ffmpeg_bin,
                    seed=seed + ((iteration - 1) * 9973),
                    t5_cpu=storymem_t5_cpu,
                    offload_model=storymem_offload_model,
                    sample_steps=storymem_sample_steps,
                    frame_num=storymem_frame_num,
                    keyframe_mode=storymem_keyframe_mode,
                )
                write_json(latest_paths["iteration_dir"] / "storymem_commands.json", {"commands": commands})
                if execute_video:
                    run_storymem_commands(commands, cwd=active_storymem_dir)
            else:
                write_json(
                    latest_paths["iteration_dir"] / "storymem_commands.json",
                    {"commands": [], "notes": "model paths not provided"},
                )
        else:
            write_json(
                latest_paths["iteration_dir"] / "storymem_commands.json",
                {"commands": [], "notes": "dry-run planning only"},
            )

        video_candidate = resolve_storymem_video(latest_paths["generated_dir"])
        final_candidate = video_candidate
        audio_result: dict[str, Any] | None = None
        if mode in {"generate", "iterate"} and generate_audio and video_candidate.exists():
            if voice_backend:
                audio_voice_backend = voice_backend
            elif current_media_audio_mode == "hybrid_voice_bed":
                audio_voice_backend = "f5_tts"
            else:
                audio_voice_backend = "ace_step_full_song" if plan.audio_mode == "full_song" else "f5_tts"
            if current_media_audio_mode in {"hybrid_voice_bed", "scene_lyrics_mix"} and audio_voice_backend in {
                "ace_step",
                "ace_step_full_song",
            }:
                audio_voice_backend = "ace_step" if ace_step_cmd else "f5_tts"
            if music_backend:
                audio_music_backend = music_backend
            elif current_media_audio_mode == "hybrid_voice_bed":
                audio_music_backend = "musicgen"
            else:
                audio_music_backend = "ace_step_full_song" if plan.audio_mode == "full_song" else "musicgen"
            if current_media_audio_mode in {"hybrid_voice_bed", "scene_lyrics_mix"} and audio_music_backend in {
                "ace_step_full_song",
            }:
                audio_music_backend = "ace_step" if ace_step_cmd else "musicgen"
            effective_media_audio_mode = media_audio_mode
            if plan.audio_mode == "voice_bed" and media_audio_mode == "full_song":
                effective_media_audio_mode = "separate_stems"

            def render_audio_candidate(media_mode: str, *, seed_offset: int = 0) -> dict[str, Any]:
                return run_audio_postprocess(
                    rhyme_file=latest_paths["iteration_dir"] / "lyrics.txt",
                    story_json=latest_paths["story"],
                    output_dir=latest_paths["generated_dir"],
                    final_video=video_candidate,
                    mode=plan.audio_mode,
                    voice_backend=audio_voice_backend,
                    music_backend=audio_music_backend,
                    media_audio_mode=media_mode,
                    audio_output_suffix=audio_output_suffix,
                    audio_voice_style=audio_voice_style,
                    ace_step_cmd=ace_step_cmd,
                    vocal_cmd=vocal_cmd,
                    backing_cmd=backing_cmd,
                    musicgen_cmd=musicgen_cmd,
                    song_cmd=song_cmd,
                    voice_ref_audio=voice_ref_audio,
                    voice_ref_text=voice_ref_text,
                    ffmpeg_bin=ffmpeg_bin,
                    seed=seed + ((iteration - 1) * 9973) + seed_offset,
                    dry_run=False,
                )

            def run_audio_alignment(candidate: Path, label: str) -> Path | None:
                whisperx_alignment_path.unlink(missing_ok=True)
                return run_whisperx_command(
                    command_template=whisperx_command,
                    audio_file=candidate,
                    output_dir=latest_paths["iteration_dir"] / label,
                    output_file=whisperx_alignment_path,
                )

            requested_media_audio_mode = (
                current_media_audio_mode if plan.audio_mode == "full_song" else effective_media_audio_mode
            )
            audio_result = render_audio_candidate(requested_media_audio_mode)
            if audio_result.get("media_output"):
                final_candidate = Path(str(audio_result["media_output"]))
            write_json(latest_paths["iteration_dir"] / "audio_postprocess_result.json", audio_result)

        whisperx_alignment_path = latest_paths["iteration_dir"] / "whisperx_alignment.json"
        if (
            mode in {"generate", "iterate"}
            and generate_audio
            and audio_aligner == "whisperx"
            and final_candidate.exists()
        ):
            run_audio_alignment(final_candidate, "whisperx")

        if (
            mode in {"generate", "iterate"}
            and generate_audio
            and audio_aligner == "whisperx"
            and plan.audio_mode == "full_song"
            and current_media_audio_mode == "full_song"
            and audio_result is not None
            and whisperx_alignment_path.exists()
        ):
            full_song_alignment = analyze_whisperx_for_plan(
                plan, whisperx_alignment_path, enforce_scene_windows=False
            )
            if _needs_scene_lyrics_audio_fallback(full_song_alignment) and full_song_candidate_count > 1:
                attempts_dir = latest_paths["iteration_dir"] / "full_song_audio_candidates"
                attempts_dir.mkdir(parents=True, exist_ok=True)
                best_alignment = full_song_alignment
                best_rank = _alignment_rank(full_song_alignment)
                best_media = attempts_dir / "candidate_00.mp4"
                best_json = attempts_dir / "candidate_00_whisperx_alignment.json"
                shutil.copy2(final_candidate, best_media)
                shutil.copy2(whisperx_alignment_path, best_json)
                candidate_records = [
                    {
                        "candidate": 0,
                        "seed_offset": 0,
                        "alignment": full_song_alignment,
                        "media_output": str(best_media),
                    }
                ]
                for candidate, retry_seed_offset in enumerate([1601, 3203, 4801, 6407, 8009, 9613, 11213], start=1):
                    if candidate >= full_song_candidate_count:
                        break
                    shutil.rmtree(latest_paths["generated_dir"] / "audio", ignore_errors=True)
                    retry_audio_result = render_audio_candidate("full_song", seed_offset=retry_seed_offset)
                    if retry_audio_result.get("media_output"):
                        final_candidate = Path(str(retry_audio_result["media_output"]))
                    run_audio_alignment(final_candidate, f"whisperx_full_song_candidate_{candidate:02d}")
                    if not whisperx_alignment_path.exists():
                        continue
                    retry_alignment = analyze_whisperx_for_plan(
                        plan, whisperx_alignment_path, enforce_scene_windows=False
                    )
                    media_copy = attempts_dir / f"candidate_{candidate:02d}.mp4"
                    json_copy = attempts_dir / f"candidate_{candidate:02d}_whisperx_alignment.json"
                    shutil.copy2(final_candidate, media_copy)
                    shutil.copy2(whisperx_alignment_path, json_copy)
                    candidate_records.append(
                        {
                            "candidate": candidate,
                            "seed_offset": retry_seed_offset,
                            "alignment": retry_alignment,
                            "media_output": str(media_copy),
                        }
                    )
                    retry_rank = _alignment_rank(retry_alignment)
                    if retry_rank < best_rank:
                        best_rank = retry_rank
                        best_alignment = retry_alignment
                        best_media = media_copy
                        best_json = json_copy
                        audio_result = retry_audio_result
                    if retry_alignment.get("passed"):
                        break
                if best_media.exists():
                    shutil.copy2(best_media, final_candidate)
                if best_json.exists():
                    shutil.copy2(best_json, whisperx_alignment_path)
                full_song_alignment = best_alignment
                audio_result["full_song_candidate_attempts"] = candidate_records
                audio_result["selected_full_song_alignment"] = best_alignment
                audio_result["selected_full_song_media"] = str(best_media)
                audio_result["candidate_policy"] = {
                    "full_song": full_song_candidate_count,
                    "voice": voice_candidate_count,
                    "music": music_candidate_count,
                }
                write_json(latest_paths["iteration_dir"] / "audio_postprocess_result.json", audio_result)
            if _needs_scene_lyrics_audio_fallback(full_song_alignment):
                failure_record = {
                    "reason": "full_song_lyric_or_timing_failure",
                    "full_song_alignment": full_song_alignment,
                    "retained_full_song_media": audio_result.get("selected_full_song_media", str(final_candidate)),
                    "policy": "retain_best_one_take_for_review_and_regenerate_one_take_next_iteration",
                }
                write_json(latest_paths["iteration_dir"] / "full_song_generation_failure.json", failure_record)
                write_json(latest_paths["iteration_dir"] / "whisperx_full_song_alignment.json", full_song_alignment)
                song_spec_path = Path(str(audio_result.get("song_spec_path") or ""))
                if song_spec_path.exists():
                    song_spec = SongSpec.from_dict(json.loads(song_spec_path.read_text(encoding="utf-8")))
                    candidate_records = audio_result.get("full_song_candidate_attempts") or [
                        {"candidate": 0, "seed_offset": 0, "alignment": full_song_alignment, "media_output": str(final_candidate)}
                    ]
                    candidates = [
                        AudioCandidate(
                            candidate_id=f"candidate_{int(item.get('candidate', 0)):02d}",
                            backend=audio_voice_backend,
                            model_version="configured-runtime",
                            seed=seed + int(item.get("seed_offset", 0)),
                            media_path=str(item.get("media_output", "")),
                            alignment=dict(item.get("alignment") or {}),
                            technical_metrics={"passed": True},
                            passed=bool((item.get("alignment") or {}).get("passed")),
                        )
                        for item in candidate_records
                    ]
                    write_json(
                        latest_paths["iteration_dir"] / "audio_candidate_manifest.json",
                        candidate_manifest(song_spec, candidates),
                    )
                audio_result["one_take_status"] = "failed_review"
                audio_result["retained_full_song_media"] = audio_result.get(
                    "selected_full_song_media", str(final_candidate)
                )
                audio_result["scene_mix_fallback"] = "disabled"
                write_json(latest_paths["iteration_dir"] / "audio_postprocess_result.json", audio_result)

        if (
            mode in {"generate", "iterate"}
            and generate_audio
            and audio_aligner == "whisperx"
            and current_media_audio_mode in {"scene_lyrics_mix", "hybrid_voice_bed"}
            and audio_result is not None
            and final_candidate.exists()
            and whisperx_alignment_path.exists()
        ):
            best_alignment = analyze_whisperx_for_plan(plan, whisperx_alignment_path)
            best_rank = _alignment_rank(best_alignment)
            attempts_dir = latest_paths["iteration_dir"] / "audio_retry_attempts"
            attempts_dir.mkdir(parents=True, exist_ok=True)
            best_media = attempts_dir / "attempt_00.mp4"
            best_json = attempts_dir / "attempt_00_whisperx_alignment.json"
            shutil.copy2(final_candidate, best_media)
            shutil.copy2(whisperx_alignment_path, best_json)
            retry_records = [
                {
                    "attempt": 0,
                    "seed_offset": 0,
                    "alignment": best_alignment,
                    "media_output": str(best_media),
                }
            ]
            if _needs_scene_lyrics_audio_fallback(best_alignment):
                scene_candidate_count = _bounded_candidate_count(voice_candidate_count, music_candidate_count)
                retry_seed_offsets = _retry_seed_offsets(scene_candidate_count, [1601, 3203, 4801, 6407])
                for attempt, retry_seed_offset in enumerate(retry_seed_offsets, start=1):
                    shutil.rmtree(latest_paths["generated_dir"] / "audio", ignore_errors=True)
                    retry_audio_result = render_audio_candidate(current_media_audio_mode, seed_offset=retry_seed_offset)
                    if retry_audio_result.get("media_output"):
                        final_candidate = Path(str(retry_audio_result["media_output"]))
                    run_audio_alignment(final_candidate, f"whisperx_scene_lyrics_mix_retry_{attempt:02d}")
                    if not whisperx_alignment_path.exists():
                        continue
                    retry_alignment = analyze_whisperx_for_plan(plan, whisperx_alignment_path)
                    media_copy = attempts_dir / f"attempt_{attempt:02d}.mp4"
                    json_copy = attempts_dir / f"attempt_{attempt:02d}_whisperx_alignment.json"
                    shutil.copy2(final_candidate, media_copy)
                    shutil.copy2(whisperx_alignment_path, json_copy)
                    retry_records.append(
                        {
                            "attempt": attempt,
                            "seed_offset": retry_seed_offset,
                            "alignment": retry_alignment,
                            "media_output": str(media_copy),
                        }
                    )
                    retry_rank = _alignment_rank(retry_alignment)
                    if retry_rank < best_rank:
                        best_rank = retry_rank
                        best_alignment = retry_alignment
                        best_media = media_copy
                        best_json = json_copy
                        audio_result = retry_audio_result
                    if retry_alignment.get("passed"):
                        break
                if best_media.exists():
                    shutil.copy2(best_media, final_candidate)
                if best_json.exists():
                    shutil.copy2(best_json, whisperx_alignment_path)
                audio_result["scene_lyrics_retry_attempts"] = retry_records
                audio_result["selected_scene_lyrics_alignment"] = best_alignment
                audio_result["selected_scene_lyrics_media"] = str(best_media)
                audio_result["candidate_policy"] = {
                    "full_song": full_song_candidate_count,
                    "voice": voice_candidate_count,
                    "music": music_candidate_count,
                    "scene_retry_candidate_limit": scene_candidate_count,
                }
                write_json(latest_paths["iteration_dir"] / "audio_postprocess_result.json", audio_result)
        subtitle_result: dict[str, Any] | None = None
        if (
            mode in {"generate", "iterate"}
            and generate_audio
            and add_transcribed_subtitles
            and audio_aligner == "whisperx"
            and final_candidate.exists()
            and whisperx_alignment_path.exists()
        ):
            subtitle_result = add_whisperx_subtitles(
                video_file=final_candidate,
                whisperx_json=whisperx_alignment_path,
                subtitle_file=latest_paths["generated_dir"] / "subtitles.ass",
                output_file=latest_paths["generated_dir"] / "generated_subtitled_with_music.mp4",
                ffmpeg_bin=ffmpeg_bin,
                result_file=latest_paths["iteration_dir"] / "subtitle_postprocess_result.json",
            )
            final_candidate = Path(str(subtitle_result["video_output"]))
            if audio_result is not None:
                audio_result["subtitle_postprocess"] = subtitle_result
                write_json(latest_paths["iteration_dir"] / "audio_postprocess_result.json", audio_result)
        latest_final_candidate = final_candidate

        latest_report = evaluate_iteration(
            plan,
            latest_paths["generated_dir"],
            final_video=final_candidate,
            subtitle_path=latest_paths["generated_dir"] / "subtitles.ass",
            whisperx_alignment_path=whisperx_alignment_path if audio_aligner == "whisperx" else None,
            ffmpeg_bin=ffmpeg_bin,
            dry_run=mode == "dry_run",
            strict_lullaby_review=strict_lullaby_review,
            review_backend=review_backend,
            vlm_command=vlm_command,
            audio_review_command=audio_review_command,
            review_frames_dir=latest_paths["iteration_dir"] / "review_frames",
            enforce_lyric_scene_windows=current_media_audio_mode != "full_song",
        )
        write_json(latest_paths["iteration_dir"] / "evaluation_report.json", latest_report.to_dict())
        write_json(
            latest_paths["iteration_dir"] / "audio_analysis.json",
            {
                "audio_scores": latest_report.audio_scores,
                "aligner": audio_aligner,
                "whisperx_command": whisperx_command,
                "audio_postprocess": audio_result,
                "subtitle_postprocess": subtitle_result,
                "evaluated_video": str(final_candidate),
            },
        )
        if latest_report.whisperx_alignment and not (latest_paths["iteration_dir"] / "whisperx_alignment.json").exists():
            write_json(latest_paths["iteration_dir"] / "whisperx_alignment.json", latest_report.whisperx_alignment)
        for reviewer in latest_report.reviewer_reports:
            write_json(
                latest_paths["iteration_dir"] / "review_reports" / f"{reviewer.reviewer}.json",
                {
                    "reviewer": reviewer.reviewer,
                    "passed": reviewer.passed,
                    "scores": reviewer.scores,
                    "failure_reasons": reviewer.failure_reasons,
                    "evidence": reviewer.evidence,
                    "backend": review_backend,
                    "vlm_command": vlm_command,
                    "audio_review_command": audio_review_command,
                },
            )
        revision = build_revision_plan(plan, latest_report)
        write_json(latest_paths["iteration_dir"] / "revision_plan.json", revision.to_dict())
        if latest_report.passed or mode != "iterate":
            break
        if revision.target_scenes:
            first_target = min(revision.target_scenes)
            reuse_generated_video_dir = latest_paths["generated_dir"] if first_target > 1 else None
            visual_regeneration_start_scene = first_target if first_target > 1 else None
        elif revision.regenerate_audio:
            reuse_generated_video_dir = latest_paths["generated_dir"]
            visual_regeneration_start_scene = None
            if (
                current_media_audio_mode == "full_song"
                and _is_whisperx_timing_failure(latest_report)
            ):
                current_media_audio_mode = "hybrid_voice_bed" if media_audio_mode == "hybrid_voice_bed" else "scene_lyrics_mix"
        else:
            reuse_generated_video_dir = None
            visual_regeneration_start_scene = None
        plan = apply_revision_plan(plan, revision)

    final_dir = root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if latest_report and latest_report.passed:
        source = latest_final_candidate or resolve_storymem_video(latest_paths["generated_dir"])
        if source.exists():
            shutil.copy2(source, final_dir / "final_video.mp4")
    summary = {
        "mode": mode,
        "passed": latest_report.passed if latest_report else False,
        "iterations_run": int(latest_paths.get("iteration_dir", root / "iterations" / "001").name),
        "latest_iteration": str(latest_paths.get("iteration_dir", "")),
        "final_video": str(final_dir / "final_video.mp4"),
        "latest_candidate_video": str(latest_final_candidate) if latest_final_candidate else None,
        "planner_backend": planner_backend,
        "planner_command": planner_command,
        "review_backend": review_backend,
        "audio_aligner": audio_aligner,
        "transcribed_subtitles": add_transcribed_subtitles,
        "media_audio_mode": current_media_audio_mode,
    }
    write_json(root / "run_manifest.json", summary)
    return summary
