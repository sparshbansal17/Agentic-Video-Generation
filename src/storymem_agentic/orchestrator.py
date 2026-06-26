from __future__ import annotations

import json
import shutil
import shlex
import subprocess
from pathlib import Path
from string import Template
from typing import Any

from .audio_director import build_audio_plan
from .agents import CommandAgentBackend
from .backends import write_backend_manifest
from .feedback import apply_revision_plan, build_revision_plan
from .media_evaluator import evaluate_iteration
from .mixer import build_mix_manifest, write_mix_manifest
from .planner import PromptPlannerAgent
from .runner import run_audio_postprocess
from .schemas import EvaluationReport, NurseryRhymeInput, ProductionPlan, RevisionPlan, SceneHint
from .story_writer import rhyme_text_from_plan, story_from_plan


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
    )
    input_data.validate()
    return input_data


def write_iteration_artifacts(root: Path, iteration: int, plan: ProductionPlan, *, dry_run: bool) -> dict[str, Path]:
    iteration_dir = root / "iterations" / f"{iteration:03d}"
    generated_dir = iteration_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    story = story_from_plan(plan)
    first_shot_story = {
        **story,
        "scenes": [
            {
                **story["scenes"][0],
                "video_prompts": [story["scenes"][0]["video_prompts"][0]],
                "first_frame_prompt": [story["scenes"][0]["first_frame_prompt"][0]],
                "cut": [story["scenes"][0]["cut"][0]],
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
    t5_cpu: bool = False,
    offload_model: bool = False,
    sample_steps: int | None = None,
    frame_num: int | None = None,
    keyframe_mode: str = "hps",
) -> list[list[str]]:
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
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        str(nproc_per_node),
        "--lora_rank",
        "128",
        "--keyframe_mode",
        keyframe_mode,
        ]
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
    t5_cpu: bool = False,
    offload_model: bool = False,
    sample_steps: int | None = None,
    frame_num: int | None = None,
    keyframe_mode: str = "hps",
) -> list[str]:
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
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        str(nproc_per_node),
        "--lora_rank",
        "128",
        "--keyframe_mode",
        keyframe_mode,
        "--mi2v",
    ]
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
        str(reason).startswith("line_") or str(reason) == "wer_above_threshold"
        for reason in report.whisperx_alignment.get("failure_reasons", [])
    )


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
    planner_backend: str = "mock",
    planner_command: str | None = None,
    review_backend: str = "mock",
    vlm_command: str | None = None,
    whisperx_command: str | None = None,
    audio_aligner: str = "whisperx",
    strict_lullaby_review: bool = True,
    generate_audio: bool = True,
    media_audio_mode: str = "full_song",
    audio_output_suffix: str = "_with_music",
    audio_voice_style: str = "warm clear toddler nursery singalong vocal, exact lyrics, gentle adult lead with soft childlike brightness, music box, celesta, glockenspiel star twinkles, soft strings, light harp, slow 3/4 sway, vocals forward",
    ace_step_cmd: str | None = None,
    vocal_cmd: str | None = None,
    backing_cmd: str | None = None,
    musicgen_cmd: str | None = None,
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
    )
    planner = PromptPlannerAgent(
        CommandAgentBackend(planner_command) if planner_backend == "command" and planner_command else None
    )
    plan = planner.plan(nursery_input)
    write_json(root / "nursery_rhyme_input.json", nursery_input.to_dict())
    planner_output = {
        "backend": planner_backend,
        "command": planner_command,
        "used_fallback": planner.used_fallback,
        "error": planner.last_error,
        "fallback_policy": (
            "continued_with_deterministic_local_plan"
            if planner_backend == "command" and planner.used_fallback
            else None
        ),
        "prompt": planner.last_prompt,
        "schema": planner.last_schema,
        "context": planner.last_context,
        "response": planner.last_response,
    }
    write_json(
        root / "planner_agent_output.json",
        planner_output,
    )

    iteration_count = 1 if mode in {"dry_run", "generate"} else max_iterations
    latest_report: EvaluationReport | None = None
    latest_paths: dict[str, Path] = {}
    latest_final_candidate: Path | None = None
    reuse_generated_video_dir: Path | None = None
    visual_regeneration_start_scene: int | None = None
    current_media_audio_mode = media_audio_mode

    for iteration in range(1, iteration_count + 1):
        cached_paths = existing_iteration_paths(root, iteration)
        cached_report_path = cached_paths["iteration_dir"] / "evaluation_report.json"
        cached_revision_path = cached_paths["iteration_dir"] / "revision_plan.json"
        if mode == "iterate" and cached_report_path.exists() and cached_revision_path.exists():
            latest_paths = cached_paths
            if cached_paths["production_plan"].exists():
                plan = ProductionPlan.from_dict(json.loads(cached_paths["production_plan"].read_text(encoding="utf-8")))
            latest_report = EvaluationReport.from_dict(json.loads(cached_report_path.read_text(encoding="utf-8")))
            cached_revision = RevisionPlan(**json.loads(cached_revision_path.read_text(encoding="utf-8")))
            latest_final_candidate = (
                cached_paths["generated_dir"] / "generated_subtitled_with_music.mp4"
                if (cached_paths["generated_dir"] / "generated_subtitled_with_music.mp4").exists()
                else resolve_storymem_video(cached_paths["generated_dir"])
            )
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
                    current_media_audio_mode = "scene_lyrics_mix"
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
                if visual_regeneration_start_scene is not None and storymem_dir and t2v_model_path and i2v_model_path and lora_weight_path:
                    targeted_story = write_targeted_story_json(
                        latest_paths["story"],
                        latest_paths["iteration_dir"] / f"story_from_scene_{visual_regeneration_start_scene:02d}.json",
                        visual_regeneration_start_scene,
                    )
                    commands = [
                        build_storymem_continuation_command(
                            story_json=targeted_story,
                            output_dir=latest_paths["generated_dir"],
                            storymem_dir=storymem_dir,
                            t2v_model_path=t2v_model_path,
                            i2v_model_path=i2v_model_path,
                            lora_weight_path=lora_weight_path,
                            nproc_per_node=nproc_per_node,
                            ffmpeg_bin=ffmpeg_bin,
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
                        run_storymem_commands(commands, cwd=storymem_dir)
                else:
                    write_json(
                        latest_paths["iteration_dir"] / "storymem_commands.json",
                        {
                            "commands": [],
                            "notes": f"reused accepted video artifacts from {reuse_generated_video_dir} for audio-only revision",
                        },
                    )
            elif storymem_dir and t2v_model_path and i2v_model_path and lora_weight_path:
                commands = build_storymem_commands(
                    story_json=latest_paths["story"],
                    first_shot_story_json=latest_paths["story_t2v_first_shot"],
                    output_dir=latest_paths["generated_dir"],
                    storymem_dir=storymem_dir,
                    t2v_model_path=t2v_model_path,
                    i2v_model_path=i2v_model_path,
                    lora_weight_path=lora_weight_path,
                    nproc_per_node=nproc_per_node,
                    ffmpeg_bin=ffmpeg_bin,
                    t5_cpu=storymem_t5_cpu,
                    offload_model=storymem_offload_model,
                    sample_steps=storymem_sample_steps,
                    frame_num=storymem_frame_num,
                    keyframe_mode=storymem_keyframe_mode,
                )
                write_json(latest_paths["iteration_dir"] / "storymem_commands.json", {"commands": commands})
                if execute_video:
                    run_storymem_commands(commands, cwd=storymem_dir)
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
            audio_voice_backend = "ace_step_full_song" if plan.audio_mode == "full_song" else "f5_tts"
            audio_music_backend = "ace_step_full_song" if plan.audio_mode == "full_song" else "musicgen"
            effective_media_audio_mode = media_audio_mode
            if plan.audio_mode == "voice_bed" and media_audio_mode == "full_song":
                effective_media_audio_mode = "separate_stems"
            audio_result = run_audio_postprocess(
                rhyme_file=latest_paths["iteration_dir"] / "lyrics.txt",
                story_json=latest_paths["story"],
                output_dir=latest_paths["generated_dir"],
                final_video=video_candidate,
                mode=plan.audio_mode,
                voice_backend=audio_voice_backend,
                music_backend=audio_music_backend,
                media_audio_mode=current_media_audio_mode
                if plan.audio_mode == "full_song"
                else effective_media_audio_mode,
                audio_output_suffix=audio_output_suffix,
                audio_voice_style=audio_voice_style,
                ace_step_cmd=ace_step_cmd,
                vocal_cmd=vocal_cmd,
                backing_cmd=backing_cmd,
                musicgen_cmd=musicgen_cmd,
                ffmpeg_bin=ffmpeg_bin,
                seed=seed + ((iteration - 1) * 9973),
                dry_run=False,
            )
            if audio_result.get("media_output"):
                final_candidate = Path(str(audio_result["media_output"]))
            write_json(latest_paths["iteration_dir"] / "audio_postprocess_result.json", audio_result)

        whisperx_alignment_path = latest_paths["iteration_dir"] / "whisperx_alignment.json"
        if (
            mode in {"generate", "iterate"}
            and generate_audio
            and audio_aligner == "whisperx"
            and final_candidate.exists()
            and not whisperx_alignment_path.exists()
        ):
            run_whisperx_command(
                command_template=whisperx_command,
                audio_file=final_candidate,
                output_dir=latest_paths["iteration_dir"] / "whisperx",
                output_file=whisperx_alignment_path,
            )
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
            review_frames_dir=latest_paths["iteration_dir"] / "review_frames",
        )
        write_json(latest_paths["iteration_dir"] / "evaluation_report.json", latest_report.to_dict())
        write_json(
            latest_paths["iteration_dir"] / "audio_analysis.json",
            {
                "audio_scores": latest_report.audio_scores,
                "aligner": audio_aligner,
                "whisperx_command": whisperx_command,
                "audio_postprocess": audio_result,
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
            if current_media_audio_mode == "full_song" and _is_whisperx_timing_failure(latest_report):
                current_media_audio_mode = "scene_lyrics_mix"
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
        "media_audio_mode": current_media_audio_mode,
    }
    write_json(root / "run_manifest.json", summary)
    return summary
