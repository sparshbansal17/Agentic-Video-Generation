import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template

import cv2
import json5


DEFAULT_STYLE_PROMPT = (
    "gentle adult lullaby, warm bedtime nursery rhyme, clear intelligible vocal, "
    "sing only the provided lyrics in order, exactly once per written line, "
    "do not substitute, omit, invent, or reorder words, "
    "music box and celesta melody, glockenspiel star twinkles, light harp arpeggios, "
    "soft strings pad, very gentle brushed percussion, slow 3/4 sway, major key, "
    "dreamy starry night, magical but calm, child-friendly, warm and comforting"
)
MAX_AUDIO_PROMPT_CHARS = 1600


@dataclass
class AudioConfig:
    story_script_path: str
    output_dir: str
    final_video: str | None = None
    audio_mode: str = "full_song"
    vocal_backend: str = "ace_step"
    backing_backend: str = "ace_step"
    audio_voice_style: str = "gentle adult lullaby"
    audio_output_suffix: str = "_with_music"
    audio_skip_on_error: bool = False
    audio_dry_run: bool = False
    lyrics_model: str | None = None
    lyrics_file: str | None = None
    ace_step_cmd: str | None = None
    vocal_cmd: str | None = None
    backing_cmd: str | None = None
    musicgen_cmd: str | None = None
    song_cmd: str | None = None
    voice_ref_audio: str | None = None
    voice_ref_text: str | None = None
    ffmpeg_bin: str | None = None
    seed: int = 0


def _story_video_path(output_dir: str, final_video: str | None) -> Path:
    if final_video:
        return Path(final_video)
    output_path = Path(output_dir)
    return output_path / f"{output_path.name}.mp4"


def _read_story(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json5.load(f)


def _video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps and frames:
        return float(frames / fps)
    raise RuntimeError(f"Could not determine video duration for {path}")


def _scene_summaries(story: dict) -> list[str]:
    summaries = []
    for scene in story.get("scenes", []):
        prompts = scene.get("video_prompts", [])
        if prompts:
            summaries.append(str(prompts[0]).strip())
    return summaries


def _repeated_word_warnings(lines: list[str]) -> list[str]:
    warnings = []
    token_re = re.compile(r"[A-Za-z0-9']+")
    for line_index, line in enumerate(lines, start=1):
        counts: dict[str, int] = {}
        originals: dict[str, str] = {}
        for match in token_re.finditer(line):
            token = match.group(0)
            key = token.lower()
            counts[key] = counts.get(key, 0) + 1
            originals.setdefault(key, token)
        for key, count in counts.items():
            if count > 1:
                warnings.append(
                    f'Line {line_index}: "{originals[key]}" appears {count} times; sing all {count} as separate audible words.'
                )
    return warnings


def _dedupe_sentences(text: str) -> str:
    seen = set()
    output = []
    for part in re.split(r"(?<=[.!?])\s+|;\s+", text):
        compact = " ".join(part.split()).strip(" .;")
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(compact)
    return ". ".join(output)


def _audio_style_prompt(story: dict, config: AudioConfig) -> str:
    scene_notes = []
    for i, summary in enumerate(_scene_summaries(story), start=1):
        compact = " ".join(summary.split())
        scene_notes.append(f"scene {i}: {compact[:95]}")
    metadata = story.get("agentic_metadata", {})
    plan_music_prompt = _dedupe_sentences(str(metadata.get("music_prompt", "")).strip())
    style = (
        f"STYLE/TONE: {DEFAULT_STYLE_PROMPT}, voice feel: {config.audio_voice_style}, "
        f"planner audio direction: {plan_music_prompt}."
    )
    scene_prefix = (
        "COMPACT SCENE CONTEXT: arrangement follows the full story arc with a soft opening, curious bedtime moment, "
        "gentle lift over the sleeping world, sparkling highlight, and peaceful goodnight cadence, "
    )
    scene_context = scene_prefix + ", ".join(scene_notes)
    remaining = MAX_AUDIO_PROMPT_CHARS - len(style) - 8
    if remaining > 80:
        scene_context = scene_context[:remaining]
    else:
        scene_context = ""
    # Lyrics and timing are separate structured inputs. Duplicating them in the
    # style caption reduces adherence and lets backend caption rewriters alter words.
    prompt = style + ("\n" + scene_context if scene_context else "")
    return prompt


def _fallback_lyrics(story: dict) -> str:
    scene_lyrics = [
        str(scene.get("lyric_line", "")).strip()
        for scene in story.get("scenes", [])
        if str(scene.get("lyric_line", "")).strip()
    ]
    if scene_lyrics:
        return "\n".join(scene_lyrics)

    title = story.get("story_name", "Little Star")
    if "star" in title.lower():
        return "\n".join([
            "Twinkle, twinkle, little star,",
            "How I wonder what you are!",
            "Up above the world so high,",
            "Like a diamond in the sky.",
            "Twinkle, twinkle, little star,",
            "How I wonder what you are!",
        ])
    return "\n".join([
        "Softly now the story gleams,",
        "Guiding little bedtime dreams.",
        "Warm and bright, the night is new,",
        "Gentle wishes shine for you.",
    ])


def _local_llm_lyrics(story: dict, duration: float, model_name: str | None) -> str | None:
    if not model_name:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception:
        return None

    scenes = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(_scene_summaries(story)))
    prompt = (
        "Write a short child-friendly nursery rhyme for a bedtime video. "
        f"The song must fit about {duration:.1f} seconds, use simple words, "
        "and be 8 to 10 very short lines mapped to the scene order. Return only the lyrics.\n\n"
        f"Story: {story.get('story_overview', '')}\nScenes:\n{scenes}\n"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **tokens,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        text = tokenizer.decode(output[0][tokens["input_ids"].shape[-1]:], skip_special_tokens=True)
        lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
        return "\n".join(lines[:8]) or None
    except Exception:
        return None


def _lyrics_override(path: str | None) -> str | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    return text or None


def _write_metadata(config: AudioConfig, story: dict, duration: float, work_dir: Path, lyrics: str) -> Path:
    style_prompt = _audio_style_prompt(story, config)
    metadata = {
        "story_name": story.get("story_name"),
        "story_overview": story.get("story_overview"),
        "duration_seconds": duration,
        "audio_mode": config.audio_mode,
        "vocal_backend": config.vocal_backend,
        "backing_backend": config.backing_backend,
        "voice_style": config.audio_voice_style,
        "style_prompt": style_prompt,
        "style_prompt_limit_chars": MAX_AUDIO_PROMPT_CHARS,
        "seed": config.seed,
        "scene_summaries": _scene_summaries(story),
        "lyrics": lyrics,
    }
    prompt_path = work_dir / "audio_prompt.json"
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return prompt_path


def _write_audio_prompt(path: Path, style_prompt: str, story: dict, config: AudioConfig, duration: float, lyrics: str) -> Path:
    metadata = {
        "story_name": story.get("story_name"),
        "duration_seconds": duration,
        "audio_mode": config.audio_mode,
        "vocal_backend": config.vocal_backend,
        "backing_backend": config.backing_backend,
        "voice_style": config.audio_voice_style,
        "style_prompt": style_prompt,
        "style_prompt_limit_chars": MAX_AUDIO_PROMPT_CHARS,
        "seed": config.seed,
        "lyrics": lyrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def _render_command(template: str, values: dict[str, str]) -> list[str]:
    quoted_values = {key: shlex.quote(value) for key, value in values.items()}
    rendered = Template(template).safe_substitute(quoted_values)
    return shlex.split(rendered)


def _run_template(template: str | None, values: dict[str, str], label: str) -> None:
    if not template:
        raise RuntimeError(
            f"Missing {label} command template. Set the matching CLI option or environment variable."
        )
    cmd = _render_command(template, values)
    subprocess.run(cmd, check=True)


def _require_template(template: str | None, label: str) -> str:
    if not template:
        raise RuntimeError(
            f"Missing {label} command template. Provide the matching CLI option or environment variable."
        )
    return template


def _requires_voice_reference(config: AudioConfig) -> bool:
    return config.vocal_backend in {"f5_tts", "cosyvoice"}


def _validate_voice_reference(config: AudioConfig) -> None:
    if not _requires_voice_reference(config):
        return
    if not config.voice_ref_audio:
        raise RuntimeError(
            f"{config.vocal_backend} requires --voice-ref-audio or VOICE_REF_AUDIO for exact lyric voice generation."
        )
    if not Path(config.voice_ref_audio).exists():
        raise RuntimeError(f"voice reference audio does not exist: {config.voice_ref_audio}")
    if not (config.voice_ref_text or "").strip():
        raise RuntimeError(
            f"{config.vocal_backend} requires --voice-ref-text or VOICE_REF_TEXT matching the reference audio."
        )


def _vocal_template(config: AudioConfig) -> str:
    if config.vocal_cmd:
        return config.vocal_cmd
    env_by_backend = {
        "f5_tts": "F5_TTS_CMD",
        "cosyvoice": "COSYVOICE_CMD",
        "ace_step": "ACE_STEP_CMD",
        "ace_step_full_song": "ACE_STEP_CMD",
    }
    env_name = env_by_backend.get(config.vocal_backend)
    if env_name and os.getenv(env_name):
        return str(os.getenv(env_name))
    if config.vocal_backend in {"ace_step", "ace_step_full_song"}:
        return _require_template(config.ace_step_cmd or os.getenv("ACE_STEP_CMD"), "ACE-Step vocal")
    return _require_template(None, f"{config.vocal_backend} vocal")


def _backing_template(config: AudioConfig) -> str:
    if config.backing_cmd:
        return config.backing_cmd
    if config.backing_backend == "musicgen":
        return _require_template(config.musicgen_cmd or os.getenv("MUSICGEN_CMD"), "MusicGen backing")
    if config.backing_backend == "stable_audio":
        return _require_template(os.getenv("STABLE_AUDIO_CMD"), "Stable Audio backing")
    return _require_template(config.ace_step_cmd or os.getenv("ACE_STEP_CMD"), "ACE-Step backing")


def _song_template(config: AudioConfig) -> str:
    if config.song_cmd:
        return config.song_cmd
    if config.vocal_backend == "yue_full_song":
        return _require_template(os.getenv("YUE_CMD"), "YuE full-song")
    return _require_template(config.ace_step_cmd or os.getenv("ACE_STEP_CMD"), "ACE-Step full-song")


def _find_ffmpeg(config: AudioConfig) -> str:
    candidates = [config.ffmpeg_bin, os.getenv("FFMPEG_BIN"), shutil.which("ffmpeg")]
    for candidate in candidates:
        if candidate:
            resolved = shutil.which(candidate) or candidate
            if Path(resolved).exists():
                return str(resolved)
    raise RuntimeError("ffmpeg was not found. Set FFMPEG_BIN to a usable ffmpeg binary.")


def _scene_clip_paths(output_dir: Path) -> list[Path]:
    scene_pattern = re.compile(r"^\d{2}_\d{2}\.mp4$")
    return sorted(path for path in output_dir.glob("*.mp4") if scene_pattern.match(path.name))


def _scene_timing(output_dir: Path) -> tuple[list[float], list[float]]:
    scene_paths = _scene_clip_paths(output_dir)
    if not scene_paths:
        return [], []
    durations = [_video_duration(path) for path in scene_paths]
    starts = []
    elapsed = 0.0
    for duration in durations:
        starts.append(elapsed)
        elapsed += duration
    return starts, durations


def _lyric_lines_for_scenes(story: dict, lyrics: str, max_lines: int = 4) -> list[str]:
    scene_lines = [
        str(scene.get("lyric_line", "")).strip()
        for scene in story.get("scenes", [])
        if str(scene.get("lyric_line", "")).strip()
    ]
    if scene_lines:
        return scene_lines[:max_lines]
    return [line.strip() for line in lyrics.splitlines() if line.strip()][:max_lines]


def _normalize_full_song(ffmpeg: str, source: Path, output: Path, duration: float) -> None:
    audio_filter = (
        "aresample=48000,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "loudnorm=I=-15:TP=-1.5:LRA=11,"
        "afade=t=in:st=0:d=0.7,"
        f"afade=t=out:st={max(duration - 1.2, 0):.3f}:d=1.2,"
        f"apad=pad_dur={duration:.3f},"
        f"atrim=0:{duration:.3f}"
    )
    subprocess.run([
        ffmpeg, "-y", "-i", str(source),
        "-af", audio_filter,
        "-ar", "48000", "-ac", "2",
        str(output),
    ], check=True)


def _mix_stems(ffmpeg: str, vocals: Path, backing: Path, output: Path, duration: float) -> None:
    fade_out_start = max(duration - 1.2, 0)
    filter_complex = (
        "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "highpass=f=90,loudnorm=I=-16:TP=-1.5:LRA=9,asplit=2[voc_sc][voc_mix];"
        "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "loudnorm=I=-20:TP=-2:LRA=11[back];"
        "[back][voc_sc]sidechaincompress=threshold=0.04:ratio=6:attack=20:release=250[ducked];"
        "[ducked][voc_mix]amix=inputs=2:weights='0.45 1.0':duration=longest:dropout_transition=0,"
        "loudnorm=I=-15:TP=-1.5:LRA=10,"
        "afade=t=in:st=0:d=0.7,"
        f"afade=t=out:st={fade_out_start:.3f}:d=1.2,"
        f"apad=pad_dur={duration:.3f},atrim=0:{duration:.3f}[a]"
    )
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(vocals),
        "-stream_loop", "-1", "-i", str(backing),
        "-filter_complex", filter_complex,
        "-map", "[a]",
        "-ar", "48000", "-ac", "2",
        str(output),
    ], check=True)


def _estimated_line_vocal_duration(line: str, scene_duration: float) -> float:
    words = re.findall(r"[A-Za-z0-9']+", line)
    estimated = 0.58 * max(len(words), 1) + 0.9
    return max(1.8, min(scene_duration - 0.15, estimated))


def _prepare_scene_vocal(
    ffmpeg: str,
    source: Path,
    output: Path,
    scene_duration: float,
    *,
    active_duration: float | None = None,
) -> None:
    trim_duration = max(min(active_duration or scene_duration - 0.05, scene_duration - 0.05), 0.1)
    fade_duration = min(0.08, max(scene_duration - trim_duration, 0.0))
    fade_start = max(trim_duration - fade_duration, 0)
    audio_filter = (
        "aresample=48000,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "highpass=f=100,"
        "loudnorm=I=-13:TP=-1.5:LRA=8,"
        f"atrim=0:{trim_duration:.3f},"
        f"afade=t=out:st={fade_start:.3f}:d={fade_duration:.3f},"
        f"apad=pad_dur={scene_duration:.3f},"
        f"atrim=0:{scene_duration:.3f}"
    )
    subprocess.run([
        ffmpeg, "-y", "-i", str(source),
        "-af", audio_filter,
        "-ar", "48000", "-ac", "2",
        str(output),
    ], check=True)


def _mix_scene_lyrics(
    ffmpeg: str,
    backing: Path,
    scene_vocals: list[Path],
    scene_starts: list[float],
    output: Path,
    duration: float,
) -> None:
    fade_out_start = max(duration - 0.35, 0)
    inputs = [ffmpeg, "-y", "-stream_loop", "-1", "-i", str(backing)]
    for vocal in scene_vocals:
        inputs.extend(["-i", str(vocal)])

    filters = [
        "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "loudnorm=I=-28:TP=-3:LRA=11[back]"
    ]
    vocal_labels = []
    for index, start in enumerate(scene_starts[:len(scene_vocals)], start=1):
        delay_ms = max(int(round(start * 1000)), 0)
        label = f"v{index}"
        filters.append(
            f"[{index}:a]adelay={delay_ms}:all=1,"
            f"apad=pad_dur={duration:.3f},atrim=0:{duration:.3f}[{label}]"
        )
        vocal_labels.append(f"[{label}]")

    filters.append(
        "".join(vocal_labels)
        + f"amix=inputs={len(vocal_labels)}:duration=longest:normalize=0,"
        + f"atrim=0:{duration:.3f},asplit=2[vocals_sc][vocals_mix]"
    )
    filters.append(
        "[back][vocals_sc]sidechaincompress=threshold=0.02:ratio=12:attack=5:release=500[ducked]"
    )
    filters.append(
        "[ducked][vocals_mix]amix=inputs=2:weights='0.18 1.7':duration=longest:dropout_transition=0,"
        "loudnorm=I=-14:TP=-1.5:LRA=9,"
        "afade=t=in:st=0:d=0.25,"
        f"afade=t=out:st={fade_out_start:.3f}:d=0.35,"
        f"apad=pad_dur={duration:.3f},atrim=0:{duration:.3f}[a]"
    )
    subprocess.run([
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[a]",
        "-ar", "48000", "-ac", "2",
        str(output),
    ], check=True)


def _generate_scene_lyrics_mix(
    config: AudioConfig,
    story: dict,
    ffmpeg: str,
    work_dir: Path,
    values: dict[str, str],
    lyrics: str,
    mixed_song: Path,
    duration: float,
    output_dir: Path,
) -> None:
    scene_starts, scene_durations = _scene_timing(output_dir)
    lyric_lines = _lyric_lines_for_scenes(story, lyrics, max_lines=max(len(scene_starts), 1))
    if not lyric_lines:
        raise RuntimeError("scene_lyrics_mix requires at least one lyric line.")
    if len(scene_starts) < len(lyric_lines) or len(scene_durations) < len(lyric_lines):
        raise RuntimeError(
            "scene_lyrics_mix requires one scene clip named like 01_01.mp4 for each lyric line."
        )

    scene_work_dir = work_dir / "scene_lyrics_mix"
    scene_work_dir.mkdir(parents=True, exist_ok=True)
    timing_metadata = {
        "duration_seconds": duration,
        "lines": [],
    }

    backing_prompt = (
        "instrumental only, no singing, no spoken words, no humming, karaoke backing track for the planned nursery rhyme, "
        "music box and celesta melody, glockenspiel star twinkles, light harp, soft strings, gentle 3/4 lullaby, "
        "child-friendly Cocomelon style, warm and bright, keep steady timing for separate vocal lines"
    )
    backing_lyrics = "[instrumental]"
    backing_lyrics_path = scene_work_dir / "backing_lyrics.txt"
    backing_lyrics_path.write_text(backing_lyrics + "\n", encoding="utf-8")
    backing_prompt_path = _write_audio_prompt(
        scene_work_dir / "backing_prompt.json",
        backing_prompt,
        story,
        config,
        duration,
        backing_lyrics,
    )
    backing = scene_work_dir / "backing.wav"

    _validate_voice_reference(config)
    backing_template = _backing_template(config)
    if not backing.exists():
        _run_template(
            backing_template,
            dict(
                values,
                lyrics_file=str(backing_lyrics_path),
                prompt_file=str(backing_prompt_path),
                music_prompt=backing_prompt,
                output_file=str(backing),
                duration=f"{duration:.3f}",
                seed=str(config.seed + 1000),
                mode="backing",
            ),
            "backing-track generation",
        )

    vocal_template = _vocal_template(config)
    prepared_vocals = []
    for index, line in enumerate(lyric_lines, start=1):
        scene_duration = scene_durations[index - 1]
        line_lyrics_path = scene_work_dir / f"line_{index:02d}_lyrics.txt"
        line_lyrics_path.write_text(line + "\n", encoding="utf-8")
        line_prompt = (
            f"EXACT NUMBERED LYRIC: {index}. {line}\n"
            f"TIMING WINDOW: line {index} must fit inside {scene_starts[index - 1]:.3f}-{scene_starts[index - 1] + scene_duration:.3f}s after mixing.\n"
            + "\n".join(_repeated_word_warnings([line]))
            + "\n"
            "a cappella sung nursery-rhyme vocal only, no instrumental backing, no intro, "
            "the first word starts immediately at 0.0 seconds, clear toddler singalong pronunciation, "
            "gentle adult lead vocal with soft childlike brightness, child-friendly Cocomelon style, "
            "sing the line once, stop after the final word, do not add extra words, "
            f"sing exactly this one lyric line and nothing else: {line}"
        )
        vocal_duration = _estimated_line_vocal_duration(line, scene_duration)
        line_prompt_path = _write_audio_prompt(
            scene_work_dir / f"line_{index:02d}_prompt.json",
            line_prompt,
            story,
            config,
            vocal_duration,
            line,
        )
        raw_vocal = scene_work_dir / f"line_{index:02d}_raw.wav"
        prepared_vocal = scene_work_dir / f"line_{index:02d}_prepared.wav"
        if not raw_vocal.exists():
            _run_template(
                vocal_template,
                dict(
                    values,
                    lyrics_file=str(line_lyrics_path),
                    prompt_file=str(line_prompt_path),
                    text=line,
                    gen_text=line,
                    voice_style=config.audio_voice_style,
                    ref_audio=str(config.voice_ref_audio or ""),
                    ref_text=str(config.voice_ref_text or ""),
                    output_file=str(raw_vocal),
                    duration=f"{vocal_duration:.3f}",
                    seed=str(config.seed + index),
                    mode="vocals",
                ),
                f"scene {index} vocal generation",
            )
        _prepare_scene_vocal(ffmpeg, raw_vocal, prepared_vocal, scene_duration, active_duration=vocal_duration)
        prepared_vocals.append(prepared_vocal)
        timing_metadata["lines"].append(
            {
                "line_index": index,
                "text": line,
                "scene_start_seconds": scene_starts[index - 1],
                "scene_end_seconds": scene_starts[index - 1] + scene_duration,
                "scene_duration_seconds": scene_duration,
                "estimated_vocal_duration_seconds": vocal_duration,
                "raw_vocal": str(raw_vocal),
                "prepared_vocal": str(prepared_vocal),
                "repeated_word_warnings": _repeated_word_warnings([line]),
            }
        )

    with open(scene_work_dir / "timing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(timing_metadata, f, indent=2)

    _mix_scene_lyrics(
        ffmpeg,
        backing,
        prepared_vocals,
        scene_starts,
        mixed_song,
        duration,
    )


def _mux_video(ffmpeg: str, video: Path, audio: Path, output: Path) -> None:
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(video),
        "-i", str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(output),
    ], check=True)


def generate_audio_for_story(config: AudioConfig) -> Path | None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = _story_video_path(config.output_dir, config.final_video)
    if not video_path.exists():
        raise FileNotFoundError(f"Missing final StoryMem video: {video_path}")

    story = _read_story(config.story_script_path)
    duration = _video_duration(video_path)
    work_dir = output_dir / "audio"
    work_dir.mkdir(parents=True, exist_ok=True)

    lyrics = (
        _lyrics_override(config.lyrics_file)
        or _local_llm_lyrics(story, duration, config.lyrics_model)
        or _fallback_lyrics(story)
    )
    lyrics_path = work_dir / "lyrics.txt"
    lyrics_path.write_text(lyrics + "\n", encoding="utf-8")
    prompt_path = _write_metadata(config, story, duration, work_dir, lyrics)

    if config.audio_dry_run:
        return prompt_path

    ffmpeg = _find_ffmpeg(config)
    values = {
        "lyrics_file": str(lyrics_path),
        "prompt_file": str(prompt_path),
        "song_spec_file": str(work_dir / "song_spec.json"),
        "style_prompt": _audio_style_prompt(story, config),
        "duration": f"{duration:.3f}",
        "seed": str(config.seed),
    }

    render_mode = "scene_lyrics_mix" if config.audio_mode == "hybrid_voice_bed" else config.audio_mode
    mixed_song = work_dir / "mixed_song.wav"
    if render_mode == "full_song":
        source_song = work_dir / "song.wav"
        values["output_file"] = str(source_song)
        _run_template(_song_template(config), values, config.vocal_backend)
        _normalize_full_song(ffmpeg, source_song, mixed_song, duration)
    elif render_mode == "scene_lyrics_mix":
        _generate_scene_lyrics_mix(config, story, ffmpeg, work_dir, values, lyrics, mixed_song, duration, output_dir)
    else:
        vocals = work_dir / "vocals.wav"
        backing = work_dir / "backing.wav"
        _validate_voice_reference(config)
        vocal_template = _vocal_template(config)
        backing_template = _backing_template(config)
        vocal_values = dict(
            values,
            output_file=str(vocals),
            mode="vocals",
            text=lyrics,
            gen_text=lyrics,
            voice_style=config.audio_voice_style,
            ref_audio=str(config.voice_ref_audio or ""),
            ref_text=str(config.voice_ref_text or ""),
        )
        backing_values = dict(
            values,
            output_file=str(backing),
            mode="backing",
            music_prompt=_audio_style_prompt(story, config),
        )
        _run_template(vocal_template, vocal_values, "vocal generation")
        _run_template(backing_template, backing_values, "backing-track generation")
        _mix_stems(ffmpeg, vocals, backing, mixed_song, duration)

    final_output = video_path.with_name(f"{video_path.stem}{config.audio_output_suffix}{video_path.suffix}")
    _mux_video(ffmpeg, video_path, mixed_song, final_output)
    return final_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and mix StoryMem nursery-rhyme audio.")
    parser.add_argument("--story_script_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--final_video", default=None)
    parser.add_argument("--audio_mode", choices=["full_song", "separate_stems", "scene_lyrics_mix", "hybrid_voice_bed"], default="full_song")
    parser.add_argument("--vocal_backend", default="ace_step")
    parser.add_argument("--backing_backend", choices=["ace_step", "musicgen", "stable_audio"], default="ace_step")
    parser.add_argument("--audio_voice_style", default="gentle adult lullaby")
    parser.add_argument("--audio_output_suffix", default="_with_music")
    parser.add_argument("--audio_skip_on_error", action="store_true")
    parser.add_argument("--audio_dry_run", action="store_true")
    parser.add_argument("--lyrics_model", default=None)
    parser.add_argument("--lyrics_file", default=None)
    parser.add_argument("--ace_step_cmd", default=None)
    parser.add_argument("--vocal_cmd", default=None)
    parser.add_argument("--backing_cmd", default=None)
    parser.add_argument("--musicgen_cmd", default=None)
    parser.add_argument("--song_cmd", default=None)
    parser.add_argument("--voice_ref_audio", default=os.getenv("VOICE_REF_AUDIO"))
    parser.add_argument("--voice_ref_text", default=os.getenv("VOICE_REF_TEXT"))
    parser.add_argument("--ffmpeg_bin", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = AudioConfig(**vars(args))
    try:
        output = generate_audio_for_story(config)
    except Exception as exc:
        if config.audio_skip_on_error:
            print(f"Skipping StoryMem audio stage after error: {exc}")
            return
        raise
    print(output)


if __name__ == "__main__":
    main()
