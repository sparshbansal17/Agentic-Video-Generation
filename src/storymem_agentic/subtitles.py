from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubtitleCue:
    start_seconds: float
    end_seconds: float
    text: str


def _timed_whisperx_words(data: dict[str, Any]) -> list[dict[str, Any]]:
    words = data.get("word_segments") or []
    if not words:
        words = [word for segment in data.get("segments", []) for word in segment.get("words", [])]
    return [
        word
        for word in words
        if str(word.get("word", "")).strip()
        and word.get("start") is not None
        and word.get("end") is not None
    ]


def _starts_phrase(word: str) -> bool:
    stripped = re.sub(r"^[^A-Za-z]+", "", word)
    return bool(stripped and stripped[0].isupper())


def whisperx_cues(
    alignment: dict[str, Any],
    *,
    pause_seconds: float = 0.9,
    max_words: int = 7,
    max_chars: int = 42,
    max_duration_seconds: float = 6.0,
) -> list[SubtitleCue]:
    """Create readable cues exclusively from WhisperX-observed, timed words."""
    words = _timed_whisperx_words(alignment)
    if not words:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        text = str(word["word"]).strip()
        if current:
            previous_end = float(current[-1]["end"])
            gap = float(word["start"]) - previous_end
            proposed_text = " ".join(
                [*(str(item["word"]).strip() for item in current), text]
            )
            duration = float(word["end"]) - float(current[0]["start"])
            phrase_boundary = len(current) >= 2 and _starts_phrase(text)
            if (
                gap >= pause_seconds
                or phrase_boundary
                or len(current) >= max_words
                or len(proposed_text) > max_chars
                or duration > max_duration_seconds
            ):
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    readable_groups: list[list[dict[str, Any]]] = []
    group_index = 0
    while group_index < len(groups):
        group = groups[group_index]
        if len(group) == 1 and group_index + 1 < len(groups):
            following = groups[group_index + 1]
            combined = [*group, *following]
            combined_text = " ".join(str(item["word"]).strip() for item in combined)
            combined_duration = float(combined[-1]["end"]) - float(combined[0]["start"])
            if (
                len(combined) <= max_words
                and len(combined_text) <= max_chars
                and combined_duration <= max_duration_seconds
            ):
                readable_groups.append(combined)
                group_index += 2
                continue
        readable_groups.append(group)
        group_index += 1
    groups = readable_groups

    cues: list[SubtitleCue] = []
    for index, group in enumerate(groups):
        start = float(group[0]["start"])
        end = float(group[-1]["end"])
        if index + 1 < len(groups):
            next_start = float(groups[index + 1][0]["start"])
            end = min(end + 0.25, max(end, next_start - 0.05))
        else:
            end += 0.25
        cues.append(
            SubtitleCue(
                start_seconds=max(0.0, start),
                end_seconds=max(start + 0.05, end),
                text=" ".join(str(item["word"]).strip() for item in group),
            )
        )
    return cues


def load_whisperx_cues(path: str | Path) -> tuple[list[SubtitleCue], str | None]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return whisperx_cues(data), data.get("language")


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def write_ass_subtitles(
    path: str | Path,
    cues: list[SubtitleCue],
    *,
    play_res_x: int = 832,
    play_res_y: int = 480,
) -> Path:
    if not cues:
        raise ValueError("WhisperX produced no timed spoken words for subtitles")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    events = [
        "Dialogue: 0,"
        f"{_ass_time(cue.start_seconds)},{_ass_time(cue.end_seconds)},"
        f"Default,,0,0,0,,{_ass_text(cue.text)}"
        for cue in cues
    ]
    content = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {play_res_x}",
            f"PlayResY: {play_res_y}",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,34,&H00FFFFFF,&H000000FF,&H7A000000,&H66000000,-1,0,0,0,100,100,0,0,1,2,1,2,40,40,34,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *events,
            "",
        ]
    )
    output.write_text(content, encoding="utf-8")
    return output


def default_subtitled_video_path(video_path: str | Path) -> Path:
    video = Path(video_path)
    if video.stem.endswith("_with_music"):
        stem = f"{video.stem.removesuffix('_with_music')}_subtitled_with_music"
    else:
        stem = f"{video.stem}_subtitled"
    return video.with_name(f"{stem}{video.suffix}")


def _subtitle_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def burn_subtitles(
    *,
    video_file: str | Path,
    subtitle_file: str | Path,
    output_file: str | Path | None = None,
    ffmpeg_bin: str | None = None,
) -> Path:
    video = Path(video_file)
    subtitles = Path(subtitle_file)
    output = Path(output_file) if output_file else default_subtitled_video_path(video)
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg was not found. Set --ffmpeg-bin or FFMPEG_BIN.")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"subtitles='{_subtitle_filter_path(subtitles)}'",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output),
        ],
        check=True,
    )
    return output


def add_whisperx_subtitles(
    *,
    video_file: str | Path,
    whisperx_json: str | Path,
    subtitle_file: str | Path,
    output_file: str | Path | None = None,
    ffmpeg_bin: str | None = None,
    result_file: str | Path | None = None,
) -> dict[str, Any]:
    cues, language = load_whisperx_cues(whisperx_json)
    subtitle_path = write_ass_subtitles(subtitle_file, cues)
    output_path = burn_subtitles(
        video_file=video_file,
        subtitle_file=subtitle_path,
        output_file=output_file,
        ffmpeg_bin=ffmpeg_bin,
    )
    result = {
        "version": "1.0",
        "source": "whisperx_observed_speech",
        "planner_lyrics_used": False,
        "language": language,
        "video_input": str(video_file),
        "whisperx_alignment": str(whisperx_json),
        "subtitle_file": str(subtitle_path),
        "video_output": str(output_path),
        "cue_count": len(cues),
        "transcript": " ".join(cue.text for cue in cues),
        "cues": [asdict(cue) for cue in cues],
    }
    if result_file:
        result_path = Path(result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
