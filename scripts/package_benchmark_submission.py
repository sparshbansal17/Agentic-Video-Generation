#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(ffprobe: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def duration(probe_value: dict[str, Any]) -> float:
    return float(probe_value["format"]["duration"])


def case_from_manifest(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in manifest["cases"]:
        if case["case_id"] == case_id:
            return dict(case)
    raise ValueError(f"unknown benchmark case: {case_id}")


def srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def write_locked_subtitles(case: dict[str, Any], path: Path) -> None:
    lyrics = list(case["lyrics"])
    step = float(case["target_duration_seconds"]) / len(lyrics)
    entries = []
    for index, lyric in enumerate(lyrics, 1):
        entries.append(
            f"{index}\n{srt_timestamp((index - 1) * step)} --> {srt_timestamp(index * step)}\n{lyric}\n"
        )
    path.write_text("\n".join(entries), encoding="utf-8")


def ffmpeg_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a common-renderer benchmark submission")
    parser.add_argument("--system", choices=("automv", "movieagent", "storymem_agentic"), required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-video", type=Path, required=True)
    parser.add_argument("--locked-audio", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/home/bansa125/bin/ffmpeg")
    parser.add_argument("--ffprobe", default="/home/bansa125/bin/ffprobe")
    parser.add_argument("--generation-seconds", type=float)
    parser.add_argument("--notes")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    case = case_from_manifest(manifest, args.case_id)
    story = load_json(args.plan)
    if len(story.get("scenes", [])) != int(case["expected_scenes"]):
        raise ValueError("plan scene count does not match the locked case")
    if not args.visual_video.is_file() or not args.locked_audio.is_file():
        raise FileNotFoundError("visual video and locked audio must exist before packaging")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_video = args.output_dir / "final.mp4"
    subtitle_path = args.output_dir / "subtitles.srt"
    write_locked_subtitles(case, subtitle_path)
    subtitle_filter_path = ffmpeg_filter_path(subtitle_path)
    visual_probe = probe(args.ffprobe, args.visual_video)
    audio_probe = probe(args.ffprobe, args.locked_audio)
    locked_audio_hash = sha256(args.locked_audio)
    expected_audio_hash = case.get("locked_audio_sha256")
    if expected_audio_hash and locked_audio_hash != expected_audio_hash:
        raise ValueError(
            f"locked audio checksum mismatch: {locked_audio_hash} != {expected_audio_hash}"
        )
    target = float(case["target_duration_seconds"])
    visual_duration = duration(visual_probe)
    audio_duration = duration(audio_probe)
    # Normalize timing once for all common-renderer systems. setpts changes only presentation
    # timestamps; the separately synthesized locked master is muxed identically for every system.
    started = time.monotonic()
    subprocess.run(
        [
            args.ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(args.visual_video),
            "-i",
            str(args.locked_audio),
            "-filter:v",
            (
                f"setpts={target / visual_duration:.12f}*PTS,fps=24,"
                f"subtitles='{subtitle_filter_path}'"
            ),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{target:.6f}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(final_video),
        ],
        check=True,
    )
    packaging_seconds = time.monotonic() - started
    final_probe = probe(args.ffprobe, final_video)
    streams = final_probe.get("streams", [])
    metrics = {
        "system": args.system,
        "case_id": args.case_id,
        "track": case["track"],
        "raw_visual_duration_seconds": visual_duration,
        "locked_audio_duration_seconds": audio_duration,
        "final_duration_seconds": duration(final_probe),
        "target_duration_seconds": target,
        "absolute_duration_error_seconds": abs(duration(final_probe) - target),
        "has_video_stream": any(stream.get("codec_type") == "video" for stream in streams),
        "has_audio_stream": any(stream.get("codec_type") == "audio" for stream in streams),
        "subtitles_burned": True,
        "subtitle_sha256": sha256(subtitle_path),
        "locked_audio_sha256": locked_audio_hash,
        "raw_visual_sha256": sha256(args.visual_video),
        "final_video_sha256": sha256(final_video),
        "packaging_seconds": packaging_seconds,
        "normalization": "single common setpts/fps24/H.264/AAC operation",
    }
    metrics_path = args.output_dir / "delivery_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    notes = args.notes or (
        "Repository StoryMem Agentic output packaged with the locked conditioned audio."
        if args.system == "storymem_agentic"
        else (
            "Shared-generator agent-design track: published agent contract with the common "
            "StoryMem renderer and locked conditioned audio; not a native-backend result."
        )
    )
    submission = {
        "system": args.system,
        "case_id": args.case_id,
        "final_video": "final.mp4",
        "evaluation_report": "delivery_metrics.json",
        "execution_trace": str(args.trace.resolve()) if args.trace else None,
        "wall_time_seconds": args.generation_seconds,
        "estimated_cost_usd": 0.0,
        "regenerated_seconds": 0.0,
        "notes": notes,
    }
    (args.output_dir / "submission.json").write_text(
        json.dumps(submission, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
