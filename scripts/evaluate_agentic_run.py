from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2


def _video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or not frames:
        return 0.0
    return float(frames / fps)


def _media_streams(ffmpeg_bin: str, path: Path) -> dict[str, bool]:
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    probe = result.stdout + result.stderr
    return {"has_video": "Video:" in probe, "has_audio": "Audio:" in probe}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a compact StoryMem agentic test run.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--final-video", required=True)
    parser.add_argument("--expected-clips", type=int, default=3)
    parser.add_argument("--ffmpeg-bin", default="/home/bansa125/bin/ffmpeg")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    final_video = Path(args.final_video)
    clips = sorted(output_dir.glob("[0-9][0-9]_01.mp4"))
    audio_dir = output_dir / "audio"
    streams = _media_streams(args.ffmpeg_bin, final_video) if final_video.exists() else {"has_video": False, "has_audio": False}
    duration = _video_duration(final_video) if final_video.exists() else 0.0

    checks = {
        "final_video_exists": final_video.exists(),
        "clip_count_ok": len(clips) == args.expected_clips,
        "has_video_stream": streams["has_video"],
        "has_audio_stream": streams["has_audio"],
        "duration_ok": 8.0 <= duration <= 25.0,
        "audio_plan_exists": (audio_dir / "audio_plan.json").exists(),
        "mix_manifest_exists": (audio_dir / "mix_manifest.json").exists(),
        "backend_manifest_exists": (audio_dir / "backend_invocations.json").exists(),
    }
    recommendations = []
    if not checks["clip_count_ok"]:
        recommendations.append("regenerate_video_with_expected_scene_count")
    if not checks["has_audio_stream"]:
        recommendations.append("rerun_audio_postprocess_or_switch_audio_backend")
    if not checks["duration_ok"]:
        recommendations.append("adjust_target_duration_or_scene_count")
    if not recommendations:
        recommendations.append("accept")

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "clip_count": len(clips),
        "duration_seconds": round(duration, 3),
        "final_video": str(final_video),
        "recommendations": recommendations,
    }
    report_path = output_dir / "agentic_media_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    tuning_path = output_dir / "agentic_tuning_report.json"
    tuning_path.write_text(
        json.dumps(
            {
                "status": "accepted" if report["passed"] else "needs_iteration",
                "next_actions": recommendations,
                "source_evaluation": str(report_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
