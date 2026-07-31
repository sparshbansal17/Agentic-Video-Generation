from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_history_markdown(report: dict[str, Any]) -> str:
    runs = report["run_metrics"]
    quality = report["terminal_quality_diagnostics"]
    audio = report["audio_diagnostics"]
    provenance = report["provenance"]
    configuration = report["configuration_breakdown"]
    return "\n".join(
        [
            "# Agentic AV Benchmark: Retrospective Results",
            "",
            "> These measurements describe local, media-backed StoryMem Agentic runs. They are "
            "not head-to-head baseline results because AutoMV, MAVIN, and MovieAgent outputs are "
            "not installed in this workspace.",
            "",
            "## Coverage",
            "",
            f"- Reports discovered: {provenance['reports_discovered']}",
            f"- Reports with non-empty MP4 evidence: {provenance['reports_with_nonempty_mp4']}",
            f"- Media-backed runs: {runs['media_backed_runs']}",
            "",
            "## Pipeline metrics",
            "",
            f"- First media-backed iteration acceptance: "
            f"{runs['first_media_backed_iteration_pass_rate']}",
            f"- Latest media-backed iteration acceptance: "
            f"{runs['latest_media_backed_iteration_pass_rate']}",
            f"- Mean iterations/run: {runs['mean_iterations_per_run']}",
            f"- Repair success after a media-backed failure: {runs['repair_success_rate']}",
            f"- Mean scene pass rate: {quality['mean_scene_pass_rate']}",
            f"- Mean heuristic prompt-adherence proxy: "
            f"{quality['mean_heuristic_prompt_adherence_proxy']}",
            f"- Command-review terminal runs: {configuration['command_review_terminal_runs']}",
            f"- Command-review terminal acceptance: "
            f"{configuration['command_review_terminal_pass_rate']}",
            f"- Final video-stream pass rate: {quality['artifact_pass_rates'].get('has_video_stream')}",
            f"- Final audio-stream pass rate: {quality['artifact_pass_rates'].get('has_audio_stream')}",
            f"- Subtitle pass rate: {quality['artifact_pass_rates'].get('has_subtitles')}",
            "",
            "## Audio diagnostics",
            "",
            f"- Terminal runs with WER: {audio['terminal_runs_with_wer']}",
            f"- Mean WER: {audio['mean_wer']}",
            f"- Mean lyric-line completeness: {audio['mean_line_completeness']}",
            f"- Mean absolute line-boundary drift (seconds): "
            f"{audio['mean_absolute_line_boundary_drift_seconds']}",
            "",
            "## Interpretation",
            "",
            report["comparability_warning"],
            " " + quality["prompt_adherence_note"],
            "",
        ]
    )


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "retrospective.json"
    markdown_path = destination / "retrospective.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_history_markdown(report), encoding="utf-8")
    return json_path, markdown_path
