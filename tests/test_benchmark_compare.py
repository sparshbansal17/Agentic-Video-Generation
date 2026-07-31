from __future__ import annotations

import json
from pathlib import Path

from storymem_agentic.benchmark.compare import compare_submissions


def test_compare_submissions_joins_delivery_and_planning(tmp_path: Path) -> None:
    manifest = {
        "benchmark_id": "test",
        "systems": ["storymem_agentic", "automv", "mavin", "movieagent"],
        "cases": [
            {
                "case_id": "case",
                "locked_audio_sha256": "audio-hash",
            }
        ],
    }
    submission_dir = tmp_path / "submissions" / "automv" / "case" / "seed_000"
    submission_dir.mkdir(parents=True)
    (submission_dir / "final.mp4").write_bytes(b"media")
    (submission_dir / "delivery.json").write_text(
        json.dumps(
            {
                "has_video_stream": True,
                "has_audio_stream": True,
                "absolute_duration_error_seconds": 0.0,
                "locked_audio_sha256": "audio-hash",
            }
        )
    )
    (submission_dir / "submission.json").write_text(
        json.dumps(
            {
                "system": "automv",
                "case_id": "case",
                "final_video": "final.mp4",
                "evaluation_report": "delivery.json",
            }
        )
    )
    plan_dir = tmp_path / "plans" / "automv" / "case" / "seed_000"
    plan_dir.mkdir(parents=True)
    (plan_dir / "planning_metrics.json").write_text(
        json.dumps({"scene_count_accuracy": 1.0, "planning_seconds": 12.0})
    )

    report = compare_submissions(
        manifest=manifest,
        submissions_root=tmp_path / "submissions",
        plans_root=tmp_path / "plans",
    )

    assert report["validation_errors"] == []
    assert report["rows"][0]["delivery_pass"] is True
    assert report["rows"][0]["audio_checksum_matches_manifest"] is True
    assert report["rows"][0]["scene_count_accuracy"] == 1.0
    assert report["rows"][0]["planning_seconds"] == 12.0


def test_compare_submissions_includes_planning_only_systems(tmp_path: Path) -> None:
    manifest = {
        "benchmark_id": "test",
        "systems": ["storymem_agentic", "automv", "movieagent"],
        "cases": [{"case_id": "case", "locked_audio_sha256": "audio-hash"}],
    }
    for system, seconds in (("automv", 12.0), ("movieagent", 34.0)):
        plan_dir = tmp_path / "plans" / system / "case" / "seed_000"
        plan_dir.mkdir(parents=True)
        (plan_dir / "planning_metrics.json").write_text(
            json.dumps(
                {
                    "scene_count_accuracy": 1.0,
                    "planning_seconds": seconds,
                }
            )
        )

    report = compare_submissions(
        manifest=manifest,
        submissions_root=tmp_path / "submissions",
        plans_root=tmp_path / "plans",
    )

    assert [row["system"] for row in report["rows"]] == ["automv", "movieagent"]
    assert all(row["delivery_pass"] is None for row in report["rows"])
    assert all(row["generation_seconds"] is None for row in report["rows"])
    assert report["rows"][1]["planning_seconds"] == 34.0
    assert "Planning-only" in report["rows"][0]["notes"]
