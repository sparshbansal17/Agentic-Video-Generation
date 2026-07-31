from __future__ import annotations

import json
from pathlib import Path

import pytest

from storymem_agentic.benchmark.adapters import submission_coverage
from storymem_agentic.benchmark.cli import main
from storymem_agentic.benchmark.history import evaluate_history
from storymem_agentic.benchmark.metrics import detection_metrics, localization_metrics
from storymem_agentic.benchmark.schema import (
    BenchmarkValidationError,
    validate_manifest,
    validate_submission,
)


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "benchmarks" / "agentic_av_v1" / "manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _write_iteration(
    root: Path,
    run_id: str,
    iteration: int,
    *,
    passed: bool,
    with_media: bool = True,
    wer: float = 0.0,
) -> None:
    destination = root / run_id / "iterations" / f"{iteration:03d}"
    destination.mkdir(parents=True)
    if with_media:
        generated = destination / "generated"
        generated.mkdir()
        (generated / "final.mp4").write_bytes(b"not-a-real-video-but-nonempty-evidence")
    report = {
        "passed": passed,
        "scene_reports": [
            {
                "scene_num": 1,
                "passed": passed,
                "scores": {"vlm_prompt_adherence": 0.8},
            }
        ],
        "reviewer_reports": [
            {"reviewer": "ArtifactReviewAgent", "passed": passed},
            {
                "reviewer": "WhisperXLyricTimingAgent",
                "passed": wer <= 0.08,
                "evidence": {
                    "word_error_rate": wer,
                    "lines": [
                        {
                            "matched_ratio": 1.0 - wer,
                            "start_drift_seconds": 0.2,
                            "end_drift_seconds": -0.4,
                        }
                    ],
                },
            },
        ],
        "regeneration_targets": [1] if not passed else [],
    }
    (destination / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_manifest_covers_primary_systems_and_both_tracks() -> None:
    manifest = validate_manifest(_manifest())
    assert len(manifest["cases"]) == 12
    assert manifest["seeds"] == [0, 1, 2]
    assert {case["track"] for case in manifest["cases"]} == {
        "conditioned_song_to_video",
        "end_to_end_prompt_to_song_video",
    }


def test_manifest_rejects_duplicate_case() -> None:
    manifest = _manifest()
    manifest["cases"].append(dict(manifest["cases"][0]))
    with pytest.raises(BenchmarkValidationError, match="duplicate case_id"):
        validate_manifest(manifest)


def test_detection_and_localization_metrics() -> None:
    detection = detection_metrics(
        ["truncated_final_lyric", "character_identity_swap"],
        ["truncated_final_lyric", "unsafe_content"],
    )
    assert detection == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    localization = localization_metrics([2, 3], [3, 4])
    assert localization["scene_f1"] == 0.5
    assert localization["collateral_scenes"] == 1
    assert localization["missed_scenes"] == 1


def test_submission_validation_and_coverage_do_not_invent_missing_results(
    tmp_path: Path,
) -> None:
    manifest = validate_manifest(_manifest())
    submission = {
        "system": "automv",
        "case_id": "conditioned_twinkle",
        "final_video": "final.mp4",
    }
    location = tmp_path / "automv" / "conditioned_twinkle" / "submission.json"
    location.parent.mkdir(parents=True)
    location.write_text(json.dumps(submission), encoding="utf-8")
    validate_submission(submission, manifest=manifest)
    report = submission_coverage(tmp_path, manifest)
    assert report["systems"]["automv"]["submitted"] == 1
    assert report["systems"]["storymem_agentic"]["submitted"] == 0
    assert report["systems"]["mavin"]["coverage"] == 0.0
    with pytest.raises(BenchmarkValidationError, match="missing or empty"):
        validate_submission(
            submission,
            manifest=manifest,
            base_dir=location.parent,
            require_media=True,
        )


def test_history_excludes_report_only_runs_and_measures_recovery(tmp_path: Path) -> None:
    _write_iteration(tmp_path, "recovered", 1, passed=False, wer=0.2)
    _write_iteration(tmp_path, "recovered", 2, passed=True)
    _write_iteration(tmp_path, "accepted", 1, passed=True)
    _write_iteration(tmp_path, "report_only", 1, passed=True, with_media=False)
    report = evaluate_history(tmp_path)
    assert report["provenance"]["reports_discovered"] == 4
    assert report["provenance"]["reports_with_nonempty_mp4"] == 3
    assert report["run_metrics"]["media_backed_runs"] == 2
    assert report["run_metrics"]["first_media_backed_iteration_pass_rate"] == 0.5
    assert report["run_metrics"]["latest_media_backed_iteration_pass_rate"] == 1.0
    assert report["run_metrics"]["repair_success_rate"] == 1.0
    assert report["audio_diagnostics"]["mean_wer"] == 0.0


def test_cli_validates_and_writes_retrospective(tmp_path: Path) -> None:
    assert main(["validate", "--manifest", str(MANIFEST)]) == 0
    results = tmp_path / "results"
    _write_iteration(results, "one", 1, passed=True)
    output = tmp_path / "benchmark_results"
    assert (
        main(
            [
                "history",
                "--results-root",
                str(results),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "retrospective.json").is_file()
    assert "not head-to-head" in (output / "retrospective.md").read_text(encoding="utf-8")
