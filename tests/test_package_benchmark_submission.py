from __future__ import annotations

from scripts.package_benchmark_submission import (
    case_from_manifest,
    srt_timestamp,
    write_locked_subtitles,
)


def test_case_from_manifest_selects_locked_case() -> None:
    manifest = {"cases": [{"case_id": "one"}, {"case_id": "two", "value": 2}]}
    assert case_from_manifest(manifest, "two") == {"case_id": "two", "value": 2}


def test_write_locked_subtitles_uses_equal_case_windows(tmp_path) -> None:
    output = tmp_path / "subtitles.srt"
    write_locked_subtitles(
        {"lyrics": ["one", "two", "three", "four"], "target_duration_seconds": 24},
        output,
    )
    rendered = output.read_text()
    assert "00:00:00,000 --> 00:00:06,000\none" in rendered
    assert "00:00:18,000 --> 00:00:24,000\nfour" in rendered
    assert srt_timestamp(3661.234) == "01:01:01,234"
