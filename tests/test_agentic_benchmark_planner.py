from pathlib import Path

import pytest

from scripts.run_agentic_benchmark_planner import load_case


def test_load_case_reads_locked_case(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"cases": [{"case_id": "locked", "lyrics": ["line"]}]}')
    assert load_case(manifest, "locked")["lyrics"] == ["line"]


def test_load_case_rejects_unknown_case(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"cases": []}')
    with pytest.raises(ValueError, match="unknown benchmark case"):
        load_case(manifest, "missing")
