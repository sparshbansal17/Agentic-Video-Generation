from __future__ import annotations

import pytest

from storymem_agentic.benchmark.media import _case


def test_media_case_selects_locked_case() -> None:
    manifest = {"cases": [{"case_id": "one"}, {"case_id": "two", "value": 2}]}
    assert _case(manifest, "two") == {"case_id": "two", "value": 2}
    with pytest.raises(ValueError, match="unknown benchmark case"):
        _case(manifest, "missing")
