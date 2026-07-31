from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import PRIMARY_SYSTEMS, validate_submission


ADAPTERS: dict[str, dict[str, Any]] = {
    "storymem_agentic": {
        "native_input": "prompt_and_optional_lyrics",
        "adapter": "Run the repository pipeline once; do not hand-edit intermediate plans.",
    },
    "automv": {
        "native_input": "finished_song",
        "adapter": "Use the locked benchmark song as AutoMV's input; disable manual shot edits.",
    },
    "mavin": {
        "native_input": "hierarchical_multishot_caption",
        "adapter": "Convert the locked case specification once, using the published scripting pipeline.",
    },
    "movieagent": {
        "native_input": "script_and_character_bank",
        "adapter": "Use the locked case script and derive its character bank without manual revisions.",
    },
}


def submission_coverage(root: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate submission JSON files and report coverage without inventing missing scores."""
    directory = Path(root)
    case_ids = {case["case_id"] for case in manifest["cases"]}
    found: dict[str, set[str]] = {system: set() for system in PRIMARY_SYSTEMS}
    errors: list[dict[str, str]] = []
    for path in sorted(directory.glob("**/submission.json")) if directory.exists() else []:
        try:
            submission = json.loads(path.read_text(encoding="utf-8"))
            validate_submission(submission, manifest=manifest, base_dir=path.parent)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        found[submission["system"]].add(submission["case_id"])
    return {
        "expected_cases_per_system": len(case_ids),
        "systems": {
            system: {
                "submitted": len(found[system]),
                "coverage": len(found[system]) / len(case_ids) if case_ids else None,
                "missing_case_ids": sorted(case_ids - found[system]),
                "adapter": ADAPTERS[system],
            }
            for system in PRIMARY_SYSTEMS
        },
        "validation_errors": errors,
    }
