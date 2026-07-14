#!/usr/bin/env python3
"""Score reference transcription and synthesized voice candidates."""

import argparse
import json
import re
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def edit_distance(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def transcript(path: Path) -> str:
    data = json.loads(path.read_text())
    return " ".join(str(segment.get("text", "")).strip() for segment in data.get("segments", [])).strip()


def score(expected: str, actual: str) -> float:
    expected_words = words(expected)
    return edit_distance(expected_words, words(actual)) / max(1, len(expected_words))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--max-wer", type=float, default=0.25)
    args = parser.parse_args()

    candidates = {
        "reference": (args.reference_text, args.output_dir / "reference_whisperx"),
        "f5": (args.target_text, args.output_dir / "f5_whisperx"),
        "cosyvoice": (args.target_text, args.output_dir / "cosyvoice_whisperx"),
    }
    report = {}
    for name, (expected, directory) in candidates.items():
        files = sorted(directory.glob("*.json")) if directory.exists() else []
        if not files:
            continue
        actual = transcript(files[0])
        report[name] = {"expected": expected, "transcript": actual, "wer": score(expected, actual)}

    viable = [name for name in ("f5", "cosyvoice") if report.get(name, {}).get("wer", 1.0) <= args.max_wer]
    report["max_wer"] = args.max_wer
    report["viable_backends"] = viable
    (args.output_dir / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if viable and report.get("reference", {}).get("wer", 1.0) <= args.max_wer else 3


if __name__ == "__main__":
    raise SystemExit(main())
