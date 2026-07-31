from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapters import submission_coverage
from .history import evaluate_history
from .report import write_report
from .schema import validate_manifest, validate_submission


def _json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic audio-video benchmark tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a manifest and submissions")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--submission", action="append", default=[])
    validate.add_argument("--require-media", action="store_true")

    history = subparsers.add_parser("history", help="score media-backed local run history")
    history.add_argument("--results-root", default="results")
    history.add_argument("--output-dir", default="benchmark_results/agentic_av_v1")

    coverage = subparsers.add_parser("coverage", help="report baseline submission coverage")
    coverage.add_argument("--manifest", required=True)
    coverage.add_argument("--submissions-root", default="benchmark_submissions")
    coverage.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        manifest = validate_manifest(_json(args.manifest))
        for raw_path in args.submission:
            path = Path(raw_path)
            validate_submission(
                _json(path),
                manifest=manifest,
                base_dir=path.parent,
                require_media=args.require_media,
            )
        print(f"valid: {args.manifest}; submissions={len(args.submission)}")
        return 0
    if args.command == "coverage":
        manifest = validate_manifest(_json(args.manifest))
        report = submission_coverage(args.submissions_root, manifest)
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
            print(destination)
        else:
            print(rendered, end="")
        return 0
    report = evaluate_history(args.results_root)
    json_path, markdown_path = write_report(report, args.output_dir)
    print(json_path)
    print(markdown_path)
    return 0
