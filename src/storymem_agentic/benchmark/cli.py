from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapters import submission_coverage
from .advanced_media import write_advanced_media_score
from .aggregate import write_advanced_aggregate
from .compare import write_comparison
from .history import evaluate_history
from .media import write_clip_media_score
from .planning import write_planning_score
from .report import write_report
from .schema import validate_manifest, validate_submission
from .vlm_media import write_vlm_media_score
from .vlm_plan import write_plan_vlm_score
from .vlm_plan_panel import write_plan_vlm_panel_score
from .vlm_panel import write_vlm_panel_score


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

    planning = subparsers.add_parser("score-plan", help="score a normalized agent plan")
    planning.add_argument("--manifest", required=True)
    planning.add_argument("--case-id", required=True)
    planning.add_argument("--plan", required=True)
    planning.add_argument("--provenance")
    planning.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare", help="compare media-backed benchmark submissions")
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--submissions-root", required=True)
    compare.add_argument("--plans-root", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--seed", type=int, default=0)

    media = subparsers.add_parser("score-media", help="score raw pre-subtitle video with CLIP")
    media.add_argument("--manifest", required=True)
    media.add_argument("--case-id", required=True)
    media.add_argument("--video", required=True)
    media.add_argument("--output", required=True)
    media.add_argument("--device", default="cpu")
    media.add_argument("--clip-cache")

    advanced_media = subparsers.add_parser(
        "score-media-advanced", help="score dense video semantics and temporal quality"
    )
    advanced_media.add_argument("--manifest", required=True)
    advanced_media.add_argument("--case-id", required=True)
    advanced_media.add_argument("--video", required=True)
    advanced_media.add_argument("--output", required=True)
    advanced_media.add_argument("--device", default="cpu")
    advanced_media.add_argument("--clip-cache")
    advanced_media.add_argument("--dino-model", default="facebook/dinov2-base")
    advanced_media.add_argument("--dino-cache")
    advanced_media.add_argument("--frames-per-scene", type=int, default=8)

    vlm_media = subparsers.add_parser(
        "score-media-vlm", help="score a blinded scene contact sheet with a VLM rubric"
    )
    vlm_media.add_argument("--manifest", required=True)
    vlm_media.add_argument("--case-id", required=True)
    vlm_media.add_argument("--video", required=True)
    vlm_media.add_argument("--output", required=True)
    vlm_media.add_argument("--model-path", required=True)
    vlm_media.add_argument("--device-map", default="auto")
    vlm_media.add_argument("--mosaic-output")

    vlm_plan = subparsers.add_parser(
        "score-plan-vlm", help="score normalized plan semantics with a blinded VLM rubric"
    )
    vlm_plan.add_argument("--manifest", required=True)
    vlm_plan.add_argument("--case-id", required=True)
    vlm_plan.add_argument("--plan", required=True)
    vlm_plan.add_argument("--output", required=True)
    vlm_plan.add_argument("--model-path", required=True)
    vlm_plan.add_argument("--device-map", default="auto")

    vlm_panel = subparsers.add_parser(
        "score-media-vlm-panel", help="compare three system-blind videos in cyclic positions"
    )
    vlm_panel.add_argument("--manifest", required=True)
    vlm_panel.add_argument("--case-id", required=True)
    vlm_panel.add_argument("--candidate", action="append", required=True)
    vlm_panel.add_argument("--output", required=True)
    vlm_panel.add_argument("--per-system-output-root", required=True)
    vlm_panel.add_argument("--seed-dir", default="seed_000")
    vlm_panel.add_argument("--model-path", required=True)
    vlm_panel.add_argument("--device-map", default="auto")

    plan_panel = subparsers.add_parser(
        "score-plan-vlm-panel", help="compare three system-blind normalized plans"
    )
    plan_panel.add_argument("--manifest", required=True)
    plan_panel.add_argument("--case-id", required=True)
    plan_panel.add_argument("--candidate", action="append", required=True)
    plan_panel.add_argument("--output", required=True)
    plan_panel.add_argument("--per-system-output-root", required=True)
    plan_panel.add_argument("--seed-dir", default="seed_000")
    plan_panel.add_argument("--model-path", required=True)
    plan_panel.add_argument("--device-map", default="auto")

    aggregate = subparsers.add_parser(
        "aggregate-advanced", help="aggregate advanced media metrics across renderer seeds"
    )
    aggregate.add_argument("--manifest", required=True)
    aggregate.add_argument("--submissions-root", required=True)
    aggregate.add_argument("--case-id", required=True)
    aggregate.add_argument("--seeds", nargs="+", type=int, required=True)
    aggregate.add_argument("--output", required=True)
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
    if args.command == "score-plan":
        result = write_planning_score(
            manifest_path=args.manifest,
            case_id=args.case_id,
            plan_path=args.plan,
            provenance_path=args.provenance,
            output_path=args.output,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "compare":
        json_path, markdown_path = write_comparison(
            manifest_path=args.manifest,
            submissions_root=args.submissions_root,
            plans_root=args.plans_root,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(json_path)
        print(markdown_path)
        return 0
    if args.command == "score-media":
        result = write_clip_media_score(
            manifest_path=args.manifest,
            case_id=args.case_id,
            video_path=args.video,
            output_path=args.output,
            device=args.device,
            clip_cache=args.clip_cache,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "score-media-advanced":
        result = write_advanced_media_score(
            manifest_path=args.manifest,
            case_id=args.case_id,
            video_path=args.video,
            output_path=args.output,
            device=args.device,
            clip_cache=args.clip_cache,
            dino_model=args.dino_model,
            dino_cache=args.dino_cache,
            frames_per_scene=args.frames_per_scene,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "score-media-vlm":
        result = write_vlm_media_score(
            manifest_path=args.manifest,
            case_id=args.case_id,
            video_path=args.video,
            output_path=args.output,
            model_path=args.model_path,
            device_map=args.device_map,
            mosaic_output=args.mosaic_output,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "score-plan-vlm":
        result = write_plan_vlm_score(
            manifest_path=args.manifest,
            case_id=args.case_id,
            plan_path=args.plan,
            output_path=args.output,
            model_path=args.model_path,
            device_map=args.device_map,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "score-media-vlm-panel":
        candidates = {}
        for item in args.candidate:
            if "=" not in item:
                raise ValueError("--candidate must use system=video_path")
            system, video_path = item.split("=", 1)
            candidates[system] = video_path
        result = write_vlm_panel_score(
            manifest_path=args.manifest,
            case_id=args.case_id,
            candidates=candidates,
            output_path=args.output,
            per_system_output_root=args.per_system_output_root,
            seed_dir=args.seed_dir,
            model_path=args.model_path,
            device_map=args.device_map,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "score-plan-vlm-panel":
        candidates = {}
        for item in args.candidate:
            if "=" not in item:
                raise ValueError("--candidate must use system=plan_path")
            system, plan_path = item.split("=", 1)
            candidates[system] = plan_path
        result = write_plan_vlm_panel_score(
            manifest_path=args.manifest,
            case_id=args.case_id,
            candidates=candidates,
            output_path=args.output,
            per_system_output_root=args.per_system_output_root,
            seed_dir=args.seed_dir,
            model_path=args.model_path,
            device_map=args.device_map,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "aggregate-advanced":
        result = write_advanced_aggregate(
            manifest_path=args.manifest,
            submissions_root=args.submissions_root,
            case_id=args.case_id,
            seeds=args.seeds,
            output_path=args.output,
        )
        print(json.dumps(result, indent=2))
        return 0
    report = evaluate_history(args.results_root)
    json_path, markdown_path = write_report(report, args.output_dir)
    print(json_path)
    print(markdown_path)
    return 0
