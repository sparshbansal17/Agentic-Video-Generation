#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storymem_agentic.orchestrator import run_workflow


def load_case(manifest: Path, case_id: str) -> dict[str, Any]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for case in data.get("cases", []):
        if case.get("case_id") == case_id:
            return dict(case)
    raise ValueError(f"unknown benchmark case: {case_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run StoryMem Agentic on a locked benchmark case")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--max-plan-revisions", type=int, default=6)
    args = parser.parse_args()

    case = load_case(args.manifest, args.case_id)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "planner_attempt_*.json",
        "planner_draft_*.json",
        "planner_revision_*.json",
        "plan_review_*.json",
    ):
        for stale in args.run_dir.glob(pattern):
            stale.unlink()
    for filename in (
        "storymem_story.json",
        "native_plan.json",
        "planning_trace.json",
        "planning_metrics.json",
        "provenance.json",
    ):
        stale = args.output_dir / filename
        if stale.exists():
            stale.unlink()
    # Keep the venv launcher path. Resolving its symlink selects the system Python,
    # which does not contain the benchmark's torch/transformers installation.
    python = Path(sys.executable)
    repo = Path(__file__).resolve().parents[1]
    planner_debug = args.run_dir / "planner_raw_output.txt"
    critic_debug = args.run_dir / "plan_critic_raw_output.txt"
    planner_command = (
        f"{python} {repo / 'scripts/local_lullaby_planner.py'} --model {args.model} "
        f"--debug-output {planner_debug}"
    )
    critic_command = (
        f"{python} {repo / 'scripts/local_plan_critic.py'} --model {args.model} "
        f"--debug-output {critic_debug}"
    )
    required = ", ".join(case.get("required_entities", []))
    topic = case["prompt"]
    if required:
        topic += f" Required continuity entities: {required}."

    started = time.monotonic()
    result = run_workflow(
        topic_or_name=topic,
        lyrics="\n".join(case["lyrics"]),
        output_dir=args.run_dir,
        mode="dry_run",
        target_duration=float(case["target_duration_seconds"]),
        clip_count=int(case["expected_scenes"]),
        max_iterations=1,
        seed=0,
        planner_backend="command",
        planner_command=planner_command,
        plan_critic_command=critic_command,
        max_plan_revisions=args.max_plan_revisions,
        review_backend="mock",
        audio_aligner="none",
        generate_audio=False,
    )
    elapsed = time.monotonic() - started
    planner_report_path = args.run_dir / "planner_agent_output.json"
    planner_report = json.loads(planner_report_path.read_text(encoding="utf-8"))
    if (
        planner_report.get("error")
        and (
            not planner_report.get("validation_passed")
            or not planner_report.get("critic_passed")
        )
    ):
        raise RuntimeError(f"agentic planner failed: {planner_report['error']}")
    if not planner_report.get("validation_passed") or not planner_report.get("critic_passed"):
        raise RuntimeError("agentic planner did not pass deterministic and semantic review")
    iteration = Path(result["latest_iteration"])
    story_source = iteration / "storymem_story.json"
    plan_source = iteration / "production_plan.json"
    if not story_source.is_file() or not plan_source.is_file():
        raise FileNotFoundError(f"agentic planning did not produce benchmark artifacts: {iteration}")

    shutil.copy2(story_source, args.output_dir / "storymem_story.json")
    shutil.copy2(plan_source, args.output_dir / "native_plan.json")
    step_paths = sorted(set(args.run_dir.glob("plan*_*.json")))
    trace = [
        {
            "artifact": str(path.relative_to(args.run_dir)),
            "payload": json.loads(path.read_text(encoding="utf-8")),
        }
        for path in step_paths
    ]
    (args.output_dir / "planning_trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    attempts = max(1, len(list(args.run_dir.glob("planner_attempt_*.json"))))
    source_files = (
        repo / "src/storymem_agentic/planner.py",
        repo / "scripts/local_lullaby_planner.py",
        repo / "scripts/local_plan_critic.py",
    )
    source_hasher = hashlib.sha256()
    for source in source_files:
        source_hasher.update(str(source.relative_to(repo)).encode("utf-8"))
        source_hasher.update(b"\0")
        source_hasher.update(source.read_bytes())
        source_hasher.update(b"\0")
    provenance = {
        "system": "storymem_agentic",
        "case_id": args.case_id,
        "model": args.model,
        "source_contract": "planner.py + local_lullaby_planner.py + local_plan_critic.py",
        "source_sha256": source_hasher.hexdigest(),
        "agent_calls": int(planner_report.get("agent_call_count", 1 + 2 * attempts)),
        "planning_seconds": elapsed,
        "run_dir": str(args.run_dir.resolve()),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
