#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from local_lullaby_planner import extract_json, generate_with_transformers


def build_user_prompt(payload: dict[str, Any]) -> str:
    context = payload.get("context", {})
    contract = context.get("review_contract", {})
    return (
        f"{payload.get('prompt', '')}\n\n"
        "Review contract JSON:\n"
        f"{json.dumps(contract, indent=2)}\n\n"
        "Deterministic validation report (binding; do not duplicate its issues):\n"
        f"{json.dumps(context.get('deterministic_report', {}), indent=2)}\n\n"
        "Planner-owned review plan to review:\n"
        f"{json.dumps(context.get('review_plan', {}), indent=2)}\n\n"
        "Return one JSON object only. Use passed=false if you find any additional semantic issue. "
        "Each issue must identify code, scene_num, message, an editable review_plan field, suggested_change, and "
        "evidence as an object with observed, expected, and source strings. Do not review fields absent from review_plan."
    )


def validate_review(review: dict[str, Any]) -> None:
    if not isinstance(review.get("passed"), bool):
        raise ValueError("critic review requires boolean passed")
    if not isinstance(review.get("issues"), list):
        raise ValueError("critic review requires issues array")
    if review["passed"] and review["issues"]:
        raise ValueError("critic cannot pass while reporting issues")
    if not review["passed"] and not review["issues"]:
        raise ValueError("critic rejection requires at least one actionable issue")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen semantic critic for StoryMem production plans")
    parser.add_argument("--model", default=os.environ.get("LOCAL_PLANNER_MODEL", "Qwen/Qwen2-VL-7B-Instruct"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("LOCAL_CRITIC_MAX_NEW_TOKENS", "2048")))
    parser.add_argument("--debug-output")
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read())
    raw = generate_with_transformers(args.model, build_user_prompt(payload), args.max_new_tokens)
    if args.debug_output:
        with open(args.debug_output, "w", encoding="utf-8") as handle:
            handle.write(raw)
    try:
        review = extract_json(raw)
        validate_review(review)
    except Exception as exc:
        print(f"local_plan_critic invalid model output: {exc}", file=sys.stderr)
        print(raw[:4000], file=sys.stderr)
        return 4
    review.setdefault("scores", {})
    review.setdefault("revision_notes", [])
    print(json.dumps(review))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
