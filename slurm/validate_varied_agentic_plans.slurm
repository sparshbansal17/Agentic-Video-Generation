#!/bin/bash
#SBATCH --job-name=validate-varied-plans
#SBATCH --output=logs/validate_varied_plans_%j.out
#SBATCH --error=logs/validate_varied_plans_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00

set -euo pipefail
umask 002

AGENTIC_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
VENV_DIR="${VENV_DIR:-${AGENTIC_DIR}/.venv}"
VALIDATION_ROOT="${VALIDATION_ROOT:-${AGENTIC_DIR}/results/varied_planner_validation_${SLURM_JOB_ID}}"
PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
ONLY_SLUG="${ONLY_SLUG:-}"
MAX_PLAN_REVISIONS="${MAX_PLAN_REVISIONS:-8}"
PRODUCTION_PLAN_PATH="${PRODUCTION_PLAN_PATH:-}"

module load python
source "${VENV_DIR}/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${AGENTIC_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"
mkdir -p "${AGENTIC_DIR}/logs" "${VALIDATION_ROOT}" "${TMPDIR}"

TOPICS=(
  "Row, Row, Row Your Boat"
  "Rain, Rain, Go Away"
  "Rock-a-bye Baby"
  "Hickory Dickory Dock"
)
SLUGS=(row_boat rain_away rockabye hickory_dock)
LYRICS_FILES=(
  "${AGENTIC_DIR}/story/row_boat_first4_lyrics.txt"
  "${AGENTIC_DIR}/story/rain_away_first4_lyrics.txt"
  "${AGENTIC_DIR}/story/rockabye_first4_lyrics.txt"
  "${AGENTIC_DIR}/story/hickory_dock_first4_lyrics.txt"
)

for index in "${!TOPICS[@]}"; do
  topic="${TOPICS[$index]}"
  slug="${SLUGS[$index]}"
  if [ -n "${ONLY_SLUG}" ] && [ "${slug}" != "${ONLY_SLUG}" ]; then
    continue
  fi
  lyrics_file="${LYRICS_FILES[$index]}"
  output="${VALIDATION_ROOT}/${slug}"
  planner_cmd="${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_lullaby_planner.py --model ${PLANNER_MODEL} --debug-output ${output}/planner_raw_output.txt"
  critic_cmd="${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_plan_critic.py --model ${PLANNER_MODEL} --debug-output ${output}/plan_critic_raw_output.txt"
  production_plan_args=()
  if [ -n "${PRODUCTION_PLAN_PATH}" ]; then
    production_plan_args=(--production-plan "${PRODUCTION_PLAN_PATH}")
  fi

  python -m storymem_agentic.cli plan \
    --topic "${topic}" \
    --lyrics-file "${lyrics_file}" \
    --output-dir "${output}" \
    --clip-count 4 \
    --target-duration 20 \
    --seed "$((710 + index))" \
    "${production_plan_args[@]}" \
    --planner-backend command \
    --planner-command "${planner_cmd}" \
    --plan-critic-command "${critic_cmd}" \
    --max-plan-revisions "${MAX_PLAN_REVISIONS}" \
    --review-backend mock \
    --audio-aligner none \
    --no-generate-audio
done

VALIDATION_ROOT="${VALIDATION_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["VALIDATION_ROOT"])
failed = False
for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    planner = json.loads((run_dir / "planner_agent_output.json").read_text())
    plan = json.loads((run_dir / "iterations" / "001" / "production_plan.json").read_text())
    scenes = plan["scenes"]
    lyric_by_index = {item["index"]: item["text"] for item in plan["lyric_segments"]}
    prompts = [scene["video_prompt"] for scene in scenes]
    distinct = len(set(prompts)) == len(prompts)
    lengths = [len(prompt) for prompt in prompts]
    relationships_complete = all(
        scene.get("narrative_function")
        and scene.get("relationship_kind")
        and scene.get("relationship_rationale")
        and (scene["scene_num"] == 1 or scene.get("relationship_change"))
        for scene in scenes
    )
    weak_changes = {"no change", "none", "unchanged", "same", "n/a", "na"}
    honest_repetition = all(
        not (
            current.get("relationship_kind") in {"continuation", "reprise"}
            and all(current.get(field) == previous.get(field) for field in ("setting", "subjects", "action", "camera"))
            and {str(value).strip().lower() for value in current.get("relationship_change", [])}.issubset(weak_changes)
        )
        for previous, current in zip(scenes, scenes[1:])
    )
    valid = (
        not planner["used_fallback"]
        and planner["validation_passed"]
        and bool(plan.get("arc_summary"))
        and relationships_complete
        and honest_repetition
        and len(scenes) == 4
        and distinct
        and all(300 <= length <= 900 for length in lengths)
    )
    failed = failed or not valid
    print(json.dumps({
        "run": run_dir.name,
        "valid": valid,
        "used_fallback": planner["used_fallback"],
        "validation_passed": planner["validation_passed"],
        "validation_issue_count": planner["validation_issue_count"],
        "scene_count": len(scenes),
        "distinct_prompts": distinct,
        "relationships_complete": relationships_complete,
        "honest_repetition": honest_repetition,
        "prompt_lengths": lengths,
        "lyrics": [lyric_by_index[scene["lyric_segment_index"]] for scene in scenes],
        "prompts": prompts,
    }, indent=2))
if failed:
    raise SystemExit("one or more planner outputs failed structural validation")
PY

echo "VALIDATION_ROOT=${VALIDATION_ROOT}"
