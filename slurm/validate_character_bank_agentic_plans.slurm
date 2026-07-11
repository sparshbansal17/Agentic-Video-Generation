#!/bin/bash
#SBATCH --job-name=validate-charbank-plans
#SBATCH --output=logs/validate_charbank_plans_%j.out
#SBATCH --error=logs/validate_charbank_plans_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:45:00

set -euo pipefail
umask 002

AGENTIC_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
VENV_DIR="${VENV_DIR:-${AGENTIC_DIR}/.venv}"
VALIDATION_ROOT="${VALIDATION_ROOT:-${AGENTIC_DIR}/results/character_bank_planner_validation_${SLURM_JOB_ID}}"
PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"

module load python
source "${VENV_DIR}/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${AGENTIC_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"
mkdir -p "${AGENTIC_DIR}/logs" "${VALIDATION_ROOT}" "${TMPDIR}"

CHARACTER_BANK="${VALIDATION_ROOT}/generated_character_bank.json"
cat > "${CHARACTER_BANK}" <<'JSON'
{
  "characters": [
    {
      "id": "garden_child",
      "role": "child",
      "description": "same preschool child in yellow raincoat with red boots and round curious face",
      "visual_anchors": ["yellow raincoat", "red boots", "round curious face"],
      "allowed_variants": ["hood up", "hood down", "holding a safe toy umbrella"],
      "negative_constraints": ["no text on coat", "no frightened expression"]
    },
    {
      "id": "clock_mouse",
      "role": "mouse",
      "description": "same tiny brown clock mouse with cream belly and soft rounded ears",
      "visual_anchors": ["tiny brown mouse", "cream belly", "rounded ears"],
      "allowed_variants": ["beside clock", "on clock base", "holding a crumb"],
      "negative_constraints": ["no sharp teeth", "no scary expression"]
    },
    {
      "id": "meadow_lamb",
      "role": "lamb",
      "description": "same small fluffy white lamb with blue ribbon collar and toy-like rounded face",
      "visual_anchors": ["fluffy white wool", "blue ribbon collar", "rounded face"],
      "allowed_variants": ["beside child", "near school path", "in meadow flowers"],
      "negative_constraints": ["no dirty wool", "no distressed pose"]
    },
    {
      "id": "sleepy_star",
      "role": "star",
      "description": "same small golden sleepy star with soft glow and rounded safe points",
      "visual_anchors": ["golden glow", "sleepy smile", "rounded safe points"],
      "allowed_variants": ["near window", "above rooftops", "over toy hills"],
      "negative_constraints": ["no harsh flare", "no sharp points"]
    },
    {
      "id": "boat_duck",
      "role": "animal_friend",
      "description": "same tiny plush yellow duck with orange scarf and gentle button-like eyes",
      "visual_anchors": ["yellow plush duck", "orange scarf", "button-like eyes"],
      "allowed_variants": ["in toy boat", "on riverbank", "beside paper boat"],
      "negative_constraints": ["no realistic wet feathers", "no text on scarf"]
    }
  ]
}
JSON

TOPICS=(
  "Mary Had a Little Lamb"
  "Rain, Rain, Go Away"
  "Hickory Dickory Dock"
  "Row, Row, Row Your Boat"
)
SLUGS=(mary_lamb rain_away hickory_dock row_boat)
LYRICS=(
  $'Mary had a little lamb\nIts fleece was white as snow\nAnd everywhere that Mary went\nThe lamb was sure to go\n'
  $'Rain, rain, go away\nCome again another day\nAll the children want to play\nRain, rain, go away\n'
  $'Hickory dickory dock\nThe mouse ran up the clock\nThe clock struck one\nThe mouse ran down\n'
  $'Row, row, row your boat\nGently down the stream\nMerrily, merrily, merrily, merrily\nLife is but a dream\n'
)

for index in "${!TOPICS[@]}"; do
  topic="${TOPICS[$index]}"
  slug="${SLUGS[$index]}"
  output="${VALIDATION_ROOT}/${slug}"
  lyrics_file="${output}/lyrics.txt"
  mkdir -p "${output}"
  printf "%s" "${LYRICS[$index]}" > "${lyrics_file}"
  planner_cmd="${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_lullaby_planner.py --model ${PLANNER_MODEL} --debug-output ${output}/planner_raw_output.txt"
  critic_cmd="${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_plan_critic.py --model ${PLANNER_MODEL} --debug-output ${output}/plan_critic_raw_output.txt"

  python -m storymem_agentic.cli plan \
    --topic "${topic}" \
    --lyrics-file "${lyrics_file}" \
    --output-dir "${output}" \
    --clip-count 4 \
    --target-duration 20 \
    --seed "$((820 + index))" \
    --character-bank "${CHARACTER_BANK}" \
    --planner-backend command \
    --planner-command "${planner_cmd}" \
    --plan-critic-command "${critic_cmd}" \
    --review-backend mock \
    --audio-aligner none \
    --no-generate-audio
done

VALIDATION_ROOT="${VALIDATION_ROOT}" python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["VALIDATION_ROOT"])
failed = False
for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    planner = json.loads((run_dir / "planner_agent_output.json").read_text())
    report = json.loads((run_dir / "plan_validation_report.json").read_text())
    characters = json.loads((run_dir / "character_bank_resolution.json").read_text())["characters"]
    plan = json.loads((run_dir / "iterations" / "001" / "production_plan.json").read_text())
    prompts = [scene["video_prompt"] for scene in plan["scenes"]]
    descriptions = " ".join(scene["description"] for scene in plan["scenes"]).lower()
    selected_descriptions = [
        character["description"].lower()
        for character in characters
        if character["description"].lower() in descriptions
    ]
    valid = (
        not planner["used_fallback"]
        and planner["validation_passed"]
        and report["passed"]
        and len(plan["scenes"]) == 4
        and len(set(prompts)) == 4
        and bool(selected_descriptions)
        and all(len(prompt) < 900 for prompt in prompts)
    )
    failed = failed or not valid
    print(json.dumps({
        "run": run_dir.name,
        "valid": valid,
        "used_fallback": planner["used_fallback"],
        "validation_passed": report["passed"],
        "issues": report["issues"],
        "selected_character_descriptions": selected_descriptions,
        "prompt_lengths": [len(prompt) for prompt in prompts],
    }, indent=2))
if failed:
    sys.exit("one or more character-bank planner validations failed")
PY

echo "VALIDATION_ROOT=${VALIDATION_ROOT}"
