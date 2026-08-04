#!/bin/bash
#SBATCH --job-name=advanced-plan-metrics-ai
#SBATCH --output=logs/advanced_plan_metrics_%j.out
#SBATCH --error=logs/advanced_plan_metrics_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=01:00:00
#SBATCH --requeue

set -euo pipefail
umask 002

ROOT="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
CASE_ID="${CASE_ID:-conditioned_twinkle}"
SEED_DIR="${SEED_DIR:-seed_000}"
QWEN_MODEL="${QWEN_MODEL:-${ROOT}/.hf-cache/transformers/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac}"

module load python
source "${ROOT}/.venv/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}/src:${ROOT}"
export HF_HOME="${ROOT}/.hf-cache"
export TMPDIR="${ROOT}/.tmp"
export MPLCONFIGDIR="${ROOT}/.tmp/matplotlib"
mkdir -p "${ROOT}/logs" "${TMPDIR}" "${MPLCONFIGDIR}"

candidate_args=()
for system in storymem_agentic automv movieagent; do
  plan="${ROOT}/results_baselines/plans/${system}/${CASE_ID}/${SEED_DIR}/storymem_story.json"
  candidate_args+=(--candidate "${system}=${plan}")
done

python -m storymem_agentic.benchmark score-plan-vlm-panel \
  --manifest "${ROOT}/benchmarks/agentic_av_v1/manifest.json" \
  --case-id "${CASE_ID}" \
  "${candidate_args[@]}" \
  --output "${ROOT}/results_baselines/reports/${CASE_ID}_${SEED_DIR}/plan_panel_report.json" \
  --per-system-output-root "${ROOT}/results_baselines/plans" \
  --seed-dir "${SEED_DIR}" \
  --model-path "${QWEN_MODEL}"
