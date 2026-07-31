#!/bin/bash
#SBATCH --job-name=baseline-plan-smoke
#SBATCH --output=logs/baseline_plan_smoke_%j.out
#SBATCH --error=logs/baseline_plan_smoke_%j.err
#SBATCH --partition=smallgpu
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
# The smallgpu partition allocates one of its two L40 GPUs per 64 CPU slots.
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00

set -euo pipefail
umask 002

ROOT="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
CASE_ID="${CASE_ID:-conditioned_twinkle}"
SYSTEM="${SYSTEM:?Set SYSTEM to automv or movieagent}"
REPO_ROOT="${REPO_ROOT:-${ROOT}/results_baselines/repos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results_baselines/plans}"

module load python
source "${ROOT}/.venv/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="${ROOT}/.hf-cache"
export TRANSFORMERS_CACHE="${ROOT}/.hf-cache/transformers"
export TMPDIR="${ROOT}/.tmp"
export MPLCONFIGDIR="${ROOT}/.tmp/matplotlib"
mkdir -p "${ROOT}/logs" "${TMPDIR}" "${MPLCONFIGDIR}"

if [ "${SYSTEM}" = "automv" ]; then
  EXTERNAL_REPO="${REPO_ROOT}/AutoMV"
else
  EXTERNAL_REPO="${REPO_ROOT}/MovieAgent"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${SYSTEM}/${CASE_ID}/seed_000"
EXTRA_ARGS=()
if [ "${REVISE_EXISTING:-0}" = "1" ]; then
  EXTRA_ARGS+=(--revise-existing)
fi
if [ "${RESUME_STAGES:-0}" = "1" ]; then
  EXTRA_ARGS+=(--resume-stages)
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/run_published_baseline_planner.py \
  --system "${SYSTEM}" \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --case-id "${CASE_ID}" \
  --external-repo "${EXTERNAL_REPO}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"

python -m storymem_agentic.benchmark score-plan \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --case-id "${CASE_ID}" \
  --plan "${OUTPUT_DIR}/storymem_story.json" \
  --provenance "${OUTPUT_DIR}/provenance.json" \
  --output "${OUTPUT_DIR}/planning_metrics.json"
