#!/bin/bash
#SBATCH --job-name=agentic-benchmark-plan
#SBATCH --output=logs/agentic_benchmark_plan_%j.out
#SBATCH --error=logs/agentic_benchmark_plan_%j.err
#SBATCH --partition=highmem
#SBATCH --qos=normal
#SBATCH --account=aarc
#SBATCH --cpus-per-task=96
#SBATCH --time=03:00:00

set -euo pipefail
umask 002

ROOT="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
CASE_ID="${CASE_ID:-conditioned_twinkle}"
RUN_DIR="${ROOT}/results_baselines/native_runs/storymem_agentic/${CASE_ID}/seed_000"
OUTPUT_DIR="${ROOT}/results_baselines/plans/storymem_agentic/${CASE_ID}/seed_000"

module load python
source "${ROOT}/.venv/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="${ROOT}/.hf-cache"
export TRANSFORMERS_CACHE="${ROOT}/.hf-cache/transformers"
export TMPDIR="${ROOT}/.tmp"
export MPLCONFIGDIR="${ROOT}/.tmp/matplotlib"
mkdir -p "${ROOT}/logs" "${RUN_DIR}" "${OUTPUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/run_agentic_benchmark_planner.py \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --case-id "${CASE_ID}" \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}"

python -m storymem_agentic.benchmark score-plan \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --case-id "${CASE_ID}" \
  --plan "${OUTPUT_DIR}/storymem_story.json" \
  --provenance "${OUTPUT_DIR}/provenance.json" \
  --output "${OUTPUT_DIR}/planning_metrics.json"
