#!/bin/bash
#SBATCH --job-name=advanced-av-metrics-ai
#SBATCH --output=logs/advanced_av_metrics_%j.out
#SBATCH --error=logs/advanced_av_metrics_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
umask 002

ROOT="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
CASE_ID="${CASE_ID:-conditioned_twinkle}"
SEED_DIR="${SEED_DIR:-seed_000}"
SYSTEMS="${SYSTEMS:-storymem_agentic automv movieagent}"
RUN_INDEPENDENT_VLM_DIAGNOSTIC="${RUN_INDEPENDENT_VLM_DIAGNOSTIC:-0}"
CLIP_CACHE="${CLIP_CACHE:-/home/bansa125/.cache/clip}"
DINO_CACHE="${DINO_CACHE:-${ROOT}/.hf-cache/transformers}"
QWEN_MODEL="${QWEN_MODEL:-${ROOT}/.hf-cache/transformers/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac}"

module load python
source "${ROOT}/.venv/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}/src:${ROOT}"
export HF_HOME="${ROOT}/.hf-cache"
export TRANSFORMERS_CACHE="${DINO_CACHE}"
export TORCH_HOME="${ROOT}/.torch-cache"
export TMPDIR="${ROOT}/.tmp"
export MPLCONFIGDIR="${ROOT}/.tmp/matplotlib"
mkdir -p "${ROOT}/logs" "${TMPDIR}" "${MPLCONFIGDIR}"

if [ ! -d "${QWEN_MODEL}" ]; then
  echo "Missing cached Qwen2-VL model: ${QWEN_MODEL}" >&2
  exit 2
fi

for system in ${SYSTEMS}; do
  video="${ROOT}/results_baselines/raw_renders/${system}/${CASE_ID}/${SEED_DIR}/${SEED_DIR}.mp4"
  output_dir="${ROOT}/results_baselines/submissions/${system}/${CASE_ID}/${SEED_DIR}"
  if [ ! -s "${video}" ]; then
    echo "Missing raw benchmark video: ${video}" >&2
    exit 2
  fi
  mkdir -p "${output_dir}"
  python -m storymem_agentic.benchmark score-media-advanced \
    --manifest "${ROOT}/benchmarks/agentic_av_v1/manifest.json" \
    --case-id "${CASE_ID}" \
    --video "${video}" \
    --output "${output_dir}/advanced_media_metrics.json" \
    --device cuda \
    --clip-cache "${CLIP_CACHE}" \
    --dino-cache "${DINO_CACHE}" \
    --frames-per-scene 8
  if [ "${RUN_INDEPENDENT_VLM_DIAGNOSTIC}" = "1" ]; then
    python -m storymem_agentic.benchmark score-media-vlm \
      --manifest "${ROOT}/benchmarks/agentic_av_v1/manifest.json" \
      --case-id "${CASE_ID}" \
      --video "${video}" \
      --output "${output_dir}/vlm_media_metrics.json" \
      --model-path "${QWEN_MODEL}" \
      --mosaic-output "${output_dir}/vlm_contact_sheet.png"
  fi
done
