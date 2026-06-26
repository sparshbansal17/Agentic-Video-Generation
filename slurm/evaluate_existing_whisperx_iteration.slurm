#!/bin/bash
#SBATCH --job-name=agentic-whisperx-eval
#SBATCH --output=logs/agentic_whisperx_eval_%j.out
#SBATCH --error=logs/agentic_whisperx_eval_%j.err
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
ITERATION_DIR="${ITERATION_DIR:?Set ITERATION_DIR to an existing agentic iteration directory}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"

mkdir -p "${AGENTIC_DIR}/logs" "${AGENTIC_DIR}/.tmp"

module load python
source "${VENV_DIR}/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${AGENTIC_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${AGENTIC_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${AGENTIC_DIR}/.torch-cache}"
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"

WHISPERX_BIN="${WHISPERX_BIN:-${AGENTIC_DIR}/.venv-whisperx/bin/whisperx}"
if [ -z "${WHISPERX_CMD:-}" ]; then
  WHISPERX_CMD="${WHISPERX_BIN} \${audio_file} --model small --language en --device cuda --compute_type float16 --vad_method silero --output_format json --output_dir \${output_dir}"
fi
REVIEWER_MODEL="${REVIEWER_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
if [ -z "${VLM_CMD:-}" ]; then
  VLM_CMD="${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_lullaby_reviewer.py --model ${REVIEWER_MODEL} --debug-output ${ITERATION_DIR}/vlm_review_raw_output.txt"
fi

echo "Evaluating existing iteration with fresh WhisperX"
echo "iteration_dir: ${ITERATION_DIR}"
echo "whisperx_bin: ${WHISPERX_BIN}"
echo "reviewer_model: ${REVIEWER_MODEL}"

cd "${AGENTIC_DIR}"
python scripts/evaluate_existing_whisperx_iteration.py \
  --iteration-dir "${ITERATION_DIR}" \
  --whisperx-command "${WHISPERX_CMD}" \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --review-backend command \
  --vlm-command "${VLM_CMD}" \
  --exit-zero
