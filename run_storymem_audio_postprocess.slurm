#!/bin/bash
#SBATCH --job-name=storymem-audio
#SBATCH --output=/scratch/gautschi/bansa125/StoryMem/logs/storymem_audio_%j.out
#SBATCH --error=/scratch/gautschi/bansa125/StoryMem/logs/storymem_audio_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00

set -euo pipefail
umask 002

SCRIPT_DIR="${STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"
cd "${SCRIPT_DIR}"

mkdir -p logs

module load conda/2026.03

CONDA_ENV="${CONDA_ENV:-${SCRIPT_DIR}/.conda/storymem}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${SCRIPT_DIR}/.torch-cache}"
export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRIPT_DIR}/.pip-cache}"

STORY_SCRIPT="${STORY_SCRIPT:-./story/storybook_star.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/storybook_star_11662274}"
FINAL_VIDEO="${FINAL_VIDEO:-}"
AUDIO_MODE="${AUDIO_MODE:-full_song}"
VOCAL_BACKEND="${VOCAL_BACKEND:-ace_step}"
BACKING_BACKEND="${BACKING_BACKEND:-ace_step}"
AUDIO_VOICE_STYLE="${AUDIO_VOICE_STYLE:-gentle adult lullaby}"
AUDIO_OUTPUT_SUFFIX="${AUDIO_OUTPUT_SUFFIX:-_with_music}"
AUDIO_SKIP_ON_ERROR="${AUDIO_SKIP_ON_ERROR:-0}"
AUDIO_DRY_RUN="${AUDIO_DRY_RUN:-0}"
LYRICS_MODEL="${LYRICS_MODEL:-}"
LYRICS_FILE="${LYRICS_FILE:-}"
ACE_STEP_CMD="${ACE_STEP_CMD:-}"
VOCAL_CMD="${VOCAL_CMD:-}"
BACKING_CMD="${BACKING_CMD:-}"
MUSICGEN_CMD="${MUSICGEN_CMD:-}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
SEED="${SEED:-0}"

if [ -z "${ACE_STEP_CMD}" ]; then
  ACE_STEP_CMD="${SCRIPT_DIR}/audio_tools/.venv/acestep/bin/python ${SCRIPT_DIR}/run_acestep_story_audio.py --lyrics-file \${lyrics_file} --prompt-file \${prompt_file} --duration \${duration} --seed \${seed} --output \${output_file} --checkpoint-path ${SCRIPT_DIR}/audio_tools/ace_step_checkpoints"
fi

ARGS=(
  --story_script_path "${STORY_SCRIPT}"
  --output_dir "${OUTPUT_DIR}"
  --audio_mode "${AUDIO_MODE}"
  --vocal_backend "${VOCAL_BACKEND}"
  --backing_backend "${BACKING_BACKEND}"
  --audio_voice_style "${AUDIO_VOICE_STYLE}"
  --audio_output_suffix "${AUDIO_OUTPUT_SUFFIX}"
  --seed "${SEED}"
)

if [ -n "${FINAL_VIDEO}" ]; then
  ARGS+=(--final_video "${FINAL_VIDEO}")
fi
if [ "${AUDIO_SKIP_ON_ERROR}" = "1" ]; then
  ARGS+=(--audio_skip_on_error)
fi
if [ "${AUDIO_DRY_RUN}" = "1" ]; then
  ARGS+=(--audio_dry_run)
fi
if [ -n "${LYRICS_MODEL}" ]; then
  ARGS+=(--lyrics_model "${LYRICS_MODEL}")
fi
if [ -n "${LYRICS_FILE}" ]; then
  ARGS+=(--lyrics_file "${LYRICS_FILE}")
fi
if [ -n "${ACE_STEP_CMD}" ]; then
  ARGS+=(--ace_step_cmd "${ACE_STEP_CMD}")
fi
if [ -n "${VOCAL_CMD}" ]; then
  ARGS+=(--vocal_cmd "${VOCAL_CMD}")
fi
if [ -n "${BACKING_CMD}" ]; then
  ARGS+=(--backing_cmd "${BACKING_CMD}")
fi
if [ -n "${MUSICGEN_CMD}" ]; then
  ARGS+=(--musicgen_cmd "${MUSICGEN_CMD}")
fi
if [ -n "${FFMPEG_BIN}" ]; then
  ARGS+=(--ffmpeg_bin "${FFMPEG_BIN}")
fi

python story_audio.py "${ARGS[@]}"
