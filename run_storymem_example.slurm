#!/bin/bash
#SBATCH --job-name=storymem-star
#SBATCH --output=/scratch/gautschi/bansa125/StoryMem/logs/storymem_star_%j.out
#SBATCH --error=/scratch/gautschi/bansa125/StoryMem/logs/storymem_star_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --time=12:00:00

set -euo pipefail
umask 002

SCRIPT_DIR="${STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"
cd "${SCRIPT_DIR}"

mkdir -p logs results

module load conda/2026.03

CONDA_ENV="${CONDA_ENV:-${SCRIPT_DIR}/.conda/storymem}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Keep caches in the project/scratch area instead of filling $HOME.
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${SCRIPT_DIR}/.torch-cache}"
export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRIPT_DIR}/.pip-cache}"

NGPUS="${NGPUS:-8}"
MASTER_PORT="${MASTER_PORT:-9999}"
DISTRIBUTED_ARGS=(--nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}")

STORY_SCRIPT="${STORY_SCRIPT:-./story/storybook_star.json}"
OUTPUT_PATH="${OUTPUT_PATH:-results/storybook_star_${SLURM_JOB_ID}}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-./models/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-./models/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-./models/StoryMem/Wan2.2-MI2V-A14B}"
SIZE="${SIZE:-832*480}"
MAX_MEMORY_SIZE="${MAX_MEMORY_SIZE:-10}"
ENABLE_AUDIO="${ENABLE_AUDIO:-0}"
AUDIO_MODE="${AUDIO_MODE:-full_song}"
VOCAL_BACKEND="${VOCAL_BACKEND:-ace_step}"
BACKING_BACKEND="${BACKING_BACKEND:-ace_step}"
AUDIO_VOICE_STYLE="${AUDIO_VOICE_STYLE:-gentle adult lullaby}"
AUDIO_OUTPUT_SUFFIX="${AUDIO_OUTPUT_SUFFIX:-_with_music}"
AUDIO_SKIP_ON_ERROR="${AUDIO_SKIP_ON_ERROR:-1}"
AUDIO_DRY_RUN="${AUDIO_DRY_RUN:-0}"
LYRICS_MODEL="${LYRICS_MODEL:-}"
ACE_STEP_CMD="${ACE_STEP_CMD:-}"
VOCAL_CMD="${VOCAL_CMD:-}"
BACKING_CMD="${BACKING_CMD:-}"
MUSICGEN_CMD="${MUSICGEN_CMD:-}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"

if [ -z "${ACE_STEP_CMD}" ]; then
  ACE_STEP_CMD="${SCRIPT_DIR}/audio_tools/.venv/acestep/bin/python ${SCRIPT_DIR}/run_acestep_story_audio.py --lyrics-file \${lyrics_file} --prompt-file \${prompt_file} --duration \${duration} --seed \${seed} --output \${output_file} --checkpoint-path ${SCRIPT_DIR}/audio_tools/ace_step_checkpoints"
fi

AUDIO_ARGS=()
if [ "${ENABLE_AUDIO}" = "1" ]; then
  AUDIO_ARGS+=(--enable_audio)
  AUDIO_ARGS+=(--audio_mode "${AUDIO_MODE}")
  AUDIO_ARGS+=(--vocal_backend "${VOCAL_BACKEND}")
  AUDIO_ARGS+=(--backing_backend "${BACKING_BACKEND}")
  AUDIO_ARGS+=(--audio_voice_style "${AUDIO_VOICE_STYLE}")
  AUDIO_ARGS+=(--audio_output_suffix "${AUDIO_OUTPUT_SUFFIX}")
  if [ "${AUDIO_SKIP_ON_ERROR}" = "1" ]; then
    AUDIO_ARGS+=(--audio_skip_on_error)
  fi
  if [ "${AUDIO_DRY_RUN}" = "1" ]; then
    AUDIO_ARGS+=(--audio_dry_run)
  fi
  if [ -n "${LYRICS_MODEL}" ]; then
    AUDIO_ARGS+=(--lyrics_model "${LYRICS_MODEL}")
  fi
  if [ -n "${ACE_STEP_CMD}" ]; then
    AUDIO_ARGS+=(--ace_step_cmd "${ACE_STEP_CMD}")
  fi
  if [ -n "${VOCAL_CMD}" ]; then
    AUDIO_ARGS+=(--vocal_cmd "${VOCAL_CMD}")
  fi
  if [ -n "${BACKING_CMD}" ]; then
    AUDIO_ARGS+=(--backing_cmd "${BACKING_CMD}")
  fi
  if [ -n "${MUSICGEN_CMD}" ]; then
    AUDIO_ARGS+=(--musicgen_cmd "${MUSICGEN_CMD}")
  fi
  if [ -n "${FFMPEG_BIN}" ]; then
    AUDIO_ARGS+=(--ffmpeg_bin "${FFMPEG_BIN}")
  fi
fi

mkdir -p "${OUTPUT_PATH}"

torchrun "${DISTRIBUTED_ARGS[@]}" pipeline.py \
  --story_script_path "${STORY_SCRIPT}" \
  --t2v_model_path "${T2V_MODEL_PATH}" \
  --i2v_model_path "${I2V_MODEL_PATH}" \
  --lora_weight_path "${LORA_WEIGHT_PATH}" \
  --size "${SIZE}" \
  --max_memory_size "${MAX_MEMORY_SIZE}" \
  --output_dir "${OUTPUT_PATH}" \
  --ffmpeg_bin "${FFMPEG_BIN}" \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size "${NGPUS}" \
  --offload_model \
  --lora_rank 128 \
  --mi2v \
  --t2v_first_shot

torchrun "${DISTRIBUTED_ARGS[@]}" pipeline.py \
  --story_script_path "${STORY_SCRIPT}" \
  --t2v_model_path "${T2V_MODEL_PATH}" \
  --i2v_model_path "${I2V_MODEL_PATH}" \
  --lora_weight_path "${LORA_WEIGHT_PATH}" \
  --size "${SIZE}" \
  --max_memory_size "${MAX_MEMORY_SIZE}" \
  --output_dir "${OUTPUT_PATH}" \
  --ffmpeg_bin "${FFMPEG_BIN}" \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size "${NGPUS}" \
  --offload_model \
  --lora_rank 128 \
  --mi2v \
  "${AUDIO_ARGS[@]}"
