#!/bin/bash
#SBATCH --job-name=storymem-nonagentic-test
#SBATCH --output=logs/storymem_nonagentic_test_%j.out
#SBATCH --error=logs/storymem_nonagentic_test_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --time=01:30:00
#SBATCH --requeue

set -euo pipefail
umask 002

SCRIPT_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
UPSTREAM_STORYMEM_DIR="${UPSTREAM_STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"
cd "${SCRIPT_DIR}"

mkdir -p logs results .hf-cache .torch-cache .tmp .pip-cache

module load python

VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Python venv does not exist or is not executable: ${VENV_DIR}" >&2
  echo "Create it with: scripts/bootstrap_venv.sh" >&2
  exit 2
fi
source "${VENV_DIR}/bin/activate"

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${SCRIPT_DIR}/.torch-cache}"
export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRIPT_DIR}/.pip-cache}"

NGPUS="${NGPUS:-4}"
ULYSSES_SIZE="${ULYSSES_SIZE:-1}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
DISTRIBUTED_ARGS=(--nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}")

STORY_SCRIPT="${STORY_SCRIPT:-story/storybook_star_toddler_playroom_singalong.json}"
RUN_NAME="${RUN_NAME:-nonagentic_$(basename "${STORY_SCRIPT}" .json)_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-results/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_PATH}/pipeline.log}"

if [ -d "${SCRIPT_DIR}/models" ]; then
  DEFAULT_MODEL_ROOT="${SCRIPT_DIR}/models"
else
  DEFAULT_MODEL_ROOT="${UPSTREAM_STORYMEM_DIR}/models"
fi
T2V_MODEL_PATH="${T2V_MODEL_PATH:-${DEFAULT_MODEL_ROOT}/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${DEFAULT_MODEL_ROOT}/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-${DEFAULT_MODEL_ROOT}/StoryMem/Wan2.2-MI2V-A14B}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-41}"
MAX_MEMORY_SIZE="${MAX_MEMORY_SIZE:-10}"
SEED="${SEED:-0}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
KEYFRAME_MODE="${KEYFRAME_MODE:-simple}"

for REQUIRED_PATH in "${VENV_DIR}" "${T2V_MODEL_PATH}" "${I2V_MODEL_PATH}" "${LORA_WEIGHT_PATH}"; do
  if [ ! -e "${REQUIRED_PATH}" ]; then
    echo "Required path does not exist: ${REQUIRED_PATH}" >&2
    exit 2
  fi
done
if [ ! -x "${FFMPEG_BIN}" ]; then
  echo "ffmpeg is not executable: ${FFMPEG_BIN}" >&2
  exit 2
fi

# Default to a cheap generation smoke test. Set RUN_FULL_STORY=1 to continue
# into the non-agentic M2V and ffmpeg concat pass after the first shot.
RUN_FULL_STORY="${RUN_FULL_STORY:-0}"

COMMON_ARGS=(
  --story_script_path "${STORY_SCRIPT}"
  --t2v_model_path "${T2V_MODEL_PATH}"
  --i2v_model_path "${I2V_MODEL_PATH}"
  --lora_weight_path "${LORA_WEIGHT_PATH}"
  --size "${SIZE}"
  --frame_num "${FRAME_NUM}"
  --max_memory_size "${MAX_MEMORY_SIZE}"
  --output_dir "${OUTPUT_PATH}"
  --log_file "${LOG_FILE}"
  --seed "${SEED}"
  --ffmpeg_bin "${FFMPEG_BIN}"
  --keyframe_mode "${KEYFRAME_MODE}"
  --ulysses_size "${ULYSSES_SIZE}"
  --offload_model
  --lora_rank 128
  --mi2v
)

if [ "${NGPUS}" -gt 1 ]; then
  COMMON_ARGS+=(--dit_fsdp --t5_fsdp)
fi

mkdir -p "${OUTPUT_PATH}"

echo "Running non-agentic StoryMem first-shot test"
echo "story: ${STORY_SCRIPT}"
echo "output: ${OUTPUT_PATH}"
echo "distributed: nproc_per_node=${NGPUS}, ulysses_size=${ULYSSES_SIZE}"

torchrun "${DISTRIBUTED_ARGS[@]}" pipeline.py \
  "${COMMON_ARGS[@]}" \
  --t2v_first_shot

if [ "${RUN_FULL_STORY}" = "1" ]; then
  echo "First shot finished; running non-agentic full-story continuation"
  torchrun "${DISTRIBUTED_ARGS[@]}" pipeline.py \
    "${COMMON_ARGS[@]}"
else
  echo "First-shot smoke test finished. Set RUN_FULL_STORY=1 to run the M2V/concat continuation."
fi
