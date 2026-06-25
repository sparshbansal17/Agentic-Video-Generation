#!/bin/bash
#SBATCH --job-name=storymem-parent-flash
#SBATCH --output=logs/storymem_parent_flash_%j.out
#SBATCH --error=logs/storymem_parent_flash_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --time=01:30:00

set -euo pipefail
umask 002

AGENTIC_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
STORYMEM_DIR="${STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"
VENV_DIR="${VENV_DIR:-${AGENTIC_DIR}/.venv}"

mkdir -p "${AGENTIC_DIR}/logs" "${AGENTIC_DIR}/results"
cd "${STORYMEM_DIR}"

module load python

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Python venv does not exist or is not executable: ${VENV_DIR}" >&2
  exit 2
fi
source "${VENV_DIR}/bin/activate"

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${STORYMEM_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${AGENTIC_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${AGENTIC_DIR}/.torch-cache}"
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${AGENTIC_DIR}/.pip-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${AGENTIC_DIR}/.tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

NGPUS="${NGPUS:-4}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
DISTRIBUTED_ARGS=(--nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}")

DEFAULT_STORY_SCRIPT="${AGENTIC_DIR}/story/storybook_star_toddler_playroom_singalong.json"
STORY_SCRIPT="${STORY_SCRIPT:-${DEFAULT_STORY_SCRIPT}}"
RUN_NAME="${RUN_NAME:-twinkle_parent_flash_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${AGENTIC_DIR}/results/${RUN_NAME}}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-${STORYMEM_DIR}/models/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${STORYMEM_DIR}/models/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-${STORYMEM_DIR}/models/StoryMem/Wan2.2-MI2V-A14B}"
SIZE="${SIZE:-832*480}"
MAX_MEMORY_SIZE="${MAX_MEMORY_SIZE:-10}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"

for REQUIRED_PATH in "${VENV_DIR}" "${STORYMEM_DIR}/pipeline.py" "${STORY_SCRIPT}" "${T2V_MODEL_PATH}" "${I2V_MODEL_PATH}" "${LORA_WEIGHT_PATH}"; do
  if [ ! -e "${REQUIRED_PATH}" ]; then
    echo "Required path does not exist: ${REQUIRED_PATH}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_PATH}"

echo "Running parent StoryMem non-agentic pipeline with FlashAttention"
echo "storymem_dir: ${STORYMEM_DIR}"
echo "story: ${STORY_SCRIPT}"
echo "output: ${OUTPUT_PATH}"
echo "distributed: nproc_per_node=${NGPUS}, ulysses_size=${NGPUS}"
python - <<'PY'
import flash_attn
import wan.modules.attention as attention
print("flash_attn:", getattr(flash_attn, "__version__", "ok"))
print("flash_attn_available:", attention.FLASH_ATTN_2_AVAILABLE, attention.FLASH_ATTN_3_AVAILABLE)
PY

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
  --mi2v
