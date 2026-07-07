#!/bin/bash
#SBATCH --job-name=storymem-manual-twinkle-4gpu
#SBATCH --output=logs/storymem_manual_twinkle_4gpu_%j.out
#SBATCH --error=logs/storymem_manual_twinkle_4gpu_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
umask 002

AGENTIC_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
STORYMEM_DIR="${STORYMEM_DIR:-${AGENTIC_DIR}}"
VENV_DIR="${VENV_DIR:-${AGENTIC_DIR}/.venv}"

mkdir -p "${AGENTIC_DIR}/logs" "${AGENTIC_DIR}/results" "${AGENTIC_DIR}/.tmp"

module load python

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Python venv does not exist or is not executable: ${VENV_DIR}" >&2
  exit 2
fi

source "${VENV_DIR}/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${AGENTIC_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${AGENTIC_DIR}/.torch-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${AGENTIC_DIR}/.tmp/matplotlib}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

RUN_NAME="${RUN_NAME:-twinkle_manual_4gpu_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${AGENTIC_DIR}/results/${RUN_NAME}}"
STORY_SCRIPT="${STORY_SCRIPT:-${AGENTIC_DIR}/story/twinkle_manual_4clip_storymem.json}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/gautschi/bansa125/StoryMem/models}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-${MODEL_ROOT}/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${MODEL_ROOT}/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-${MODEL_ROOT}/StoryMem/Wan2.2-MI2V-A14B}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
NGPUS="${NGPUS:-4}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
DISTRIBUTED_ARGS=(--nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}")
SIZE="${SIZE:-832*480}"
MAX_MEMORY_SIZE="${MAX_MEMORY_SIZE:-10}"
SEED="${SEED:-402}"
export STORY_SCRIPT

for REQUIRED_PATH in \
  "${VENV_DIR}" \
  "${STORYMEM_DIR}/pipeline.py" \
  "${STORYMEM_DIR}/wan" \
  "${STORYMEM_DIR}/extract_keyframes.py" \
  "${STORY_SCRIPT}" \
  "${T2V_MODEL_PATH}" \
  "${I2V_MODEL_PATH}" \
  "${LORA_WEIGHT_PATH}"; do
  if [ ! -e "${REQUIRED_PATH}" ]; then
    echo "Required path does not exist: ${REQUIRED_PATH}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_PATH}"

cd "${AGENTIC_DIR}"
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import json
import os
from pathlib import Path

story_path = Path(os.environ["STORY_SCRIPT"])
story = json.loads(story_path.read_text(encoding="utf-8"))
assert isinstance(story.get("story_overview"), str) and story["story_overview"].strip()
scenes = story.get("scenes")
assert isinstance(scenes, list) and len(scenes) == 4
for expected, scene in enumerate(scenes, start=1):
    assert scene["scene_num"] == expected
    assert isinstance(scene.get("video_prompts"), list) and len(scene["video_prompts"]) == 1
    assert isinstance(scene.get("cut"), list) and len(scene["cut"]) == 1
    assert isinstance(scene["cut"][0], bool)
    assert scene["video_prompts"][0].strip()
print("validated_manual_storymem_json:", story_path)
print("scene_count:", len(scenes))
print("first_prompt:", scenes[0]["video_prompts"][0])
PY

cd "${STORYMEM_DIR}"
export PYTHONPATH="${STORYMEM_DIR}"

echo "Running hand-authored direct StoryMem Twinkle pipeline"
echo "job_id: ${SLURM_JOB_ID}"
echo "run_name: ${RUN_NAME}"
echo "output: ${OUTPUT_PATH}"
echo "story_script: ${STORY_SCRIPT}"
echo "gpus: ${NGPUS}"

torchrun "${DISTRIBUTED_ARGS[@]}" pipeline.py \
  --story_script_path "${STORY_SCRIPT}" \
  --t2v_model_path "${T2V_MODEL_PATH}" \
  --i2v_model_path "${I2V_MODEL_PATH}" \
  --lora_weight_path "${LORA_WEIGHT_PATH}" \
  --size "${SIZE}" \
  --max_memory_size "${MAX_MEMORY_SIZE}" \
  --output_dir "${OUTPUT_PATH}" \
  --ffmpeg_bin "${FFMPEG_BIN}" \
  --seed "${SEED}" \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size "${NGPUS}" \
  --lora_rank 128 \
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
  --seed "${SEED}" \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size "${NGPUS}" \
  --lora_rank 128 \
  --mi2v

echo "Hand-authored direct StoryMem Twinkle run completed"
echo "generated: ${OUTPUT_PATH}"
