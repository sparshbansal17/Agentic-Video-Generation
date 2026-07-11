#!/bin/bash
#SBATCH --job-name=storymem-agentic-video-4gpu
#SBATCH --output=logs/storymem_agentic_video_4gpu_%j.out
#SBATCH --error=logs/storymem_agentic_video_4gpu_%j.err
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

RUN_NAME="${RUN_NAME:-twinkle_video_only_4gpu_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${AGENTIC_DIR}/results/${RUN_NAME}}"
TOPIC="${TOPIC:-Twinkle Twinkle Little Star}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/gautschi/bansa125/StoryMem/models}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-${MODEL_ROOT}/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${MODEL_ROOT}/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-${MODEL_ROOT}/StoryMem/Wan2.2-MI2V-A14B}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
NGPUS="${NGPUS:-4}"
SEED="${SEED:-401}"
CLIP_COUNT="${CLIP_COUNT:-3}"
TARGET_DURATION="${TARGET_DURATION:-15}"
PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
PLANNER_CMD="${PLANNER_CMD:-${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_lullaby_planner.py --model ${PLANNER_MODEL} --debug-output ${OUTPUT_PATH}/planner_raw_output.txt}"
PLAN_CRITIC_CMD="${PLAN_CRITIC_CMD:-${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_plan_critic.py --model ${PLANNER_MODEL} --debug-output ${OUTPUT_PATH}/plan_critic_raw_output.txt}"
export OUTPUT_PATH

for REQUIRED_PATH in \
  "${VENV_DIR}" \
  "${STORYMEM_DIR}/pipeline.py" \
  "${STORYMEM_DIR}/wan" \
  "${STORYMEM_DIR}/extract_keyframes.py" \
  "${T2V_MODEL_PATH}" \
  "${I2V_MODEL_PATH}" \
  "${LORA_WEIGHT_PATH}" \
  "${AGENTIC_DIR}/scripts/local_lullaby_planner.py"; do
  if [ ! -e "${REQUIRED_PATH}" ]; then
    echo "Required path does not exist: ${REQUIRED_PATH}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_PATH}"

cd "${AGENTIC_DIR}"
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Running 4-GPU video-only agentic StoryMem test"
echo "job_id: ${SLURM_JOB_ID}"
echo "run_name: ${RUN_NAME}"
echo "output: ${OUTPUT_PATH}"
echo "topic: ${TOPIC}"
echo "storymem_dir: ${STORYMEM_DIR}"
echo "planner_model: ${PLANNER_MODEL}"
echo "gpus: ${NGPUS}"
echo "clip_count: ${CLIP_COUNT}"
echo "target_duration: ${TARGET_DURATION}"
echo "t2v_model_path: ${T2V_MODEL_PATH}"
echo "i2v_model_path: ${I2V_MODEL_PATH}"
echo "lora_weight_path: ${LORA_WEIGHT_PATH}"

python -m storymem_agentic.cli generate \
  --topic "${TOPIC}" \
  --output-dir "${OUTPUT_PATH}" \
  --target-duration "${TARGET_DURATION}" \
  --clip-count "${CLIP_COUNT}" \
  --max-iterations 1 \
  --seed "${SEED}" \
  --storymem-dir "${STORYMEM_DIR}" \
  --t2v-model-path "${T2V_MODEL_PATH}" \
  --i2v-model-path "${I2V_MODEL_PATH}" \
  --lora-weight-path "${LORA_WEIGHT_PATH}" \
  --nproc-per-node "${NGPUS}" \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --planner-backend command \
  --planner-command "${PLANNER_CMD}" \
  --plan-critic-command "${PLAN_CRITIC_CMD}" \
  --review-backend mock \
  --audio-aligner none \
  --no-generate-audio \
  --execute-video

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_PATH"]) / "iterations" / "001"
storymem = json.loads((root / "storymem_story.json").read_text(encoding="utf-8"))
commands = json.loads((root / "storymem_commands.json").read_text(encoding="utf-8"))["commands"]
print("storymem_story:", root / "storymem_story.json")
print("storymem_top_keys:", sorted(storymem))
print("scene_count:", len(storymem["scenes"]))
print("scene_keys:", sorted(storymem["scenes"][0]))
print("first_command_script:", Path(commands[0][commands[0].index("--story_script_path") + 1]).name)
print("second_command_script:", Path(commands[1][commands[1].index("--story_script_path") + 1]).name)
print("first_prompt:", storymem["scenes"][0]["video_prompts"][0])
PY

echo "4-GPU video-only agentic StoryMem test completed"
echo "iteration: ${OUTPUT_PATH}/iterations/001"
echo "generated: ${OUTPUT_PATH}/iterations/001/generated"
