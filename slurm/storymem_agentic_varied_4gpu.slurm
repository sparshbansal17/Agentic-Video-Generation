#!/bin/bash
#SBATCH --job-name=storymem-varied-4gpu
#SBATCH --output=logs/storymem_varied_4gpu_%j.out
#SBATCH --error=logs/storymem_varied_4gpu_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
umask 002

: "${TOPIC:?Set TOPIC to the nursery rhyme name}"
: "${RUN_SLUG:?Set RUN_SLUG to a filesystem-safe run name}"
: "${LYRICS_FILE:?Set LYRICS_FILE to the canonical lyric text file}"

AGENTIC_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
STORYMEM_DIR="${STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"
VENV_DIR="${VENV_DIR:-${AGENTIC_DIR}/.venv}"
OUTPUT_PATH="${OUTPUT_PATH:-${AGENTIC_DIR}/results/${RUN_SLUG}_agentic_4gpu_${SLURM_JOB_ID}}"
MODEL_ROOT="${MODEL_ROOT:-${STORYMEM_DIR}/models}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-${MODEL_ROOT}/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${MODEL_ROOT}/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-${MODEL_ROOT}/StoryMem/Wan2.2-MI2V-A14B}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
REVIEWER_MODEL="${REVIEWER_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
SEED="${SEED:-720}"
CHARACTER_BANK="${CHARACTER_BANK:-}"
FINALIZED_PLAN="${FINALIZED_PLAN:-}"
EXECUTE_VIDEO="${EXECUTE_VIDEO:-1}"
MEDIA_AUDIO_MODE="${MEDIA_AUDIO_MODE:-full_song}"
CHARACTER_BANK_ARGS=()
if [ -n "${CHARACTER_BANK}" ]; then
  CHARACTER_BANK_ARGS=(--character-bank "${CHARACTER_BANK}")
fi
FINALIZED_PLAN_ARGS=()
if [ -n "${FINALIZED_PLAN}" ]; then
  FINALIZED_PLAN_ARGS=(--production-plan "${FINALIZED_PLAN}")
fi
VIDEO_EXECUTION_ARGS=(--execute-video)
if [ "${EXECUTE_VIDEO}" = "0" ]; then
  VIDEO_EXECUTION_ARGS=(--no-execute-video)
fi

module load python
source "${VENV_DIR}/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${AGENTIC_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${AGENTIC_DIR}/.torch-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"
mkdir -p "${AGENTIC_DIR}/logs" "${OUTPUT_PATH}" "${TMPDIR}"

PLANNER_CMD="${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_lullaby_planner.py --model ${PLANNER_MODEL} --debug-output ${OUTPUT_PATH}/planner_raw_output.txt"
PLAN_CRITIC_CMD="${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_plan_critic.py --model ${PLANNER_MODEL} --debug-output ${OUTPUT_PATH}/plan_critic_raw_output.txt"
VLM_CMD="${VENV_DIR}/bin/python ${AGENTIC_DIR}/scripts/local_lullaby_reviewer.py --model ${REVIEWER_MODEL} --debug-output ${OUTPUT_PATH}/vlm_review_raw_output.txt"
ACE_STEP_CMD="${STORYMEM_DIR}/audio_tools/.venv/acestep/bin/python ${AGENTIC_DIR}/run_acestep_story_audio.py --lyrics-file \${lyrics_file} --prompt-file \${prompt_file} --duration \${duration} --seed \${seed} --output \${output_file} --checkpoint-path ${STORYMEM_DIR}/audio_tools/ace_step_checkpoints"
WHISPERX_BIN="${WHISPERX_BIN:-${AGENTIC_DIR}/.venv-whisperx/bin/whisperx}"
WHISPERX_CMD="${WHISPERX_BIN} \${audio_file} --model small --language en --device cuda --compute_type float16 --vad_method silero --output_format json --output_dir \${output_dir}"

for path in \
  "${STORYMEM_DIR}/pipeline.py" \
  "${T2V_MODEL_PATH}" \
  "${I2V_MODEL_PATH}" \
  "${LORA_WEIGHT_PATH}" \
  "${WHISPERX_BIN}" \
  "${STORYMEM_DIR}/audio_tools/.venv/acestep/bin/python"; do
  if [ ! -e "${path}" ]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

python -m storymem_agentic.cli iterate \
  --topic "${TOPIC}" \
  --lyrics-file "${LYRICS_FILE}" \
  --output-dir "${OUTPUT_PATH}" \
  --target-duration 20 \
  --clip-count 4 \
  --max-iterations 2 \
  --seed "${SEED}" \
  "${CHARACTER_BANK_ARGS[@]}" \
  "${FINALIZED_PLAN_ARGS[@]}" \
  --storymem-dir "${STORYMEM_DIR}" \
  --t2v-model-path "${T2V_MODEL_PATH}" \
  --i2v-model-path "${I2V_MODEL_PATH}" \
  --lora-weight-path "${LORA_WEIGHT_PATH}" \
  --nproc-per-node 4 \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --planner-backend command \
  --planner-command "${PLANNER_CMD}" \
  --plan-critic-command "${PLAN_CRITIC_CMD}" \
  --review-backend command \
  --vlm-command "${VLM_CMD}" \
  --audio-aligner whisperx \
  --whisperx-command "${WHISPERX_CMD}" \
  --generate-audio \
  --media-audio-mode "${MEDIA_AUDIO_MODE}" \
  --audio-output-suffix _with_music \
  --ace-step-cmd "${ACE_STEP_CMD}" \
  --strict-lullaby-review \
  "${VIDEO_EXECUTION_ARGS[@]}"

echo "OUTPUT_PATH=${OUTPUT_PATH}"
