#!/bin/bash
#SBATCH --job-name=storymem-agentic-3clip
#SBATCH --output=logs/storymem_agentic_3clip_%j.out
#SBATCH --error=logs/storymem_agentic_3clip_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --time=01:30:00
#SBATCH --requeue

set -euo pipefail
umask 002

AGENTIC_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
STORYMEM_DIR="${STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"
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
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${AGENTIC_DIR}/.pip-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${AGENTIC_DIR}/.tmp/matplotlib}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

NGPUS="${NGPUS:-8}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
DISTRIBUTED_ARGS=(--nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}")

RUN_NAME="${RUN_NAME:-twinkle_agentic_3clip_8gpu_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${AGENTIC_DIR}/results/${RUN_NAME}}"
RHYME_FILE="${RHYME_FILE:-${AGENTIC_DIR}/story/twinkle_3clip_agentic_lyrics.txt}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-${STORYMEM_DIR}/models/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${STORYMEM_DIR}/models/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-${STORYMEM_DIR}/models/StoryMem/Wan2.2-MI2V-A14B}"
SIZE="${SIZE:-832*480}"
MAX_MEMORY_SIZE="${MAX_MEMORY_SIZE:-10}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
SEED="${SEED:-301}"

if [ -z "${ACE_STEP_CMD:-}" ]; then
  ACE_STEP_CMD="${STORYMEM_DIR}/audio_tools/.venv/acestep/bin/python ${AGENTIC_DIR}/run_acestep_story_audio.py --lyrics-file \${lyrics_file} --prompt-file \${prompt_file} --duration \${duration} --seed \${seed} --output \${output_file} --checkpoint-path ${STORYMEM_DIR}/audio_tools/ace_step_checkpoints"
fi

for REQUIRED_PATH in "${VENV_DIR}" "${STORYMEM_DIR}/pipeline.py" "${RHYME_FILE}" "${T2V_MODEL_PATH}" "${I2V_MODEL_PATH}" "${LORA_WEIGHT_PATH}"; do
  if [ ! -e "${REQUIRED_PATH}" ]; then
    echo "Required path does not exist: ${REQUIRED_PATH}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_PATH}"

echo "Running 3-clip full agentic StoryMem test"
echo "run_name: ${RUN_NAME}"
echo "output: ${OUTPUT_PATH}"
echo "rhyme: ${RHYME_FILE}"
echo "distributed: nproc_per_node=${NGPUS}, ulysses_size=${NGPUS}"

cd "${AGENTIC_DIR}"
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python -m storymem_agentic.cli dry-run \
  --rhyme-file "${RHYME_FILE}" \
  --output-dir "${OUTPUT_PATH}" \
  --target-duration 15 \
  --clip-count 3 \
  --max-iterations 1 \
  --seed "${SEED}"

STORY_SCRIPT="${OUTPUT_PATH}/iterations/001/story.json"
if [ ! -s "${STORY_SCRIPT}" ]; then
  echo "Agentic planner did not create StoryMem story JSON: ${STORY_SCRIPT}" >&2
  exit 2
fi
echo "generated_story: ${STORY_SCRIPT}"

cd "${STORYMEM_DIR}"
export PYTHONPATH="${STORYMEM_DIR}"

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

cd "${AGENTIC_DIR}"
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}"
FINAL_VIDEO="${OUTPUT_PATH}/${RUN_NAME}_subtitled.mp4"

python -m storymem_agentic.cli postprocess-audio \
  --rhyme-file "${RHYME_FILE}" \
  --story-json "${STORY_SCRIPT}" \
  --output-dir "${OUTPUT_PATH}" \
  --final-video "${FINAL_VIDEO}" \
  --mode full_song \
  --voice-backend ace_step_full_song \
  --music-backend ace_step_full_song \
  --media-audio-mode full_song \
  --audio-output-suffix _with_music \
  --audio-voice-style "warm clear toddler nursery singalong vocal, exact Twinkle lyrics, gentle adult lead with soft childlike brightness, music box, celesta, glockenspiel star twinkles, soft strings, light harp, slow 3/4 sway, vocals forward" \
  --ace-step-cmd "${ACE_STEP_CMD}" \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --seed "${SEED}"

AUDIO_VIDEO="${OUTPUT_PATH}/${RUN_NAME}_subtitled_with_music.mp4"
set +e
python scripts/evaluate_agentic_run.py \
  --output-dir "${OUTPUT_PATH}" \
  --final-video "${AUDIO_VIDEO}" \
  --expected-clips 3 \
  --ffmpeg-bin "${FFMPEG_BIN}"
EVAL_STATUS="$?"
set -e

if [ "${EVAL_STATUS}" -ne 0 ]; then
  echo "Evaluation requested iteration; see ${OUTPUT_PATH}/agentic_tuning_report.json" >&2
  exit "${EVAL_STATUS}"
fi

echo "Agentic 3-clip test accepted: ${AUDIO_VIDEO}"
