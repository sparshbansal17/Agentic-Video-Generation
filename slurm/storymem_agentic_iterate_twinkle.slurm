#!/bin/bash
#SBATCH --job-name=storymem-agentic-iterate-twinkle
#SBATCH --output=logs/storymem_agentic_iterate_twinkle_%j.out
#SBATCH --error=logs/storymem_agentic_iterate_twinkle_%j.err
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-${AGENTIC_DIR}/.tmp/matplotlib}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

RUN_NAME="${RUN_NAME:-twinkle_agentic_iterate_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${AGENTIC_DIR}/results/${RUN_NAME}}"
LYRICS_FILE="${LYRICS_FILE:-${AGENTIC_DIR}/story/twinkle_3clip_agentic_lyrics.txt}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-${STORYMEM_DIR}/models/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${STORYMEM_DIR}/models/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-${STORYMEM_DIR}/models/StoryMem/Wan2.2-MI2V-A14B}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
NGPUS="${NGPUS:-8}"
SEED="${SEED:-301}"

if [ -z "${ACE_STEP_CMD:-}" ]; then
  ACE_STEP_CMD="${STORYMEM_DIR}/audio_tools/.venv/acestep/bin/python ${AGENTIC_DIR}/run_acestep_story_audio.py --lyrics-file \${lyrics_file} --prompt-file \${prompt_file} --duration \${duration} --seed \${seed} --output \${output_file} --checkpoint-path ${STORYMEM_DIR}/audio_tools/ace_step_checkpoints"
fi

WHISPERX_BIN="${WHISPERX_BIN:-}"
if [ -z "${WHISPERX_BIN}" ]; then
  if [ -x "${AGENTIC_DIR}/.venv-whisperx/bin/whisperx" ]; then
    WHISPERX_BIN="${AGENTIC_DIR}/.venv-whisperx/bin/whisperx"
  elif [ -x "${VENV_DIR}/bin/whisperx" ]; then
    WHISPERX_BIN="${VENV_DIR}/bin/whisperx"
  elif command -v whisperx >/dev/null 2>&1; then
    WHISPERX_BIN="$(command -v whisperx)"
  fi
fi

if [ -z "${WHISPERX_CMD:-}" ] && [ -n "${WHISPERX_BIN}" ]; then
  WHISPERX_CMD="${WHISPERX_BIN} \${audio_file} --model small --language en --device cuda --compute_type float16 --vad_method silero --output_format json --output_dir \${output_dir}"
fi

for REQUIRED_PATH in \
  "${VENV_DIR}" \
  "${STORYMEM_DIR}/pipeline.py" \
  "${LYRICS_FILE}" \
  "${T2V_MODEL_PATH}" \
  "${I2V_MODEL_PATH}" \
  "${LORA_WEIGHT_PATH}" \
  "${STORYMEM_DIR}/audio_tools/.venv/acestep/bin/python" \
  "${AGENTIC_DIR}/run_acestep_story_audio.py"; do
  if [ ! -e "${REQUIRED_PATH}" ]; then
    echo "Required path does not exist: ${REQUIRED_PATH}" >&2
    exit 2
  fi
done

if [ -z "${WHISPERX_CMD:-}" ]; then
  echo "WhisperX is required for full agentic lyric-timing review, but no whisperx executable was found." >&2
  echo "Set WHISPERX_BIN=/path/to/whisperx or WHISPERX_CMD='... \${audio_file} ... \${output_dir}'." >&2
  exit 2
fi

mkdir -p "${OUTPUT_PATH}"

cd "${AGENTIC_DIR}"
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Running implemented agentic iterate pipeline"
echo "job_id: ${SLURM_JOB_ID}"
echo "run_name: ${RUN_NAME}"
echo "output: ${OUTPUT_PATH}"
echo "lyrics: ${LYRICS_FILE}"
echo "storymem_dir: ${STORYMEM_DIR}"
echo "gpus: ${NGPUS}"

python -m storymem_agentic.cli iterate \
  --topic "Twinkle Twinkle Little Star" \
  --lyrics-file "${LYRICS_FILE}" \
  --output-dir "${OUTPUT_PATH}" \
  --target-duration 15 \
  --clip-count 3 \
  --max-iterations 1 \
  --seed "${SEED}" \
  --storymem-dir "${STORYMEM_DIR}" \
  --t2v-model-path "${T2V_MODEL_PATH}" \
  --i2v-model-path "${I2V_MODEL_PATH}" \
  --lora-weight-path "${LORA_WEIGHT_PATH}" \
  --nproc-per-node "${NGPUS}" \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --planner-backend mock \
  --review-backend mock \
  --allow-mock-review \
  --audio-aligner whisperx \
  --whisperx-command "${WHISPERX_CMD}" \
  --generate-audio \
  --media-audio-mode full_song \
  --audio-output-suffix _with_music \
  --audio-voice-style "warm clear toddler nursery singalong vocal, exact Twinkle lyrics, gentle adult lead with soft childlike brightness, music box, celesta, glockenspiel star twinkles, soft strings, light harp, slow 3/4 sway, vocals forward" \
  --ace-step-cmd "${ACE_STEP_CMD}" \
  --strict-lullaby-review \
  --execute-video

echo "Implemented agentic iterate pipeline completed"
echo "manifest: ${OUTPUT_PATH}/run_manifest.json"
echo "iteration: ${OUTPUT_PATH}/iterations/001"
