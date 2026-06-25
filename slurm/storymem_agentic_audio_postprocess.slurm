#!/bin/bash
#SBATCH --job-name=storymem-audio-agentic
#SBATCH --output=/scratch/gautschi/bansa125/storymem-agentic/logs/storymem_audio_agentic_%j.out
#SBATCH --error=/scratch/gautschi/bansa125/storymem-agentic/logs/storymem_audio_agentic_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00

set -euo pipefail
umask 002

AGENTIC_DIR="${AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
PARENT_STORYMEM_DIR="${PARENT_STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"

cd "${AGENTIC_DIR}"
mkdir -p logs .tmp

module load python

export PYTHONNOUSERSITE=1
export PYTHONPATH="${AGENTIC_DIR}/src:${AGENTIC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${PARENT_STORYMEM_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${PARENT_STORYMEM_DIR}/.torch-cache}"
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${AGENTIC_DIR}/.tmp/matplotlib}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${AGENTIC_DIR}/.pip-cache}"

mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

PYTHON_BIN="${PYTHON_BIN:-${AGENTIC_DIR}/.venv/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"

OUTPUT_DIR="${OUTPUT_DIR:-${AGENTIC_DIR}/results/twinkle_parent_flash_4gpu_bansa125_20260616_181459}"
FINAL_VIDEO="${FINAL_VIDEO:-${OUTPUT_DIR}/twinkle_parent_flash_4gpu_bansa125_20260616_181459_subtitled.mp4}"
STORY_JSON="${STORY_JSON:-${AGENTIC_DIR}/story/storybook_star_toddler_playroom_singalong.json}"
RHYME_FILE="${RHYME_FILE:-${AGENTIC_DIR}/story/twinkle_first4_lyrics.txt}"

AUDIO_PLAN_MODE="${AUDIO_PLAN_MODE:-full_song}"
MEDIA_AUDIO_MODE="${MEDIA_AUDIO_MODE:-full_song}"
VOICE_BACKEND="${VOICE_BACKEND:-ace_step_full_song}"
MUSIC_BACKEND="${MUSIC_BACKEND:-ace_step_full_song}"
AUDIO_OUTPUT_SUFFIX="${AUDIO_OUTPUT_SUFFIX:-_with_music}"
AUDIO_VOICE_STYLE="${AUDIO_VOICE_STYLE:-warm clear toddler nursery singalong vocal, exact Twinkle lyrics, gentle adult lead with soft childlike brightness, music box, celesta, glockenspiel star twinkles, soft strings, light harp, slow 3/4 sway, vocals forward}"
SEED="${SEED:-125}"

if [ -z "${ACE_STEP_CMD:-}" ]; then
  ACE_STEP_CMD="${PARENT_STORYMEM_DIR}/audio_tools/.venv/acestep/bin/python ${AGENTIC_DIR}/run_acestep_story_audio.py --lyrics-file \${lyrics_file} --prompt-file \${prompt_file} --duration \${duration} --seed \${seed} --output \${output_file} --checkpoint-path ${PARENT_STORYMEM_DIR}/audio_tools/ace_step_checkpoints"
fi

echo "Running integrated agentic audio postprocess"
echo "output_dir: ${OUTPUT_DIR}"
echo "final_video: ${FINAL_VIDEO}"
echo "story_json: ${STORY_JSON}"
echo "rhyme_file: ${RHYME_FILE}"
echo "audio_plan_mode: ${AUDIO_PLAN_MODE}"
echo "media_audio_mode: ${MEDIA_AUDIO_MODE}"
echo "python: ${PYTHON_BIN}"

"${PYTHON_BIN}" -m storymem_agentic.cli postprocess-audio \
  --rhyme-file "${RHYME_FILE}" \
  --story-json "${STORY_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --final-video "${FINAL_VIDEO}" \
  --mode "${AUDIO_PLAN_MODE}" \
  --voice-backend "${VOICE_BACKEND}" \
  --music-backend "${MUSIC_BACKEND}" \
  --media-audio-mode "${MEDIA_AUDIO_MODE}" \
  --audio-output-suffix "${AUDIO_OUTPUT_SUFFIX}" \
  --audio-voice-style "${AUDIO_VOICE_STYLE}" \
  --ace-step-cmd "${ACE_STEP_CMD}" \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --seed "${SEED}"
