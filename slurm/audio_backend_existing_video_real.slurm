#!/bin/bash
#SBATCH --job-name=audio-existing-real
#SBATCH --output=/scratch/gautschi/bansa125/storymem-agentic/logs/audio_existing_real_%j.out
#SBATCH --error=/scratch/gautschi/bansa125/storymem-agentic/logs/audio_existing_real_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --requeue

set -uo pipefail
umask 002

REPO_ROOT="${REPO_ROOT:-/scratch/gautschi/bansa125/storymem-agentic}"
TOOLS_DIR="${TOOLS_DIR:-${REPO_ROOT}/audio_tools}"
SOURCE_ITERATION="${SOURCE_ITERATION:-${REPO_ROOT}/results/twinkle_context_scene_full_4gpu_20260630_001741/iterations/004}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/results/audio_existing_video_real/${SLURM_JOB_ID:-manual}}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-/home/bansa125/bin/ffprobe}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

export HF_HOME="${HF_HOME:-/scratch/gautschi/bansa125/home-cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${TOOLS_DIR}/cache/torch}"
export AUDIOCRAFT_CACHE_DIR="${AUDIOCRAFT_CACHE_DIR:-${TOOLS_DIR}/models/audiocraft}"
export TOOLS_DIR
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}" "${TMPDIR}"
cd "${REPO_ROOT}"

module load python

SOURCE_GENERATED="${SOURCE_ITERATION}/generated"
STORY_JSON="${SOURCE_ITERATION}/story.json"
RHYME_FILE="${SOURCE_ITERATION}/lyrics.txt"
VOICE_REF_AUDIO="${VOICE_REF_AUDIO:-${SOURCE_GENERATED}/audio/mixed_song.wav}"
VOICE_REF_TEXT="${VOICE_REF_TEXT:-Twinkle, twinkle, little star. How I wonder what you are.}"

ACE_STEP_CMD="${TOOLS_DIR}/bin/storymem-acestep --lyrics-file \${lyrics_file} --prompt-file \${prompt_file} --duration \${duration} --seed \${seed} --output \${output_file}"
F5_TTS_CMD="${TOOLS_DIR}/bin/storymem-f5tts-line --ref_audio \${ref_audio} --ref_text \${ref_text} --gen_text \${text} --output_file \${output_file}"
COSYVOICE_CMD="${TOOLS_DIR}/bin/storymem-cosyvoice-line --text \${text} --prompt-audio \${ref_audio} --prompt-text \${ref_text} --output \${output_file}"
MUSICGEN_CMD="${TOOLS_DIR}/bin/storymem-musicgen-bed --prompt \${music_prompt} --duration \${duration} --seed \${seed} --output \${output_file}"
STABLE_AUDIO_CMD="${TOOLS_DIR}/bin/storymem-stable-audio-bed --prompt \${music_prompt} --duration \${duration} --seed \${seed} --output \${output_file}"
export STABLE_AUDIO_CMD

write_status() {
  local case_name="$1"
  local status="$2"
  local reason="$3"
  printf '{"case":"%s","status":"%s","reason":"%s"}\n' "${case_name}" "${status}" "${reason}" > "${RUN_ROOT}/${case_name}.json"
}

prepare_case_dir() {
  local case_name="$1"
  local out_dir="${RUN_ROOT}/${case_name}/generated"
  mkdir -p "${out_dir}"
  cp "${SOURCE_GENERATED}"/[0-9][0-9]_[0-9][0-9].mp4 "${out_dir}/"
  cp "${SOURCE_GENERATED}/generated_subtitled.mp4" "${out_dir}/generated_subtitled.mp4"
  cp "${SOURCE_GENERATED}/subtitles.ass" "${out_dir}/subtitles.ass"
  printf '%s\n' "${out_dir}"
}

run_case() {
  local case_name="$1"
  shift
  local out_dir
  out_dir="$(prepare_case_dir "${case_name}")"
  local start end rc output_video
  start="$(date +%s)"
  echo "=== ${case_name} ==="
  set +e
  "${PYTHON_BIN}" -m storymem_agentic.cli postprocess-audio \
    --rhyme-file "${RHYME_FILE}" \
    --story-json "${STORY_JSON}" \
    --output-dir "${out_dir}" \
    --final-video "${out_dir}/generated_subtitled.mp4" \
    --ffmpeg-bin "${FFMPEG_BIN}" \
    "$@"
  rc=$?
  set -e
  end="$(date +%s)"
  output_video="$(find "${out_dir}" -maxdepth 1 -type f -name 'generated_subtitled_*.mp4' | sort | tail -1)"
  if [ "${rc}" -eq 0 ] && [ -n "${output_video}" ]; then
    "${FFPROBE_BIN}" -hide_banner -v error -show_streams "${output_video}" > "${RUN_ROOT}/${case_name}_ffprobe.txt" 2>&1
    printf '{"case":"%s","status":"ok","elapsed_seconds":%s,"output_video":"%s"}\n' \
      "${case_name}" "$((end - start))" "${output_video}" > "${RUN_ROOT}/${case_name}.json"
  else
    printf '{"case":"%s","status":"failed","elapsed_seconds":%s,"returncode":%s,"output_video":"%s"}\n' \
      "${case_name}" "$((end - start))" "${rc}" "${output_video}" > "${RUN_ROOT}/${case_name}.json"
  fi
}

set -e

echo "Running real audio backend tests against existing video"
echo "source_iteration: ${SOURCE_ITERATION}"
echo "run_root: ${RUN_ROOT}"
echo "voice_ref_audio: ${VOICE_REF_AUDIO}"

if [ ! -f "${SOURCE_GENERATED}/generated_subtitled.mp4" ] || [ ! -f "${STORY_JSON}" ] || [ ! -f "${RHYME_FILE}" ]; then
  echo "Missing required source iteration files." >&2
  exit 2
fi

if [ ! -f "${VOICE_REF_AUDIO}" ]; then
  echo "Missing voice reference audio: ${VOICE_REF_AUDIO}" >&2
  exit 2
fi

run_case ace_full_song \
  --mode full_song \
  --voice-backend ace_step_full_song \
  --music-backend ace_step_full_song \
  --media-audio-mode full_song \
  --audio-output-suffix "_ace_full_song" \
  --audio-voice-style "warm clear toddler nursery singalong vocal, exact Twinkle lyrics, gentle music box and celesta" \
  --ace-step-cmd "${ACE_STEP_CMD}" \
  --seed 701

run_case f5_musicgen_stems \
  --mode voice_bed \
  --voice-backend f5_tts \
  --music-backend musicgen \
  --media-audio-mode separate_stems \
  --audio-output-suffix "_f5_musicgen_stems" \
  --audio-voice-style "gentle exact lyric lullaby voice, clear child-safe pronunciation" \
  --vocal-cmd "${F5_TTS_CMD}" \
  --musicgen-cmd "${MUSICGEN_CMD}" \
  --voice-ref-audio "${VOICE_REF_AUDIO}" \
  --voice-ref-text "${VOICE_REF_TEXT}" \
  --seed 711

run_case f5_musicgen_scene \
  --mode voice_bed \
  --voice-backend f5_tts \
  --music-backend musicgen \
  --media-audio-mode hybrid_voice_bed \
  --audio-output-suffix "_f5_musicgen_scene" \
  --audio-voice-style "gentle exact lyric lullaby voice, clear child-safe pronunciation" \
  --vocal-cmd "${F5_TTS_CMD}" \
  --musicgen-cmd "${MUSICGEN_CMD}" \
  --voice-ref-audio "${VOICE_REF_AUDIO}" \
  --voice-ref-text "${VOICE_REF_TEXT}" \
  --seed 721

run_case cosy_musicgen_scene \
  --mode voice_bed \
  --voice-backend cosyvoice \
  --music-backend musicgen \
  --media-audio-mode hybrid_voice_bed \
  --audio-output-suffix "_cosy_musicgen_scene" \
  --audio-voice-style "gentle exact lyric lullaby voice, clear child-safe pronunciation" \
  --vocal-cmd "${COSYVOICE_CMD}" \
  --musicgen-cmd "${MUSICGEN_CMD}" \
  --voice-ref-audio "${VOICE_REF_AUDIO}" \
  --voice-ref-text "${VOICE_REF_TEXT}" \
  --seed 731

if [ -n "${HF_TOKEN:-}" ]; then
  run_case f5_stable_stems \
    --mode voice_bed \
    --voice-backend f5_tts \
    --music-backend stable_audio \
    --media-audio-mode separate_stems \
    --audio-output-suffix "_f5_stable_stems" \
    --audio-voice-style "gentle exact lyric lullaby voice, clear child-safe pronunciation" \
    --vocal-cmd "${F5_TTS_CMD}" \
    --voice-ref-audio "${VOICE_REF_AUDIO}" \
    --voice-ref-text "${VOICE_REF_TEXT}" \
    --seed 741
else
  write_status f5_stable_stems skipped "HF_TOKEN is not set; Stable Audio is gated"
fi

ALIGN_AUDIO="$(find "${RUN_ROOT}" -path '*/generated/audio/mixed_song.wav' -type f | sort | head -1)"
if [ -n "${ALIGN_AUDIO}" ] && [ -x "${TOOLS_DIR}/bin/storymem-whisperx" ]; then
  mkdir -p "${RUN_ROOT}/whisperx"
  set +e
  "${TOOLS_DIR}/bin/storymem-whisperx" "${ALIGN_AUDIO}" \
    --model small --language en --device cuda --compute_type float16 \
    --output_format json --output_dir "${RUN_ROOT}/whisperx"
  rc=$?
  set -e
  if [ "${rc}" -eq 0 ]; then
    write_status whisperx ok "aligned ${ALIGN_AUDIO}"
  else
    write_status whisperx failed "return code ${rc} aligning ${ALIGN_AUDIO}"
  fi
else
  write_status whisperx skipped "no generated mixed_song.wav or missing wrapper"
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
root = Path("${RUN_ROOT}")
summary = {}
for path in sorted(root.glob("*.json")):
    try:
        summary[path.stem] = json.loads(path.read_text())
    except Exception as exc:
        summary[path.stem] = {"status": "unreadable", "reason": str(exc)}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
print(json.dumps({"run_root": str(root), "summary": summary}, indent=2))
PY
