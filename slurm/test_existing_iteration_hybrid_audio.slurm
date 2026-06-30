#!/bin/bash
#SBATCH --job-name=agentic-hybrid-audio-test
#SBATCH --output=logs/agentic_hybrid_audio_test_%j.out
#SBATCH --error=logs/agentic_hybrid_audio_test_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail
umask 002

AGENTIC_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
VENV_DIR="${VENV_DIR:-${AGENTIC_DIR}/.venv}"
ITERATION_DIR="${ITERATION_DIR:?Set ITERATION_DIR to an existing agentic iteration directory}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-/home/bansa125/bin/ffprobe}"

mkdir -p "${AGENTIC_DIR}/logs" "${AGENTIC_DIR}/.tmp"

module load python
source "${VENV_DIR}/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${AGENTIC_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${AGENTIC_DIR}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${AGENTIC_DIR}/.torch-cache}"
export TMPDIR="${TMPDIR:-${AGENTIC_DIR}/.tmp}"

cd "${AGENTIC_DIR}"

SOURCE_GENERATED="${ITERATION_DIR}/generated"
TEST_ROOT="${ITERATION_DIR}/hybrid_audio_existing_video_test_${SLURM_JOB_ID}"
TEST_GENERATED="${TEST_ROOT}/generated"
REF_AUDIO="${TEST_ROOT}/voice_reference.wav"
REF_TEXT="soft gentle lullaby reference voice"

mkdir -p "${TEST_GENERATED}"
cp "${SOURCE_GENERATED}"/[0-9][0-9]_[0-9][0-9].mp4 "${TEST_GENERATED}/"
cp "${SOURCE_GENERATED}/generated_subtitled.mp4" "${TEST_GENERATED}/generated_subtitled.mp4"
cp "${SOURCE_GENERATED}/subtitles.ass" "${TEST_GENERATED}/subtitles.ass"

PYTHON_BIN="${VENV_DIR}/bin/python"
FAKE_BACKEND="${AGENTIC_DIR}/scripts/fake_audio_backend.py"

"${PYTHON_BIN}" "${FAKE_BACKEND}" \
  --kind voice \
  --text "${REF_TEXT}" \
  --output_file "${REF_AUDIO}" \
  --duration 1.5

VOCAL_CMD="${PYTHON_BIN} ${FAKE_BACKEND} --kind voice --text \${text} --gen_text \${gen_text} --voice_style \${voice_style} --ref_audio \${ref_audio} --ref_text \${ref_text} --lyrics_file \${lyrics_file} --prompt_file \${prompt_file} --output_file \${output_file} --duration \${duration} --seed \${seed} --mode \${mode}"
MUSICGEN_CMD="${PYTHON_BIN} ${FAKE_BACKEND} --kind music --music_prompt \${music_prompt} --lyrics_file \${lyrics_file} --prompt_file \${prompt_file} --output_file \${output_file} --duration \${duration} --seed \${seed} --mode \${mode}"
SONG_CMD="${PYTHON_BIN} ${FAKE_BACKEND} --kind song --lyrics_file \${lyrics_file} --prompt_file \${prompt_file} --output_file \${output_file} --duration \${duration} --seed \${seed} --mode \${mode}"
export STABLE_AUDIO_CMD="${PYTHON_BIN} ${FAKE_BACKEND} --kind music --music_prompt \${music_prompt} --lyrics_file \${lyrics_file} --prompt_file \${prompt_file} --output_file \${output_file} --duration \${duration} --seed \${seed} --mode \${mode}"

echo "Testing existing generated videos with hybrid voice-bed audio"
echo "iteration_dir: ${ITERATION_DIR}"
echo "test_root: ${TEST_ROOT}"
echo "source_video: ${SOURCE_GENERATED}/generated_subtitled.mp4"

"${PYTHON_BIN}" -m storymem_agentic.cli postprocess-audio \
  --rhyme-file "${ITERATION_DIR}/lyrics.txt" \
  --story-json "${ITERATION_DIR}/story.json" \
  --output-dir "${TEST_GENERATED}" \
  --final-video "${TEST_GENERATED}/generated_subtitled.mp4" \
  --mode full_song \
  --voice-backend f5_tts \
  --music-backend musicgen \
  --media-audio-mode hybrid_voice_bed \
  --audio-output-suffix "_hybrid_f5_musicgen" \
  --audio-voice-style "gentle exact lyric lullaby voice, clear child-safe pronunciation" \
  --vocal-cmd "${VOCAL_CMD}" \
  --musicgen-cmd "${MUSICGEN_CMD}" \
  --song-cmd "${SONG_CMD}" \
  --voice-ref-audio "${REF_AUDIO}" \
  --voice-ref-text "${REF_TEXT}" \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --seed 111

rm -rf "${TEST_GENERATED}/audio"

"${PYTHON_BIN}" -m storymem_agentic.cli postprocess-audio \
  --rhyme-file "${ITERATION_DIR}/lyrics.txt" \
  --story-json "${ITERATION_DIR}/story.json" \
  --output-dir "${TEST_GENERATED}" \
  --final-video "${TEST_GENERATED}/generated_subtitled.mp4" \
  --mode full_song \
  --voice-backend cosyvoice \
  --music-backend stable_audio \
  --media-audio-mode hybrid_voice_bed \
  --audio-output-suffix "_hybrid_cosy_stable" \
  --audio-voice-style "gentle exact lyric lullaby voice, clear child-safe pronunciation" \
  --vocal-cmd "${VOCAL_CMD}" \
  --song-cmd "${SONG_CMD}" \
  --voice-ref-audio "${REF_AUDIO}" \
  --voice-ref-text "${REF_TEXT}" \
  --ffmpeg-bin "${FFMPEG_BIN}" \
  --seed 222

"${FFPROBE_BIN}" -hide_banner -v error -show_streams "${TEST_GENERATED}/generated_subtitled_hybrid_f5_musicgen.mp4" >/dev/null
"${FFPROBE_BIN}" -hide_banner -v error -show_streams "${TEST_GENERATED}/generated_subtitled_hybrid_cosy_stable.mp4" >/dev/null

python - <<PY
import json
from pathlib import Path
root = Path("${TEST_GENERATED}")
manifest = json.loads((root / "audio_postprocess_manifest.json").read_text())
outputs = sorted(str(path) for path in root.glob("generated_subtitled_hybrid_*.mp4"))
summary = {
    "test_root": "${TEST_ROOT}",
    "outputs": outputs,
    "latest_manifest_inputs": manifest["inputs"],
}
print(json.dumps(summary, indent=2))
PY
