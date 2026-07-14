#!/usr/bin/env bash
#SBATCH --job-name=voice-canary
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:45:00
#SBATCH --output=logs/voice_canary_%j.out
#SBATCH --error=logs/voice_canary_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/gautschi/bansa125/storymem-agentic}"
TOOLS_DIR="${TOOLS_DIR:-${REPO_ROOT}/audio_tools}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/voice_canary/${SLURM_JOB_ID:-manual}}"
VOICE_REF_AUDIO="${VOICE_REF_AUDIO:?Set VOICE_REF_AUDIO to a clean speech clip}"
VOICE_REF_TEXT="${VOICE_REF_TEXT:?Set VOICE_REF_TEXT to its exact transcript}"
TARGET_TEXT="${TARGET_TEXT:-Rock-a-bye baby on the tree top}"
MAX_REF_SECONDS="${MAX_REF_SECONDS:-12}"

export HF_HOME="${HF_HOME:-/scratch/gautschi/bansa125/home-cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${OUT_DIR}" "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

duration="$(${TOOLS_DIR}/.venvs/f5tts/bin/python -c 'import soundfile as s,sys; x,r=s.read(sys.argv[1]); print(len(x)/r)' "${VOICE_REF_AUDIO}")"
python - "${duration}" "${MAX_REF_SECONDS}" <<'PY'
import sys
if float(sys.argv[1]) > float(sys.argv[2]):
    raise SystemExit(f"reference audio is {float(sys.argv[1]):.2f}s; maximum is {sys.argv[2]}s")
PY

"${TOOLS_DIR}/bin/storymem-whisperx" "${VOICE_REF_AUDIO}" --model small --language en \
  --device cuda --compute_type float16 --output_format json --output_dir "${OUT_DIR}/reference_whisperx"

if [ "${DISCOVER_ONLY:-0}" = "1" ]; then
  echo "Reference transcript written to ${OUT_DIR}/reference_whisperx"
  exit 0
fi

"${TOOLS_DIR}/bin/storymem-f5tts-line" --ref_audio "${VOICE_REF_AUDIO}" \
  --ref_text "${VOICE_REF_TEXT}" --gen_text "${TARGET_TEXT}" --output_file "${OUT_DIR}/f5.wav"
"${TOOLS_DIR}/bin/storymem-whisperx" "${OUT_DIR}/f5.wav" --model small --language en \
  --device cuda --compute_type float16 --output_format json --output_dir "${OUT_DIR}/f5_whisperx"

if [ -x "${TOOLS_DIR}/bin/storymem-cosyvoice-line" ]; then
  "${TOOLS_DIR}/bin/storymem-cosyvoice-line" --text "${TARGET_TEXT}" \
    --prompt-audio "${VOICE_REF_AUDIO}" --prompt-text "${VOICE_REF_TEXT}" --output "${OUT_DIR}/cosyvoice.wav"
  "${TOOLS_DIR}/bin/storymem-whisperx" "${OUT_DIR}/cosyvoice.wav" --model small --language en \
    --device cuda --compute_type float16 --output_format json --output_dir "${OUT_DIR}/cosyvoice_whisperx"
fi

python scripts/score_voice_canary.py --output-dir "${OUT_DIR}" --reference-text "${VOICE_REF_TEXT}" \
  --target-text "${TARGET_TEXT}" --max-wer "${MAX_WER:-0.25}"
