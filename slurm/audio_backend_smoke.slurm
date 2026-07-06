#!/usr/bin/env bash
#SBATCH --job-name=audio-backend-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/audio_backend_smoke_%j.out
#SBATCH --error=logs/audio_backend_smoke_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/gautschi/bansa125/storymem-agentic}"
TOOLS_DIR="${TOOLS_DIR:-${REPO_ROOT}/audio_tools}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/audio_backend_smoke/${SLURM_JOB_ID:-manual}}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"

export HF_HOME="${HF_HOME:-/scratch/gautschi/bansa125/home-cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${TOOLS_DIR}/cache/torch}"
export AUDIOCRAFT_CACHE_DIR="${AUDIOCRAFT_CACHE_DIR:-${TOOLS_DIR}/models/audiocraft}"

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

cat > "${OUT_DIR}/lyrics.txt" <<'EOF'
Twinkle, twinkle, little star,
How I wonder what you are!
EOF

cat > "${OUT_DIR}/prompt.json" <<'EOF'
{
  "style_prompt": "gentle nursery lullaby, clear exact lyrics, music box and celesta",
  "lyrics": "Twinkle, twinkle, little star,\nHow I wonder what you are!"
}
EOF

record_skip() {
  local name="$1"
  local reason="$2"
  printf '{"backend":"%s","status":"skipped","reason":"%s"}\n' "$name" "$reason" > "${OUT_DIR}/${name}_manifest.json"
}

run_backend() {
  local name="$1"
  shift
  local start end
  start="$(date +%s)"
  if "$@"; then
    end="$(date +%s)"
    printf '{"backend":"%s","status":"ok","elapsed_seconds":%s}\n' "$name" "$((end - start))" > "${OUT_DIR}/${name}_manifest.json"
  else
    end="$(date +%s)"
    printf '{"backend":"%s","status":"failed","elapsed_seconds":%s}\n' "$name" "$((end - start))" > "${OUT_DIR}/${name}_manifest.json"
  fi
}

if [ -x "${TOOLS_DIR}/bin/storymem-acestep" ]; then
  run_backend ace_step "${TOOLS_DIR}/bin/storymem-acestep" \
    --lyrics-file "${OUT_DIR}/lyrics.txt" --prompt-file "${OUT_DIR}/prompt.json" \
    --duration 10 --seed 11 --output "${OUT_DIR}/ace_step.wav"
else
  record_skip ace_step "missing wrapper"
fi

if [ -n "${VOICE_REF_AUDIO:-}" ] && [ -n "${VOICE_REF_TEXT:-}" ] && [ -x "${TOOLS_DIR}/bin/storymem-f5tts-line" ]; then
  run_backend f5_tts "${TOOLS_DIR}/bin/storymem-f5tts-line" \
    --ref_audio "${VOICE_REF_AUDIO}" --ref_text "${VOICE_REF_TEXT}" \
    --gen_text "Twinkle, twinkle, little star," --output_file "${OUT_DIR}/f5_tts.wav"
else
  record_skip f5_tts "missing VOICE_REF_AUDIO/VOICE_REF_TEXT or wrapper"
fi

if [ -n "${VOICE_REF_AUDIO:-}" ] && [ -n "${VOICE_REF_TEXT:-}" ] && [ -x "${TOOLS_DIR}/bin/storymem-cosyvoice-line" ]; then
  run_backend cosyvoice "${TOOLS_DIR}/bin/storymem-cosyvoice-line" \
    --text "Twinkle, twinkle, little star," --prompt-audio "${VOICE_REF_AUDIO}" \
    --prompt-text "${VOICE_REF_TEXT}" --output "${OUT_DIR}/cosyvoice.wav"
else
  record_skip cosyvoice "missing VOICE_REF_AUDIO/VOICE_REF_TEXT or wrapper"
fi

if [ -x "${TOOLS_DIR}/bin/storymem-musicgen-bed" ]; then
  run_backend musicgen "${TOOLS_DIR}/bin/storymem-musicgen-bed" \
    --prompt "instrumental only, no vocals, gentle music box lullaby" \
    --duration 10 --seed 13 --output "${OUT_DIR}/musicgen.wav"
else
  record_skip musicgen "missing wrapper"
fi

if [ -x "${TOOLS_DIR}/bin/storymem-stable-audio-bed" ] && [ -n "${HF_TOKEN:-}" ]; then
  run_backend stable_audio "${TOOLS_DIR}/bin/storymem-stable-audio-bed" \
    --prompt "instrumental only, no vocals, gentle music box lullaby" \
    --duration 10 --seed 17 --output "${OUT_DIR}/stable_audio.wav"
else
  record_skip stable_audio "missing wrapper or HF_TOKEN/model terms"
fi

ALIGN_SOURCE="${OUT_DIR}/f5_tts.wav"
if [ ! -f "${ALIGN_SOURCE}" ]; then
  ALIGN_SOURCE="${OUT_DIR}/ace_step.wav"
fi
if [ -f "${ALIGN_SOURCE}" ] && [ -x "${TOOLS_DIR}/bin/storymem-whisperx" ]; then
  run_backend whisperx "${TOOLS_DIR}/bin/storymem-whisperx" "${ALIGN_SOURCE}" \
    --model small --language en --device cuda --compute_type float16 \
    --output_format json --output_dir "${OUT_DIR}/whisperx"
else
  record_skip whisperx "no generated audio or missing wrapper"
fi

python scripts/audio_backend_preflight.py --tools-dir "${TOOLS_DIR}" --repo-root "${REPO_ROOT}" \
  --output "${OUT_DIR}/preflight.json" > "${OUT_DIR}/preflight_stdout.json" || true

echo "smoke outputs: ${OUT_DIR}"
