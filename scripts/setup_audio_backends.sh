#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="/scratch/gautschi/bansa125/storymem-agentic/audio_tools"
INSTALL_F5=1
INSTALL_COSYVOICE=1
INSTALL_MUSICGEN=1
INSTALL_STABLE_AUDIO=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tools-dir)
      TOOLS_DIR="$2"
      shift 2
      ;;
    --skip-f5-tts)
      INSTALL_F5=0
      shift
      ;;
    --skip-cosyvoice)
      INSTALL_COSYVOICE=0
      shift
      ;;
    --skip-musicgen)
      INSTALL_MUSICGEN=0
      shift
      ;;
    --skip-stable-audio)
      INSTALL_STABLE_AUDIO=0
      shift
      ;;
    --install-stable-audio)
      INSTALL_STABLE_AUDIO=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export TOOLS_DIR
export HF_HOME="${HF_HOME:-/scratch/gautschi/bansa125/home-cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${TOOLS_DIR}/cache/torch}"
export AUDIOCRAFT_CACHE_DIR="${AUDIOCRAFT_CACHE_DIR:-${TOOLS_DIR}/models/audiocraft}"

mkdir -p \
  "${TOOLS_DIR}/bin" \
  "${TOOLS_DIR}/repos" \
  "${TOOLS_DIR}/.venvs" \
  "${TOOLS_DIR}/models" \
  "${TOOLS_DIR}/cache/torch" \
  "${TOOLS_DIR}/logs" \
  "${REPO_ROOT}/configs/audio"

ACE_PYTHON="/scratch/gautschi/bansa125/StoryMem/audio_tools/.venv/acestep/bin/python"
ACE_REPO="/scratch/gautschi/bansa125/StoryMem/audio_tools/ACE-Step"
ACE_CHECKPOINT="/scratch/gautschi/bansa125/StoryMem/audio_tools/ace_step_checkpoints"
WHISPERX_BIN="${REPO_ROOT}/.venv-whisperx/bin/whisperx"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"

cat > "${TOOLS_DIR}/bin/storymem-acestep" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="\${HF_HOME:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="\${HUGGINGFACE_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export TORCH_HOME="\${TORCH_HOME:-${TORCH_HOME}}"
export PYTHONPATH="${ACE_REPO}:\${PYTHONPATH:-}"
cd "${REPO_ROOT}"
exec "${ACE_PYTHON}" "${REPO_ROOT}/run_acestep_story_audio.py" --checkpoint-path "${ACE_CHECKPOINT}" "\$@"
EOF

cat > "${TOOLS_DIR}/bin/storymem-whisperx" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="\${HF_HOME:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="\${HUGGINGFACE_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
exec "${WHISPERX_BIN}" "\$@"
EOF

cat > "${TOOLS_DIR}/bin/storymem-f5tts-line" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="\${HF_HOME:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="\${HUGGINGFACE_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export TORCH_HOME="\${TORCH_HOME:-${TORCH_HOME}}"
if [ -z "\${VOICE_REF_AUDIO:-}" ] && [[ " \$* " != *" --ref_audio "* ]]; then
  echo "storymem-f5tts-line requires --ref_audio/--ref_text or VOICE_REF_AUDIO/VOICE_REF_TEXT." >&2
fi
exec "${TOOLS_DIR}/.venvs/f5tts/bin/f5-tts_infer-cli" "\$@"
EOF

cat > "${TOOLS_DIR}/bin/storymem-cosyvoice-line" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="${TOOLS_DIR:-/scratch/gautschi/bansa125/storymem-agentic/audio_tools}"
export HF_HOME="${HF_HOME:-/scratch/gautschi/bansa125/home-cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${TOOLS_DIR}/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TOOLS_DIR}/cache/triton}"
export PYTHONPATH="${TOOLS_DIR}/repos/CosyVoice:${TOOLS_DIR}/repos/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH:-}"
mkdir -p "${TRITON_CACHE_DIR}"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-${TOOLS_DIR}/models/cosyvoice/CosyVoice2-0.5B}"
exec "${TOOLS_DIR}/.venvs/cosyvoice/bin/python" "${TOOLS_DIR}/bin/storymem_cosyvoice_wrapper.py" --model-dir "${MODEL_DIR}" "$@"
EOF

cat > "${TOOLS_DIR}/bin/storymem-musicgen-bed" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="${TOOLS_DIR:-/scratch/gautschi/bansa125/storymem-agentic/audio_tools}"
export HF_HOME="${HF_HOME:-/scratch/gautschi/bansa125/home-cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${TOOLS_DIR}/cache/torch}"
export AUDIOCRAFT_CACHE_DIR="${AUDIOCRAFT_CACHE_DIR:-${TOOLS_DIR}/models/audiocraft}"
exec "${TOOLS_DIR}/.venvs/audiocraft/bin/python" "${TOOLS_DIR}/bin/storymem_musicgen_bed.py" "$@"
EOF

cat > "${TOOLS_DIR}/bin/storymem-stable-audio-bed" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="${TOOLS_DIR:-/scratch/gautschi/bansa125/storymem-agentic/audio_tools}"
export HF_HOME="${HF_HOME:-/scratch/gautschi/bansa125/home-cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${TOOLS_DIR}/cache/torch}"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Stable Audio requires HF_TOKEN and accepted model terms." >&2
  exit 66
fi
exec "${TOOLS_DIR}/.venvs/stable-audio/bin/python" "${TOOLS_DIR}/bin/storymem_stable_audio_bed.py" "$@"
EOF

cat > "${TOOLS_DIR}/bin/storymem_cosyvoice_wrapper.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one CosyVoice line.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-audio", "--prompt_audio", dest="prompt_audio", required=True)
    parser.add_argument("--prompt-text", "--prompt_text", dest="prompt_text", required=True)
    parser.add_argument("--output", "--output_file", dest="output", required=True)
    args = parser.parse_args()

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    model = CosyVoice2(args.model_dir)
    prompt_speech = load_wav(args.prompt_audio, 16000)
    result = next(model.inference_zero_shot(args.text, args.prompt_text, prompt_speech, stream=False))
    audio = result["tts_speech"].detach().cpu().numpy().T
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, model.sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

cat > "${TOOLS_DIR}/bin/storymem_musicgen_bed.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one instrumental MusicGen bed.")
    parser.add_argument("--prompt", "--music_prompt", dest="prompt", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", "--output_file", dest="output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
    model_name = args.model or os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
    inputs = processor(text=[args.prompt], padding=True, return_tensors="pt").to(device)
    sample_rate = int(model.config.audio_encoder.sampling_rate)
    max_new_tokens = max(1, int(args.duration * 50))
    wav = model.generate(**inputs, max_new_tokens=max_new_tokens)[0].detach().cpu().numpy()
    audio = wav.T if wav.ndim == 2 else np.asarray(wav)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

cat > "${TOOLS_DIR}/bin/storymem_stable_audio_bed.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Stable Audio wrapper placeholder.")
    parser.add_argument("--prompt", "--music_prompt", dest="prompt", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", "--output_file", dest="output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.parse_args()
    raise SystemExit(
        "Stable Audio checkpoint/model invocation is gated. Install stable-audio-tools, set HF_TOKEN, "
        "accept model terms, then replace this wrapper with the approved local generation command."
    )


if __name__ == "__main__":
    main()
PY

chmod +x "${TOOLS_DIR}/bin"/storymem-* "${TOOLS_DIR}/bin"/storymem_*.py

if [ "${INSTALL_F5}" -eq 1 ]; then
  bash "${REPO_ROOT}/scripts/setup_f5_tts.sh" > "${TOOLS_DIR}/logs/setup_f5_tts.log" 2>&1 || echo "F5-TTS setup failed; see logs/setup_f5_tts.log" >&2
fi
if [ "${INSTALL_COSYVOICE}" -eq 1 ]; then
  bash "${REPO_ROOT}/scripts/setup_cosyvoice.sh" > "${TOOLS_DIR}/logs/setup_cosyvoice.log" 2>&1 || echo "CosyVoice setup failed; see logs/setup_cosyvoice.log" >&2
fi
if [ "${INSTALL_MUSICGEN}" -eq 1 ]; then
  bash "${REPO_ROOT}/scripts/setup_musicgen.sh" > "${TOOLS_DIR}/logs/setup_musicgen.log" 2>&1 || echo "MusicGen setup failed; see logs/setup_musicgen.log" >&2
fi
if [ "${INSTALL_STABLE_AUDIO}" -eq 1 ]; then
  bash "${REPO_ROOT}/scripts/setup_stable_audio.sh" > "${TOOLS_DIR}/logs/setup_stable_audio.log" 2>&1 || echo "Stable Audio setup skipped/failed; see logs/setup_stable_audio.log" >&2
fi

cat > "${REPO_ROOT}/configs/audio/local.yaml" <<EOF
# Generated by scripts/setup_audio_backends.sh.
tools_dir: ${TOOLS_DIR}
cache:
  hf_home: ${HF_HOME}
  huggingface_hub_cache: ${HUGGINGFACE_HUB_CACHE}
  torch_home: ${TORCH_HOME}
  audiocraft_cache_dir: ${AUDIOCRAFT_CACHE_DIR}
ffmpeg_bin: ${FFMPEG_BIN}
candidate_policy:
  ace_step_full_song: 4
  f5_tts: 4
  cosyvoice: 4
  musicgen: 1
  stable_audio: 1
allow_scene_mix_debug: false
backends:
  ace_step_full_song:
    kind: song
    env: ACE_STEP_CMD
    command: >-
      ${TOOLS_DIR}/bin/storymem-acestep --lyrics-file '\${lyrics_file}' --prompt-file '\${prompt_file}'
      --duration \${duration} --seed \${seed} --output '\${output_file}'
  f5_tts:
    kind: voice
    env: F5_TTS_CMD
    command: >-
      ${TOOLS_DIR}/bin/storymem-f5tts-line --ref_audio '\${ref_audio}' --ref_text '\${ref_text}'
      --gen_text '\${text}' --output_file '\${output_file}'
  cosyvoice:
    kind: voice
    env: COSYVOICE_CMD
    command: >-
      ${TOOLS_DIR}/bin/storymem-cosyvoice-line --text '\${text}' --prompt-audio '\${ref_audio}'
      --prompt-text '\${ref_text}' --output '\${output_file}'
  musicgen:
    kind: music
    env: MUSICGEN_CMD
    command: >-
      ${TOOLS_DIR}/bin/storymem-musicgen-bed --prompt '\${music_prompt}'
      --duration \${duration} --seed \${seed} --output '\${output_file}'
  stable_audio:
    kind: music
    env: STABLE_AUDIO_CMD
    gated: true
    command: >-
      ${TOOLS_DIR}/bin/storymem-stable-audio-bed --prompt '\${music_prompt}'
      --duration \${duration} --seed \${seed} --output '\${output_file}'
  whisperx:
    kind: aligner
    env: WHISPERX_CMD
    command: >-
      ${TOOLS_DIR}/bin/storymem-whisperx '\${audio_file}' --model small --language en
      --device cuda --compute_type float16 --vad_method silero --output_format json --output_dir '\${output_dir}'
EOF

python "${REPO_ROOT}/scripts/audio_backend_preflight.py" \
  --tools-dir "${TOOLS_DIR}" \
  --repo-root "${REPO_ROOT}" \
  --output "${TOOLS_DIR}/logs/setup_summary.json" || true

echo "Audio backend setup wrote ${REPO_ROOT}/configs/audio/local.yaml"
echo "Preflight summary: ${TOOLS_DIR}/logs/setup_summary.json"
