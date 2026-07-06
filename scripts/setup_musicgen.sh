#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="${TOOLS_DIR:-/scratch/gautschi/bansa125/storymem-agentic/audio_tools}"
VENV_DIR="${TOOLS_DIR}/.venvs/audiocraft"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

mkdir -p "${TOOLS_DIR}/.venvs" "${TOOLS_DIR}/models/audiocraft" "${TOOLS_DIR}/logs"

pip_install() {
  if "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    "${VENV_DIR}/bin/python" -m pip install "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "${VENV_DIR}/bin/python" "$@"
  else
    "${VENV_DIR}/bin/python" -m ensurepip --upgrade
    "${VENV_DIR}/bin/python" -m pip install "$@"
  fi
}

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
elif ! "${VENV_DIR}/bin/python" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >/dev/null 2>&1 \
  && ! "${VENV_DIR}/bin/python" -c "from transformers import MusicgenForConditionalGeneration" >/dev/null 2>&1; then
  rm -rf "${VENV_DIR}"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
fi

pip_install --upgrade pip wheel setuptools
pip_install --index-url "${TORCH_INDEX_URL}" torch torchaudio
pip_install "transformers>=4.40" accelerate soundfile scipy
