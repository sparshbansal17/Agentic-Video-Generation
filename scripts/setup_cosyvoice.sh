#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="${TOOLS_DIR:-/scratch/gautschi/bansa125/storymem-agentic/audio_tools}"
REPO_DIR="${TOOLS_DIR}/repos/CosyVoice"
VENV_DIR="${TOOLS_DIR}/.venvs/cosyvoice"
MODEL_DIR="${TOOLS_DIR}/models/cosyvoice/CosyVoice2-0.5B"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
BUILD_CONSTRAINTS="${TOOLS_DIR}/logs/cosyvoice-build-constraints.txt"

mkdir -p "${TOOLS_DIR}/repos" "${TOOLS_DIR}/models/cosyvoice" "${TOOLS_DIR}/logs"

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

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "${REPO_DIR}"
else
  git -C "${REPO_DIR}" submodule update --init --recursive
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
fi

cat > "${BUILD_CONSTRAINTS}" <<'EOF'
setuptools<81
EOF
export PIP_CONSTRAINT="${PIP_CONSTRAINT:-${BUILD_CONSTRAINTS}}"

pip_install --upgrade pip wheel "setuptools<81"
pip_install -r "${REPO_DIR}/requirements.txt"

if [ ! -d "${MODEL_DIR}" ]; then
  pip_install "huggingface_hub[cli]"
  "${VENV_DIR}/bin/huggingface-cli" download FunAudioLLM/CosyVoice2-0.5B \
    --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False
fi
