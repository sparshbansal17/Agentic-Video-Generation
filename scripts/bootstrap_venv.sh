#!/bin/bash
set -euo pipefail

# Cluster-friendly bootstrap. Uses the Python module requested for 3.11, then creates
# a local venv. If uv is available, use it; otherwise fall back to pip.
module load python

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

EXTRAS="${EXTRAS:-dev,agents}"
if command -v uv >/dev/null 2>&1; then
  uv pip install -e ".[${EXTRAS}]"
else
  python -m pip install -e ".[${EXTRAS}]"
fi

python -m unittest discover -s tests
