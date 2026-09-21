#!/usr/bin/env bash
# Linux: install into a project-local venv; never alter a system Python environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}"
case "$TORCH_CHANNEL" in
  cu124) TORCH_VERSION=2.5.1 ;;
  cu128) TORCH_VERSION=2.9.1 ;;
  cpu) TORCH_VERSION=2.5.1 ;;
  *) echo 'TORCH_CHANNEL must be cu124, cu128 or cpu' >&2; exit 2 ;;
esac
"$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12"'
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL" --extra-index-url https://pypi.org/simple
.venv/bin/python -m pip install -r requirements-field-qa.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_field_qa_environment.py --output outputs/environment.json
echo 'Ready. Run: source .venv/bin/activate'
