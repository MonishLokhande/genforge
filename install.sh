#!/usr/bin/env bash
# One-shot setup after `git clone`: installs uv if missing, then creates the .venv.
# Extra args pass straight to `uv sync`:  ./install.sh --extra flow --group robotics
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync "$@"
echo "Installed. Try: uv run forge list"
