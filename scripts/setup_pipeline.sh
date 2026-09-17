#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.uv-cache"
uv sync --frozen --extra dev --extra pipeline
.venv/bin/python scripts/fetch_robots.py
if [[ "${1:-}" == "--blender" ]]; then
  .venv/bin/python scripts/fetch_blender.py
fi
.venv/bin/pytest -q
