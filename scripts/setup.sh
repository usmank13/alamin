#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.uv-cache"
uv sync --frozen --extra dev
.venv/bin/python scripts/fetch_robots.py
