#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.uv-cache"
extras=(--extra pipeline --extra dev)
robots=true
blender=false
tests=false
for argument in "$@"; do
  case "$argument" in
    --minimal) robots=false ;;
    --blender) blender=true ;;
    --agent) extras+=(--extra agent) ;;
    --prototypes) extras+=(--extra prototypes) ;;
    --retrieval) extras+=(--extra retrieval) ;;
    --test) tests=true ;;
    *) echo "Unknown option: $argument" >&2; exit 2 ;;
  esac
done
uv sync --frozen "${extras[@]}"
if "$robots"; then .venv/bin/python scripts/fetch_robots.py; fi
if "$blender"; then .venv/bin/python scripts/fetch_blender.py; fi
.venv/bin/pipeline doctor --smoke
if "$tests"; then .venv/bin/pytest -q; fi
