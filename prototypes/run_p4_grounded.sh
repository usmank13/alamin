#!/usr/bin/env bash
# Offline sandbox: source files read-only, only pilot outputs writable.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/prototypes/representative
ulimit -t 180
ulimit -v 4194304
exec timeout 240 bwrap --unshare-all --die-with-parent --new-session \
  --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --proc /proc --dev /dev --tmpfs /tmp --dir /work \
  --ro-bind "$PWD" /work --bind "$PWD/outputs/prototypes/representative" /work/outputs/prototypes/representative \
  --clearenv --setenv PATH /usr/bin --setenv PYTHONPATH /work/src --setenv MUJOCO_GL osmesa \
  --setenv OPENBLAS_NUM_THREADS 1 --setenv OMP_NUM_THREADS 1 --setenv LP_NUM_THREADS 1 --chdir /work \
  /work/.venv/bin/python -m prototypes.p4_grounded
