#!/usr/bin/env bash
# Linux sandbox for the reviewed agent-authored prototype code, not arbitrary downloaded code.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/prototypes/p4
ulimit -t 120
ulimit -v 4194304
exec timeout 180 bwrap --unshare-all --die-with-parent --new-session \
  --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --proc /proc --dev /dev --tmpfs /tmp --dir /work \
  --ro-bind "$PWD" /work --bind "$PWD/outputs/prototypes/p4" /work/outputs/prototypes/p4 \
  --clearenv --setenv PATH /usr/bin --setenv OPENBLAS_NUM_THREADS 1 \
  --setenv OMP_NUM_THREADS 1 --chdir /work \
  /work/.venv/bin/python -m prototypes.p4
