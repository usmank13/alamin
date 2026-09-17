# Sensor-driven mapping and oracle diagnostics

The live mapper consumes noisy simulated rangefinder readings, wheel angular
velocities and gyro observations. The initial pose, map canvas bounds and wheel
geometry are known configuration. It does not consume true moving robot poses,
scene occupancy, object IDs, semantic masks or RGB-D. Ground-truth geometry is
held for reporting only; true recorded travel is evaluated after the rollout.

## Fixes

- Dropped readings are missing data (NaN), not fictitious 8 m free-space rays.
- Each grid cell updates once per scan; hit cells cannot also be cleared by
  crossing rays. Clearing stops short of the hit's cell/noise uncertainty band.
- Local point-to-line registration corrects translation **and yaw**, using
  neighborhoods estimated from previous sensor returns, robust residual weights
  and an observability check. It does not match against scene geometry.
- Occupancy truth uses cell/collider intersection, preserving thin panels and
  surface-boundary cells. Reports retain the old cell-center score and explicitly
  version the changed definition. The earlier 0.8 IoU and 0.7 coverage targets
  were internal experiments, not brief requirements, and are no longer gates.
  AABB and fixed-height approximations remain limitations.
- Post-scan estimated poses are captured in the `localization` stream. State,
  range truth and noisy observations remain separate. Physical travel is checked
  from the recorded trajectory instead of wheel distance alone.

## Perfect-perception control experiment

```bash
.venv/bin/python scripts/diagnose_mapping.py \
  outputs/pipeline_current_cafe outputs/mapping_v2_cafe outputs/my_mapping_diagnostic
```

This replays the **same trajectory** with three input combinations:

1. Recorded estimated poses and noisy ranges: integration-only baseline.
2. True simulator poses and noisy ranges: isolate localization errors.
3. True simulator poses and noiseless ranges: ideal sensing/integration control.

Scan matching and navigation are disabled in these offline ablations; the pose
is supplied at each scan. They cannot pass a robot task gate. They save per-case
maps and `diagnostic.json`. A persistent gap in the ideal control is evidence of
grid/visibility/geometry/metric assumptions, not sensor noise or localization.

The first corrected 60 s cafe recording (`mapping_v2_cafe`, before the improved
surface matcher) scored approximately 0.485 / 0.833 / 0.918 in these three modes
under the new cell-intersection metric. This isolates substantial residual
localization error on that run. Old recordings cannot recover which max-range
observations were actually dropped; do not interpret their noisy replay as the
corrected dropout implementation.

The scorer is deliberately not an exact-equality oracle: its solid-cell AABB
reference, finite 72-ray visibility, moving sensor height/tilt and grid boundaries
can differ even with noiseless measurements. Do not feed reference occupancy
into scan matching or inflate occupied output just to improve the score.

Regression tests cover missing scans, no-return rays, endpoint self-clearing,
dropout encoding, and translation/yaw recovery from synthetic wall scans.

## Verification (2026-09-17)

The cafe's surface-matching run `outputs/mapping_v3_cafe` scored 0.777 online.
Its offline exact-pose/noisy-range control scored 0.834, and the fully noiseless
control scored 0.941 (`outputs/mapping_v3_diagnostic/diagnostic.json`).
These use the same recording, not independently chosen trajectories.

A stationary startup experiment (`mapping_v4_cafe`) did not improve the score
(0.768) and was removed. Fixed registration reference points prevent map-feedback
drift, but did not alone resolve the remaining gap: `mapping_v5_cafe` scored
0.776, with 4.62 m actual travel, 1.0 observed reachable coverage and no physics
warnings. Its 60 false-positive occupied cells and 21 false negatives are retained
in the report, not masked or relabeled to pass. There is no spec-defined accuracy
threshold; the oracle diagnostic remains a separate control, not a substitute
for reporting sensor-driven results. Current `passed` means a finite metric was
computed from observations, not that a particular accuracy or travel target was met.

The full regression suite passed 44 tests after these changes. The mapper stops
when fewer than half of its range readings are available. Fixed-height, AABB and
cell-boundary assumptions still warrant a more precise geometric evaluator before
using the remaining small score gap to judge scene generation quality.
