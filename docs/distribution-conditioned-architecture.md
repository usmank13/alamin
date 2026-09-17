# Distribution-conditioned architecture

Implemented 2026-09-17. This is the single-room/hall milestone: semantic intent
from the agent, joint spatial parameters from data, independent geometric checks,
and the existing SceneIR/compiler/render/export path. It is not a kitchen generator.

## What works, and what is not claimed

- Open-ended room labels and opening roles; no domain-specific placement branches.
- Rectangular and supported concave boundaries, doors/windows, and required
  same-wall/opposite-wall opening relationships.
- Seeded joint sampling, source lineage, bounded rejection and explicit unsupported results.
- Robot-disc entry clearance/connectivity when a footprint is supplied.
- Independent source-parameter and compiled-mesh checks, without agent inspection.
- Existing furnishings can be placed on the sampled boundary using the old
  heuristic placement machinery. That furnishing layout is **not** empirically grounded.
- Existing asset geometry/articulation routes are unchanged. A mounting-height
  value moved into asset metadata for the new generic placement path.

**Important scale limitation:** source SVGs have no verified absolute scale.
We sample dimensionless geometry, then scale to the requested interior area.
Opening widths scale with the room: some 90 m² test halls have 5 m-wide openings.
Those are source-proportional apertures, not measured or calibrated real-world
doorway dimensions. The system reports this instead of silently imposing a
handwritten maximum. Metric aperture evidence is a remaining data requirement.

Manufacturing is an extensibility test only. We have no manufacturing corpus.
The available real annotations are CubiCasa5K with its
[CC BY-NC 4.0 license](https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE),
not an approved commercial data source. Geometric checks cannot establish that
a label such as “manufacturing workroom” describes a functionally adequate factory.

## Interfaces and authority

SceneProgram-v1 and both existing backends remain supported and unchanged by
default. `--layout-backend architecture` requires SceneProgram-v2, which adds:

- `space.shape`: `sampled`, `rectangle`, `l_shape`, or `concave`; no annexes yet.
- `architecture.openings`: request ID, door/window kind, count, semantic role,
  required flag and origin (`explicit` or `inferred`).
- `architecture.relationships`: `same_wall` or `opposite_wall` between two
  opening-request IDs, with required/origin fields.
- `architecture.unsupported_requirements`: unsupported explicit requirements,
  such as a specified absolute door width or a multi-room request. Nonempty lists
  fail honestly; they cannot disappear during a repair.

The complete declarative example is `examples/architecture_hall.json`. V2 permits
an empty object inventory for architecture-only generation. With a required count
for an opening kind, that kind's total must equal the sum of required counts.
Unspecified kinds can be sampled. Inferred optional requests are reported when
unsatisfied, not silently claimed to have been implemented.

The agent emits semantic intent, not coordinates, source IDs, distribution
parameters or asset measurements. Explicit architecture requirements are frozen
across repairs. Inferred preferences can change with a logged failure/revision.
Natural-language interpretation itself is not independently certified; the checks
verify the executable intent, not every possible interpretation of the prompt.

The corpus is supplied by the caller, not authored by the agent. Unknown labels
or configurations with fewer than three independent supporting training sources
fail unless `--allow-prior-backoff` is supplied. Backoff uses generic architecture
evidence and is named `generic_architecture_backoff`; it is not evidence for the
requested domain. The minimum of three sources is an engineering support guard,
not a statistical confidence guarantee.

`architecture` is an architecture-only generation command. `generate` with the
architecture backend additionally accepts existing asset requests; all artifacts
still use SceneIR-v2. Reconstruction remains a separate producer of that schema.

## How generation works

1. Ingest source room/wall/opening polygons, including SVG transforms. Verify
   opening/wall intersection and a unique room-boundary host. Ambiguities, missing
   architectural annotations, overlaps and missing entrances are excluded visibly.
2. Normalize to unit interior area, consistent winding and a canonical rotation.
   Do not augment by reflection. Retain topology, host-edge indices, normalized
   opening positions and relative widths as a complete joint record.
3. Select compatible training records using the supplied semantic label, shape
   and executable opening constraints. Labels only select data; they do not
   activate hand-authored arrangement rules.
4. Sample an eligible configuration and two records from different source groups.
   Interpolate the entire boundary/opening parameter vector with one shared
   weight. This preserves relationships between the sampled parameters; no
   marginal-histogram matching or best-of-N typicality ranking is used.
5. Scale to requested area and construct walls/openings deterministically. Accept
   the first feasible sample, with at most 64 proposals. Record every rejection.
   If the budget is exhausted, return `ARCHITECTURE_UNSAT`, not the legacy heuristic.
6. Compile SceneIR and independently recheck it against trusted source parameters
   and actual MuJoCo collision meshes. Persist validation, provenance and previews.

Walls use the existing 0.12 m engineering thickness; height, sill and lintel values
are explicit defaults. Generated openings are structural cuts, not rigged door
assets. Inserting a stock articulated door that needs resizing currently fails
with `DOOR_ASSET_FIT_UNSUPPORTED` rather than creating a misleading fit.

The generic furnishing adapter reads the actual boundary and opening reservations.
It uses asset mounting metadata and support surfaces; it has no kitchen/office/
manufacturing branch. Conservative swept bounds can reject geometrically feasible
placements; a richer furnishing solver remains separate work.

## Independent checks

Checks recompute requested area, boundary validity, wall envelope, opening host,
width/shape, counts, relationships, adjacency metadata, source identity and
interpolated parameters. A generated success flag is not evidence.

Robot clearance uses a circular footprint plus an explicit margin. It checks
required door widths, approach points and connection through eroded free space.
Optional furnishings contribute package-derived conservative swept obstacles,
not generator-supplied bounding boxes. Without a radius, robot access is marked
unverified. This does not prove manipulator reachability or full motion planning.

Compiled geometry is compared at floor level and wall heights 0.4, 1.2 and 2.5 m.
Numerical comparison uses a 10 µm seam grid, 1 mm boundary tolerance and 0.0001
relative area tolerance. These are geometry tolerances, not distribution targets.

Architecture physics validation has no five-articulated-object quota. The existing
brief profile still does. Static URDF verification reports `static_package_load_only`
and `articulation_verified=false`; a successful load is not articulation proof.

Stock mobile-base insertion derives its approach pose from the actual entrance
boundary, including inward heading, rather than assuming an entrance at the origin.
Native contact checks reject occupied spawn candidates. Mapping starts from that
known configured pose; live ground truth remains scoring-only. Legacy v1 placement
is unchanged. This is placement compatibility, not a full navigation guarantee.

## Corpus and measured results

Added 36 annotations from the official archive, excluding the previous sample's
building IDs. Train/calibration/test assignments were written before download.
The additional payload was 3,053,692 bytes. Earlier 24 inspected plans are development
training evidence, not reused as untouched evaluation data.

Combined extraction: 60 source plans, 404 usable room records—272 train, 61
calibration and 71 test. Splits are enforced by dataset building ID and identical
file hash. Near-duplicate buildings with different IDs remain a limitation.

Primary experiment: 32 seeds, 90 m², one required entrance, generic architecture
backoff, robot radius 0.3 m plus 0.05 m margin. The held-out target is filtered by
intent but includes configurations absent from eligible training groups.

| Measurement | Result |
| --- | ---: |
| Geometry/compiled checks passed | 32 / 32 |
| Unique accepted layouts | 32 / 32 |
| Exact training-layout matches, 1e-6 parameter rounding | 0 |
| Near training layouts, normalized parameter RMS < 0.001 | 1 |
| Near-duplicate generated layouts at the same tolerance | 0 |
| Joint feature energy distance: generated to test | 0.3933 |
| Fixed heuristic architecture to test | 5.5633 |
| Real calibration to test | 0.1681 |
| Supported configuration coverage of matching test rooms | 57.9% |

Distances are computed jointly over aspect, rectangularity, vertex/opening counts,
opening positions/widths and clear-wall fraction, standardized using training data.
The matching context has 162 training, 36 calibration and 38 test rooms. Test data
is scored after generation and never used to select proposals. No threshold was
tuned to make these scenes pass. The comparison is against the old constant
architecture baseline, not a claim to outperform general learned layout methods.

The geometry improvement does not eliminate data sparsity. Eligible one-entrance
training examples are mostly rectangles (95 records versus three L-shaped rooms),
so the sampler remains heavily rectangular. Separate real-data L-shaped and
opposite-entry examples passed. Unsupported shapes/configurations are still visible
failures rather than occasions for invented evidence.

A larger robot stress run (radius 1.5 m, deliberately large) accepted 16 scenes
from 26 proposals. Ten proposals failed clearance; one also failed connectivity.
Proposal-to-test energy was 0.3547, while accepted-to-test energy became 1.0327.
This shows why post-constraint distribution checking matters. The target is not
filtered for the stress robot footprint, so the shift is diagnostic, not evidence
that enforcing necessary clearance is wrong.

An actual agent-driven prompt for a manufacturing workroom completed on its first
attempt in approximately 21 seconds. It supplied one explicit entrance and an
inferred window preference; the backend used labeled generic backoff. Static export
and PyBullet loading passed. This tests the end-to-end boundary, not manufacturing
realism. Supplied-program tests separately isolate the sampler from LLM behavior.

A one-second stock mobile-base sensor/mapping smoke test on that agent-generated
room completed with no MuJoCo warnings. Its recorded streams are under
`outputs/architecture_grounding/agent_workroom_mapping_smoke/`. The short rollout
checks integration only, not mapping accuracy or complete exploration.

Regression verification: `.venv/bin/python -m pytest -q` passes all 90 tests,
including 24 architecture tests. Static URDF export emits one nonfatal SciPy
Euler-angle gimbal-lock warning; the exported package loads successfully.

Artifacts:

- `outputs/architecture_grounding/evaluation_final/index.html`: 32-seed gallery/metrics.
- `outputs/architecture_grounding/agent_workroom/index.html`: actual agent run.
- `outputs/architecture_grounding/agent_workroom/render/cycles.png`: Cycles overview.
- `outputs/architecture_grounding/concave_hall/index.html`: real-data L-shaped example.
- `outputs/architecture_grounding/opposite_entries/index.html`: two-entry connectivity example.
- `outputs/architecture_grounding/access_stress_large/report.json`: constraint-induced bias.
- `outputs/architecture_grounding/frozen_corpus/audit.json`: source exclusions and split counts.

## Commands

Use new output directories. Downloads are explicit; generation never fetches data.
The existing installation already includes required dependencies.

```bash
# Additional annotation sample; the old 24-plan sample must exist first.
.venv/bin/python scripts/fetch_floorplan_sample.py --count 36 \
  --exclude-manifest vendor/layout_reference/cubicasa_sample/sources.json \
  --freeze-splits --output vendor/layout_reference/new_architecture_sample

.venv/bin/pipeline architecture-fit \
  --sample vendor/layout_reference/cubicasa_sample \
  --sample vendor/layout_reference/new_architecture_sample \
  --output outputs/new_architecture_priors

# Omit --program to use the agent; supply it to replay intent deterministically.
.venv/bin/pipeline architecture \
  --prompt 'A 90 square metre working hall with one entrance; sample the room shape and window arrangement.' \
  --program examples/architecture_hall.json \
  --priors outputs/new_architecture_priors/priors.json \
  --allow-prior-backoff --robot-radius .3 --seed 12 --output outputs/new_hall

.venv/bin/pipeline architecture-check outputs/new_hall \
  --priors outputs/new_architecture_priors/priors.json \
  --allow-prior-backoff --robot-radius .3
.venv/bin/pipeline render outputs/new_hall --view overview
.venv/bin/pipeline export outputs/new_hall --verify
.venv/bin/pipeline run outputs/new_hall --flow mapping --seconds 1 --tier state \
  --output outputs/new_hall_mapping_smoke

.venv/bin/pipeline architecture-evaluate --program examples/architecture_hall.json \
  --priors outputs/new_architecture_priors/priors.json \
  --allow-prior-backoff --robot-radius .3 --seeds 32 --output outputs/new_hall_eval
```

To combine a v2 program's supported furnishings with the architecture, use
`pipeline generate --layout-backend architecture` with the same prior/config flags.
Its output explicitly identifies furnishing placement as heuristic. Multi-room
generation, furnished joint priors, absolute aperture measurements and arbitrary
blueprint image extraction remain outside this milestone.
