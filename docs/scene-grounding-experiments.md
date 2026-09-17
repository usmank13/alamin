# Scene-level grounding: first implementation and findings

This is the historical first experiment. The subsequent
[distribution-conditioned architecture implementation](distribution-conditioned-architecture.md)
adds a joint sampler and frozen held-out evaluation; it does not supersede the
limitations of the furnishing experiment reported here.

2026-09-17. This is an implemented feasibility slice of
[the plan](scene-level-grounding-plan.md), not completion of the full scene generator.

## Result

The reference-to-simulation path works for the sampled architecture. The naive
empirical layout backend does **not** yet establish on-distribution generation.
Both findings are useful: geometry/schema correctness and distributional quality
are now independently testable, without an LLM judging its own output.

Local inspection artifacts:

- `outputs/scene_grounding/experiment_v2/index.html`: full experiment gallery.
- `outputs/scene_grounding/experiment_v2/reconstruction/003/index.html`: example
  source/compiled overlay and native MuJoCo view.
- `outputs/scene_grounding/experiment_v2/reconstruction/003/render/cycles.png`:
  Cycles overview of the same compiled geometry.
- `outputs/scene_grounding/experiment_v2/report.json`: all failures and metrics.
- `outputs/scene_grounding/corpus_v4/audit.json`: corpus coverage and omissions.
- `outputs/scene_grounding/cli_empirical`: public generation-command smoke test,
  including a verified articulated URDF/PyBullet export.
- `outputs/scene_grounding/reconstruction_export`: reconstructed architecture
  exported and loaded in PyBullet (load smoke test, not articulation certification).

Earlier output directories are retained experiments. None are checked-in datasets.

## What changed

1. `generators.py` provides a layout boundary with heuristic and empirical backends.
   Both emit the same SceneIR-v1; the compiler does not dispatch on backend.
2. `reference_layout.py` ingests real SVG annotations with affine transforms,
   source hashes, explicit missingness, and dimensionless feature extraction.
   `reconstruction.py` is another producer, emitting SceneIR-v2 architecture.
3. SceneIR-v2 adds generic wall polygons and polygonal door/window openings.
   `architecture.py` triangulates/extrudes these into separate convex collision
   and visual meshes. Existing SceneIR-v1 scenes remain supported.
4. `scene_intent.py` freezes the first program's required inventory/relations and
   space requirements across repairs. This is a compatibility intent contract,
   **not** the full functional-zone/group DSL and not proof of correct prompt parsing.
5. `scene_checks.py` independently checks required inventory/relations, asset
   identity, dimensions, room containment and support against resolved packages.
   It does not trust cached instance bounds or a generator's success flags.
6. Prior bundles carry train/calibration/test source groups, joint feature rows,
   source identities and content hashes. Unknown sources, changed hashes and
   cross-partition source leakage are rejected. Hashes detect corruption, not
   authenticity against an adversary able to rewrite the trusted corpus itself.
7. The space-type enum is now open-ended. Unknown categories still fail as before.
   Asset construction, physical properties and articulation routes are unchanged.
8. Cycles supports a generic overview camera for multi-room architecture. The
   existing interior view remains available; no diffusion renderer was added.

The old heuristic remains the default. Empirical generation is explicit and
experimental. Generation's `validated` status still refers to implemented checks,
not empirical scene realism, full robot connectivity, or full brief acceptance.

## Real corpus and evidence limitations

Source: the [official CubiCasa5K repository](https://github.com/CubiCasa/CubiCasa5k)
and its [published archive](https://zenodo.org/records/2613548).
The publisher's [license](https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE)
is CC BY-NC 4.0. This local feasibility sample is **not** an approved commercial
corpus. Verify rights before any commercial use or redistribution.

Fetched 24 SVG annotations using HTTP byte ranges: 2,903,162 bytes instead of the
5,469,495,706-byte archive. Selection is evenly spaced through sorted archive
members, not a representative sample. Each member's SHA-256 and archive path are
retained in `vendor/layout_reference/cubicasa_sample/sources.json`.

The audit extracted 209 interior room annotations with 69 skipped annotations
(missing, degenerate or invalid polygons). All skips remain in per-source records.
Splitting by complete source annotation gives 133 training, 49 calibration and
27 test room rows. This prevents room-level leakage within a plan; deduplication
of different files depicting the same building is not implemented yet.

Absolute scale is **not verified**. Reconstruction explicitly normalizes each
source's interior area to 90 m². It recovers source proportions/topology, not the
original building's metric dimensions. Height, sill and lintel values are declared
engineering defaults. Hidden SVG dimension labels are not accepted as validated
measurements. The CLI also accepts an explicitly supplied metres-per-unit value,
which remains labeled as user-supplied rather than source-verified.

The sample has room labels such as Kitchen, Storage and Bedroom, but no demonstrated
coverage of commercial kitchens, workshops or diverse working spaces. It contains
fixed-fixture annotations, not complete inventories of furniture and clutter.
Empirical furnishing features currently compare only BaseCabinet, WallCabinet and
Refrigerator annotations against their matching asset categories. Other fixtures
are retained in source records but not forced into unrelated asset categories.
Annotation granularity (a fixture run versus individual modules) is not normalized;
count-based comparisons are exploratory and must not become acceptance gates yet.

## Reconstruction test

Pipeline: source SVG -> normalized reference -> explicit scale -> common SceneIR
-> MuJoCo compilation -> recover collision-mesh footprints -> compare to source.

All 24 reconstructed architectures passed:

- Minimum floor-footprint IoU: 0.999998997.
- Minimum wall-footprint IoU at 1.2 m: 0.999985341.
- Maximum compared boundary discrepancy: approximately 0.0000142 m.

MuJoCo float32 mesh seams are merged on a declared 0.00001 m grid for comparison.
Acceptance tolerances are 0.001 m boundary distance and 0.0001 relative area error;
these are numerical reconstruction tolerances, **not** dataset-plausibility targets.
The source-vs-compiled comparison is independent of the generator's cached bounds.

Doors and windows are real boolean cuts in wall volumes. Only floor and wall
cross-sections are measured here; complete room-connectivity, lintel-height and
per-opening topology verification remain future tests. Room adjacency metadata
uses a declared 2 cm proximity heuristic, not an independently certified graph.

Fixtures are not reconstructed as invented 3D assets. Their 2D footprints appear
in orange in source overlays; omission records explain what is missing. Thus this
is **architecture reconstruction**, not a furnished or robot-ready reconstruction.
The match is against imported valid polygons; skipped source annotations prevent
a claim of complete source recovery. Synthetic unit tests additionally cover
oblique walls, concave floors, transform composition and deliberately displaced walls.

## Distribution experiment

Controlled experiment: same supported inventory and 36 m² area for every run;
three semantic labels (kitchen, storage, workshop), two backends, four seeds.
The inventory is supplied deliberately to isolate scene layout from asset selection
and LLM behavior. No agent invocation is included in these measurements.

The empirical prototype samples room aspect from training data and selects among
12 seeded placements using nearest-neighbor distance in a standardized **joint**
feature vector: aspect, rectangularity, comparable-fixture coverage, wall gap and
count. It does not copy floorplan coordinates. This is best-of-N proposal selection,
not the planned hierarchical functional-group optimizer.

All 24 runs passed current scene and physics checks. The common compiler/export
path also passed a separate public-CLI empirical generation and five-joint
URDF/PyBullet actuation test.

Held-out diagnostics, lower joint energy distance is closer for these features:

| Condition | Heuristic | Empirical | Real calibration-to-test baseline |
| --- | ---: | ---: | ---: |
| Kitchen, five features | 4.501 | 5.159 | 1.173 |
| Generic architecture fallback, two features | 0.329 | 0.257 | 0.060 |

Kitchen empirical output had three unique layouts across four seeds, versus four
for the baseline. Candidate pools overlap and best-of-N selection can collapse
onto the same preferred arrangement. Storage has no held-out test rooms in this
small split, so no storage generalization score is claimed. Workshop requires
explicit backoff and compares only generic architecture statistics, **not** workshop
furnishing evidence. Unsupported roles fail unless that backoff is explicitly enabled.

The held-out partition is scored only after generation. No threshold is fitted to
make these outputs pass. Calibration ranks are diagnostic, and the report explicitly
sets `full_scene_distribution_verified=false`.

### What this suggests next

Do not scale up best-of-N selection as the final generator. It can reduce a local
training cost without matching held-out distributions or preserving diversity.
The next scene-level slice should:

1. Normalize annotation granularity and define comparable functional groups and
   relative arrangements; avoid treating source cabinet runs as module counts.
2. Condition spatial priors on the immutable requested inventory/activity/area,
   rather than comparing any three-cabinet scene with an unconditional kitchen pool.
3. Sample conditional joint arrangements with diversity, then apply constrained
   geometric repair; avoid always selecting the most typical candidate.
4. Add real evidence for furnished working spaces and expand source-group auditing
   before claiming cross-domain generalization. Keep a new untouched evaluation split
   for subsequent model choices rather than repeatedly tuning on these five kitchens.
5. Implement functional-zone geometry, group-relative constraints and robot access
   checks; the current scene labels do not implement these semantics.

## Reproduce

Install updated optional dependencies first:

```bash
uv sync --extra pipeline --extra dev
.venv/bin/python scripts/fetch_floorplan_sample.py --count 24 \
  --output vendor/layout_reference/new_sample
.venv/bin/pipeline layout-audit vendor/layout_reference/new_sample \
  --output outputs/layout_corpus
.venv/bin/pipeline reconstruct outputs/layout_corpus/references/003.json \
  --area-m2 90 --output outputs/layout_reconstruction
.venv/bin/pipeline render outputs/layout_reconstruction --view overview
MUJOCO_GL=osmesa .venv/bin/python scripts/experiment_scene_grounding.py \
  --corpus outputs/layout_corpus --output outputs/layout_experiment --seeds 4 --render
```

Output directories must not already exist. Downloading is explicit; generation
and tests do not fetch data automatically. A fitted corpus stays outside agent
authorship and is supplied by the caller:

```bash
.venv/bin/pipeline generate \
  --prompt 'A kitchen workspace with work surfaces, storage cabinets and supported containers.' \
  --program outputs/layout_experiment/kitchen_empirical_0/program.json \
  --layout-backend empirical --priors outputs/layout_corpus/priors.json \
  --output outputs/layout_generated
```

Omit `--program` to use the existing agent intent path; the distribution experiment
above intentionally does not exercise it. `--allow-prior-backoff` permits labeled
generic architecture fallback for unsupported room labels; it does not manufacture
domain-specific evidence.

Regression coverage lives in `tests/test_scene_grounding.py`. Synthetic fixtures
are labeled as such and never presented as real-data evidence. Existing mapping
accuracy thresholds have also been removed to align with the brief: error metrics
remain reported, not passed against an invented 0.8 IoU target.
