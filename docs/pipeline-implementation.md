# Scene pipeline: implementation and remaining release gates

This began as a working, kitchen-focused asset/scene vertical slice, not a claim that the full trial
brief is satisfied. The generator is `src/scene_pipeline`; `sim_harness` remains
an independent MJCF/MJZ consumer. Historical experiments stay in `prototypes`.

Categories outside the registry are now sourced with verified evidence and realized as
sized template geometry, relations are solved rather than checked, and the validator
gates the brief's scale, mass and access criteria; see
[open vocabulary and gates](open-vocabulary-and-gates.md). Scene-level generation is
explicitly domain-agnostic. See
[distribution-conditioned architecture](distribution-conditioned-architecture.md)
for the implemented joint boundary/opening sampler and its current evidence limits, and
[scene grounding experiments](scene-grounding-experiments.md) for the new generator
boundary, real-floorplan reconstruction, empirical prototype and negative held-out
result. Functional grouping and broad distributional generalization remain open;
the tables below describe the earlier vertical slice unless noted otherwise.

## Accepted decisions

- A staged, durable implementation; no artificial two-day delivery constraint.
- Initial kitchen-focused asset vocabulary, with explicit failure for required unknown categories;
  the scene schema/generator interface is not a kitchen-specific system.
- Engineering defaults are permitted when versioned and clearly distinguished
  from sourced measurements. No compliance or calibrated-dynamics claims.
- G1 reusable families and G3 native articulated retrieval with **uniform** scale.
  G2 remains supervised offline authoring, not execution of arbitrary agent code.
  G4 needs segmented parts and anchors; arbitrary anisotropic rescaling is rejected.
- Cycles is the only offline renderer. No diffusion renderer or placeholder for one.
- URDF plus actual CPU PyBullet loading and actuation is the first independent
  export target. USD is deferred.
- Agents choose intent and revise programs; deterministic tools own geometry,
  placement and validation. Generated scene files are not manually repaired.

## Setup and a reproducible first run

From the repository root, on Linux with uv and the existing OSMesa libraries:

```bash
bash scripts/setup_pipeline.sh --blender
.venv/bin/pipeline generate \
  --prompt 'A small cafe kitchen with storage, preparation surfaces and containers' \
  --program examples/cafe_program.py --seed 1 --output outputs/my_cafe
.venv/bin/pipeline render outputs/my_cafe
.venv/bin/pipeline export outputs/my_cafe --format urdf --verify
.venv/bin/sim outputs/my_cafe/scene.mjz --viewer --backend glfw
```

Omit `--blender` to skip the approximately 350 MB official binary download.
Cycles currently uses CPU; native MuJoCo physics and OSMesa capture require no
CUDA. This does not diagnose or change the RTX 3070 driver. Cold installation
under fifteen minutes has **not** been measured. Menagerie downloads and Blender
downloads need network access. PyBullet may build from source on Python 3.12.

Omit `--program` to use an authenticated local Codex CLI. The nested invocation
uses structured JSON output, a read-only sandbox and an isolated working folder.
It may use the configured account's inference quota. It does not inherit arbitrary
project agent code as its execution policy. The DSL itself is an AST interpreter,
not Python execution.

```bash
.venv/bin/pipeline generate --prompt 'A 60 square metre kitchen with three base cabinets, two drawer units, two prep tables and twelve containers' \
  --seed 4 --output outputs/agent_kitchen --max-iterations 5 --timeout 900
```

Outputs refuse to overwrite existing generation, rollout and dataset directories.
Run all commands from the repository root so vendor paths resolve. URDF packages
and MJZ bundles themselves are relocatable.

## Stage boundaries and current status

| Milestone | Implemented | Still needed for the full plan |
| --- | --- | --- |
| M0 contracts and evidence | Strict SceneProgram schema, literal DSL, versioned IR checks, field-bound source extraction, explicit defaults, hashes | Full nested IR schema; persistent evidence database/fetch pipeline; source-grounded category distributions |
| M1 assets | G1 cabinet/drawer/table/shelf/door/container families; G3 three native RoboCasa fixture adapters; scale and inertia transforms; candidate validation/promotion | Broader asset retrieval/search and validated corpus; isolated G2 authoring workflow; measured category physical profiles |
| M2 layout | Seeded candidates, rectangle/L footprint and rectangular annexes, opening reservations, conservative 3D sweeps, supported clutter, relation reports | Zone geometry/role constraints; run grammar; annealing; robot-radius connectivity; density targets; walk-in and wash-station support |
| M3 compiler/validation/render | Shared compiled geometry, portable MJZ, stable semantic IDs, five-second settling, mask-independent overlap and 21-pose joint sweeps, Cycles stills and inspection gallery | Dynamic endpoint tests for every asset; category mass/metric assertions; comprehensive PBR materials; semantic/reachability release gates |
| M4 agent runner | Real Codex structured output, bounded repair iterations, transcripts, failure reports and stage timings | Program/fetch cache replay; complete token/spend accounting; hard cancellation across every deterministic stage; three held-out prompts |
| M5 robots/data | Stock base and Panda flows, placement candidates, native sensors, explicit noise, timestamped HDF5, loader, tiered ten-variant collector | Robust mapping accuracy; broader contact manipulation; production flow videos; ten successful full-length randomizations |
| M6 portability/release | URDF package, independent PyBullet motor tests, mass/inertia/COM comparisons, relocated-package regression | Full axis/transform/bounds comparison, broader articulated corpus, two complete acceptance environments, measured cold setup and release writeup |

`validated` currently means the implemented scene checks passed, **not** all
milestones or all brief gates passed. Joint inspection GIFs are kinematic and
explicitly labeled. A preview or a parsed URDF never stands in for physics proof.

## Asset and evidence policy

See [object provenance and the richer kitchen](object-provenance.md) for the
per-object source classifier, mixed procedural/retrieved example, inspection
gallery and the future visual-only generated-clutter boundary.

`pipeline registry` lists the finite supported vocabulary. Template dimensions,
panel thickness, densities, damping, clearances and sensor noise are engineering
defaults in `registry.py` and `sensors.py`, not manufacturer specifications.
Class IDs come from an append-only taxonomy rather than each scene's inventory.
Instance strings remain the durable identity; integer renderer IDs are local.

RoboCasa sources use the locally downloaded native files under
`vendor/robocasa_native`, with original file hashes retained. The adapter preserves
part hierarchy, anchors, contact conventions and prerequisite door states. It
uses an explicit effective shell-density proxy (180 kg/m³) instead of accepting
the source's inferred solid-box masses. This is **not** actual material density,
measured appliance mass, or calibrated dynamics. Uniform scale applies length,
mass and inertia factors s, s³ and s⁵. Source files are never edited.

```bash
.venv/bin/pipeline assets build microwave --output outputs/library
.venv/bin/pipeline assets validate outputs/library/ASSET_KEY
.venv/bin/pipeline assets promote outputs/library/ASSET_KEY
```

Promotion requires a passing report. Missing retrieved sources fail explicitly;
the basic setup installs stock robots, not gated third-party datasets. Existing
download provenance/instructions remain in `prototypes/README.md`. Generated
scene validation checks all composed assets even when their cache state is still
`candidate`. Asset-only acceptance does not imply successful scene placement.

Current probes: microwave and fridge passed the engineering-profile checks;
dishwasher failed passive settling (about 7 mm drift) and must remain unpromoted.
The adapter does not repair that failure by relaxing tolerances.

## Robot flows and data

```bash
.venv/bin/pipeline run outputs/my_cafe --flow mapping \
  --output outputs/my_cafe_mapping --seconds 60 --tier full
.venv/bin/pipeline run outputs/my_cafe --flow interaction \
  --output outputs/my_cafe_interaction --seconds 60 --tier full
.venv/bin/pipeline replay outputs/my_cafe_interaction
.venv/bin/pipeline dataset outputs/my_cafe \
  --output outputs/my_cafe_variants --variants 10 --seconds 60
```

Flows command **robot actuators only**. Panda IK uses a separate scratch state;
the live drawer receives contact forces, not commanded joint positions. Its
success check requires the completed pull/push/release sequence and both open
and final closed endpoint errors within 5 mm, plus measured robot/object contact.
Placement searches a small set of clear initial poses; this is not general
manipulation reachability planning.

The mapper uses noisy wheel speed, gyro and 72 range readings. A robust local
surface scan matcher corrects translation and yaw; ground truth is used separately
for evaluation. RGB-D, accelerometer and semantic masks are recorded but are not
mapping inputs. Known spawn pose, configured map bounds and robot wheel geometry
are explicit priors: this is not globally unlocalized SLAM.

The versioned 10 cm occupancy metric uses **cell/collider intersection**, not
cell-center containment; the legacy center score remains in each report. Scores
across these versions are not directly comparable. Collider AABBs at a fixed
height remain an approximation, not a benchmark-grade surface model. Coverage is
observed reachable free-space coverage, not proof that the robot physically
visited every area. Task travel is evaluated from the recorded true trajectory
after control finishes, rather than accepting inflated wheel-odometry distance.

Range dropouts are NaN with `valid=false` and never clear the map. A real no-return
measurement is 8 m with `valid=false` and can provide free-space evidence. Hit
cells win over crossing free rays within a scan, and clearing stops before the
endpoint uncertainty band. See [mapping diagnostics](mapping-diagnostics.md).

HDF5 streams include 100 Hz state/IMU (and wrist wrench for the arm), 20 Hz range
for the base, and 10 Hz 320×240 RGB-D plus instance/class masks in `full` mode.
Every stream uses simulation seconds on the 2 ms clock. `rig.json` records
intrinsics, mounting and synthetic noise parameters. Truth and observation fields
are separate. Camera noise uses an independent RNG so capture tier does not
alter controller observations. `ctrl` is applied over the following interval.

```python
from scene_pipeline.dataset import inspect, load_stream
print(inspect('outputs/my_cafe_mapping/data.h5'))
samples = load_stream('outputs/my_cafe_mapping/data.h5', 'range', 0, 100)
```

Join different-rate streams by timestamp, not by array index. Scene ground-truth
queries are available through `scene_pipeline.semantics.snapshot(model, data,
manifest)` after `mj_forward`. HDF5 state capture also records all body poses.
The dataset collector retains failed variants, reports overall failure if any
flow fails, and writes a data card. It never converts a short capture smoke test
into a successful task trial.

`pipeline replay ROLLOUT` uses ffmpeg to encode `replay.mp4` from recorded physical
states, including state-only runs. Its overlay and metadata distinguish this
replay from the scripted inspection GIF. It does not rerun or alter the task.

## Measured local evidence

- `outputs/pipeline_agent_probe2`: real unattended Codex generation, one reported
  relation failure followed by a successful program revision (two attempts).
- `outputs/pipeline_probe/kitchen`: compiled/validated template scene, actual
  1024-pixel CPU Cycles still, and independently actuated URDF export.
- `outputs/pipeline_cafe_v2`: second deterministic declarative scene and gallery.
- `outputs/pipeline_retrieved_profiles`: retained retrieved-asset pass/fail reports.
- `outputs/pipeline_dataset_pilot`: ten **one-second** variants; first three RGB-D,
  remaining seven state/range/IMU. Capture plumbing passed; task acceptance did not.
- `outputs/pipeline_mapping_metric`: 15-second mapping probe, approximately 0.466
  IoU, 0.991 observed reachable coverage. Below target and explicitly failed.
- `outputs/pipeline_contact_roundtrip`: 60-second contact manipulation passed;
  open error 2.49 mm, final closed error 0.061 mm, no physics warnings. Includes
  recorded-state `replay.mp4`; consult its report rather than older, weaker probes.
- `outputs/pipeline_current_cafe`: regenerated with the current taxonomy/adapter;
  independent export passed five motor-driven joints and mass/inertia/COM checks.
- Regression suite: 44 tests passed, including mapping sensor/oracle regressions, capture-tier reproducibility,
  unnamed robot collider ownership and final-close endpoint checking.

Artifacts are ignored local outputs, not tracked golden fixtures. Tests generate
fresh fixtures. `cost.json` leaves unknown API spend null, never zero. Historical
artifacts can carry an older adapter/taxonomy hash; regenerate to use current code.

## Next release gates

Prioritize the failed mapping baseline, complete spatial/reachability constraints,
and add the actual walk-in/wash categories before claiming the original two
acceptance prompts are supported. Then run full-length variants and held-out
prompts, produce task videos, and verify the remaining cross-engine invariants.
Do not add diffusion rendering, unrestricted in-loop code execution, or G4
deformation as shortcuts around these gates.
