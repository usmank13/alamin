# Running and reproducing the pipeline

Run the commands below from the repository root. The core CLI works without an
agent: an external agent or person writes a SceneProgram, and deterministic tools
build and check it. Prompt-only generation is an optional adapter described later.
Use fresh output directories for generation, dataset collection and robot captures.
A failed run remains evidence; fix inputs and write a new run instead of editing
compiled geometry or weakening its checks.

Quick links: [setup](#setup-and-first-scene), [capture formats](#capture-formats),
[GPU rendering](#rendering-on-cpu-or-gpu), [portable agent workflow](#portable-agent-workflow),
[reproduction](#resources-caches-and-reproduction), [home-kitchen batch](#reproduce-the-primary-home-kitchen-batch),
[fal decoration](#optional-fal-textures-and-clutter),
[optional model backends](#agent-backends).

## Setup and first scene

On Linux, install Git, uv, and your system rendering libraries. Ubuntu package
names are `libosmesa6` for software rendering, `libglfw3` for the desktop viewer,
and `ffmpeg` for videos. Setup does not install system packages or GPU drivers.
Python 3.12 and dependencies come from the committed `uv.lock`.

```bash
bash scripts/setup_pipeline.sh --minimal
source .venv/bin/activate
pipeline doctor --smoke
pipeline generate --program examples/cafe_program.py \
  --prompt 'A compact cafe kitchen with a prep table, three cabinets, two drawer units, a storage shelf, and eight containers.' \
  --seed 1 --output outputs/first_scene
pipeline inspect outputs/first_scene --no-open
```

Open the printed gallery path. This fixture requires no credentials, retrieved
asset catalog or robot downloads. It verifies deterministic generation with a
supplied program; it does not evaluate natural-language interpretation.

For robot flows and Cycles rendering, install the additional resources:

```bash
bash scripts/setup_pipeline.sh --blender
```

Without `--minimal`, setup fetches stock robots from a pinned MuJoCo Menagerie
revision. `--blender` fetches the Blender binary; an existing `blender` on PATH also
works. Other setup flags are `--retrieval`, `--agent`, `--prototypes`, and `--test`.
`doctor` lists missing optional tools without making them mandatory for templates.
Network access is needed for dependency/resource downloads.

Richer programs may require RoboCasa fixtures or retrieved assets; these are not
included by default. Fetch fixtures with `python prototypes/fetch_robocasa.py` and
verify them with `python prototypes/fetch_robocasa.py --verify`; read the source and
license notes in [ROBOCASA_DATASET.md](../prototypes/ROBOCASA_DATASET.md). Static
catalog acquisition is described under [retrieval](#retrieve-and-insert-an-asset).
Floorplan corpora are also explicit acquisitions, never automatic generation inputs.

## Scene → render/export → robot capture → variants

Continue with `outputs/first_scene` from the quick start:

```bash
pipeline render outputs/first_scene --view overview --resolution 1024 --samples 32
pipeline export outputs/first_scene --format urdf --verify
pipeline run outputs/first_scene --flow mapping --seconds 60 \
  --tier full --seed 0 --output outputs/first_mapping
pipeline run outputs/first_scene --flow interaction --seconds 60 \
  --tier full --seed 0 --output outputs/first_interaction
pipeline dataset outputs/first_scene --variants 10 --start-seed 100 --seconds 60 \
  --output outputs/first_variants
pipeline inspect outputs/first_variants --no-open
pipeline costs outputs/first_scene outputs/first_mapping outputs/first_interaction \
  outputs/first_variants --output outputs/first_costs.md
```

`render` produces `render/cycles.png` from the compiled scene using CPU Cycles.
`export --verify` tests the independent URDF package in PyBullet. Mapping uses the
stock small omnidirectional base; interaction runs the Panda against a qualifying
drawer. An arbitrary scene may lack the geometry needed by a particular flow.
`--seconds` is simulated time, not a wall-time limit. `--no-video` skips encoding;
`pipeline replay ROLLOUT_DIRECTORY` regenerates a video from recorded states.

The dataset collector replays saved programs, evidence, priors and asset references
without a model call. Seeds resample placement and clutter counts; lighting, tint,
texture repeat and optional fal texture-set seeds vary as recorded in `dataset.json`.
The first three variants have full RGB-D capture, and the remainder use the `state`
tier (state/range/IMU without RGB-D). To capture RGB-D for additional variants, run
`pipeline run VARIANT --flow mapping --tier full --output NEW_DIRECTORY` explicitly.
The collector does not generate variant stills or previews; export them afterward:

```bash
for variant in outputs/first_variants/variant_*; do
  pipeline preview "$variant"
  pipeline render "$variant" --view overview
  pipeline inspect "$variant" --no-open
done
pipeline inspect outputs/first_variants --no-open
```

Read `dataset.json` and each run's `report.json`: a directory or gallery alone does
not establish completion. Failures remain in the batch summary. The local
baseline contains five commercial-kitchen and five warehouse variants. The primary
collection now targets ten home-kitchen variants; those earlier batches remain
supplementary. Use `--variants 10` on one accepted scene for the single-scene
requirement, and consult the published checklist for verified completion.

Optional semantic navigation uses the same recorder:

```bash
pipeline run outputs/first_scene --flow navigate --goal 'go to the prep table' \
  --policy planner --seconds 60 --output outputs/first_navigation
```

See [bonus features and measured examples](bonuses.md) for semantic-navigation
results, goal-resolution behavior, action chunks and the Stretch mobile manipulator.
The planner resolves the target through the semantic manifest. `--policy vlm`
uses the configured model backend and can incur API costs; it is not needed for
mapping, drawer interaction or ordinary planner navigation.

## Capture formats

A flow directory contains `data.h5`, `rig.json`, `report.json`, a portable
`rollout_scene.mjz`, and, when enabled, `replay.mp4`. Mapping adds reconstruction
maps and ground-truth comparison metrics. `rig.json` and HDF5 metadata describe
sensor rates, frames, units and synthetic noise. Ground truth and noisy observations
are separate; this is not hardware-calibrated sensing.

```python
from scene_pipeline.dataset import inspect, load_stream

path = "outputs/first_mapping/data.h5"
print(inspect(path))  # validates clocks/field lengths and lists available streams
samples = load_stream(path, "state", 0, 10)  # sample indices 0 through 9
print(samples["time"])
```

`start` and `stop` are sample indices, not seconds. Different-rate streams must be
joined by their `time` fields, never by array index. The mapping report distinguishes
visibility coverage from physical robot traversal.

The separate `sim` diagnostic command writes final-frame arrays at the simulation
`capture_time` in `report.json`: `depth.npy` is metric depth; `segmentation.npy`
is int32 `[height,width,2]` with MuJoCo `(object_id, object_type)` and background
`(-1,-1)`. Only `mjOBJ_GEOM` IDs index the ground-truth geometry list. These are
model-local IDs, not semantic class labels. `state.npz` stores time, qpos, qvel,
controls, activations and world body poses; read it with `allow_pickle=False`.
`ground_truth.json` records wxyz quaternions, masses/inertias, conservative geometry
AABBs, joints and controls. Call `mj_forward` before a live
`sim_harness.ground_truth.snapshot(model, data)`.

## Rendering on CPU or GPU

MuJoCo physics stays on CPU. MuJoCo's offscreen camera rendering uses software
OSMesa by default; a working NVIDIA driver and accessible GPU can use EGL:

```bash
MUJOCO_GL=egl pipeline doctor --smoke
MUJOCO_GL=egl pipeline run outputs/first_scene --flow mapping --seconds 60 \
  --output outputs/first_mapping_gpu
```

Set `MUJOCO_GL` before Python imports MuJoCo. `nvidia-smi` checks driver visibility;
a successful EGL smoke test checks context creation but does not identify the
rendering device. Containers/sandboxes also need GPU device and driver-library
access. Identify the actual EGL renderer with:

```bash
MUJOCO_GL=egl python - <<'PY'
import mujoco
from OpenGL import GL
context = mujoco.GLContext(16, 16)
context.make_current()
print(GL.glGetString(GL.GL_VENDOR).decode())
print(GL.glGetString(GL.GL_RENDERER).decode())
context.free()
PY
```

Physics, HDF5 writes and
ffmpeg encoding still consume CPU; benchmark a representative capture before
extrapolating throughput. The current Cycles bridge explicitly selects CPU and
has no GPU CLI switch. Desktop inspection uses GLFW and needs a display.

## Portable agent workflow

The canonical skill is [`skills/scene-pipeline/SKILL.md`](../skills/scene-pipeline/SKILL.md).
Any agent that can read instructions and execute commands can use it. Point your agent at this tracked file, or copy/link its
folder into your harness's skill location; automatic discovery depends on the harness.
The instructions require this repository and resolve examples/docs from its root,
so preserve that association if the skill is installed elsewhere. This repository does not install itself into global agent settings.

An external agent can run `pipeline registry`, write a SceneProgram JSON, then call
`pipeline generate --program ...`. That path never starts a nested agent. For novel
categories, provide user-quoted, axis-labeled dimensions or a URL with complete
width/depth/height fields. A caller cannot bypass evidence verification by supplying
`dimension_basis`. JSON/DSL inputs are data, never executed Python.

## Visual inspection

`inspect` accepts a scene, retrieved asset, evaluation batch, or completed dataset.
Dataset pages link each variant's checks, HDF5 and recorded mapping video.

```bash
pipeline inspect /absolute/path/to/scene_or_evaluation
pipeline inspect /absolute/path/to/scene --no-open   # print page path only
pipeline inspect /absolute/path/to/scene --viewer    # real MuJoCo, GLFW
pipeline preview /absolute/path/to/scene
pipeline render /absolute/path/to/scene --view overview
pipeline replay /absolute/path/to/rollout
```

HTML pages are static and work without a server. They show available perspective,
plan and provenance views, validation/failure reports, Cycles stills, and task
videos. Missing artifacts are not linked as if they existed. Plan views are
bounding-footprint diagnostics, not collision proof. The animation is a kinematic
sweep, **not** a robot interaction. Replay reads recorded physical states; launching
the simulator alone starts simulation and does not replay a controller.

## Resources, caches and reproduction

Installed commands work outside the checkout with absolute scene/output paths.
Global options belong **before** the subcommand:

```bash
pipeline --resource-root /absolute/path/to/repo \
  --cache-dir /absolute/path/to/cache \
  --asset-store /absolute/path/to/assets doctor --smoke
```

| Resource | Default / override | Needed for |
| --- | --- | --- |
| Python packages | `.venv`, locked by `uv.lock` | All commands |
| Stock robots, fixture sources, Blender | `vendor/` under repo; `--resource-root` or `SCENE_PIPELINE_RESOURCE_ROOT` | New flows, source-based generation, Cycles |
| Evidence and fal downloads | `$XDG_CACHE_HOME/scene-pipeline` or `~/.cache/scene-pipeline`; `--cache-dir` or `SCENE_PIPELINE_CACHE` | Verified source reuse; cached API assets |
| Static/generated asset packages | `$XDG_CACHE_HOME/scene-pipeline/assets` or `~/.cache/scene-pipeline/assets`; `--asset-store` or `SCENE_PIPELINE_ASSET_STORE` | Asset search/import and referenced generation |
| Generated artifacts | Explicit `--output` | Scenes, runs, datasets and reports |

The fal cache is the `fal/` child of the configured cache root; dimension evidence
uses `dimensions.json`. Evidence records archive source text, fields and hashes.
The decoration driver below uses its own `--cache`, pointing directly to the fal
folder; `restyle_scene.py` defaults to `vendor/fal_cache/fal` unless
`SCENE_PIPELINE_CACHE` is set. Caches are not committed. Keep secrets in process environment variables or
an ignored `.env`, never in ScenePrograms, source control or exported artifacts.

A scene's `scene.mjz` embeds compiled geometry and assets; loading it needs no vendor
tree. Preserve the entire scene folder for validation, galleries, export and dataset
regeneration: program/intent, IR, semantic manifest, provenance, checks, assets,
generation configuration, evidence, and any prior bundle are part of that contract.
An MJZ alone cannot regenerate the authoring process. Existing rollout archives plus
HDF5 states can replay without re-running a controller; new flows need stock robots.

For reproduction, retain the Git revision, `uv.lock`, source program and prompt,
seeds/configuration, asset packages and hashes, prior/evidence snapshots, cached fal
responses and rendering-device details. Frozen inputs make reruns inspectable;
live models, mutable providers, different drivers and simulator/renderer versions
can still change output. A seed does not guarantee the same future API result.
`fal_calls.json`, `agent_calls.json` and `cost.json` distinguish reused assets from
new spend; missing provider billing remains unknown. Do not describe cache hits as
the total acquisition cost.

Fresh-machine timing under fifteen minutes and three unseen prompt-only evaluations
have **not** been certified. Existing fixed-program and warm-cache runs establish
composition and capture behavior, not those acceptance claims. See
[brief-status.md](brief-status.md) and the local deliverable checklist for evidence.

## Retrieve and insert an asset

This is a direct tool workflow for any command-capable agent, not a new agent
framework. Install `--retrieval` alongside whichever other setup extras you use:

```bash
bash scripts/setup_pipeline.sh --minimal --retrieval
pipeline --asset-store vendor/asset_library asset-index
pipeline --asset-store vendor/asset_library asset-search 'pallet'
pipeline --asset-store vendor/asset_library asset-fetch gazebo:euro_pallet --category pallet
```

`asset-fetch` returns `asset_ref`, dimensions, capabilities and a package path.
Inspect that path, then put `asset_ref` on the corresponding SceneProgram object.
Alternatively, an agent can supply a selected `asset_request.candidate_id` and
query, as in [warehouse_program.py](../examples/warehouse_program.py).
Use the same asset store for generation. A supplied program never starts a nested
agent; query-only selection is a convenience for the configured runtime.

The current catalog indexes pinned OSRF Gazebo and AWS house/warehouse/hospital model
repositories. These are downloaded data, not installed Gazebo/ROS executables.
The importer supports single-link **static rigid SDF props**, with source textures
and separate primitive/decomposed colliders. It rejects articulated models rather
than freezing them silently. Existing RoboCasa articulation remains available via
the registry. Native source scale is not a manufacturer measurement; do not override
dimensions on an asset reference. Packages carry provenance and source licenses
into scenes and URDF exports. A bowl with a solid source collider is not a verified
container; a static pallet jack is not a lifting mechanism.

The registry is not a closed vocabulary for generated dressing. `asset-generate`
accepts any category and registers its generated visual in the same asset library:

```bash
pipeline asset-generate --category tea_towel --prompt 'a folded cotton tea towel' \
  --size-m .25 --placement support --max-fal-usd .4
pipeline asset-search 'tea towel'
```

Set `FAL_KEY` or `FAL_API_KEY` in the environment for direct CLI calls. Use the
returned `asset_ref` in subsequent scenes; it replays without a fal call. The
one-prompt runner also accepts agent-created `generated_request` objects containing
`prompt`, `size_m`, `placement` and `physical_use: "visual_only"` when clutter is
enabled. The estimated largest extent is labeled unverified; no collision geometry,
support surface or articulation is invented. The importer explicitly converts GLB
Y-up into MuJoCo Z-up. Source prompts, hashes and provider request IDs travel with
the package. Retrieval selection can decline all candidates; an unrelated room
keyword no longer forces a match. This is not deterministic semantic verification.

Run the small public-asset exercise (no model calls) with:

```bash
.venv/bin/python scripts/exercise_asset_retrieval.py --output outputs/assets_check \
  --priors outputs/architecture_grounding/frozen_corpus/priors.json
pipeline inspect outputs/assets_check
```

Omit `--priors` for asset-only tests. Failures stay in the report and return exit 1.

## Optional fal textures and clutter

For an authored program that includes dressing, enable fal explicitly:

```bash
# Export FAL_KEY or FAL_API_KEY in your shell first.
pipeline generate --program examples/pbr_kitchen_program.py \
  --prompt 'A 60 square metre working prep kitchen with countertop clutter.' \
  --seed 17 --materials fal --clutter fal --max-fal-usd 6 \
  --output outputs/decorated_kitchen
```

This richer fixture requires its source assets to be available; inspect its program
and registry entries before running it. PATINA supplies PBR finishes and Hunyuan3D
supplies visual-only clutter. Clutter has no collision geometry and cannot serve as
a manipulation target. Both MuJoCo and Cycles consume the compiled material/mesh
assets; RGB-D must be re-recorded after changing them. Existing captures cannot be
made consistent with new textures by replacing thumbnails or replay videos.

Uncached requests need credentials and can cost money. `--max-fal-usd` checks
estimated list-price spend, not an account billing ceiling. Request-digest cache
hits avoid repeat charges only for matching cached jobs. Dataset collection from
a fal-enabled scene can fetch new texture seeds; review its inherited budget and
cache before starting. Direct `pipeline` commands read environment variables;
`create_scene.py`, `restyle_scene.py` and `decorate_variants.py fetch` also read
supported credentials from `.env` as data.

For one floor finish on a saved scene, preserving layout:

```bash
python scripts/restyle_scene.py outputs/first_scene --output outputs/first_oak \
  --floor-prompt 'Seamless natural oak plank flooring, matte finish, neutral illumination' \
  --tile-m 2 --max-fal-usd 0.10
```

`--tile-m` sizes the whole repeated patch, not an individual plank. This runner
writes new validation, previews and a Cycles still; it does not re-record a flow.

For the local **five kitchen + five warehouse** collection, the dedicated driver
adds distinct themed finishes and seeded prop selection/placement without moving
original physical objects. It requires that source collection, not just a checkout:

```bash
python scripts/decorate_variants.py plan --source deliverables/outputs \
  --output outputs/decorated_variants --cache vendor/fal_cache/fal
# Inspect outputs/decorated_variants/plan.json before the paid fetch.
python scripts/decorate_variants.py fetch --output outputs/decorated_variants \
  --cache vendor/fal_cache/fal --max-fal-usd 8
python scripts/decorate_variants.py prepare --output outputs/decorated_variants \
  --cache vendor/fal_cache/fal
MUJOCO_GL=egl python scripts/decorate_variants.py capture \
  --output outputs/decorated_variants --domain kitchen --cache vendor/fal_cache/fal
MUJOCO_GL=egl python scripts/decorate_variants.py capture \
  --output outputs/decorated_variants --domain warehouse --cache vendor/fal_cache/fal
```

Use `MUJOCO_GL=osmesa` when EGL is unavailable. The driver records source IR hashes,
material requests, decoration seeds, physical-invariant comparisons, fresh scene
checks, and new 60-second captures. It retains the source batch's three full/two
state tiers. Successful stages can be reused in the same staging directory; an
interrupted capture without a completed report must be moved aside before retrying.
The plan stores absolute source paths: do not relocate its source mid-run. Start
from an undecorated source snapshot to repeat the pass; applying it to the published
decorated variants is not a clean reproduction. The source tree stays unchanged,
and publishing/packaging the staging results is a separate step.

## Reproduce the primary home-kitchen batch

[`scripts/collect_home_kitchen.py`](../scripts/collect_home_kitchen.py) builds the
single-scene primary dataset from a saved home kitchen and existing fal downloads.
It requires the source scene's authoring files (`input_program.json`, `program.json`,
`ir.json`, `evidence_cache.json` and `assets/`), the applicable vendor resources,
Blender, stock robots and ffmpeg. The published base home kitchen is the default
source; a source `.mjz` alone is insufficient. This is replay of a saved program,
not a fresh prompt-only experiment.

The helper imports kitchen theme descriptions from
[`scripts/decorate_variants.py`](../scripts/decorate_variants.py); keep both scripts
in the checkout. `--cache` points directly to the fal cache folder, normally
`vendor/fal_cache/fal`. The five required PBR themes and all other requests needed
by source regeneration must already be cached. There is no paid fetch stage:
missing cached textures stop preparation, model calls are bypassed by the supplied
program, and the generation spend allowance is zero. This means **zero new paid or
model calls for replay**, not zero historical asset-acquisition cost. Copy the
verified cache from the producing environment when reproducing elsewhere.

```bash
python scripts/collect_home_kitchen.py prepare \
  --source deliverables/outputs/home_kitchen/scene \
  --output outputs/home_kitchen_reproduction --cache vendor/fal_cache/fal \
  --start-seed 300
MUJOCO_GL=egl python scripts/collect_home_kitchen.py capture \
  --source deliverables/outputs/home_kitchen/scene \
  --output outputs/home_kitchen_reproduction --cache vendor/fal_cache/fal \
  --start-seed 300
python scripts/collect_home_kitchen.py finalize \
  --source deliverables/outputs/home_kitchen/scene \
  --output outputs/home_kitchen_reproduction --cache vendor/fal_cache/fal \
  --start-seed 300
pipeline inspect outputs/home_kitchen_reproduction --no-open
```

Use `MUJOCO_GL=osmesa` when EGL is unavailable. `prepare` generates and validates
all ten layouts and writes previews, articulation animations, 1024 px Cycles stills,
and `variant.json` records. Indices 0–9 use layout seeds 300–309; source clutter
counts/placement, light intensity and texture repeat vary. Five cached fal kitchen
themes are reused across distinct layouts; they are not ten new API generations.
The source's generated props retain their visual-only status.

`capture` records 60 simulated seconds per variant, with full RGB-D for indices
0–2 and state/range/IMU for 3–9. It writes rendering-device metadata into completed
reports. For concurrent capture after preparation, start workers in separate shells
using **disjoint** index sets and the same source/output/cache/seed settings:

```bash
# Worker A
MUJOCO_GL=egl python scripts/collect_home_kitchen.py capture \
  --source deliverables/outputs/home_kitchen/scene \
  --output outputs/home_kitchen_reproduction --cache vendor/fal_cache/fal \
  --start-seed 300 --indices 0 1 2 3 4
# Worker B, in another shell
MUJOCO_GL=egl python scripts/collect_home_kitchen.py capture \
  --source deliverables/outputs/home_kitchen/scene \
  --output outputs/home_kitchen_reproduction --cache vendor/fal_cache/fal \
  --start-seed 300 --indices 5 6 7 8 9
```

Do not run these workers alongside an unpartitioned capture of the same output.
`--indices` also restricts preparation, useful for a specific retry; run initial
preparation in one process so shared source packages are initialized once. Capture
can wait for a prepared variant, but workers must never write the same index.
Concurrent workers share CPU, GPU, memory and disk bandwidth; concurrency is not
an assurance of higher throughput.

After **all** capture workers finish, run the unpartitioned `finalize` command.
It always checks indices 0–9, irrespective of `--indices`, requires passing scene
and capture records and 60-second durations, checks HDF5 clock/field alignment,
then writes `dataset.json`, `DATA_CARD.md` and the batch gallery. A prepared scene
or a running capture is not a completed dataset.

Use a new output root for changed source programs, seeds, themes or code. Successful
preparation is skipped when `variant.json` exists, and capture is skipped when
`mapping/report.json` exists; this is stage reuse, not configuration-identity
validation. Inspect those records before resuming. Preserve/move aside an incomplete
scene or mapping directory before retrying its index; do not silently overwrite
failure evidence. Final publication into `deliverables/` remains a separate step.

## Agent backends

For a complete prompt-to-scene run (agent generation, deterministic validation,
Cycles still and HTML inspection page), use the thin wrapper:

```bash
.venv/bin/python scripts/create_scene.py 'A standard tiled home kitchen with everyday countertop clutter.' \
  --materials fal --clutter fal --max-fal-usd 5 --output outputs/my_home_kitchen
```

Only the prompt is required; otherwise a timestamped output directory is created.
The wrapper reads supported API credentials from the repo `.env`, without executing
it. Existing environment variables take precedence. Fal is opt-in and paid;
generated clutter is visual-only, not contact geometry. The cap is a list-price
estimate, not an account billing limit. Default agent authentication is your Codex
CLI login; `--agent-backend openrouter --model PROVIDER/MODEL` is also supported.
No authored SceneProgram is supplied: the agent creates it internally. Attempts,
failures and costs are retained, and a partial/failed run returns a nonzero status.
Repairs are small patches: unchanged object requests retain their asset/evidence
bindings, and failed retrieval queries are remembered for the rest of the run.
The heuristic layout tries up to three deterministic placements before asking the
agent to repair. The agent can revise room fields explicitly marked as its own
`space.inferred_fields`, and change the asset used for a required category; it
cannot drop required inventory, change quoted dimensions or weaken required
relations. Supplied programs and unannotated room fields remain fixed. Initial
prompt interpretation is still agent-authored, not independently certified.

Before fal spending, `attempts/N/preflight/report.json` records a layout check of
known geometry and the presence of supports for required generated clutter.
This does **not** certify the size/fit of a mesh that has not been generated, or
physics stability. Actual generated geometry still goes through the full checks.
Repairs are recorded in `attempts/N/repair.json`; placement trials are in
`layout_candidates.json`. There is no cross-run resume or guaranteed completion
when assets/providers are unavailable.

Default layout is heuristic; empirical conditioning requires `--layout-backend
empirical --priors PATH`. A successful physics check does not establish realism
or distribution matching. Rendering does not launch a robot rollout or dataset job.

The default remains an authenticated Codex CLI. For a standalone API-backed runner:

```bash
bash scripts/setup_pipeline.sh --minimal --agent
# Set OPENROUTER_API_KEY in your environment; do not put it in programs or logs.
.venv/bin/pipeline generate --agent-backend openrouter --model PROVIDER/MODEL \
  --prompt 'A 60 square metre workshop with three base cabinets, two drawer units, a prep table and a shelf.' \
  --max-iterations 3 --timeout 900 --max-cost-usd 5 --output outputs/api_workshop
```

The optional dependency is Pydantic AI's OpenRouter adapter, isolated behind
`scene_pipeline.runtime.Runtime`. Intent, dimension sourcing and optional VLM
navigation use the same boundary. There are no arbitrary code/shell tools in the
API worker. Local JSON Schema validation runs even when a provider advertises
structured output. Search results are evidence proposals, not verified dimensions.
The pipeline independently fetches and verifies complete fields.

Provider calls run in bounded subprocesses. The generation deadline includes nested
sourcing and retries; synchronous geometry/physics stages are checked at stage
boundaries and are not preempted. Calls, usage and reported cost are saved in
`agent_calls.json`; unknown cost stays unknown. Cost limits stop **between calls**,
and stop subsequent calls if billed usage is unavailable. They cannot guarantee a
hard billing ceiling; use an account/key credit limit for that. Search capability,
search limits and billing vary by provider. Never silently substitute another model.

Offline transport testing uses `--agent-backend recorded --responses FILE`.
The file is a list of `{ "kind": "program", "output": { ...SceneProgram... } }`
entries, followed by `dimension_evidence` or `chunk` entries when those calls are
expected. Repairs use `{ "kind": "repair", "output": { "objects": [...] } }`:
objects are complete replacements keyed by ID or additions; omitted objects stay
unchanged. Optional `remove_objects`, `space`, `relations` and `architecture`
fields remove IDs or replace sections. Intent checks apply after merging.
This tests orchestration, not model capability. Python callers can inject
`Runtime(backend='recorded', responses=[...])` into `generate(..., runtime=...)`.

## Cross-domain evaluation

Freeze a prior bundle first using the existing `architecture-fit` command and real
SVG annotations from `scripts/fetch_floorplan_sample.py`. A supplied bundle records
sources, partitions, licensing and scale limitations; no empirical claims are made
for synthetic unit-test fixtures. The existing local bundle, when present, is
`outputs/architecture_grounding/frozen_corpus/priors.json`.

```bash
pipeline evaluate examples/evaluation.json --priors PATH_TO_PRIORS.json \
  --output outputs/domain_regression --flows --no-cycles
pipeline inspect outputs/domain_regression

# Eight live runs: two models, four domains, one seed. Requires key and spend cap.
pipeline evaluate examples/evaluation_domains.json --priors PATH_TO_PRIORS.json \
  --live --models PROVIDER/MODEL_A PROVIDER/MODEL_B --seeds 7 --max-cost-usd 10 \
  --output outputs/domain_models
```

The default regression matrix is four domains × two backends × two seeds. The live
smoke uses architecture generation, explicit generic backoff, Cycles, export and
short mapping, plus the first qualifying drawer interaction in each domain.
`--no-cycles` explicitly skips path tracing. Regression runs can opt into full
interaction with `--interactions`. Accepted scenes alone proceed to robot tasks.

`evaluation.json` deliberately shares known storage mechanics plus one novel dimensioned
object. It measures composition and proxy handling, **not realistic domain-specific
equipment coverage**. Its supplied dimensions do not benchmark live web sourcing.
`evaluation_domains.json` instead contains live-only kitchen, clinic, electronics,
and warehouse/material-handling prompts. The warehouse requests racks, pallets,
cartons, carts, a pallet jack, packing and staging—not a cabinet quota. These are
coverage probes: the current library does not implement all requested equipment.
Capability probes are reported separately as `unsupported_or_unverified`, even if
labeled proxy geometry compiles. Warehouse rolling/lifting probes are diagnostic,
not acceptance requirements for a static scene. Explicit `required_capabilities`
gate task-focused cases; `capability_gap` is not a successful required-task result. Only
hinged/sliding affordance presence currently has a deterministic coverage check;
it is not proof of task success. Inventory labels are strict, authored categories;
synonyms may require a reviewed benchmark update, not automatic substitutions.
Spatial workflow and domain-specific functional checks remain incomplete.

Expectations are authored separately and never shown to the
model. Each run starts from the same evidence-cache snapshot; none can warm another
model's cache. `--cache-snapshot FILE` supplies verified archived evidence. These
mechanical fixtures have separate field-binding/transport tests for sourcing.

`--resume` reuses completed runs only when frozen inputs, code and settings match.
Interrupted scenes are preserved and the run gets a new attempt directory. A changed
experiment requires a new output root. `evaluation.json` includes every failure;
`index.html` compares all runs. Exit code 1 means at least one run/stage failed, not
that the report could not be produced. Paid tests are never part of pytest.

## Tests and remaining limits

```bash
.venv/bin/python -m pytest -q
pipeline doctor --smoke
```

See [brief-status.md](brief-status.md) for current evidence and release gaps.
No fixed IoU threshold is imposed. Generated aperture widths remain proportional
to uncalibrated source floorplans, and furnished layouts are heuristic. A physics
pass does not establish functional suitability, medical/industrial compliance,
or broad distributional generalization.
