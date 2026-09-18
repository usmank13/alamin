# Running and inspecting the pipeline

The core is a Python library and CLI, not a coding-agent harness. Agents supply
declarative intent and evidence; deterministic tools own dimensions, geometry,
validation and compilation. MuJoCo physics is CPU-based. CUDA is optional, and
Cycles currently uses CPU. No diffusion renderer is used.

## First run without an agent or datasets

On Linux with Git, uv, and OSMesa installed (`libosmesa6` on Ubuntu):

```bash
bash scripts/setup_pipeline.sh --minimal
.venv/bin/pipeline doctor --smoke
.venv/bin/pipeline generate --program examples/cafe_program.py \
  --prompt 'A compact cafe kitchen with three cabinets, two drawer units, a prep table, a shelf and eight containers.' \
  --seed 1 --output outputs/first_scene
.venv/bin/pipeline inspect outputs/first_scene
```

This fixture checks installation; it is not evidence of natural-language generation.
Output directories are never overwritten. Choose another directory for another run.
The scripts resolve their own repo directory. Installed `pipeline` commands work
from other directories with absolute input/output paths. Use
`--resource-root /path/to/repo` before the subcommand, or
`SCENE_PIPELINE_RESOURCE_ROOT`, to locate `vendor/` outside an editable checkout.

`setup_pipeline.sh` fetches stock robots unless `--minimal` is supplied.
Optional flags: `--blender`, `--agent`, `--prototypes`, `--retrieval`, `--test`.
RoboCasa fixtures are a separate explicit acquisition; see
[`prototypes/ROBOCASA_DATASET.md`](../prototypes/ROBOCASA_DATASET.md).
Floorplan ingestion is also explicit; generation never downloads a prior corpus.
`doctor` reports missing optional assets/tools without requiring them for templates.
Viewer dependencies are GLFW plus a working desktop display; video encoding needs
ffmpeg. Setup does not change system packages or GPU drivers.

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

## Portable agent workflow

The canonical skill is [`skills/scene-pipeline/SKILL.md`](../skills/scene-pipeline/SKILL.md).
Any agent that can read instructions and execute commands can use it. Copy/link its
folder into your harness's skill location if desired; automatic discovery depends
on that harness. This repository does not install itself into global agent settings.

An external agent can run `pipeline registry`, write a SceneProgram JSON, then call
`pipeline generate --program ...`. That path never starts a nested agent. For novel
categories, provide user-quoted, axis-labeled dimensions or a URL with complete
width/depth/height fields. A caller cannot bypass evidence verification by supplying
`dimension_basis`. JSON/DSL inputs are data, never executed Python.

The writable dimension cache defaults to `$XDG_CACHE_HOME/scene-pipeline` (or
`~/.cache/scene-pipeline`). Override with global `--cache-dir` or
`SCENE_PIPELINE_CACHE`. New records archive source text, fields and hashes; old
hash-only entries are not silently trusted. Concurrent updates use locking and
atomic replacement. Each scene saves original intent, resolved intent, evidence
and generation configuration so variants can replay without model calls.

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

## Deliverable collection

After a scene validates, the existing collector runs randomized mapping captures:

```bash
pipeline dataset outputs/my_kitchen --output outputs/kitchen_dataset --variants 5 --seconds 60
pipeline dataset outputs/my_warehouse --output outputs/warehouse_dataset --variants 5 --start-seed 101 --seconds 60
pipeline inspect outputs/kitchen_dataset
pipeline inspect outputs/warehouse_dataset
```

This requested split is ten runs across two domains, not the original brief's ten
variants of one scene. Each five-run batch captures RGB-D in its first three runs
and state/range/IMU in the other two (six full, four lightweight in total). Both use
the same simulated clock, loader and data-card format. Keep failed variants visible.
These fixed-program experiments test composition and capture, not an unattended
held-out prompt benchmark. The prompt remains the source of inventory and relations;
do not patch generated geometry to fix a layout.

PBR materials and optional visual-only clutter from `pbr-materials` are merged.
`generate --materials fal --max-fal-usd LIMIT` requires configured fal credentials
and may spend money; the default is still flat materials. Tests use synthetic API
responses, which establish integration, not real generated-texture quality.

To exercise a single fal material on an existing creation without regenerating its
layout, the small runner below reads `FAL_API_KEY` or `FAL_KEY` from `.env` (as data,
not shell code). The source scene stays untouched; output contains the maps, source
request, cost record, fresh validation, MuJoCo preview and 1024 px Cycles render.
The API sees the material prompt only, not your scene files.

```bash
.venv/bin/python scripts/restyle_scene.py outputs/deliverables_cleanup/kitchen \
  --output outputs/kitchen_wood \
  --floor-prompt 'Seamless natural oak plank flooring, matte finish, neutral illumination' \
  --tile-m 2 --max-fal-usd 0.10
pipeline inspect outputs/kitchen_wood
```

`--tile-m` is the physical size of the whole repeated texture patch, not a board's
width. Overrides are stored in this scene's appearance metadata; registry defaults
for other scenes do not change. This is a generated PBR material followed by Cycles,
not a diffusion repaint of the scene. Other pipeline commands accept either key
name in the process environment; only this runner explicitly reads a dotenv file.

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
  --live --models PROVIDER/MODEL_A PROVIDER/MODEL_B --max-cost-usd 10 \
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
