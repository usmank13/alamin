---
name: scene-pipeline
description: Build and inspect grounded robotics workspaces with this repository's scene pipeline, including asset selection, declarative scene programs, validation, rendering and dataset capture. Use for prompt-to-scene tasks or diagnosing generation reports, not general MuJoCo code changes.
---

# Scene pipeline

This skill is harness agnostic: it needs a shell, file access and the installed
repository CLI. It requires no agent-specific tools, session API or nested model.
Locate the repository root (contains `pyproject.toml`, `src/scene_pipeline/` and
`examples/`); run commands there. Use `.venv/bin/pipeline` or activate `.venv`.
If this skill is installed elsewhere, locate the checkout before resolving the
repository-relative references below. Do not assume Codex is installed.

Read `docs/running.md` for setup, flow/capture commands, optional resources and
reproduction. `docs/brief-status.md` distinguishes evidence from remaining limits.
Neither private `docs/agent/` notes nor a particular harness is required.

## Build from intent

The agent chooses inventory, requirements, asset candidates and relationships;
the tools resolve dimensions, geometry, placement, physics and compilation.

1. Run `pipeline doctor --smoke` when installation is uncertain. Discover supported
   categories and declared defaults with `pipeline registry`; the registry is not
   a closed list of allowed content.
2. Write a SceneProgram JSON or a single literal expression using `scene`, `space`,
   `place` and `relation`. Start from `examples/cafe_program.py`; see
   `src/scene_pipeline/dsl.py` for signatures and `src/scene_pipeline/contracts.py`
   for the JSON schemas. Python-shaped inputs are interpreted as data: imports,
   arbitrary expressions and executable Python are rejected.
3. Generate into a new directory:

   ```bash
   pipeline generate --program PATH --prompt 'USER REQUEST' --seed 17 --output NEW_DIRECTORY
   ```

   A supplied program makes no nested agent calls. Resolve asset selection and
   dimension evidence explicitly before using it. Keep the program's prompt aligned
   with the request. Omitting `--program` selects an optional runtime backend and
   may invoke a model; that is a separate workflow, not a prerequisite for this skill.
4. Read `validation.json`, `scene_checks.json`, `cost.json`, `manifest.json` and provenance, then
   inspect previews with `pipeline inspect NEW_DIRECTORY --no-open`. A `partial`
   result and `unmet.json` describe a loadable but unaccepted scene. Revise the
   declarative input or selected asset and use a fresh directory on failure. Do not
   edit generated MJCF or relax validators to manufacture a pass.

Represent arrangements with relations such as `against_wall`, `near`, `in_row` and
`under`. Preserve user-required inventory, quoted sizes and required relations;
mark a relation required only when the user requires it. Geometry checks cannot
certify semantic identity or an object's intended function.

## Choose assets and dimensions

For retrieved geometry, select/import directly rather than delegate to another agent:

```bash
pipeline asset-index                            # downloads the pinned catalog
pipeline asset-search 'pallet'
pipeline asset-fetch gazebo:euro_pallet --category pallet
```

Inspect the returned package with `pipeline inspect PATH --no-open`, then use its
returned `asset_ref` on the SceneProgram object. Keep the same asset store across
commands (`pipeline --asset-store PATH SUBCOMMAND`). Search results are candidates,
not verified semantic matches. Decline unrelated candidates and report gaps.
The SDF importer supports static rigid props; it does not invent joints or certify
manipulation. RoboCasa articulated assets use the registry route and need their
source fixtures. Retain package provenance, licenses and hashes.

For suitable parametric objects, keep the user's category and select a supported
family (`box`, `table`, `shelf`, `cabinet`, `drawer`). Registry dimensions are labeled
engineering defaults, not measurements. For novel dimensions, provide matching
axis-labeled values and the verbatim `dimension_evidence.prompt_quote`, or a source
URL with complete labeled width/depth/height fields. Tools verify external evidence.
Do not set harness-owned `dimension_basis` to assert verification. Native mesh units
are not measured product specifications. Sized proxies do not establish function;
do not substitute unrelated equipment without the user's agreement. Automatic
sourcing is an optional runtime feature; `--program` does not start that runtime.

## Optional generated appearance

When fal use and spending are in scope, missing visual-only dressing can be generated:

```bash
pipeline asset-generate --category tea_towel --prompt 'a folded cotton tea towel' \
  --size-m .25 --placement support --max-fal-usd .4
```

Set `FAL_KEY` or `FAL_API_KEY` in the process environment; direct CLI calls do not
load `.env`. Inspect the result and reuse its `asset_ref`. Alternatively an object
can have `generated_request` with `prompt`, `size_m`, `placement` and
`physical_use: "visual_only"`; enable it with `generate --clutter fal`.
`--materials fal` adds PBR finishes. Use an explicit `--max-fal-usd` budget for
uncached jobs and retain the request/cost records. Matching cached jobs avoid new
spend; provider billing and future output are not guaranteed by a seed.

Generated props have no collision geometry, articulation or support surfaces.
`size_m` is an unverified largest-extent estimate. They cannot satisfy functional
or contact-rich equipment requirements. Mark user-required dressing required so
failure cannot silently remove it. G2 physical asset authoring and part-aware G4
articulation are separate tasks, not a fallback that this skill silently invents.

## Inspect, capture and hand off

Use `pipeline preview SCENE`, `pipeline render SCENE --view overview`, and
`pipeline export SCENE --format urdf --verify` as requested. Cycles consumes the
compiled geometry; MJZ is a MuJoCo bundle, not independent-format export. A preview
animation is a kinematic sweep, not a recorded robot interaction. Desktop
`pipeline inspect SCENE --viewer` needs GLFW and a display.

`pipeline run SCENE --flow mapping --output NEW_RUN --seconds 60` records a flow;
`--flow interaction` needs a qualifying drawer. `pipeline dataset SCENE --output
NEW_BATCH --variants 10 --seconds 60` captures three full RGB-D variants and seven
state/range/IMU variants. Export variant previews/stills separately. Decorate before
capture, and re-record observations if textures/clutter change. Preserve the
boundary between noisy observations and truth, and join streams by timestamps.

For handoff, include portable scenes, input/configuration and evidence, per-run
checks, dataset summaries/data cards, requested renders/videos, and cost records.
Keep debug attempts out of a curated deliverable copy while retaining source-run
failure evidence when available. Report limitations and incomplete runs explicitly.
A supplied-program fixture, recorded-response adapter or warm asset library does
not establish unseen-prompt generalization or a timed cold installation.
