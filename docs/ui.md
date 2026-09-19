# Local pipeline dashboard

From the repository root:

```bash
.venv/bin/python -m scene_pipeline.web
# For NVIDIA EGL camera rendering:
MUJOCO_GL=egl .venv/bin/python -m scene_pipeline.web
```

Open **http://127.0.0.1:8765**. `--port` changes the port; `--root` selects the
repository. `pipeline-ui` is also installed by the next environment sync. No Node
build, frontend dependencies, or additional Python packages are required.

The dashboard lists scenes under `outputs/`, preferring `best_layout` when present
and hiding attempts, asset packages and work directories. It shows scene previews,
floor plans, validation, agent logs, recorded replays and occupancy maps. Reports
and HDF5 recordings can be opened or downloaded. A partial scene remains marked
partial; completing a render does not make the underlying scene valid.

## Create a scene

Choose OpenRouter, Codex CLI, or supplied SceneProgram JSON. Supported credentials
are read as data from the root `.env`, with existing environment values taking
precedence. Keys stay on the server. Restart after changing credentials. Codex
uses the installed CLI/login; leave its model blank to use the CLI default.

Generation exposes layout seed, total attempt count (default five, including the
first), and optional fal textures/clutter with a separate budget.

**Codex follows the repository [pipeline skill](../skills/scene-pipeline/SKILL.md)**
in one tool-capable session. The launcher reads the current skill, includes it in
the agent prompt, and saves `skill_snapshot.md` and `agent_prompt.txt` with the run.
The agent works from the repo, inspects the shared asset library, writes scene JSON,
runs the ordinary supplied-program CLI, reads reports, and repairs failures. It is
instructed to prefer existing assets and reuse identical selections. The CLI enforces
the shared attempt limit, original scene requirements, appearance settings and
remaining fal budget. Attempts use seeds separated by three. The agent has workspace
write access and network access for source/asset retrieval, with instructions to
write scene inputs and caches rather than modify repository code or past scenes.
The launcher independently revalidates the selected result before publishing it.

OpenRouter continues to use the structured-JSON adapter and its existing repair
loop. It does not run a shell agent or execute the skill. Supplied JSON builds once.
The queue shows the current attempt and stage; `progress.json` and `cost.json`
record the outcome and stop reason. `agent_calls.json` records the Codex session's
usage, and `agent_trajectory.jsonl` retains its redacted tool events, including on
timeout. Missing tools, provider failures, exhausted budgets and the 20-minute
timeout can stop work early. Partial scenes remain labeled partial.

The same Codex skill workflow is available from the terminal:

```bash
.venv/bin/python -m scene_pipeline.cli generate --use-skill --agent-backend codex \
  --prompt 'A practical small kitchen' --output outputs/kitchen_skill \
  --seed 42 --max-iterations 5 --timeout 1200
```

As with other direct CLI commands, configure resource/cache paths and credentials
in the environment; `.env` loading and repo-local cache defaults belong to the UI.

OpenRouter model cost limits are shared across all attempts and checked between
requests. Codex CLI does not report billed USD, so UI Codex runs use the attempt
and time limits instead of an unenforceable dollar budget; token usage is retained.
fal limits are list-price estimates, not hard account billing limits.
JSON mode runs once and makes no nested model call; its description must match the JSON prompt.
The worker generates, renders any loadable result, and refreshes the gallery.
New scenes go into `outputs/ui/scene_<job-id>/`.

## Run a behavior

- **Explore & map:** the stock omnidirectional base builds a range occupancy map.
- **Open & close a drawer:** the Franka Panda targets a selected drawer unit.
  The existing sequence needs at least 60 simulated seconds. Target selection
  reorders a private IR lookup copy; compiled geometry and manifest are unchanged.
- **Navigate to an object:** the same omnidirectional base uses the existing
  semantic goal planner. This is not a VLM policy or live teleoperation.

Choose simulated duration, capture tier and seed. Full capture includes RGB-D and
semantic cameras; state capture omits those camera streams. Both request physical
state replay video. Recorded output goes under the scene's `runs/` directory.
Frames/maps appear when their artifacts are ready, not as a live simulator feed.
Partial scenes can be tested, but scene and run checks are reported independently.

## Jobs and local access

One worker runs one job at a time. Queue status, stage and CLI logs are available
in the UI. Cancellation terminates the child process group; it retains artifacts
and cannot undo provider charges already incurred. Active jobs are marked
interrupted after a restart and never automatically resumed. Job metadata and
redacted log access live under `outputs/.ui/`; that directory is not served as
artifacts. This is a local demo server, bound only to 127.0.0.1, with same-origin
submission protection and no arbitrary shell-command endpoint. Do not expose it
as a public service.

Existing failed runs and models with unresolved access checks remain useful for
demonstrations. A replay is recorded physical motion; the separate articulation
preview in the pipeline gallery is a kinematic sweep.

Validation: `.venv/bin/python -m pytest -q tests/test_web.py` exercises job input,
model arguments/budgets, JSON mode, path confinement, unchanged drawer targeting,
credential redaction, queue failure handling, cancellation and restart behavior.
