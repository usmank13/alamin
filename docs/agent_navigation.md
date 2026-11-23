# Agentic navigation in MuJoCo

## Actual node: observation and conversation test

`python -m scene_pipeline.rhizome_agent` launches the checkout's actual
`src/ai_agent/ai_agent.main.py` in a separate process, using the normal Rhizome
relay and services. It does not construct an AgentInterface or copy an agent
loop. The selected task must enable `ai_agent`; Rhizome's native NodeConfig
loads its node YAML and task overrides. This is a **stationary observation
milestone**, not yet a launcher for every node in that task.

MuJoCo physics, rendering, and the agent run in independent processes. The bridge
publishes measured pose, camera intrinsics, optical frames and the description
resolved from Rhizome's CAD. Front/rear 640×480 RGB-D uses Rhizome's existing
GStreamer SHM protocol, including 0.1 mm depth units and capture timestamps.
The run records both transmitted images and images decoded by Rhizome's own
`ShmStreamPipeline`. Physics targets wall time. This stationary runner records
scheduling overruns greater than 0.5 seconds and catches up without applying
commands; stale sensor snapshots are dropped. An overrun means navigation
timing is not validated. This is not a pauseable or accelerated simulation clock.

Dependencies in addition to the existing `agent-nav` and pipeline environment:
Linux `bubblewrap`, GStreamer (`shmsrc`, `shmsink`, `appsrc`, `appsink`,
`videorate` and GI typelibs), Python `PyGObject`, and OpenCV. On Ubuntu the SHM
elements are in `gstreamer1.0-plugins-bad`. PyGObject must match the interpreter
running the selected Python 3.12 host extensions. A project-local SHM plugin
installed for the current workstation can be selected with:

```bash
export GST_PLUGIN_PATH="$PWD/outputs/host_runtime_deps/gst${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"
```

Choose the checkout and verified build with `RHIZOME_ROOT` and
`RHIZOME_HOST_BUILD` as described below. The current native CLI helpers can be
built into that same host installation without rebuilding its Python modules:

```bash
make -C "$RHIZOME_ROOT" BUILD_DIR="$(dirname "$RHIZOME_HOST_BUILD")" JOBS=4 CCACHE= \
  "$RHIZOME_HOST_BUILD/bin/aigen_echo" \
  "$RHIZOME_HOST_BUILD/bin/aigen_topics" \
  "$RHIZOME_HOST_BUILD/bin/aigen_services" \
  "$RHIZOME_HOST_BUILD/bin/set_nav_source" \
  "$RHIZOME_HOST_BUILD/bin/set_drive_mux"
```

The runner stages Rhizome's own `aigen` script and these helpers. `status` and
`tail` need the real systemd/journal environment and are unsupported here;
control services are absent until their nodes are connected. The launcher does
not fabricate successful CLI replies. Bubblewrap gives each run private
`/tmp`, `/data`, `/opt/aigen`, `/etc/aigen` and HOME paths; model networking
remains available. This is path/process isolation, not a security boundary.

Credential-free wiring check (fresh output directory each time):

```bash
.venv/bin/python -m scene_pipeline.rhizome_agent \
  --scene outputs/two_fields_corner --task dual-agent \
  --chassis-configuration gen-3.0 --payload-configuration raptor_4x_30in \
  --seconds 30 --output outputs/actual_agent_check
```

For the credentialed session, create a private JSON file containing ordinary
Rhizome `ai_agent` configuration overrides, for example:

```json
{
  "claude_auth": "subscription",
  "keys": {"claude_oauth": "/absolute/path/to/provisioned/claude_oauth.token"}
}
```

Alternatively use `"claude_auth": "api_key"` and `keys.anthropic`. Gemini tools
also require `keys.gemini`. These are **file paths**, never credential values.
Keep them outside the runtime's private paths (`/tmp`, `/data`, `/opt/aigen`,
`/etc/aigen`). Telegram is disabled in the staged configuration; memory, voice
and captures are redirected to the run directory. Other agent behavior remains
owned by Rhizome's node and task configuration.

```bash
.venv/bin/python -m scene_pipeline.rhizome_agent \
  --scene outputs/two_fields_corner --task dual-agent \
  --chassis-configuration gen-3.0 --payload-configuration raptor_4x_30in \
  --agent-config /absolute/path/to/private-agent-config.json \
  --mission "This is an observation-only simulator with motion disconnected. Grab the front and rear camera frames and describe what you see. Do not attempt to move." \
  --seconds 180 --output outputs/actual_agent_observe
```

Read `ai_agent.log` and `topics.jsonl` for the real conversation and tool results.
`report.json` distinguishes node/sensor readiness from model and motion testing;
accepting `agent-send` alone is not proof of a successful model response.
`install.json`, `source.json` and `cli_programs.json` record the selected build
and executable hashes. A Python-only agent change is picked up on the next run.

Still missing: native navigation/mux/drive and arm control integration, weeding,
multi-device routing, and full task scheduling. No autonomy/health state is
fabricated. CPU rendering suffices for observation checks but has not established
the camera rates needed for autonomous navigation; retain capture timestamps
and examine frame age before attempting movement.

## Legacy navigation adapter

`scripts/run_agent_nav.py` is the legacy agent adapter (Claude plans,
Gemini Robotics-ER picks image points) against an outdoor scene. It does not
instantiate the AiAgent node; the replacement is tracked in the
[host contract and handoff](rhizome_host_contract.md). The agent's
own `Conversation` and tools run behind a simulated `AgentInterface`; native
LocalControl drives the wheels and nav_model's obstacle grid runs in shadow
mode on rendered RGB-D, as on the rover. Agent time is simulation time: each
LLM call lands at its start plus its real latency, so the rover keeps rolling
while the agent thinks.

## Setup

- `uv sync --inexact --extra agent-nav` (torch CPU, the Anthropic/Gemini clients,
  dataclasses-json for rhizome's robot description).
- A rhizome checkout with `src/ai_agent` (`--rhizome` or `RHIZOME_ROOT`)
  and host extensions built from it:
  `.venv/bin/python -m scene_pipeline.rhizome_build --root "$RHIZOME_ROOT" --output outputs/rhizome_host --jobs 4`.
  Select it with `RHIZOME_HOST_BUILD` or `--host-build`. Rebuild after native
  source changes; Python-only changes load directly from the checkout.
- The nav_model obstacle-grid package (`--nav-package`, default
  `outputs/nav_obstacle_package/nav_grid_4_nav`); it runs on CPU.
- `GEMINI_API_KEY`. Claude runs on the Claude Agent SDK with the Claude Code
  login (`--llm sdk`, default; unset `ANTHROPIC_API_KEY`) or on an API key as
  on the rover (`--llm api`).
- Scenes: `scripts/build_outdoor_scene.py OUT construction|urban`.

## Run

```
GEMINI_API_KEY=... .venv/bin/python scripts/run_agent_nav.py \
  --scene outputs/outdoor_construction --host-build outputs/rhizome_host/host \
  --spawn -6 0 0 --target loader --seconds 1500 --output outputs/run1 \
  --mission "Find the front loader and stop about 2 m from it."
```

Type operator messages on stdin while it runs. `--target` scores a
goal-directed run (reached = within `--target-radius`, surface to surface).
The run ends on a collision (the failure names the contact geoms), the
duration, or Ctrl-C. The output directory holds `events.jsonl` (tool calls,
events, agent messages), the frames the agent looked at, the recorded state
(`rollout_scene.mjz` + `data.h5`), `trajectory.png` and `report.json` (path
length, camera coverage, objects seen, failure).

- `scripts/replay_video.py RUN` renders a run to a 1080p MP4: a chase view with
  who is driving (agent / nav_model), a mission timeline and a minimap, and a
  side panel with the robot's front nav camera (re-rendered at the run's own
  settings, nav_model's row lines while it drives) and the agent's trace
  (thinking, messages, tool calls, events). Playback runs at `--fast` (8x) and
  drops to `--slow` (2x) around agent activity. `--start/--end` cut a clip;
  `--scene DIR` draws the run in another build of the same layout; `--rhizome
  ROOT` gives runs recorded before the production skin the real robot look.
- For a photoreal chase view: `--export-hero DIR` writes the scene, poses and
  camera path; `vendor/blender/.../blender --background --python
  scripts/blender_hero.py -- DIR SKY.hdr SKY_YAW_RAD [--tree T.gltf --bush
  B.gltf]` renders `DIR/frames`; `--hero-frames DIR/frames` composites them
  under the same HUD. The robot-view panel always stays the simulator's image.
- Splat captures (`capture_object`, `capture_area`) land in
  `RUN/captures/<name>`; `scripts/splat.py prep|train|eval` turns one into a
  Gaussian splat and scores it against MuJoCo ground truth.

## What is and isn't simulated

- Simulated by the harness: `/nav/source`, drive_mux and `/agent/enable`
  (as `aigen_cli` replies), the operator (re-engages AUTONOMY 15 s after the
  agent's own cancel_nav), and drive_control's `/drive/limits` lease.
- Not run: navigation (no GPS route; map == odom), drive_mux invariants,
  hardware drive control. Depth is ideal (no stereo noise or holes).
