# Agentic navigation in MuJoCo

## Stationary observation

`python -m scene_pipeline.rhizome_agent` launches the checkout's actual
`src/ai_agent/ai_agent.main.py` in a separate process, using the normal Rhizome
relay and services. It does not construct an AgentInterface or copy an agent
loop. For stationary observation the selected task must enable `ai_agent`. Rhizome's native NodeConfig
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

Select matching source and binaries with `RHIZOME_ROOT` and
`RHIZOME_HOST_BUILD` (see [host contract](rhizome_host_contract.md)). The current native CLI helpers can be
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

## Native mission drive

Use a Rhizome checkout with the external motor backend, the `sim-drive` task,
and the `ai_agent` and `navigation` nodes. Build the native executables into a
fresh directory using the external-motors checkout.

```bash
.venv/bin/python -m scene_pipeline.rhizome_build --root "$RHIZOME_ROOT" \
  --output outputs/mission-host --binaries navigation local_control drive_mux drive_control
export RHIZOME_HOST_BUILD="$PWD/outputs/mission-host/host"
```

Build the CLI helpers listed under stationary observation if your agent uses them. The navigation run stages
the wheel-control task with the native navigation and agent nodes. The operator
grants autonomy and sends the mission through Rhizome services. The agent chooses
waypoints. LocalControl, drive_mux and drive_control send commands to MuJoCo via
the motor adapter. The checker watches the native goals, transcript and measured
odometry until the robot moves and stays stopped near its final waypoint.

```bash
.venv/bin/python -m scene_pipeline.rhizome_agent --mission-drive \
  --scene outputs/two_fields_corner --task sim-drive \
  --chassis-configuration gen-3.0 --payload-configuration raptor_4x_30in_gen3 \
  --agent-config /absolute/path/to/private-agent-config.json \
  --mission "Drive straight ahead 1.2 meters." --seconds 90 \
  --output outputs/native_agent_drive
```

`report.json` records the measured travel, final speed, stop duration and
distance to the last waypoint. `topics.jsonl` and `physics.jsonl` contain the
native messages and simulated joint state. This check measures arrival at an
agent-issued waypoint. Score a field or obstacle objective separately.

The earlier runner's SDK loop, simulated tool implementations and automatic
autonomy re-engagement have been retired. Historical recordings can still be
opened with `scripts/replay_agent_nav.py`.
The `scripts/run_agent_nav.py` entry point now invokes `--mission-drive` and
takes the native launcher flags. Use `--agent-config` to select the provider.
The old interactive operator inbox, `--llm` switch, replay recording and
splat capture workflow belonged to the retired runner.

### Repeat with a local model response

To exercise the control path without a provider account, start the test
fixture from the Alamin checkout. It prints a loopback URL.

```bash
python tests/fixtures/native_mission_provider.py
```

Set `ANTHROPIC_BASE_URL` to that URL in the mission launch environment. Use
`claude_auth: api_key` and set `keys.anthropic` in the JSON config to the
absolute path of `tests/fixtures/native_mission_dummy.key`. Run the exact
1.2-meter mission above. The fixture returns one model tool call. Rhizome
dispatches it, publishes the waypoint and controls the drive to arrival.
If the GStreamer plugin lives under a private mount, place it in the
selected host build and point `GST_PLUGIN_PATH` there.
