# Running Rhizome against an Alamin world

MuJoCo owns the simulated robot and world. Rhizome owns navigation, agent
decisions, tracking, weeding planning, coordination and stopping policies.
Wheel and arm commands must change MuJoCo actuators; subsequent sensor messages
and measured joint positions must come from that physics state. Rhizome's
`PhysicsEnvironment` remains the weeding planner's internal model, with measured
motor feedback. Its motor simulation cannot replace physical arm feedback.

## Available now

The actual-node observation runner is now available as
`python -m scene_pipeline.rhizome_agent`; see the
[launch and credential handoff](agent_navigation.md#stationary-observation).
It runs the unmodified AiAgent entry point with live MuJoCo sensor topics and
native RGB-D SHM streams. It is a stationary subset of the selected task, not
yet full navigation or weeding integration. The existing navigation adapter
below remains separate. The same entry point also provides
[native wheel and arm acceptance checks](native_controls.md), without agent
credentials. These connect the real control nodes to MuJoCo motor feedback;
they are separate checks from the default stationary agent observation mode.

Select one checkout and a host build produced from its native sources:

```bash
export RHIZOME_ROOT=/absolute/path/to/rhizome-checkout
.venv/bin/python -m scene_pipeline.rhizome_build --root "$RHIZOME_ROOT" \
  --output outputs/rhizome_host --jobs 4
export RHIZOME_HOST_BUILD="$PWD/outputs/rhizome_host/host"
# Optional: an interpreter with Rhizome's Python inference dependencies.
export RHIZOME_INFERENCE_PYTHON=/absolute/path/to/python

.venv/bin/pipeline rhizome-inspect --task dual-agent \
  --output outputs/dual-agent-contract.json
```

Explicit paths override environment variables. If no host path is supplied,
the default is the selected checkout's `build/host`. A verified manifest is
required. Verification compares the **selected checkout's** native source
fingerprint and the actual extension bytes, including duplicate/missing modules.
Changing only Python or YAML does not require a native rebuild. The checkout's
`src` remains the source of Python code and configuration.

Inspection reads the task's service membership, node defaults, includes and
JSON merge-patch overrides. It preserves YAML `on`/`off` keys and null deletion.
It records hashes of all configuration inputs. Missing hardware includes such
as `payloads.yml` remain explicit; `--config-dir` can supply those inputs.
This command does **not** start nodes or certify a runnable simulated task.

`dual-agent` leaves NavModel in shadow mode and selects `/nav/goals` for
LocalControl. `nav-model-drive` enables NavModel driving and selects
`/nav_model/goals`. The eventual runner must read these choices from the selected
task, without overriding them to fit a scenario. Switching these startup modes
during a run requires an explicit Rhizome-owned control interface.

The agent mission drive runs the native node and wheel-control services in the
same simulation. See the [mission drive setup](agent_navigation.md#native-mission-drive).
The `agriculture-drive` runner also verifies the selected install and accepts
finite JSON LocalControl overrides.

## Investigation: run the normal host processes

The preferred integration is Rhizome's normal node executables and Python entry
points, communicating through its existing proxy/IPC transport. The fixture is
useful for focused tests; embedding every production node in it creates lifecycle
and subprocess problems that normal processes already solve. This replaces the
earlier proposal to require new node factories, agent lifecycle hooks, an
in-process CLI dispatcher and an auth abstraction.

Local probes in `outputs/rhizome_host_probe` established the following without
changing Rhizome source:

- `ipc_probe.py` / `result.json`: a normal `core.Relay` connected separate
  publisher, subscriber and service-client processes. The exact published string
  arrived, and `proxy/ping` returned `pong`. A temporary working directory isolates
  HOST_BUILD's relative `build/host/aigen-proxy-*` IPC endpoints. This used a local
  `dev_machine` relay; rover-to-payload forwarding was not tested.
- `agent_probe.py` / `agent_result.json`: the actual `ai_agent.main.py` entry point
  started, consumed a real resolved RobotDescription message and answered
  `agent-send` (empty-message validation) and `agent-stop`. The worker used
  configured missing credential paths and remained idle. No model calls,
  navigation or cameras were exercised. Without a drive mux, the stop response
  proves service dispatch only, not physical stopping.
- A minimal `bwrap` invocation successfully created private filesystem, PID and
  network namespaces. This makes it possible to stage the expected `/opt/aigen`,
  `/etc/aigen`, `/data` and camera paths without altering a host installation.
  A complete packaged task has not yet been launched inside that namespace.

### Existing interfaces to reuse

**Task/configuration.** Native `NodeConfig::from_file` already performs includes
and task overrides. `AIGEN_CONFIG_PATH` adds configuration search roots, but task
selection still reads `/opt/aigen/task/task.*`. Stage the selected task at that
path inside the private runtime, using Rhizome's existing task-generation rules.
Alamin's inspector is for validation and provenance, not a second runtime config
implementation. No configurable task-path API is required for this approach.

**Agent and CLI.** Launch the real entry point with its normal node config; it
constructs its own AgentInterface, tools, memory, coverage and event loop. Existing
config supports authentication/credential paths. The agent's subprocess can
reach IPC, unlike a fixture-only bus. Stage Rhizome's own `aigen` script and host
CLI binaries at their expected paths. Echo, topics and mux/source commands have
existing native implementations. `status` depends on systemd and `tail` on the
journal; a plain process launcher does not supply them. Treat these as explicit
unsupported diagnostics until a Rhizome-owned host implementation exists, rather
than fabricating successful output or building a whole simulated operating
system. CLI commands were inspected, not executed in the probe.

**Cameras.** Publish the existing GStreamer shared-memory format at the configured
camera paths. For navigation combo frames this is RGBA followed by uint16 depth;
the first eight bytes of the depth plane carry the capture timestamp. The agent
already consumes this format. No new image-source API is intrinsically necessary.
The local Python/GStreamer dependencies are now provisioned. The actual-node
observation runner verifies front/rear delivery through Rhizome's own SHM reader.
Private `/tmp` paths provide run isolation. Agent-invoked camera tools still
require the separate credentialed conversation test.

**Build/packaging.** Rhizome's make rules already build host executables, copy
Python entry points and stage configuration/scripts. Alamin's current build helper
only produces five extension modules. Extend that packaging to request the
selected task's executable dependencies and retain live checkout Python/config
inputs. This is Alamin build/launcher work, not a reason to expose every native
node as a new Python class.

### Rhizome integration status

1. **Configurable host device identity and proxy endpoints — pending.**
   `include/base_node/endpoints.hpp` fixes HOST_BUILD's device index to `1`.
   A rover and two payloads need distinct identities, and existing cross-device
   forwarding assumes rover network addresses/interfaces. The C++ endpoint and
   proxy configuration types already carry most of these settings; expose a
   consistent host override and ensure all identity consumers use it. Preserve
   existing topic suffixing and forwarding inside Rhizome. Separate local IPC
   directories alone do not solve multi-payload routing.

2. **Motor backend at the hardware boundary — implemented and tested.**
   `src/base_control/protocol.hpp` defines `CanMotorProtocol`; the shared host
   `MotorManager` now selects an external socket backend through
   `AIGEN_MOTOR_SOCKET`. Both drive and arm nodes reach this hook without changes
   to their control logic. Hardware builds retain CAN. The Rhizome transport
   forwards operations, while Alamin applies torque and returns measured MuJoCo
   feedback. The adapter implements enable/reset, encoder offsets, errors and
   command liveness, and explicitly rejects unsupported hardware operations.
   See [external motors](external_motors.md) for the supported motor model and
   limitations. Native MotorManager-to-MuJoCo integration is tested; full
   control-node startup and homing remain unverified.

Neither change belongs in `src/ai_agent`. A Python-only agent change should be
picked up by launching the new checkout entry point with the same compatible
native build. The initial probes required no Rhizome edits. The subsequent
motor-backend implementation changes the shared host MotorManager boundary in a separate
worktree; it does not change AiAgent.

### Clock: a separate constraint, not a timestamp shim

Real-time operation could use Rhizome's existing clock if physics and sensor
production keep pace. Alamin still advances MuJoCo and produces the observations;
agent inference must run independently while the world continues stepping.
This would be a restricted real-time mode, not fulfillment of pause/accelerated
simulation-time semantics.

The local short benchmark (`timing.json`) measured three seconds of physics in
0.081 wall seconds, but steady-state front/rear 640x480 RGB-D rendering reached
only 0.50 frame pairs/second. EGL selected CPU `llvmpipe`; no inference, IPC or crop
cameras were included. Therefore full real-time feasibility is **not** established.
First check usable GPU rendering and the task's required sensor rates.

If the runner must operate slower/faster than wall time or pause, proper clock
support is additional Rhizome infrastructure. Native scheduling, stale-input
checks and request timeouts use `CLOCK_MONOTONIC` timerfds in
`src/base_node/base_node_impl.cpp`; Python agent event waits also use loop time.
Replacing `robot_time::now()` alone would not synchronize them. Scope a shared
scheduler/time source across these consumers, keeping external network/auth
timeouts on wall time. Do not hide overruns by rewriting observation timestamps
or disabling stale-input protections. This is the main unresolved scope risk
for the original requirement that Alamin own the clock.

## Remaining Alamin integration and acceptance

First package and launch the selected task through its real entry points, with
separate device runtimes and the existing proxy transport. Then connect the
MuJoCo sensor/actuator bridge, using the existing agriculture capture loop where
applicable. The launcher supervises processes and records failures; Rhizome owns
payload coordination. Alamin publishes measured sensors and motor feedback and
applies commands from the host motor backend. Agent I/O must not block physical stepping;
LLM latency must remain observable while the robot continues to move.

The point-to-point mission now runs the checkout's agent, navigation and wheel
controls against MuJoCo. The previous navigation CLI delegates to that launcher.
The following broader scenarios remain separate acceptance targets:

- The construction truck-then-loader mission through the actual AiAgent node.
- A Python-only change under `src/ai_agent` picked up by a new rollout, with no
  Alamin edits or unnecessary native rebuild.
- NavModel row driving in the west plot of `outputs/two_fields_corner`, with
  tracked crop/weed inputs, planner target selection, changing arm commands and
  measured MuJoCo joint motion in the same recording.
- Real stopping on stale inputs, disabled autonomy, lost required health and
  worker shutdown, plus correct task mode selection and cross-payload topics.

The first two plots, road and scenery are committed separately. Scene clearance
and settling checks do not establish autonomous inter-field navigation. Native
integration tests require `RHIZOME_ROOT`/`RHIZOME_HOST_BUILD`: an explicitly
selected bad install fails instead of silently skipping or falling back.
