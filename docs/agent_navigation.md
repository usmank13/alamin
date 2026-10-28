# Agentic navigation in MuJoCo

`scripts/run_agent_nav.py` runs rhizome's on-rover `ai_agent` (Claude plans,
Gemini Robotics-ER picks image points) against an outdoor scene. The agent's
own `Conversation` and tools run behind a simulated `AgentInterface`; native
LocalControl drives the wheels and nav_model's obstacle grid runs in shadow
mode on rendered RGB-D, as on the rover. Agent time is simulation time: each
LLM call lands at its start plus its real latency, so the rover keeps rolling
while the agent thinks.

## Setup

- `uv sync --inexact --extra agent-nav` (torch CPU, the Anthropic/Gemini clients,
  dataclasses-json for rhizome's robot description).
- A rhizome checkout with `src/ai_agent` (`--rhizome`, default `../rhizome`)
  and host extensions built from it:
  `.venv/bin/python -m scene_pipeline.rhizome_build --root ../rhizome --output outputs/rhizome_host --jobs 4`.
  Rebuild after any rhizome message change.
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

- `scripts/replay_agent_nav.py RUN` renders a run to MP4.
- Splat captures (`capture_object`, `capture_area`) land in
  `RUN/captures/<name>`; `scripts/splat.py prep|train|eval` turns one into a
  Gaussian splat and scores it against MuJoCo ground truth.

## What is and isn't simulated

- Simulated by the harness: `/nav/source`, drive_mux and `/agent/enable`
  (as `aigen_cli` replies), the operator (re-engages AUTONOMY 15 s after the
  agent's own cancel_nav), and drive_control's `/drive/limits` lease.
- Not run: navigation (no GPS route; map == odom), drive_mux invariants,
  hardware drive control. Depth is ideal (no stereo noise or holes).
