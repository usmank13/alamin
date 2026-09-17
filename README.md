# MuJoCo simulation harness

Standalone simulation scaffolding for the [trial brief](trial-task-text-to-simulations.md).
The scene generator can use any architecture: its handoff is an MJCF file with
resolvable assets. This package loads it, optionally attaches stock robots, and
provides a viewer and a physics/render smoke check.

Minimal experiments for the pipeline docs live separately in
[prototypes/README.md](prototypes/README.md): P4 procedural/agent-authored geometry,
P5 URDF rescaling/CoACD, and small validation counterexamples.

The hand-authored example remains an installation fixture, **not** a generated
scene or submission. A decoupled generator now lives in `src/scene_pipeline`:
templates/retrieval, seeded layout, validation, Codex repair loop, Cycles,
URDF/PyBullet export, robot sensor flows and HDF5 collection. See the
[pipeline implementation guide](docs/pipeline-implementation.md) for setup,
commands, measured evidence and the still-unmet full-brief gates.
The [richer kitchen and object provenance guide](docs/object-provenance.md)
documents a complete mixed-source example and how each object's origin is tracked.
The [scene grounding experiments](docs/scene-grounding-experiments.md) cover real
floorplan reconstruction, interchangeable layout backends and deterministic
checks, including where the initial empirical layout sampler still falls short.
The newer [distribution-conditioned architecture](docs/distribution-conditioned-architecture.md)
backend samples joint room/opening configurations from real annotations, with
agent-authored intent, independent checks and held-out diversity/coverage reports.

## Setup

Requires Linux, Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Python 3.12 is selected automatically. From this directory:

```bash
bash scripts/setup.sh
```

This installs locked dependencies into `.venv` and fetches three robot directories
from a pinned [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
revision into ignored `vendor/`. Downloads require network access. Setup preserves
locally modified robot checkouts by refusing to overwrite them. No CUDA dependency.
The existing `.git` directory on this machine is empty; initialize version control
separately when ready.

## Run

Desktop viewer (close the window to stop):

```bash
.venv/bin/sim examples/lab.xml --robots examples/robots.json --viewer --backend glfw
```

Headless CPU physics and software-rendered RGB/depth:

```bash
.venv/bin/sim examples/lab.xml --robots examples/robots.json --backend osmesa --seconds 10
.venv/bin/pytest
```

Results: `outputs/smoke/rgb.png`, `depth.npy` (480×640 metric depth),
`segmentation.npy`, `ground_truth.json`, `state.npz`, and `report.json`.
The command fails on physics warnings, resets, nonfinite state,
or invalid renders. It holds the arm at its home controls and leaves wheel targets
at zero; the integration test additionally drives the wheels and moves an arm joint.
Physics wall time excludes model compilation and rendering.

See [brief-status.md](docs/brief-status.md) for requirement-by-requirement coverage
and the remaining work that needs generator, sensor, export or task decisions.

### Diagnostic capture format

All final-frame artifacts use `report.json`'s `capture_time` in simulation seconds.
Derived poses are refreshed with `mj_forward` before capture. These are noiseless
diagnostics, not a configured robot sensor suite or a training dataset.

- `segmentation.npy`: int32 `[480,640,2]`, channels `(object_id, object_type)` from
  MuJoCo's renderer; background is `(-1,-1)`. For `mjOBJ_GEOM`, object ID indexes
  the `geoms` array in `ground_truth.json`. Other types must not be interpreted
  as geometry IDs. These are model-local IDs, not semantic labels.
- `ground_truth.json`: version 1; simulation time, body world poses (wxyz
  quaternions), masses/principal inertias, geometry-to-body mapping, geometry world
  AABBs, contact masks, joint types/limits/positions/velocities, and named controls.
  Bounds conservatively enclose each transformed local geometry bounding box;
  they are not tight semantic object bounds. Infinite planes have null bounds.
- `state.npz`: `time` scalar, `qpos[nq]`, `qvel[nv]`, `ctrl[nu]`, `act[na]`,
  `body_position[nbody,3]`, `body_quaternion_wxyz[nbody,4]`. Joint qpos and qvel
  widths differ for free/ball joints. Read with `np.load(path, allow_pickle=False)`.
- `report.json`: MuJoCo version/backend, timestep/integrator/solver/gravity,
  simulation duration, separate load/physics/render-capture wall timings, warnings,
  and initial/final contacts with signed separation in metres. Negative distance
  indicates penetration; this is a diagnostic, not an automatic clearance pass.

For live programmatic access, call `sim_harness.ground_truth.snapshot(model, data)`
after `mujoco.mj_forward(model, data)`. It returns the same JSON-compatible structure.

Load your own generated scene, with or without robots:

```bash
.venv/bin/sim /path/to/generated/scene.xml --backend osmesa
.venv/bin/sim /path/to/generated/scene.xml --robots /path/to/placements.json --viewer
```

`--output DIR` separates runs. `--viewer-seconds 5` closes a viewer after five
wall-clock seconds for diagnostics. The viewer uses native MuJoCo mouse controls
and actuator sliders. These commands run from the repo root.

## Integration contract

- MJCF uses metres and Z-up. It owns gravity, timestep, solver settings, lights,
  cameras, and world geometry. Assets/includes resolve relative to the XML.
- Optional robot manifest: a JSON array, illustrated by `examples/robots.json`.
  `model` points to a robot-only MJCF, relative to the manifest (absolute paths
  also work). `name` is a unique instance name. `pos` is a translation in metres;
  `quat` is a unit quaternion in **wxyz** order. Both default to identity.
- Placement transforms the robot's existing root pose; it is not a replacement
  for that pose. The mobile robot's source root is already 0.03 m above its origin.
- MuJoCo `MjSpec.attach` composes meshes, joints, tendons, actuators, defaults and
  contact exclusions. All robot names gain an instance prefix, e.g.
  `arm/joint1` and `base/wheel1_speed`. Multiple instances are supported.
- Optional `keyframe` initializes a fixed robot's scalar joints and controls.
  Floating/ball-joint keyframe initialization is explicitly unsupported; omit it
  for these robots. Without it, MuJoCo defaults are used. Source robot keyframes
  are not copied into the final scene. Scene keyframes are not automatically selected.
- Parent scene settings win on global option conflicts; MuJoCo reports conflicts.
  The sample uses `implicitfast`, a 2 ms timestep, and `noslip_iterations=1` to
  accommodate the mobile model. Pick suitable settings for each generated world.
- File-backed meshes, textures, and heightfields are embedded for portability.
  Six-file cubemap textures are currently rejected; use a single texture file.
- Placement clearance is the caller's responsibility. This harness does not
  repair collisions or supply task policies. The default camera frames the sample;
  reposition it interactively for larger worlds.

Python entry point for future controllers and capture code:

```python
import mujoco
from sim_harness.scene import load_scene

model, data = load_scene("examples/lab.xml", "examples/robots.json")
data.actuator("arm/actuator1").ctrl[:] = 0.2
mujoco.mj_step(model, data)
```

Set `MUJOCO_GL` before importing MuJoCo in Python when rendering. The CLI's
`--backend` does this for you. `scene.py` has no generation dependency; `cli.py`
  is only a runner, and the `examples/` files can be replaced independently.

## Export boundary

The brief requires a **second independent format with articulation intact**.
The pipeline now exports URDF scene packages and verifies them in PyBullet; USD
is deferred. A native MuJoCo bundle alone does not meet that requirement.
See [docs/export-contract.md](docs/export-contract.md) for fidelity and scope.

For portable MuJoCo inspection/replay now:

```bash
.venv/bin/sim examples/lab.xml --robots examples/robots.json --backend osmesa --export-mujoco outputs/composed.mjz
.venv/bin/sim outputs/composed.mjz --viewer --backend glfw
```

The `.mjz` archive contains composed MJCF, embedded assets, and a `harness_initial`
keyframe with initial state and controls. Exports refuse to overwrite an existing
file. The CLI restores that keyframe when loading harness bundles. No vendor paths
are needed for replay. Integration tests cover scene-only loading, robot
placement/control, multiple instances/rotation, invalid names, and export/reload.

Exporters can call `compose_scene(scene, robots)` from `sim_harness.scene` to obtain
`(spec, model, data)`. Keeping the specification available avoids forcing later
exporters to reconstruct articulation from rendered images or flattened meshes.

## Graphics on this machine

The machine has an RTX 3070, but `nvidia-smi` currently cannot communicate with
its driver. `osmesa` provides working CPU offscreen rendering without that driver.
For Ubuntu hosts missing graphics libraries, install `libosmesa6`, `libegl1`,
and `libgl1` using the system package manager. They are already present here.

Once the NVIDIA driver works, use `--backend egl` for headless GPU rendering;
`glfw` is for the desktop viewer. EGL can also select Mesa, so successful EGL
rendering alone does not prove NVIDIA is in use. CUDA is not required by this
native MuJoCo harness; accelerated training/MJX is a separate integration.

## Models and references

Try the Stretch 2 mobile manipulator (differential drive, lift, telescoping arm,
wrist, and gripper):

```bash
bash scripts/setup.sh
.venv/bin/sim examples/stretch_lab.xml --robots examples/stretch.json --viewer --backend glfw
.venv/bin/sim examples/stretch_lab.xml --robots examples/stretch.json --backend osmesa --seconds 10 --output outputs/stretch
```

`stretch_lab.xml` reuses the original world but selects Stretch's stock elliptic
contact cone and two no-slip iterations. The integration test drives forward,
raises the lift, extends the coupled telescoping joints, and turns, checking
response, uprightness and absence of physics warnings. The stock `forward` and
`turn` actuators apply motor effort, **not velocity targets**. This verifies model
compatibility, not navigation or manipulation task success. Export/reload tests
also cover Stretch's textures, tendons, and equality constraints.

- Panda: `franka_emika_panda/panda.xml`, including its stock gripper and home pose.
- Mobile manipulator: `hello_robot_stretch/stretch.xml` (Stretch 2, Clear BSD license).
- Mobile base: `robot_soccer_kit/robot_soccer_kit.xml`, a small three-wheel
  omnidirectional robot. Its upstream README explicitly states that no accurate
  system identification was performed. Treat it as a harness fixture, not a
  calibrated full-size service robot.
- Assets stay in their upstream directories with model-specific licenses and
  READMEs; inspect those before redistributing.
- [MuJoCo Python/model composition documentation](https://mujoco.readthedocs.io/en/stable/python.html)
- Exact upstream commit is recorded in `scripts/fetch_robots.py`; Python versions
  and dependency hashes are recorded in `uv.lock`.
