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

## Scene pipeline

`src/scene_pipeline` turns a prompt into a validated MuJoCo scene and runs the robot
flows. Setup adds the pipeline extras and, optionally, the Blender binary for Cycles:

```bash
bash scripts/setup_pipeline.sh --blender
```

One command per brief deliverable, run from the repository root:

```bash
# Prompt -> validated scene; omit --program to let the Codex agent write the SceneProgram
.venv/bin/pipeline generate --prompt 'A 60 square metre working prep kitchen ...' \
  --program examples/rich_kitchen_program.py --seed 17 --output outputs/kitchen
# Optional fal layers (export FAL_KEY): PATINA PBR material sets per finish, Hunyuan3D visual-only decor
.venv/bin/pipeline generate --prompt 'A 60 square metre working prep kitchen ...' \
  --program examples/pbr_kitchen_program.py --seed 17 --materials fal --clutter fal --max-fal-usd 6 \
  --output outputs/pbr_kitchen
.venv/bin/pipeline render outputs/kitchen                        # 1024 px CPU Cycles still, same IR
.venv/bin/pipeline export outputs/kitchen --format urdf --verify  # URDF loaded and actuated in PyBullet
# Flows write data.h5, report.json and replay.mp4 (ffmpeg); --tier state skips RGB-D
.venv/bin/pipeline run outputs/kitchen --flow mapping --output outputs/kitchen_mapping --seconds 60
.venv/bin/pipeline run outputs/kitchen --flow interaction --output outputs/kitchen_interaction --seconds 60
.venv/bin/pipeline run outputs/kitchen --flow navigate --goal 'go to the walk-in fridge' \
  --output outputs/kitchen_nav --seconds 60
.venv/bin/pipeline run outputs/kitchen --flow navigate --goal 'go to the prep table' --policy vlm \
  --output outputs/kitchen_nav_vlm --seconds 20
# Ten randomized variants (layout seed, clutter counts, light, tint, texture repeat, PBR texture-set seed); three with RGB-D
.venv/bin/pipeline dataset outputs/kitchen --output outputs/kitchen_dataset --variants 10 --seconds 60
# Stage wall-clock and Codex token table from the artifacts above
.venv/bin/pipeline costs outputs/kitchen outputs/kitchen_mapping outputs/kitchen_interaction \
  outputs/kitchen_nav outputs/kitchen_dataset --output docs/cost-table.md
```

Materials, clutter and render/sim sync. `--materials fal` asks fal PATINA for one seamless PBR
set (base colour, normal, roughness, metalness) per registry finish; `--clutter fal` offers the agent
decor categories (`mug`, `kettle`, `potted_plant`, ...) that Hunyuan3D turns into static, contact-free
meshes sized to a declared band. Every fal job is cached by request digest under
`$SCENE_PIPELINE_CACHE` (default `~/.cache/scene-pipeline/fal`), so repairs and variants never pay twice;
`fal_calls.json` and `cost.json` record list-price spend, with cache hits at zero. The compiler writes each
map once as MuJoCo material texture layers (`rgb`, `normal`, `roughness`, `metallic`, `texuniform` with a
metric repeat from the finish's tile size) and rebinds every asset's `finish_*` placeholder to them, so the
MJZ the simulator loads is the textured scene and its RGB camera sees the albedo. `pipeline render` reads the
same layers back from the compiled model into Principled BSDF: one scene, two consumers. Without the flags
or a key, finishes stay the declared flat engineering colours and `manifest.json` says so under `materials`.
Decor carries no collision geometry, so an arm can sweep through it; it is dressing, not a manipulation target.

Semantic navigation resolves the goal text through the scene's semantic manifest (an
instance id, or a category alias such as `walk-in`, `refrigerator` or `drawer`), never
through coordinates. The robot drives the mapping stack (noisy odometry, scan matching,
online occupancy) behind an action-chunk contract: a policy returns K body-frame
`[vx, vy, wz]` actions that are executed open-loop before it is queried again, the
interface a learned VLA fills later. `planner` (default) plans on the online map toward
the free cell beside the goal; `vlm` sends the base camera frame, the instruction, the
goal offset and a range summary to the Codex CLI with a strict JSON action schema
(10-40 s per call, so keep `--seconds` small). Every chunk is recorded in the `action`
stream with the pose estimate. Success is scored after the rollout from ground truth
only: true final distance to the goal bounds within 0.6 m.

Read the HDF5 streams by time, not by index:

```python
from scene_pipeline.dataset import inspect, load_stream
print(inspect('outputs/kitchen_mapping/data.h5'))
chunks = load_stream('outputs/kitchen_nav/data.h5', 'action', 0, 10)
```

Writeup: [docs/writeup.md](docs/writeup.md). Measured stage timings and token usage:
[docs/cost-table.md](docs/cost-table.md). Design notes live in the ignored `docs/agent/`.

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
