# Scene pipeline

Generate validated MuJoCo workspaces, inspect the same geometry in Cycles, and
record robot mapping or drawer-interaction datasets. A declarative SceneProgram
specifies inventory and relations; Python tools resolve assets, place and validate
them, compile scenes, and collect synchronized observations and ground truth.

Start with [running and reproducing the pipeline](docs/running.md). The
[harness-agnostic agent skill](skills/scene-pipeline/SKILL.md) works with any agent
that can read files and execute commands. Supplying `--program` requires neither
Codex nor an API-backed agent. Prompt-only generation has optional model adapters.

## Quick start

Requires Linux, Git, uv and system rendering libraries (`libosmesa6` on Ubuntu).
Python 3.12 and Python dependencies are selected from `uv.lock`. From the repo root:

```bash
bash scripts/setup_pipeline.sh --minimal
source .venv/bin/activate
pipeline generate --program examples/cafe_program.py \
  --prompt 'A compact cafe kitchen with a prep table, three cabinets, two drawer units, a storage shelf, and eight containers.' \
  --seed 1 --output outputs/first_scene
pipeline inspect outputs/first_scene --no-open
```

Open the printed `index.html` to inspect previews and checks. This is a supplied
program fixture, not a cold natural-language evaluation. Generation and capture
need new output directories; reports and galleries can be refreshed in place.

Add stock robots and Blender, then capture and render that scene:

```bash
bash scripts/setup_pipeline.sh --blender
pipeline run outputs/first_scene --flow mapping --seconds 60 \
  --output outputs/first_mapping
pipeline render outputs/first_scene --view overview
pipeline inspect outputs/first_mapping --no-open
```

Video encoding requires ffmpeg. MuJoCo physics and the current Cycles bridge use
CPU. MuJoCo camera capture can use NVIDIA EGL with `MUJOCO_GL=egl`; software
rendering defaults to OSMesa. See the run guide for setup, GPU checks, optional fal
textures/clutter, variants, data loading and reproducibility limits.

## Where to look

| Need | Entry point |
| --- | --- |
| Setup, end-to-end commands and reproduction | [docs/running.md](docs/running.md) |
| Instructions for a command-capable agent | [skills/scene-pipeline/SKILL.md](skills/scene-pipeline/SKILL.md) |
| Evidence and outstanding brief requirements | [docs/brief-status.md](docs/brief-status.md) |
| Architecture and limitations | [docs/writeup.md](docs/writeup.md) |
| Scene examples | [examples/](examples/) |
| Core pipeline and independent MuJoCo loader | [src/scene_pipeline/](src/scene_pipeline/), [src/sim_harness/](src/sim_harness/) |

The local `deliverables/` snapshot, when present, contains grouped scenes, datasets,
galleries and a checklist. It and the downloaded `vendor/`, generated `outputs/`
and private historical `docs/agent/` notes are excluded from Git. Reproduction does
not require the private notes. The tracked cost table is a historical measurement;
use `pipeline costs` on the artifacts being reviewed for a current table.

## Independent simulation harness

The `sim` CLI loads MJCF or portable MJZ archives independently of scene generation:

```bash
.venv/bin/sim examples/lab.xml --robots examples/robots.json --viewer --backend glfw
.venv/bin/sim examples/lab.xml --robots examples/robots.json \
  --backend osmesa --seconds 10 --output outputs/smoke
.venv/bin/sim outputs/first_scene/scene.mjz --viewer --backend glfw
```

Stock robot models come from `setup_pipeline.sh` without `--minimal` (or the smaller
`setup.sh` for harness-only dependencies). The headless diagnostic writes one RGB,
depth and segmentation frame, `ground_truth.json`, `state.npz` and `report.json`.
Those noiseless final-frame diagnostics differ from the multi-rate HDF5 datasets
created by `pipeline run`. See [the capture formats](docs/running.md#capture-formats).

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
