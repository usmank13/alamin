# Articulated Gen 3 robot

For a complete single-arm perception-to-contact test with independent weed/crop
scoring, see [weeding acceptance](weeding.md).

The Onshape export at `/data/URDF EXPORT GEN 3.0` contains **four two-axis
weeding arms, two independent rocker joints, and four wheel joints**. Alamin
imports the glTF geometry and URDF pivots into a portable MuJoCo robot, adds
collision proxies and bounded actuators, and composes it with any existing
agricultural field scene. No Blender, ROS or source checkout is required to
load the resulting `scene.mjz` or `robot.mjz`.

```bash
uv sync --extra pipeline --extra rhizome --extra dev

MUJOCO_GL=osmesa .venv/bin/pipeline agriculture robot outputs/raptor30_field \
  --urdf '/data/URDF EXPORT GEN 3.0' --output outputs/gen3_articulated

MUJOCO_GL=egl .venv/bin/pipeline run outputs/gen3_articulated \
  --flow agriculture-drive --config examples/agriculture/gen3_drive.json \
  --output outputs/gen3_motion --tier state

.venv/bin/pipeline inspect outputs/gen3_articulated --viewer
```

Use a new output directory for each command. `--urdf` accepts the export folder
or its URDF file. `--mesh-faces` sets the simplifier's target per unique mesh
(default 30,000; topology can prevent reaching that target). CAD face seams are
welded before simplification. Visual meshes are reused across repeated links;
simple collision shapes keep the dynamics independent of tessellation density.
CAD colors are retained, so the rendering differs from the supplied photograph.

## Assembly and payload layouts

The default `--layout URDF` preserves exported arm mount locations. Arms 1–4
are ordered from robot left to right, at chassis Y positions
`[0.4826, 0.1524, -0.1524, -0.4826]` metres. Each has its own `gen3_N` camera.
Chassis +X is forward, +Y left, +Z up; the tools extend rearward.

The URDF contains no optical frames. Default cameras use estimated
`x=-0.18, z=-0.299` mounts, aligned laterally with each arm, downward-facing
orientation, and Inicio's D405 optics. Some chassis/wheel occlusion is retained
in RGB, depth and masks. These mounts require calibration before interpreting
perception accuracy as representative of the real robot.

To select an Inicio camera layout, add:

```bash
--layout RAPTOR_30 --inicio-root ../ml-aigen-inicio
# or --layout RAPTOR_22
```

The importer reads camera poses from the selected Inicio manifest (including
its Blender-unit conversion), keeps its camera names and relocates the arm
pivots to the corresponding camera Y positions. This is an **explicit assembly
assumption**, not a measured camera-to-arm calibration: camera support brackets
are part of the fixed chassis mesh and are not relocated. The export alone
does not establish the lateral camera-to-arm offsets for either preset.
The default URDF layout is the geometry-preserving choice. The profile records
the chosen mapping and manifest hash so it can be replaced with measured mounts.

### Custom payload counts and lateral positions

Pass `--payloads` to define the installed assembly explicitly. For example,
[`payloads_three.json`](../examples/agriculture/payloads_three.json) contains:

```json
{
  "payloads": [
    {"id": "arm_left", "lateral_m": 0.5},
    {"id": "arm_center", "lateral_m": 0.0},
    {"id": "arm_right", "lateral_m": -0.5}
  ]
}
```

```bash
MUJOCO_GL=osmesa .venv/bin/pipeline agriculture robot outputs/gen3_articulated \
  --urdf '/data/URDF EXPORT GEN 3.0' \
  --payloads examples/agriculture/payloads_three.json \
  --output outputs/gen3_three_payloads
```

The list defines the physical payload count. Omitted payloads have no arm bodies,
joints, actuators or optical camera frames in the compiled model. To retain four
arms with uneven spacing, use
[`payloads_four_offset.json`](../examples/agriculture/payloads_four_offset.json).
There is no four-payload limit; the requested assembly must fit the field and
pass initial collision checks. An empty list produces a chassis without arm
payloads. Omitting `--payloads` preserves the original four-arm import/preset.

IDs are stable `arm_` names, independent of position or list order. Coordinates
are chassis metres, with +Y to the robot's left. Each instance uses the corrected
first CAD arm as its template. Its arm bodies, collision proxies, joints, tool
site, optical camera and native perception/planner mount translate together;
the fore/aft and vertical offsets and camera-to-arm transform are preserved.
For example, `arm_left` owns `gen3_left`, `arm_left_yaw` and `arm_left_pitch`.
The recipe and resolved mounts are stored in `robot_profile.json` and provenance.
If combined with an Inicio preset, the first payload's camera/arm pair supplies
the template before lateral translation; other preset placements are replaced.

Use the installed IDs in the existing native controller configuration:

```json
"physical_arms": ["arm_left", "arm_center", "arm_right"]
```

Camera selection, neural target transforms, native workers, feedback and HDF5
streams follow the profile. `physical_arms` selects which installed payloads
are controlled; `--payloads` determines which payloads physically exist. Existing
controllers or benchmarks that name removed arms are rejected. Define a new
benchmark with plant assignments matching the new IDs when changing count.

This reconfiguration does not redesign the fixed chassis mesh: its exported
camera/support brackets remain visible at their original positions. Lateral
mounts, mass/inertia and collision/actuator estimates are not mechanically
validated layouts. The field footprint conservatively includes the tool sweep;
initial intersections are checked when composing the robot with the field.

Two export-pose corrections are recorded in `robot_profile.json`:

- One otherwise identical arm (`arm_j2_1`) has a 0.604232-radian pose baked into
  its visual/inertial frame. It uses the common `arm_j2` link frame so all four
  arms share the same motor-zero convention.
- The left rocker starts at +0.0698087 radians, canceling the assembly's baked
  tilt and bringing all four wheel centers to the same neutral height.

## Dynamics and arm commands

The export has continuous joints, no collision geometry or actuator limits,
and implausibly small masses. The simulation uses explicitly estimated masses:
70 kg chassis, 8 kg per rocker, 5 kg per wheel, 0.6 kg per yaw assembly and
0.55 kg per arm/tool. CAD inertia tensors are scaled by the assigned mass ratio.
These are estimates, not identified vehicle parameters.

Rockers are passive hinges with ±0.22 rad travel around their neutral pose,
800 Nm/rad springs and 60 Nm·s/rad damping. Wheel radius is 0.20748 m; each
wheel has a velocity actuator limited to 35 Nm. Axis signs are derived from
the assembled model, including the reversed left-wheel axes.

Arm position actuators use 100 Nm/rad stiffness, 5 Nm·s/rad damping and a 12 Nm
torque limit. Estimated motor limits are ±0.65 rad yaw and [-0.15, 1.15] rad
pitch. Initial motor pitch is 0.65 rad. MuJoCo joint angles equal the **negative**
of Rhizome motor yaw and pitch: motor yaw is clockwise-positive, and positive
motor pitch lowers the rearward-pointing tool. The native ARM frame is rotated
180° around chassis Z. Its origin is the yaw axis at the pitch-joint height.

`arm_commands` is independent of wheel commands. For example:

```json
{"arm_commands": {"arm_1": [
  {"time": 1, "yaw_rad": 0.25, "pitch_rad": 0.25},
  {"time": 3, "yaw_rad": -0.25, "pitch_rad": 0.6}
]}}
```

Commands use motor radians, occur on the 100 Hz clock and hold until replaced.
Uncommanded arms hold their initial controls. Out-of-range requests fail
validation. HDF5 contains `arm_1` through `arm_4` with commanded/measured angles,
velocities, actuator torques and tool-tip world positions; `suspension` records
both rocker angles/velocities. Full qpos/qvel/ctrl are retained for replay.
The Gen 3 replay camera follows the robot so payload motion is visible.

## Rhizome perception to a physical simulated arm

```bash
MUJOCO_GL=egl .venv/bin/pipeline run outputs/gen3_articulated \
  --flow agriculture-drive --config examples/agriculture/gen3_rhizome_neural.json \
  --output outputs/gen3_neural_control --no-video --timeout 600
```

`controller.physical_arm: "arm_1"` selects one payload for the native
`PhysicsEnvironment`. All four cameras run the supplied crop model and native
trackers. Payload positions and rotations are resolved from the compiled robot
profile; stale explicit target transforms are rejected. The selected camera's
predicted row profile and ODOM tracks feed the native planner. MuJoCo measured
motor angles are sent back every 10 ms; `SimMotorController` runs in PLAYBACK
mode solely to expose the planner's commands, with its internal motor/soil
simulation disabled. Bounded commands drive the actual MuJoCo position actuators.
The remaining arms hold position unless given timed commands.

The HDF5 `rhizome_physical_arm` stream records native requested commands and the
same MuJoCo feedback. `arm_N` records applied commands after actuator-limit
clipping. JSONL RPC records let you verify each feedback sample independently.
For explicit oracle fixtures, use `perception: "oracle_visible"`; neural mode
never falls back to generator keypoints or terrain sampling.

This is a useful software/control integration, **not a validated weeding
simulation**. The stock Rhizome planner tool shape is approximate, actuator
behavior and mounts are uncalibrated, and native hardware `arm_control` is not
running. Use `controller.physical_arms` to connect multiple native planners; see [multi-arm acceptance](weeding.md#four-arm-native-planning). Tools contact
rigid terrain and robot proxies; plants remain visual, with no weed removal,
crop-damage or deformable-soil model. A successful rollout means valid dynamics
and completed duration, not successful targeting or weeding; inspect target IDs
and measured motion separately.

## Verification

```bash
MUJOCO_GL=osmesa .venv/bin/python -m pytest \
  tests/test_gen3.py tests/test_agriculture.py \
  tests/test_rhizome.py tests/test_rhizome_perception.py -q
```

Gen 3 tests use the local export (`GEN3_URDF` overrides its path) and skip asset
integration checks when it is absent. They check assembly symmetry, portable
reload, forward/reverse/turn directions, all four arm actuators, passive rocker
response, frame transforms, invalid commands and native weed targeting with
MuJoCo feedback. Native/model checks retain their optional checkout/runtime
requirements.
