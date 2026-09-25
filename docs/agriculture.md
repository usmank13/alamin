# Agricultural simulation

The first integration stage adds a wheel-driven Element with the RAPTOR_30
four-camera payload. Inicio generates the plants, crop rows, terrain and optional
debris offline; Alamin compiles portable MuJoCo scenes and records driving runs.
The [Gen 3 importer](gen3.md) can replace the original rigid robot with four
articulated arms and passive rocker suspension while retaining the field.
The [Rhizome integration](rhizome.md) adds native waypoint control, native plant
tracking and a software-only weeding planner/arm path alongside timed driving.

## Generate, drive, inspect

Use Alamin's existing `pipeline` Python dependencies. Generation additionally
needs the sibling Inicio checkout with `src/tools/export_sim_bundle.py`, Blender
5.x, and Inicio's L-Py environment. The existing Alamin Blender 4.2 installation
is not suitable for this exporter. No model API or cloud credentials are needed.

From the Alamin repository:

```bash
.venv/bin/pipeline agriculture generate \
  --config examples/agriculture/raptor30.json \
  --inicio-root ../ml-aigen-inicio \
  --blender "$HOME/Downloads/blender-5.1.2-linux-x64/blender" \
  --output outputs/raptor_field

.venv/bin/pipeline run outputs/raptor_field \
  --flow agriculture-drive --config examples/agriculture/drive.json \
  --output outputs/raptor_drive

.venv/bin/pipeline inspect outputs/raptor_drive --no-open
.venv/bin/pipeline inspect outputs/raptor_field --viewer
.venv/bin/pipeline validate outputs/raptor_field
.venv/bin/pipeline render outputs/raptor_field --view overview
```

Output directories must be new. Default camera capture uses software rendering;
set `MUJOCO_GL=egl` before the command for an available NVIDIA EGL renderer.
Capture is allowed to run slower than real time. Use `--tier state --no-video`
for fast dynamics experiments, or `--seconds 3` for a shorter camera smoke run.
Replay video is generated when ffmpeg is available unless `--no-video` is set;
it can also be generated later with `pipeline replay <rollout>`.

The example contains four soybean rows, 30-inch row spacing, Palmer amaranth,
parthenium and furrowed soil. It uses wider in-row spacing than Inicio's production
RAPTOR_30 example to keep the first scene compact. Its stochastic crop skipping
means the realized count is recorded in the output rather than fixed in advance.

## Configuration and composition

The generation config accepts Inicio's single-scenario JSON or YAML run format.
The `lsystem` block goes directly through Inicio's existing population pipeline,
including its species, growth stages, row spacing, flat/furrow/bed terrain and
`terrain.debris` settings. Existing Inicio units still apply to input fields:
`field_bounds`, `field_center`, `row_spacing`, and `col_spacing_*` use Blender
units (10 BU = 1 metre); terrain keys ending in `_m` use metres. The exported
bundle and all Alamin simulation settings use metres.

Optional Alamin settings in that config:

```json
{
  "simulation_export": {"workers": 1},
  "simulation": {
    "spawn": {"position_m": [-2, 0], "yaw_rad": 0},
    "props": [
      {
        "name": "crate",
        "model": "../my_assets/crate.mjz",
        "position_m": [2, 0, 0],
        "quaternion_wxyz": [1, 0, 0, 0]
      }
    ]
  }
}
```

Props are asset-only MJCF or MJZ models, such as those produced by Alamin's asset
tools. Paths resolve relative to the generation config. The importer packs their
file-backed assets into the bundle, namespaces geometry, and assigns obstruction
class 9. Terrain remains the environment: do not use an entire indoor world as a
prop. Spawn requires the whole robot footprint within the field and clear wheel
patches. The chassis starts just above the terrain and settles under gravity.

The drive config is JSON; see `examples/agriculture/drive.json`. `seconds` can be
overridden on the CLI. `commands` start at time zero, have strictly increasing
times on the 100 Hz control clock, and hold until the next command or run end.
`forward_mps` is along chassis +X; positive `yaw_rate_rps` turns left about +Z.
Commands are bounded to ±1 m/s and ±1 rad/s, converted to four wheel targets and
applied through torque-limited velocity actuators. This is open-loop skid steering,
so the achieved chassis twist differs from the command. The example includes an
initial settling interval and a final stopping interval.

The optional drive `spawn` overrides generation placement. Cameras default to
640×360; `camera.width` and `camera.height` may select a 16:9 resolution between
64×36 and 1280×720. RGB, depth and semantic capture always share camera poses,
intrinsics and timestamps. The four camera poses come from Inicio's RAPTOR_30
manifest; all are fixed to the chassis in this stage.

## Portable artifacts and data

Generated scene directories contain:

- `scene.mjz`: complete field plus the robot at the default spawn.
- `environment.mjz` and `robot.mjz`: independently composable field and robot.
- `bundle/`: producer geometry, terrain samples, identities, resolved config,
  source revision and checksums, plus packaged Alamin props when present.
- `robot_profile.json`: dimensions, actuator limits, estimated dynamics and
  their provenance; `manifest.json`: plants, rows and geom-to-semantic mappings.
- `preview.png`, `index.html`, validation report and Inicio export log.

Copy a bundle and run `pipeline agriculture import <bundle> --output <scene>` to
compile it without Inicio, Blender or L-Py. Copying a compiled scene directory
likewise requires neither sibling repository to drive or inspect it. The standalone
`sim` harness can load `scene.mjz` directly. To put Element in another MJCF world,
use `robot.mjz` in the harness robot manifest with an explicit ground-clear pose;
its root is the CAD chassis origin, about 0.838 m above flat ground. The caller
owns clearance and timestep when composing custom worlds.

Export defaults to one L-Py worker and a pinned Python hash seed for reproducible
geometry within the same source/environment versions. More workers are allowed,
but Inicio's shared leaf-pool selection counters then depend on task scheduling:
placement, stages and identities remain seeded while leaf selection may vary.
Keep the bundle for exact reuse; its geometry is checksummed independently of
generation settings.

Rollouts contain `rollout_scene.mjz`, `data.h5`, `rig.json`, `drive_config.json`,
`manifest.json`, camera previews, trajectory, reports, and optional replay video.
HDF5 remains compatible with the existing `dataset.load_stream` and `inspect` APIs:

| Stream | Rate | Contents |
| --- | --- | --- |
| `state` | 100 Hz | Full qpos/qvel/ctrl, chassis pose, wheel state, actuator forces, contact count |
| `commands` | 100 Hz | Requested twist, bounded twist, wheel targets |
| `imu` | 100 Hz | Ideal and noisy gyro/accelerometer; synthetic uncalibrated noise |
| `camera_raptor_30_1` through `_4` | 10 Hz | RGB, metric optical-axis depth, semantic/instance masks, world camera pose |

All timestamps use simulation seconds on a 2 ms physics clock. Different-rate
streams must be joined by time. The camera rotation matrices use +X right, +Y up,
−Z optical forward (Blender/MuJoCo convention). Camera observations are ideal;
the depth stream does not reproduce D405 stereo holes, noise or range clipping.
Masks, exact poses and plant metadata are ground truth, not perception outputs.

The native Inicio ontology is preserved: background 0, weed/crop/grass foliage
1/2/3, robot occlusion 4, weed/crop/grass stems 5/6/7, barrier 8, obstruction 9.
Stem and foliage geoms share a plant instance ID while retaining separate classes.
Soil and debris are background. Instance IDs are specified in the HDF5 metadata.

## Fidelity and validation

This is a functional prototype, not a calibrated digital twin. Wheel radius and
wheelbase are measured from the CAD mesh; track width is the RAPTOR_30 manifest's
1.6764 m. Tire visuals are translated to that nominal track, and original wheel
centres remain in the profile. Rhizome's 1.558 m Gen2.1 default is not substituted.
Chassis mass/inertia, contact friction and actuator response are explicit estimates.
The model uses a rigid chassis, fixed deck and four driven wheels; no suspension,
arms, soil deformation, plant contact or weed-removal physics is represented.
Uneven soil can unload diagonal wheels and stall turns; this was observed in the
furrow maneuver check. A passed rollout means it completed without numerical
warnings, field exit, tipping or obstacle contact, not that it tracked the
requested chassis twist. Wheel targets, feedback and chassis state are recorded
so these errors can be measured independently of navigation controller behavior.

Terrain is resampled to a heightfield at approximately 2.5 cm spacing, capped at
1025 samples per axis; the actual spacing is stored in the bundle. The same
heightfield provides rendering and contacts. Rock collision shapes are ellipsoid
proxies for rocks at least 4 cm across. Other debris and plants are visual only.
Mesh complexity is reduced for simulation, and material base colors replace
Inicio's procedural Blender shaders. Appearance therefore differs from Inicio's
training renders.

`tests/test_agriculture.py` covers driving direction, turning and stopping,
heightfield axes and metric elevations, asset integrity, spawn rejection,
multi-camera timestamps/depth/identity, portable replay state, bounded commands,
and identical physics with and without camera capture. The full Inicio exporter
is exercised separately with the commands above and its export log retained.

Rhizome's Box2D `PhysicsEnvironment` remains the weeding planner. The current
integration runs its software motor model; MuJoCo will supply physical arm/tool
state after payload articulation and feedback adapters are added.
