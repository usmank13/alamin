# Rhizome vision driving in MuJoCo

`controller.nav_model` runs Rhizome's `NavModel` from the `nav-model-drive`
task using the supplied `nav_rows_jdof5rzx_nav` package. The production
TorchScript loader, camera projection, row-history fitting, lane selection,
vanishing-point filtering, driving gates and mapless turn planner run in the
inference worker. Its `/nav_model/goals` feed the compiled `LocalControl`, whose
linear/angular commands drive the MuJoCo wheels.

The original RGB-only package receives RGB, fixed camera calibration and truth
odometry. The obstacle-grid package additionally receives rendered depth as
described below. Neither receives plant identities, generated row lines or
segmentation. Crop perception is independent and may still use `oracle_visible`
or `neural` for weeding. Synchronous inference pauses simulated time; it does
not simulate inference delay or dropped frames.

## Run

The downloaded archive was extracted, without modifying the original, into
`outputs/nav_model_package/nav_rows_jdof5rzx_nav`. `archive.json` records the
source archive hash. The package includes its own numerical self-test, run at
worker startup. The default uses CPU, `model.ts`, and arrangement 2 (two rows).

Build matching current Rhizome extensions in a fresh directory:

```bash
.venv/bin/python -m scene_pipeline.rhizome_build --root ../rhizome \
  --output outputs/rhizome_host_nav --jobs 4
```

Then run the existing early soybean scene:

```bash
MUJOCO_GL=egl .venv/bin/pipeline run outputs/soybean30_early_gen3 \
  --flow agriculture-drive --config examples/agriculture/soybean30_nav_model.json \
  --output outputs/soybean30_nav_model --timeout 1800

.venv/bin/pipeline replay outputs/soybean30_nav_model --viewer --paused
```

Use fresh output directories for rebuilds/reruns. The config selects
`../ml-aigen-tools/venv312/bin/python`, which supplies Torch, OpenCV, SciPy,
Shapely, dataclasses-json and PyYAML. It selects the new host build explicitly; do not mix current
source with older host binaries.

The existing inference environment needed these additional dependencies:

```bash
uv pip install --python ../ml-aigen-tools/venv312/bin/python \
  'shapely==2.1.2' 'dataclasses-json==0.6.7'
```

## Cameras and clocks

### Obstacle-grid checkout and package

`examples/agriculture/soybean30_obstacle_drive.json` selects the local
`../rhizome-nav-grid` checkout (`usman/nav-obstacle-grid`) and its matching
`nav_grid_4_nav` package in `outputs/nav_obstacle_package/`. The archive's source
path and SHA-256 are retained in `archive.json`. The older `nav_rows_jdof5rzx_nav`
package has no depth/obstacle outputs and cannot run this version of the node.

Run the prepared scene with one command:

```bash
.venv/bin/python scripts/run_soybean_obstacle.py --viewer
```

This checks the selected native build (rebuilding into a fresh directory when
needed), runs 25 simulated seconds, saves a timestamped rollout and videos, and
opens recorded replay paused. Omit `--viewer` for a headless run. `--seconds`,
`--scene`, `--config`, `--output`, `--no-video`, and `--renderer osmesa` are
available; `--disable-stopping` runs the same perception with its obstacle switch
off. Scene generation and model export are separate preparation steps.

The equivalent explicit build/run commands are:

```bash
.venv/bin/python -m scene_pipeline.rhizome_build --root ../rhizome-nav-grid \
  --output outputs/rhizome_host_obstacle --jobs 4

MUJOCO_GL=egl .venv/bin/pipeline run outputs/soybean30_obstacle_gen3 \
  --flow agriculture-drive --config examples/agriculture/soybean30_obstacle_drive.json \
  --output outputs/soybean30_obstacle_drive --timeout 1800

.venv/bin/pipeline replay outputs/soybean30_obstacle_drive --viewer --paused
```

`nav_model.depth: true` captures aligned navigation-camera RGB and optical-axis
depth, including the cube and plants. The adapter converts metres to Rhizome's
GRAY16 units (0.1 mm), with zero for invalid or unrepresentable rays; the package
applies its own depth range. Both metric `depth` and uint16 `depth_raw` are saved
in HDF5. These are ideal rendered observations, without stereo noise or holes;
no obstacle coordinates, instance IDs, masks or terrain grids enter the detector.
The robot profile supplies the dimensions used by Rhizome's swept envelope.

`nav_model.obstacle_detection: true` supplies the navigation-settings enable
switch. Rhizome runs its actual stateful grid, semantic obstacle head, debounce,
`/vision/obstacle` publication and `OBSTACLE` goal gate. Empty goals stop native
LocalControl and the simulator's wheel commands; this does not exercise the
hardware `drive_control` directional clip or brake electronics. Set the switch
false, retaining depth, for a comparison with identical model inputs and raw
grid predictions but obstacle gating disabled.

The adapter feeds the latest commanded twist back as `/drive`. Obstacle health,
lights and snapshot service requests are logged in `rhizome/messages.jsonl`;
incident files stay under the rollout's `rhizome/obstacle_incidents`. No external
hardware services are called. Videos show gate, front/rear block, grid decision
and semantic probability. `report.json` includes obstacle frame counts and first
stop time; use `data.h5` to verify actual motion and clearance separately.

### Shared camera configuration

The nav cameras are added to the rollout assembly, so saved replay contains them.
They capture 640×480 RGB at 10 Hz, independently of the 16:9 crop cameras.
Defaults use Rhizome's nominal front/rear optical poses (±0.40949 m longitudinal,
−0.089478 m vertical, approximately 21.8 degrees downward). Square-pixel focal
length is estimated at 400 px. Fixed mounting height comes from the CAD ground
height plus camera offset, not a terrain query during inference. These values
need physical calibration before interpreting navigation accuracy.

`nav_model.cameras` can select `front`, `rear`, or both. `mounts` may override
`position_m`, optical `euler_xyz_rad`, `fx_px`/`fy_px` (equal), and `height_m`
per selected camera. `arrangement` accepts 0 (unknown), 1, 2 or 3;
`steer_source` accepts `rows`, `vp` or `fused`.

The adapter gives the Python nav node simulation time; its capture history and
freshness gates use that clock. Native LocalControl receives model goals at
100 ms intervals. Its wall-clock watchdog is replaced by an equivalent 0.5 s
simulation-clock goal watchdog. Empty, unhealthy or expired model goals stop
the wheel command. There is no waypoint or ground-truth steering fallback.
The native transform graph selects a slice strictly before the requested time.
Synchronous camera/odom transforms are therefore stamped 1 ns before capture;
using the capture timestamp itself would select the previous pose and corrupt
the accumulated depth grid, especially on the first frame.

## Weeding and scope

A controller may combine `nav_model` with `physical_arms` and its existing crop
`perception` setting. Remove any static `waypoints`: mapless model guidance
replaces them. A separate navigation worker owns the two nav cameras;
per-payload workers keep their own crop cameras, trackers and planners. The
capture camera list must contain the selected payload cameras and nav cameras.
Navigation gates and TURN guidance hold arms at zero yaw/pitch through native
motor-state overrides. This is a simulation adapter for the task's drive and
arm gating, not the complete hardware mux/service system.

`examples/agriculture/soybean30_nav_weeding_oracle.json` combines vision driving
with four oracle-weeding payloads. Swap the crop `perception` to `neural` and add
the `neural` block from `soybean30_neural_rows.json` to use neural crop perception;
keep this example's current host build and inference Python. Navigation always
uses the learned nav model regardless of that crop-perception selection.

This integration currently selects **mapless mode** (`use_map: false`). It does
not run the RTK navigation service, mapped-route handoff, geofence, drive_mux,
or hardware drive_control. Mapless K-turn generation remains Rhizome's own
implementation, but an end-to-end headland maneuver requires a larger field
than this small soybean patch. A successful capture is not an agricultural
navigation performance or crop-clearance certification.

## Inspect

`report.json` records frame counts, driving-frame counts and gate counts.
`rhizome/messages.jsonl` records every prediction and generated goal list;
combined multi-arm runs place navigation messages under `rhizome/navigation/`.
`data.h5` contains RGB and optical poses in `camera_nav_front` / `camera_nav_rear`,
and numerical navigation diagnostics in `nav_model`. The inspection page shows
last-frame overlays with predicted row lines, vanishing point and gate state.
Raw model/package hashes, effective parameters and self-test error are retained
in worker provenance. Replay uses recorded MuJoCo states and needs no inference.
With video enabled, `navigation.mp4` shows the recorded front/rear RGB with the
model's row lines, vanishing point and driving status for each capture. It is
also embedded in the inspection page.

## Recorded checks

Host extensions were built from Rhizome revision
`501177a2642e891c069e3cd44bbdddb03313a419`. The supplied FP32 package's self-test
maximum absolute output error was `3.05e-5`.

Two eight-second runs completed without MuJoCo warnings, each with 81 paired
front/rear captures, 81 driving predictions, and no oracle navigation inputs:

| Capture | Travel | Starting lateral offset | Final lateral offset |
| --- | ---: | ---: | ---: |
| `outputs/soybean30_nav_model` | 1.281 m | 0 cm | 0.14 cm |
| `outputs/soybean30_nav_model_offset` | 1.251 m | 12 cm | 9.74 cm |

The centered run's maximum absolute lateral excursion was 0.85 cm. The offset
run also started with a 0.06 rad heading error; its config is
`examples/agriculture/soybean30_nav_model_offset.json`. These are short captures
on a finite field, not a convergence or robustness benchmark. In particular,
the rear view begins near the field edge and the model still predicts rows in
partly empty views. Predicted visibility is not a guarantee that a real row exists.

Native-node tests cover movement from model-generated goals, stops on missing
or expired captures, independence from wall-clock inference delays, and immediate
arm holds and worker cleanup with three and four payloads. The real-package
capture test forbids all oracle plant/row/terrain observation functions.

The combined run at `outputs/soybean30_nav_weeding_oracle` completed six seconds
and 0.977 m of travel without simulation warnings. All 61 paired nav captures
produced driving goals while four independent crop-camera/planner streams ran.
Arm 1 entered pitching-down, striking and follow-through during this short
capture. This validates concurrent operation; no new crop-clearance claim or
contact score is inferred from the planner states.

Validation completed with **74 passing tests**, including six navigation tests
and 68 existing native, neural-perception, agriculture, Gen 3 and weeding checks.
Logs are saved at `outputs/nav_tests_final.log` and
`outputs/nav_regressions_final.log`.

### Soybean cube stop

With obstacle checkout revision `a4a6874272391c067fe363d3e578106497e53a8f`
and `nav_grid_4_nav`, the corrected 25-second rollout is
`outputs/soybean30_obstacle_drive_v2`. It travels 1.484 m before holding:

- Elevation-grid obstacle gate at 8.90 s; first zero wheel command at 8.91 s.
- Planar speed below 1 cm/s at 9.11 s; no cube contact or simulation warning.
- Final conservative footprint clearance is 1.00 m; nearest physical collision
  geometry is 1.43 m from the cube. The two use different robot envelopes.
- The RGB semantic obstacle head stays below threshold: this validates the
  depth-grid branch and the NavModel stopping gate, not that head's accuracy.
- The otherwise identical cube-free control, `outputs/soybean30_obstacle_clear_v2`,
  drives 1.974 m over 12 s without any raw grid stop or obstacle gate.
- With stopping disabled, `outputs/soybean30_obstacle_disabled_v2` drives into the
  cube at 17.91 s (2.952 m travel). This expected comparison failure confirms that
  ordinary row following does not stop for the cube on its own.

`obstacle_test.json` in the corrected rollout records motion/clearance measurements.
The initial `soybean30_obstacle_drive` and `soybean30_obstacle_clear_check` runs
predate the capture-transform timing fix and are retained as diagnostic failures;
their immediate stops are not successful cube detections. Twelve adapter tests
pass, including native debounce, stop/resume, disabled-switch behavior, capture
pose consistency, and the existing RGB-only navigation and multi-arm checks.
