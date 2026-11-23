# Localization and obstacle-grid rollout

The runner sends a straight outbound segment and a reverse segment to native LocalControl. The obstacle grid receives both nav cameras at 10 Hz and can stop motion in either direction. By default the runner stows the Gen3 weeding arms at zero motor pitch so the tools clear the ground in reverse. The route starts after one simulated second of suspension settling. The example requests 40 m at 0.45 m/s on each leg. LocalControl switches legs within its arrival tolerance.

Run from the alamin repository with the existing courses and host build.

```sh
.venv/bin/python scripts/run_localization.py --mode truth --output outputs/s1_truth
.venv/bin/python scripts/run_localization.py --mode odometry --output outputs/s1_odom
```

Use a fresh output directory per run. To recreate the courses or build against another Rhizome revision, choose fresh destinations and pass the resulting paths to the runner.

```sh
.venv/bin/python scripts/build_localization_courses.py \
    --source outputs/soybean30_gen3 --output outputs/localization_courses_v2
.venv/bin/python -m scene_pipeline.rhizome_build --root ../rhizome \
    --sim-odometry --jobs 4 --output outputs/localization_host_v2
.venv/bin/python scripts/run_localization.py --scene outputs/localization_courses_v2/rolling \
    --host-build outputs/localization_host_v2/host --package outputs/nav_obstacle_package/nav_grid_4_nav
```

The courses share the same Element robot, spawn and 64 × 12 m footprint. Their passive rocker springs carry preload calculated from the stowed chassis’s weight distribution. This levels the sprung body on flat ground while retaining the original spring stiffness, damping and wheel travel. The per-rocker settings are in each generated robot profile under `suspension[].springref_rad`. `flat` provides clear level ground. `rolling` rises 0.24 m from trough to crest with a maximum grade of 3.1%. `hazards` places a 0.40 m block and a 0.40 m deep ditch beside each other at x = 6 m, on opposite sides of the wheel corridor. A blocked approach is an expected outcome on that course. Inspect the hole and obstacle cell counts in each nav tick for the two features.

The synchronous agriculture rollout runs the native `pyzome.nav.Odometry` binding in its controller worker. Each 10 ms MuJoCo sample publishes the motor measurement last and uses its arrival to trigger one native tick. The node retains its configured 0.01 s integration period. This keeps inference wall time out of the estimator. The host build adds a shared simulation clock and a nav-extension robot-description setter when the checkout lacks them. The example resolves the Gen3 chassis and payload from Rhizome CAD and uses the course wheel radius and trackwidth. This checkout takes yaw from the supplied attitude and uses the gyro to compute yaw rate.

## Fault scenarios

Start with `examples/agriculture/localization_drive.json`. The production odometry parameters live in the second YAML document of Rhizome’s `src/nav/odometry.node.yml`. Override individual parameters under `controller.localization.params`. Other controller configurations default to `truth` when the mode is absent. The runner accepts `--distance` in metres and `--speed` from 0.3 to 0.6 m/s. All fault times are simulation seconds from t = 0.

The following commands prepare configurations for the six scenarios. S1 uses the two commands above. The example configurations select odometry for S2–S6.

```sh
base=examples/agriculture/localization_drive.json
jq '.controller.localization |= (.chassis_pitch_deg = 1.5 | .rtk.outages = [[0,3600]])' "$base" > /tmp/s2.json
jq '.controller.localization.chassis_pitch_deg = 1.5' "$base" > /tmp/s3.json
jq '.controller.localization.rtk |= (.outages = [[25,70]] | .blips = [[75,75.4,0.5]])' /tmp/s3.json > /tmp/s4.json
jq '.controller.localization.params.disable_tilt = true' "$base" > /tmp/s6.json

.venv/bin/python scripts/run_localization.py --config /tmp/s2.json --output outputs/s2
.venv/bin/python scripts/run_localization.py --config /tmp/s3.json --output outputs/s3
.venv/bin/python scripts/run_localization.py --config /tmp/s4.json --output outputs/s4
.venv/bin/python scripts/run_localization.py --config /tmp/s3.json --scene outputs/localization_courses_balanced/rolling --output outputs/s5
.venv/bin/python scripts/run_localization.py --config /tmp/s6.json --output outputs/s6
```

Positive `chassis_pitch_deg` pitches the sprung body nose down relative to the wheel-supported plane. The cameras, IMU, and RTK antenna follow its measured pose. A +1.5° command on flat ground gives roughly −2.6 cm/m of outbound open-loop z error. The rocker references transition over 0.45 seconds and the passive dynamics determine the measured pitch. Use the balanced courses with stowed arms. During a physical-pitch experiment the scripted drive limits acceleration to 0.8 m/s² to keep the rockers loaded through reversal. Obstacle stops still cut the command immediately.

For the local 15 m memory package, add `--package /data/nav_packages/nav_grid_6_mem_ju8zgtfa` to the run command. The runner accepts `--chassis-pitch-deg 1.5` or a time schedule such as `--chassis-pitch-deg '[[0,1.5],[45,-1.5]]'`. Add `--turnaround-pitch-deg -1.5` to hold that angle from the controller's actual switch to reverse. `imu_pitch_offset_deg` independently biases the IMU reading and defaults to zero. For drift scoring, inspect peak and turnaround error since signed travel changes direction on the return. RTK outage and blip intervals are half-open. At 0.45 m/s, the 45 s S4 outage spans about 20 m of uninterrupted travel. Pass the newly exported grid package with `--package PATH` to evaluate distance-based forgetting.

For a short positive control, approach the hazard pair from x = 2.3 m. A lateral offset of −1 m puts the ditch in the swept corridor and the block outside it. Use y = 0 for the centerline approach.

```sh
jq '.spawn = {"position_m":[2.3,-1],"yaw_rad":0}' "$base" > /tmp/ditch.json
.venv/bin/python scripts/run_localization.py --mode truth --scene outputs/localization_courses_balanced/hazards \
    --config /tmp/ditch.json --seconds 7 --output outputs/ditch
```

## Shuttle route

A shuttle repeats a forward leg and a reverse leg between the same endpoints. This example first drives 4 m from the settled start, then shuttles between the 4 m and 7 m marks six times. Omitting `--outbound` starts the shuttle at the settled start. The equivalent controller settings are `"out_and_back_m": 4` and `"shuttle": [3, 6]`.

```sh
.venv/bin/python scripts/run_localization.py --outbound 4 --shuttle 3 6 \
    --chassis-pitch-deg -1.5 --reverse-pitch-deg 1.5 \
    --package /data/nav_packages/nav_grid_6_mem_ju8zgtfa --output outputs/shuttle_rtk
```

`--reverse-pitch-deg` selects the physical pitch at each switch to reverse. The next forward command restores the current `chassis_pitch_deg` setting. Each route leg can be shortened by LocalControl’s arrival tolerance.

## Scores and traces

`localization_score.json` gives simulation status, route completion, actual furthest and final horizontal distance from the settled start, peak absolute z error, signed turnaround and final z errors, and event times. The score subtracts the initial odom–truth height difference at t = 1 s, since odometry starts at zero and MuJoCo reports world elevation. Times use the MuJoCo clock. Rhizome headers use that clock plus one second to avoid zero timestamps.

`legs` lists each commanded segment in order, starting at index zero, including the optional outbound segment. Each entry gives its first and last active sample times and cumulative truth xy distances, signed end error, signed error at the largest absolute excursion, and peak time. The final score also includes any settling time after the route finishes. `first_violation_s` and `first_violation_distance_m` record the first nav tick with a positive raw violating cell count and the accumulated horizontal truth path length at that tick. They are present overall and for each prediction direction, and are null if there was no violation. Travel starts at the settled reference sample.

Each direction’s score includes the number of blocked prediction ticks, first blocked time, raw grid-stop ticks, maximum violating cell count and accumulated violating cell-ticks, including separate counts for holes and obstacles when available. A cell seen on multiple ticks contributes each time. Check `scored_predictions` when interpreting zero counts. Exit status 0 means the simulation and round trip completed, 1 means the simulation failed, and 2 means the simulation finished with an incomplete course. An explicit `--seconds` permits a partial course, including a short smoke or expected stop. Its exit status reflects simulation health and its summary still reports completion.

`rhizome/messages.jsonl` contains one JSON object per request, with `op`, `simulation_time`, `input` and `output`.

- Each 100 Hz `step` records truth chassis `input.position` (xyz, m) and `input.quaternion` (wxyz), and the consumed `output.localization.pose` and `twist` (v m/s, w rad/s). `stamp_ns` is the odometry header time in nanoseconds. `leg` is `out`, `back` or `done`. In odometry mode `path_offset` is the native slope estimate in m/m and `rtk` is the latest antenna fix as fed, with measurement time, height, status and reported vertical accuracy. Fresh RTK kinematics appear in `input.sensors` every twentieth step.
- Each 10 Hz `nav_infer` records the published `/vision/obstacle` result under `output.obstacle`. `output.prediction.obstacle_fwd` and `obstacle_rev` contain `blocked`, `grid_stop`, `violations`, `clearance_m` and `nearest` for current packages. The nearest cell includes its height. The score also reads `output.violation_cells.hole` and `output.violation_cells.obstacle`, each a pair of cell counts in forward, reverse order from the package’s per-cell zone IDs.
- Every 10 simulated seconds with obstacle detection active, `rhizome/grid/{time:.1f}.npz` stores `time`, `ground_z_m` in ODOM, the `valid` mask, `origin_xy_m` at the center index and `cell_m` (metres per cell). The spatial arrays use [x, y] axis order. Packages that expose the fitted plane and violation zones also supply `plane_z_m`, `zone_id` (forward and reverse), and `hole_id`. On shuttle routes, `before_first_violation.npz`, `first_violation.npz`, and `after_first_violation.npz` capture the neighboring camera ticks when available. Use the origin in each snapshot to compare the same ODOM column.
- `data.h5` records truth state, IMU, commands and nav results. The example omits stored images to limit long-run disk usage. Both rendered RGB/depth streams still reach nav_model.

The 100 Hz HDF5 fields `state/chassis_pitch_deg` and `state/commanded_chassis_pitch_deg` contain world-referenced true body pitch and the requested suspension offset in degrees. On a grade, the true angle includes the terrain slope.
