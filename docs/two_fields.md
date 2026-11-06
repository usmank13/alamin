# Two soybean plots connected by a dirt road

The ready-to-load scene is `outputs/two_fields_corner/scene.mjz`. View it with:

```bash
.venv/bin/pipeline inspect outputs/two_fields_corner --viewer
```

`outputs/two_fields_corner/index.html` contains the overview, front-camera preview and
validation report. `road.png` and `west_plot.png` provide closer views.

The cultivated rectangles are **8 × 5.4 m**. The west plot occupies
x = −18…−10 m, y = ±2.7 m. The second plot is northeast of it at
x = 2.3…7.7 m, y = 19…27 m, **rotated 90 degrees** so its rows run north–south.
The nearest plot edges are **20.42 m apart**. Each plot has six soybean rows at
the existing 0.762 m spacing, with 692 soybeans and 80 scattered weeds. The
existing early-stage Inicio soybean meshes are reused in two adjacent tiles per
plot. Individual plant IDs, root positions, row endpoints and foliage geometry
all follow the plot's rotation; roots are placed on the new ground surface.

A **3.6 m wide dirt track** leaves the west field heading east, curves left
through **90 degrees with a 4 m centreline radius**, then runs north into the
second field. The road between field entrances is approximately 32.3 m long.
Textured cultivated soil, shallow furrows, wheel ruts, two trees, six bushes
and a parked 4.2 m tractor provide appearance and obstacles. A tree stands
inside the bend; the tractor sits on a gravel pull-off south of the approach.
Tree trunks, bush envelopes and the tractor's convex hull are solid; leaves and
crops are visual geometry.

The Gen 3 RAPTOR_30 starts at **(−16.5, 0)** facing east. `mission.json` contains
the task, a goal at **(5, 25.5)** facing north, a 0.6 m positional tolerance and
reference waypoints. These waypoints are benchmark truth and should be withheld
when testing a vision-only agent's route discovery. `scene.mjz` includes
front/rear navigation cameras plus the four payload cameras. The separate robot
bundle keeps the original rig so existing runners can install their own
navigation cameras.

## Run an agent

Use the setup from [agent navigation](agent_navigation.md), including matching
Rhizome source/build paths, model package and credentials. For example:

```bash
.venv/bin/python scripts/run_agent_nav.py \
  --scene outputs/two_fields_corner \
  --rhizome ../rhizome-orin-agent \
  --host-build outputs/rhizome_host_agent/host \
  --seconds 240 --output outputs/two_fields_corner_agent \
  --mission "Leave the west soybean plot, follow the dirt road past the parked tractor and turn left around the tree, enter the northeast soybean plot with north-south rows, and stop near its far end. Stay on the road between plots and avoid vegetation and equipment."
```

The agent runner records its usual trajectory and collision report. It does not
consume `mission.json` or automatically score this positional goal; compare the
final pose with that file's goal and tolerance. Do not use `--target` with a plot
name: that option expects a physical manifest object.

For a supplied-waypoint LocalControl baseline, adjust the checkout/build paths
in `reference_drive.json` to a matching installed pair, then run:

```bash
MUJOCO_GL=egl .venv/bin/pipeline run outputs/two_fields_corner \
  --flow agriculture-drive --config outputs/two_fields_corner/reference_drive.json \
  --tier state --output outputs/two_fields_corner_reference --timeout 600
```

This baseline uses simulator pose and supplied waypoints; it does not test
visual road recognition or obstacle avoidance.

## Rebuild and checks

```bash
MUJOCO_GL=egl .venv/bin/python scripts/build_two_fields.py \
  --source outputs/soybean30_early_gen3 \
  --output outputs/two_fields_corner_new
.venv/bin/pytest -q tests/test_two_fields.py
```

Rebuilding requires the existing Inicio bundle/Gen 3 robot and the local Kenney
Car Kit `tractor.glb` (override with `--tractor`). See [soybean30.md](soybean30.md)
for generating the source field. The builder refuses to overwrite output
folders, verifies source bundle hashes, embeds scene assets, preserves the
tractor's CC0 license, and writes source hashes and limitations. Loading the
finished MJZ does not require those source directories.

The builder checks a 3-second passive settling simulation, warnings, a
conservative swept robot envelope at 454 route samples against solid props,
and portable archive reload. These are scene checks, not autonomous-navigation
success. The terrain is 36 × 41 m; stay inside those bounds. Soil and tires are
rigid, dynamics are uncalibrated, and the procedural scenery/tractor are
stylized rather than photorealistic. The plant tiles repeat the source layout.
