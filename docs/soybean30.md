# Soybean field: oracle and neural Rhizome runs

## Corrected oracle rollout

The original oracle recordings below used stale native binaries whose legacy
barrier logic connected separate rows across the working lane. Current runs use
the matching host build and explicit separate row geometry; see the
[diagnosis and correction](soybean30_oracle_diagnosis.md).

After the [host build setup](rhizome.md), run and watch the corrected case:

```bash
MUJOCO_GL=egl .venv/bin/pipeline run outputs/soybean30_early_gen3 \
  --flow agriculture-drive --config examples/agriculture/soybean30_oracle_rows.json \
  --output outputs/soybean30_early_oracle_rows --timeout 1800

.venv/bin/pipeline replay outputs/soybean30_early_oracle_rows --viewer --paused
```

Use a fresh output directory for a rerun. `planner_debug: true` saves native
geometry views every simulated second under each `rhizome/arm_N/planner/`.
The JSON trace also records geometry validity, row IDs and boundary counts.
All visible crop detections and the existing stem margins are retained.
`soybean30_neural_rows.json` selects the same host build and the upgraded neural
package; neural geometry is predicted, with no oracle fallback.

The corrected 20-second physical run completed without MuJoCo warnings. All four
arms entered pitching-down, striking and follow-through. Open
`outputs/soybean30_oracle_rows_fix/index.html` for the before/after videos and
native row views, or `outputs/soybean30_early_oracle_rows/field_weeding.html`
for per-arm contacts.

| Unique plants | Original oracle | Corrected oracle |
| --- | ---: | ---: |
| Detected weeds | 37 | 37 |
| Selected weeds | 19 | 16 |
| Ordered weed contacts | 0 | 12 |
| Crop-proxy contacts | 0 | 20 |

The corrected weed contacts came from arms 1–4 in counts 3, 2, 6 and 1.
Crop-proxy contacts were 0, 5, 14 and 2 per arm (one crop was contacted by two
arms). These use the same fixed 6 cm-radius, 15 cm-axis crop proxies as before.
The result establishes restored strikes after the row-representation fix;
**it does not establish safe crop clearance**. Arm/tool geometry, contact proxies
and native avoidance still need calibration and validation. The robot travelled
2.658 m and ended 4.21 cm from the final goal, just outside the configured 4 cm
arrival tolerance when the 20-second capture ended.

Validation: 77 focused tests passed across the agriculture, Gen 3, weeding,
replay, native adapter, neural adapter and build-manifest checks. The final native,
neural and build subset passed all 20 tests. The neural core still matches the
original package's numerical reference; the upgraded geometry package was also
exported successfully through the documented CLI. A full new 20-second neural
field comparison has not been recorded; historical neural counts below belong
to the legacy binaries/postprocessor and should not be treated as a current
algorithm comparison.

## Early-stage field and interactive playback

`examples/agriculture/soybean30_early.json` uses soybean growth values **0.14–0.33**,
matching the small-soybean range in Inicio's
`configs/soybean_raptor30_g6_packed_dense_example.yaml`. These are generator growth
values, not agronomic V-stage labels. The other field settings remain unchanged.
The exported soybeans are 4.9–15.9 cm tall (median 8.4 cm), compared with
12.5–80.2 cm (median 30.3 cm) in the original field. There are still 346 soybeans
and 40 weeds, at the same XY positions, with identical terrain and weed roots.
The crop roots' export height offsets change by at most 0.14 mm.

The new scene is `outputs/soybean30_early_gen3`; paired recordings are
`outputs/soybean30_early_oracle` and `outputs/soybean30_early_neural`.
Open `outputs/soybean30_early_comparison/index.html` for videos and diagnostics.
The original larger-canopy scene and runs remain available.

From the repository directory, open the actual saved rollout in MuJoCo:

```bash
.venv/bin/pipeline replay outputs/soybean30_early_neural --viewer --paused
# Or view the matching oracle run:
.venv/bin/pipeline replay outputs/soybean30_early_oracle --viewer --paused
```

This requires a graphical desktop; the command selects GLFW automatically even
if `MUJOCO_GL` is set to a headless backend. It replays recorded states without
stepping physics or rerunning inference. The neural model and Rhizome processes
are not needed for playback. Close the window to exit.

| Control | Action |
| --- | --- |
| Space | Play/pause |
| Left/right arrows | Seek backward/forward one second |
| Comma/period | Step one recorded frame backward/forward and pause |
| Home | Restart |
| `[` / `]` | Halve/double playback speed |
| F | Toggle camera following the chassis |
| L | Toggle looping (on initially) |
| Left mouse drag | Orbit |
| Right mouse drag | Pan |
| Mouse wheel | Zoom |

Orbit, pan and zoom also work while playing. Follow mode preserves your chosen
angle and pan offset. Toggle it off to watch the robot pass a fixed viewpoint.
Omit `--paused` to play immediately; add `--speed 0.5` for half speed. The MuJoCo
camera selector also exposes the robot's installed cameras. Use the replay
keyboard controls above for the recorded timeline; this is not a live controller
session. `pipeline inspect ... --viewer` instead opens a fresh simulation of a
scene and does not replay its saved controller actions.

To reproduce this smaller scene, use the commands below with
`--config examples/agriculture/soybean30_early.json` at generation and replace
the output/input prefix `outputs/soybean30_` with `outputs/soybean30_early_`.
Keep the existing oracle and neural drive configs. Then rebuild its comparison:

```bash
.venv/bin/python examples/agriculture/compare_soybean30.py \
  --oracle outputs/soybean30_early_oracle \
  --neural outputs/soybean30_early_neural \
  --output outputs/soybean30_early_comparison
```

Both early-stage runs completed 20 seconds without MuJoCo warnings and reached
the route goal, travelling about 2.71 m. Each has 2,001 recorded robot states,
201 frames per camera, and 10,001 tool poses per arm.

| Unique associated plants, early-stage field | Oracle | Neural |
| --- | ---: | ---: |
| Detected weeds | 37 | 27 |
| Selected weeds | 19 | 17 |
| Ordered weed contacts | 0 | 10 |
| Crop-proxy contacts | 0 | 15 |

Neural recorded 22,117 predictions and zero oracle detections. Its ordered weed
contacts came from arm_1 (4) and arm_3 (6); crop-proxy contacts came from arm_2
(11) and arm_3 (4). Scoring retains the same fixed near-stem proxies described
below, including the 15 cm crop axis height: they are not fitted to these smaller
plants and do not establish biological damage. The oracle/neural input differences
described below still apply. Neither rollout removes plants.

[Oracle non-striking diagnosis](soybean30_oracle_diagnosis.md) isolates the
crop observation map as a cause of target cancellation and documents a mismatch
between the loaded native host binaries and the current Rhizome checkout.

Playback validation: 42 focused tests passed, including saved-state restoration,
pause/seek/step/speed/loop behavior and existing capture/weeding checks. Desktop
smoke tests verified paused startup and animated playback; a four-second viewer
session at 0.5× replayed 1.98 simulation seconds and shut down cleanly.

## Original larger-canopy field

This experiment uses Inicio's `configs/raptor_30.yaml`: seed 42, six rows of
60 soybean planting sites, 5% crop skipping, soybean stages 0.4–0.8, 20 Palmer
amaranth (stages 0.2–0.3) and 20 parthenium (stages 0.15–0.35). The copied config
makes the existing 30-inch row spacing and 2–3 inch in-row spacing explicit:
7.62 and 0.508–0.762 Blender units respectively (10 BU = 1 m).

The realized field contains **346 soybeans and 40 weeds**, with the weeds
scattered across x = ±1.5 m and y = ±1.2 m. Inicio supplies furrowed terrain
with a 0.762 m period, 0.1 m nominal amplitude and its normal soil-noise settings.
Terrain tessellation is set to 100 subdivisions, subdivision level 1; export
uses one plant-generation worker. Plant sizes, placements and species labels
are not adjusted based on inference or weeding results.

The Gen 3 chassis uses Inicio's RAPTOR_30 camera/payload layout: lateral arm
positions +0.622, +0.1397, −0.1397 and −0.622 m. Each camera has its own native
tracker/planner and actual MuJoCo yaw/pitch feedback. Both configurations use
640×360 camera captures at 10 Hz, 100 Hz native control, 500 Hz physics/tool
poses, a 20-second duration, and the same LocalControl path from (−1.4, 0) to
(1.3, 0) at up to 0.18 m/s. The path covers part of the field; these results are
not whole-field recall or efficacy measurements.

## What oracle means here

`oracle_visible` substitutes simulator-derived perception for the neural model.
Rendered instance segmentation decides whether a plant is visible. For visible
plants the adapter sends the known root position and crop/weed class, confidence
1 and fixed 2 cm keypoint/rectangle sizes. In the corrected adapter, visible crops
also identify their generated row; each row is sent as its own line with a stable
ID. These are row centerlines, not canopy polygons. Ground heights are sampled from the
terrain to supply the arm's row profile. Rhizome's tracker, target selection,
arm planner and feedback loop still run normally.

This is an idealized perception baseline, not a mesh collision query or a
complete perfect sensor. Any visible part can reveal the known root even when
the root is occluded; sizes are simplified, and plants outside camera visibility
are not supplied. Neural mode uses the supplied multicrop package's RGB/depth
predictions and row profiles, with no oracle detection fallback. Neural mode also
forwards predicted density/weed-pressure metrics; the oracle adapter does not
synthesize equivalent density outputs. Both modes
receive simulator chassis/camera poses and ideal rendered depth.

## Reproduce

Use fresh output directories for each run:

```bash
MUJOCO_GL=osmesa .venv/bin/pipeline agriculture generate \
  --config examples/agriculture/soybean30_realistic.json \
  --inicio-root ../ml-aigen-inicio \
  --blender "$HOME/Downloads/blender-5.1.2-linux-x64/blender" \
  --output outputs/soybean30_inicio --timeout 900

MUJOCO_GL=osmesa .venv/bin/pipeline agriculture robot outputs/soybean30_inicio \
  --urdf '/data/URDF EXPORT GEN 3.0' --layout RAPTOR_30 \
  --inicio-root ../ml-aigen-inicio --output outputs/soybean30_gen3

MUJOCO_GL=egl .venv/bin/pipeline run outputs/soybean30_gen3 \
  --flow agriculture-drive --config examples/agriculture/soybean30_oracle_visible.json \
  --output outputs/soybean30_oracle --timeout 1800

MUJOCO_GL=egl .venv/bin/pipeline run outputs/soybean30_gen3 \
  --flow agriculture-drive --config examples/agriculture/soybean30_neural.json \
  --output outputs/soybean30_neural --timeout 1800
```

The neural config uses `/data/models/edge_packages/multicrop_td_wzmle4cf_grid`
and the existing Rhizome-compatible inference interpreter. CPU inference runs
slower than simulation time. Camera capture disables multisampling for aligned
RGB/depth/segmentation passes: blending integer segmentation colours at dense
leaf edges otherwise produces incorrect or out-of-range object IDs. This was
reproduced on the initial field run and verified with the saved robot pose.

## Contact diagnostics

`targeting_diagnostics` now records actual tool-box poses at 500 Hz for active
physical arms in ordinary fields as well as controlled benchmarks. Analyze each
completed rollout with:

```python
from scene_pipeline.weeding_field import analyze
from scene_pipeline.inspection import scene_page
for root in ("outputs/soybean30_oracle", "outputs/soybean30_neural"):
    analyze(root)
    scene_page(root)
```

`field_weeding.html` links to the per-arm JSON report and contact events;
`index.html` includes camera images, trajectory and physics replay. The field
analysis checks the swept tool box against vertical capsules at generated plant
roots: weeds have 15 mm radius/40 mm axis height, crops 60 mm radius/150 mm axis
height. These are the same diagnostic tolerances as the controlled benchmark,
not measured plant meshes or biological damage models. A spatial broad phase
avoids evaluating distant plants without skipping possible intersections.

Ordered weed contacts require detection, tracking and selection of an associated
weed before a new tool-contact entry. Incidental/preselection contacts are
reported separately. Associations use nearest same-class roots within 6 cm in
XY, so densely spaced soybean neighbours can be ambiguous. Field results report
counts and events; they do not impose the controlled benchmark's one-assigned-
weed-per-arm pass criterion. The normal rollout `passed` flag still describes
simulation completion, not successful weeding or crop safety.

Build the side-by-side report, replay links and field/contact maps with:

```bash
.venv/bin/python examples/agriculture/compare_soybean30.py
```

Open `outputs/soybean30_comparison/index.html`. The comparison checks that both
recordings share the source scene, initial pose and camera rig, and that neural
mode recorded no oracle predictions. It can be rebuilt from saved captures
without loading the neural model or running another simulation.

## Recorded results for the original larger canopy

Both runs completed 20 simulation seconds without warnings and reached the goal.
The oracle travelled 2.661 m; neural travelled 2.711 m. Each recorded 201 frames
per camera, 2,001 native feedback samples and 10,001 tool poses per arm. All four
native feedback streams match their corresponding MuJoCo joint recordings.

| Unique associated plants | Oracle | Neural |
| --- | ---: | ---: |
| Detected weeds | 36 | 14 |
| Selected weeds | 14 | 6 |
| Ordered weed contacts | 0 | 4 |
| Crop-proxy contacts | 1 | 2 |

The four neural weed contacts were all made by arm_1 at 4.888, 7.844, 14.708 and
16.548 s. Neural arm_2 contacted two crop proxies at 6.732 and 6.750 s. Oracle
arm_1 contacted one crop proxy at 5.102 s. Oracle planners only entered SEARCHING
and LOCKED_ON during this run. Neural planners also entered pitching-down,
striking and follow-through states. The trace establishes this behavior, not a
single causal explanation for the difference. In particular, object predictions,
row profiles, density inputs and subsequent physical trajectories differ between
modes. Four weed contacts do not establish superior or safe neural weeding.

The neural capture used 4,390 predicted detections and zero oracle detections;
the oracle supplied 13,858 visible-plant detections (including repeated plants
across frames/cameras). The geometric scorer uses independent field truth after
capture. No vegetation was removed and no biological crop damage was simulated.

The oracle process was killed during its initial replay export, after writing
its complete capture and successful simulation report. The replay was recovered
from those saved states in a separate process, without rerunning physics. Future
agricultural replay exports reuse the already-compiled model to avoid holding a
second large field model in memory. The replay source hashes match the captures.
