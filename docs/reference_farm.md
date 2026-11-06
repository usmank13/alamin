# Reference-image farm

This independent scene follows the supplied annotated aerial image: the robot
starts in the northeast clearing, a slightly slanted eastern dirt track meets the
bottom cross-track, and the three marked fields keep their relative shapes and
positions. North is +Y, east is +X. All cultivation rows run north–south.

The image has no scale or georeferencing. The chosen **0.12 metres per pixel**
produces an **87.72 × 125.88 m** landscape. The dirt tracks are **4.2 m wide** to
provide clearance for the GEN3 robot. Orange outlines appear only in the annotated
preview, not as artificial walls in the scene.

```bash
MUJOCO_GL=egl .venv/bin/python scripts/build_reference_farm.py \
  --output outputs/reference_farm

# Portable scene, articulated GEN3 robot and initial state:
.venv/bin/sim outputs/reference_farm/scene.mjz --viewer --backend glfw

# Geometry and semantic regression checks:
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_reference_farm.py -q
```

For the existing supplied-waypoint LocalControl flow, use a matching installed
`RHIZOME_ROOT` / `RHIZOME_HOST_BUILD` pair:

```bash
MUJOCO_GL=egl .venv/bin/pipeline run outputs/reference_farm \
  --flow agriculture-drive --config outputs/reference_farm/reference_drive.json \
  --tier state --output outputs/reference_farm_drive --timeout 1200
```

This optional drive uses simulator pose and supplied waypoints. It has not been
run for this scene and does not validate vision-based navigation.

The builder requires the existing local `outputs/soybean30_early_gen3` bundle,
robot archive and profile. Pass `--source` to use a compatible alternative.
Output directories must be new; rebuilding to another directory preserves the
previous scene. The script uses no API, network access, paid generation or key.
It reuses existing project Python dependencies and helper functions.

The tall eastern plot is harvested/cultivated soil, the west strip is a green
row crop, and the south plot continues beyond the image. Background strips,
hedgerows and trees fill the remaining landscape. Shared foliage meshes and
low-poly background rows limit geometry overhead. The western work patch at
**[20, 52] m** contains **346 individually identified soybean plants and 40 weeds**,
using positions and row IDs from the local source bundle and simplified shared
leaf geometry. Its six semantic rows are rotated north–south. The clear approach
alley terminates at **[20, 48] m**; use the existing row/weeding workflow after
approach. Crop/weed identities and class IDs remain in `manifest.json`; distant
background rows are scenery and are not individual weeding targets.

`mission.json` provides the NE start and destinations in all three fields, with
sampled paths down the east road and across the cross-track to each southern
field entry. The default destination is **[59, 68] m** in the east field;
`reference_drive.json` provides corresponding waypoint-controller input. A route
is geometrically clear but has not been driven autonomously. The west destination
is the clear approach alley, before the soybean patch.

Generated artifacts:

- `scene.mjz`: portable environment, articulated robot, cameras and initial state.
- `environment.mjz`, `robot.mjz`, `robot_profile.json`: separate reusable components.
- `topdown.png`: north-up MuJoCo render; `layout.png`: field/start annotations.
- `overview.png`, `nav_front.png`, `index.html`: visual inspection.
- `manifest.json`, `mission.json`, `generation.json`: semantics, route goals and provenance.
- `validation.json`: archive reload, physical route clearance and three seconds
  of settling, including drift and physics warnings.

The builder verifies an enclosing robot disc against physical tree/hedge bounds
along all three routes. The road width exceeds that disc by at least 0.15 m on
each side. Crops and leaves are visual-only; trunks and hedge envelopes collide.
Soil is rigid, dynamics are uncalibrated, and the image reconstruction is an
approximation, not survey data or a claim of real agricultural performance.
