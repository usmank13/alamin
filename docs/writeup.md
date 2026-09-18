# Writeup: text to simulation

## Architecture and metric accuracy

An agent supplies a declarative SceneProgram: inventory, relationships and scene
semantics. Deterministic consumers resolve assets, place geometry, validate it,
compile MuJoCo, export URDF and collect data. The agent can be a command-capable
coding harness using the portable skill, or the configured runtime. Supplied
programs never require a nested model call. The current cross-domain fixtures are
agent-authored programs, not evidence of cold, unattended prompt generalization.

Dimensions come from explicit engineering defaults, axis-bound user quotations,
verified source fields, or retrieved model units. These bases remain distinct:
native mesh scale is not a measured product specification. RoboCasa articulated
assets retain their joints; the new catalog tool imports static SDF props, textures
and separate convex colliders. An agent searches, selects, imports and inserts an
asset reference without per-object code. Licenses and provenance travel with it.

The optional architecture backend samples joint room/opening statistics from
frozen real floorplan records; it does not copy a floorplan verbatim. Its current
corpus is residential, not industrial, and furnishing placement remains heuristic.
The present kitchen/warehouse collection uses the simpler heuristic backend.
Independent checks cover dimensions, inventory, support, room containment,
clearance, passive stability and sampled articulation. They do not certify semantics
or functional suitability. MuJoCo previews, Cycles and export consume the same
compiled geometry, preventing independently invented render scenes.

## Why MuJoCo, and its costs

MuJoCo provides established CPU contact physics and stock Menagerie robots. Its
convex-collision requirement makes collider preparation explicit. The tradeoffs
are a separate Cycles path tracer, mesh conversion work, and independent export
verification. URDF packages are loaded and motor-actuated in PyBullet; they preserve
articulation, mass and inertia but do not promise identical contact behavior or
PBR fidelity. USD remains unimplemented.

## Observed failures

Real retrieval does not guarantee simulation readiness. Earlier PartNet-derivative
doors penetrated at rest; a RoboCasa dishwasher drifted during settling. Of eight
new static asset candidates, seven import; a drill's degenerate collider fails.
A nursing station imports correctly but is taller than the test room, so placement
is rejected. A bowl's source collider is solid, and the pallet jack has no moving
mechanism: neither is advertised as a contact-rich task asset.

The first warehouse layout blocked entrance access. Revising declarative placement
requirements fixed that scene without modifying generated files or weakening
checks; failed attempts remain available. Another seed leaves a pallet approach
blocked. Validated layouts are still simplified: wall-biased furniture, generous
empty floor and sparse small-scale detail. Visual inspection supplements, rather
than replaces, deterministic checks.

## Hardware, throughput and scale

The development machine has an RTX 3070 Ti Laptop GPU. Earlier measurements
used CPU MuJoCo, software RGB-D and CPU Cycles; CUDA is not required. The
historical 60-second kitchen mapping capture took 560 wall seconds, while a 1024-pixel Cycles still took 43.
Current timings are recorded per artifact and must not be conflated with those
earlier runs. NVIDIA EGL is available for new MuJoCo camera captures; physics and
the current Cycles bridge remain on CPU. Record the backend and device with new timing results.

At 1,000 environments/day, agent latency, asset coverage, CPU rendering and the
sequential collector dominate. Cached assets avoid repeat conversion. Tiered
capture avoids rendering every variant. Training, broad parallel orchestration
and additional agent infrastructure are outside this pass.

## Deliverables and remaining limits

Optional fal PBR materials and visual-only generated clutter have both offline
integration tests and a live home-kitchen exercise. The commercial kitchen and
warehouse validate and export independently. The variant refresh adds varied fal
finishes and props, then re-records observations; completion follows the published
dataset reports, not the presence of decorated stills. No photorealism claim follows
from enabling PBR. Mapping reports compare range reconstruction
with scene ground truth; the brief imposes no fixed IoU threshold.

The collected baseline has ten 60-second mapping runs, five per domain, rather
than ten variants of one scene. HDF5 carries synchronized observations and truth;
each completed batch has a data card and recorded videos. Consult
[brief-status.md](brief-status.md) for completion status and
[cost-table.md](cost-table.md) for measured costs.

Customers would still need broader articulated assets, calibrated dynamics,
industrial floorplan priors, general manipulation and validated mobile-manipulator
flows. The live model comparison is paused, and the fifteen-minute cold-machine
installation target has not been certified. These limits are not solved by a
loadable scene or an attractive render.
