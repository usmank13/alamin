# Writeup: text to simulation

## Architecture and metric accuracy

An agent supplies a declarative SceneProgram: inventory, relationships and semantics.
Deterministic tools resolve assets, place and validate geometry, compile MuJoCo,
export URDF and record robot data. The tracked skill works with any command-capable
agent; supplying a program never starts a nested model. Commercial-kitchen and
warehouse fixtures use authored programs. The home kitchen exercised live prompt
interpretation with one repair and a warm asset library, not a cold generalization test.

Dimensions retain distinct bases: engineering defaults, axis-bound user quotations,
verified source fields, or retrieved model units. Native mesh scale is not a measured
product specification. RoboCasa fixtures preserve articulation; catalog imports
provide static SDF props with textures and separate convex colliders. Provenance,
licenses and hashes accompany asset packages.

The optional architecture backend samples frozen residential floorplan statistics;
industrial priors remain unavailable. These deliverables use heuristic furnished
placement. Checks cover dimensions, inventory, support, containment, clearance,
passive stability and sampled articulation. They do not certify semantics, calibrated
dynamics or functional suitability. MuJoCo, Cycles and export consume the same
compiled geometry. Optional fal PBR textures and generated clutter improve appearance;
newly imported generated props include approximate static convex collision hulls,
with unverified geometry and physical defaults recorded in provenance.

## Simulator and export tradeoffs

MuJoCo provides CPU contact physics and stock Menagerie robots. Convex collision
requirements make collider preparation explicit. Costs include mesh conversion,
a separate Cycles bridge and independent export verification. Kitchen and warehouse
URDF packages load and actuate in PyBullet, preserving articulation, mass and inertia;
identical contact behavior and PBR fidelity are not promised. USD remains unimplemented.

## Results and observed failures

The primary dataset has ten accepted home-kitchen variants, each with 60 simulated
seconds of mapping: three full RGB-D and seven state/range/IMU captures. Together
they contain 1,803 camera frames and 60,010 state samples, with recorded videos,
scene previews, Cycles stills, data cards and mapping metrics. Five commercial-kitchen
and five warehouse variants are supplementary; a separate Panda drawer interaction
provides contact-driven articulation evidence. Visibility coverage is distinct from
physical traversal; sensor noise is synthetic and uncalibrated.

Primary base seeds were 300–309. Seed 308 failed robot-access validation and was
replaced by 1308 under a bounded deterministic retry policy. Rejected inputs/checks
remain available and no acceptance check was weakened. This dataset is selected for
passing validation, not evidence of 100% scene-generation success. Five previously
cached fal themes vary across layouts; primary replay made zero new model or paid
fal calls. Historical asset acquisition still had costs.

Earlier retrieval failures include a degenerate drill collider, a nursing station
taller than its test room, and a dishwasher that drifted during settling. A static
pallet jack cannot lift; a solid bowl collider does not make a usable container.
Warehouse entrance/access failures required revised declarative placement.

The semantic-navigation bonus resolved the refrigerator and drove 3.010 m, but
stopped after 22.5 seconds: estimated goal distance was 0.474 m versus 0.924 m truth,
so it failed the 0.6 m arrival criterion. The report preserves this localization
failure. Stretch loading and drive/lift/extension/turn tests pass, but autonomous
mobile manipulation is unproven. Generated-mesh articulation is unfinished; no live
VLM navigation success is claimed. [Bonus evidence](bonuses.md) gives examples.

## Hardware, throughput and scale

The RTX 3070 Ti Laptop GPU renders MuJoCo cameras/replays through NVIDIA EGL;
physics, Cycles and ffmpeg remain on CPU. Primary full captures took 893.5, 731.2
and 385.6 wall seconds; state captures took 89.8–101.1 seconds. Concurrent load varied,
so these are artifact timings, not controlled GPU speedup measurements, and exclude
later replay encoding. Per-run reports retain renderer identity and timings.

At 1,000 environments/day, asset coverage, agent latency, simulation, rendering and
storage need measured capacity planning. Cached assets, tiered capture and disjoint
workers reduce repeated work, but this small batch does not establish production
throughput. Provider billing remains incomplete for earlier model calls.

Cold-machine installation under fifteen minutes and three unseen unattended prompts
remain unverified. Broader articulated assets, calibrated dynamics, industrial
layouts and general manipulation need further evidence. The published checklist
and [brief status](brief-status.md) distinguish collected artifacts from those gaps.
