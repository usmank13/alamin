# Brief coverage audit

## Current pipeline status (2026-09-16)

The decoupled generator is now implemented as a working vertical slice. See
[pipeline-implementation.md](pipeline-implementation.md) for commands, evidence,
milestone status and remaining release gates. This is not a completed submission.

| Requirement | Current evidence and limitations |
| --- | --- |
| Prompt-generated scene | Real Codex structured-output repair run; seeded DSL route; required unsupported categories fail |
| Metric geometry and articulation | Explicit engineering defaults; G1 families and native RoboCasa adaptation; five-object fixture checks; no calibrated physical claim |
| Layout | Rectangle/L footprints, annex partitions, collision/sweep reservations and supported clutter; full zones/connectivity/grammar pending |
| Stable simulation | Five-second passive settle and mask-independent geometric checks; 21-pose sweeps are not dynamic endpoint proof |
| Visuals | Real 1024-pixel Cycles render and labeled inspection gallery; limited materials, no diffusion |
| Semantics | Stable class taxonomy, instance manifest, masks, live object pose/bounds/joint query |
| Independent export | URDF loaded and motor-actuated in PyBullet, mass/inertia/COM checks, relocation regression; additional frame/bounds checks pending |
| Robots and sensors | Stock base/Panda flows; RGB-D, range, IMU and arm force/torque; explicit synthetic noise and rig metadata |
| Mapping | Working baseline and reported error metrics; the brief specifies no IoU/coverage threshold |
| Manipulation | Actuator-only contact controller and open/final-close checks; limited to one drawer family and approach candidates |
| Dataset | HDF5 writer/loader, common simulation clock, tiered collector and data card; ten one-second pilot variants, not ten successful full tasks |
| Release | Full original acceptance environments, broader category coverage, held-out prompts, task videos and cold-install timing remain pending |

## Historical harness-only audit

The table below records the earlier harness-only baseline, before the pipeline
implementation. Its "pending" entries are not current implementation status.
Hand-authored fixtures still do not satisfy prompt-generated scene requirements.

| Brief requirement | Current coverage / remaining work |
| --- | --- |
| Real simulator and reproducible setup | MuJoCo, locked Python dependencies, pinned stock assets, setup command. Fresh-machine under-15-minute acceptance not measured. |
| Prompt to 30–120 m² environment | Generator-owned; not implemented here. |
| Metric scale, plausible mass/inertia, static/dynamic classification | Load and expose model data; cannot establish physical plausibility without generator dimensions/material provenance. |
| Separate visual/convex collision geometry | Preserve source definitions and expose contact masks; decomposition and provenance validation remain generator-owned. |
| Five articulated environment objects and full-range clearance | Robot joints work; fixture is not five articulated environment objects. Object generation and clearance validation remain pending. |
| Stable physics, timestep/integrator report | Headless checks for warnings/resets/nonfinite state; explicit solver/timestep/gravity report and initial/final contact distances. No proof of full-range clearance. |
| PBR, scene density, offline path-traced still | Pending generation/material and offline-render integration. MuJoCo renders are not path traced. |
| Semantic class and instance ID in scene | Pending object identity/label contract. Model-local geom/body IDs are not semantic classes or persistent object IDs. |
| Live poses, bounds, joint state | Python snapshot API and JSON; per-geometry conservative world bounds, body poses and live joints. Semantic object aggregation remains pending. |
| Segmentation | Native renderer object-ID/type mask with geom-to-body/name mapping. Semantic and persistent instance masks remain pending. |
| Independent-format portability | Native MJZ works; independent articulated USD/URDF/SDF export remains pending; see export-contract.md. |
| Fixed arm, mobile base, bonus mobile manipulator | Panda, Robot Soccer Kit, Stretch; placement, control response and native export tests. |
| RGB/depth, joint state | Same-time 640×480 RGB/depth and exact state snapshot. Diagnostic camera only; no configured robot sensor rig. |
| Sensor rates, intrinsics/extrinsics and noise | Pending sensor design. No arbitrary noise parameters or claim that ground truth is noisy observation. |
| LiDAR, IMU, force/torque | Pending rig configuration. Contact distances are not force/torque sensor measurements. |
| Mapping and articulated interaction flows | Pending task/controller design, sensor input and ground-truth comparison; smoke runs are not flows. |
| Synchronized streams, standard dataset container/loader | Final-frame capture shares simulation time; no time-series dataset. State NPZ/JSON are diagnostics, not the required HDF5/LeRobot/rosbag dataset. |
| Tiered capture, ten randomizations, data card | Pending dataset/randomization design and completed flows. |
| Two generated scenes, flow videos, error report | Pending generator/flow implementation. |
| Stage costs/timing | Load/compile, physics and render/capture wall times recorded; generation API spend and marginal per-environment cost remain pending. |
| 800-word writeup and cold unseen prompts | Pending complete pipeline and measured results. |

No new scene-generation schema, semantic taxonomy, data-container choice, sensor
mounting/noise defaults, physics exporter, or controller was selected in this pass.
