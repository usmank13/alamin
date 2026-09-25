# Weeding acceptance

The controlled benchmark exercises rendered visibility or neural perception →
native Rhizome tracking → target selection → MuJoCo yaw/pitch actuation →
independently measured tool contact. It places one Inicio weed and one crop on
flat terrain, raises the tools initially, and drives the selected payload past
both plants. The native planner is unchanged.

## Build and run the same scene in both modes

Start from an articulated Gen 3 field created with [the robot importer](gen3.md).
The example source IDs refer to the plants in the existing `raptor30_field`
example. For another field, choose IDs from its `bundle/bundle.json`, or omit
`source_id` to use the first plant of the requested category. Geometry is copied
and translated without changing its scale or relabeling its species.

```bash
MUJOCO_GL=osmesa .venv/bin/pipeline agriculture weeding-scene \
  outputs/gen3_articulated --config examples/agriculture/weeding_scene.json \
  --output outputs/weeding_single_arm

MUJOCO_GL=egl .venv/bin/pipeline run outputs/weeding_single_arm \
  --flow agriculture-drive --config outputs/weeding_single_arm/oracle_visible.json \
  --output outputs/weeding_oracle --timeout 600

MUJOCO_GL=egl .venv/bin/pipeline run outputs/weeding_single_arm \
  --flow agriculture-drive --config outputs/weeding_single_arm/neural.json \
  --output outputs/weeding_neural --timeout 600
```

Use new output directories for new runs. `MUJOCO_GL=osmesa` works without GPU
rendering. The generated configs share the scene, selected camera, waypoints,
speed, native planner parameters, initial pose and contact criteria. The neural
config uses the supplied model package and the existing inference interpreter;
override `model_package`, `inference_python` or `rhizome_root` in the scene config
for another installation. Only the selected camera is captured, through
`camera.names`; all four physical arms remain in the simulation.

The oracle baseline uses ground-truth plant locations only when their rendered
segmentation is visible. Neural mode receives RGB, depth, intrinsics and poses,
with predicted row profiles and no oracle fallback. Scoring reads ground truth
after the rollout and does not feed it to the neural model or native controller.

## Inspect and diagnose

Open `weeding.html` in either rollout directory. This self-contained page has a
time slider, top and side views, tool trajectory, target/protection volumes,
camera footprint and a table of native track flags. `?t=6.3` opens near a chosen
simulation time. The physics video and captured camera images are also linked.

`controller.targeting_diagnostics: true` works outside the benchmark too. Every
native control step records each track's class, ARM-frame position, size,
reach-circle and reach-triangle membership, valid-candidate flag, strike-ready
flag, target lock, native elimination flag, crop protection radii, strike and
follow-through markers. It also records the native reach-circle radius and
projected tube length. These are available in `rhizome/messages.jsonl`.

The flags are observations of native state, not a complete causal explanation
of every rejection. For example, the earlier generated-field run had weed
tracks within reach but zero valid-candidate samples. A diagnostic rerun showed
that all 134 in-reach weed samples overlapped a native crop protection circle.
That satisfies a crop-protection exclusion condition in Rhizome's candidate
filter; the trace does not establish that this was the only applicable gate.
The controlled scene separates the weed and crop so this stage can be tested.

## Acceptance criteria

The rollout must finish without a physics failure, and the weed must pass these
stages in order:

1. A correctly classified detection lies within 6 cm of the known weed stem XY.
2. A correctly classified native track lies within that association radius.
3. The native planner locks a target associated with that weed.
4. The actual MuJoCo tool enters the weed target volume after selection.
5. The tool never intersects the crop protection volume during the run.

The crop's own detection/tracking timestamps are reported as well. Association
uses location and class, not a model track ID copied into the scene. Plant IDs
and native track IDs remain separate. A hit before selection, or continuous
contact that started before selection, cannot satisfy the ordered hit criterion.

In a benchmark scene, top-level `report.json.passed` and the CLI exit code include
these acceptance criteria. `simulation_passed` reports physics completion
separately. A missed weed can therefore fail acceptance with no physics error.
Ordinary agricultural rollouts retain their existing completion criteria.

## Independent contact measurement

The `weeding_tool` HDF5 stream records the selected tool collision box's world
position, orientation and half-extents at **500 Hz**. Evaluation sweeps this
oriented box between consecutive recorded poses. Plants are vertical capsule
volumes defined in the scenario: a 15 mm radius weed target with a 40 mm axis
segment, and a 60 mm radius crop protection volume with a 150 mm axis segment
in the supplied example. Capsule end caps extend beyond those axis endpoints.
These volumes describe benchmark tolerances, not measured plant mechanics.

The box/capsule calculation includes height and orientation. A blade passing
above a plant is a miss. Translation and rotation are subdivided so the maximum
motion of any box point per sample is at most 1 mm; inflating the target by half
that bound prevents tunnelling under this interpolated rigid-motion model.
Reported clearances are conservative by at most 0.5 mm. This assumption covers
interpolation between 2 ms physics samples, not arbitrary motion between them.

Native `is_eliminated` is recorded for debugging but never used as proof of a
geometric hit. Plants remain visible and exert no contact forces. No weed death,
crop damage, removal animation or deformable soil is inferred from these events.
Hardware fidelity remains a separate later stage. Multi-arm planning is described below.

Artifacts:

- `weeding_report.json`: stage results, timestamps, minimum clearances and source hashes.
- `weeding_events.jsonl`: ordered perception/selection milestones, native state
  transitions, and geometric contact entry/exit events.
- `weeding_trace.json` / `weeding.html`: interactive diagnostic data and view.
- `data.h5`: camera capture, control/feedback and 500 Hz tool geometry.
- `rhizome/messages.jsonl`: original native requests, replies and per-track flags.

## Regression checks

```bash
MUJOCO_GL=osmesa .venv/bin/python -m pytest \
  tests/test_weeding.py tests/test_gen3.py tests/test_rhizome.py \
  tests/test_rhizome_perception.py tests/test_agriculture.py -q
```

The geometric tests cover translation/rotation tunnelling, height-aware misses,
crop contacts, absent target selection, initial incidental contact and failed
simulation. Existing native integration checks exercise target selection and
real MuJoCo feedback; perception tests forbid oracle calls in neural mode.
The two commands above provide the complete scene-based acceptance runs.

## Four-arm native planning

Use `controller.physical_arms: ["arm_1", "arm_2", "arm_3", "arm_4"]`
for concurrent simulated payloads. This replaces `physical_arm`/`shadow_arm`.
Each arm owns its assembly camera, native tracker, transform history, row
profile and native PhysicsEnvironment in a separate worker process. The first
listed arm's worker runs the shared LocalControl navigation node. All planners
receive the same simulation time and chassis pose, but their own measured yaw
and pitch at 100 Hz. Calls are synchronous; this is not a real-time throughput
claim. Registry detection expiry follows simulation time using the native
configured timeout and removal API; rendering/inference delays do not age
tracks. Native spatial expiry and tracker frame-age rules remain active. Worker failures stop the rollout and close all payload workers.

The camera list must match the enabled payloads; omitting `camera.names` selects
their cameras automatically. No detection is filtered by its benchmark plant
assignment: overlapping cameras can legitimately see the same plant. Track IDs
are scoped to an arm, so `arm_1:0` and `arm_2:0` are different tracks.

```bash
MUJOCO_GL=osmesa .venv/bin/pipeline agriculture weeding-scene \
  outputs/gen3_articulated --config examples/agriculture/weeding_multi_scene.json \
  --output outputs/weeding_four_arms_spaced

MUJOCO_GL=egl .venv/bin/pipeline run outputs/weeding_four_arms_spaced \
  --flow agriculture-drive --config outputs/weeding_four_arms_spaced/oracle_visible.json \
  --output outputs/weeding_four_oracle_clock --timeout 600

MUJOCO_GL=egl .venv/bin/pipeline run outputs/weeding_four_arms_spaced \
  --flow agriculture-drive --config outputs/weeding_four_arms_spaced/neural.json \
  --output outputs/weeding_four_neural_spaced --timeout 900
```

The multi-arm fixture assigns one weed and one crop to each arm with each
plant's `arm` field. It places the crops farther along the rows, leaving room
for the four parallel weed strikes. This is a controlled integration fixture,
not evidence of successful operation at arbitrary plant density.

Acceptance requires **every arm** to detect, track, select and contact its own
weed, and **every tool to avoid every crop**, including other arms' assigned
crops. One arm cannot earn credit for another arm's target. The top-level
`weeding.html` links to four independent timelines and `weeding_report.json`
contains per-arm results. Original native messages and provenance are under
`rhizome/arm_N/`; native feedback streams are `rhizome_arm_N`, applied controls
and joint states are `arm_N`, and 500 Hz tool geometry is `weeding_tool_arm_N`.
The single-arm configuration and artifact names remain supported. Native peer-arm
semaphore/barrier messages are not exchanged in this adapter; this stage exercises
independent payload planning on the same physical robot.

### Measured four-arm results

The 10-second runs in `outputs/weeding_four_oracle_clock` and
`outputs/weeding_four_neural_spaced` both complete physics without warnings.
All four payloads detect, track and select their assigned weeds. The geometric
acceptance result is currently **FAIL**, because every assigned weed must be hit:

| Arm | Visibility baseline | Packaged neural model |
| --- | --- | --- |
| arm_1 | Contact at 6.946 s | Contact at 6.412 s |
| arm_2 | Contact at 6.428 s | Contact at 6.398 s |
| arm_3 | Miss: 7.5 mm minimum clearance | Miss: 112.9 mm minimum clearance |
| arm_4 | Miss: 4.5 mm minimum clearance | Contact at 6.368 s |

Every tool avoids every crop volume in both runs. Neural arm_3 associates a
locked target with its assigned weed at 4.190 s, later than the other payloads
(0.88–0.90 s). Its timeline exposes the detections, native choices and measured
motion for further investigation; these results do not establish a single cause
for the miss. The neural run records 2,338 predictions and zero oracle detections.
Each payload has 1,001 native feedback samples, 101 camera captures and 5,001 tool
poses. Recorded native measured angles match the corresponding MuJoCo joints.
`outputs/weeding_four_arms_spaced/comparison.json` contains the paired results.

The original closer-crop runs are preserved separately in
`outputs/weeding_four_oracle` and `outputs/weeding_four_neural`. They predate the
simulation-clock expiry correction and are diagnostic artifacts, not current
acceptance evidence. Hardware calibration, production `arm_control`, peer-arm
coordination and biological weed removal remain outside this integration.
