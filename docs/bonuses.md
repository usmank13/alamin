# Bonus features and measured examples

The brief names three bonuses: semantic navigation, a mobile manipulator/humanoid,
and articulation of a generated mesh. This page separates implemented behavior,
measured examples and remaining gaps. Follow [running.md](running.md) for setup;
commands below assume an activated `.venv` and stock robot resources. Saved
`outputs/` paths are local evidence, excluded from Git and absent in a fresh clone.

| Feature | Implementation | Evidence boundary |
| --- | --- | --- |
| Semantic navigation | Text/instance goal → semantic manifest → online-map planner → robot wheel actions | Measured example below; one scene/goal is not a navigation benchmark |
| Mobile manipulator | Stock Stretch 2 model, programmatic placement, drive/lift/extension/turn controls, portable MJZ | Loading/control checks, not autonomous mobile manipulation |
| Action-chunk policy interface | Planner and optional VLM emit body-frame velocity chunks into the same recorder/controller | Working interface, not a trained VLA or demonstrated live VLM policy |
| Articulating a generated mesh | Not completed | Generated fal props have static collision only; retrieved/procedural articulation is separate |

## Semantic navigation

The goal is resolved from `manifest.json` before the rollout. An exact instance ID
selects that object; known category phrases such as “refrigerator” or “prep table”
select matching instances. If several match, the nearest to the spawn is selected
and alternatives are recorded. This is explicit alias/category matching, not a
general language-understanding benchmark. An absent goal fails with `UNKNOWN_GOAL`.
“Walk-in fridge” is deliberately rejected when the scene only has a reach-in fridge.

The planner uses noisy wheel odometry, IMU, range observations and an online
occupancy map. Static semantic geometry supplies the target's location; this is
not object discovery from the camera. Runtime robot truth is used for final scoring,
not localization. Arrival passes when the final true base position is within
0.6 m of the goal's XY bounds, not when an image merely appears near the object.
`declared_arrival`, `reached`, estimated/true distances and path length are separate
report fields; estimator error or getting stuck can therefore remain visible.

A reproducible example with a supplied template program requires no API calls:

```bash
pipeline generate --program examples/cafe_program.py \
  --prompt 'A compact cafe kitchen with a prep table, three cabinets, two drawer units, a storage shelf, and eight containers.' \
  --seed 1 --output outputs/bonus_cafe
pipeline run outputs/bonus_cafe --flow navigate --goal 'go to the prep table' \
  --policy planner --seconds 60 --tier state --seed 0 \
  --output outputs/bonus_cafe_navigation
pipeline inspect outputs/bonus_cafe_navigation --no-open
```

Use `--tier full` for synchronized RGB-D as well. `state` retains the action,
localization, range, odometry, IMU and exact state streams; the optional observer
replay renders saved physics states and is not an onboard RGB-D recording.
A navigation command may return exit 1 after writing a valid unsuccessful rollout.
Preserve its report instead of presenting the existence of data/video as success.

### Saved home-kitchen example

The measured run uses the existing home kitchen (`outputs/home_kitchen_fal_final`)
and “go to the refrigerator,” with planner policy, seed 0, state tier and a maximum
of 60 simulated seconds. It **did not meet the arrival criterion**: the estimated
pose caused an early arrival declaration, while the true robot remained too far away.

| Recorded field | Result |
| --- | --- |
| Resolved semantic goal | `refrigerator_0`, category `fridge` |
| Simulated duration / declared arrival | 22.50 s; stopped early with `declared_arrival=true` |
| Estimated / true final goal distance | 0.474 m / 0.924 m |
| Acceptance radius | 0.600 m; `reached=false`, `passed=false` |
| Actual traveled path | 3.010 m |
| Recorded action chunks | 91, each shaped `[25,3]` |
| Physics warnings / model-policy failures | None / none |
| Capture wall time | 116.41 s, CPU state capture; excludes later replay rendering |

Open the local [gallery](../outputs/bonus_navigation_home/index.html),
[report](../outputs/bonus_navigation_home/report.json) or
[recorded video](../outputs/bonus_navigation_home/replay.mp4). The same directory
contains `data.h5`, `rig.json`, `rollout_scene.mjz`, map artifacts and
`reproduction.json` with the exact invocation. The replay is 113 frames at 5 fps,
480×360, rendered from the stored states with EGL after capture.
The report's 0.779 occupancy IoU is a reconstruction metric, not navigation success.
The run demonstrates semantic resolution, real control, recording and truthful
failure scoring. It does not establish reliable goal-reaching, and no prompts,
tolerances or estimator settings were changed to turn this result into a pass.

```bash
pipeline run outputs/home_kitchen_fal_final --flow navigate \
  --goal 'go to the refrigerator' --policy planner --seconds 60 --tier state \
  --seed 0 --no-video --output outputs/bonus_navigation_home
pipeline replay outputs/bonus_navigation_home
pipeline inspect outputs/bonus_navigation_home --no-open
```

### Action chunks and the optional VLM

Policies produce arrays of body-frame `[vx, vy, wz]` commands, played at 100 Hz.
The planner supplies 25 actions (0.25 simulated seconds); the VLM supplies up to
eight commands, each held for 25 control steps, making a 200-action/2-second chunk.
A range-based reflex remains below the policy. HDF5 `action/chunk` records each
chunk with estimated pose and target position, so another controller can use the
same contract. There is no learned policy, training run or temporal ensembling.

The optional VLM receives the goal, estimated pose, range-sector summaries, an
available camera frame, and static object positions from the scene manifest.
Those positions are a known-map prior: this is not camera-only visual navigation.
Model-call failures produce a stationary chunk and are recorded in
`policy_failures`, rather than being treated as success. The initial query can
precede the first camera sample.

A live call requires configured credentials and a vision-capable model. This is
an example invocation, **not a measured successful result**:

```bash
# Requires --agent setup extra and OPENROUTER_API_KEY; this can incur API costs.
pipeline run outputs/bonus_cafe --flow navigate --goal 'go to the prep table' \
  --policy vlm --tier full --seconds 10 --agent-backend openrouter \
  --model PROVIDER/VISION_MODEL --max-cost-usd 2 --timeout 900 \
  --output outputs/bonus_cafe_vlm
```

No saved live VLM navigation result was found in the audited outputs. Model budgets
are checked between calls, not enforced as hard provider billing ceilings. The
existing semantic-navigation regression is a 0.1-second interface/goal-resolution
smoke test and accepts successful or unsuccessful arrivals; it is not success-rate
evidence. It also verifies that an absent fridge produces `UNKNOWN_GOAL`.

## Stretch mobile manipulator

The harness composes the stock Menagerie Stretch 2 through
[examples/stretch.json](../examples/stretch.json), preserving its differential
drive, lift, coupled telescoping arm, wrist, gripper, tendons and constraints.
The example configures placement programmatically:

```bash
sim examples/stretch_lab.xml --robots examples/stretch.json \
  --backend osmesa --seconds 10 --output outputs/stretch_example \
  --export-mujoco outputs/stretch_example.mjz
sim outputs/stretch_example.mjz --viewer --backend glfw
```

The passive CLI run holds default controls. The dedicated regression actively
checks driving, lifting, extension and turning:

```bash
python -m pytest -q tests/test_scene.py::test_stretch_drive_lift_extend_and_turn
```

That ten-second simulation checks 5–20 cm forward travel, lift to 0.20 m, combined
arm extension to 0.25 m (each within 1 cm), coupling mismatch below 2 mm, a changed
heading, upright orientation, finite state and zero physics warnings. Stock
`forward` and `turn` commands apply motor effort, not velocity targets. The test
passed during this documentation check; these are fixture tolerances, not calibrated
real-robot accuracy measurements.

Saved historical artifacts:

| Artifact | Recorded observation |
| --- | --- |
| [Stretch load report](../outputs/stretch_verified/report.json) | 10 simulated seconds, 28 bodies, 19 joints, 8 actuators, no warnings; CPU physics time 0.148 s |
| [Portable scene](../outputs/stretch_verified/scene.mjz) | Portable composed Stretch scene with embedded dependencies |
| [Reload report](../outputs/stretch_replay/report.json) | Export reload ran 2 simulated seconds, same counts and no warnings; CPU physics time 0.029 s |

These reports predate the current richer diagnostic schema. Their timings exclude
rendering and are not end-to-end performance measurements. Stretch has not been
integrated into an autonomous mapping-plus-manipulation task or a demonstrated
sensor-complete training dataset. No humanoid is implemented.

## Generated-mesh articulation: unfinished

Fal meshes receive approximate static convex collision hulls here. No implementation segments an
arbitrary generated mesh, discovers parts/attachment frames and authors functioning
joints from its prompt. Enabling fal clutter therefore does not claim this bonus.

The separate [prototype experiments](../prototypes/README.md) include procedural
opening-bin/swivel-spout proxies and retrieved articulated appliances. The
[representative follow-up](../prototypes/REPRESENTATIVE.md#g4-an-explicit-unresolved-input-requirement)
rejects part-aware G4 adaptation when corresponding visible part segmentation is
missing. Named handle collision boxes are insufficient when the visible handle
is fused into a door mesh. Those experiments establish narrower authoring/retrieval
behavior, not articulation of newly generated fal meshes.
