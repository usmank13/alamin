# 2-day work trial: text to simulation environment

Build a pipeline that turns a text prompt into a metrically accurate 3D environment and loads it into a real robotics simulator. Then put robots in it, run autonomous flows, and collect training-grade data.

We score whether a robotics team can load the scene, run it, and collect usable training data without touching it. Resemblance to the prompt is not the test.

---

## Priorities

**Metric accuracy: 25% of the score, and the hardest unsolved part.** Counter at 900mm, robot fits through the doorway, cabinet mass plausible. No API does this for you. Generative 3D returns geometry at arbitrary scale, and an LLM will hand you a wrong dimension with full confidence.

**Visual fidelity: 10%**, shared with scene density and the portability export.

A scene that renders well but is dimensionally wrong is what a generative pipeline gives you by default, and it is useless for robotics. Grey boxes at exact dimensions will train policies fine. If you have to choose, choose the grey boxes.

Spend about 60% of your time on prompt to dimensioned layout, and mesh to physics-ready geometry. Robots, sensors, flows and data have known solutions and stock components.

Sequencing warning: robots and flows show visible progress fastest and pull time away from the scoring. If a robot is moving while your scenes are still at arbitrary scale, you are building in the wrong order.

---

## Resources and hardware

Provided:

* LLM API keys, any provider
* fal, for text to 3D, image to 3D, image generation, texture generation
* Any other hosted API, paid service, dataset or account you want. Ask early instead of engineering around it.

Open source is fully in play: any library, model, dataset or asset pack. No credit for rewriting what exists. Reuse aggressively, cite what you used, spend your time on the unsolved parts.

No GPU is supplied, and do not build around one. Accelerators get expensive at volume and we want the architecture that holds without them. Heavy generative steps go behind hosted APIs, local work stays light. Training or fine-tuning is out of scope. If you want an accelerator somewhere, say where and why in the writeup.

---

## Rules

Simulator, format, language, libraries and architecture are yours. Three constraints:

1. **A real, third-party simulator** that robotics teams already use. Your own physics engine does not qualify. Neither does a renderer standing in for one.
2. **Reproducible.** Fresh machine, your README, under 15 minutes to a running scene. If we cannot run it, we cannot score it.
3. **Everything comes from the prompt.** No hand-authored scenes, no manual fixes to generated files. If you touched it by hand, it does not count.

### Suggested stack

Take it or argue with it. A good argument against it counts for more than going along with it.

* **Simulator:** MuJoCo. Strong contact physics, no accelerator needed, easy to script. MJCF is used in real labs, and its convex collision requirement forces proper collision geometry instead of reused render meshes.
* **Robot models:** MuJoCo Menagerie. Validated, correctly massed, ready to drop in.
* **Alternatives:** PyBullet (simplest install, weakest fidelity), Drake (strongest physics and control, weak perception), Gazebo Harmonic (the ROS 2 and mapping standard, fiddly to install), Genesis (fast and differentiable, less stable support). Pick deliberately and say why.
* **Assets and textures:** fal. **Convex decomposition:** V-HACD or CoACD. **Offline rendering:** Blender Cycles.

On another simulator, translate the format-specific requirements below to its equivalents. We check the output, not which tags produced it.

---

## Part 1: text to scene

### Input

One natural language prompt describing a real indoor working space, 30 to 120 m². Example: *"a mid-size commercial kitchen with two prep stations, a walk-in fridge, and a cluttered dry-goods shelf."*

### Metric accuracy

Highest weighted axis. fal will not solve it for you.

* Real-world scale throughout: counters at counter height, doorways at door width, handles at handle height.
* Metres, consistent axes, defined world origin, sane solver and compiler settings.
* Plausible mass and inertia on every dynamic body, derived from geometry and material density, not left at defaults.
* Correct static vs dynamic classification.

### Collision geometry

Render meshes and collision meshes are different things. MuJoCo enforces this at compile time. Other simulators let you get it wrong silently.

* Every collision shape is a primitive or a convex hull from decomposition (V-HACD, CoACD or equivalent).
* Visual and collision geometry separate, with visual-only geometry excluded from contact.
* Contact filtering configured so the scene does not melt on play.

### Articulation

Most of a room is static: walls, counters, clutter. The things a robot would interact with have to move.

* At least 5 articulated objects on hinge and slide joints: doors, drawers, cabinets, a fridge door, a lever or valve.
* Correct axes, ranges, damping, stiffness.
* Every articulation stable under physics. Report your timestep and integrator, and show it holding.
* No interpenetrating geometry at rest, and clearance for every articulated object to move through its full range. A cabinet door blocked by a chair is a layout failure.

No generative 3D API returns articulated assets, and rigging an arbitrary generated mesh into a drawer is a research problem, not a two-day one. Three routes, in order of cost:

1. **Retrieval** (recommended). [PartNet-Mobility](https://sapien.ucsd.edu/browse) holds thousands of articulated objects across dozens of categories, already rigged with correct joints. Your work is matching, rescaling to target dimensions, and translating the joint spec into your simulator's format.
2. **Parametric library.** Cabinets, drawers, doors and fridges generated procedurally with correct joints. Covers what retrieval misses, with exact dimensional control.
3. **Generated meshes** from fal or equivalent, for non-articulated dressing.

Any mix of the three is a good answer.

**Bonus:** articulate a *generated* mesh. Segment it into parts and author joints programmatically, so articulation follows from the prompt rather than from what you pre-built. A partial result with an honest account of where it breaks counts for more than five more retrieved objects.

### Visual quality

Simulator renderers are not path tracers, so handle that explicitly.

* Real PBR textures and materials, fal-generated or from a library. Not flat colours.
* Meaningful scene density and clutter. An empty room with four objects is a fail.
* One offline render per environment, path traced, driven from the same scene description as the sim. One still at about 1024px is enough, denoising expected.
* Document how render and sim stay in sync. It has to be one scene, not two that resemble each other.

An API-side beautification pass over a sim render is fine if labelled clearly. It is not the simulation.

### Semantics and ground truth

* Per-object semantic class and instance ID, carried in the scene file and queryable at runtime.
* Programmatic access to ground truth poses, bounding boxes, live joint state.
* Segmentation masks from the render path.

### Portability proof

Export one scene to a second, independent format and show it loading there with articulation intact. USD with `UsdPhysics` schemas is the most valuable target. URDF and SDF also work. Partial fidelity is expected. Document what survives and what does not.

---

## Part 2: robots, sensors, flows, data

### Robots

Two required, third is bonus. Placed and configured programmatically, not by hand:

1. Mobile base, differential or holonomic drive
2. Fixed manipulator: Franka Panda, UR-class or equivalent, mounted somewhere sensible
3. Bonus: mobile manipulator or humanoid

Use stock model libraries (Menagerie or equivalent). We evaluate placement, configuration, sensor rigging and control, not whether you can model a robot.

### Sensor suite

Use native sensors where they exist. Keep resolutions modest and justify your choices.

* RGB camera, declared intrinsics, up to 640x480
* Depth or RGB-D from the render path
* Rangefinder array standing in for LiDAR
* IMU: accelerometer plus gyro
* Joint position and velocity
* Force/torque and contact on manipulators

Every sensor needs a declared rate, extrinsics in the robot frame, and a noise model. Noiseless sensors do not produce useful training data. Say what you injected and why.

### Flows

Two autonomous flows, headless and unattended, no human driving, each capped at 60 seconds of simulated time.

1. **Mapping.** The mobile base traverses the space and builds an occupancy grid or point cloud from range data. Compare against ground truth from the scene definition and report the error as a number. This is the cleanest proof that your environment is metrically real.
2. **Articulated interaction.** The manipulator contacts one articulated object and drives its joint through the full range. A clean grasp is not required. Working articulation under contact physics is.

**Bonus:** semantic navigation. "Go to the walk-in fridge," resolved through the semantic layer instead of hardcoded coordinates.

### Data collection

Rendering throughput is the bottleneck, so dataset design is part of what we evaluate.

* Tiered capture: full sensor streams including RGB-D for about 3 runs, then state, range and IMU only for the rest. Justify the tiering.
* All streams time-synchronized on one clock.
* Robot state: joint positions, velocities, control inputs, base pose.
* Ground truth alongside observation: true poses, semantic and instance masks, depth, joint state.
* A standard container (HDF5, LeRobot-style or rosbag) with a documented schema and a working loader.
* Domain randomization across at least 10 variants of one scene: lighting, textures, object placement, clutter. Parallelize however you like.
* A data card: what is in it, how much, known biases.

---

## Deliverables

1. Repo. One command generates a scene from a prompt, a second runs a flow and produces a dataset.
2. At least two environments from your own prompts in your primary format. Three if you have time.
3. One offline render per environment, plus video of each flow.
4. Mapping error report: reconstruction against ground truth, with the number and the method behind it.
5. Dataset from at least 10 randomized runs, with the data card.
6. Cost and timing table: wall clock and API spend per environment, broken down by stage. We want the marginal cost of one environment.
7. Writeup, 800 words max:
   * Architecture, and where the metric accuracy comes from
   * Why you chose that simulator, and what it cost you
   * The failure taxonomy you observed
   * Where hardware limits shaped your design, and what you would do differently without them
   * What breaks first at 1,000 environments a day
   * What a robotics customer would ask for that this pipeline cannot deliver today

---

## Held-out evaluation

End of day 2 we run your pipeline cold on 3 unseen prompts, unattended. Then we load one output in your target simulator and check scale, articulation, collision geometry and stability by hand. We follow your README, and how far we get counts toward the result.

---

## Scoring

| Axis | Weight |
|---|---|
| Metric accuracy and physical plausibility: scale, mass, collision geometry | 25% |
| Scene-file correctness: articulation, semantics, stable simulation, clean load with zero manual fixes | 20% |
| Robots and sensors: both types working, correct rigging, honest noise models | 15% |
| Flows running end to end unattended, and the mapping error number | 15% |
| Data quality: synchronization, schema, tiering rationale, randomization design | 15% |
| Visual quality, scene density, and the portability export | 10% |

Noted but unweighted: sim to real reasoning, randomization beyond lighting and colour, cost discipline, and evidence about what actually degrades downstream policy performance.
