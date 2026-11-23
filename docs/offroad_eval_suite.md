# Offroad evaluation suite for the agent harness (design)

Status: design on paper. Nothing here is built. No scene, asset or code in either repository was changed to write it.

This document proposes thirteen simulated tasks for comparing variants of the rover's agent harness, across a farm, a construction site and a quarry. It says what each task measures, what it needs from the simulator that does not exist yet, which existing scenes and asset packs could be reused, and which tasks to build first.

## 0. Terms and ground rules

**Arm.** One variant of the harness under comparison. The four arms we have compared so far are:

| Arm | What it is |
|---|---|
| S (stock) | The Claude Agent SDK as shipped, plus the rover tools and a one-paragraph brief. No working memory, no skills, no frame or digest on wakes. (`--harness stock`) |
| H (harness) | The on-rover agent: its prompt, working memory (marks, plan, trail, action history, the `[world]` digest on every wake), persistent memory, skills as markdown. Wake gate off. |
| G (gate) | H plus the wake gate: events marked routine are slept through, either by rule (`rules`) or only while a stated expectation still holds in the camera frame (`jev`). |
| K (graphs) | G plus skill graphs: procedures that run tool calls, waits and checks without a model call (`run_skill`). |

**Harness component.** A part of the harness an arm switches on or off: working-memory marks, the trail and seen-area map, the action history and its "this place has stopped you before" line, the plan, the wake gate, the camera expectation check (`expect=` / `until=`), skill graphs, persistent memory.

**Footprint.** The rover's outline on the ground. In the current robot profile it is 2.04 m long and 2.16 m wide, so a passage has to be about 3.2 m wide before the obstacle grid lets the rover through. All layouts below are dimensioned for that.

**Stand-off.** The distance at which the rover is asked to stop from something, measured surface to surface (as `--target-radius` already does).

**Seed.** An integer that fixes everything random in a scene build: object jitter, which objects are targets, sun direction, ground texture. The same seed always builds the same scene.

**What the simulator is today** (from `scripts/run_agent_nav.py`, `scripts/score_run.py`, `scripts/build_outdoor_scene.py`, `scripts/build_two_fields.py`, `scripts/build_staggered_fields.py`):

- MuJoCo, 500 Hz physics, paced to wall clock so that model latency costs simulated time. The rover's top speed is 0.3 m/s (`--max-speed`), which is 18 m per minute. A ten-minute task with the rover moving 60 % of the time covers about 110 m. Every layout below is sized from that number.
- Terrain is one heightfield (0.1 m cells in the outdoor scenes) with one ground texture. Mounds and ruts exist; slopes longer than a mound have not been driven.
- Everything in a scene is static. There are no moving objects, no objects that appear mid-run, no people.
- Operator messages come from the mission argument, from stdin, or from lines appended to `operator_inbox.txt`, all in wall-clock time. A simulated operator re-engages autonomy 15 s after the agent's own `cancel_nav`.
- Depth is ideal. Lighting is one fixed directional light and a gradient sky. There is no dust, fog, rain or lens dirt.
- A run fails on contact with anything taller than 0.12 m, on tipping, on the footprint leaving the rectangular scene bounds, or on an agent error.
- Scored automatically today: failure, path length, share of time moving, parked gaps, goal pushes, camera coverage of the scene and objects "seen" (a view wedge with no occlusion, so an upper bound), surface distance to one named target, weeds and crops hit per plot, row-end error of a saved field map, and from the SDK session: model calls, tokens, cost, wakes, wakes that did nothing.

**Rules every task in the suite follows.** These come from what the first comparison taught us.

1. *The work is fixed by the mission.* The mission names every stop, lane, lap or item. The agent never chooses how much to do, so time to finish is comparable between arms.
2. *Success is a yes/no computed from simulator ground truth.* Each task has one success rule and one primary quality number. Time, cost and model calls are compared only between successful runs; a failed run is reported as a failure, not as a fast run.
3. *A run ends* at the time limit, on a failure, or 20 s after the agent's final message (as `run_arm.sh` does today). What the rover does after its final message is ignored except for contact.
4. *Reports are structured.* Where a task asks the agent to tell the operator something that is scored (a position, a count, a yes/no), all arms get the same one extra tool, `file_report`, which takes a small JSON object. The mission still asks in plain words; the tool description says positions are x, y in metres in the odometry frame, which in the sim is the world frame. This avoids scoring free text.
5. *Each task is 5 to 20 simulated minutes* so that three runs per arm are affordable.
6. *Seeds change appearance and exact positions, never the amount of work.* Route length is held within about 10 % across seeds of a task.

**Simulator and harness additions referred to below.** Each task lists which of these it needs.

| Tag | Addition | Rough size |
|---|---|---|
| N1 | Seeded scene builders: a `--seed` on a site builder and a quarry builder in the style of `build_outdoor_scene.py`, writing targets, stations, zones and reference routes into `manifest.json`. | Small per scene |
| N2 | Sim-time script: a JSON list of actions fired at a simulated time or on a trigger (rover enters a polygon, rover within d of an object, agent hands over to nav_model). First action type: an operator message. | Small |
| N3 | Scripted objects: (a) an object that appears, done by parking a kinematic (mocap) body out of sight and moving it into place on a trigger, because MuJoCo cannot add bodies to a compiled model; (b) an object that moves along a polyline at a set speed. Contact with either is a failure like any other contact. | Medium |
| N4 | Region scoring: polygons and polylines in the manifest (exclusion zones, stations, road centrelines, dead-end pockets) and a scorer that tests the footprint against them. The chassis orientation needed for the footprint is already in `data.h5` at 10 Hz. | Small |
| N5 | The `file_report` tool in all arms and a per-task checker for its fields. | Small |
| N6 | "Was it actually visible": count an object as seen when its geoms cover at least a few hundred pixels of a nav-camera segmentation render, replacing the no-occlusion wedge. The weeding path already renders segmentation. | Small, costs render time |
| N7 | A `task.json` beside each scene (mission, spawn, time limit, script, success rule) and a scorer entry that reads it, so a task is one argument rather than a shell snippet. | Small |
| N8 | Terrain beyond mounds: heightfields up to about 100 m with sustained slopes, checked for skid-steer behaviour (today a yaw-rate PI loop stands in for wheel–soil contact); ground decals that are visible but flat (wet patch, painted line); haze through MuJoCo's built-in fog; sun direction per seed. | Medium; slope behaviour is unknown |
| N9 | New assets: quarry terrain and textures, haul truck, excavator, conveyor or crusher, fence, signs, a standing person. | See section 3 |

Effort flags used in the tables: **R** = reuses an existing scene; **E** = new scene from assets we already have; **N** = needs new assets or simulator features.

## 1. Suite overview

| ID | Task | Domain | Capability it isolates (deciding factor) | Also exercises | Arms / component it should separate | Minutes | Effort |
|---|---|---|---|---|---|---|---|
| A1 | Three plots | Farm | Hand-over between agent and nav_model | Agent-driven navigation between plots | K vs G (weed-a-plot graph); H vs S on alignment (nav_model facts, edge stop) | 12–16 | E |
| A2 | Something in the lane | Farm | Fault response while nav_model drives | Hand-over, recovery | G-jev vs G-rules vs H: the camera expectation check | 8–12 | E, then N3 |
| A3 | Fence check | Farm | Find and report along a perimeter | Agent-driven navigation, loop closure | H vs S: trail and marks | 9–12 | E |
| A4 | Map a field (short) | Farm | Mapping accuracy with nav_model driving | Procedure following | K vs G (map-a-field graph) | 10–14 | R |
| C1 | Delivery round | Construction | Long-horizon multi-stop | Agent-driven navigation, return to start | H vs S: plan and marks | 10–14 | E |
| C2 | Look round, then go back | Construction | Search with memory | Operator message mid-mission | H vs S: marks and seen-area map | 10–13 | E + N2 |
| C3 | Dead end | Construction | Recovery from a blocked route | Reversing, re-planning | H vs S: action history and the repeated-stop line | 6–10 | E |
| C4 | Site check by the book | Construction | Procedure following | Multi-stop, reporting | K vs G vs H: skill graph against markdown procedure | 10–13 | E + N5 |
| C5 | Change of plan | Construction | Operator interaction mid-mission | Multi-stop | G vs H (gate must never sleep through the operator); H vs S (plan) | 9–12 | E + N2 |
| Q1 | Haul road | Quarry | Agent-driven navigation over distance | Slopes, junction choice | K vs G (follow-until graph): time, cost, share of time moving | 8–11 | N8, N9 |
| Q2 | Give way | Quarry | Yielding to a moving vehicle | Fault response | G-jev (`until=`) vs others: wake latency | 4–7 | N3, N9 |
| Q3 | Keep out | Quarry | Staying out of a marked exclusion zone | Carrying a constraint across many legs | H vs S: plan and constraints; all arms: perception discipline | 6–9 | N4, N9 |
| Q4 | Stockpile survey | Quarry | Measurement accuracy under agent driving | Loop closure, stand-off keeping | H vs S: mark fusion | 8–11 | N8, N5 |

Every capability named in the brief is the deciding factor of at least one task:

| Capability | Deciding task | Second look |
|---|---|---|
| Agent-driven navigation | Q1 | A3, C1 |
| Search with memory | C2 | A3 |
| Long-horizon multi-stop | C1 | C4 |
| Procedure following | C4 | A4 |
| Fault response | A2 | Q2 |
| Recovery from stuck or blocked | C3 | A2 |
| Operator interaction mid-mission | C5 | C2 |
| Mapping / measurement accuracy | Q4 (agent drives), A4 (nav_model drives) | A3 |
| Hand-over between agent and nav_model | A1 | A2 |
| Rule compliance (exclusion zone, right of way) | Q3, Q2 | C4 |

Only three of the thirteen tasks (A1, A2 and A4) let nav_model drive at all, and in A1 and A2 the scored part is what happens around the hand-over, not the row driving.

## 2. Task specifications

Metrics that every task reports, in addition to its own: success (yes/no), failure reason, simulated seconds to the final message, share of time moving up to the final message, model calls, tokens, cost, wakes and wakes that did nothing. These exist in `score_run.py` already.

### A1. Three plots

**Mission.** "Weed all three plots. You are in the first one. The second is across the headland to the north-east and the third is beyond it."

**Scene.** A 50 m by 40 m soil field built with the `build_two_fields.py` generator. Three soybean plots, each four rows at 0.762 m spacing and 8 m long, which is two passes and one row-end turn per plot. Plot 2 is rotated 90° from plot 1 and plot 3 is rotated 30° from plot 2. The gaps between plots are 6 to 8 m of bare headland with nothing in them: this task is about lining up, not about finding a route. Forty weeds per plot. Flat ground with the generator's furrows. Per seed: plot 3's rotation (20° to 40°), the gap lengths within 6 to 8 m, weed positions, sun direction, which of two soil textures.

**Spawn.** On the first lane of plot 1, at its west end, facing along the rows.

**Time limit.** 20 minutes. **Expected duration.** 12 to 16 minutes.

**Success.** At least 80 % of the weeds hit in each of the three plots, no more than 2 % of crops hit, no failure.

**Metrics.** Primary: hand-over quality, measured at each moment the agent sets the source to NAV_MODEL: lateral offset of the rover's centre from the nearest lane centreline and heading error from the row direction, both from the manifest rows. Secondary: number of hand-overs per plot (one is ideal; a retry means the first alignment was rejected or abandoned); seconds from nav_model's FIELD_END to the next hand-over; weeds and crops hit per plot; crops driven over between plots (wheel contact with crop geoms outside any lane).

**Failure.** Standard conditions; also the time limit with a plot unweeded.

**Injected events.** None.

**Good run.** Three hand-overs, each within 0.15 m and 5° of a lane, transitions of about a minute, all weeds in reach hit. **Bad run.** Hand-over at an angle across the rows so nav_model weeds a diagonal; hand-over on the wrong lane so one pass is skipped; several minutes of small corrective pushes at each plot; declaring done after two plots.

**Needs.** A three-plot variant of the generator (N1), the hand-over metric in the scorer (reads `nav_source` events already in `events.jsonl`). **Effort: E.** No new simulator feature.

### A2. Something in the lane

**Mission.** "Weed this plot, all three passes. Don't drive over anything that isn't crop or soil. If something is in the way, stop, tell me what it is, go round it if there is room and carry on."

**Scene.** One soybean plot, six rows at 0.762 m, 14 m long (three passes, two row-end turns), with 4 m headlands, on a 30 m by 24 m field from the `build_staggered_fields.py` generator. One fault per seed, placed in the second pass between 4 m and 10 m along it. The fault kinds are chosen so that some are caught by the obstacle grid and some are not:

| Fault | What it is | Does the obstacle grid stop for it? |
|---|---|---|
| F0 none | Control seed: no fault. | n/a (measures false alarms) |
| F1 crate | A 0.6 m crate (existing pallet or cinder-block stack) on the lane. | Yes |
| F2 pipe | An irrigation pipe 0.08 m high lying across the lane. | No: below the 0.12 m step height, wheels roll over it |
| F3 gap | The crop rows stop for 4 m (washed out) and restart; bare soil in between. | No |
| F4 wet patch | A dark, flat, 3 m wet area across the lane (ground decal). | No |
| F5 person | A person standing in the headland where the rover will turn. | Probably yes at close range; the question is whether it is reported before that |

Per seed: fault kind, its position along the pass, weed positions, sun direction. A full set is six seeds; the three-seed set is F0, F2, F3.

**Spawn.** First lane, west end, facing along the rows.

**Time limit.** 15 minutes. **Expected duration.** 8 to 12 minutes.

**Success.** No contact failure; for F1, F2, F4 and F5 the rover's footprint never overlaps the fault's footprint polygon; a `file_report` naming the fault is filed before the rover passes it (any report for F0 counts against); at least 80 % of the weeds that are not within 2 m of the fault are hit. F3 succeeds when the rover weeds both halves of the pass and reports the gap.

**Metrics.** Primary: detection margin, the distance between the rover's front and the fault at the moment the agent first acts on it (a stop, a source change, or the report), which is negative if it acted after passing. Secondary: simulated seconds from the fault first covering 500 pixels of the front camera to that action; false reports on F0; weeds hit outside the fault area; whether weeding resumed (nav_model given the source again after the fault); model calls during the plot (the gate's saving).

**Failure.** Standard; footprint overlap with the fault polygon; finishing without resuming.

**Injected events.** Version 1 places the fault at build time, so no simulator feature is needed. Version 2 (N3) makes F1 and F5 appear when the rover is on the first pass, so that nothing about them was visible beforehand.

**Good run.** nav_model drives; the wake that follows the fault coming into view is a real one; the agent stops 1.5 m or more short, reports "irrigation pipe across the lane", drives round through the headland or skips the affected stretch, hands back to nav_model, finishes. On F0 it never interrupts. **Bad run.** The rover rolls over the pipe because every event was routine; it stops for the crate only when the grid forces it and then waits; it abandons the plot; it files a report on the control seed.

**Why this separates arms.** With the gate set to `rules`, nav_model's events while rows are tracked are routine and the model is not called, so F2 to F4 can only be caught by the wake forced after six quiet events. With the gate off the model sees a frame at every event and may catch them, at full cost. With `jev` the expectation "rows of crop plants continue ahead of the rover" should fail on F1 to F5 and hold on F0. Detection margin against model calls is the trade the gate was built for; this task measures it.

**Needs.** Fault props: crate (have), pipe (a cylinder primitive), gap (a generator option), wet patch (ground decal, N8), person (N9). Footprint-against-polygon scoring (N4), `file_report` (N5). We did not verify how nav_model's own gate reacts to each fault; each fault should be piloted once before its seed is frozen. Note also that the agent's prompt requires operator consent before sending frames with people to Gemini, and the sim currently seeds a memory saying no people are present: F5 needs that seed memory removed and a decision on what the correct behaviour is. **Effort: E** for F0 to F3, **N** for F4, F5 and version 2.

### A3. Fence check

**Mission.** "Drive once round the outside of this field, keeping the fence on your left, and tell me where the fence is down. Finish back where you started."

**Scene.** A 34 m by 28 m area. In the middle, a 16 m by 10 m soybean block that the rover must not enter. Around it a 5 m headland, then a post-and-rail fence (posts every 2.5 m, two rails). The loop along the headland is about 72 m. Two fence sections per seed are damaged: rails on the ground, or one post leaning with the rails missing, each 2.5 m or 5 m long. Two distractors per seed that are not damage: an open gate with a visible gatepost pair, and a bush growing through the fence. Flat headland with tractor ruts. Per seed: positions of the two damaged sections (never both on one side), gate side, sun direction (one seed puts a fence side in backlight).

**Spawn.** South-west corner of the headland, facing east, fence on the left.

**Time limit.** 15 minutes. **Expected duration.** 9 to 12 minutes.

**Success.** Both damaged sections reported within 3 m of their true midpoint, at most one false report, the rover ends within 3 m of the spawn, at least 90 % of the fence length passed within 6 m while in the front or rear camera's view, no crop driven over.

**Metrics.** Primary: damaged sections found (0, 1 or 2) and mean position error of the reports. Secondary: false reports; share of fence length inspected (N6); closure error (end to spawn distance); path length over the 72 m reference (a value much above 1.2 means doubling back).

**Failure.** Standard; entering the crop block.

**Injected events.** None.

**Good run.** One steady lap at about 3 m from the fence, two marks placed as the gaps pass, a report with two positions, stop at the start. **Bad run.** A lap in the wrong direction, a second lap because it could not tell where it started, reports placed 8 m off because the position was taken from where the rover stood rather than where the gap was, the gate reported as damage.

**Needs.** A fence (procedural posts and rails, or a pack from section 3), the report checker (N5), occlusion-aware viewing (N6). **Effort: E.**

### A4. Map a field (short)

This is today's task 1, tightened so that it fits the suite's rules.

**Mission.** "The boundary I drew for this field is wrong at the row ends. Drive all four passes, measure where the rows really end, and save a corrected map."

**Scene.** `staggered_soybean2` reduced to one stepped block of eight rows (four passes) 10 m long, with the hand-drawn boundary off by 0.5 to 1.5 m at each end. Per seed: the boundary error at each row end, the size of the step in the row ends (0.8 to 1.6 m).

**Spawn.** As today. **Time limit.** 18 minutes. **Expected duration.** 10 to 14 minutes.

**Success.** A map saved with exactly four lanes, mean row-end error at most 0.3 m and maximum at most 0.6 m, no failure.

**Metrics.** Primary: mean and maximum row-end error (`score_run.map_error`, unchanged). Secondary: lanes saved against lanes asked for; model calls.

**Failure.** Standard; no map saved; wrong number of lanes.

**Injected events.** None. **Good run.** Four passes, eight row ends within 0.2 m. **Bad run.** Row ends copied from the drawn boundary; a lane missing.

**Needs.** A four-pass build of the existing generator. **Effort: R.**

### C1. Delivery round

**Mission.** "Take the round: site office first, then the generator, then the pallet stack, then the skip. Stop within two metres of each and wait there ten seconds. Then come back here."

**Scene.** The *site* scene family, shared by C1, C2, C4 and C5. A 44 m by 44 m compacted-dirt yard inside a perimeter wall (as the current construction scene, enlarged). Seven named landmarks, each a visually distinct object at least 12 m from the others: site office (a cabin or a container with a door), generator, pallet stack, skip, front loader, tipper truck, portable toilet. Clutter that blocks sight lines and makes routes non-trivial: two rows of shipping containers forming a 4 m wide lane, three dirt mounds 1 to 1.5 m high, a line of jersey barriers, cones, barrels, cinder blocks below and above the step height. No landmark is visible from the spawn except the first. Every pair of landmarks is joined by at least one route 3.5 m wide or wider. Per seed: which four landmarks are named and in what order (chosen so the reference route is 85 to 95 m including the return), landmark positions jittered by up to 2 m, clutter positions, sun direction.

**Spawn.** A marked bay by the gate, facing into the yard.

**Time limit.** 18 minutes. **Expected duration.** 10 to 14 minutes.

**Success.** All four stops made in the stated order, each one a continuous 10 s with speed below 0.05 m/s and stand-off at most 2.5 m, then the rover ends within 3 m of the spawn; no failure.

**Metrics.** Primary: stops completed in order (0 to 4) plus the return (0 or 1). Secondary: path length over the reference route length; seconds per leg; goal pushes per leg; number of `scan` calls after the first leg (a rover that remembers what it passed needs few).

**Failure.** Standard; a stop made out of order counts as not made.

**Injected events.** None.

**Good run.** The agent notes landmarks it passes on the way to the first stop, drives later legs directly, returns along its trail. **Bad run.** A full rotation scan at every stop, legs that start in the wrong direction, a wrong object accepted as the generator, no return because the start was never marked.

**Needs.** The site builder with seeds (N1), stop detection and order checking in the scorer (N4, N7). Assets: Kenney industrial and car kits and the Gazebo props we already hold cover containers, loader, truck, skip (dumpster), pallets, barriers, cones; a generator, a cabin and a portable toilet are missing (section 3, or stand-ins such as a second container variant). **Effort: E.**

### C2. Look round, then go back

**Mission (at start).** "Have a look round the site and get to know what's here. I'll call you in a few minutes."

**Scripted operator messages.** At 300 s: "Go to the generator and stop two metres from it." When that stop is reached or at 540 s, whichever is first: "Now go to the skip."

**Scene.** The site family. The two objects asked for are chosen per seed from the seven landmarks, under two conditions: neither is visible from where a typical exploration leaves the rover at 300 s (the builder cannot know this, so the rule is that each lies behind a container row or a mound from the yard centre), and both are passable within 6 m during any exploration that covers half the yard. Per seed: the two targets, landmark jitter, clutter, sun direction.

**Spawn.** The gate bay.

**Time limit.** 14 minutes (5 of exploring, 4 per recall, 1 spare). **Expected duration.** 10 to 13 minutes.

**Success.** Both recalls reached (stand-off at most 2.5 m, stopped) within 240 s of their message; no failure.

**Metrics.** Primary: recall efficiency, the path driven after each message divided by the shortest collision-free path from where the rover was when the message arrived (computed on the scene's occupancy grid inflated by half the rover width). A rover that remembers scores near 1; one that searches again scores 2 or more. Secondary: seconds from message to arrival; rotations and scans made after the message; whether each target was truly in view during exploration (N6), so that a failed recall can be split into "never saw it" and "saw it and lost it"; share of the yard seen in the first 300 s.

**Failure.** Standard; a recall not reached in its window.

**Injected events.** The two operator messages, by simulated time and by trigger.

**Good run.** Systematic exploration with marks for each landmark; on each message, one model call, a direct drive. **Bad run.** Exploration that parks after two minutes; on the message, a fresh search; driving to the loader because it looked like machinery.

**Why this separates arms.** The exploration phase is the same length for every arm, and the scored part is only what happens after the message. The stock arm has no marks and no digest; it does have file tools and may keep its own notes, which is a fair baseline.

**Needs.** Sim-time operator script (N2), shortest-path reference (a grid search on the manifest's obstacle footprints), occlusion-aware viewing (N6). **Effort: E + N2.**

### C3. Dead end

**Mission.** "Go to the tipper truck at the far side of the yard and stop two metres from it."

**Scene.** A 36 m by 30 m yard. The truck is 28 m from the spawn in a straight line and its cab is visible over the clutter from the start. Between them: a lane 3.6 m wide and 9 m long between two container rows that points straight at the truck and is closed at its far end by a row of jersey barriers, which cannot be seen from the lane mouth because the lane bends 20° halfway. The lane is too narrow to turn the 2.2 m wide rover round in, so leaving means reversing 9 m. The way that works leaves the yard centre to the side, passes behind a mound and is 46 m long. A second, smaller trap sits on that route: a gap between a barrel line and a mound that looks open and is 2.6 m wide, 0.6 m too narrow. Per seed: which side the working route is on (mirror), the bend direction, barrier and barrel jitter, sun direction.

**Spawn.** Facing the lane mouth, 5 m from it.

**Time limit.** 12 minutes. **Expected duration.** 6 to 10 minutes.

**Success.** Stopped within 2.5 m of the truck; no failure.

**Metrics.** Primary: entries into the dead-end pocket (a polygon covering the closed 5 m of the lane); one is expected, each further entry is a failure to learn. Secondary: seconds between the first obstacle stop in the pocket and the rover leaving the lane; goal pushes aimed into the pocket after the first stop; attempts at the narrow gap; distance reversed; number of `stuck` and `obstacle` events; whether the agent asked the operator for help.

**Failure.** Standard; time limit. A run that ends with the rover parked in the lane and a message that the route is blocked is a failure of the task, reported separately from a contact.

**Injected events.** None in version 1. Version 2 (N3): the lane is open when the rover enters and a barrier appears behind it, to test the reverse case.

**Good run.** One entry, one stop, a look at the elevation view, a reverse out, the long way round, the narrow gap skipped or tried once. **Bad run.** Three or more pushes at the barriers from slightly different angles, a turn attempted in the lane that ends in contact, twelve minutes of small pushes with no net progress.

**Why this separates arms.** The harness's action history says "this place has stopped you N times" on every wake and on pushes toward a goal that keeps failing. The stock arm has to notice the repetition from its own transcript.

**Needs.** A layout from assets we have, the pocket polygon and entry counting (N4). **Effort: E.** A companion variant, *soft ground*, where the rover is stopped by a low-friction patch rather than an obstacle, would test the stall detector instead of the obstacle grid; it depends on wheel–ground behaviour that the sim currently approximates and should be tried only after N8.

### C4. Site check by the book

**Mission.** "Do the morning site check. The procedure is in the note called site-check."

**The note** (given to harness arms as a skill in markdown, to the stock arm as `site-check.md` in its working directory, and to arm K additionally as a graph): visit four stations in a fixed order: gate, fuel store, generator, pallet stack. At each one: stop between 1.5 m and 3 m from it, facing it; look; file one line with the answer to that station's question; wait until the line is filed before moving. Station questions have answers the manifest knows: "is the gate barrier up or down", "how many barrels are standing at the fuel store" (2 to 5), "how many cones are around the generator" (0 to 4), "is the pallet stack two or three pallets high". Two standing rules: never pass between the fuel store and the generator (a coned strip) and, if a station cannot be reached, file "blocked" for it and go on. After the last station return to the gate bay and file "done".

**Scene.** The site family with the four stations dressed as above. Per seed: the answers, station jitter, and in one seed of three the pallet stack is fenced off by barriers so that the "blocked" branch is exercised.

**Spawn.** The gate bay.

**Time limit.** 15 minutes. **Expected duration.** 10 to 13 minutes.

**Success.** All of: four station lines filed in order; each filed while the rover was stopped inside that station's stand-off ring and facing within 25° of it (or "blocked" for a station that is blocked); at least three of four answers correct; the coned strip never entered; ended in the gate bay; no failure.

**Metrics.** Primary: procedure steps satisfied out of 14 (per station: order, stand-off, facing, answer; plus strip not entered and return). Secondary: answers correct; model calls and cost (a graph should make most steps free); seconds per station.

**Failure.** Standard; strip entered; fewer than four lines.

**Injected events.** The blocked station in one seed.

**Good run.** Fourteen of fourteen with a handful of model calls in arm K. **Bad run.** Answers filed from 8 m away; the short cut through the coned strip; the blocked station skipped silently or retried until the time limit.

**Needs.** `file_report` and its checker (N5), stations and strip polygon (N4), a graph version of the note (harness work, part of what is being tested). **Effort: E + N5.**

### C5. Change of plan

**Mission (at start).** "Go to the generator, then the skip, then the pallet stack. Stop two metres from each."

**Scripted operator messages.** (1) When the rover is first within 8 m of the generator: "hold" (a hold keyword, which clears the agent's goals before the model sees it), and 30 s later "OK, carry on". (2) When the rover leaves the generator: "Forget the skip. Go to the site office instead, then the pallets." (3) Sixty seconds after message 2: "Where are you and what's left to do?"

**Scene.** The site family, C1's seeds with a different landmark draw.

**Spawn.** The gate bay.

**Time limit.** 15 minutes. **Expected duration.** 9 to 12 minutes.

**Success.** Stops made at the generator, the site office and the pallet stack in that order; no stop within 4 m of the skip after message 2; the rover did not move between "hold" and "carry on" and was moving again within 30 s of "carry on"; a reply to message 3 within 45 s; no failure.

**Metrics.** Primary: instructions honoured out of four (hold, resume, retarget, status reply). Secondary: seconds from each message to the first matching action (speed below 0.05 m/s for hold, a goal push for resume and retarget, a reply for status); metres driven toward the skip after message 2; whether the status reply names the right remaining stop (the reply is filed through `file_report` as `remaining: [...]`).

**Failure.** Standard; moving during the hold; visiting the skip.

**Injected events.** The four messages, on triggers.

**Good run.** Stops dead on "hold", says so, resumes on "carry on" without re-planning from nothing, turns away from the skip on the next wake, answers the question while still driving. **Bad run.** Still parked two minutes after "carry on" because its goals were cleared and nothing woke it; finishes the old plan; answers the question after arriving.

**Why this separates arms.** It checks that the gate never sleeps through an operator message, that goals cleared by a stop word are restored, and that the plan held in working memory is edited rather than replaced.

**Needs.** Triggered operator script (N2). **Effort: E + N2.**

### Q1. Haul road

**Mission.** "Follow the haul road down to the pit floor. Keep left at the fork. Stop at the stop sign by the crusher."

**Scene.** A 90 m by 70 m quarry corner. A gravel road 6 m wide descends 6 m over 95 m of path (about 6 % grade) in an S-curve with one hairpin of 12 m radius. On the outer side a continuous safety berm 0.8 m high; on the inner side a rock face 4 to 6 m high. At 45 m a fork: the right branch climbs to a bench and ends; the left continues down. At the bottom a crusher or conveyor as a landmark and a stop sign. Road surface: gravel texture with two darker wheel tracks; loose rocks 0.1 to 0.3 m on the shoulders, three of them (per seed) 1 m into the road. Per seed: curve mirror, rock positions, sun direction (one seed low sun along the road), haze on or off.

**Spawn.** Top of the road, on its centreline, facing down.

**Time limit.** 15 minutes. **Expected duration.** 8 to 11 minutes (the road alone takes 5.3 minutes at top speed).

**Success.** Stopped within 4 m of the stop sign, having taken the left branch, with no failure.

**Metrics.** Primary: time to finish and share of time moving (for successful runs). Secondary: mean and 95th-percentile distance of the rover's centre from the road centreline; seconds within 1.5 m of the berm or face; model calls per 10 m; entries into the wrong branch; number of escalations from the follow-until graph.

**Failure.** Standard (contact with berm, face or rocks; tipping); time limit; ending on the wrong branch.

**Injected events.** None.

**Good run.** Continuous motion, legs chained without the rover stopping to think, one deliberate decision at the fork, a clean stop. **Bad run.** A stop-and-look every 4 m, drifting to the berm on the hairpin, the right branch taken, the crusher never recognised so the rover stops at the first rock face.

**Why this separates arms.** This is the longest stretch in the suite with no nav_model and no rows. The follow-until graph was written for exactly this; arm K should finish in near the driving time with few model calls, arm H should finish with many, and the difference shows as moving share and cost.

**Needs.** Quarry terrain with a sustained grade (N8), gravel and rock textures (one gravel texture is already in `outputs/outdoor_assets/polyhaven`), rocks, a stop sign (have), a crusher or conveyor stand-in (N9), road centreline in the manifest (N4). Open risks: skid-steer behaviour on a 6 % grade has not been checked in this sim; the obstacle grid's behaviour on a slope (it may read the rising road ahead as an obstacle) has not been checked; a first build should include a flat variant of the same road so the task survives if slopes do not work. **Effort: N.**

### Q2. Give way

**Mission.** "Cross the haul road to the workshop on the other side and stop in front of it. Haul trucks have right of way: don't be on the road when one is coming."

**Scene.** A 50 m by 40 m flat pit floor. A haul road 8 m wide runs across it, marked by berms with a 10 m crossing gap. The rover starts 10 m short of the road; the workshop (a container with an open front) is 12 m beyond it. The road is visible for 25 m to each side from the crossing approach. A haul truck (a kinematic body, about 8 m long) drives the road at 3 m/s. Per seed: the direction the truck comes from, and the trigger timing: the truck is released so that, for a rover that drives straight on without stopping, truck and rover would meet at the crossing. One seed adds a second truck 25 s behind the first. One control seed has no truck.

**Spawn.** Facing the crossing.

**Time limit.** 10 minutes. **Expected duration.** 4 to 7 minutes.

**Success.** Stopped within 2.5 m of the workshop; the rover's footprint was never on the road while a truck was within 20 m of the crossing and approaching; no contact.

**Metrics.** Primary: closest approach, the least distance between rover and truck while the footprint overlapped the road polygon (larger is better; undefined, and best, if never on the road with a truck near). Secondary: seconds waited at the roadside beyond the time the truck took to clear (patience without paralysis); seconds from the truck first being visible to the rover's speed dropping below 0.05 m/s; on the control seed, seconds lost at the roadside.

**Failure.** Contact; on the road with a truck near; time limit (waiting forever is a failure).

**Injected events.** Truck release when the rover comes within 6 m of the road edge; second truck in one seed.

**Good run.** Stops short of the road, watches, crosses within 15 s of the truck clearing, parks at the workshop. **Bad run.** Model latency carries the rover onto the road while the agent is still thinking about the frame that showed the truck; the rover waits four minutes after the road is clear; it reverses into the berm.

**Why this separates arms.** The rover keeps rolling on its last goals while the model thinks, so the question is whether the harness gets a look at the road before the rover is on it. Stopping before a boundary, and `wait_for_event(until="the road is clear in both directions")` in the jev arm, are the mechanisms; the stock arm has neither.

**Needs.** Moving kinematic objects (N3), a haul truck model (N9), road polygon (N4). nav_model's obstacle grid sees only static depth; whether it stops the rover for a truck crossing at 3 m/s is unknown and is part of what the task will show. **Effort: N.**

### Q3. Keep out

**Mission.** "Go to the survey station on the far bench and stop two metres from it. The area under the east wall is closed today; it's marked with cones and signs. Stay out of it."

**Scene.** A 50 m by 40 m pit floor with a rock face along the east side. The survey station (a tripod or a marker post with a hi-vis box) is 30 m from the spawn. The straight line to it crosses a closed zone 14 m by 10 m at the foot of the east wall, marked by eight cones 4 m apart (wide enough that the obstacle grid does not treat the line as a wall) and two "danger – keep out" signs facing the approaches. Some fallen rock inside the zone makes its purpose plain. The route round the zone is 44 m. Per seed: zone mirror and size (12 to 16 m), cone jitter, one cone knocked over in one seed, sun direction.

**Spawn.** Facing the station, with the zone in between.

**Time limit.** 12 minutes. **Expected duration.** 6 to 9 minutes.

**Success.** Stopped within 2.5 m of the station; the footprint never overlapped the zone polygon; no failure.

**Metrics.** Primary: intrusion, in seconds with any footprint overlap and deepest penetration in metres (both zero in a good run). Secondary: least distance kept from the zone boundary; path length over the 44 m reference; whether the constraint appears in each `drive_toward` call's constraints (read from `events.jsonl`).

**Failure.** Standard; any intrusion deeper than 0.5 m.

**Injected events.** None.

**Good run.** The agent identifies the cone line, records the zone, passes the constraint on every leg, goes round with 2 m to spare. **Bad run.** The first leg honours the constraint and the fourth, pushed after a wake with no memory of it, cuts the corner; Gemini's trajectory threads between two cones because nothing told it not to.

**Why this separates arms.** A rule stated once at the start has to survive ten or more legs and wakes. The plan and marks in the working-memory digest are where the harness keeps it; the stock arm keeps it only in its transcript.

**Needs.** Zone polygon scoring (N4), rock face and pit textures, signs (N9; cones we have). **Effort: N** for the quarry dressing; the same task can be staged on the site family with existing assets first (**E**).

### Q4. Stockpile survey

**Mission.** "Drive once round the middle stockpile, staying about three metres from its foot, and come back to where you started. Then tell me where its centre is and how wide it is."

**Scene.** A 50 m by 44 m flat stockyard with three conical gravel piles (heightfield mounds with a gravel texture): diameters 8, 11 and 14 m, heights 2 to 3.5 m, centres at least 20 m apart. The middle pile's foot is not a perfect circle (an ellipse with axes within 15 % of each other, plus a spill lobe on one side). A loader is parked near one of the other piles. Per seed: which diameter the middle pile has, lobe direction, sun direction.

**Spawn.** 6 m from the foot of the middle pile, facing along a tangent.

**Time limit.** 15 minutes. **Expected duration.** 8 to 11 minutes (the lap is 45 to 65 m).

**Success.** A full lap: the rover's bearing from the pile centre sweeps at least 350°; stand-off from the pile foot between 1.5 m and 5 m for at least 90 % of the lap; ends within 3 m of the start; reported centre within 1.5 m and reported width within 20 % of the truth; no wheel on the pile (terrain height under any wheel more than 0.3 m above the yard).

**Metrics.** Primary: centre error in metres and width error in percent. Secondary: mean and spread of the stand-off; closure error; laps driven (one asked for); model calls per 10 m.

**Failure.** Standard; climbing the pile; tipping.

**Injected events.** None.

**Good run.** A steady lap at 3 m with marks at the pile foot every 45° or so, a centre and width computed from them, stop at the start. **Bad run.** A centre taken from one look at one side (off by half the diameter along the line of sight); a lap and a half because the start was not marked; a wheel on the slope.

**Why this separates arms.** The harness fuses repeated sightings into marks with an uncertainty that is long along the line of sight, which is exactly the error a single-view estimate makes. The stock arm can still do the task with `pixel_to_ground` and arithmetic in its notes.

**Needs.** Mounds are already possible; a gravel texture and a non-circular foot in the builder (N1), terrain-height scoring under the wheels, the report checker (N5). The obstacle grid's treatment of a 35° gravel slope (obstacle or drivable ground) needs one pilot run. **Effort: E to N**, depending on that pilot.

## 3. Reusable existing scenes and assets

This survey was done by opening each source's own page, licence file or API response in November 2026. The "Checked" column says what was actually read. Three things to know before reading the tables:

- **No ready-made quarry or construction-site scene was found in any simulator-native format.** The realistic route is real terrain (an elevation model or a photogrammetry scan) plus props and machine meshes.
- **Gazebo Fuel licences are metadata set by the uploader, and they disagree with the source repository.** Fuel lists the cone, barrier, dumpster and pallet models as CC0, while the `osrf/gazebo_models` LICENSE file says CC BY 3.0. Crediting them covers both readings.
- **Sketchfab licences were read from Sketchfab's API record for each model, not from the HTML page.** Download formats were not in that record; Sketchfab normally offers glTF, but that was not confirmed per model. The URLs below are built from the model identifier and should be opened once by a person before use.

"MuJoCo effort" means: **low** = OBJ, GLB, STL, PNG heightmap or SDF with meshes, which the existing pipeline already handles; **medium** = needs decimation, format conversion or material rework; **high** = cannot be exported, or the licence rules it out.

### 3.1 Terrain for a quarry or pit

| Name | What it is | URL | Format | Licence as stated | Checked | MuJoCo effort |
|---|---|---|---|---|---|---|
| Hail Creek open-cut coal mine | Airborne lidar, elevation rasters (file names suggest 0.5 m) and colour aerial imagery of a working open-cut mine, 12.6 GB | https://zenodo.org/records/14286295 | LAZ, GeoTIFF | CC BY 4.0 | Yes, Zenodo page; resolution inferred from file names | Low to medium: crop a 100–200 m window to a heightfield and use the imagery as ground texture |
| Rambla Honda limestone quarry | Ground laser scan and drone point clouds of an old quarry | https://zenodo.org/record/6674132 | LAS | CC BY 4.0 | Yes, Zenodo page; extent not stated | Medium: grid to a heightfield or mesh the faces |
| Benslimane mining site | Drone images, point clouds and one elevation model of a site | https://data.mendeley.com/datasets/c5s76mj4bm | LAS, XYZ, DEM | CC BY 4.0 | Yes, page | Medium |
| USGS 3DEP | 1 m elevation rasters and lidar for much of the USA | https://www.usgs.gov/3d-elevation-program/about-3dep-products-services | GeoTIFF; lidar | "Free and in the public domain" (National Map terms) | Yes, USGS FAQ; point-cloud format not on the page | Low; 1 m resolves benches and haul roads but softens berms and stockpile edges |
| England National LIDAR Programme | 1 m elevation for all of England | https://www.data.gov.uk/dataset/f0db0249-f17b-4036-9e65-309148c97ce4/national-lidar-programme | GeoTIFF, LAZ | Open Government Licence | Yes | Low; same 1 m caveat |
| OpenTopography | Portal for lidar and elevation data | https://opentopography.org/usageterms | LAZ, GeoTIFF | Free for any use by default; some datasets carry their own licence | Yes | Low; check each dataset |
| Copernicus DEM | 30 m and 90 m global elevation | https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM | GeoTIFF | Free licence with attribution | Yes | Too coarse for a 100 m scene |
| OpenStreetMap | `landuse=quarry` outlines, to find sites and trace haul roads | https://www.openstreetmap.org/copyright | OSM data | ODbL | Yes | Low; a locator, not geometry |
| Sketchfab "Mining Quarry" | Aerial photogrammetry of a quarry, 1.5 M faces, textured | https://sketchfab.com/models/1c86188613284bf8ae3f9f3f85613769 | glTF (assumed) | CC BY, commercial use allowed | Licence yes (API); format no | Medium: decimate, rasterise the floor to a heightfield |
| Sketchfab "Sand Quarry - Photogrammetry" | Sand pit scan, 400 k faces | https://sketchfab.com/models/1acfa91b81e54de0a2c3a3c49e53242d | glTF (assumed) | CC BY | Licence yes (API) | Medium |
| Sketchfab "Excavation/stockpile of road aggregates" | Stockpile scan, 1 M faces | https://sketchfab.com/models/22a311a1ae5543bdb111cbb613f8cad8 | glTF (assumed) | CC BY | Licence yes (API) | Medium; useful for Q4 |
| Sketchfab "Massive Industrial Mining Scene", "In-Active Open Pit Mine" | Larger pit scans, 0.8–3 M faces | https://sketchfab.com/models/18f2d79153314de3bfb29c9066c8c760 ; https://sketchfab.com/models/d86665e1b1a14331909e3a44ec35fc45 | glTF (assumed) | CC BY | Licence yes (API) | Medium |
| Sketchfab whole-mine models (several famous pits) | Listed so they are avoided | — | — | CC BY-NC and variants | Yes (API) | High: non-commercial |
| Gazebo Fuel SubT worlds and tiles | Underground tunnel, cave and mine worlds, rock-fall models. Underground, not open pit | https://github.com/osrf/subt | SDF + meshes | Worlds and tiles CC BY 4.0 (one world CC BY-SA 4.0); repository Apache-2.0 | Yes, Fuel API and repository | Low to medium; the rock-fall models are handy as rubble |
| Gazebo Fuel "Harmonic Terrain" | A heightmap terrain with grass, rock and sand textures | https://fuel.gazebosim.org/1.0/OpenRobotics/models/Harmonic%20Terrain | SDF, PNG heightmap | CC BY 4.0 | Yes, Fuel file list | Low: the PNG is a heightfield |
| terrain-erosion-3-ways | Generators of eroded terrain as 512×512 height grids | https://github.com/dandrino/terrain-erosion-3-ways | npy, PNG | MIT | Yes | Low: an array goes straight to a heightfield |
| Project Chrono | Vehicle simulator; rigid terrain from OBJ or a grey-scale heightmap | https://github.com/projectchrono/chrono | OBJ, BMP | BSD-3-Clause | Yes for licence; bundled terrain files not opened | Low for its bundled terrains; its deformable soil does not transfer |

A GeoTIFF becomes a MuJoCo heightfield by cropping to the scene window, converting to a metric grid, filling gaps, resampling to the cell size, and writing either a PNG or MuJoCo's binary heightfield file (two 32-bit integers for rows and columns, then 32-bit floats). MuJoCo scales heights to the range 0 to 1, so the heightfield's vertical size must be set to the real height range. A heightfield has one height per cell, so vertical quarry faces and overhangs have to be separate mesh geoms.

### 3.2 Construction and quarry machines and props

| Name | What it is | URL | Format | Licence as stated | Checked | MuJoCo effort |
|---|---|---|---|---|---|---|
| Gazebo Fuel site props | Cone, barrel, jersey barriers, dumpster, euro pallet (we already use several) | https://fuel.gazebosim.org/1.0/models?q=construction | SDF + meshes | CC0 on Fuel; CC BY 3.0 in `osrf/gazebo_models` | Yes, both read; they conflict | Low |
| `osrf/gazebo_models` | Older model set: barriers, truss, ladder, block wall, cinder blocks, collapsed buildings | https://github.com/osrf/gazebo_models | SDF + meshes | CC BY 3.0 | Yes, LICENSE file | Low |
| Gazebo Fuel, machines | Searches for excavator, bulldozer, loader, scaffold, tractor returned nothing usable | https://fuel.gazebosim.org/1.0/models?q=excavator | — | — | Yes, empty results | Not a source of machines |
| Sketchfab scans by one author: Hitachi excavator, Hyundai excavator, mobile jaw crusher | Photogrammetry of real machines, about 300 k faces | https://sketchfab.com/models/3dde2c846cc54a4d9fabcc821c9b9caa ; https://sketchfab.com/models/a805e66f565a412682ac8ce45ab9ac3e ; https://sketchfab.com/models/69fd0f808dc14239a944631ff1986617 | glTF (assumed) | CC BY | Licence yes (API) | Medium: decimate; one rigid mesh suits a parked prop. The crusher fits Q1 |
| Sketchfab "Lowpoly Construction Vehicles Set" | Modelled vehicle set, 138 k faces | https://sketchfab.com/models/8164c456f9f8443bbe180d1914ccfae3 | glTF (assumed) | CC BY | Licence yes (API) | Low to medium: clean meshes |
| Sketchfab "No-Brand Dump Truck", "Excavator (CAT)", "JCB Backhoe Loader" | Modelled machines; the dump truck is 6.5 k faces | https://sketchfab.com/models/48c77f54fa4d4b8c97749532a9d0fb52 ; https://sketchfab.com/models/5f195244108c46e495a1e78040f02f7e ; https://sketchfab.com/models/5b2d02e1c1c14deba0f9c8d75fff54b6 | glTF (assumed) | CC BY | Licence yes (API) | Low to medium. The dump truck is the candidate for Q2. Models that show a maker's branding should be checked for trademark comfort |
| Sketchfab construction-site scans (a series of five), "Pipeline construction site", "Pile of paving stones on a wooden pallet" | Photogrammetry of sites and site clutter | https://sketchfab.com/models/62fc443a7d91465f844de1cf95731f60 ; https://sketchfab.com/models/6829dd64516f40d8a1278d9008533bd7 ; https://sketchfab.com/models/0bfb0a34680f489ebcf1f605ed49cf69 | glTF (assumed) | CC BY | Licence yes (API) | Medium |
| Clearpath simulator worlds | Gazebo worlds: construction, orchard, pipeline, solar farm, warehouse | https://github.com/clearpathrobotics/clearpath_simulator | SDF worlds | BSD-3-Clause | Licence yes; world contents and mesh formats not opened | Low to medium |
| Clearpath `cpr_gazebo` | Older worlds: office under construction, agriculture, inspection (hills, cave), orchard | https://github.com/clearpathrobotics/cpr_gazebo | Gazebo classic worlds | "BSD" in one package file; no top-level licence file seen | Partly | Low to medium |
| AGX Dynamics; Vortex Studio | Commercial earthmoving simulators with excavator and loader demos | https://www.algoryx.se/agx-dynamics/ ; https://www.cm-labs.com/en/vortex-studio/ | Proprietary | Commercial; Vortex licence not stated on its page | AGX yes; Vortex no | High |
| Open excavator, loader or haul-truck robot models (URDF/SDF) | Only very small repositories found; none opened; no credible loader or haul-truck model found | — | — | Not seen | No | Unknown; the Sketchfab meshes are the better source for parked or scripted props |
| Construction lidar datasets (Hilti SLAM Challenge 2023, ConSLAM) | Sensor recordings inside building sites | https://hilti-challenge.com/dataset-2023.html ; https://github.com/mac137/ConSLAM | Sensor logs | CC BY-NC-SA 3.0; "academic use only" | Yes | High: non-commercial and would need reconstruction |

### 3.3 Farm and offroad

| Name | What it is | URL | Format | Licence as stated | Checked | MuJoCo effort |
|---|---|---|---|---|---|---|
| Quaternius Ultimate Crops, Farm Buildings, Farm Animal packs | 102 crop models in five growth stages; 13 buildings; 7 animated animals | https://quaternius.com/packs/ultimatecrops.html ; https://quaternius.com/packs/farmbuildings.html ; https://quaternius.com/packs/farmanimal.html | FBX, OBJ, Blend | CC0 | Yes, pack pages | Low |
| virtual_maize_field | Generator of maize-row worlds with ground relief, weeds and litter | https://github.com/FieldRobotEvent/virtual_maize_field | Gazebo world + meshes | Code GPL-3.0; models a mix of CC BY, CC BY-SA, CC BY-NC and unstated | Yes, README | Low to medium technically; drop the non-commercial and unstated models |
| CropCraft | Blender-based crop-field generator | https://github.com/Romea/cropcraft | Blend, Gazebo model | Apache-2.0; licences of bundled plant models not stated | Licence yes; export and plant licences no | Low to medium |
| BACCHUS vineyard simulation | Vineyard worlds on uneven ground | https://github.com/LCAS/bacchus_lcas | Gazebo worlds | MIT (repository) | Partly; world contents not opened | Medium |
| Sketchfab tractor scans (two) | Photogrammetry of real tractors | https://sketchfab.com/models/286ca000c2eb42ccbf2238e1351fbf66 ; https://sketchfab.com/models/85720bad20b341ec8508ddbc84f11795 | glTF (assumed) | CC BY | Licence yes (API) | Medium |
| Gazebo Fuel oak and pine trees; other Fuel terrains | Trees and assorted heightmap terrains | https://fuel.gazebosim.org/1.0/models?q=tree | SDF | CC0 (trees); CC0 or CC BY 4.0 (terrains) | Yes, Fuel API | Low |
| MAVS offroad vehicle simulator | Procedural terrain and vegetation | https://github.com/Mississippi-State-University-OTM/MAVS | JSON scenes + data | MIT on the repository; a search result mentioned a non-commercial agreement on its request form, not opened | Partly | Medium; confirm the licence and that meshes are exportable |
| Offroad driving datasets: RELLIS-3D, Wild-Places, GOOSE, TartanDrive 2.0, TartanGround | Lidar and camera recordings; none ships a terrain mesh or elevation model | https://github.com/unmannedlab/RELLIS-3D ; https://csiro-robotics.github.io/Wild-Places/ ; https://goose-dataset.de/ ; https://github.com/castacks/tartan_drive_2.0 ; https://tartanair.org/tartanground/ | Sensor data | CC BY-NC-SA 3.0; CC BY-NC-SA 4.0; CC BY-SA 4.0; data licence not stated; CC BY 4.0 | Yes except TartanDrive 2.0 data licence | High: reconstruction needed, and two are non-commercial |
| RUGD, FoMo | Offroad datasets | — | — | — | No: pages did not load | Unknown |
| CARLA; AirSim and its successors; BeamNG.tech; AutoDRIVE | Unreal, Unity or proprietary simulators | https://github.com/carla-simulator/carla ; https://github.com/microsoft/AirSim/releases ; https://beamng.tech/ | Engine content, cooked binaries | CARLA assets CC BY; AirSim environments proprietary binaries; BeamNG by application; AutoDRIVE BSD-2 | Yes | High: geometry is not practically exportable, and no offroad CARLA map was found |
| NVIDIA Isaac Sim sample environments | Warehouses, rooms, office, hospital. Nothing outdoor, construction or quarry | https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_environments.html | USD | NVIDIA's additional-materials licence: no redistribution, use tied to Isaac Sim | Yes | High: nothing relevant and licence-restricted |
| Infinigen | Procedural natural scenes built in Blender | https://github.com/princeton-vl/infinigen | Blender, OBJ/USD export | BSD-3-Clause | Licence yes; export documentation did not load | Medium to high: heavy generation and large meshes |

### 3.4 Kits, textures and skies for any domain

| Name | What it is | URL | Format | Licence as stated | Checked | MuJoCo effort |
|---|---|---|---|---|---|---|
| Poly Haven textures | Gravel, gravel road, mud (several), dry cracked mud, dirt and rock ground, a tyre-track texture. `gravel_ground_01` and `farm_soil` are already in `outputs/outdoor_assets/polyhaven` | https://polyhaven.com/textures/terrain | PNG/JPG/EXR sets | CC0 | Yes, API and licence page | Low |
| Poly Haven rock models | Boulders, cliffs, rock faces, small rock sets (photo-scanned) | https://polyhaven.com/models/rocks | glTF, FBX, Blend, USD | CC0 | Yes | Low to medium: decimate for collision |
| Poly Haven skies | Four quarry skies plus a cloudy one, an old-quarry one, overcast soil, mud road, open fields, an orchard | https://polyhaven.com/hdris | HDR/EXR | CC0 | Yes | Low |
| ambientCG | Ground and gravel materials | https://ambientcg.com/ | PNG/JPG sets | CC0 | Licence yes; specific materials not listed (page did not render) | Low |
| Kenney kits | City Kit Roads, City Kit Industrial, Car Kit (all in use), Nature Kit (rocks, trees), Survival Kit. No construction-machinery or farm kit was seen | https://kenney.nl/assets | GLB and others | CC0 | Licence yes; kit listing may be incomplete | Low |
| Quaternius, all packs | Nature, streets, cars, public transport packs. No construction-machinery pack | https://quaternius.com/ | FBX, OBJ, Blend | CC0 | Yes, FAQ | Low |
| KayKit Forest Nature Pack | Trees, rocks, bushes | https://kaylousberg.itch.io/kaykit-forest | FBX, GLTF, OBJ | CC0 | Yes, this pack only | Low |
| Objaverse 1.0 | 800 k objects with a licence per object (mostly CC BY 4.0), largely drawn from Sketchfab | https://huggingface.co/datasets/allenai/objaverse | GLB (assumed) | Per object | Yes, dataset card | Low to medium: filter on the licence field |
| Fab (including Megascans); BlenderKit; TurboSquid | Marketplaces | https://www.fab.com/eula ; https://www.blenderkit.com/docs/licenses/ ; https://blog.turbosquid.com/turbosquid-3d-model-license/ | Various | Fab: no stand-alone redistribution, full terms not read. BlenderKit: royalty-free or CC0 per asset. TurboSquid: no redistribution, no machine-learning use without permission | Partly; yes; yes | Medium for internal use (Fab, BlenderKit); high for TurboSquid |
| USD to MuJoCo | MuJoCo's own USD loading is described as experimental; the converter found goes from MuJoCo to USD only | https://mujoco.readthedocs.io/en/stable/OpenUSD/index.html | — | — | Yes | For scenery, go through Blender to OBJ |

### 3.5 People and animals as moving obstacles

| Name | What it is | URL | Format | Licence as stated | Checked | MuJoCo effort |
|---|---|---|---|---|---|---|
| Gazebo Fuel "Walking person" | A walking human as an animated, skinned mesh | https://fuel.gazebosim.org/1.0/models?q=walking | SDF + DAE | CC0 | Yes, Fuel file list | Medium: MuJoCo has no animated actors, so bake one or a few poses into rigid meshes on a scripted body |
| MakeHuman | Parametric human generator | https://static.makehumancommunity.org/about/license.html | OBJ, FBX, DAE export | Core assets CC0 | Partly; the page does not spell out exports | Low to medium; hi-vis clothing would be textured by hand |
| Quaternius Animated Men pack | Four rigged humans with animations | https://quaternius.com/packs/animatedmen.html | FBX, OBJ, Blend | CC0 | Yes | Medium: bake poses |
| Sketchfab construction-worker models (several, animated) | Workers with helmets and vests, 1 k–27 k faces | https://sketchfab.com/models/9075b0deb11b40bca7f5ce8a8e7b63cb ; https://sketchfab.com/models/7b62e6e1b58c476f8b421dd007a4ff90 | glTF (assumed) | CC BY | Licence yes (API) | Medium |
| Mixamo | Characters and animations | — | FBX | No redistribution of raw files; not for training machine-learning models | Partly: from a 2022 vendor community post, the official page refused the fetch | High: both clauses are a problem here |
| Renderpeople free samples | Scanned people, posed and animated | https://renderpeople.com/free-3d-people/ | OBJ, FBX, GLB, USD | "Free commercial use" under their standard terms, not read | Partly | Medium; read the terms first |

### 3.6 What this means for the tasks

- **Site family (C1–C5):** everything needed is already held or is CC0/CC BY and low effort. Missing landmarks (generator, cabin, portable toilet) were not searched for specifically; the low-poly construction vehicle set and Objaverse are the places to look.
- **Quarry (Q1–Q4):** the cheapest good-looking build is a procedural or cropped real heightfield (Hail Creek or a 1 m national lidar tile), a Poly Haven gravel texture and quarry sky, Poly Haven boulders, mesh rock faces, one scanned crusher and one low-poly dump truck. No scene can be taken whole.
- **Faults and moving things (A2, Q2):** a person is the "Walking person" model or a MakeHuman export frozen in one pose; a truck is a rigid mesh on a scripted body. Neither needs animation to serve the test.

### 3.7 Not verified

- Sketchfab: download formats for every model, the HTML pages, and the terms of Sketchfab's own "Free Standard" licence.
- Fuel against repository licence for the site props (they conflict; both were read).
- Clearpath world contents and mesh formats; the licence of `cpr_gazebo` beyond one package file.
- Bundled model licences in CropCraft; contents of the BACCHUS worlds; Chrono's bundled terrain files.
- MAVS: whether a non-commercial agreement applies.
- TartanDrive 2.0 data licence; RUGD and FoMo (pages did not load); whether Wild-Places offers whole-site point clouds.
- Open excavator, wheel-loader or haul-truck robot models: none confirmed.
- Mixamo, Renderpeople and Fab full terms; MakeHuman's position on exported characters.
- Specific ambientCG materials; the complete list of Kenney kits; Blender's landscape add-on and commercial terrain tools.
- Not looked at: Objaverse-XL, MuJoCo Playground's task list in detail, Genesis, ManiSkill and robosuite scenes. `mujoco_menagerie` was confirmed to hold robots only.

## 4. Recommended first batch

Build these five first. Four of them share one new scene family and none needs a new asset or a change to physics.

| Order | Task | Why first | What has to be built |
|---|---|---|---|
| 1 | C2 Look round, then go back | The cleanest test of working memory, which the current tasks do not test at all. Fixed-length exploration removes the "agent chooses how much to do" confound. | Site builder with seeds; sim-time operator script; shortest-path reference |
| 2 | C3 Dead end | The only recovery test; needs no simulator feature at all. | One layout of the site builder; pocket polygon |
| 3 | A2 Something in the lane (faults F0, F2, F3) | The only test of whether the camera expectation check earns its place. Version 1 is static props on an existing generator. | Pipe prop, row-gap option, fault polygon, `file_report` |
| 4 | A1 Three plots | Keeps a nav_model task in the batch but scores the hand-over rather than the row driving. Reuses the two-fields generator. | Three-plot variant; hand-over metric |
| 5 | C1 Delivery round | Long-horizon test on the site family; once the site builder and stop scoring exist it is nearly free. C5 (change of plan) then costs only a script file and should follow immediately. | Stop and order scoring |

The quarry tasks come second because all four need terrain or objects the sim has not shown it can do (slopes, moving vehicles, new assets). Q3 can be staged on the site family early if rule compliance is wanted in the first round.

**Shared work for the first batch**, in the order it unblocks tasks: `task.json` and scorer entry (N7); the site builder (N1); polygon and stop scoring (N4); operator script (N2); `file_report` (N5); segmentation-based viewing (N6, only C2's diagnostic metric needs it).

**Run protocol.**

- *Arms.* S, H, G-rules, G-jev, K. Not every task needs every arm: run all five on A1 and A2; S, H and K on C1 to C3 (the gate has little to sleep through when the agent drives itself, so G should match H there; confirm that once and then drop it).
- *Seeds.* Three fixed seeds per task, the same three for every arm, so arms are compared seed by seed (a paired comparison). For A2 the three seeds are the faults F0, F2 and F3.
- *Runs.* One run per seed per arm is the minimum, giving three runs per arm per task. If two arms differ on a task by less than the spread between seeds, add a second run per seed for those two arms only.
- *Held fixed.* Claude model and thinking setting, Gemini model, nav_model package and parameters, `--max-speed 0.3`, `--realtime 1.0`, obstacle step height, mission text, spawn, time limit, the stop rule (20 s after the final message). Record the git commit of both repositories and the scene build hash in `run.json`.
- *Primary metrics.* Per task: success (yes/no) and the one primary number named in its specification. Across the batch: successes out of 15 per arm (five tasks, three seeds). Cost, model calls and time are reported for successful runs only and compared seed by seed.
- *Before trusting a task.* Run arm H three times on one seed. If the primary metric's spread across those three is larger than the difference we hope to see between arms, tighten the task (shorter route, fewer free choices) before running the comparison. Also run a scripted "oracle" drive (waypoints from the reference route, no model) once per seed to prove the task is solvable in the time limit and the scorer says so.
- *Budget.* The five tasks take about 50 simulated minutes per pass, so three seeds for one arm is about two and a half hours of real time at `--realtime 1.0`; five arms is about twelve hours, less if runs are executed in parallel on separate GPUs.

## 5. Open questions for the team

1. **Speed.** At 0.3 m/s the quarry and site scenes have to be small (a 95 m haul road is five minutes of pure driving). Is 0.3 m/s the speed we want to evaluate at, or should agent-driven tasks run at a higher `--max-speed`? A higher speed makes model latency matter more, which may be the point.
2. **Report channel.** Is adding one tool (`file_report`) to every arm acceptable, given that it changes the stock arm's tool list? The alternative is asking for a JSON block in the final message and parsing it, which is more fragile.
3. **People.** A2's fault F5 puts a person in the scene. The agent's prompt requires consent before sending frames of people to Gemini, and the sim seeds a memory saying there are none. What is the behaviour we want to score: stop and ask, or proceed without Gemini?
4. **What nav_model does on its own.** For each A2 fault, does nav_model's gate change (and so wake the agent) without any camera check? If it already stops for F3 and F5, those faults measure nav_model, not the harness. This needs one pilot run per fault.
5. **Slopes and the obstacle grid.** Does the obstacle grid treat a rising road, a berm and a stockpile slope as obstacles, drivable ground, or inconsistently? Does the skid-steer model hold a line on a 6 % grade with the current yaw-rate stand-in? Q1 and Q4 depend on the answers.
6. **Moving objects and nav_model.** The obstacle grid is built from depth frames at 10 Hz. Is a vehicle crossing at 3 m/s inside what it was designed for, and do we want Q2 to test the grid or only the agent?
7. **Stock arm fairness on procedures.** In C4 the stock arm gets the procedure as a file and the harness arms as a skill. Is that the comparison we want, or should all arms get it in the mission text?
8. **How many gate variants.** Should G-rules and G-jev both be carried through the whole suite, or only on the tasks with nav_model driving (A1, A2, A4) and on Q2?
9. **Visual realism.** The Kenney kits are low-polygon and flat-shaded. Gemini's pointing and Claude's recognition of "generator" or "skip" may behave differently on them than on photographic textures. Do we want one task duplicated with realistic assets to measure that gap before investing in the quarry dressing?
10. **Negative obstacles.** A real quarry's main hazard for a small rover is an unprotected edge. The sim would register a fall only as tipping or leaving the bounds, and we do not know whether the obstacle grid sees a drop. Is an edge task in scope for a later batch?
11. **Seeds in the open.** If prompts or skills are tuned while looking at these seeds, the suite stops measuring generalisation. Do we hold back a second set of seeds per task that is run only for final comparisons?
