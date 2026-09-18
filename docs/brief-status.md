# Brief status and deliverable evidence

This is a working prototype, not a completed cold-evaluation submission. Historical
design notes remain private under ignored `docs/agent/`; maintained operating
instructions are [running.md](running.md).

| Brief area | Evidence | Remaining limit |
| --- | --- | --- |
| Prompt-to-scene | Declarative programs, bounded intent/repair loop, selectable Codex/OpenRouter transport | Live OpenRouter comparison not yet run; requires key, two model IDs and budget |
| Metric accuracy | Explicit defaults, per-axis prompt evidence, complete source fields, archived evidence, compiled metric/mass checks | Source selection may be semantically wrong even when fields match; default masses are not calibrated |
| Grounded architecture | Joint room/opening sampling from frozen real records, independent geometry/lineage checks | Apertures scale with requested room area; no calibrated doorway prior; residential data does not validate factories |
| Furnishings/generalization | Generic placement, support, relation and access checks; direct catalog search/import with portable asset references | No empirical furnishing/group prior; static props do not establish manipulation capability |
| Articulation | Five-object brief profile, passive settle, finite sweeps, paired endpoint checks | Finite kinematic checking is not dynamic full-range proof for every asset |
| Robots/flows | Stock mobile base and Panda, noisy sensor streams, mapping, actuator-only drawer interaction, semantic navigation | General manipulation and learned policies unproven; VLM navigation uses a labeled static-map prior |
| Export | Independent URDF/PyBullet loading, articulation and inertia tests | Static-only loading is labeled separately; USD deferred |
| Visuals | Shared-geometry Cycles bridge; merged optional fal PBR layers and visual-only clutter; inspection galleries | Live PBR quality not tested in this pass; default remains flat; not photorealistic across domains |
| Data | Synchronized HDF5, tiered capture, backend/evidence-preserving variants; mixed-domain ten-run collection started | Completion and review of all ten captures still pending |
| Reproducibility | Locked dependencies, headless defaults, doctor, relocatable scene packages, configurable resource/cache roots | Clean-machine fifteen-minute target has not been certified |

## Evaluation evidence

After integrating `origin/pbr-materials` at `d529b9f`, the suite passed 161 tests
(non-fatal Euler-angle and upstream COLLADA deprecation warnings).
The optional OpenRouter adapter was tested with a mocked HTTP completion, structured
output and provider-reported billing; no live provider calls were made. The portable
skill passes its frontmatter validator. A built wheel installed into a fresh
temporary environment generated a validated template scene from outside the repo
with no API key and no agent on PATH; headless rendering also passed. This used
existing system libraries and cached dependencies, not a cold-machine timing trial.

The initial fixed-program matrix under `outputs/generalization_hardening/` has 16
runs: four domains, two backends, two seeds. Ten validated, five were partial and
one failed. All eight heuristic runs passed; architecture-plus-furnishing runs
exposed access and placement limitations. Failures and diagnostic views are retained.
This is intentionally not described as an eight-run live model comparison.

A separate local check at `outputs/generalization_hardening_checks/kitchen_interaction/`
completed a 60-second actuator-driven Panda drawer open/close sequence with no
physics warnings (2.49 mm opening and 0.062 mm closing endpoint error), recording
601 RGB-D/segmentation frames and 6,001 state/IMU/wrench samples. The rollout took
375 seconds before video encoding. This verifies the full capture path, not
warehouse task coverage. A same-geometry Cycles overview was also rendered for the
first heuristic kitchen fixture (8 samples, 640 px).

The cases share supported storage mechanics and one explicitly dimensioned novel
object per domain. They test mechanism portability/composition, not independent
domain realism. The corpus remains residential; generic backoff is recorded.
Deterministic expected inventories/dimensions are checked separately from physics.
There is no LLM/visual approval gate and no newly invented mapping accuracy target.

The separate `examples/evaluation_domains.json` live-only suite probes realistic
equipment breadth, including warehouse/material handling (racks, pallets, carts,
pallet jack, packing/staging). It has not been run against live models. Unsupported
or unverified capabilities are reported separately; only explicitly required
capabilities gate acceptance. Warehouse lifting/rolling probes are diagnostic,
not required for static scene construction. The existing cabinet-heavy results must not be cited as warehouse or
manufacturing generalization evidence. Inventory and generic geometry checks do
not yet verify industrial workflow, loaded lifting or rolling-cart tasks.

Direct asset search/import is now exposed to any command-capable agent and documented
in the portable skill. It returns the same package contract used downstream.
`outputs/retrieved_assets_final/` exercises eight real assets: seven import successfully;
the drill fails on a degenerate collision mesh. Static scene composition and URDF
verification pass for warehouse, kitchen and workshop subsets. The nursing station
imports but exceeds the test room's ceiling, so the clinic scene correctly fails.
These small tests are integration evidence, not realistic finished environments.
New SDF imports are static rigid props only; RoboCasa remains the existing articulated
retrieval path. G2 authoring and part-aware G4 remain separate prototypes.

## Fixes from the audit

The live tiled-home-kitchen exercise is under `outputs/home_kitchen_fal_final/`.
After tool fixes and library enrichment, the unchanged natural-language prompt
completed with two agent calls (one repair), 19 placed objects, six articulated
instances and eight visual-only fal clutter instances. The agent added a prep table
when the cooktop did not fit the chosen counters. All implemented scene/physics
checks passed; the optional microwave was dropped for lack of safe placement.
The final run reused cached assets with zero new fal spend; all four exploratory
runs together estimated $4.85 in fal charges (not provider-reported billing).
Earlier failed/partial runs are retained in `outputs/home_kitchen_fal*`.
This is a warm-library prompt-to-scene success, not a cold unrestricted retrieval
benchmark. Layout is heuristic, not distribution-verified. Visual review finds
recognizable kitchen equipment but schematic arrangement and over-bright lighting.

The exercise added open-vocabulary visual-only `asset-generate`/`generated_request`
support using the existing asset-reference contract, source-catalog abstention,
explicit GLB up-axis conversion, previous/frozen-intent repair context, and generic
large-first placement with rotated support candidates. Generated props can reuse
the library without network calls. Physical-object checks remain separate from
visual-only dressing. The portable skill documents the new tool; the regression
suite passes 171 tests.

- Swapped prompt axes and shipping-dimension substring matches are rejected.
- Resolved dimension claims cannot bypass the external evidence boundary.
- New caches archive evidence, validate on reuse, and do not modify package files.
- Evidence fetching has public-address/redirect restrictions and size/time bounds.
- Semantic hash collisions fail instead of silently overwriting the class map.
- Variants retain the original backend, prior bundle, evidence and validation profile.
- All model paths share a provider adapter, including optional VLM action chunks.
- Reported costs include nested sourcing calls; unavailable billing remains unknown.
- The misleading walk-in-to-reach-in navigation alias was removed.

## Current deliverable pass

`outputs/deliverables_cleanup/index.html` collects the inspection links. The kitchen
(60 m²) and warehouse (80 m²) both validate, have 1024 px Cycles stills, and pass
independent articulated URDF loading. Visual review finds recognizable equipment
but simplified, wall-heavy layouts and plain materials; this is not photorealism.
The first warehouse attempts failed access checks and remain archived. The accepted
scene uses revised declarative placement requirements, not edited generated geometry.

Five kitchen captures (seeds 100–104) and five warehouse captures (101–105) have
been started, each at 60 simulated seconds. All ten layouts passed preflight;
this does not imply ten completed captures. The first kitchen capture completed
with 601 RGB-D frames, 6,001 state/IMU samples, 1,201 range samples, IoU 0.639,
99.91% observed reachable coverage and 2.16 cm mean occupied-cell error. The
recorded robot travelled 1.46 m; visibility coverage is not physical traversal.
There were no physics warnings. Dataset completion is indicated by each batch's
`dataset.json`, not by this snapshot. Six runs use full capture and four lightweight.
Sampled onboard RGB frames at 0, 30 and 60 seconds are coherent with the low-mounted
stock base camera, but are dominated by table legs and show limited late-run motion.
This capture is useful plumbing/geometry evidence, not a strong exploration-policy
demonstration. The small base and its viewpoint limit realism for industrial robots.

The separate 60-second Panda interaction passed, with 2.49 mm opening error,
0.061 mm closing error and 17,750 contact samples. Its recorded video was sampled
visually and shows open–close–release. A two-second warehouse mapping smoke also
passed and has a reviewed replay; it is not full-duration dataset evidence.
The replay camera was corrected to frame room floors, since imported meshes
inflated MuJoCo's default view extent. Recorded states and physics were unchanged.

## Next evidence to collect

The user has paused live model comparisons and requested cleanup rather than new
capabilities. Current collection scope is five kitchen plus five warehouse runs,
with visual review, synchronized data, Cycles stills and recorded videos. This is
an explicit variation from the brief's ten variants of one scene. Measure a
cold-machine installation separately; keep the writeup/cost table tied to actual runs.
Do not tune prompts or tolerances after seeing held-out results and retain the same
experiment identity. New iterations need new manifests/output folders.
