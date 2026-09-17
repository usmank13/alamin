# LLM + Resolver Design Options and Tie-Break Prototypes

2026-09-16 · @u_LYweu6308DfErXx86q_U5w

## Decision record after the prototypes

The implementation uses reusable G1 families and source-aware native G3 adapters
with uniform scale. Preserve native collision/articulation when usable; do not
re-decompose already suitable native colliders unconditionally. G2 is supervised
offline library growth, not unrestricted code execution inside scene generation.
G4 requires actual segmented affordance parts and attachment frames; a jointed
hierarchy alone does not make a baked visual handle separately resizable.
P4's sourced examples do not establish an open-vocabulary dimensional guarantee,
and the small real-asset P5 probes do not establish the proposed 15-object pass
rate. Engineering defaults and modified physical profiles must remain explicit.
See [implementation and release gates](pipeline-implementation.md) for the selected
architecture, current evidence, and unresolved acceptance work.

## Scope

Decided: the pipeline is an agent skill for a coding agent (Claude Code or Codex), and the agent takes the place of the human modeler. It drives deterministic tools for layout, geometry, physics and validation; the open decisions are now about how objects are parameterized into geometry and physics (D7), not about where the LLM sits.

| Decision | Question | Status |
| --- | --- | --- |
| D1 | What does the agent emit as the scene program? | Decided: constraint DSL (C) |
| D2 | Where does the LLM sit? | Decided: agent operating the pipeline as a skill (C) |
| D3 | How is an open category name matched to the registry? | Folded into D2: agent judgment with a registry search tool |
| D4 | Where do dimensions come from? | Decided: agent extracts from fetched sources with a verbatim check (B); trust policy deferred |
| D5 | Which parameterization template does a new category get? | Decided: agent chooses, given template descriptions and a retrieval tool |
| D6 | When does a new asset enter the library? | Decided: verify-then-cache (A) |
| D7 | How are objects parameterized into geometry and physics? | Open: two prototypes |

The fixed constraint stands: no number produced from the model's memory reaches the simulator. The agent reads dimensions from sources it fetches, and geometry is produced by tools from those dimensions.

## D1: what the LLM emits

Decided: the agent writes the scene program in a constraint DSL (option C). The DSL carries categories, counts, zones, relations and grouping rules, and the solver treats it as the full specification of soft constraints on top of the fixed hard constraints from the standards table.

| Option | LLM output | Why not / why |
| --- | --- | --- |
| A. Flat inventory | Categories + counts + zones | Loses prompt intent |
| B. Inventory + relations | A plus a fixed relation vocabulary | Vocabulary proved too coarse for grouping ("three prep tables in a row, one under the pass window") |
| **C. Constraint DSL** | Infinigen-style constraints: counts, relations, groups, per-zone density | Chosen. Expressive, composable, and the agent can iterate on it when the solver reports unsatisfiable constraints |
| D. Direct layout | Coordinates | Violates the fixed constraint |

Design notes for C: the DSL is a small Python-embedded language (functions returning constraint objects), so the agent writes ordinary code and the solver can report which constraint failed by name. Every constraint is soft except the ones the standards table marks hard. An unsatisfiable program returns the violating constraint set to the agent, which revises the program rather than the solver relaxing dimensions.

## D2 + D3: the agent as modeler

Decided: a coding agent (Claude Code or Codex) runs the pipeline as a skill and takes the role a human modeler would have. It writes the scene program, resolves categories, sources dimensions, picks parameterizations and reacts to validator reports; every geometric and physical computation is a tool it calls.

| Option | Shape | Why not / why |
| --- | --- | --- |
| A. Single call | One structured call, then code | Cannot recover from a resolver miss |
| B. Staged calls | Program call plus one call per unresolved category | Bounded, but every recovery path has to be anticipated in code |
| **C. Agent operating the skill** | Agent with tools for solver, resolver, registry, fetch, compiler, validator; iterates on failures | Chosen. Recovery paths are the agent's judgment, which is the point of replacing the modeler |
| D. LLM as solver critic | Solver proposes, LLM judges | LLM judging geometry is the weakness we design around |

**Tool surface the skill exposes.** `registry.search(name)` returns candidate categories with aliases and templates; `dims.lookup(category)` and `dims.fetch(query)` return dimensions with source and quote; `asset.build(template, params)` produces a physics-valid asset; `layout.solve(program)` returns a layout or the failing constraint set; `scene.compile(ir)` and `scene.validate(path)` return the MJCF and a report. The agent never computes geometry itself.

**Category matching (D3) is folded in.** The agent calls `registry.search` and decides whether a hit is the same thing; a synonym like "lowboy" in a kitchen context is the kind of judgment the agent is there for. No embedding threshold, no separate matcher.

**Reproducibility note.** Tools are seeded and deterministic; the agent's trajectory is not. The held-out run is repeatable at the tool level (cached fetches, cached programs by prompt hash) even if the agent would take a different path on a rerun. Harness-level guarantees (max iterations, transcript caching, a fallback path when the agent stalls) are deferred until they become a problem.

## D3: category matching

Folded into D2. The agent matches open names to registry entries using `registry.search` and its own judgment; the earlier options (closed vocabulary, embedding threshold, LLM matcher, hybrid) were ways to do this without an agent. If false matches show up in practice, the first fix is richer alias lists in the registry, not a separate matcher.

## D4: dimension sourcing and trust policy

Decided: option B. The agent fetches spec sheets, catalogs and dataset entries, extracts (w, d, h, unit, quote, URL), and the `dims.fetch` tool accepts the result only if the quote appears verbatim in the fetched page. Resolution order is unchanged: offline table, ShapeNetSem statistics, web, drop.

| Option | Method | Why not / why |
| --- | --- | --- |
| A. Structured-page scrape | Per-site parsers | Brittle; per-site code |
| **B. Agent extraction with verbatim check** | Agent reads pages; tool verifies the quote | Chosen. Site-agnostic; the verbatim check blocks numbers from memory |
| C. Dataset-only | No web | Kept as an offline flag for runs without network access |

Trust policy (two-source agreement, distribution fitting) is deferred. It is an agent-harness concern, and the agent can be asked to cross-check when a value looks off. Every accepted dimension is written back to the offline table with its source, so the table grows and the web is touched once per category.

## D5: parameterization assignment

Decided: the agent assigns the template. It calls `registry.search` and a retrieval tool over PartNet-Mobility and ShapeNetSem, reads the template descriptions from D7, and picks: retrieval if a rigged or sized match exists, a procedural template if the object is box-like or otherwise fits one, drop if neither. The `asset.build` tool rejects parameter sets outside a template's declared domain, which is the guard against forcing a valve into the cabinet template.

| Option | Method | Why not / why |
| --- | --- | --- |
| **A. Agent picks** | Agent chooses from template descriptions plus retrieval results | Chosen; same judgment the modeler would apply |
| B. Retrieval-first, box fallback | Code-only rule | Subsumed: it is the default heuristic the agent is told to follow |
| C. Shape-prior test | Image fill ratio decides box vs not | Available as a tool if template errors show up |
| D. Curated map | Fixed table | Blocks library growth |

## D6: library growth and verification gate

Pick now: a new asset enters the cache only after passing the object-level validator, and a scene may use an unverified asset only in the same run that produced it. The alternatives either slow every run or let bad assets persist.

| Option | Rule | Trade-off |
| --- | --- | --- |
| A. Verify-then-cache | Asset is validated in isolation (compile, settle, joint sweep) before entering the cache; the current scene uses it immediately | Cache never holds a broken asset; one extra validation per new category |
| B. Cache-then-verify | Asset enters the cache; scene validator failures mark it bad | Faster first run; a bad asset can poison later scenes until it fails |
| C. Quarantine tier | New assets live in a per-run cache; promotion to the shared library requires passing in 3 scenes | Strongest guarantee |

Verdict: **pick A**. Object-level validation costs a few seconds of simulation and removes the whole class of "a cached asset broke a later scene" bugs. Promotion counts (option C) can be added as metadata without changing the rule.

## D7: parameterizing objects into geometry and physics

Open. This is where the metric guarantee is actually manufactured, and the literature offers more routes than we should ship at once. Two prototypes decide between the candidates below; everything else is either picked now or deferred.

**Dimension parameterization (how sizes are sampled).** Pick now: fit a Gaussian or KDE on log-dimensions per category from ShapeNetSem, ABO and 3D-FUTURE instances, sample from it, clamp to the standards table. This keeps width, depth and height correlated. PCA on the log-dimension matrix is the same thing with fewer parameters and is worth doing for categories with 20+ instances; with fewer, the full covariance is fine.

**Geometry parameterization (how a sized object becomes a mesh with joints).**

| Route | Method | Articulation | Metric control | Cost to build | Status |
| --- | --- | --- | --- | --- | --- |
| G1. Fixed procedural templates | Hand-written Python generators: box-composition with a split grammar along width, a shelving template, a table template, each with categorical parameters | Yes, derived from module geometry | Exact | Days per template family | Candidate for P4 |
| G2. Agent-authored parametric generators | Agent writes a CadQuery / build123d / trimesh generator per new category from a skeleton, runs it in a sandbox, keeps it only if it passes the object validator (Infinigen's generator-as-code, authored on demand) | Yes if the skeleton enforces a joint spec | Exact when it works | Minutes per category, failure-prone | Candidate for P4 |
| G3. Retrieval + uniform bbox rescale | PartNet-Mobility or ShapeNetSem mesh scaled to sampled dimensions | Yes for PartNet-Mobility | Exact bbox; internal proportions distort | Hours | Candidate for P5 |
| G4. Retrieval + part-aware rescale | Scale the carcass to target, keep handles, hinges and knobs at real size (GAPartNet-style labels or PartNet-Mobility's part hierarchy) | Yes | Exact bbox and part sizes | Days | Candidate for P5 |
| G5. Inverse procedural fit | Find reference images or meshes, estimate template parameters from it: bounding box for scale, VLM for door and drawer counts (CropCraft's IPM idea) | Yes, via the template | Exact | Days | Deferred; bonus route |
| G6. Generated mesh + rescale | fal Tripo / Hunyuan3D output scaled to sampled dims, CoACD | No | bbox only | Hours | Pick now, non-articulated dressing only |
| G7. Learned articulated generation | CAGE, SINGAPO, NAP, URDFormer | Yes | Weak | GPU, training | Out of scope |
| G8. Generated mesh + healing | fal mesh, then a repair pipeline: canonical orientation (up and front axes from a VLM or symmetry), watertight repair and floater removal, anisotropic rescale to sampled (w, d, h) rather than one scale factor, part segmentation (SAMPart3D / PartSLIP) with joints authored on the segmented parts (Articulate-Anything style), CoACD per part | Yes when segmentation and joint authoring succeed; else falls back to G6 | Exact bbox; part sizes only as good as segmentation | Days; articulation half is failure-prone | Candidate for P6 if retrieval coverage proves thin |

**Collision geometry.** Pick now: primitives (box, cylinder) emitted directly by G1/G2 templates; CoACD for every retrieved or generated mesh, capped at 16 hulls per part; V-HACD only if CoACD runtime becomes the bottleneck at volume. A primitive-fitting fallback (fit boxes and cylinders to a mesh's parts) is deferred.

**Mass and inertia.** Pick now: density per material class from the registry, mass = density × collision volume per part, inertia from collision geoms. Hollow objects (cabinets, fridges) use a shell density so a 600 mm base cabinet lands near 25–40 kg rather than the 200 kg a solid box gives.

The two live questions: (1) hand-written templates (G1) or agent-authored generators (G2), and (2) is uniform rescale (G3) good enough for retrieved articulated objects, or does handle-size distortion break grasping and clearance (G4)? Prototypes P4 and P5 settle both. A third, G8, sits between G5 and G6: it keeps the generated mesh's appearance and heals its scale and structure instead of fitting a template to it. Its non-articulated half (orientation, repair, anisotropic rescale) is a cheap upgrade to G6 and can be picked now; its articulated half is the spec's bonus and becomes P6 only if P5 shows PartNet-Mobility coverage is too thin for the categories prompts actually produce.

## Tie-break prototypes

Two prototypes remain, both on D7. P1–P3 from the earlier draft are retired: D1 is decided (C), and D3 and D5 are the agent's judgment rather than a mechanism to benchmark.

| Prototype | Decides | Build | Test set | Measure | Decision rule |
| --- | --- | --- | --- | --- | --- |
| P4 | G1 fixed templates vs G2 agent-authored generators | G1: box-composition template with split grammar, \~300 lines. G2: a generator skeleton (function signature, joint-spec return type, sandbox runner) and a prompt asking the agent to author a generator per category | 12 articulated categories: 6 box-like (base cabinet, drawer unit, reach-in fridge, walk-in door, wall cabinet, dishwasher) and 6 not (swing-arm faucet, gate valve, lever handle, tilting kettle, hinged lid bin, rolling rack) | (a) object-validator pass rate on first attempt; (b) pass rate after up to 3 agent revisions; (c) dimensional error of the produced asset vs requested (w, d, h); (d) wall-clock and tokens per category | Ship G1 for box-like categories regardless. Adopt G2 for the rest only if (b) ≥ 80% and (c) ≤ 1%; otherwise non-box articulated objects go to retrieval (P5) and G2 stays an offline library-building mode |
| P5 | G3 uniform rescale vs G4 part-aware rescale for retrieved articulated objects | G3: bbox scale + CoACD + URDF→MJCF. G4: same, plus per-part scale factors that hold handle, knob and hinge parts at their source size and re-solve joint origins | 15 PartNet-Mobility objects across cabinets, fridges, microwaves, faucets, valves, each rescaled to 3 target sizes at 0.7×, 1.0×, 1.4× of source | (a) validator pass rate (settle, sweep, no interpenetration); (b) handle and knob dimensions after rescale vs real-world range from the dimension table; (c) whether a Franka gripper (80 mm max aperture) can close on the handle at all three scales | Ship G3 if (a) ≥ 90% and (c) holds at all scales; otherwise G4 for categories with graspable parts and G3 for the rest |

Order: P5 first, because its URDF→MJCF and CoACD path is needed regardless of P4's result and surfaces the known MuJoCo import issues (dropped inertials, mesh paths, spinning bodies) early. P4 second, and its G1 half is production code either way.

Both prototypes run headless against the object validator only; no scene, no solver, no robot.

## Defaults if no prototype is run

If building starts before P4 and P5 finish, these are the safe ends of each open trade-off.

| Decision | Default | Why it is the safe end |
| --- | --- | --- |
| D7 geometry, box-like categories | G1 fixed templates | Exact dimensions, joints by construction, no sandbox |
| D7 geometry, non-box articulated | G3 retrieval + uniform rescale | Rigged assets exist; distortion is bounded at scales near 1.0× |
| D7 geometry, dressing | G6 generated mesh + rescale | Non-articulated, so only the bbox matters |
| D7 dimensions | Log-dimension Gaussian per category, clamped to standards | Cheap; keeps proportions plausible |
| D7 collision | Primitives for templates, CoACD for meshes | Convex by construction; MuJoCo compiles |
| D7 mass | Density × collision volume with shell density for hollow classes | Plausible without per-object tuning |
| D2 harness | Max 20 agent iterations per scene, then fail the run with the last validator report | Bounded held-out runtime |
