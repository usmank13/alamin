# Scene-level grounding and generalization plan

Status: implementation underway, 2026-09-17. The first feasibility slice and its
measured limitations are documented in [scene-grounding-experiments.md](scene-grounding-experiments.md).
Architecture reconstruction passes; the initial empirical sampler does not yet
demonstrate improved held-out kitchen distributions. The phases below are not all complete.
The next milestone, [distribution-conditioned architecture](distribution-conditioned-architecture.md),
is implemented for individual rooms/halls. It improves held-out architectural
statistics but does not complete functional-zone or furnished-layout generation.

## Objective and scope

Generate diverse indoor working spaces from prompts, with semantic intent supplied
by an LLM and spatial realizations grounded in empirical distributions of real
setups. Do not replicate individual floorplans. Do not specialize the engine to
kitchens. Keep the working asset adapters, physics compilation, Cycles rendering,
robot integration and export paths; adapt their input boundary only as needed.

The central architectural commitment is a generator-independent, versioned scene
schema. The compiler must not know whether a scene was produced by an empirical
sampler, a constraint optimizer, an imported layout, or a future learned model.
LLM inspection is not an acceptance gate. Deterministic checks guard the boundary
between semantic proposals and accepted spatial output.

This plan refines the scene-level portions of the original core plan. It does not
claim that their proposed real-layout statistics already exist in the code.

## 1. Contracts and authority boundaries

```text
Prompt -> SceneIntent -> conditioned empirical priors -> layout generator
                                                     -> SceneIR candidate
                                                     -> independent validation
                                                     -> existing downstream tools
```

Define three separate contracts rather than conflating intent, evidence and geometry:

1. **SceneIntent:** activities, functional roles, room/zone relationships, object
   requirements, groups, preferences and explicit user constraints. Open-ended
   semantic labels map to generic spatial capabilities. No LLM-invented metric
   measurements, distribution parameters or finished coordinates.
2. **PriorBundle:** normalized empirical features, conditional distributions,
   source records, sample counts, applicability domain, uncertainty, units,
   extraction version and train/calibration/test split identifiers.
3. **SceneIR:** fully resolved metric architecture, openings, zones, object poses,
   support/group relationships and stable asset references. A common evidence
   manifest records which values were user-specified, asset-derived, sampled,
   optimized or explicitly defaulted. All generators emit this same contract.

User-specified measurements are allowed and labeled as user inputs. Spatial
measurements otherwise come from data or deterministic geometry. Engineering
defaults remain possible but must not be reported as empirical grounding.

Each semantic requirement records its origin: explicit prompt requirement,
agent-inferred intent, or corpus-derived suggestion. The agent cannot silently
reclassify an explicit requirement as optional during repair. For the evaluation
suite, use independently authored expected requirements; do not score success
against only the generating agent's own interpretation of the prompt.

Generic predicates include support, containment, adjacency, alignment, facing,
membership, access and connectivity. Domain labels and role mappings are data,
not per-domain solver branches. Unsupported mappings are surfaced, not converted
into a convenient but semantically different catalog item.

## 2. Real-data grounding

First inventory usable real-layout data. Dataset selection is an implementation
checkpoint, not settled by this plan. Audit licenses, access, metric scale,
annotations, domain coverage and source biases before acquiring a large corpus.

Separate evidence types:

- Architectural plans support room shape, aspect, topology, openings and circulation.
- Furnished plans or annotated 3D scenes support occupancy, counts, placements,
  co-occurrence, orientation, spacing and functional group structure.
- Unannotated images do not provide verified metric placement distributions.

An architectural-only source must not be credited with furnishing evidence.
Unknown scale must remain unknown; use dimensionless features where legitimate,
but do not infer metres from pixels without evidence. Automated label extraction
is permissible only with explicit uncertainty and an audited evaluation sample.

Normalize sources into a reference schema with room polygons, opening locations,
object footprints and labels where present, relationship annotations where
supported, and per-field evidence. Retain missingness instead of fabricating labels.
Deduplicate and split by original building/scene/source family before fitting, so
variants of one layout cannot leak into both training and evaluation.

Initial empirical features:

- Room area/aspect, concavity, room adjacency and opening positions.
- Object count and footprint coverage conditioned on usable area and activity.
- Wall attachment, nearest-neighbor distances and relative orientations.
- Support occupancy, cluster sizes and group-relative arrangements.
- Free-space connectivity, bottleneck widths and access-region obstruction.

Preserve dependencies: independently matching marginal histograms can still
produce incoherent scenes. Start with interpretable conditional distributions
and joint samples of local arrangements; test conditional neighbor sampling
against a constrained empirical sampler. No model training is required initially.
Continuous reference adaptation is optional, not the required generation method.

## 3. Generalization and fallback policy

Condition priors on spatially meaningful roles and capabilities as well as domain
labels: storage, work surfaces, circulation, seated work, service and supported
clutter. These are an initial extensible vocabulary, not a complete ontology.

Use hierarchical backoff:

1. Compatible domain/activity and room-scale evidence.
2. Shared functional-group or relation evidence across domains.
3. Generic geometric priors with clearly reduced empirical coverage.
4. Explicit unsupported result when required content cannot be grounded safely.

Record the backoff path and effective source count. Do not call a novel domain
empirically validated merely because its geometry compiles. Compositional
generalization is plausible; universal domain coverage is not an acceptance claim.

Unusual explicit requests may legitimately differ from the reference population.
Keep physical feasibility hard, report distributional novelty separately, and
never erase the user's unusual requirement to improve a typicality score.

## 4. Interchangeable layout generators

Define a backend interface accepting SceneIntent, PriorBundle, resolved asset
bounds/affordances, robot footprint, seed and budget. Return either a SceneIR
candidate with evidence or a structured failure identifying conflicting constraints.

Keep the current heuristic as a labeled ungrounded baseline. Implement an initial
empirical-proposal plus constrained-optimization backend:

1. Sample architecture/topology conditioned on area and semantic requirements.
2. Allocate functional zones and generic object groups.
3. Sample joint local arrangements and role-relative placements from real data.
4. Optimize group/object transforms against empirical costs and intent constraints.
5. Place supported objects using empirical density and support relationships.

Sampled statistical targets are soft; explicit user requirements, geometry,
support validity and required access remain hard. Asset dimensions are not
distorted to make placement succeed. The existing resolver supplies asset geometry;
expanding its vocabulary is not part of this workstream.

Generic group constraints should support aligned modules, shared work surfaces
and storage/workstation groupings. This phase adds scene composition semantics,
not a new family of kitchen-only object generators. If an existing asset cannot
represent a required component, report the asset coverage limitation separately.

## 5. Independent deterministic validation

Validation reads the emitted SceneIR and actual resolved geometry, recomputes
features, and checks them against frozen evidence and constraints. It must not
trust the generator's reported success flags or its claimed feature values.

### Per-scene hard checks

- Schema integrity, units, references and evidence-chain integrity.
- Explicit counts and executable relation/zone/group requirements.
- Room containment, collision, support contact and articulation clearance.
- Connectivity and required approach access using the selected robot footprint.
- Existing compilation and physics checks, retained as separate reports.

### Empirical plausibility checks

Score conditional density, adjacency, spacing, orientation and group structure
against a held-out calibration partition. Report sample support and out-of-domain
status. Calibrate thresholds to held-out real scenes and deliberately corrupted
scenes; do not invent a universal numeric pass target.

A single scene can be scored for plausibility, but cannot establish distribution
matching. Across batches, compare joint/conditional features, diversity, coverage
and near-duplicate rates. Include real-to-real split variation as a baseline.
Reserve test data for final evaluation rather than tuning costs against it.

These checks constrain hallucination, but do not prove every open-ended semantic
claim. An unknown activity or unverifiable label is reported as unverified.
Neither an agent's approval nor a good histogram match is proof that a room
fulfills its intended function. Human inspection is useful during development,
not required in the unattended runtime acceptance path.

### Adversarial tests

Inject unsupported source IDs, invented metric values, shuffled categories,
incorrect support parents, missing required objects, inaccessible zones,
blocked articulations, duplicate layouts and marginally plausible but jointly
incoherent arrangements. Confirm the corresponding checks detect each corruption.
Also retain valid unusual scenes to measure over-rejection.

## 6. Bounded repair and downstream compatibility

Use typed failures to select resampling, local optimization, another compatible
prior, or an explicit unsupported result. Let the agent clarify inferred semantic
choices, not alter evidence, tolerances, fixed measurements or user requirements.
Freeze explicit requirements at the run boundary and log every revision.

All backends pass the same validator and emit the same downstream schema. Preserve
stable IDs, support relationships and provenance through MJCF, Cycles, robot
insertion and existing export/data paths. Render diagnostic top-down overlays,
zone/access maps and Cycles views for development, without using agent image
inspection as a correctness gate.

## 7. Delivery sequence and exit criteria

| Phase | Work | Exit evidence |
| --- | --- | --- |
| A: Contracts and baseline | Version contracts; isolate generator interface; capture current baseline; define requirement provenance | Existing examples compile/export unchanged through a compatibility adapter; two producers can emit the same IR |
| B: Corpus feasibility | Audit candidate data; normalize a small real corpus; split sources; extract features | Reproducible feature report with metric/annotation coverage, licenses, missingness and explicit domain gaps |
| C: Minimal grounded generator | Fit simple conditional/joint priors; sample and optimize using current assets | Unattended prompt-to-IR runs with source-linked spatial decisions, no hand-edited layouts, visual overlays |
| D: Independent gates | Geometry/intent checks, calibrated plausibility, adversarial suite, typed repair | Known corruptions caught; false-rejection results on held-out real/valid unusual scenes; honest unsupported cases |
| E: Generalization study | Compare baseline and grounded variants across seeds and held-out semantic compositions | Joint-distribution, diversity, fidelity and grounding-coverage reports; no per-prompt solver code |
| F: End-to-end integration | Run Cycles, robots and export verification on accepted scenes; update operating docs | Same artifact contract and downstream behavior across generator backends; reproducible commands and cost/timing reports |

Do not scale corpus acquisition or commit to a complex optimizer until Phase B
demonstrates useful annotations and domain support. Corpus ingestion and contract
work can proceed in parallel; empirical validation depends on the data audit.

Evaluation should span at least two materially different activity types using
supported assets, plus a deliberately unsupported/out-of-domain case. Hold out
semantic combinations and source scenes, not just random seeds. Use a leave-domain-out
test only if the corpus actually supports multiple domains; otherwise explicitly
limit the generalization claim.

Compare: current heuristic; independent-marginal priors; conditional/joint priors;
and joint priors with constraint optimization. This distinguishes improvements
from empirical evidence versus improvements from geometry checking alone.

## Deferred work

Asset-library expansion, new articulation routes, diffusion rendering, learned
layout models and broad raster-floorplan reconstruction are outside this focused
iteration. Cycles stays in place. Mapping remains a reported diagnostic without
the previously invented 0.8 IoU acceptance requirement.

The first milestone is not another decorated kitchen. It is a small reproducible
experiment demonstrating that the same semantic-to-schema flow produces varied,
physically coherent scenes whose measurable spatial structure is better grounded
in held-out real setups than the current heuristic.
