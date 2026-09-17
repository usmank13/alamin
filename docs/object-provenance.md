# Object provenance and the richer kitchen

`scene_pipeline.provenance.classify` is a deterministic metadata classifier, not
an LLM guessing an object's origin from its appearance. It checks the resolver
route against the source record and rejects contradictory/unknown combinations.

## Separate origin from suitability

| Origin | Current source | Dimensions | Physical use |
| --- | --- | --- | --- |
| `procedural` | Local G1 template families | Declared engineering design parameters | Simulation candidate; run scene/asset checks |
| `retrieved` | Native RoboCasa / Lightwheel G3 fixtures | Native asset scale, not measured product specifications | Simulation candidate; modified effective-density proxy is disclosed |
| `generated` | Reserved G6/G8 metadata, e.g. a future fal job | Unverified generated geometry | **Visual only**, no contact-rich certification |

`source_classification` includes origin, provider, route, source ID, content key,
dimension basis, physical basis, asset validation state, license information and
allowed use. `contact_rich_certified` is deliberately false: neither retrieval nor
a basic settle/sweep pass certifies a general manipulation task. Scene validation
is separate from asset-cache promotion and remains in `validation.json`.

The original `provenance` record is retained alongside the classification: native
file SHA-256 hashes, attribution, license, scale and the physical-profile changes
are not replaced by a coarse label. The compiler derives the label from the
resolved asset rather than trusting a label supplied by the scene program.

## Where to inspect it

- `assets/KEY/asset.json`: asset source record and classification.
- `ir.json`: classification on each generated object instance.
- `manifest.json`: authoritative classification alongside semantic instance IDs;
  also embedded in the MJZ scene's `semantic_manifest` custom text.
- `provenance.json`: per-instance inventory, source counts and architectural origin.
- `index.html` and `provenance.png`: object table and color-coded cutaway
  (blue procedural, orange retrieved, purple generated).
- Runtime `semantics.snapshot`, HDF5 scene metadata and URDF package sidecar carry
  the same classifications. Old artifacts without classifications remain legacy
  artifacts; they are not silently relabeled.

## Rich scene exercised

The reproducible declarative input is `examples/rich_kitchen_program.py`.
`outputs/rich_kitchen_final` contains a 60 m² kitchen with **44 object instances**:
42 procedural and two retrieved. It includes a real native fridge and countertop
microwave, two prep tables, a serving counter, four base cabinets, two drawer
units, two wall cabinets, two shelves, 12 containers, eight jars, six bottles and
three trays. There are ten articulated object instances and 15 scalar joints.

All small items are placed on support surfaces by the solver; the microwave is
supported at countertop height. Mesh bounds use actual transformed vertices,
avoiding inflated bounds from rotated local boxes. Required wall relations now
filter placement candidates. None of this is a manual edit to generated MJCF.

The scene passed the existing five-second stability and sampled clearance checks.
It has a genuine 1024px Cycles render and an exported URDF package verified in
PyBullet, including all 15 joint endpoint checks and mass/inertia/COM comparisons.
These checks are not a robot-contact proof for every retrieved joint. The known
unstable dishwasher was intentionally not included or mislabeled as verified.

```bash
.venv/bin/pipeline generate --program examples/rich_kitchen_program.py \
  --prompt 'A stocked 60 square metre prep kitchen with retrieved appliances' \
  --seed 17 --output outputs/my_rich_kitchen
.venv/bin/pipeline render outputs/my_rich_kitchen
.venv/bin/pipeline export outputs/my_rich_kitchen --verify
.venv/bin/sim outputs/my_rich_kitchen/scene.mjz --viewer --backend glfw
```

To exercise the unattended agent, omit `--program` and supply the full inventory
description from the example's prompt. The controlled example is authored intent,
not a claim that its inventory was produced by the unattended agent.

The unattended run is `outputs/rich_kitchen_agent_v3`: the same natural-language
description produced **51 instances** (49 procedural, two retrieved) with ten
articulated objects. Counts of small items were unspecified in that prompt, so the
agent selected 12 containers, ten jars, eight bottles and six trays. After the
pipeline fixes below, generation validated on the first attempt in approximately
24 seconds. Its program, trajectory, IR, source inventory and validation report
are retained alongside its gallery and render. The regression suite has 52 tests.

Earlier failed unattended runs were retained, not erased. They exposed hard
relations invented by the agent and a lower-shelf placement intersecting a corner
upright. The agent instruction now distinguishes requested constraints from
optional suggestions and forbids dropping requested inventory during repair.
Usable shelf support bounds exclude corner uprights, and the solver enforces
required wall relations while proposing placements. The corrected generated
files were produced afresh; no failed scene was patched by hand.

This is richer than the initial rooms, not a finished photorealistic kitchen:
procedural objects remain simplified; finishes approximate materials; zone names
are not yet a full workflow-layout solver. No walk-in, sink or dishwasher is
implied by the included cabinets.

## Later: optional fal clutter

No fal request, paid generation or generated-asset importer is implemented here.
The classification contract already distinguishes generated decorative geometry
from simulation candidates. The compiler rejects generated visual-only packages
with contact geoms, joints or a dynamic instance flag; it never silently upgrades
their suitability based on appearance.

A future importer should retain provider/model/version, prompt, request ID,
retrieval timestamp, source mesh hashes, license, scale/orientation transforms and
placement constraints. Start with static dressing outside robot contact paths.
If coarse collision proxies are later needed for navigation, describe their
capability separately; do not equate a proxy with graspable, contact-ready mesh
geometry. Contact use requires a separately reviewed physical representation.
