---
name: scene-pipeline
description: Generate grounded robotics workspaces using this repository's scene pipeline tools, or diagnose a generation report. Use for prompt-to-scene work, not arbitrary MuJoCo code changes.
---

# Scene pipeline

Use `pipeline registry` to discover reusable categories and declared defaults,
not as a closed list of allowed scene content.
The agent chooses intent, inventory and relationships; tools compute dimensions,
geometry, placement and physical checks. Read `docs/running.md` (relative to the
repository root) for commands and `docs/brief-status.md` for current limits.
These are tracked operating references, not private design notes.

Write a SceneProgram JSON or a single expression using `scene`, `space`, `place`
and `relation`. The DSL is interpreted as data; arbitrary Python is rejected.
Use `pipeline generate --program PATH --prompt TEXT --output NEW_DIRECTORY` to
run deterministic tools without starting a nested agent. This works from any
harness with command execution; do not assume Codex is installed. Never edit the
generated MJCF or change a validator to
make a failing scene pass.

For objects needing retrieved geometry, use the exposed tools directly:

```bash
pipeline asset-index                            # once, downloads the catalog
pipeline asset-search 'pallet'
pipeline asset-fetch gazebo:euro_pallet --category pallet
```

Select a search result, inspect the returned package with `pipeline inspect PATH`,
then set the object's `asset_ref` to its returned key in the SceneProgram. Use the
same global `--asset-store PATH` on commands if overriding the default. This is
search → import/validate → insert; no nested agent is needed. The new SDF importer
supports static rigid props only; it does not add joints or certify manipulation.
Existing RoboCasa articulation uses the registry route. For setup and examples,
read the retrieval section of `docs/running.md`.

When a parametric template is appropriate, keep the user's own category name,
pick a family (box, table, shelf, cabinet, drawer) and either quote the user's
stated size verbatim as `dimension_evidence.prompt_quote` with matching axis-labeled
dimensions, or provide a source URL and complete labeled
width/depth/height fields. Automatic sourcing requires an explicitly configured
agent backend; `--program` alone does not start one. Never write a
dimension without evidence. Do not substitute an appliance with an unrelated proxy
without explicit user permission. A retrieved articulated asset is not verified
merely because it loads. Do not supply the harness-owned `dimension_basis` to
assert that evidence is verified. Sized proxies do not establish object function.

For missing visual-only clutter, generate and register a new entry rather than
relabeling a weak retrieval match. With configured fal credentials and an authorized
spend budget:

```bash
pipeline asset-generate --category tea_towel --prompt 'a folded cotton tea towel' \
  --size-m .25 --placement support --max-fal-usd .4
```

Inspect the package and use its `asset_ref`, just like a retrieved asset. Search
also lists registered entries for reuse. Alternatively, the SceneProgram object
can carry `generated_request` with `prompt`, `size_m`, `placement` and
`physical_use: "visual_only"`; `generate --clutter fal` submits and registers it.
`size_m` is an explicitly unverified estimate of largest extent, not a manufacturer
measurement. Generated dressing has no contact geometry, joints or support surfaces;
it cannot satisfy functional-equipment requests. Stock G6 decor remains available.
Mark user-required clutter required so generation/placement failures cannot silently
erase it. If retrieval offers no credible match, reject it and revise the search or
report the missing item. A geometry check alone cannot establish semantic identity.

Express the arrangement with relations (against_wall, near, in_row, under); the
solver optimizes them. Mark a relation required only when the user requires it.
On failure, revise only the declarative intent or choose another supported asset.
Do not approve engineering defaults or promote failing assets. G2 authoring is an
explicit offline task; G4 requires corresponding visual/collision part
segmentation and attachment frames. A `partial` result is a loadable scene with
`unmet.json`; it is not a validated scene.

Inspect validation and provenance alongside previews. Kinematic animations are
not robot interactions; MJZ is not independent-format export. Cycles is the sole
offline renderer. Preserve the boundary between ground truth and noisy observations.

Use `pipeline doctor --smoke` to diagnose installation. Use `pipeline inspect PATH`
for a scene/batch gallery, or `--viewer` for the actual MuJoCo scene. A viewer needs
a desktop display; HTML inspection does not. Batch evaluation retains failures and
requires explicit model IDs and spend limits for live calls. Never treat a
fixed-program fixture or recorded-response run as evidence of model generalization.
