---
name: scene-pipeline
description: Generate grounded robotics workspaces using this repository's scene pipeline tools, or diagnose a generation report. Use for prompt-to-scene work, not arbitrary MuJoCo code changes.
---

# Scene pipeline

Use `pipeline registry` to discover supported categories and declared defaults.
The agent chooses intent, inventory and relationships; tools compute dimensions,
geometry, placement and physical checks. See `docs/pipeline-implementation.md`
for command examples and current acceptance status.

Write a SceneProgram JSON or a single expression using `scene`, `space`, `place`
and `relation`. The DSL is interpreted as data; arbitrary Python is rejected.
Use `pipeline generate --program PATH --prompt TEXT --output NEW_DIRECTORY` to
run deterministic tools. Never edit the generated MJCF or change a validator to
make a failing scene pass.

For a category the registry lacks, keep the user's own name, pick the closest
template family (box, table, shelf, cabinet, drawer) and either quote the user's
stated size verbatim as `dimension_evidence.prompt_quote` or leave dimensions out
so the harness sources and verifies them from a fetched page. Never write a
dimension without evidence. Do not substitute an appliance with an unrelated proxy
without explicit user permission. A retrieved articulated asset is not verified
merely because it loads. See `docs/open-vocabulary-and-gates.md`. Decor categories
(route G6: mug, kettle, potted_plant, ...) are fal-generated, visual-only dressing
without contact geometry; they appear in the registry only under `--clutter fal`, are
dropped with a reason otherwise, so request them with `required=False`.

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
