# Open-vocabulary categories, evidence-bound dimensions and brief-facing gates

Added 2026-09-17. This closes the largest held-out gap: a prompt naming anything
outside the 14 registry categories used to burn the whole repair budget and emit
nothing. The brief scores grey boxes at exact dimensions; it does not score honest
failure. Nothing here lets a number from the agent's memory reach the simulator.

## Unmodeled categories

Resolution order for a category the registry does not know, in `orchestrator.resolve_program`,
before intent is frozen:

1. **User-stated size.** The agent copies the user's words into
   `dimension_evidence.prompt_quote` and states `dimensions_m`. The quote must occur
   verbatim in the prompt and every axis must equal a number-with-unit in that quote
   (or, for a registry category, its declared default). Basis `user_quoted`.
2. **Dimension cache** `src/scene_pipeline/dimensions.json`, keyed by category, holding
   dimensions, source URL, the verbatim quotes, the page hash, access date and the
   template family. Basis `sourced`. A held-out run touches the web once per category.
3. **Web sourcing.** A second bounded `codex --search exec` call returns evidence only:
   a URL, an identity string and, per axis, `[label, number, unit]` exactly as printed.
   The harness fetches the page itself (the Codex sandbox has no network) and
   `evidence.measurement` accepts an axis only if label, number and unit occur together
   as one field and the identity is present. A real quote paired with a number from
   elsewhere on the page is rejected (the P4 counterexample). Verified values are
   written to the cache. Basis `sourced`.
4. **Nothing verifiable** still raises `UNKNOWN_CATEGORY`; see degradation below.

A quote may fix only some axes ("a 1.2 m wide autoclave"). The bound axes keep the
user's values and the remaining axes come from the cache, sourcing the category once
if needed; basis `user_quoted_and_sourced`, with `quoted_axes` and `sourced_axes` in
`dimension_sources.json`. A quote that binds no axis is refused. The first real
unattended run did exactly this on its first attempt, which is why the mixed case exists.

Resolved programs carry `dimensions_m` and `dimension_basis` and no raw evidence, so a
repair that keeps the same category compares equal under the intent freeze and the
agent cannot relax a dimension. The agent also names the closest template family
(`box`, `table`, `shelf`, `cabinet`, `drawer`; default `box`). Box-like articulated
unknowns such as lockers get hinge or slide joints from the existing families at the
sourced size. Registry categories accept a user-quoted override the same way, so
"counter at 900 mm" is representable. Open-vocabulary instances get a stable hashed
class id in 1000..9999 (`registry.class_id`) and provenance `sourced_template` with
`dimension_basis` on the classification.

## Relations and zones are solved, not merely checked

`layout.solve` now ranks candidate poses by a soft cost before taking the first
collision-free one: `near`, `in_row`, `under` and `against_wall` residuals to already
placed partners, weighted double when required, plus distance to a zone anchor. With
two or more zone labels each label owns a seeded wall midpoint; one label means no
anchor and unchanged single-zone layouts. `in_row` also adds candidates flush beside
each placed row member at the panel gap, so runs are exact by construction. Required
relations are still verified afterwards; they now rarely fail, and the agent
instruction no longer forbids requesting them.

## Validator additions (`validation.brief_checks`)

Read off the compiled model, gated in `validation.validate_scene`:

- **Metric bands** (`registry.TOP_BANDS_M`, `MASS_BANDS_KG`, `DOOR_CLEAR_WIDTH_M`): top
  surface height of counters, prep tables, base cabinets and drawer units; per-category
  mass summed over each instance's bodies; door clear width. Bands are wide on purpose.
- **Robot access**: a disc of `POLICY['robot_radius_m']` (0.25 m default, override with
  `--robot-radius`) through the compiled static colliders in the 0.1–1.0 m band must
  connect the entrance to every fixture's approach point in front of it. Not
  manipulation reachability.
- **Pairwise articulation**: neighbours whose swept volumes overlap are opened together
  and checked with the mask-independent geometric overlap.
- **Density**: objects per m², articulated and supported counts. Reported, not gated.

Opening widths are gated in the brief profile only. The sampled-architecture profile
still scales apertures with area and reports them; making its door width an
engineering default with the position still sampled is the pending change there.

The mass band caught a real defect: retrieved RoboCasa fixtures weighed 3712 kg
(fridge) and 328 kg (microwave) in compiled scenes, because the adapter's inertia
group range did not survive the MJZ round trip and the group-1 visual meshes were
weighed at MuJoCo's default 1000 kg/m³. Visual geoms now carry zero density
(`native-mjcf-v8`); the shell proxy yields 93 kg, 12 kg and 41 kg for fridge,
microwave and dishwasher, inside their bands.

## Unattended-run hardening found by running it

- The Codex child now gets a closed stdin. Inherited from a non-terminal parent (cron,
  a background job, a CI runner) Codex prints "Reading additional input from stdin"
  and waits until the harness timeout, leaving no trajectory. Reproduced: 120 s hang
  with an open pipe, 7 s with `/dev/null`.
- The agent-facing schema is passed through `orchestrator.strict`: OpenAI strict
  structured outputs reject `oneOf` and any property missing from `required`, so
  optional fields become nullable and required, and nulls are stripped from the
  agent's output before `validate_program`. The harness-owned `dimension_basis` is
  removed from that copy entirely.

## Degradation and loop economics (`orchestrator.generate`)

- If no attempt validates but some attempt compiled, the best one by passed-check
  count is emitted with `status: partial`, `passed: false`, exit code 1, and
  `unmet.json` listing failed checks, failed sweeps, dropped objects and unsatisfied
  relations. The intent freeze is unchanged; degradation is a logged harness decision.
- Two consecutive identical deterministic failures end the loop instead of paying for
  up to 20 agent calls that cannot change the outcome.

## Verification

`tests/test_open_vocabulary.py`: prompt-quoted grey box with a supported container;
bare numbers, unbound quotes and evidence-free unknowns rejected; web evidence bound,
cached with family and reused without a refetch; the wrong-number counterexample;
sourced cabinet family articulates; `in_row` flush at the 4 mm gap; brief gates pass
on the cafe example and access fails for a 1.5 m disc; partial output with
`unmet.json`; loop stops after two identical failures; partially quoted unknown
categories source the remaining axes; the agent-facing schema is strict-mode
compatible. Full suite: 99 passed.
