# Initial P4/P5 findings — 2026-09-16

Implementation and commands: [prototypes/README.md](../prototypes/README.md).
Raw measurements: `outputs/prototypes/SUMMARY.md` and per-pilot `results.json`.
These are minimal engineering pilots, not the full adoption benchmarks in the design doc.

## P4: interface works; coverage/reliability still unmeasured

- A cabinet and drawer from the G1 box family passed compilation, 5 s settling,
  sampled articulation sweep, idealized servo endpoint tests and ≤1% dimension error.
- Two Codex-authored G2 geometry proxies (swivel arm and hinged-lid bin) passed
  those checks. The lid's controlled bad/good variants demonstrate that mechanical
  validity alone does not catch a 5% height error.
- Box envelopes use the cited nominal cabinet dimensions. Non-box unit-cube
  envelopes and density are explicitly synthetic test inputs, not real asset data.
- No independent authoring trials or per-category token/time telemetry were measured.
  The variants do not constitute a measured agent-repair success rate.

Direction: proceed with G1 for box-like objects; keep G2 as a validated experimental
extension. These tests establish feasibility of the interface, not broad category quality.

## P5: source validity dominates this pilot

The official UCSD download host timed out. The dataset agent prepared a pinned
public ManipGen derivative subset: generated cabinets with PartNet handles.
Three door assets × 0.7/1.0/1.4 scales × G3/G4 yielded 18 cases. Rebuilding colliders
from visual meshes with CoACD yielded a second 18-case diagnostic run.

- All cases compile. Uniform bbox scaling matches requested extents to floating-point
  precision (relative error below 3×10⁻¹⁶).
- **0/18 strict physics passes** for source colliders, and **0/18** for the quick
  visual-mesh CoACD rebuild. Source 1× rest overlaps are 14.78, 14.53 and 9.86 mm;
  the rebuild's overlaps are 10.45, 18.34 and 16.63 mm. These precede the scaling decision.
- G4 preserves handle sizes and their offset relative to the mount plane in this
  corpus. G3's handle extents all remain below the rough 80 mm aperture heuristic;
  actual Panda approach/closure and real-world size distributions were not tested.
- Three additional extracted drawer URDFs have duplicate joint names and missing
  link references. The importer rejects them without modifying source files.

Direction: retain G3 provisionally for its simplicity, but gate source assets at
1× before evaluating scaling. This corpus does **not** settle G3 versus G4. The
next discriminating test needs a clean articulated source and actual grasp geometry.

## Changes suggested to the core plan

1. **Do not use filtered contacts as the sole geometric validator.** In one source
   asset, parent/child exclusion changes observed overlap from 14.53 mm to 0.34 mm,
   crossing the planned 2 mm pass threshold without repairing geometry.
2. **Bind extracted numbers, units and axis labels to evidence.** Quote membership
   alone accepts a real quote paired with fabricated dimensions. The pilot's narrow
   parser rejects unit mismatch; general extraction still needs this guard.
3. **Use 3D checks to recover safe vertical stacking.** Complete projected sweep
   bounds are conservative: overlapping 2D bounds can correspond to separated 3D
   objects. Closed footprints alone remain insufficient for articulation clearance.
4. **Keep source structural validation and provenance ahead of physics conversion.**
   Compile success is insufficient; default native URDF import can fuse fixed parts,
   and inferred inertias/contact geometry do not establish physical accuracy.

Verification: 14 regression tests passed, including source transforms, scale/joint
math, domain rejection, overlap detection, quote binding, and the existing harness.
P4 executed inside a network-disabled Bubblewrap sandbox. Failures in the P5 result
tables are measured asset-validation outcomes, not failing regression tests.
