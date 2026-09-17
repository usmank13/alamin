# Minimal P4/P5 experiments

## Representative follow-up (start here)

The new [representative tests](REPRESENTATIVE.md) replace synthetic non-box
envelopes with manufacturer-grounded P4 proxies and use original RoboCasa
microwave, dishwasher, and refrigerator fixtures for P5. Open
`outputs/prototypes/representative/index.html` for every test's closed/open/collision
views, animation, evidence, and portable MJZ. The older experiments below remain
available as historical diagnostics; their failure rates do not select G3/G4.

These are independent engineering pilots for the two pipeline design documents,
not the scene generator, production asset library, or the proposed full benchmark.
No new LLM service is used: Codex authored the two G2 candidates in this session.
The core `sim_harness` remains unchanged.

## Reproduce

From the repository root on Linux:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --extra dev --extra prototypes
python prototypes/fetch_manipgen.py --download
.venv/bin/python -m prototypes.run_pilots
.venv/bin/pytest -q
```

The downloader reuses the verified archive if already present. It is 552 MB; the
selected extracted geometry is only about 507 KB. Provenance, hashes and selected
paths are in [DATASET.md](DATASET.md) and [dataset_sources.json](dataset_sources.json).
The official ManiSkill UCSD asset host timed out; the original full PartNet corpus
requires access approval. This pilot uses a public **ManipGen derivative**: generated
cabinet bodies with PartNet handles. It is not representative of the five P5 categories.

P4 runs under Bubblewrap with network disabled, read-only source files, a writable
output directory, 4 GiB virtual-memory limit, 120 s CPU limit and 180 s wall limit.
Requires the system `bwrap` and `timeout` commands. Nested sandboxes may require
running this command outside the outer sandbox. There is no silent unsandboxed fallback.
Only reviewed local prototype code is executed; the downloaded dataset contains
geometry/XML, and no downloaded code or pickle files are run.

Outputs go to ignored `outputs/prototypes/`:

- `SUMMARY.md`: measured results and collision-filter counterexample.
- `p4/results.json`: candidate outcomes, dimensional errors, validation timing,
  input provenance and generator hash. Each emitted scene has `scene.xml`, meshes,
  and successful cases also have an articulated `preview.png`.
- `p5/results.json`: three handles × three scales × two routes, using source colliders.
- `p5_visual/results.json`: same experiment with explicit collider reconstruction
  from source visual meshes, using CoACD.
- `microchecks.json`: dimension-extraction and 2D/3D counterexamples.

Inspect any result with the existing harness:

```bash
.venv/bin/sim outputs/prototypes/p4/base_cabinet_G1_r0/scene.xml --viewer --backend glfw
```

## P4: what this tests

G1 covers a single hinged cabinet and a single drawer tray. It rejects the two
non-box category requests rather than forcing them into a cabinet. This is a small
box-composition seed, **not** the planned full split grammar. G2 has two simple
agent-authored generators: a hinged-lid bin and a simple swiveling faucet proxy.

The cabinet envelope/thickness come from the nominal dimensions on the
[IKEA product page](https://www.ikea.com/us/en/p/sektion-base-cabinet-white-30265386/),
with conversions computed from [fixtures/dimensions.json](fixtures/dimensions.json).
The output is not an IKEA product replica. Non-box cases deliberately use a
**synthetic one-metre cube** to stress the parameterization; these are not realistic
faucet/bin dimensions and must not enter the dimension registry.

The lid has a controlled bad/good design pair: positioning the lid above the
requested envelope makes it 5% too tall; moving its pivot to the correct height
removes that error. These are **not measured first-attempt/three-revision agent
trials**, so no G2 success-rate estimate or adoption threshold is claimed.
Per-category authoring tokens and wall time were not available and are null;
build/validation wall times are measured, not substituted for authoring cost.

Initial direction: keep G1 for supported box-like fixtures. G2 can create simple
mechanisms through this interface, but keep it an explicitly validated experimental
or offline route until tested on sourced real envelopes and more categories.

## P5: what this tests

Three cabinet doors with horizontal, vertical and round handle geometry are tested
at 0.7×, 1× and 1.4×. `p5.py` implements a small explicit URDF translator (mesh/box,
fixed/revolute/prismatic), transforms source geometry correctly, scales slide
limits/joint origins, and rebuilds mass/inertia from collision geometry. It also
records a native MuJoCo URDF-import probe for comparison.

- G3 scales all geometry and joint origins uniformly.
- G4 holds the handle geometry at source size, scales its tangential position,
  and preserves its offset relative to the door's +X mounting plane. This uses
  the corpus's separate `handle_link` and zero-frame joint convention. It is
  **not** a general part-labeling or attachment solver; rotated/offset mount
  frames are rejected by this prototype rather than silently mishandled.
- CoACD runs with seed 0, at most 16 hulls per input mesh, and reduced pilot search
  settings. Already convex inputs take an exact hull shortcut. Decomposition is
  cached by mesh bytes/settings version. A hard hull cap does not guarantee that
  concavities and cavities survive; visual-collider rebuilding tests that directly.
- Handle extents and a rough 80 mm aperture heuristic are reported. No actual
  Franka closure, approach clearance or grasp was tested. It is **not** P5(c).
- Source sizes follow the URDF's metre convention, without independent product
  measurement. There is no real-world handle-size distribution check yet.

Initial direction: these samples do not select G3 versus G4. Source collider
problems dominate even at 1×. All sampled handles remain below the chosen aperture
heuristic at all scales, so this pilot gives no observed grasp-driven reason to
pay for G4. Keep G3 as the simpler provisional path, validate the source at 1×
first, and retain G4 as a targeted option where an actual handle/attachment test
fails. Do not interpret the collision failure rates as evidence against rescaling.

## Shared validator: precise scope

The pilot checks compilation, five simulated seconds of passive settling,
maximum sampled COM/joint drift, 2 mm penetration tolerance, 21 kinematic poses
per joint, and dynamic reach to each endpoint using idealized gravity-compensated
servo forces. Hinge tolerance is 0.5 degrees; slide tolerance is 5 mm. Endpoint
servo damping is integrated implicitly to avoid numerical instability on small
inertias. Simulation warnings and reset clocks fail the check.

Limitations matter:

- Root is fixed; no floor/support stability, tipping, interaction policy, multi-joint
  swept-volume proof, or calibrated actuator effort limits are tested.
- Template collision shapes are convex box/cylinder meshes, not a production
  primitive emitter. Mass is collision-volume × **experimental density 500 kg/m³**.
  This is a sensitivity input, not sourced material density. Overlapping hulls can
  overcount volume. Category mass plausibility is not certified.
- Parent-child filtering is disabled for the geometric check. Blanket exclusions
  can hide collisions; the final report shows a concrete source-collider example.
- A candidate can be mechanically valid and dimensionally wrong; the P4 gate
  explicitly combines physics validation with bbox error ≤1%.
- Negative tests verify that actual overlap fails the validator and invalid
  categories/source structure are rejected. Fixtures do not get promoted to a cache.

## Other small findings

1. **Verbatim quote membership is not a value validator.** The counterexample
   supplies a genuine quote with fabricated extracted numbers. A narrow deterministic
   parser binds values and units to text; a general source/axis parser remains work.
2. **2D full-sweep bounds are conservative.** Vertically separated objects can
   overlap in projection while being collision-free in 3D. A 3D narrow-phase check
   can recover valid stacked layouts; this is not evidence that full projected
   sweep bounds miss 3D intersections. Closed footprints alone are insufficient.
3. **Blanket contact exclusions weaken the validator.** Keep an unfiltered geometric
   overlap/sweep test separate from simulation contact masks.
4. **Source URDFs need a structural gate.** The additional extracted drawers have
   duplicate joint names and missing links. They are preserved unchanged and the
   importer rejects them. Native MuJoCo also fuses fixed bodies by default, which
   can discard per-part identity needed for semantics and protected-part scaling.

The next useful experiment is a few clean, independently dimensioned articulated
assets with an actual gripper approach/closure check—not a larger count of these
same confounded samples.
