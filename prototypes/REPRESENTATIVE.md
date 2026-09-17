# Representative retrieved / grounded follow-up

Small engineering tests, independent of the main scene-generation system. No new
LLM service or full RoboCasa installation. Codex authored the P4 proxy generators;
P5 loads retrieved assets directly, retaining their meshes, textures, joints,
collision masks, and source physics settings.

## Inspect and reproduce

Open `outputs/prototypes/representative/index.html`. Each of the 14 cases has
`comparison.png` (closed, open, collision geometry), `sweep.gif`, `result.json`,
and a portable `scene.mjz` with embedded dependencies and initial state.
Renders are actual MuJoCo output; animations are kinematic, not robot executions.
Scale variants share a camera distance, rather than fitting each variant to the
same apparent size. Collision views preserve the source shapes, not rebuilt hulls.

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --extra dev --extra prototypes
.venv/bin/python prototypes/fetch_robocasa.py
.venv/bin/python -m prototypes.fetch_p4_evidence
bash prototypes/run_p4_grounded.sh
MUJOCO_GL=osmesa LP_NUM_THREADS=1 .venv/bin/python -m prototypes.p5_native
.venv/bin/pytest -q
.venv/bin/sim outputs/prototypes/representative/p5_Refrigerator055_G3_1/scene.mjz --viewer --backend glfw
```

Fetches need network; all subsequent tests are offline. P4 uses Bubblewrap,
read-only source mounts, no network, a 4 GiB virtual-memory cap, 180 s CPU / 240 s
wall limits, and one software-renderer thread. Nested sandboxes may need an outer
permission grant. No silent unsandboxed fallback. System tools: `bwrap`, `timeout`,
and Poppler (`pdftotext`, `pdftoppm`) if the publisher PDF download succeeds.

## P4: what is actually grounded

- Rubbermaid 1883566 bin: publisher HTML cached and hashed. Product (not packaging)
  length/width/height fields checked together before conversion: width 12.12 in,
  depth 18.88 in, height 28.50 in. The proxy has tapered walls and an opening lid.
- T&S 133X-L12 swivel gooseneck: primary publisher specification read through web
  retrieval. Printed metric dimensions 145 / 261 / 148 mm drive reach / height /
  outlet-height inputs. Direct PDF retrieval returned anti-bot HTML; this failure
  is preserved in the cache manifest rather than falsely labeling it a PDF.
  Drawing interpretation is provisional; no pixel-level drawing verification.

See `fixtures/grounded_products.json` for URLs, short evidence excerpts, retrieval
method, and explicit assumptions. Bin wall thickness/taper, spout tube diameter,
joint travel and experimental density are **not sourced product measurements**.
These are dimension-grounded procedural proxies, not CAD reconstructions. Bin
pedal linkage and faucet hydraulic internals are omitted. Both pass the existing
small compiler/settle/sweep/idealized-servo checks; this is not a G1/G2 success-rate
benchmark, agent repair study, or a semantic shape-quality score.

## P5: faithful retrieval before adaptation

Three fixtures from the official RoboCasa/Lightwheel distribution:
Microwave075, Dishwasher051, Refrigerator055. Source revision, all downloaded
file hashes, licenses and importer conventions are in
[ROBOCASA_DATASET.md](ROBOCASA_DATASET.md) and `robocasa_sources.json`.
Raw asset files remain unchanged. No PartNet access agreement was accepted.

- Nine G3 cases: each native asset at 0.75x, 1x, 1.25x. Geometry, pivots, slide
  limits/reference positions, and annotation sizes scale together. Density is
  retained; explicit mass/inertia use cubic/quintic scaling. Damping, springs,
  actuator gear and armature remain unchanged: **not dynamic similarity**.
- Three additional 1x runtime-style baselines: upstream scene compiler's
  `inertiagrouprange="0 0"` excludes visual/region geometry from inferred inertia.
  This reproduces that particular convention, not full RoboCasa scene sizing,
  solver configuration, placement or task controls.
- Raw 1x model arrays match the original native source. Uniform visible extents
  match requested scaling within numerical precision. Portable archives reload;
  mass/range checks allow 1e-5 relative error because MJCF decimal serialization
  and recomputed mesh inertia introduce a few ppm of difference.
- Source closed configurations have zero reported collision penetration. The
  raw dishwasher moves roughly 5.5–7.2 mm in the 5-second passive tests; this is
  reported, not silently marked stable. Sweeping drawers with doors closed
  creates large overlaps in the dishwasher/fridge. A second diagnostic opens
  other limited hinges first; this removes the reported drawer overlaps in all
  tested variants. Exact per-joint results are retained. This is not
  an inferred manipulation plan or continuous collision proof.

The adapter regression suite catches a mesh-cache pitfall: downsizing a copied,
already-compiled spec can retain stale bounds. Non-unit scaling reparses the
embedded spec before transformation; 1x is a genuine no-op copy.

## G4: an explicit unresolved input requirement

These assets have named handle **collision boxes**, but their visible handles are
baked into entire door/freezer meshes. G4 is rejected with an explicit reason;
there is no fake handle-preserving render that edits only colliders. The next
G4 test needs verified matching visual/collision segmentation and attachment
anchors, or a source asset that already supplies separate handle parts. This
follow-up does **not** select G3 versus G4 or validate gripper access/graspability.

Initial direction: native retrieval plus uniform scaling is a useful baseline;
source-aware import conventions and configuration-dependent validation are
essential. Keep part-aware adaptation conditional on usable semantic parts.
Neither a successful render nor a collision-filtered pose check makes an asset
fully robot-ready. Mass calibration, task affordances, grasp/approach tests and
full-scene support checks remain separate work.

Assets and derived RoboCasa renders: RoboCasa Team / Lightwheel, CC BY 4.0.
Exports include attribution and report the scale/compiler modification. Source
documents are cached as evidence, not redistributed as generated geometry.
