# Brief status and evidence

The pipeline is a working prototype. [running.md](running.md) is the maintained
run/reproduction guide; the [agent skill](../skills/scene-pipeline/SKILL.md) is
tracked and usable from any command-capable harness. Private historical notes in
ignored `docs/agent/` are not needed to operate it.

The local `deliverables/README.md` and `deliverables/CHECKLIST.md`, when present,
are the authoritative artifact inventory. They link the grouped scenes, variants,
interaction, data cards, costs and verification results. Generated files and the
large deliverable snapshot are not part of a fresh checkout. Consult saved run
reports instead of inferring completion from a directory name or this document.

## Established evidence and limits

| Area | Available evidence | Limit |
| --- | --- | --- |
| Scene generation | Supplied declarative programs, intent-preserving repair loop, optional Codex/OpenRouter adapters | Fixed programs are not cold prompt-only evaluations; live model comparison remains unverified |
| Dimensions/provenance | Declared defaults, axis-bound prompt evidence, archived source fields, compiled metric/mass checks | Semantic source selection can be wrong; default masses are not calibrated |
| Architecture/layout | Heuristic furnished placement and optional frozen floorplan priors with lineage checks | Current empirical corpus is residential; doorway scaling and industrial realism are not calibrated |
| Assets | Procedural templates, RoboCasa articulation, static catalog search/import, generated visual dressing | Static props do not establish manipulation capability; generated dressing has no contact geometry |
| Articulation | Passive settling, sampled sweeps/endpoints and a contact-driven Panda drawer sequence | No dynamic full-range proof for every articulation; general manipulation remains unproven |
| Export | Articulated URDF packages independently loaded and actuated in PyBullet | USD deferred; cross-engine contact/PBR equivalence not claimed |
| Visuals | Same compiled geometry for MuJoCo and CPU Cycles; live fal PBR/generated-clutter exercise | Recognizable equipment does not establish photorealism; appearance must match capture provenance |
| Mapping/data | Multi-rate synchronized HDF5, range-map metrics, six full and four state-tier captures in the collected baseline | Synthetic sensor noise; small omnibase and low camera; visibility coverage differs from traversal |
| Reproduction | Locked dependencies, pinned resources, doctor, portable archives, configurable cache/resource paths | Cold-machine timing and unseen-prompt acceptance not certified |

The collected baseline has a commercial kitchen, warehouse and home kitchen, ten
completed 60-second mapping runs split five kitchen/five warehouse, and a separate
60-second Panda interaction. The commercial kitchen and warehouse were generated
from supplied programs. The home kitchen exercised actual model calls and cached
fal assets with one repair; that is warm-library prompt-to-scene evidence.

The requested fal variant refresh adds varied textures and visual-only clutter,
then re-records captures so RGB-D and scene appearance agree. Staging progress lives
under `outputs/fal_variant_refresh/`; use each domain's `dataset.json` and published
deliverable reports to determine which refreshed batches are complete. Do not
substitute undecorated captures or partial GPU test runs for the decorated batches.

## Reproduction and evaluation boundaries

Regression tests cover geometry, evidence binding, asset packages, capture clocks,
loading/export and mocked provider transport. Paid calls are outside pytest. Run
`python -m pytest -q` for the installed checkout; historical counts are not a claim
that the current working tree has passed a new full-suite run.

An earlier installed-wheel smoke generated a template outside the repository with
no API key or agent on PATH, using cached dependencies and existing system libraries.
It was not a cold-machine installation measurement. Fixed-program multi-domain
matrices and small retrieval exercises similarly test composition and failure
handling, not realistic unrestricted domain generalization. Failed assets/layouts
remain useful evidence and must not be relabeled as successful environments.

## Remaining brief requirements

- The ten collected variants span two scenes. The literal brief needs at least ten
  variants of one scene: generate five more of either domain, or agree on the split.
- Measure a clean-machine installation separately and run the three unseen prompt
  evaluations without tuning inputs after observing held-out results.
- Complete API cost accounting where provider billing was unavailable; distinguish
  initial acquisition, cache reuse, generation, capture, render and video overhead.
- Keep the submission writeup and exported source snapshot aligned with the final
  published artifacts. Refresh the measured cost table from those exact runs.

The task does not gain those missing acceptance results merely through a loadable
scene, a passing geometric check or an attractive render.
