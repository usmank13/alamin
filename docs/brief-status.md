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
| Mapping/data | Primary ten home-kitchen captures (three full/seven state), ten supplementary captures, synchronized HDF5 and range-map metrics | Synthetic sensor noise; small omnibase and low camera; visibility coverage differs from traversal |
| Reproduction | Locked dependencies, pinned resources, doctor, portable archives, configurable cache/resource paths | Cold-machine timing and unseen-prompt acceptance not certified |

The collected baseline has a commercial kitchen, warehouse and home kitchen, ten
completed 60-second mapping runs split five kitchen/five warehouse, and a separate
60-second Panda interaction. The commercial kitchen and warehouse were generated
from supplied programs. The home kitchen exercised actual model calls and cached
fal assets with one repair; that is warm-library prompt-to-scene evidence.

The primary collection is complete and verified: ten home-kitchen variants, each
with 60 simulated seconds of mapping, three full RGB-D and seven state-tier runs.
It contains 1,803 camera frames and 60,010 state samples, plus ten recorded videos
and scene visualizations. Published evidence is under
`deliverables/outputs/home_kitchen/variants/`; its `dataset.json` records passing
results, exact factors and devices. Commercial-kitchen and warehouse batches are
supplementary. All observations were recorded after decoration.

Accepted layout seeds are 300–307, 1308 and 309. Candidate 308 failed robot-access
validation; its inputs/checks/costs are retained in `rejected_layouts/`. The bounded
replacement policy preserves acceptance checks. These are accepted-layout datasets,
not a claim of 100% generation success. Five cached fal themes supply varied
finishes; primary generation made zero new paid fal or model calls. That does not
erase the historical cost of producing the cached assets.

Primary full captures took 893.5, 731.2 and 385.6 wall seconds; state captures took
89.8–101.1 seconds, excluding later replay encoding. Concurrent workload varied.
All recorded renderer identities are NVIDIA EGL on the RTX 3070 Ti Laptop GPU;
physics, Cycles and ffmpeg remain CPU work. This is not a controlled speedup study.

The [bonus evidence](bonuses.md) includes a recorded semantic-navigation failure:
the refrigerator resolved correctly, but estimated arrival at 0.474 m corresponded
to a true distance of 0.924 m, outside the 0.6 m criterion. Stretch control/loading
checks pass; autonomous mobile manipulation and generated-mesh articulation remain
unproven. No live VLM navigation success is claimed.

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

- Measure a clean-machine installation separately and run the three unseen prompt
  evaluations without tuning inputs after observing held-out results.
- Complete API cost accounting where provider billing was unavailable; distinguish
  initial acquisition, cache reuse, generation, capture, render and video overhead.
- Keep the submission writeup and exported source snapshot aligned with the final
  published artifacts. Refresh the measured cost table from those exact runs.

The task does not gain those missing acceptance results merely through a loadable
scene, a passing geometric check or an attractive render.
