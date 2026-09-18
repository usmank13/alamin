# Deliverable audit

Audited against the seven deliverables in the original task brief, the design/status
notes in `docs/agent`, and the files themselves on 2026-09-17.

| Brief deliverable | Status | Evidence / remaining work |
| --- | --- | --- |
| 1. Repo: prompt → scene; flow → dataset | Collected; cold acceptance unverified | `code/` includes CLI, setup, lockfile, loader, tests and instructions. No fresh-machine under-15-minute measurement or completed three-prompt cold unattended evaluation found. |
| 2. At least two environments | Collected: three | Commercial kitchen, warehouse and home kitchen archives compile at their copied locations; existing validation passes. First two use supplied declarative programs; home kitchen has two actual agent calls with one repair, using a warm asset library. Do not call the first two cold prompt-only successes. |
| 3. One offline render per environment; each flow on video | Collected | Three 1024×1024 Cycles stills, ten mapping videos and one 60-second interaction video. Video streams were probed; this pass did not watch every frame. |
| 4. Mapping error report | Collected and consolidated | `reports/mapping-report.md`; all ten IoU/distance metrics independently recomputed from saved maps. High visibility coverage does not imply the base traversed the whole room. |
| 5. At least ten randomized runs of one scene, with data card | Partial against literal brief | Ten completed 60-second runs exist, but split five kitchen/five warehouse. Six full RGB-D and four lightweight. Both batches pass their recorded checks. Need five additional distinct variants of either scene (or agreement to retain the prior mixed-domain scope). |
| 6. Cost/timing table by stage | Collected; billing incomplete | Refreshed `reports/cost-table.md` includes all completed batches and three scenes. Home-kitchen model billing is unavailable; its zero new fal spend reflects cached assets. Cold asset acquisition, earlier exploratory API spend, and some preprocessing/video-encoding costs are not a complete marginal-cost measurement. |
| 7. Writeup ≤800 words | Draft collected: 678 words | `reports/writeup.md` covers the requested topics but predates completed batches and the live fal exercise. Refresh completion, PBR/API claims and cost references before submission. |

Existing technical evidence includes primitive/convex collision handling, density
and source records, semantic manifests, sensor rig/noise metadata, HDF5 loader,
stock mobile base and Panda, and two independently verified articulated URDF exports.
The home kitchen supplies the live PBR/generated-clutter evidence; the other two
scenes mainly use default finishes and source textures. Generated clutter is visual
only, and the warehouse pallet jack is a static prop.

Remaining evidence gaps beyond the seven-item inventory:

- Physical stability/clearance reports use passive settling and finite kinematic
  sweeps. A Panda contact-driven full-range drawer sequence exists; dynamic
  full-range proof for every scene articulation has not been established.
- Sensor noise is synthetic, not calibrated. The collected `state` stream stores
  exact qpos/qvel truth; noisy wheel velocity is present in mapping odometry, but
  a general noisy joint-position/velocity observation stream is not demonstrated.
- Generalization, physical calibration, broad manipulation and realistic industrial
  layouts remain limited; passing artifact checks does not certify these qualities.

Not counted as finished evidence: `outputs/pipeline_dataset_pilot` has ten one-second
variants with `passed: false`; the warehouse mapping smoke lasts two seconds; failed
warehouse layout/preflight attempts remain in the original repository. They were
not substituted for the ten completed runs. Optional third-environment export,
USD, generated-mesh articulation and bonus navigation are not required missing items.

Collection checks are recorded in `reports/verification.json`: three archives
compile, still sizes match, eleven HDF5 files pass clock/length checks and bounded
loader reads, eleven video streams probe successfully, and all 2,165 copied files
match their SHA-256 hashes. No copied source changed during the audited interval.
Physics and URDF results are existing reports, not fresh acceptance runs.
