# Deliverable audit

Audited against the original brief and the generated artifacts. The primary collection now uses one home-kitchen scene; commercial kitchen and warehouse batches are supplementary.

| Brief deliverable | Status | Evidence / remaining work |
| --- | --- | --- |
| Repo: prompt → scene; flow → dataset | Collected; cold acceptance unverified | Source, CLI, setup, lockfile, tests, loader, reproduction guide and portable skill are in code/. No fresh-machine under-15-minute measurement or completed three-unseen-prompt evaluation. |
| At least two environments | Collected: three | Home kitchen, commercial kitchen and warehouse. First two commercial/warehouse fixtures use supplied programs; home kitchen has warm-library live model evidence with repair. |
| Offline render per environment; flows on video | Collected | Three base-scene stills, twenty variant stills/videos, a 60-second Panda interaction and a separately labeled unsuccessful navigation example. |
| Mapping error report | Collected and verified | Twenty 60-second runs, metrics independently recomputed from saved maps. Visibility coverage is distinct from physical traversal. |
| Ten randomized runs of one scene, tiering and data card | **Collected: ten home-kitchen variants** | Three full RGB-D and seven state/range/IMU. Lighting, fal textures, placement and clutter vary. Exact factors, synchronized HDF5, schema, loader and biases are documented. Ten supplementary runs are separate. |
| Cost/timing table by stage | Collected; historical billing incomplete | Current prepare/capture timings and estimated new fal spend; primary batch reuses cached assets with no new paid/model calls. Earlier acquisition/provider billing and some encoding overhead are not fully accounted. |
| Writeup ≤800 words | Updated: 655 words | reports/writeup.md matches current scope, hardware, outcomes and limitations. |

Verification: reports/primary-verification.json, reports/decoration-verification.json and reports/collection-verification.json. Current hashes are in MANIFEST.json; earlier collection-only checks are historical under reports/history/.

Remaining limits: clean-machine installation and unseen-prompt evaluation; complete historical cost accounting; calibrated dynamics/noise; broad manipulation and dynamic full-range articulation evidence. Joint state is simulator truth, not a separate noisy joint-encoder stream. Generated clutter is visual-only. Rejected home seed 308 is retained separately; accepted layouts do not imply a 100% generation success rate.

Bonuses: semantic navigation resolves named goals and records action chunks, but the measured refrigerator example failed true-distance arrival. Stretch load/reload and control regression evidence exists. No saved live VLM navigation success, humanoid, USD export or generated-mesh articulation is claimed. These optional gaps do not invalidate the completed tiered primary dataset.
