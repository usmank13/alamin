# Cost and timing table

Historical partial measurement, retained with its original values. This table
predates the complete ten-run collection and the fal/GPU refresh. Generate a
current table with `pipeline costs` over the exact artifacts being reviewed; the
local deliverable snapshot has its own consolidated reports.

| artifact | stage | wall s | sim s | attempts | tokens in/out | API spend USD | passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen | generate/layout | 0.3 | n/a | n/a | n/a | n/a | n/a |
| kitchen | generate/compile | 0.1 | n/a | n/a | n/a | n/a | n/a |
| kitchen | generate/validation | 3.1 | n/a | n/a | n/a | n/a | n/a |
| kitchen | generate/agent+io | 2.8 | n/a | 1 | n/a | 0 | n/a |
| kitchen | render/cycles | 7.4 | n/a | n/a | n/a | n/a | n/a |
| kitchen | export/urdf+pybullet | 39.2 | n/a | n/a | n/a | n/a | n/a |
| warehouse_final | generate/layout | 0.1 | n/a | n/a | n/a | n/a | n/a |
| warehouse_final | generate/compile | 0.2 | n/a | n/a | n/a | n/a | n/a |
| warehouse_final | generate/validation | 1.1 | n/a | n/a | n/a | n/a | n/a |
| warehouse_final | generate/agent+io | 2.3 | n/a | 1 | n/a | 0 | n/a |
| warehouse_final | render/cycles | 8.4 | n/a | n/a | n/a | n/a | n/a |
| warehouse_final | export/urdf+pybullet | 2.6 | n/a | n/a | n/a | n/a | n/a |
| kitchen_interaction | flow/interaction | 126.8 | 60.0 | n/a | n/a | n/a | True |
| mapping | flow/mapping | 636.4 | 60.0 | n/a | n/a | n/a | True |
| warehouse_mapping_smoke | flow/mapping | 4.3 | 2.0 | n/a | n/a | n/a | True |
| **total** | | **835.1** | | | | | |

Wall seconds are measured for this run; physics and current Cycles rendering use the CPU. `n/a` spend means billing was unavailable; pass `--usd-per-mtok-in/--usd-per-mtok-out` to price them. Deterministic stages have no API spend. fal rows are list price at request time with cache hits at 0, not provider billing.
