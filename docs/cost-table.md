# Cost and timing table

| artifact | stage | wall s | sim s | attempts | tokens in/out | API spend USD | passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen | generate/layout | 0.4 | n/a | n/a | n/a | n/a | n/a |
| kitchen | generate/compile | 0.2 | n/a | n/a | n/a | n/a | n/a |
| kitchen | generate/validation | 3.4 | n/a | n/a | n/a | n/a | n/a |
| kitchen | generate/agent+io | 3.0 | n/a | 1 | n/a | n/a | n/a |
| kitchen | render/cycles | 43.0 | n/a | n/a | n/a | n/a | n/a |
| kitchen | export/urdf+pybullet | 33.3 | n/a | n/a | n/a | n/a | n/a |
| kitchen_mapping | flow/mapping | 560.0 | 60.0 | n/a | n/a | n/a | True |
| kitchen_interaction | flow/interaction | 407.9 | 60.0 | n/a | n/a | n/a | True |
| kitchen_nav | flow/navigate | 33.1 | 4.0 | n/a | n/a | n/a | True |
| kitchen_nav_shelf | flow/navigate | 245.1 | 27.0 | n/a | n/a | n/a | True |
| kitchen_nav_prep | flow/navigate | 326.9 | 36.0 | n/a | n/a | n/a | True |
| kitchen_nav_vlm | flow/navigate | 333.0 | 20.0 | n/a | n/a | n/a | True |
| **total** | | **1989.4** | | | | | |

Wall seconds are measured on this CPU-only machine. `n/a` spend means the Codex CLI reports tokens but no price; pass `--usd-per-mtok-in/--usd-per-mtok-out` to price them. Deterministic stages have no API spend.
