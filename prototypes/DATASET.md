# ManipGen cabinet/handle pilot

Ready-to-use minimal pilot: the following three URDFs, relative to the workspace:

| Asset | URDF | Handle XYZ extent (meters, rounded) |
| --- | --- | --- |
| Horizontal handle | `vendor/partnet_pilot/manipgen_subset/partnet/meshdata/door-33507-18-0/coacd.urdf` | 0.0303 × 0.1013 × 0.0160 |
| Vertical handle | `vendor/partnet_pilot/manipgen_subset/partnet/meshdata/door-40402-16-0/coacd.urdf` | 0.0264 × 0.0104 × 0.1222 |
| Compact round handle | `vendor/partnet_pilot/manipgen_subset/partnet/meshdata/door-30857-13-0/coacd.urdf` | 0.0318 × 0.0396 × 0.0396 |

Form descriptions are approximate geometric observations, not upstream semantic labels. Six assets (507,461 raw bytes) had finished extracting before scope was reduced to a 2–3 asset pilot. The three additional drawers remain available, but are not necessary for the minimal pilot: `drawer-46254-5-0`, `drawer-47167-8-0`, and `drawer-47710-9-0`, each with `coacd.urdf` under the same `meshdata/` directory. No additional corpora are needed.

## URDF integration conventions

- Links: `skeleton` (root), `door_link` (moving front), `handle_link` (separate handle). Drawers retain these names and even `robot name="door"`; infer category from the directory or `meta.yaml`, not robot/link names.
- `door_joint`: `skeleton` → `door_link`. Doors are revolute around local +Z, limits 0 to π/2. Drawers are prismatic along +X, limits 0 to 0.3. `handle_joint` fixes `handle_link` to `door_link`. All joint origins in these samples are zero, with zero RPY.
- `handle_link` uses `decomposed.obj` for both visual and collision; it is not merged into the front-panel mesh. Its visual/collision origins are zero and mesh scales are `1 1 1`. Handle placement is baked into OBJ vertex coordinates: do not recenter the handle without compensating its transform. Exact bounds and joint attributes are in `dataset_sources.json`.
- `cabinet_body.obj` is the root visual; its URDF origin is nonzero and must be applied. Front panels use box geometry. Drawers also use `drawer_body.obj` on `door_link`. All mesh filenames resolve relative to the URDF directory, with no external mesh dependencies.
- Standard URDF conventions imply meters/radians; the source does not separately declare a unit system. Dimensions are consistent with those conventions (cabinet depth 0.35, drawer travel 0.3). +X points outward from the cabinet front, Z is vertical, Y spans the front width. Coordinates refer to the asset frame, not the handle center or a separately calibrated grasp frame.
- No inertial elements are present in any of these six URDFs. They are raw geometry/kinematic inputs; the downstream physics pipeline must supply its own mass/inertia policy.
- **All three extracted drawers have an upstream malformed joint:** an extra `door_joint` refers to nonexistent parent `sektion` and child `drawer_top`. This duplicates the valid prismatic joint name. Raw files are intentionally unchanged; strict loaders may reject them. The three door samples have no missing-link or duplicate-joint defects in the structural inventory.
- Root collision is a single solid box enclosing the cabinet dimensions, not the hollow cabinet visual mesh. Downstream contact/physics validation should account for this approximation; this preparation step does not certify physical validity.
- `meta.yaml` is retained verbatim. Its `joint_val: 1` is not treated as a documented initial configuration. Grasp `.npy` files were not extracted or loaded; no dataset code or pickle payload was executed.

## Provenance and license

Source: [minliu01/ManipGen-PartNet](https://huggingface.co/datasets/minliu01/ManipGen-PartNet), pinned revision `5459c7323a1c04e0207ad30fc28f78e62ff168fc`. The dataset card describes procedural cabinet bodies combined with handles sampled from [PartNet](https://partnet.cs.stanford.edu/), for [ManipGen](https://mihdalal.github.io/manipgen/). This is **not** the original 15-asset, five-category P5 dataset and is not a collection of original PartNet-Mobility cabinet scenes.

The exact [pinned README](https://huggingface.co/datasets/minliu01/ManipGen-PartNet/raw/5459c7323a1c04e0207ad30fc28f78e62ff168fc/README.md) is preserved at `vendor/partnet_pilot/manipgen_subset/SOURCE_README.md`. Its YAML metadata declares `license: mit`. No README or LICENSE file was present inside the archive. Upstream PartNet handle terms were not independently resolved: the dataset-card MIT declaration should not be interpreted as independently established relicensing of upstream geometry. No license text has been fabricated.

The card calls its list `trainset3419.txt` while describing 2655 training objects; the actual archive has `partnet/trainset2655.txt` and 2657 URDF assets. Selection here is a small geometry pilot, not a claim of training-split membership or category coverage. The official ManiSkill UCSD host timeout and gated full PartNetMobility source were not pursued further.

Archive: `vendor/partnet_pilot/manipgen-partnet.tar.gz`, 552,155,633 bytes.

SHA-256: `a4120e4eb663f2976f7067547e19c55a09dc839d2fc4e9d06a98731c8448d555`.

`dataset_sources.json` records the pinned source, README hash, exact per-file hashes and sizes, raw URDF paths, links, joint parameters, handle bounds, and structural defects.

## Reproduce

From the workspace, with standard-library Python:

```sh
python prototypes/fetch_manipgen.py
```

This verifies the completed archive and extracts only the fixed six-asset selection. If the archive is absent, explicitly opt into fetching the same pinned archive:

```sh
python prototypes/fetch_manipgen.py --download
```

No other corpus is downloaded. Downloads have a 30-second socket timeout, a 30-minute total budget, size/hash checks, and progress reports. Extraction streams the archive into staging, rejects traversal paths and nonregular members, allows only specified asset filenames, caps each file at 16 MiB and the total extracted asset data at 64 MiB, and verifies all expected files exist before publishing. An existing identical subset is accepted; differing files are never overwritten. The manifest is deterministically regenerated. The script does not repair or execute dataset content.

Validation performed: archive size/hash match, all 27 extracted asset files inventoried with SHA-256, all mesh references resolve locally, all six XML documents parsed and structural defects reported, and repeat extraction checked for identical output. Simulation validation is owned by the separate prototype/physics work.
