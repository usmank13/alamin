# Native RoboCasa articulated fixture pilot

Three complete, unmodified fixture folders are ready. No RoboCasa/robosuite installation is required to load them in MuJoCo. This is original distributed RoboCasa/Lightwheel fixture geometry, not ManipGen or a replacement claim for original PartNet-Mobility scenes.

| Fixture | Native MJCF, relative to workspace | Main-region width × depth × height | Door motion |
| --- | --- | --- | --- |
| Microwave075 | `vendor/robocasa_native/fixtures/microwaves/Microwave075/model.xml` | 0.626584 × 0.512752 × 0.352458 | `microjoint`, +Z hinge, −π/2 to 0 |
| Dishwasher051 | `vendor/robocasa_native/fixtures/dishwashers/Dishwasher051/model.xml` | 0.650982 × 0.649131 × 0.971048 | `door_joint`, +X hinge, 0 to 0.6235987756 |
| Refrigerator055 | `vendor/robocasa_native/fixtures/fridges/Refrigerator055/model.xml` | 0.823178 × 0.753878 × 1.804854 | `fridge_door_joint`, +Z hinge, 0 to π/2 |

Dimensions above are twice the `reg_main` box half-sizes, **not** a measured union of all collision/visual geometry or guaranteed handle-inclusive bounds. Raw scale is unchanged. All three compile directly with MuJoCo 3.13 in the workspace environment, and `mj_forward` succeeds. Their qpos dimensions are 4, 4, and 3 respectively, with all-zero `qpos0`.

## Frames, closed reference, and scaling

All three explicitly set `<compiler angle="radian"/>`; length values follow the source's meter-scale convention. +Z is up, X spans width, and the front is −Y. Body/mesh positions refer to the native fixture frame; origins are not necessarily object centers, hinge centers, or ground contact points. Mesh scales default to one. MJCF quaternions are **w x y z**. Mesh/texture paths are relative to each XML; preserve the sibling `visuals/` directory.

Use zero joint positions for the closed/stowed reference, with buttons unpressed and the microwave tray at zero angle. The preserved `fixture.py` normalizes joint ranges and reverses the normalized direction for ranges with a negative lower limit; this confirms that the microwave closes at its upper endpoint (zero), while the other selected door joints close at their lower endpoint (zero). This is a reference configuration, not a guarantee of zero contact or a dynamically settled state.

Additional articulations:

- Microwave: `Button001_joint`, `Button002_joint` slide along +Y from 0 to 0.0025; `Disc001_joint` rotates around +Z and is **unlimited**, despite an XML `range="0 360"`. Since compiler angles are radians, do not reinterpret that range as degrees or impose it as a joint limit.
- Dishwasher: `button_power_joint` slides +Y through 0–0.002; `rack0_joint` and `rack1_joint` slide −Y through 0–0.2 and 0–0.3. Preserve the source door's unusually short 0.6235987756-radian limit, rather than assuming 90 degrees.
- Refrigerator: `freezer0_door_joint` and `fridge_drawer0_joint` are **slides**, both −Y through 0–0.5. The freezer's joint name includes `door` but it is not a hinge.

For uniform geometric scaling, multiply body/geom/site/joint-anchor positions, primitive half-sizes, mesh scales, slide limits and any slide reference positions by the scale. Keep angles, quaternions, and axis directions unchanged. Native source positions and pivots must be transformed together; do not center individual meshes. Source mass/inertia/contact/actuator parameters require an explicit downstream policy if geometric scale changes; geometry scaling alone is not dynamic similarity. Anisotropic scaling of rotated primitives or articulated bodies needs frame-aware treatment and cannot generally be implemented by multiplying every local XYZ field by a world-axis scale vector.

The official registry applies additional scene-dependent sizing: microwave default `[0.8, -1, -1]`, dishwasher `[0.65, null, null]`, bottom-freezer fridge `[0.78, 0.80, 1.75]`. These raw fixtures have **not** had that runtime sizing applied. Treat the microwave `-1` entries as source registry sentinels, not negative dimensions. For this pilot, native scale and the `reg_main` dimensions provide a concrete reference; the preserved registry/source files support any later reproduction of scene sizing.

## Handle semantics for G4 and visuals

Handles are **geoms on articulated bodies**, not separate links/joints. Identify geometry by the named, active XML elements rather than assuming every OBJ filename is used: the source folders retain some unused visual files.

| Fixture | Moving body | Named contact geometry |
| --- | --- | --- |
| Microwave | `door` | `door_handle_main` |
| Dishwasher | `door` | `door_handle_main` |
| Refrigerator main door | `fridge_door` | `fridge_door_handle_main`, `fridge_door_handle_1`, `fridge_door_handle_2` |
| Refrigerator freezer drawer | `freezer0` | `freezer0_door_handle_main`, `freezer0_door_handle_1`, `freezer0_door_handle_2` |

All named handles above are collision boxes. Their `size` is local **half-extents**, and `pos`/`quat` are relative to the owning body. The microwave handle box has a substantial rotation: do not read its three half-sizes as world-axis dimensions. For G4, transform the box corners (or use compiled `geom_xpos`/`geom_xmat`) in the requested qpos state. Named `_main` geoms describe a graspable segment, not necessarily the whole handle including all supports. The microwave also has unnamed support collision boxes. Full attributes and owner bodies are in `robocasa_sources.json`.

The visible handles are embedded in the active door/freezer meshes (`door.obj`, `fridge_door.obj`, `freezer0.obj`), with materials and texture coordinates retained. Named collision boxes provide semantic contact targets but are not exact visual segmentations.

**G4 limitation:** these sources do not supply a standalone visual handle segmentation. If G4 requires independently identified visual handle geometry, it should remain unresolved/fail closed until segmentation is supplied and validated. A named collision box is evidence for a contact target, not proof that the visual handle was segmented or a justification for replacing the entire door mesh.

Visual geoms use group 1 and `contype="0" conaffinity="0"`. Collision geoms use group 0, mostly boxes/cylinders, and inherit contact masks unless explicitly overridden. `region` geoms are noncontact, transparent annotations; they are not solids. To render source appearance, show visual group 1 and hide collision group 0; retain material textures and alpha, including glass. To diagnose G4 contacts, render named handle collision geometry separately or as overlays. Do not indiscriminately convert visual meshes into contact meshes, since that changes the native collision model. Geom `group` alone does not set collision behavior; preserve masks/classes as well.

Native motor actuators are present (microwave door gear 10, most dishwasher/fridge joints gear 50). For static reference visuals, assign qpos and call `mj_forward`; no rollout is needed. Do not remove actuators or rewrite contact parameters in the archived raw sources. Per-test rendering and scaling adapters remain owned by the P5/harness implementation.

## Raw MJCF versus RoboCasa runtime inertia

**Direct fixture compilation is not the full RoboCasa runtime import.** RoboCasa's pinned README requests robosuite `master`, without a dependency commit lock. For this review, the official robosuite source was separately pinned at `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`; snapshots and hashes are included under `provenance/robosuite_upstream/`. Conclusions below apply to those inspected sources; no complete environment was installed or executed.

The import chain is `Fixture` → `MujocoXMLObjectRobocasa` → robosuite `MujocoXMLObject`. The scene uses `Task`/`MujocoWorldBase`, whose `robosuite/models/assets/base.xml` sets:

```xml
<compiler angle="radian" meshdir="meshes/" inertiagrouprange="0 0" autolimits="true"/>
```

This limits automatically inferred inertia to **group 0**. Selected fixture visual meshes and `region` geoms are group 1, so they do not contribute. `contype=0`, `conaffinity=0`, and zero alpha alone do **not** exclude a geom from inferred inertia. The raw fixture compiler lacks this group restriction, explaining the large direct-load mass sums.

The robosuite XML importer expands source default classes inline, resolves mesh/texture paths, takes the `object` body subtree, names/prefixes elements, and applies requested scale. Both visual and collision geoms remain (`obj_type="all"`); excluding visual inertia does not mean deleting the visual meshes. Scene merging carries bodies/assets/actuators/etc., not the fixture compiler or option elements, so the scene's global compiler/options govern. Fixture initialization makes region annotations transparent and records their bounds; that alone does not remove their mass in a direct load.

Measured with MuJoCo 3.13 at **native unscaled geometry**, changing only `inertiagrouprange` in memory:

| Fixture | Raw sum of body masses (kg) | Group-0-only inertia sum (kg) |
| --- | ---: | ---: |
| Microwave075 | 383.049916 | 67.477291 |
| Dishwasher051 | 1335.527190 | 228.188578 |
| Refrigerator055 | 4134.119522 | 515.267398 |

These are diagnostic compiler outputs, **not measured appliance masses or full-scene runtime mass predictions**. They include anchored root bodies; geometry overlaps and solid collision approximations also contribute. Additional official scene sizing changes the values. Selected collision geoms generally lack explicit density/mass, so the compiler's default geom density remains relevant. The same-body masses for important joints change as follows (not articulated subtree masses): microwave door 13.6978 → 5.5789 kg; dishwasher door 29.8638 → 12.8611 kg; fridge door 217.7720 → 100.2542 kg; freezer drawer 567.5669 → 105.3535 kg.

The reviewed fixture importer has **no blanket density=100, damping, frictionloss, or armature rewrite**. `MujocoXMLObjectRobocasa.set_scale` changes geometry and slide ranges; it does not rescale inertial or joint dynamics parameters explicitly. A different class in the same source file, `MJCFObject`, defaults to `density=100` and rewrites geom density/friction/solver values for movable objects. `Fixture` does **not** inherit that class, so applying its density rule to these appliances would introduce a new adapter policy. Similarly, robosuite's tiny mass (`1e-8`) for duplicated visual collision geoms is not a blanket mass override for preexisting fixture visuals; microwave/dishwasher explicitly disable collision duplication.

Source joint values persist in the reviewed import path: selected joints have damping 1, frictionloss 1, and armature 0.01, except dishwasher `door_joint` has frictionloss **10**. Preserve per-joint XML values rather than applying a common friction default. Visual mesh geoms contain friction/solver attributes but have contact disabled; do not assume their friction values apply to the separate collision shapes. No additional mass/friction rewrite was found in the reviewed `kitchen.py` scene code. This source review does not establish every possible task-specific runtime modification.

For a source-aware comparison, label the two modes explicitly: **raw fixture direct-load** versus **runtime-style group-0 inertia at native scale**. Adding the compiler group restriction to an adapter reproduces this specific upstream convention; it does not by itself reproduce scene sizing, placement, global solver/fluid options, controls, or the entire runtime. Raw downloaded assets remain unchanged.

## Source, license, and reproducibility

Official repository: [robocasa/robocasa](https://github.com/robocasa/robocasa), pinned commit `4f8a2980def75a55dff96b990745b83540425f09`. Its `robocasa/models/assets/box_links/box_links_assets.json` identifies the official Lightwheel fixture archive. Preserved under `vendor/robocasa_native/provenance/`: pinned README, MIT code LICENSE, asset-link registry, fixture source definitions (read only, never executed), and the selected category registries.

The pinned README explicitly distinguishes **code: MIT** from **assets/datasets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. Attribution: RoboCasa Team, Lightwheel fixture collection distributed through RoboCasa. Preserve attribution and license links when redistributing derived assets or visuals, and identify modifications. The upstream license declaration and code license are bundled; fetching the full CC legal text returned HTTP 403, so that text is linked rather than fabricated or mislabeled as MIT.

The Box archive is 529,745,465 bytes. Its URL is selected by the pinned Git commit, but the external archive itself is not a Git blob or immutable commit-addressed object. The manifest therefore locks **every downloaded file with SHA-256**, also records archive member CRC32 values, and explicitly does not claim a full-archive SHA-256. The initial asset fetch transferred approximately 1.93 MB (ZIP directory plus selected compressed entries), yielding 2,682,743 uncompressed fixture bytes. No full archive, full environment, or additional corpus was downloaded.

```sh
# Fetch the same three folders; existing locked files are reused and verified.
python prototypes/fetch_robocasa.py

# Offline per-file size/hash verification, no network access.
python prototypes/fetch_robocasa.py --verify
```

`--list` inspects only ZIP metadata. Fetching requires proper HTTP 206 responses and matching Content-Range values; otherwise it refuses the full download. Limits: 64 MiB per range, 160 MiB ZIP transfer per run, 128 MiB uncompressed selection, 32 MiB per member, and 2000 selected files. Traversal paths and symlink members are rejected. ZIP CRC verifies freshly downloaded entries; the saved SHA-256 manifest locks future downloads and detects source changes. Existing differing files are not overwritten. `robocasa_sources.json` inventories all raw files and metadata, compiler flags, joints, handle collision geometry, and unresolved dependencies (none).

Validation performed: all three native XMLs compile unchanged and pass `mj_forward` in local MuJoCo; mesh/texture references resolve locally; file hashes verify offline. This is a dataset-readiness check, not the independent physics/scale/visual test suite.
