# Independent-format export contract

The trial requires one scene exported to a second independent format and loaded
with articulation intact. The selected first target is now **URDF + PyBullet
DIRECT**, implemented in `scene_pipeline.portability`; USD remains deferred.

`pipeline export SCENE_DIRECTORY --format urdf --verify` writes a main fixed
environment URDF and separate free-clutter roots, with package-relative meshes,
semantic metadata and an explicit fidelity report. Verification loads the entire
scene, compares mass/inertia/initial COM and drives every scalar joint to both
limits using the independent runtime's motors. Relocated-package loading is
covered by regression tests. This does not yet prove all bounds, axes and world
transforms below, or cross-engine force/contact equivalence. Robot tendons,
equalities, ball joints and multiple joints per body are rejected; sensor/actuator
export is not supported. See [implementation status](pipeline-implementation.md).

## Boundaries

The generator owns a dimensioned scene description and emits MJCF/assets plus
semantic metadata. The harness accepts MJCF and optional robot placements without
knowing which generation method produced them. A future independent exporter can
consume that same description, or adapt the composed MuJoCo specification exposed
by `compose_scene`. Neither approach should introduce a USD dependency into the
generator or the runtime loader. Put optional USD dependencies and code in an
export adapter when it is implemented.

Use stable object IDs in MJCF names and carry semantic class/instance mappings in
a generator-owned sidecar when MJCF custom fields are insufficient. The mapping
must survive robot prefixes and export. This harness preserves MJCF names but does
not create a semantic schema or infer classes. Avoid making runtime integer body
indices persistent object IDs, since composition can change those indices.

## Required preservation

| Source property | Export requirement |
| --- | --- |
| World coordinates | Explicit metres-per-unit and Z-up; same origin and transforms |
| Object identity | Stable instance IDs and semantic labels |
| Static/dynamic bodies | Correct rigid-body flags and static scene geometry |
| Mass/inertia | Mass, centre of mass, inertia frame and tensor |
| Visual/collision geometry | Separate geometry, same scale, collider assignments |
| Hinge/slide articulation | Parent/child relationships, local anchors, axes, limits |
| Initial configuration | Object poses and articulated joint positions |
| Materials/assets | Relative or packaged asset paths; no machine-specific references |
| Contact/control behavior | Explicit fidelity report for unsupported solver, friction, actuator, tendon and equality semantics |

Angular units need explicit conversion where the target schema expects degrees.
Do not assume MuJoCo tendons, contact filtering, equality constraints, or actuator
models map exactly to another engine. Record supported, approximated, and omitted
features per export. A visual-only USD animation is not articulated physics export.

## Acceptance checks for the future exporter

1. Export a generated scene containing hinge and slide objects, then load it in an
   independent physics runtime. Merely parsing the file is insufficient
   evidence that the articulation works.
2. Compare metric bounds, object transforms, body masses, joint counts, axes and
   limits to source ground truth with declared tolerances.
3. Exercise every exported hinge/slide through its full range. Record evidence
   that joints remain connected and collision geometry remains correctly aligned.
4. Move the exported package away from the source assets and repeat the load.
5. Save a fidelity report and the exact load command/tool version alongside the
   artifact, with any unsupported features clearly listed.

The existing `.mjz` round-trip test verifies native packaging and composition
only. It is useful groundwork, but it is not the independent-format acceptance test.
