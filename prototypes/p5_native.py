"""Native retrieved MJCF: preserve baseline semantics before uniform rescaling.

Deliberately rejects unsupported model features and unsegmented G4 visual parts.
This is not a reimplementation of the complete RoboCasa environment.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from sim_harness.scene import _read_spec
from .visual_checks import bounds, gallery, render_case


def scale_native(source, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be positive and finite')
    if scale == 1:
        # True no-op baseline, without XML float serialization.
        spec = source.copy()
    else:
        spec = mujoco.MjSpec.from_string(source.to_xml(), assets=source.assets)
    # Reparse to discard compiled mesh caches. Copying an already-compiled spec
    # can retain unscaled mesh bounds on downsizing (covered by regression test).
    if any(len(getattr(spec, name)) for name in ('keys', 'equalities', 'tendons', 'flexes', 'hfields', 'skins', 'frames')):
        raise ValueError('Pilot does not support keys, constraints, tendons, flexes, heightfields, skins or frames')
    if any(j.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for j in spec.joints):
        raise ValueError('Only scalar joints supported')
    for body in spec.bodies:
        body.pos *= scale
        if body.explicitinertial:
            body.ipos *= scale
            body.mass *= scale**3
            body.inertia *= scale**5
            body.fullinertia *= scale**5
    for mesh in spec.meshes:
        mesh.scale *= scale
    for geom in spec.geoms:
        geom.pos *= scale
        geom.size *= scale
        geom.fromto *= scale
        geom.mass *= scale**3  # NaN sentinel remains NaN; density unchanged.
        geom.margin *= scale
        geom.gap *= scale
    for site in spec.sites:
        site.pos *= scale
        site.size *= scale
        site.fromto *= scale
    for joint in spec.joints:
        joint.pos *= scale
        if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint.range *= scale
            joint.ref *= scale
            joint.springref *= scale
            joint.margin *= scale
    # Keep source controls/friction/springs/armatures; no claim of dynamically
    # similar mechanisms or physical calibration after geometric rescaling.
    return spec


def probe(model):
    """Unactuated 5s and collision-filter-aware pose probes; no servo overrides."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    origin = data.xipos.copy()
    worst, drift = 0., 0.
    warnings = np.zeros(len(data.warning), dtype=int)

    def penetration():
        return max([0.] + [-float(c.dist) for c in data.contact])

    closed = penetration()
    max_q = np.zeros(model.nq)
    steps = round(5/model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        worst = max(worst, penetration())
        drift = max(drift, float(np.linalg.norm(data.xipos-origin, axis=1).max()))
        max_q = np.maximum(max_q, abs(data.qpos-initial))
        warnings = np.maximum(warnings, data.warning.number)
    sweeps = []
    for j in range(model.njnt):
        peak, open_peak = 0., 0.
        limits = model.jnt_range[j] if model.jnt_limited[j] else [-np.pi, np.pi]
        for value in np.linspace(*limits, 21):
            mujoco.mj_resetData(model, data)
            data.qpos[model.jnt_qposadr[j]] = value
            mujoco.mj_forward(model, data)
            peak = max(peak, penetration())
            # Diagnostic precondition: open all *other* limited hinges before
            # sweeping. Not an automatically inferred task/affordance graph.
            for other in range(model.njnt):
                if other != j and model.jnt_type[other] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[other]:
                    data.qpos[model.jnt_qposadr[other]] = model.jnt_range[other, np.argmax(abs(model.jnt_range[other]))]
            mujoco.mj_forward(model, data)
            open_peak = max(open_peak, penetration())
        sweeps.append(dict(joint=model.joint(j).name, penetration_m=peak,
                           other_hinges_open_penetration_m=open_peak,
                           range_source='native limits' if model.jnt_limited[j] else 'test interval +/- pi; source joint unlimited'))
    handles = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if 'handle' in name:
            handles.append(dict(name=name, full_local_box_size_m=(2*model.geom_size[i]).tolist()))
    return dict(closed_penetration_m=closed, passive_5s_max_penetration_m=worst,
                passive_5s_max_com_drift_m=drift, passive_5s_max_joint_drift=max_q.tolist(),
                warnings=warnings.tolist(), sweeps=sweeps, handles=handles,
                total_inferred_mass_kg=float(model.body_mass.sum()),
                numerically_finite=bool(np.isfinite(data.qpos).all()),
                collision_pose_check_pass=bool(max([closed]+[s['penetration_m'] for s in sweeps]) <= .002),
                caveats=['Contacts honor source masks and parent filtering; not all geometric intersections.',
                         'Each joint swept with other joints closed; drawer/door interlocks may need a task-specific sequence.',
                         'No servo, robot grasp, floor/support stability or calibrated mass verification.'])


def main():
    root = Path('outputs/prototypes/representative')
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path('prototypes/robocasa_sources.json')
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['files']:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Source hash mismatch: {entry['path']}")
    records = []
    for prefix in manifest['prefixes']:
        path = Path('vendor/robocasa_native')/prefix/'model.xml'
        source = _read_spec(path)
        native = source.compile()
        data = mujoco.MjData(native)
        mujoco.mj_forward(native, data)
        lo, hi = bounds(native, data)
        original_extent = hi-lo
        native_probe = probe(native)
        for scale in (.75, 1., 1.25):
            case = f'p5_{path.parent.name}_G3_{scale:g}'
            directory = root/case
            directory.mkdir(parents=True, exist_ok=True)
            spec = scale_native(source, scale)
            model = spec.compile()
            measurements = probe(model)
            record = dict(case=case, route='G3', scale=scale, source=str(path),
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                          baseline=native_probe, measurements=measurements,
                          policy='Native XML geometry/materials/physics; density-preserving geometry scaling. Controls, friction, damping, stiffness, armature unchanged. Not full RoboCasa runtime.',
                          g4_status='Rejected: handles have named colliders, but visual handles are baked into whole-door meshes. Requires verified part segmentation and attachment anchors; collider-only edits would be misleading.')
            record['visuals'] = render_case(model, directory, f'P5 / {path.parent.name} / native G3 {scale:g}x', distance=float(np.linalg.norm(original_extent)*2.2))
            nlo, nhi = np.array(record['visuals']['closed_bounds_m'])
            record['max_relative_extent_error'] = float(np.max(abs((nhi-nlo)-original_extent*scale)/(original_extent*scale)))
            # Make standalone exports compatible with the harness initial-state contract.
            spec.add_key(name='harness_initial', qpos=model.qpos0)
            spec.add_text(name='source_attribution', data=f'RoboCasa Team / Lightwheel; CC BY 4.0 https://creativecommons.org/licenses/by/4.0/ ; source {path}; modified uniform scale {scale}')
            spec.to_zip(str(directory/'scene.mjz'))
            roundtrip = mujoco.MjSpec.from_zip(str(directory/'scene.mjz')).compile()
            record['export_roundtrip_equal'] = bool(np.allclose(roundtrip.body_mass, model.body_mass) and np.allclose(roundtrip.jnt_range, model.jnt_range))
            record['export_relative_tolerance'] = 1e-5
            record['native_1x_equivalent'] = None if scale != 1 else all(np.array_equal(getattr(native, a), getattr(model, a)) for a in ('body_mass', 'body_inertia', 'geom_pos', 'geom_size', 'jnt_range', 'geom_contype', 'geom_conaffinity', 'dof_damping'))
            record['summary'] = f"Closed overlap {measurements['closed_penetration_m']*1000:.2f} mm; passive drift {measurements['passive_5s_max_com_drift_m']*1000:.2f} mm; G4 requires segmentation."
            (directory/'result.json').write_text(json.dumps(record, indent=2)+'\n')
            records.append(record)
            print(case, record['summary'], flush=True)
        # Reproduce one documented upstream compiler convention independently
        # of G3: raw direct-load is not the full RoboCasa runtime.
        runtime = _read_spec(path)
        runtime.compiler.inertiagrouprange = [0, 0]
        model = runtime.compile()
        case = f'p5_{path.parent.name}_runtime_inertia_1'
        directory = root/case
        directory.mkdir(parents=True, exist_ok=True)
        result = dict(case=case, route='runtime convention baseline', scale=1., source=str(path),
                      policy='Only compiler inertiagrouprange=0 0 changed to match upstream scene convention; not full RoboCasa runtime.',
                      measurements=probe(model), baseline=native_probe)
        result['visuals'] = render_case(model, directory, f'P5 / {path.parent.name} / runtime-style collision-only inertia 1x', distance=float(np.linalg.norm(original_extent)*2.2))
        runtime.add_key(name='harness_initial', qpos=model.qpos0)
        runtime.add_text(name='source_attribution', data=f'RoboCasa Team / Lightwheel; CC BY 4.0 https://creativecommons.org/licenses/by/4.0/ ; source {path}; changed inertia group to 0')
        runtime.to_zip(str(directory/'scene.mjz'))
        result['summary'] = f"Runtime-style inferred mass {result['measurements']['total_inferred_mass_kg']:.2f} kg vs raw {native_probe['total_inferred_mass_kg']:.2f} kg; includes anchored bodies, not measured product mass."
        (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        records.append(result)
        print(case, result['summary'], flush=True)
    (root/'p5_results.json').write_text(json.dumps(records, indent=2)+'\n')
    p4 = json.loads((root/'p4_results.json').read_text()) if (root/'p4_results.json').exists() else []
    gallery(root, p4+records)


if __name__ == '__main__':
    main()
