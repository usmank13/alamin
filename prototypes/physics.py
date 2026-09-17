"""Shared pilot emitter/validator. Numerical settings are experimental, not calibrated."""
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def vec(values):
    return ' '.join(format(float(v), '.12g') for v in values)


def emit(parts, joints, output, density=500):
    """parts: link/name/mesh/collision_meshes; joints: parent/child/pos/rpy/axis/range/type.

    One fixed root, SI coordinates. Derive inertias from collision hull volume.
    density=500 is a declared sensitivity-test input, not sourced material data.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = ET.Element('mujoco', model='prototype')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='true', inertiagrouprange='3 3')
    option = ET.SubElement(root, 'option', timestep='.002', integrator='implicitfast', cone='elliptic')
    ET.SubElement(option, 'flag', filterparent='disable')
    assets = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -2 4', dir='0 0 -1', directional='true')
    links = set(p['link'] for p in parts) | {j['parent'] for j in joints} | {j['child'] for j in joints}
    children = {j['child'] for j in joints}
    roots = links - children
    if len(roots) != 1:
        raise ValueError(f'Expected one root, found {roots}')
    bodies = {}

    def add(link, parent, joint=None):
        attr = dict(name=link)
        if joint:
            xyzw = Rotation.from_euler('xyz', joint['rpy']).as_quat()
            attr.update(pos=vec(joint['pos']), quat=vec(xyzw[[3, 0, 1, 2]]))
        body = ET.SubElement(parent, 'body', **attr)
        bodies[link] = body
        if joint and joint['type'] != 'fixed':
            ET.SubElement(body, 'joint', name=joint['name'], type=joint['type'],
                          axis=vec(joint['axis']), range=vec(joint['range']),
                          limited='true', damping='1', armature='.01')
        for child in (j for j in joints if j['parent'] == link):
            add(child['child'], body, child)
    add(next(iter(roots)), world)
    for i, part in enumerate(parts):
        visual = part['mesh']
        meshes = [('v', visual)] + [(f'c{k}', hull) for k, hull in enumerate(part['collision_meshes'])]
        for label, mesh in meshes:
            name = f'p{i}_{label}'
            mesh.export(output / f'{name}.obj')
            ET.SubElement(assets, 'mesh', name=name, file=f'{name}.obj')
            is_visual = label == 'v'
            ET.SubElement(bodies[part['link']], 'geom', name=name, type='mesh', mesh=name,
                          group='2' if is_visual else '3', density='0' if is_visual else str(density),
                          contype='0' if is_visual else '1', conaffinity='0' if is_visual else '1',
                          rgba='.55 .65 .75 1' if is_visual else '.7 .3 .2 0')
    path = output / 'scene.xml'
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='unicode')
    return path


def validate(path, samples=21):
    """5 s passive settle; sampled collision sweep; separate PD endpoint reach test.

    This is a pilot: fixed root, scalar joints, no floor/support dynamics,
    finite sweep sampling, no semantic plausibility or calibrated actuation.
    """
    start = time.perf_counter()
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        return dict(compiles=False, error=str(exc), passed=False)
    data = mujoco.MjData(model)
    if any(int(t) not in (2, 3) for t in model.jnt_type):
        raise ValueError('Pilot validator only supports slide/hinge joints')
    # Rest at lower limits, rather than an illegal zero state for offset joints.
    data.qpos[:] = model.jnt_range[:, 0]
    mujoco.mj_forward(model, data)
    rest = data.qpos.copy()
    initial_positions = data.xipos.copy()
    max_body_drift = 0.
    max_joint_drift = np.zeros(model.njnt)
    max_penetration = 0.
    warning_counts = np.zeros(len(data.warning), dtype=int)

    def penetration():
        return max([0.] + [-float(c.dist) for c in data.contact])

    for _ in range(2500):
        mujoco.mj_step(model, data)
        max_joint_drift = np.maximum(max_joint_drift, abs(data.qpos-rest))
        max_penetration = max(max_penetration, penetration())
        max_body_drift = max(max_body_drift, float(np.max(np.linalg.norm(data.xipos-initial_positions, axis=1))))
        warning_counts = np.maximum(warning_counts, data.warning.number)
    mujoco.mj_forward(model, data)
    max_body_drift = max(max_body_drift, float(np.max(np.linalg.norm(data.xipos-initial_positions, axis=1))))
    drift_limits = np.where(model.jnt_type == int(mujoco.mjtJoint.mjJNT_SLIDE), .005, math.radians(.5))
    sweep = []
    clock_ok = bool(np.isclose(data.time, 5.))
    original_damping = model.dof_damping.copy()
    # Integrate derivative feedback implicitly; explicit damping forces destabilize
    # small-inertia parts at this timestep and would confound the geometry experiment.
    model.dof_damping[:] = 20
    for j in range(model.njnt):
        worst = 0.
        for value in np.linspace(*model.jnt_range[j], samples):
            data.qpos[:] = rest
            data.qpos[j] = value
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            worst = max(worst, penetration())
            warning_counts = np.maximum(warning_counts, data.warning.number)
        # Independent dynamic endpoints, drives all other joints to rest.
        errors = []
        for target in model.jnt_range[j]:
            mujoco.mj_resetData(model, data)
            data.qpos[:] = rest
            desired = rest.copy()
            desired[j] = target
            for _ in range(1500):
                mujoco.mj_forward(model, data)
                # Explicit idealized servo + gravity compensation, not a robot controller.
                data.qfrc_applied[:] = data.qfrc_bias + 100 * (desired-data.qpos)
                mujoco.mj_step(model, data)
            errors.append(float(abs(data.qpos[j]-target)))
            clock_ok = clock_ok and bool(np.isclose(data.time, 3.))
            warning_counts = np.maximum(warning_counts, data.warning.number)
        sweep.append(dict(joint=model.joint(j).name, max_penetration=worst,
                          endpoint_errors=errors, endpoint_tolerance=float(drift_limits[j])))
    warnings = warning_counts.tolist()
    model.dof_damping[:] = original_damping
    passed = bool(np.all(max_joint_drift <= drift_limits) and max_body_drift <= .005
                  and max_penetration <= .002 and not any(warnings) and clock_ok
                  and all(s['max_penetration'] <= .002 and max(s['endpoint_errors']) <= s['endpoint_tolerance'] for s in sweep)
                  and np.isfinite(data.qpos).all())
    return dict(compiles=True, passed=passed, joints=model.njnt, mass=float(model.body_mass.sum()),
                max_joint_drift=max_joint_drift.tolist(), body_drift=max_body_drift,
                settle_max_penetration=max_penetration, sweep=sweep, warnings=warnings, clock_ok=clock_ok,
                wall_seconds=time.perf_counter()-start,
                limitations=['fixed root; no support test', '21 sampled poses, not continuous collision proof',
                             'idealized servo with gravity compensation', 'density 500 kg/m3 is experimental'])


def box(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def part(link, name, mesh):
    return dict(link=link, name=name, mesh=mesh, collision_meshes=[mesh.convex_hull])
