#!/usr/bin/env python3
"""Build three matching MuJoCo localization courses from a compiled Element robot."""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from scene_pipeline.agriculture import compile_environment, compose_robot, sha256
from scene_pipeline.localization_sensors import rocker_gravity_trim
from scene_pipeline.contracts import read_json, write_json
from sim_harness.export import export_mujoco
from sim_harness.scene import _read_spec


REPO = Path(__file__).resolve().parents[1]
COURSES = ('flat', 'rolling', 'hazards')
BOUNDS = [[-8., -6.], [56., 6.]]


def balance_rockers(robot, profile):
    """Preload the Gen3 rocker springs for level equilibrium with the arms stowed."""
    suspension = profile.get('suspension', [])
    if not suspension:
        return
    model = robot.compile()
    data = mujoco.MjData(model)
    for name, position in profile.get('initial_joint_positions', {}).items():
        data.joint(name).qpos[0] = position
    for arm in profile.get('arms', []):
        for key in ('pitch_joint', 'yaw_joint'):
            data.joint(arm[key]).qpos[0] = 0.
    mujoco.mj_forward(model, data)
    joints = [model.joint(rocker['joint']).id for rocker in suspension]
    trim = rocker_gravity_trim(model, data, model.body('chassis').id, joints)[0]
    for rocker, joint_id in zip(suspension, joints):
        joint = robot.joint(rocker['joint'])
        joint.springref -= trim * data.xaxis[joint_id, 1]
        rocker['springref_rad'] = joint.springref


def build(source, output, course):
    robot = _read_spec(source / 'robot.mjz')
    profile = read_json(source / 'robot_profile.json')
    balance_rockers(robot, profile)
    output.mkdir(parents=True, exist_ok=False)
    x, y = np.meshgrid(np.linspace(-8., 56., 641), np.linspace(-6., 6., 121))
    z = np.zeros_like(x)
    if course == 'rolling':
        # A level start, then a smooth 3.1% maximum grade.
        z = .12 * (1 - np.cos(2 * np.pi * x / 24))
    elif course == 'hazards':
        # Opposite sides of the same wheel corridor, visible on the same approach.
        ditch = (abs(x - 6.) <= .55) & (abs(y + .50) <= .40)
        z[ditch] = -.40
    np.savez_compressed(output / 'terrain.npz', elevation_m=z)
    environment, manifest = compile_environment(dict(
        terrain=dict(file='terrain.npz', bounds_m=BOUNDS), instances=[], rows=[],
        class_names={'0': 'terrain', '4': 'element', '9': 'obstruction'}), output)
    if course == 'hazards':
        environment.worldbody.add_geom(name='block', type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[6., .5, .20], size=[.35, .32, .20], rgba=[.45, .42, .40, 1])
        manifest['instances']['block'] = dict(id='block', category='obstruction', position_m=[6., .5, .2])
        manifest['geom_labels']['block'] = dict(instance='block', class_id=9)
    spawn = dict(position_m=[0., 0.], yaw_rad=0.)
    spec, _, data = compose_robot(environment, robot, profile, spawn, BOUNDS)
    export_mujoco(spec, data, output / 'scene.mjz')
    environment.to_zip(str(output / 'environment.mjz'))
    robot.to_zip(str(output / 'robot.mjz'))
    write_json(output / 'robot_profile.json', profile)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'generation.json', dict(domain='agriculture', spawn=spawn,
        provenance=dict(course=course, source_robot_sha256=sha256(source / 'robot.mjz'),
                        robot_sha256=sha256(output / 'robot.mjz'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO / 'outputs/soybean30_gen3')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/localization_courses_balanced',
                        help='Fresh directory for the flat, rolling and hazards courses')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for course in COURSES:
        build(args.source, args.output / course, course)
        print(args.output / course)


if __name__ == '__main__':
    main()
