"""The production skin swaps the robot's visuals and nothing that physics or a recorded qpos depends on."""
import os

import mujoco
import numpy as np
import pytest

from scene_pipeline.robot_skin import MATERIALS, reskin


def test_reskin_keeps_joints_and_replaces_visuals():
    if not os.environ.get('RHIZOME_ROOT'):
        pytest.skip('Set RHIZOME_ROOT for the production robot meshes')
    spec = mujoco.MjSpec()
    corners = np.array([[x, y, z] for x in (-.1, .1) for y in (-.05, .05) for z in (-.1, .1)], float)
    spec.add_mesh(name='block', uservert=corners.ravel())

    def body(parent, name, pos, axis=None):
        b = parent.add_body(name=name, pos=pos)
        if axis:
            b.add_joint(name=name, axis=axis)
        b.add_geom(name=name + '_visual', type=mujoco.mjtGeom.mjGEOM_MESH, meshname='block', contype=0, conaffinity=0, mass=0)
        b.add_geom(name=name + '_collision', size=[.05, 0, 0], mass=1)
        return b

    chassis = body(spec.worldbody, 'chassis', [0, 0, 1])
    chassis.add_freejoint()
    for side, y in (('right', -.71), ('left', .71)):
        rocker = body(chassis, 'rocker_' + side, [0, y, 0], [0, 1, 0])
        for end, x in (('front', .4), ('rear', -.4)):
            body(rocker, f'{end}_{side}', [x, y * .15, -.7], [0, 1, 0])
    for i, y in enumerate((.62, .14, -.14, -.62), start=1):
        yaw = body(chassis, f'arm_{i}_yaw', [-.345, y, -.37], [0, 0, 1])
        body(yaw, f'arm_{i}_pitch', [0, .035, -.08], [0, 1, 0])
    before = spec.compile()
    after = reskin(spec, os.environ['RHIZOME_ROOT']).compile()
    assert (after.nq, after.nbody, after.njnt) == (before.nq, before.nbody, before.njnt)
    np.testing.assert_allclose(after.body_mass, before.body_mass)
    names = [after.geom(g).name for g in range(after.ngeom)]
    assert not any(n.endswith('_visual') for n in names) and sum(n.startswith('skin_') for n in names) > 20
    assert {after.material(m).name for m in range(after.nmat)} == {'robot_' + m for m in MATERIALS}
    assert not any(after.geom_contype[g] for g in range(after.ngeom) if names[g].startswith('skin_'))
