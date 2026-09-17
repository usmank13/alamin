import mujoco
import numpy as np
import pytest

from sim_harness.ground_truth import snapshot, contacts


def test_live_pose_rotated_bounds_and_free_joint_dimensions():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="box" pos="1 2 3" quat="0.70710678 0 0 0.70710678">
        <freejoint name="floating"/><geom type="box" size=".1 .2 .3" mass="1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    record = snapshot(model, data)
    assert record['geoms'][0]['world_aabb'] is None
    np.testing.assert_allclose(record['geoms'][1]['world_aabb']['min'], [.8, 1.9, 2.7])
    np.testing.assert_allclose(record['geoms'][1]['world_aabb']['max'], [1.2, 2.1, 3.3])
    assert len(record['joints'][0]['qpos']) == 7
    assert len(record['joints'][0]['qvel']) == 6
    mujoco.mj_step(model, data, nstep=100)
    mujoco.mj_forward(model, data)
    record = snapshot(model, data)
    assert record['time'] == pytest.approx(.2)
    assert record['bodies'][1]['position'][2] < 3
    assert contacts(data) == []
