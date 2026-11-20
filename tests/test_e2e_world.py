import json

import mujoco
import numpy as np
import pytest

from scene_pipeline.rhizome_e2e_world import PlotWorld
from sim_harness.export import export_mujoco


def test_plot_world_measures_joints_and_scores_tool_geometry(tmp_path):
    spec = mujoco.MjSpec.from_string("""
        <mujoco>
          <compiler angle="radian"/>
          <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
          <default><joint armature=".01"/></default>
          <worldbody>
            <body pos="2 0 0"><joint name="element/wheel"/><geom type="sphere" size=".05"/></body>
            <body pos="0 0 .025">
              <joint name="element/yaw"/>
              <geom type="sphere" size=".02"/>
              <body>
                <joint name="element/pitch" axis="0 1 0"/>
                <geom name="element/pitch_tool_collision" type="box" pos=".18 0 0" size=".02 .01 .01"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <velocity name="element/wheel" joint="element/wheel" kv="1"/>
            <position name="element/yaw" joint="element/yaw" kp="1" kv="0.1" ctrlrange="-.5 .5"/>
            <position name="element/pitch" joint="element/pitch" kp="1" kv="0.1" ctrlrange="-.5 .5"/>
          </actuator>
        </mujoco>""")
    export_mujoco(spec, mujoco.MjData(spec.compile()), tmp_path / 'scene.mjz')
    profile = dict(max_forward_mps=1, max_yaw_rate_rps=1,
                   wheels=[dict(id='wheel', radius_m=.05, center_m=[0, 0, 0])],
                   arms=[dict(id='arm', yaw_joint='yaw', pitch_joint='pitch',
                              motor_to_joint_sign=[-1, 1])])
    (tmp_path / 'robot_profile.json').write_text(json.dumps(profile))
    plants = dict(weed=dict(category='weed', position_m=[.18, 0, 0]),
                  crop=dict(category='crop', position_m=[0, .5, 0]))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(instances=plants)))

    world = PlotWorld(tmp_path)
    contact = world.contact_report()
    assert contact['first_contact_s_by_arm'] == {'arm': {'weed': 0}}
    assert contact['unique_weed_contacts'] == ['weed']
    assert contact['unique_crop_contacts'] == []

    with pytest.raises(ValueError, match='integral'):
        world.advance(.003, forward_mps=0, yaw_rate_rps=0, commands={})
    assert world.data.time == 0
    world.advance(.05, forward_mps=.1, yaw_rate_rps=0, commands={'arm': {'yaw': .2, 'pitch': 0}})
    assert world.data.time == pytest.approx(.05)
    assert world.data.ctrl[world.wheels[0]] == pytest.approx(2)
    assert world.data.ctrl[world.actuators['arm'][0]] == pytest.approx(-.2)
    feedback = world.feedback()['arm']
    assert 0 < feedback['yaw'] < .2
    assert feedback['yaw_velocity'] > 0
    assert not np.any(world.contact_report()['warnings'])

    extents = dict(yaw=dict(min=-.9, max=.75), pitch=dict(min=-.1, max=1.4))
    limits = world.bind_arm_extents({'1': {'arm_0': {'config': {'extents': extents}}}})
    assert limits['arm'] == dict(payload=1, slot=0, motor_rad={'yaw': [-.9, .75], 'pitch': [-.1, 1.4]})
    assert list(world.model.joint('element/yaw').range) == [-.75, .9]
    assert list(world.model.actuator('element/yaw').ctrlrange) == [-.75, .9]
    assert list(world.model.joint('element/pitch').range) == [-.1, 1.4]
    assert list(world.model.actuator('element/pitch').ctrlrange) == [-.1, 1.4]
    applied = world.advance(.002, forward_mps=0, yaw_rate_rps=0,
                            commands={'arm': {'yaw': .7, 'pitch': 1.6}})
    assert applied == {'arm': {'yaw': .7, 'pitch': 1.4}}
    assert world.data.ctrl[world.actuators['arm'][0]] == -.7
