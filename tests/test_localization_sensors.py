import json
import math

import mujoco
import numpy as np
import pytest

from scene_pipeline.localization_sensors import Sensors, validate_localization


def rig(joint='<joint axis="0 0 1"/>', config=None):
    wheels = [dict(id=name, drive_sign=(-1 if name.endswith('left') else 1))
              for name in ('front_left', 'front_right', 'rear_left', 'rear_right')]
    joints = ''.join(f'<body><joint name="element/{w["id"]}"/><geom size=".01"/></body>' for w in wheels)
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body name="chassis" pos="3 0 1">{joint}'
        f'<geom size=".1" pos="1 0 0"/></body>{joints}</worldbody></mujoco>')
    data = mujoco.MjData(model)
    return model, data, Sensors(model, dict(wheels=wheels), 1, config or {}, 77)


def test_parked_pitched_imu_encoders_and_fault_schedule():
    model, data, sensors = rig(config=dict(imu_pitch_offset_deg=[[.2, 1.5]], rtk=dict(
        noise_sigma_m=0, outages=[[.2, .4]], blips=[[.4, .6, .5]])))
    data.qvel[0] = 1.  # chassis yaw about its own origin with an offset center of mass
    data.qvel[1:] = [-2, 2, -3, 3]
    mujoco.mj_forward(model, data)
    sample = sensors.sample(data, 0)
    assert sample['wheels'] == dict(front_left=2, front_right=-2, rear_left=3, rear_right=-3)
    assert sample['accel'] == pytest.approx([0, 0, 1])
    assert sample['gyro'] == pytest.approx([0, 0, 1])
    assert sample['rtk_kinematics']['velocity'] == pytest.approx([0, 0, 0])
    assert sample['rtk_kinematics']['omega'] == pytest.approx([0, 0, 1])
    assert 'rtk_kinematics' not in sensors.sample(data, .19)
    fault = sensors.sample(data, .2 - 1e-12)
    angle = math.radians(1.5)
    np.testing.assert_allclose(fault['accel'], [-math.sin(angle), 0, math.cos(angle)], atol=1e-12)
    assert fault['attitude'] == pytest.approx([math.cos(angle/2), 0, math.sin(angle/2), 0])
    assert fault['rtk_kinematics']['status'] == 'NO_RTK'
    assert fault['rtk_kinematics']['v_acc'] == 8.
    fix = sensors.sample(data, .4)['rtk_kinematics']
    assert (fix['dz'], fix['status'], fix['v_acc']) == (.5, 'RTK_FIXED', .014)
    assert sensors.sample(data, .6)['rtk_kinematics']['dz'] == 0.
    json.dumps(sample, allow_nan=False)


def test_free_fall_specific_force_and_seeded_noise():
    samples = []
    for _ in range(2):
        model, data, sensors = rig(joint='<freejoint/>')
        mujoco.mj_forward(model, data)
        sample = sensors.sample(data, 0)
        assert sample['accel'] == pytest.approx([0, 0, 0], abs=1e-12)
        samples.append(sample['rtk_kinematics']['dz'])
    assert samples[0] == samples[1] != 0.


@pytest.mark.parametrize('bad', [
    {'chassis_pitch_deg': [[1, 1.5], [1, -1.5]]}, {'turnaround_pitch_deg': float('nan')},
    {'mode': 'estimated'}, {'imu_pitch_offset_deg': [[1, 0], [.9, 2]]},
    {'rtk': {'noise_sigma_m': -1}}, {'rtk': {'outages': [[1, 0]]}},
    {'rtk': {'blips': [[1, 2, float('inf')]]}}, {'params': {'disable_tilt': 'yes'}},
])
def test_invalid_configs(bad):
    with pytest.raises(ValueError, match='localization'):
        validate_localization(bad)
