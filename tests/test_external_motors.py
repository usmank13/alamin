import json
import socket

import mujoco
import numpy as np
import pytest

from scene_pipeline.external_motors import MotorBinding, MotorSocketServer, MujocoMotors


@pytest.fixture
def rig():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0" timestep="0.002"/>
      <worldbody><body><joint name="wheel" axis="0 0 1"/><geom type="sphere" size=".1" mass="1"/></body>
      <body pos="1 0 0"><joint name="arm" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 .4" size=".02" mass="1"/></body></worldbody>
      <actuator><velocity joint="wheel" kv="100"/><position joint="arm" kp="100"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    now = [0.0]
    adapter = MujocoMotors(model, data, {(0, 12): MotorBinding('wheel'), (1, 21): MotorBinding('arm', -1)}, clock=lambda: now[0])
    return model, data, adapter, now


def request(op, value=None, key=(0, 12)):
    return dict(version=1, op=op, interface=key[0], motor_id=key[1], controller='ODRIVE',
                motor_config=dict(max_torque=2), value=value)


def configure(mode):
    return dict(mode=mode, kp=5, ki=0, kd=1, max_integrator=1, max_torque=1,
                ff_torque=0, position_target=0, velocity_target=0)


def ref(mode, position=0, velocity=0):
    return dict(mode=mode, position_ref=position, velocity_ref=velocity, torque_ref=0)


def start(adapter, key, mode, position=0, velocity=0):
    adapter.handle(request('configure', configure(mode), key))
    adapter.handle(request('enable', key=key))
    adapter.handle(request('command', ref(mode, position, velocity), key))


def test_wheel_and_arm_move_only_via_physics_and_obey_limits(rig):
    model, data, adapter, now = rig
    start(adapter, (0, 12), 'VELOCITY', velocity=2)
    start(adapter, (1, 21), 'POSITION', position=.4)
    np.testing.assert_array_equal(data.qpos, 0)
    for _ in range(100):
        adapter.apply()
        assert np.max(abs(data.qfrc_applied)) <= 1
        mujoco.mj_step(model, data)
    assert data.qpos[0] > .1
    assert data.qpos[1] < -.02
    feedback = adapter.handle(request('get_feedback', key=(1, 21)))
    assert feedback['position'] == -data.qpos[1]
    assert feedback['velocity'] == -data.qvel[1]
    assert feedback['torque'] == -data.qfrc_applied[1]
    # A fresh measurement does not extend the command lease.
    now[0] = 1
    assert adapter.handle(request('get_feedback'))['error_vec'] != 0
    adapter.apply()
    np.testing.assert_array_equal(data.qfrc_applied, 0)
    with pytest.raises(ValueError, match='Clear'):
        adapter.handle(request('enable'))


def test_encoder_operations_do_not_teleport_physics(rig):
    _, data, adapter, _ = rig
    data.qpos[1] = -.3
    key = (1, 21)
    adapter.handle(request('set_absolute_position', .8, key))
    assert adapter.feedback(key)['position'] == pytest.approx(.8)
    adapter.handle(request('zero_encoder', .2, key))
    assert adapter.feedback(key)['position'] == pytest.approx(.6)
    assert adapter.handle(request('zeroed', key=key))
    assert data.qpos[1] == -.3
    adapter.handle(request('reset', key=key))
    assert adapter.handle(request('zeroed', key=key))


def test_invalid_modes_values_and_unmapped_motors_fail(rig):
    _, _, adapter, _ = rig
    for req in (request('configure', configure('POSITION_TRAPEZOID')),
                request('configure', dict(configure('POSITION'), kp=float('nan'))),
                request('calibrate_encoder'), request('provision'), request('ping', key=(1, 99))):
        with pytest.raises(ValueError):
            adapter.handle(req)


def exchange(server, client, payload):
    client.send(json.dumps(payload).encode())
    server.poll(.1)
    return json.loads(client.recv(65536))


def connect(server):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    client.settimeout(1)
    client.connect(str(server.path))
    server.poll(.1)
    return client


def test_socket_disconnect_and_invalid_command_stop_motors(rig, tmp_path):
    _, data, adapter, _ = rig
    with MotorSocketServer(tmp_path / 'motors.sock', adapter) as server:
        with connect(server) as client:
            assert exchange(server, client, request('configure', configure('VELOCITY')))['ok']
            assert exchange(server, client, request('enable'))['ok']
            assert exchange(server, client, request('command', ref('VELOCITY', velocity=2)))['ok']
            adapter.apply()
            assert data.qfrc_applied[0] > 0
            assert not exchange(server, client, request('command', ref('TORQUE')))['ok']
            assert data.qfrc_applied[0] == 0
            assert exchange(server, client, request('clear'))['ok']
            assert exchange(server, client, request('enable'))['ok']
            assert exchange(server, client, request('command', ref('VELOCITY', velocity=2)))['ok']
            adapter.apply()
            assert data.qfrc_applied[0] > 0
        server.poll(.1)
        assert data.qfrc_applied[0] == 0
        assert adapter.feedback((0, 12))['error_vec'] != 0
    assert not (tmp_path / 'motors.sock').exists()


def test_second_client_cannot_take_motor_or_delete_endpoint(rig, tmp_path):
    _, _, adapter, _ = rig
    path = tmp_path / 'motors.sock'
    with MotorSocketServer(path, adapter) as server:
        with pytest.raises(OSError):
            MotorSocketServer(path, adapter)
        assert path.exists()
        with connect(server) as first, connect(server) as second:
            assert exchange(server, first, request('ping'))['ok']
            assert not exchange(server, second, request('ping'))['ok']
            assert exchange(server, first, request('ping'))['ok']


def test_missing_or_malformed_packet_returns_error(rig, tmp_path):
    _, _, adapter, _ = rig
    with MotorSocketServer(tmp_path / 'motors.sock', adapter) as server:
        with connect(server) as client:
            for payload in ([], {}, dict(request('ping'), interface=True), dict(request('ping'), version=99), dict(request('ping'), version=True)):
                assert not exchange(server, client, payload)['ok']


def test_native_motor_manager_drives_mujoco(rig, tmp_path):
    """Compile against the selected checkout; no Python imitation of the client."""
    import os
    from pathlib import Path
    import subprocess
    import time

    root = os.environ.get('RHIZOME_ROOT')
    if not root:
        pytest.skip('Set RHIZOME_ROOT and RHIZOME_HOST_BUILD to test the native motor backend')
    root = Path(root)
    host = Path(os.environ.get('RHIZOME_HOST_BUILD', root / 'build/host'))
    client = tmp_path / 'motor-client'
    command = ['g++', '-std=c++17', '-O1', '-DHOST_BUILD', '-pthread']
    for inc in ('include', 'src', 'external/include', 'external/rapidyaml/include', 'external/nanobind/include'):
        command += ['-I' + str(root / inc)]
    command += [str(Path(__file__).parent / 'fixtures/external_motor_client.cpp'),
                str(root / 'src/base_control/external_motors.cpp'), str(root / 'src/base_control/motor_iface.cpp'),
                '-L' + str(host / 'lib'), '-lrhizome', '-lrhizome-comms', '-lzmq', '-lsystemd', '-o', str(client)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    model, data, adapter, _ = rig
    # The native process and command watchdog use wall time; physics advances on
    # this thread, independently of commands arriving from the client.
    adapter.clock = time.monotonic
    with MotorSocketServer(tmp_path / 'native.sock', adapter) as server:
        with subprocess.Popen([str(client)], env={**os.environ, 'AIGEN_MOTOR_SOCKET': str(server.path)},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            deadline = time.monotonic() + 10
            while proc.poll() is None and time.monotonic() < deadline:
                server.poll(.001)
                adapter.apply()
                mujoco.mj_step(model, data)
            if proc.poll() is None:
                proc.kill()
            stdout, stderr = proc.communicate(timeout=2)
            assert proc.returncode == 0, stdout + stderr
        server.poll(.1)
        np.testing.assert_array_equal(data.qfrc_applied, 0)
    feedback = json.loads(stdout.strip().splitlines()[-1])
    assert feedback['wheel']['load_position'] > .1
    assert feedback['arm']['load_position'] > .02
    assert feedback['wheel']['position'] > .1
    assert feedback['arm']['position'] > .02
    assert data.qpos[0] > .1 and data.qpos[1] < -.02
    assert feedback['wheel']['error_vec'] == feedback['arm']['error_vec'] == 0


def test_absolute_load_ring_is_independent_of_motor_zero(rig):
    model,data,_,now=rig
    adapter=MujocoMotors(model,data,{(0,21):MotorBinding('arm',-1,load_offset_rad=3.2,encoder_zeroed=True)},clock=lambda:now[0])
    key=(0,21)
    data.qpos[1]=-.4
    before=adapter.feedback(key)
    assert before['position']==pytest.approx(.4)
    assert before['load_position']==pytest.approx(3.6)
    assert adapter.handle(request('zeroed',key=key)) is True
    adapter.handle(request('zero_encoder',.2,key))
    assert adapter.feedback(key)['position']==pytest.approx(.2)
    assert adapter.feedback(key)['load_position']==pytest.approx(3.6)
    assert data.qpos[1]==-.4
    adapter.handle(request('reset',key=key))
    assert adapter.handle(request('zeroed',key=key)) is True
    assert adapter.feedback(key)['position']==pytest.approx(.2)


def test_odrive_does_not_apply_mab_config_feedforward(rig):
    _,data,adapter,_=rig
    params=configure('POSITION');params['ff_torque']=-2
    adapter.handle(request('configure',params))
    adapter.handle(request('enable'))
    adapter.handle(request('command',ref('POSITION')))
    adapter.apply()
    assert data.qfrc_applied[0]==0


def test_watchdog_uses_odrive_error_for_native_recovery(rig):
    _,_,adapter,now=rig
    start(adapter,(0,12),'VELOCITY',velocity=1)
    now[0]=.3
    assert adapter.feedback((0,12))['error_vec']==0x01000000
