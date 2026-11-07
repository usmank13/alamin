"""MuJoCo hardware adapter for Rhizome's host external-motor protocol.

Call server.poll(), adapter.apply(), then mj_step() on the simulation thread.
The socket endpoint identifies a device; (CAN interface, motor ID) identifies a
motor within it. Bindings use output-shaft radians and Nm, not rotor units.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import selectors
import socket
import time

import mujoco
import numpy as np


@dataclass(frozen=True)
class MotorBinding:
    joint: str
    sign: int = 1
    temperature_c: float = 25.0
    load_offset_rad: float = 0.0
    encoder_zeroed: bool = False


@dataclass
class _Motor:
    binding: MotorBinding
    dof: int
    qpos: int
    config: dict | None = None
    params: dict | None = None
    reference: dict | None = None
    enabled: bool = False
    zeroed: bool = False
    offset: float = 0.0
    integral: float = 0.0
    error: int = 0
    last_command: float = -math.inf


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite number')
    return float(value)


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f'Expected fields: {names}')
    return {k: _number(v) if k != 'mode' else v for k, v in value.items()}


class MujocoMotors:
    """Ideal motor electronics around measured physics, with a command watchdog.

    This models output-shaft controllers, not electrical/thermal dynamics or
    manufacturer firmware. Trapezoid trajectories and hardware provisioning are
    unsupported and fail explicitly. Existing servos on bound joints are disabled.
    """
    def __init__(self, model, data, bindings, *, watchdog_s=0.25, clock=time.monotonic):
        if _number(watchdog_s) <= 0:
            raise ValueError('watchdog_s must be positive')
        self.model, self.data = model, data
        self.watchdog_s, self.clock = watchdog_s, clock
        self.motors = {}
        for key, binding in bindings.items():
            if (len(key) != 2 or any(type(n) is not int or n < 0 for n in key)
                    or binding.sign not in (-1, 1)):
                raise ValueError('Bindings need nonnegative (interface, motor ID) and sign +/-1')
            _number(binding.temperature_c)
            _number(binding.load_offset_rad)
            joint = model.joint(binding.joint)
            if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError('Motor bindings require hinge joints')
            dof, qpos = int(joint.dofadr[0]), int(joint.qposadr[0])
            if any(m.dof == dof for m in self.motors.values()):
                raise ValueError('A joint cannot have multiple motor owners')
            self.motors[key] = _Motor(binding, dof, qpos, zeroed=binding.encoder_zeroed)
        for m in self.motors.values():
            jid = model.joint(m.binding.joint).id
            for aid in range(model.nu):
                if (model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                        and model.actuator_trnid[aid, 0] == jid):
                    model.actuator_gainprm[aid] = 0
                    model.actuator_biasprm[aid] = 0
            data.qfrc_applied[m.dof] = 0

    def stop(self, keys, error=1):
        for key in keys:
            m = self.motors[key]
            m.enabled, m.reference, m.error = False, None, error
            self.data.qfrc_applied[m.dof] = 0

    def _expire(self, key):
        m = self.motors[key]
        if m.enabled and self.clock() - m.last_command > self.watchdog_s:
            # ODrive CAN axis_error WATCHDOG_TIMER_EXPIRED; native MotorManager handles recovery.
            self.stop([key], error=0x01000000)

    def feedback(self, key):
        self._expire(key)
        m = self.motors[key]
        sign = m.binding.sign
        return dict(error_vec=m.error, temperature=m.binding.temperature_c,
                    position=sign * float(self.data.qpos[m.qpos]) + m.offset,
                    load_position=(sign * float(self.data.qpos[m.qpos]) + m.binding.load_offset_rad) % (2 * math.pi),
                    velocity=sign * float(self.data.qvel[m.dof]),
                    torque=sign * float(self.data.qfrc_applied[m.dof]))

    def handle(self, request):
        if type(request.get('version')) is not int or request['version'] != 1:
            raise ValueError('Unsupported external motor protocol version')
        key = (request['interface'], request['motor_id'])
        if any(type(n) is not int for n in key) or key not in self.motors:
            raise ValueError(f'Unmapped motor {key}')
        if request['controller'] != 'ODRIVE':
            raise ValueError('This adapter currently models ODRIVE output-shaft controls only')
        m = self.motors[key]
        config = request['motor_config']
        if not isinstance(config, dict) or _number(config['max_torque']) <= 0:
            raise ValueError('Motor config requires positive max_torque')
        if m.config is not None and m.config != config:
            raise ValueError('Motor configuration changed during the run')
        m.config = dict(config)
        self._expire(key)
        op, value = request['op'], request.get('value')
        if op == 'ping':
            return True
        if op in ('get_feedback', 'get_errors'):
            feedback = self.feedback(key)
            return feedback if op == 'get_feedback' else feedback['error_vec']
        if op == 'reset':
            self.stop([key])
            m.error, m.integral, m.params = 0, 0.0, None
        elif op == 'clear':
            m.error = 0
        elif op == 'enable':
            if m.error:
                raise ValueError('Clear motor errors before enabling')
            m.enabled, m.reference, m.last_command = True, None, self.clock()
        elif op == 'configure':
            params = _fields(value, ('mode', 'kp', 'ki', 'kd', 'max_integrator',
                                    'max_torque', 'ff_torque', 'position_target', 'velocity_target'))
            if params['mode'] not in ('IDLE', 'VELOCITY', 'POSITION', 'IMPEDANCE', 'TORQUE'):
                raise ValueError('Unsupported motor control mode')
            if any(params[k] < 0 for k in ('kp', 'ki', 'kd', 'max_integrator', 'max_torque')):
                raise ValueError('Motor gains and limits must be nonnegative')
            m.params, m.reference, m.integral = params, None, 0.0
            self.data.qfrc_applied[m.dof] = 0
        elif op == 'command':
            ref = _fields(value, ('mode', 'position_ref', 'velocity_ref', 'torque_ref'))
            if m.params is None or ref['mode'] != m.params['mode']:
                raise ValueError('Command mode must match configured mode')
            m.reference, m.last_command = ref, self.clock()
            return self.feedback(key)
        elif op == 'zeroed':
            return m.zeroed
        elif op == 'zero_encoder':
            m.offset -= _number(value)
            m.zeroed = True
        elif op == 'set_absolute_position':
            m.offset = _number(value) - m.binding.sign * float(self.data.qpos[m.qpos])
        elif op == 'unwind':
            m.integral = 0.0
        else:
            raise ValueError(f'Unsupported motor operation: {op}')
        return None

    def apply(self):
        """Apply bounded motor torque before each physics step; never set qpos/qvel."""
        for key, m in self.motors.items():
            self._expire(key)
            torque = 0.0
            if m.enabled and not m.error and m.params is not None and m.reference is not None:
                p, r = m.params, m.reference
                f = self.feedback(key)
                if p['mode'] in ('POSITION', 'VELOCITY'):
                    velocity = r['velocity_ref']
                    if p['mode'] == 'POSITION':
                        velocity += p['kp'] * (r['position_ref'] - f['position'])
                    error = velocity - f['velocity']
                    m.integral = float(np.clip(m.integral + p['ki'] * error * self.model.opt.timestep,
                                               -p['max_integrator'], p['max_integrator']))
                    torque = (p['kd'] if p['mode'] == 'POSITION' else p['kp']) * error + m.integral
                elif p['mode'] == 'IMPEDANCE':
                    torque = p['kp'] * (r['position_ref'] - f['position']) + p['kd'] * (r['velocity_ref'] - f['velocity'])
                if p['mode'] != 'IDLE':
                    # The ODrive backend forwards command torque only; config ff_torque is MAB-specific.
                    torque += r['torque_ref']
                limit = min(p['max_torque'], m.config['max_torque'])
                torque = float(np.clip(torque, -limit, limit))
            self.data.qfrc_applied[m.dof] = m.binding.sign * torque


class MotorSocketServer:
    """Single-threaded Unix packet server; poll on the physics thread."""
    def __init__(self, path, adapter):
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError('Motor socket path must be absolute')
        self.adapter = adapter
        self.selector = selectors.DefaultSelector()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.owners = {}
        self.clients = {}
        try:
            # Never unlink an existing endpoint: it may belong to another run.
            self.listener.bind(str(self.path))
            self.path.chmod(0o600)
            self.listener.listen()
            self.listener.setblocking(False)
            self.selector.register(self.listener, selectors.EVENT_READ)
        except Exception:
            self.listener.close()
            self.selector.close()
            raise

    def _drop(self, client):
        keys = self.clients.pop(client)
        self.adapter.stop(keys)
        for key in keys:
            self.owners.pop(key, None)
        self.selector.unregister(client)
        client.close()

    def poll(self, timeout=0):
        for event, _ in self.selector.select(timeout):
            client = event.fileobj
            if client is self.listener:
                client, _ = self.listener.accept()
                client.setblocking(False)
                self.clients[client] = set()
                self.selector.register(client, selectors.EVENT_READ)
                continue
            try:
                packet, _, flags, _ = client.recvmsg(65536)
                if not packet or flags & socket.MSG_TRUNC:
                    self._drop(client)
                    continue
                try:
                    request = json.loads(packet)
                    if not isinstance(request, dict):
                        raise ValueError('Request must be an object')
                    key = (request.get('interface'), request.get('motor_id'))
                    if any(type(n) is not int for n in key) or key not in self.adapter.motors:
                        raise ValueError('Unmapped motor')
                    owner = self.owners.get(key)
                    if owner is not None and owner is not client:
                        raise ValueError('Motor is owned by another connection')
                    self.owners[key] = client
                    self.clients[client].add(key)
                    value = self.adapter.handle(request)
                    response = dict(version=1, ok=True, value=value)
                except (ValueError, KeyError, TypeError) as error:
                    self.adapter.stop(self.clients[client])
                    response = dict(version=1, ok=False, error=str(error))
                client.send(json.dumps(response, allow_nan=False).encode())
            except (OSError, ValueError):
                self._drop(client)

    def close(self):
        for client in list(self.clients):
            self._drop(client)
        self.selector.close()
        self.listener.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
