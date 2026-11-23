"""MuJoCo localization sensors and physical rocker pitch on the 100 Hz control clock."""
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import PipelineError


def _invalid(reason):
    raise PipelineError('RHIZOME_CONFIG', 'localization.' + reason)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _rows(rows, width, key):
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) != width or not all(map(_finite, row))
        or row[0] < 0 for row in rows
    ):
        _invalid(f'{key} requires finite {width}-element rows starting at a nonnegative time')


def pitch_at(schedule, t):
    if isinstance(schedule, list):
        return next((value for start, value in reversed(schedule) if start <= t), 0.)
    return schedule


def validate_localization(config):
    if not isinstance(config, dict) or set(config) - {'mode', 'params', 'imu_pitch_offset_deg', 'chassis_pitch_deg', 'turnaround_pitch_deg', 'reverse_pitch_deg', 'rtk'}:
        _invalid('config contains unknown keys or is not an object')
    mode = config.get('mode', 'truth')
    if mode not in ('truth', 'odometry'):
        _invalid('mode must be truth or odometry')
    pitches = {}
    for key in ('imu_pitch_offset_deg', 'chassis_pitch_deg'):
        pitch = config.get(key, 0.)
        if not _finite(pitch):
            _rows(pitch, 2, key)
            if any(b[0] <= a[0] for a, b in zip(pitch, pitch[1:])):
                _invalid(f'{key} step times must increase')
        pitches[key] = pitch
    if 'turnaround_pitch_deg' in config and 'reverse_pitch_deg' in config:
        _invalid('choose one of reverse_pitch_deg and turnaround_pitch_deg')
    for key in ('turnaround_pitch_deg', 'reverse_pitch_deg'):
        if key in config:
            if not _finite(config[key]):
                _invalid(f'{key} must be finite')
            pitches[key] = config[key]
    params = config.get('params', {})
    if not isinstance(params, dict) or any(
        not isinstance(key, str) or not (isinstance(value, bool) or _finite(value))
        for key, value in params.items()
    ):
        _invalid('params requires finite numbers or booleans keyed by odometry parameter')
    rtk = config.get('rtk', {})
    if not isinstance(rtk, dict) or set(rtk) - {'noise_sigma_m', 'v_acc', 'outages', 'blips'}:
        _invalid('rtk contains unknown keys or is not an object')
    rtk = {'noise_sigma_m': .007, 'v_acc': .014, 'outages': [], 'blips': [], **rtk}
    for key in ('noise_sigma_m', 'v_acc'):
        if not _finite(rtk[key]) or rtk[key] < 0:
            _invalid(f'rtk.{key} must be finite and nonnegative')
    for key, width in (('outages', 2), ('blips', 3)):
        _rows(rtk[key], width, 'rtk.' + key)
        if any(end <= start for start, end, *_ in rtk[key]):
            _invalid(f'rtk.{key} windows require end > start')
    return dict(mode=mode, params=params, rtk=rtk, **pitches)


class Sensors:
    def __init__(self, model, profile, chassis, config, seed):
        self.model = model
        self.chassis = chassis
        self.config = validate_localization(config)
        self.rng = np.random.default_rng(seed)
        self.wheels = {}
        for wheel in profile['wheels']:
            name = wheel['id']
            sign = wheel.get('drive_sign', 1) * (-1 if name.endswith('_right') else 1)
            self.wheels[name] = (int(model.joint('element/' + name).dofadr[0]), sign)
        if set(self.wheels) != {'front_left', 'front_right', 'rear_left', 'rear_right'}:
            raise ValueError('Odometry requires four named chassis wheel joints')

    def sample(self, data, t):
        tick = round(t * 100)
        t = tick / 100  # Compare scheduled boundaries on the exact control clock.
        pitch = pitch_at(self.config['imu_pitch_offset_deg'], t)
        orientation = Rotation.from_quat(data.xquat[self.chassis], scalar_first=True)
        reported = orientation * Rotation.from_euler('y', pitch, degrees=True)
        velocity, acceleration = np.empty(6), np.empty(6)
        # XBODY is the chassis frame origin. BODY is the inertial center of mass.
        origin = mujoco.mjtObj.mjOBJ_XBODY
        mujoco.mj_objectVelocity(self.model, data, origin, self.chassis, velocity, 0)
        mujoco.mj_rnePostConstraint(self.model, data)
        mujoco.mj_objectAcceleration(self.model, data, origin, self.chassis, acceleration, 0)
        # rnePostConstraint includes -gravity in cacc. Bias only the reported gravity.
        gravity = self.model.opt.gravity
        accel = orientation.inv().apply(acceleration[3:] + gravity) - reported.inv().apply(gravity)
        result = dict(
            wheels={name: float(data.qvel[dof] * sign) for name, (dof, sign) in self.wheels.items()},
            accel=(accel / 9.81).tolist(),
            gyro=orientation.inv().apply(velocity[:3]).tolist(),
            attitude=reported.as_quat(scalar_first=True).tolist(),
        )
        if tick % 20 == 0:
            rtk = self.config['rtk']
            outage = any(start <= t < end for start, end in rtk['outages'])
            blip = sum(dz for start, end, dz in rtk['blips'] if start <= t < end)
            result['rtk_kinematics'] = dict(
                position=data.xpos[self.chassis].tolist(), rotation=data.xquat[self.chassis].tolist(),
                velocity=velocity[3:].tolist(), omega=velocity[:3].tolist(),
                dz=blip + self.rng.normal(0, rtk['noise_sigma_m']),
                v_acc=8. if outage else rtk['v_acc'], status='NO_RTK' if outage else 'RTK_FIXED',
            )
        return result


def rocker_gravity_trim(model, data, chassis, joints):
    """Sprung first moment in chassis axes times gravity / combined pitch stiffness."""
    relative = (data.subtree_com - data.xanchor[joints[0]]) @ data.xmat[chassis].reshape(3, 3)
    moment = model.body_subtreemass[:, None] * relative
    sprung = moment[chassis] - moment[model.jnt_bodyid[joints]].sum(axis=0)
    return model.opt.gravity[2] * sprung / sum(model.jnt_stiffness[joints])


class ChassisPitch:
    """Offset the opposed rocker spring references relative to their level-course trim."""
    def __init__(self, model, data, profile, chassis):
        suspension = profile.get('suspension', [])
        if len(suspension) != 2 or any('springref_rad' not in r for r in suspension):
            raise ValueError('Physical chassis pitch requires a balanced two-rocker course')
        joints = [model.joint('element/' + r['joint']).id for r in suspension]
        self.model = model
        self.addresses = model.jnt_qposadr[joints]
        self.base = model.qpos_spring[self.addresses].copy()
        rotation = data.xmat[chassis].reshape(3, 3)
        self.sign = (data.xaxis[joints] @ rotation)[:, 1]
        # Gravity changes the sprung body's pitch moment as its COM rotates about the pivots.
        stowed = mujoco.MjData(model)
        stowed.qpos[:] = data.qpos
        for arm in profile.get('arms', []):
            for key in ('pitch_joint', 'yaw_joint'):
                stowed.joint('element/' + arm[key]).qpos[0] = 0.
        mujoco.mj_forward(model, stowed)
        self.gravity_trim = rocker_gravity_trim(model, stowed, chassis, joints)
        self.limits = model.jnt_range[joints]
        self.home = np.array([r['home_rad'] for r in suspension])
        self.target = self.start = self.applied = self.changed = 0.

    def apply(self, t, degrees):
        if degrees != self.target:
            nominal = self.home - self.sign * math.radians(degrees)
            if np.any(nominal < self.limits[:, 0]) or np.any(nominal > self.limits[:, 1]):
                raise ValueError(f'Chassis pitch {degrees} degrees exceeds rocker travel')
            self.start, self.target, self.changed = self.applied, degrees, t
        phase = min(1., (t - self.changed) / .45)
        self.applied = self.start + (self.target - self.start) * (.5 - .5 * math.cos(math.pi * phase))
        angle = math.radians(self.applied)
        trim = self.gravity_trim[0] * (math.cos(angle) - 1.) + self.gravity_trim[2] * math.sin(angle)
        self.model.qpos_spring[self.addresses] = self.base - self.sign * (angle + trim)
