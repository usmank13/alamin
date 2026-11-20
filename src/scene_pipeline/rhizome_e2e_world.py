"""MuJoCo feedback and independent plant contact scoring for native e2e runs."""
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from sim_harness.scene import load_scene
from .agriculture_drive import wheel_targets
from .weeding_field import field_plants, tool_contacts


class PlotWorld:
    def __init__(self, scene):
        scene = Path(scene)
        self.model, self.data = load_scene(scene / 'scene.mjz')
        self.profile = json.loads((scene / 'robot_profile.json').read_text())
        self.arms = {arm['id']: arm for arm in self.profile['arms']}
        self.wheels = [self.model.actuator('element/' + w['id']).id for w in self.profile['wheels']]
        self.joints = {}
        self.actuators = {}
        self.tools = {}
        for name, arm in self.arms.items():
            joints = ['element/' + arm[axis + '_joint'] for axis in ('yaw', 'pitch')]
            self.joints[name] = [self.data.joint(j) for j in joints]
            self.actuators[name] = [self.model.actuator(j).id for j in joints]
            self.tools[name] = self.model.geom(joints[1] + '_tool_collision').id
        self.plants = field_plants(json.loads((scene / 'manifest.json').read_text()))
        self.tree = cKDTree([p['position_m'][:2] for p in self.plants]) if self.plants else None
        self.previous = {}
        self.contacts = {name: {} for name in self.arms}
        self._score()

    def bind_arm_extents(self, payloads):
        """Use the selected Rhizome arm extents for both joint stops and position servos."""
        limits = {}
        for i, (name, arm) in enumerate(self.arms.items()):
            entry = payloads[str(i // 2 + 1)][f'arm_{i % 2}']
            bounds = {}
            for axis, sign, joint, aid in zip(('yaw', 'pitch'), arm['motor_to_joint_sign'],
                                               self.joints[name], self.actuators[name]):
                extent = entry['config']['extents'][axis]
                low, high = extent['min'], extent['max']
                if sign not in (-1, 1) or not np.isfinite([low, high]).all() or low >= high:
                    raise ValueError(f'Invalid resolved motor limits or joint sign for {name}/{axis}')
                bounds[axis] = [low, high]
                signed = sorted((low * sign, high * sign))
                self.model.jnt_range[joint.id] = signed
                self.model.jnt_limited[joint.id] = True
                self.model.actuator_ctrlrange[aid] = signed
                self.model.actuator_ctrllimited[aid] = True
            limits[name] = dict(payload=i // 2 + 1, slot=i % 2, motor_rad=bounds)
        mujoco.mj_forward(self.model, self.data)
        return limits

    def feedback(self):
        result = {}
        for name, arm in self.arms.items():
            measured = {}
            for axis, joint, sign in zip(('yaw', 'pitch'), self.joints[name], arm['motor_to_joint_sign']):
                measured[axis] = float(sign * joint.qpos[0])
                measured[axis + '_velocity'] = float(sign * joint.qvel[0])
            result[name] = measured
        return result

    def advance(self, seconds, *, forward_mps, yaw_rate_rps, commands):
        dt = self.model.opt.timestep
        steps = round(seconds / dt)
        if steps < 1 or not math.isclose(seconds, steps * dt, abs_tol=1e-10):
            raise ValueError('Duration must be a positive integral number of physics steps')
        if not np.isfinite([forward_mps, yaw_rate_rps]).all() or commands.keys() != self.arms.keys():
            raise ValueError('Expected finite wheel speeds and commands for each arm')
        target = self.data.ctrl.copy()
        target[self.wheels], _ = wheel_targets(self.profile, forward_mps, yaw_rate_rps)
        for name, command in commands.items():
            angles = np.array([command['yaw'], command['pitch']])
            if angles.shape != (2,) or not np.isfinite(angles).all():
                raise ValueError(f'Invalid position command for {name}')
            actuators = self.actuators[name]
            limits = self.model.actuator_ctrlrange[actuators]
            target[actuators] = np.clip(angles * self.arms[name]['motor_to_joint_sign'], *limits.T)
        self.data.ctrl[:] = target
        start = self.data.time
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_kinematics(self.model, self.data)
            self._score()
        mujoco.mj_forward(self.model, self.data)
        if (not math.isclose(self.data.time, start + seconds, abs_tol=1e-8)
                or not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning)):
            raise RuntimeError('Invalid MuJoCo state, clock reset, or physics warning')

        return {
            name: dict(zip(('yaw', 'pitch'), (self.data.ctrl[self.actuators[name]] *
                                              arm['motor_to_joint_sign']).tolist()))
            for name, arm in self.arms.items()
        }

    def _score(self):
        for name, gid in self.tools.items():
            position = self.data.geom_xpos[gid].copy()
            rotation = self.data.geom_xmat[gid].reshape(3, 3).copy()
            p0, r0 = self.previous.get(name, (position, rotation))
            for plant, _ in tool_contacts(self.plants, self.tree, p0, r0, position, rotation,
                                          self.model.geom_size[gid]):
                self.contacts[name].setdefault(plant['id'], float(self.data.time))
            self.previous[name] = (position, rotation)

    def contact_report(self):
        touched = {plant_id for arm in self.contacts.values() for plant_id in arm}
        return dict(first_contact_s_by_arm=self.contacts,
                    unique_weed_contacts=sorted(p['id'] for p in self.plants
                                                if p['category'] == 'weed' and p['id'] in touched),
                    unique_crop_contacts=sorted(p['id'] for p in self.plants
                                                if p['category'] == 'crop' and p['id'] in touched),
                    warnings=[int(w.number) for w in self.data.warning])
