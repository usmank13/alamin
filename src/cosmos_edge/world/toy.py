"""Minimal MuJoCo reach task that exercises the split-loop contract without
LIBERO or a Cosmos checkpoint.

A point mass tracks position targets in the plane toward a goal marker. Proprio
is (x, y, vx, vy); the "camera latent" is the observed goal position, so moving
the goal is invisible to proprio (a tier-2 visual residual) while payload and
gain changes show up only as kinematic residuals (tier 1)."""
import mujoco
import numpy as np

from .base import Perturbation

XML = """
<mujoco model="toy_reach">
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="1 1 0.01" rgba="0.8 0.8 0.8 1"/>
    <body name="ee" pos="0 0 0.02">
      <joint name="x" type="slide" axis="1 0 0" damping="4"/>
      <joint name="y" type="slide" axis="0 1 0" damping="4"/>
      <geom name="ee" type="sphere" size="0.02" mass="1" contype="0" conaffinity="0"/>
    </body>
    <site name="goal" pos="0.3 0.2 0.02" size="0.02" rgba="0 1 0 0.5"/>
  </worldbody>
  <actuator>
    <position name="ax" joint="x" kp="60"/>
    <position name="ay" joint="y" kp="60"/>
  </actuator>
</mujoco>
"""


class ToyReachWorld:
    action_dim = 2
    proprio_dim = 4
    latent_dim = 2

    def __init__(self, control_hz=50, tolerance=0.02, perturbations=()):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.control_dt = 1.0 / control_hz
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.tolerance = tolerance
        self.perturbations = sorted(perturbations, key=lambda p: p.time)
        self._mass = self.model.body_mass.copy()
        self._inertia = self.model.body_inertia.copy()
        self._gain = self.model.actuator_gainprm.copy()
        self._bias = self.model.actuator_biasprm.copy()
        self.reset(0)

    @property
    def time(self):
        return self.data.time

    def reset(self, seed):
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[:] = self._mass
        self.model.body_inertia[:] = self._inertia
        self.model.actuator_gainprm[:] = self._gain
        self.model.actuator_biasprm[:] = self._bias
        self.rng = np.random.default_rng(seed)
        angle = self.rng.uniform(0, 2 * np.pi)
        self.goal = 0.3 * np.array([np.cos(angle), np.sin(angle)])
        self.latent_noise = 0.0
        self.pending = list(self.perturbations)
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def proprio(self):
        return np.concatenate([self.data.qpos[:2], self.data.qvel[:2]]).copy()

    def latent(self):
        noise = self.rng.normal(0, self.latent_noise, 2) if self.latent_noise else 0.0
        return self.goal + noise

    def _apply(self, p: Perturbation):
        if p.kind == "object_shift":
            self.goal = self.goal + np.asarray(p.value, dtype=float)
        elif p.kind == "payload":
            # MuJoCo caches mass-derived constants; recompute them after editing the model.
            self.model.body_mass[1] = self._mass[1] * p.value
            self.model.body_inertia[1] = self._inertia[1] * p.value
            mujoco.mj_setConst(self.model, self.data)
        elif p.kind == "gain":
            self.model.actuator_gainprm[:, 0] = self._gain[:, 0] * p.value
            self.model.actuator_biasprm[:, 1] = self._bias[:, 1] * p.value
        elif p.kind == "camera_jitter":
            self.latent_noise = float(p.value)
        else:
            raise ValueError(f"unknown perturbation {p.kind!r}")

    def step(self, action):
        while self.pending and self.pending[0].time <= self.data.time + 1e-9:
            self._apply(self.pending.pop(0))
        self.data.ctrl[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    def success(self):
        return bool(np.linalg.norm(self.data.qpos[:2] - self.goal) < self.tolerance
                    and np.linalg.norm(self.data.qvel[:2]) < 0.05)

    def task_state(self):
        return {"goal": self.goal.tolist(), "pos": self.data.qpos[:2].tolist()}
