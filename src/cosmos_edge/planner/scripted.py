"""Stand-in planner for the toy world.

It plans a straight, speed-limited path to the goal it sees in the latent and
predicts its own future by rolling the chunk through a *nominal* copy of the
world, the way a world model predicts under its training distribution. Payload
or gain perturbations therefore create kinematic residuals, and goal shifts
create visual residuals, exactly as the spec's tiers assume."""
import numpy as np

from .. import interp
from ..messages import Bundle, Observation
from ..world.toy import ToyReachWorld


class ScriptedPlanner:
    def __init__(self, chunk_size=16, policy_hz=15, speed=0.4, control_hz=50):
        self.chunk_size = chunk_size
        self.policy_dt = 1.0 / policy_hz
        self.speed = speed
        self.nominal = ToyReachWorld(control_hz=control_hz)

    def get_bundle(self, obs: Observation, chunk_id: int, speculative=False):
        pos, goal = obs.proprio[:2], obs.latent[:2]
        offset = goal - pos
        dist = np.linalg.norm(offset)
        step = self.speed * self.policy_dt
        direction = offset / dist if dist else np.zeros(2)
        k = np.arange(1, self.chunk_size + 1)[:, None]
        waypoints = pos + direction * np.minimum(k * step, dist)
        actions = waypoints
        return Bundle(
            chunk_id=chunk_id,
            obs_time=obs.time,
            policy_dt=self.policy_dt,
            actions=actions,
            future_proprio=self._predict(obs.proprio, actions),
            future_latents=np.repeat(goal[None], self.chunk_size, axis=0),
            value=-float(dist),
            speculative=speculative,
        )

    def _predict(self, proprio, actions):
        """Roll the chunk through the nominal model; ŝ_i is the predicted state at
        policy step i, so ŝ_0 echoes the conditioning state."""
        w = self.nominal
        w.data.qpos[:2], w.data.qvel[:2] = proprio[:2], proprio[2:4]
        w.data.time = 0.0
        out, t = [], 0.0
        for i in range(self.chunk_size):
            out.append(w.proprio())
            while t < (i + 1) * self.policy_dt - 1e-9:
                w.step(interp.sample(actions, t / self.policy_dt))
                t += w.control_dt
        return np.array(out)
