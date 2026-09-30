"""Corrector variants: Δ = f(a_i, r_s, r_l, i/N). The runtime clamps Δ to the
authority envelope, so a corrector can only degrade tracking, never invent motion.

V0 passthrough is the open-loop NVIDIA reference client; V0.5 is a PD tracker
toward ŝ. V1 (learned, inverse-dynamics supervised) plugs in with the same call."""
import numpy as np


class Passthrough:
    name = "passthrough"

    def __call__(self, action, r_s, r_l, phase):
        return np.zeros_like(action)


class PDTracker:
    """Steer from the actual state toward the planner's predicted state.

    ``r_s`` is actual − predicted proprio laid out as (position, velocity)."""
    name = "pd"

    def __init__(self, kp=1.0, kd=0.1):
        self.kp, self.kd = kp, kd

    def __call__(self, action, r_s, r_l, phase):
        n = len(action)
        return -(self.kp * r_s[:n] + self.kd * r_s[n:2 * n])


CORRECTORS = {"passthrough": Passthrough, "pd": PDTracker}
