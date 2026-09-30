"""World interface. The world owns the virtual clock; it only advances when the
experiment loop steps it, so planner compute never costs simulated time unless
it is injected explicitly."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Perturbation:
    """Mid-rollout injection. kind: ``object_shift`` (value = xy offset),
    ``payload`` (value = mass multiplier), ``gain`` (value = actuator gain multiplier),
    ``camera_jitter`` (value = latent noise std)."""
    time: float
    kind: str
    value: object


class World(Protocol):
    control_dt: float
    action_dim: int
    proprio_dim: int
    latent_dim: int
    time: float

    def reset(self, seed: int) -> None: ...
    def proprio(self) -> np.ndarray: ...
    def latent(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> None: ...
    def success(self) -> bool: ...
    def task_state(self) -> dict: ...
