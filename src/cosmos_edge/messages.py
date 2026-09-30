"""Wire messages. Every message carries virtual send/receive timestamps so the
same objects can later move over a real transport unchanged."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bundle:
    """Planner → edge, once per chunk.

    ``actions[i]``, ``future_proprio[i]`` and ``future_latents[i]`` refer to policy
    step ``i`` after the observation the bundle was conditioned on (``obs_time``).
    """
    chunk_id: int
    obs_time: float
    policy_dt: float
    actions: np.ndarray          # (N, action_dim)
    future_proprio: np.ndarray   # (N, proprio_dim) predicted ŝ
    future_latents: np.ndarray   # (N, latent_dim) predicted L̂ (keyframes may repeat)
    value: float                 # V̂
    speculative: bool = False

    def nbytes(self):
        return sum(a.nbytes for a in (self.actions, self.future_proprio, self.future_latents)) + 4


@dataclass
class Observation:
    """Edge → planner snapshot (proprio + camera latent/frame)."""
    time: float
    proprio: np.ndarray
    latent: np.ndarray

    def nbytes(self):
        return self.proprio.nbytes + self.latent.nbytes


@dataclass
class Requery:
    """Event packet that jumps the uplink queue."""
    time: float
    reason: str                  # "chunk_low" | "visual" | "saturation" | "prediction_mismatch"
    obs: Observation
    divergence: float = 0.0


@dataclass(order=True)
class Envelope:
    """A message in flight on the virtual network."""
    deliver_at: float
    seq: int
    sent_at: float = field(compare=False)
    channel: str = field(compare=False)     # "down" | "up"
    payload: object = field(compare=False)
