"""Planner interface. Cosmos Policy 2B, Cosmos 3 Nano and Super are meant to be
config swaps behind this one call."""
from typing import Protocol

from ..messages import Bundle, Observation


class Planner(Protocol):
    chunk_size: int
    policy_dt: float

    def get_bundle(self, obs: Observation, chunk_id: int, speculative: bool = False) -> Bundle:
        """Return {action chunk, future latents, ŝ, V̂} conditioned on ``obs``."""
        ...
