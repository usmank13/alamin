"""Fast loop: interpolation, correction under an authority envelope, divergence
tiers, requery events and chunk-handoff blending. One ``tick`` per control step."""
from dataclasses import dataclass

import numpy as np

from .. import interp
from ..messages import Bundle, Observation, Requery


@dataclass
class EdgeConfig:
    prefetch_margin: float = 0.3      # request the next chunk this long before the current ends (s)
    blend_ticks: int = 4              # time-weighted handoff between chunks
    max_delta_tick: float = 0.03      # ‖Δ‖ clamp per tick (action units)
    max_delta_chunk: float = 0.5      # Σ‖Δ‖ budget per chunk
    visual_threshold: float = 0.05    # ‖r_l‖ that means the scene left the plan (tier 2)
    saturation_ticks: int = 10        # envelope pinned this long → requery (tier 3)
    mismatch_ticks: int = 2           # r_l high within this many ticks of a chunk start → planner predictions are bad
    triggers: bool = True             # False = pure prefetch, like the reference client
    align: str = "receipt"            # "receipt": chunk starts on arrival; "obs": skip the stale prefix


class EdgeRuntime:
    def __init__(self, corrector, config: EdgeConfig | None = None):
        self.corrector = corrector
        self.cfg = config or EdgeConfig()
        self.current = self.start = None
        self.queued = None                  # speculative bundle waiting for the current to finish
        self.previous = self.prev_start = None
        self.blend_left = 0
        self.budget = 0.0
        self.saturated = 0
        self.last_command = None
        self.requested = False
        self.stats = {"ticks": 0, "idle": 0, "delta_norm": [], "divergence": [], "latency": []}

    # --- downlink -----------------------------------------------------------------
    def receive(self, bundle: Bundle, now):
        self.stats["latency"].append(now - bundle.obs_time)
        if bundle.speculative and self.current is not None and self._remaining(now) > 0:
            self.queued = bundle
            return
        self._switch(bundle, now)

    def _switch(self, bundle, now):
        if self.current is not None:
            self.previous, self.prev_start = self.current, self.start
            self.blend_left = self.cfg.blend_ticks
        self.current = bundle
        self.start = bundle.obs_time if self.cfg.align == "obs" else now
        self.budget = self.cfg.max_delta_chunk
        self.saturated = 0
        self.requested = False
        self.chunk_ticks = 0

    def _index(self, bundle, start, now):
        return (now - start) / bundle.policy_dt

    def _remaining(self, now):
        return (len(self.current.actions) - 1) * self.current.policy_dt - (now - self.start)

    # --- control tick -------------------------------------------------------------
    def tick(self, now, proprio, latent):
        """Return (command, events) where events are Requery packets for the uplink."""
        self.stats["ticks"] += 1
        events = []
        if self.current is not None and self.queued is not None and self._remaining(now) <= 0:
            self._switch(self.queued, now)
            self.queued = None
        if self.current is None or self._remaining(now) < -self.current.policy_dt:
            # No valid command: hold-and-settle on the last command (stable for position control).
            self.stats["idle"] += 1
            if not self.requested:
                events.append(self._requery(now, "chunk_low", proprio, latent))
            return (self.last_command if self.last_command is not None else proprio[:len(proprio) // 2]), events

        b, i = self.current, self._index(self.current, self.start, now)
        action = interp.sample(b.actions, i)
        r_s = proprio - interp.sample(b.future_proprio, i)
        r_l = latent - interp.sample(b.future_latents, i)

        raw = self.corrector(action, r_s, r_l, i / len(b.actions))
        delta = self._clamp(raw)
        command = action + delta
        if self.blend_left and self.previous is not None:
            w = 1 - self.blend_left / (self.cfg.blend_ticks + 1)
            old = interp.sample(self.previous.actions, self._index(self.previous, self.prev_start, now))
            command = w * command + (1 - w) * old
            self.blend_left -= 1

        divergence, reason = self._divergence(r_s, r_l)
        self.stats["delta_norm"].append(float(np.linalg.norm(delta)))
        self.stats["divergence"].append(divergence)
        if not self.requested:
            if reason and self.cfg.triggers:
                events.append(self._requery(now, reason, proprio, latent, divergence))
            elif self._remaining(now) < self.cfg.prefetch_margin:
                events.append(self._requery(now, "chunk_low", proprio, latent))
        self.chunk_ticks += 1
        self.last_command = command
        return command, events

    def _clamp(self, raw):
        norm = float(np.linalg.norm(raw))
        limit = min(self.cfg.max_delta_tick, self.budget)
        pinned = norm > limit and norm > 1e-9
        delta = raw * (limit / norm) if pinned else raw
        self.budget -= float(np.linalg.norm(delta))
        self.saturated = self.saturated + 1 if pinned else 0
        return delta

    def _divergence(self, r_s, r_l):
        """Rule-based d_t in [0, 1] with the three-tier reason. The learned head replaces this."""
        visual = float(np.linalg.norm(r_l)) / self.cfg.visual_threshold
        saturation = self.saturated / self.cfg.saturation_ticks
        divergence = float(np.clip(max(visual, saturation), 0, 1))
        if visual >= 1:
            return divergence, "prediction_mismatch" if self.chunk_ticks < self.cfg.mismatch_ticks else "visual"
        if saturation >= 1:
            return divergence, "saturation"
        return divergence, None

    def _requery(self, now, reason, proprio, latent, divergence=0.0):
        self.requested = True
        return Requery(now, reason, Observation(now, proprio.copy(), latent.copy()), divergence)
