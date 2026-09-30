"""Discrete-event episode: world, edge and planner exchange messages only through
the virtual network, so a later multi-process split is a transport swap."""
from dataclasses import dataclass, field

import numpy as np

from ..edge import CORRECTORS, EdgeConfig, EdgeRuntime
from ..messages import Bundle, Observation, Requery
from ..net import LatencyTrace, VirtualNet
from ..planner import ScriptedPlanner
from ..world import ToyReachWorld


@dataclass
class RunConfig:
    seed: int = 0
    horizon: float = 6.0
    corrector: str = "pd"
    compute_time: float = 0.4          # virtual planner inference time per chunk (s)
    uplink_hz: float = 10.0            # periodic proprio/latent snapshots
    speculate: bool = False
    resync_every: int = 3              # speculative chunks before a forced real-observation chunk
    down: LatencyTrace = field(default_factory=LatencyTrace)
    up: LatencyTrace = field(default_factory=LatencyTrace)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    perturbations: tuple = ()


class PlannerServer:
    """Slow loop: one inference at a time, abortable, optionally speculative."""

    def __init__(self, planner, cfg: RunConfig, net: VirtualNet):
        self.planner, self.cfg, self.net = planner, cfg, net
        self.latest = None
        self.pending = None           # Requery waiting for the planner to be free
        self.busy_until = -1.0
        self.in_flight = None         # chunk id currently being denoised
        self.aborted = set()
        self.next_id = 0
        self.speculative_run = 0
        self.log = []

    def on_uplink(self, msg, now):
        if isinstance(msg, Observation):
            self.latest = msg
            return
        self.latest = msg.obs
        if msg.reason != "chunk_low" and self.in_flight is not None and now < self.busy_until:
            self.aborted.add(self.in_flight)
            self.busy_until = now
            self.log.append({"t": now, "abort": self.in_flight, "reason": msg.reason})
        if msg.reason != "chunk_low":
            self.speculative_run = self.cfg.resync_every  # disable speculation until re-validated
        self.pending = msg

    def step(self, now):
        if now < self.busy_until:
            return
        if self.pending is not None:
            self._launch(self.latest, now, speculative=False)
            self.pending = None

    def _launch(self, obs, now, speculative):
        bundle = self.planner.get_bundle(obs, self.next_id, speculative)
        self.in_flight = self.next_id
        self.next_id += 1
        self.busy_until = now + self.cfg.compute_time
        self.net.schedule("down", bundle, now, extra_delay=self.cfg.compute_time)
        self.log.append({"t": now, "launch": bundle.chunk_id, "obs_time": obs.time,
                         "speculative": speculative, "reason": getattr(self.pending, "reason", None)})
        if self.cfg.speculate and self.speculative_run < self.cfg.resync_every:
            # Condition the next chunk on this chunk's predicted end state.
            end = len(bundle.actions) - 1
            predicted = Observation(obs.time + end * bundle.policy_dt,
                                    bundle.future_proprio[-1], bundle.future_latents[-1])
            self._speculate_after = (now + self.cfg.compute_time, predicted)
            self.speculative_run += 1
        else:
            self._speculate_after = None
            self.speculative_run = 0

    def step_speculation(self, now):
        pending = getattr(self, "_speculate_after", None)
        if pending and now >= pending[0] and self.pending is None and now >= self.busy_until:
            self._speculate_after = None
            self._launch(pending[1], now, speculative=True)


def run_episode(cfg: RunConfig):
    world = ToyReachWorld(perturbations=cfg.perturbations)
    world.reset(cfg.seed)
    planner = ScriptedPlanner(control_hz=round(1 / world.control_dt))
    net = VirtualNet(cfg.down, cfg.up, seed=cfg.seed)
    edge = EdgeRuntime(CORRECTORS[cfg.corrector](), cfg.edge)
    server = PlannerServer(planner, cfg, net)
    uplink_every = max(1, round(1 / (cfg.uplink_hz * world.control_dt)))

    # Bootstrap: the edge's first request is sent at t=0.
    success_time, tick = None, 0
    trajectory = []
    while world.time < cfg.horizon - 1e-9:
        now = world.time
        for env in net.deliver(now):
            if env.channel == "down":
                if env.payload.chunk_id not in server.aborted:
                    edge.receive(env.payload, now)
            else:
                server.on_uplink(env.payload, now)
        server.step(now)
        server.step_speculation(now)

        proprio, latent = world.proprio(), world.latent()
        command, events = edge.tick(now, proprio, latent)
        for event in events:
            net.schedule("up", event, now)
        if tick % uplink_every == 0:
            net.schedule("up", Observation(now, proprio, latent), now)
        world.step(command)
        trajectory.append([now, *proprio[:2], *command])
        tick += 1
        if success_time is None and world.success():
            success_time = world.time

    s = edge.stats
    reasons = [e["reason"] for e in server.log if "launch" in e and e["reason"]]
    return {
        "seed": cfg.seed,
        "corrector": cfg.corrector,
        "success": success_time is not None,
        "success_time": success_time,
        "final_error": float(np.linalg.norm(world.data.qpos[:2] - world.goal)),
        "idle_fraction": s["idle"] / max(1, s["ticks"]),
        "latency_p50": _pct(s["latency"], 50), "latency_p95": _pct(s["latency"], 95),
        "latency_p99": _pct(s["latency"], 99),
        "mean_delta": float(np.mean(s["delta_norm"])) if s["delta_norm"] else 0.0,
        "requeries": {r: reasons.count(r) for r in set(reasons)},
        "aborts": len(server.aborted),
        "bytes_down": sum(m["bytes"] for m in net.log if m["channel"] == "down"),
        "bytes_up": sum(m["bytes"] for m in net.log if m["channel"] == "up"),
        "messages": net.log,
        "trajectory": trajectory,
    }


def _pct(values, q):
    return float(np.percentile(values, q)) if values else None
