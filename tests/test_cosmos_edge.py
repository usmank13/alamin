import numpy as np

from cosmos_edge.edge import EdgeConfig, EdgeRuntime, Passthrough, PDTracker
from cosmos_edge.experiments import RunConfig, run_episode
from cosmos_edge.messages import Bundle
from cosmos_edge.net import LatencyTrace, VirtualNet
from cosmos_edge.world import Perturbation


def test_virtual_net_is_reproducible_and_drops_during_outage():
    def delays(seed):
        net = VirtualNet(LatencyTrace("heavy", base=0.05, scale=0.1), LatencyTrace(outages=((1.0, 2.0),)), seed)
        for i in range(20):
            net.schedule("down", object(), i * 0.1)
            net.schedule("up", object(), i * 0.1)
        return [m.get("received") for m in net.log]

    first = delays(3)
    assert first == delays(3)
    assert first != delays(4)
    up = first[1::2]
    assert all(r is None for r in up[10:20]) and all(r is not None for r in up[:10])


def test_nominal_episode_succeeds_with_little_correction():
    r = run_episode(RunConfig(corrector="pd"))
    assert r["success"]
    assert r["mean_delta"] < 0.01


def test_episode_is_deterministic():
    cfg = RunConfig(corrector="pd", perturbations=(Perturbation(0.8, "payload", 5.0),),
                    down=LatencyTrace("jitter", base=0.1, scale=0.05))
    a, b = run_episode(cfg), run_episode(cfg)
    assert a["trajectory"] == b["trajectory"]


def test_pd_corrector_recovers_payload_that_breaks_open_loop():
    pert = (Perturbation(0.8, "payload", 5.0),)
    open_loop = run_episode(RunConfig(corrector="passthrough", perturbations=pert))
    corrected = run_episode(RunConfig(corrector="pd", perturbations=pert))
    assert not open_loop["success"] and corrected["success"]
    assert "saturation" in corrected["requeries"]


def test_object_shift_triggers_visual_requery():
    r = run_episode(RunConfig(perturbations=(Perturbation(1.0, "object_shift", (0.15, -0.1)),)))
    assert r["requeries"].get("visual", 0) + r["requeries"].get("prediction_mismatch", 0) >= 1
    assert r["success"]


def _bundle(n=8):
    zeros = np.zeros((n, 2))
    return Bundle(0, 0.0, 0.1, zeros, np.zeros((n, 4)), zeros, 0.0)


def test_authority_envelope_clamps_per_tick_and_per_chunk():
    edge = EdgeRuntime(PDTracker(kp=100), EdgeConfig(max_delta_tick=0.01, max_delta_chunk=0.025, triggers=False))
    edge.receive(_bundle(), 0.0)
    far = np.array([1.0, 1.0, 0.0, 0.0])
    norms = [np.linalg.norm(edge.tick(t * 0.02, far, np.zeros(2))[0]) for t in range(5)]
    assert np.allclose(norms, [0.01, 0.01, 0.005, 0.0, 0.0])


def test_passthrough_holds_last_command_when_chunk_runs_out():
    edge = EdgeRuntime(Passthrough(), EdgeConfig(triggers=False))
    edge.receive(_bundle(n=2), 0.0)
    command, _ = edge.tick(0.5, np.zeros(4), np.zeros(2))
    assert edge.stats["idle"] == 1 and command is not None
