# Cosmos-Edge prototype

A large remote world-model planner paired with a small on-robot corrector. The
planner predicts the future and the edge model enforces it. The research spec is
the Notion page *Remote World Models*. This branch builds the sim prototype
described in §4 of that spec.

## Current state

The five modules from the spec's software layout run end to end on a toy MuJoCo
reach task. The task stands in for LIBERO and uses a scripted planner, so no
Cosmos checkpoint is needed yet.

| Module | Contents | Status |
| --- | --- | --- |
| `src/cosmos_edge/net/` | Virtual-time event queue with `schedule(msg, delay)`. Latency traces can be fixed, jittered, heavy-tailed or replayed, with drops and outage windows. Every message is logged with virtual send and receive times. | Working |
| `src/cosmos_edge/planner/` | The `get_bundle(obs) → {actions, ŝ, L̂, V̂}` interface. `ScriptedPlanner` predicts ŝ by rolling the chunk through a nominal copy of the world. | Working stand-in |
| `src/cosmos_edge/edge/` | V0 passthrough (the NVIDIA open-loop reference) and V0.5 PD tracker toward ŝ. Includes the authority envelope (per-tick and per-chunk), rule-based three-tier divergence (visual, saturation, prediction mismatch), requery events, handoff blending, hold-and-settle, and receipt or obs alignment. | Working; the learned V1 corrector is still to do |
| `src/cosmos_edge/world/` | `ToyReachWorld` with mid-rollout perturbations: `object_shift`, `payload`, `gain` and `camera_jitter`. | Working; the LIBERO wrapper is still to do |
| `src/cosmos_edge/experiments/`, `analysis/` | Discrete-event loop, planner server with abort and speculation, and a `cosmos-edge sweep` grid with JSONL output. | Working; Hydra configs are still to do |

The toy latent is the observed goal position. A goal shift is therefore invisible
to proprio: it produces a tier-2 visual residual. Payload and gain changes show up
only as kinematic residuals (tier 1).

```bash
uv sync --extra dev
.venv/bin/pytest tests/test_cosmos_edge.py
.venv/bin/cosmos-edge sweep --seeds 10 --output outputs/edge_sweep_toy
```

Early toy-sweep observations (5 seeds, 0.4 s virtual planner compute):

- Under a 5× payload, open-loop execution fails at every latency. The PD corrector
  recovers it at 20 ms but not at 300 ms and above.
- Under an object shift at 300 ms, PD is *worse* than passthrough (0.8 vs 1.0).
  Correction toward a stale ŝ fights the replan. The mismatch fallback should
  probably suppress correction until the new chunk arrives.
- Idle time is high: 0.12 to 0.74 of ticks at 20 ms to 1 s one-way delay. The
  default prefetch margin (0.3 s) is shorter than compute plus round trip, so
  speculation and prefetch tuning are the next knobs to turn.

These numbers only validate the harness. They are not results.

## Planner dependency findings (checked 2026-09-29)

- **Cosmos Policy 2B (LIBERO)**: `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`, with code
  at `NVlabs/cosmos-policy`. It is ungated, but the weights are *non-commercial*
  (NSCLv1). `get_action()` returns a 16×7 action chunk, `future_image_predictions`
  and `value_prediction`. Future proprio is predicted as a latent frame but is not
  returned as a key, so a small code change is needed to expose ŝ. It uses 5
  denoising steps and about 6–9 GB of VRAM in BF16.
- **aarch64**: upstream only supports Linux x86-64. The torch 2.7/2.9 pins,
  flash-attn, transformer-engine and natten will probably need source builds on
  GB10. This is the first thing to de-risk on the Spark.
- **Cosmos 3**: `nvidia/Cosmos3-Nano`, `-Super`, `-Edge`, `-Nano-Policy-DROID` and
  `-Edge-Policy-DROID` are public (OpenMDW). There is **no LIBERO policy
  checkpoint**. Inverse dynamics is released
  (`"model_mode": "inverse_dynamics"` in the NVIDIA/cosmos cookbook), which the
  Stage 2 labels depend on.
- **LIBERO**: the Cosmos Policy `libero` group installs PyPI `libero` 0.1.1 (the
  lerobot fork) with `robosuite==1.4.0` and `mujoco==3.3.2`. That MuJoCo version
  has a cp312 aarch64 wheel, but it conflicts with this repo's `mujoco==3.13.0`.
  Run LIBERO in its own environment behind the `World` interface.
- **Tokenizer**: the Wan2.1 VAE (8×8 spatial, 4× temporal, 16 channels), shipped
  in the gated `nvidia/Cosmos-Predict2-2B-Video2World`.

## Next steps

1. Get Cosmos Policy 2B running on the Spark in a separate environment. Use
   `libero_10` eval to reproduce the reported success rate.
2. De-risk the future-latent accuracy (spec §6) before building further. Compare
   predicted-latent error with a static-frame baseline on held-out rollouts.
3. Add a `CosmosPolicyPlanner` behind `Planner` (exposing ŝ) and a `LiberoWorld`
   behind `World`. Swap them in for the toy.
4. Stage 1 and Stage 2 training for the learned corrector, with inverse-dynamics
   labels.
