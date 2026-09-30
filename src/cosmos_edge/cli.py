"""``cosmos-edge sweep``: latency × corrector × perturbation grid on the toy world.

Every episode is reproducible from (seed, latency trace); results and message logs
are written as JSONL with virtual send/receive timestamps."""
import argparse
import itertools
import json
from pathlib import Path

from .analysis import summarize
from .edge import CORRECTORS
from .experiments import RunConfig, run_episode
from .net import LatencyTrace
from .world import Perturbation

PERTURBATIONS = {
    "none": (),
    "payload": (Perturbation(0.8, "payload", 5.0),),
    "gain": (Perturbation(0.8, "gain", 0.4),),
    "object_shift": (Perturbation(1.0, "object_shift", (0.15, -0.1)),),
    "camera_jitter": (Perturbation(0.5, "camera_jitter", 0.01),),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cosmos-edge")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="run a latency × corrector × perturbation grid")
    sweep.add_argument("--correctors", nargs="+", default=list(CORRECTORS), choices=list(CORRECTORS))
    sweep.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS), choices=list(PERTURBATIONS))
    sweep.add_argument("--latencies", nargs="+", type=float, default=[0.02, 0.1, 0.3, 0.6, 1.0],
                       help="one-way delay in seconds, both directions")
    sweep.add_argument("--jitter", type=float, default=0.0, help="half-normal jitter scale (s)")
    sweep.add_argument("--seeds", type=int, default=10)
    sweep.add_argument("--compute-time", type=float, default=0.4)
    sweep.add_argument("--speculate", action="store_true")
    sweep.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    kind = "jitter" if args.jitter else "fixed"
    results = []
    for corrector, name, latency, seed in itertools.product(
            args.correctors, args.perturbations, args.latencies, range(args.seeds)):
        trace = LatencyTrace(kind=kind, base=latency, scale=args.jitter)
        r = run_episode(RunConfig(seed=seed, corrector=corrector, down=trace, up=trace,
                                  compute_time=args.compute_time, speculate=args.speculate,
                                  perturbations=PERTURBATIONS[name]))
        r.update(perturbation=name, latency=latency)
        results.append(r)

    rows = summarize(results)
    print(f"{'corrector':12s} {'perturbation':14s} {'latency':>7s} {'success':>7s} {'idle':>5s} {'|Δ|':>6s}")
    for row in rows:
        print(f"{row['corrector']:12s} {row['perturbation']:14s} {row['latency']:7.2f} "
              f"{row['success_rate']:7.2f} {row['idle_fraction']:5.2f} {row['mean_delta']:6.4f}")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        with open(args.output / "episodes.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps({k: v for k, v in r.items() if k != "trajectory"}) + "\n")
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
