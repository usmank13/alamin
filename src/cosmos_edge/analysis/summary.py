"""Aggregate episode results into success-vs-latency rows."""
from collections import defaultdict

import numpy as np


def summarize(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r["corrector"], r["perturbation"], r["latency"])].append(r)
    rows = []
    for (corrector, perturbation, latency), rs in sorted(groups.items()):
        rows.append({
            "corrector": corrector, "perturbation": perturbation, "latency": latency,
            "episodes": len(rs),
            "success_rate": float(np.mean([r["success"] for r in rs])),
            "idle_fraction": float(np.mean([r["idle_fraction"] for r in rs])),
            "mean_delta": float(np.mean([r["mean_delta"] for r in rs])),
            "final_error": float(np.mean([r["final_error"] for r in rs])),
        })
    return rows
