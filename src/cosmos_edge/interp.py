"""Chunk interpolation shared by the edge and by planners that predict their own
execution, so both sides agree on what "execute a_i" means at control rate."""
import numpy as np


def sample(chunk, index):
    """Linearly interpolate rows of ``chunk`` at fractional ``index``; clamps at ends."""
    index = float(np.clip(index, 0, len(chunk) - 1))
    lo = int(np.floor(index))
    hi = min(lo + 1, len(chunk) - 1)
    w = index - lo
    return (1 - w) * chunk[lo] + w * chunk[hi]
