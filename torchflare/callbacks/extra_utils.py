"""Implements extra utilities required."""
import math
from functools import partial


def _is_min(score, best, min_delta):
    return score <= (best - min_delta)


def _is_max(score, best, min_delta):
    return score >= (best + min_delta)


def init_improvement(mode: str, min_delta: float):
    """Get the scoring function and the best value according to mode.

    Args:
        mode: one of min or max.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.

    Returns:
        The scoring function and best value according to mode.
    """
    if mode == "min":

        improvement = partial(_is_min, min_delta=min_delta)
        best_score = math.inf

    else:

        improvement = partial(_is_max, min_delta=min_delta)
        best_score = -math.inf

    return improvement, best_score
