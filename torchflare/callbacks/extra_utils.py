"""Implemennts extra utilites required."""
from functools import partial
import math


def _is_min(score, best, min_delta):
    return score <= (best - min_delta)


def _is_max(score, best, min_delta):
    return score >= (best + min_delta)


def init_improvement(mode: str, min_delta):

    if mode == "min":

        improvement = partial(_is_min, min_delta=min_delta)
        best_score = math.inf

    else:

        improvement = partial(_is_max, min_delta=min_delta)
        best_score = -math.inf

    return improvement, best_score
