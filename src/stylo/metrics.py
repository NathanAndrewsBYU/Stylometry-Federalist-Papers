"""AUC, bootstrap intervals and permutation statistics (pure Python)."""

from __future__ import annotations

import math
import random
import statistics
from typing import Sequence


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score of a positive > score of a negative), ties count 1/2
    (Mann-Whitney U / (n_pos * n_neg)). Rank-based, O(n log n)."""
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum = 0.0
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for t in allv[i:j + 1] if t[1] == 1)
        i = j + 1
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def bootstrap_auc_ci(pos: Sequence[float], neg: Sequence[float], n_boot: int = 2000,
                     seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI, resampling UNITS within each class (never windows)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        p = [pos[rng.randrange(len(pos))] for _ in pos]
        n = [neg[rng.randrange(len(neg))] for _ in neg]
        vals.append(auc(p, n))
    vals.sort()
    lo = vals[int(math.floor((1 - level) / 2 * n_boot))]
    hi = vals[int(math.ceil((1 + level) / 2 * n_boot)) - 1]
    return lo, hi


def paired_bootstrap_delta_auc(pos_a, neg_a, pos_b, neg_b, n_boot: int = 2000,
                               seed: int = 0, level: float = 0.95):
    """AUC(a) - AUC(b) for two arms scored on the SAME units in the same order;
    units are resampled jointly. Returns (delta, lo, hi)."""
    if len(pos_a) != len(pos_b) or len(neg_a) != len(neg_b):
        raise ValueError("paired bootstrap needs the same units for both arms")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        ip = [rng.randrange(len(pos_a)) for _ in pos_a]
        ineg = [rng.randrange(len(neg_a)) for _ in neg_a]
        vals.append(auc([pos_a[i] for i in ip], [neg_a[i] for i in ineg])
                    - auc([pos_b[i] for i in ip], [neg_b[i] for i in ineg]))
    vals.sort()
    lo = vals[int(math.floor((1 - level) / 2 * n_boot))]
    hi = vals[int(math.ceil((1 + level) / 2 * n_boot)) - 1]
    return auc(pos_a, neg_a) - auc(pos_b, neg_b), lo, hi


def permutation_summary(observed: float, null: Sequence[float]) -> dict:
    """p = (b + 1) / (N + 1) with b = #null >= observed; z against the null."""
    null = [v for v in null if not math.isnan(v)]
    n = len(null)
    if n == 0:
        return {"n_perm": 0}
    b = sum(1 for v in null if v >= observed)
    sd = statistics.stdev(null) if n > 1 else float("nan")
    mean = statistics.fmean(null)
    return {
        "n_perm": n,
        "p_perm": (b + 1) / (n + 1),
        "null_mean": mean,
        "null_sd": sd,
        "null_max": max(null),
        "null_q025": sorted(null)[int(0.025 * (n - 1))],
        "null_q975": sorted(null)[int(math.ceil(0.975 * (n - 1)))],
        "z_perm": (observed - mean) / sd if sd and sd > 0 else float("nan"),
    }


def binom_two_sided_ok(k: int, n: int) -> float:
    """Exact two-sided binomial p-value vs 0.5 (doubling the smaller tail)."""
    lo = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    hi = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(lo, hi))
