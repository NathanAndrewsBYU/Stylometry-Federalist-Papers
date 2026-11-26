"""Balancing, fold construction and label permutation. All randomness goes
through random.Random(seed) so every split is reproducible from its seed."""

from __future__ import annotations

import random
from typing import Sequence

Pool = dict[str, list[str]]   # class -> unit ids


def balanced_subsample(ids: Sequence[str], n: int, seed: int | None) -> list[str]:
    """Same rule as the legacy code: random.Random(seed).sample(sorted ids, n),
    returned sorted. seed=None or len(ids) <= n returns all ids."""
    ids = sorted(ids)
    if seed is None or len(ids) <= n:
        return ids
    return sorted(random.Random(seed).sample(ids, n))


def stratified_group_kfold(pool: Pool, k: int, seed: int) -> list[Pool]:
    """k folds; each class's units are shuffled and dealt round-robin, so
    every fold holds out (almost) the same number of units per class. With
    equal class sizes the hold-out is symmetric (D-006)."""
    rng = random.Random(seed)
    folds: list[Pool] = [{c: [] for c in pool} for _ in range(k)]
    for c in sorted(pool):
        ids = sorted(pool[c])
        rng.shuffle(ids)
        for i, uid in enumerate(ids):
            folds[i % k][c].append(uid)
    for f in folds:
        for c in f:
            f[c].sort()
    return folds


def permute_labels(pool: Pool, seed: int) -> Pool:
    """Shuffle class labels across all pooled units, keeping class sizes."""
    classes = sorted(pool)
    ids = sorted(uid for c in classes for uid in pool[c])
    random.Random(seed).shuffle(ids)
    out: Pool = {}
    i = 0
    for c in classes:
        n = len(pool[c])
        out[c] = sorted(ids[i:i + n])
        i += n
    return out


def random_split(ids: Sequence[str], n_per_group: int, seed: int,
                 names: tuple[str, str] = ("group_a", "group_b")) -> Pool:
    """Pick 2*n ids at random and split them into two pseudo-classes
    (within-author null test)."""
    picked = random.Random(seed).sample(sorted(ids), 2 * n_per_group)
    return {names[0]: sorted(picked[:n_per_group]), names[1]: sorted(picked[n_per_group:])}
