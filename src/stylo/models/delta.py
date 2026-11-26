"""Arm A: Burrows' Delta family on most-frequent-word relative frequencies.

Variants:
  per_class  The legacy project version (cluster/traditional_stylometry.py and
             the Colab notebooks): each class has its own per-word mean and
             SD; Delta = mean |(x - mean_c) / sd_c|. Kept as the default so the
             regression check reproduces the committed CSVs exactly.
  burrows    Classic Burrows (2002): z-scores from the pooled training mean
             and SD; Delta = mean |z_x - centroid_c| (centroid = class mean z).
  cosine     Cosine Delta (Smith & Aldridge 2011): 1 - cos(z_x, centroid_c).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter

from ..representations import Window


def _freqs(window: Window, vocab: list[str]) -> list[float]:
    total = sum(window.counts.values())
    if total == 0:
        return [0.0] * len(vocab)
    # same expression as the legacy code, for bit-identical regression output
    return [(window.counts.get(w, 0) / total) * 1000 for w in vocab]


def _mean_sd(rows: list[list[float]], j: int, exact: bool) -> tuple[float, float]:
    vals = [r[j] for r in rows]
    n = len(vals)
    if exact:
        # statistics.mean/stdev, as in the legacy code: bit-identical, but slow
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if n > 1 else 0
    else:
        mean = math.fsum(vals) / n
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0
    return mean, (sd if sd > 0 else 1e-6)


class DeltaArm:
    def __init__(self, variant: str = "per_class", n_features: int = 100,
                 vocab: list[str] | None = None, exact: bool = False):
        """exact=True uses the statistics module like the legacy code (needed
        only for the bit-identical regression check; about 50x slower)."""
        self.exact = exact
        if variant not in ("per_class", "burrows", "cosine"):
            raise ValueError(variant)
        self.variant = variant
        self.name = {"per_class": "delta", "burrows": "delta_burrows", "cosine": "cosine_delta"}[variant]
        self.n_features = n_features
        self.fixed_vocab = vocab
        self.vocab: list[str] = []
        self.profile: dict = {}

    def fit(self, train: dict[str, list[Window]]) -> None:
        if self.fixed_vocab is not None:
            self.vocab = list(self.fixed_vocab)
        else:
            total = Counter()
            for c in sorted(train):
                for w in train[c]:
                    total.update(w.counts)
            ranked = sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))
            self.vocab = [w for w, _ in ranked[: self.n_features]]

        rows = {c: [_freqs(w, self.vocab) for w in ws] for c, ws in train.items()}
        if self.variant == "per_class":
            self.profile = {c: [_mean_sd(r, j, self.exact) for j in range(len(self.vocab))]
                            for c, r in rows.items()}
        else:
            allrows = [r for c in sorted(rows) for r in rows[c]]
            self.pooled = [_mean_sd(allrows, j, self.exact) for j in range(len(self.vocab))]
            self.profile = {}
            for c, r in rows.items():
                zs = [self._z(x) for x in r]
                self.profile[c] = [statistics.fmean(z[j] for z in zs) for j in range(len(self.vocab))]

    def _z(self, x: list[float]) -> list[float]:
        return [(v - m) / s for v, (m, s) in zip(x, self.pooled)]

    def score(self, window: Window) -> dict[str, float]:
        x = _freqs(window, self.vocab)
        out = {}
        if self.variant == "per_class":
            for c, prof in self.profile.items():
                zs = [abs((x[j] - prof[j][0]) / prof[j][1]) for j in range(len(x))]
                out[c] = sum(zs) / len(zs)
            return out
        z = self._z(x)
        for c, cen in self.profile.items():
            if self.variant == "burrows":
                out[c] = sum(abs(a - b) for a, b in zip(z, cen)) / len(z)
            else:
                dot = sum(a * b for a, b in zip(z, cen))
                na = math.sqrt(sum(a * a for a in z))
                nb = math.sqrt(sum(b * b for b in cen))
                out[c] = 1.0 - dot / (na * nb) if na > 0 and nb > 0 else 1.0
        return out
