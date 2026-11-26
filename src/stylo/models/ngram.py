"""Arm B1: interpolated Kneser-Ney n-gram language model, one per class.

Score = cross-entropy of the window under that class's model, in bits per
token (lower = the class's model finds the window less surprising). The
perplexity logic is the same as the GPT-2 arm, but the model only ever sees
the training data, so there is no pretraining contamination.

Implementation notes:
  - Interpolated KN with a fixed absolute discount D (default 0.75).
    The highest order uses raw counts, lower orders use continuation counts,
    and the unigram level interpolates with a uniform distribution over the
    shared training vocabulary + 1 (for unseen tokens). So every token gets
    a non-zero probability, and the base distribution is the same for every
    class.
  - Each window is one sequence, padded with <s> and terminated by </s>.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..representations import Window

BOS, EOS = "<s>", "</s>"


class _KN:
    def __init__(self, order: int, discount: float, vocab_size: int):
        self.n = order
        self.d = discount
        self.base = 1.0 / (vocab_size + 1)
        self.tables: list[dict] = [{} for _ in range(order + 1)]   # tables[k][ctx] -> Counter
        self.totals: list[dict] = [{} for _ in range(order + 1)]
        self.types: list[dict] = [{} for _ in range(order + 1)]

    def fit(self, seqs: list[list[str]]) -> None:
        n = self.n
        top = defaultdict(Counter)
        for seq in seqs:
            s = [BOS] * (n - 1) + seq + [EOS]
            for i in range(n - 1, len(s)):
                top[tuple(s[i - n + 1:i])][s[i]] += 1
        self.tables[n] = dict(top)
        for k in range(n - 1, 0, -1):
            lower = defaultdict(Counter)
            for ctx, ws in self.tables[k + 1].items():
                lctx = ctx[1:]
                for w in ws:
                    lower[lctx][w] += 1        # continuation count: distinct left extensions
            self.tables[k] = dict(lower)
        for k in range(1, n + 1):
            for ctx, ws in self.tables[k].items():
                self.totals[k][ctx] = sum(ws.values())
                self.types[k][ctx] = len(ws)

    def logprob2(self, w: str, ctx: tuple[str, ...]) -> float:
        p = self.base
        for k in range(1, self.n + 1):
            c = ctx[len(ctx) - (k - 1):] if k > 1 else ()
            table = self.tables[k].get(c)
            if not table:
                continue                        # unseen context: keep the lower-order estimate
            tot = self.totals[k][c]
            p = max(table.get(w, 0) - self.d, 0) / tot + self.d * self.types[k][c] / tot * p
        return math.log2(p)

    def cross_entropy(self, seq: list[str]) -> float:
        n = self.n
        s = [BOS] * (n - 1) + seq + [EOS]
        lp = [self.logprob2(s[i], tuple(s[i - n + 1:i])) for i in range(n - 1, len(s))]
        return -sum(lp) / len(lp)


class KneserNeyArm:
    def __init__(self, order: int = 3, discount: float = 0.75, lowercase: bool = True):
        self.name = f"ngram_kn{order}"
        self.order = order
        self.discount = discount
        self.lowercase = lowercase
        self.models: dict[str, _KN] = {}

    def _seq(self, w: Window) -> list[str]:
        return [t.lower() for t in w.tokens] if self.lowercase else list(w.tokens)

    def fit(self, train: dict[str, list[Window]]) -> None:
        seqs = {c: [self._seq(w) for w in ws] for c, ws in train.items()}
        vocab = {t for ss in seqs.values() for s in ss for t in s}
        for c in sorted(seqs):
            m = _KN(self.order, self.discount, len(vocab) + 1)   # +1 for </s>
            m.fit(seqs[c])
            self.models[c] = m

    def score(self, window: Window) -> dict[str, float]:
        seq = self._seq(window)
        return {c: m.cross_entropy(seq) for c, m in self.models.items()}
