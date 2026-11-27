"""Arm B2: compression-based cross-entropy (character level).

Score for class c = extra compressed bits needed to encode the window after
class c's training text, per character of the window:
    8 * (|LZMA(train_c + window)| - |LZMA(train_c)|) / len(window)
Lower = the window is more predictable from that class's text.

This stands in for a true PPM model (docs/project_plan.md section 1.4, arm B;
docs/DECISION-LOG.md D-011). LZMA is
fast because it is written in C, and its dictionary (8 MB at preset 6) holds
a whole training corpus, which zlib's 32 KB window cannot. A pure-Python PPM
would be too slow for 1,000 permutations. If needed, a PPM implementation
can be swapped in behind the same interface.
"""

from __future__ import annotations

import lzma

from ..representations import Window


class CompressionArm:
    def __init__(self, preset: int = 6, lowercase: bool = False):
        self.name = "lzma"
        self.preset = preset
        self.lowercase = lowercase
        self.train: dict[str, bytes] = {}
        self.base: dict[str, int] = {}

    def _text(self, w: Window) -> str:
        t = w.text
        return t.lower() if self.lowercase else t

    def _size(self, data: bytes) -> int:
        return len(lzma.compress(data, preset=self.preset))

    def fit(self, train: dict[str, list[Window]]) -> None:
        for c in sorted(train):
            ws = sorted(train[c], key=lambda w: (w.unit_id, w.index))
            data = "\n".join(self._text(w) for w in ws).encode("utf-8") + b"\n"
            self.train[c] = data
            self.base[c] = self._size(data)

    def score(self, window: Window) -> dict[str, float]:
        x = self._text(window).encode("utf-8")
        n = max(len(x), 1)
        return {c: 8.0 * (self._size(self.train[c] + x) - self.base[c]) / n for c in self.train}
