"""Model arms. Each arm has fit(train: dict[class, list[Window]]) and
score(window) -> dict[class, float] where lower = closer to that class."""

from __future__ import annotations

from typing import Callable


def make_arm_factory(name: str, **kw) -> Callable[[], object]:
    """Return a zero-argument factory for the named arm (fresh model per fold)."""
    if name in ("delta", "delta_per_class", "delta_burrows", "cosine_delta"):
        from .delta import DeltaArm
        variant = {"delta": "per_class", "delta_per_class": "per_class",
                   "delta_burrows": "burrows", "cosine_delta": "cosine"}[name]
        return lambda: DeltaArm(variant=variant, **kw)
    if name == "ngram_kn":
        from .ngram import KneserNeyArm
        return lambda: KneserNeyArm(**kw)
    if name == "lzma":
        from .compress import CompressionArm
        return lambda: CompressionArm(**kw)
    if name == "linear":
        from .linear import LinearArm
        return lambda: LinearArm(**kw)
    raise ValueError(f"unknown arm {name!r}")


ARM_NAMES = ("delta", "delta_burrows", "cosine_delta", "ngram_kn", "lzma", "linear")
