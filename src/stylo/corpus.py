"""Corpus loading. A corpus root holds one subfolder per class, each with
one .txt file per unit (paper or chapter): <root>/<class>/<unit_id>.txt."""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass

from .text import CLEANERS


@dataclass(frozen=True)
class Unit:
    unit_id: str      # file stem, e.g. "federalist_10" -- the grouping key
    cls: str          # folder name, e.g. "madison"
    text: str         # cleaned text
    number: int | None  # paper/chapter number parsed from the id, if any


def load_corpus(root: str, cleaner: str = "federalist") -> dict[str, Unit]:
    clean = CLEANERS[cleaner]
    units: dict[str, Unit] = {}
    for cls in sorted(os.listdir(root)):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            continue
        for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
            uid = os.path.splitext(os.path.basename(path))[0]
            if uid in units:
                raise ValueError(f"duplicate unit id {uid!r} in {root}")
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = clean(f.read())
            m = re.search(r"(\d+)$", uid)
            units[uid] = Unit(uid, cls, text, int(m.group(1)) if m else None)
    if not units:
        raise FileNotFoundError(f"no units found under {root!r}")
    return units


def by_class(units: dict[str, Unit]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for uid in sorted(units):
        out.setdefault(units[uid].cls, []).append(uid)
    return out


def corpus_hash(units: dict[str, Unit]) -> str:
    h = hashlib.sha256()
    for uid in sorted(units):
        u = units[uid]
        h.update(f"{uid}\t{u.cls}\t".encode("utf-8"))
        h.update(u.text.encode("utf-8"))
    return h.hexdigest()[:16]
