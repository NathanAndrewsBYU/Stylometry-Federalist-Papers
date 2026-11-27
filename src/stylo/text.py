"""Text cleaning, tokenization and the fixed function-word list."""

from __future__ import annotations

import hashlib
import os
import re

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Word tokens: letters and apostrophes (same character class as the legacy
# Delta tokenizer, so lowercased word tokens match legacy/cluster/ exactly).
WORD_RE = re.compile(r"[A-Za-z']+")
# Full token stream: words, digit runs, or any single non-space symbol
# (punctuation), in order.
TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9']")


def clean_federalist(raw: str) -> str:
    """Identical to legacy/cluster/stylometry_lib.clean_text: drop [...] editorial
    brackets and a trailing PUBLIUS signature, collapse whitespace."""
    text = re.sub(r"\[.*?\]", "", raw)
    text = re.sub(r"PUBLIUS\.?\s*$", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


CLEANERS = {"federalist": clean_federalist}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def is_word(token: str) -> bool:
    return WORD_RE.fullmatch(token) is not None


def load_function_words(name: str = "function_words_en.txt") -> frozenset[str]:
    path = os.path.join(_DATA_DIR, name)
    with open(path, encoding="utf-8") as f:
        words = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return frozenset(words)


def function_words_hash(name: str = "function_words_en.txt") -> str:
    words = sorted(load_function_words(name))
    return hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()[:16]
